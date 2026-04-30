# Deep Analysis: Allocator Strategy for 32B GRPO on Aurora

**Date:** 2026-04-25
**Purpose:** Independent validation of assumptions before further allocator work.

## Executive Summary

After examining the PyTorch source code on Aurora (`frameworks/2025.3.1`, torch 2.10.0a0),
we have identified **two significant findings** that change the framing of the allocator problem:

1. **FSDP2 does NOT use `recordStream` at all** — it uses event-based synchronization instead.
   The `recordStream` no-op bug may not be the cause of the step-1 crashes in custom allocators.
2. **The 10 MB/step external growth is from CCL internal state**, not from VA churn.
   A caching allocator prevents the *crash trigger* (GC freeing CCL-registered blocks), but
   does NOT prevent the external growth itself. Even with perfect caching, runs would
   eventually hit the same wall — just much later (~700 steps instead of ~28).

These findings suggest we may be solving the wrong problem with increasingly complex
allocator strategies, and that a simpler approach may work.

---

## Finding 1: FSDP2 Does Not Use `recordStream`

### Evidence

On the actual Aurora framework (`/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/`):

```
# FSDP2 composable API — ZERO record_stream calls:
$ grep -rn 'record_stream' torch/distributed/fsdp/_fully_shard/*.py
(no results)

# FSDP2 uses event-based synchronization instead:
$ grep -n 'wait_stream\|wait_event' torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py
273:    all_gather_stream.wait_stream(all_gather_copy_in_stream)
361:    device_handle.current_stream().wait_event(all_gather_event)
531:    reduce_scatter_stream.wait_stream(current_stream)
```

FSDP2 synchronizes cross-stream access via `wait_stream()` and `wait_event()` — it records
an event after the AllGather completes on the comm stream, then the compute stream waits
for that event before using the buffer. **It never calls `tensor.record_stream()`.**

The `record_stream` calls found in the codebase are in:
- Old FSDP1 (`torch/distributed/fsdp/_runtime_utils.py`) — not used by this project
- `torch/distributed/utils.py` — only in the `_to_map` helper for `PackedSequence`, not
  in the AllGather/ReduceScatter path

### Implication

The XPUPluggableAllocator C++ class (`XPUPluggableAllocator.cpp:79-86`) has a working
`recordStream` dispatch mechanism:
```cpp
void XPUPluggableAllocator::recordStream(const c10::DataPtr& ptr, c10::Stream stream) {
    if (record_stream_fn_) {
        auto xpu_stream = c10::xpu::XPUStream(stream);
        record_stream_fn_(ptr.get(), &xpu_stream.queue());
    }
}
```
This is a no-op when `record_stream_fn_` is not set (which is the case — the Python
wrapper `XPUPluggableAllocator.__init__` never calls `set_record_stream_fn`).

**But FSDP2 never calls `recordStream` anyway.** So the no-op `recordStream` is irrelevant
to the FSDP2 AllGather/ReduceScatter path. The cross-stream safety for FSDP2 buffers is
handled by `wait_event` / `wait_stream`, which operate at the stream level, not the
allocator level.

### Question this raises

If `recordStream` isn't the mechanism FSDP2 uses, then **why did 6 of the 7 custom
allocator variants crash at step 1?**

Possible explanations:
1. **The crashes were from a different cross-stream race**, not the one `recordStream`
   is designed to prevent. FSDP2's `wait_event` provides ordering guarantees between
   *streams*, but the *allocator* can still recycle a block before the event is recorded
   if PyTorch's allocation scheduling runs ahead of the event insertion.
2. **The crashes were from the OFI MR registration problem**, not cross-stream safety at
   all. A fresh `sycl::malloc_device` call in the allocator produces a new VA, which
   triggers an OFI registration. If that VA is freed and then another address is allocated,
   the old OFI registration points to invalid memory. This would produce the same
   `banned:1` GPU fault signature but has nothing to do with `recordStream`.
3. **The `queue->wait()` in `delayfree` worked for a different reason than assumed.**
   `queue->wait()` doesn't just synchronize cross-stream access — it also forces all
   pending DMA operations (including CCL's OFI transfers) to complete before the memory
   is freed. This protects against OFI stale handles, not just cross-stream races.

**Recommendation:** Before building another allocator variant, confirm which of these
is actually happening. A diagnostic test: run `usm_delayfree_alloc.so` (the one that
passed) with `USM_ALLOC_DEBUG=1` and log every alloc/free. Count how many unique VAs
are created across step 0 and step 1. If the same VA is being reused across steps
(pooling working), the crash in other variants is from cross-stream races. If new VAs
are being created each step (pooling not working at the sizes involved), the crash is
from OFI MR accumulation.

---

## Finding 2: External Growth is CCL-Internal, Not VA-Churn

### Evidence

From `ccl_external_memory_growth_32b.md`:

> "The PyTorch caching allocator maintains stable VAs (`torch_resv` flat at 62.04 GiB),
> so OFI MR registrations should be stable. The ~10 MB/step external growth comes from
> a different source — likely CCL internal bookkeeping, IPC handle metadata, or Level
> Zero driver state associated with the collective operations themselves."

This means: **even with a perfect caching allocator that never frees anything and keeps
all VAs stable, the 10 MB/step external memory growth would still happen.** The external
growth is from CCL's own internal state, not from the allocator.

### Memory Budget with Caching Allocator at 32B (Linear Bucketing)

| Component | Size |
|-----------|------|
| Working set (actual allocations) | ~55 GiB |
| Linear bucketing overhead (~15% avg) | ~2-3 GiB |
| **Total cached** | **~57-58 GiB** |
| External at step 0 | ~1.5 GiB |
| **l0_free at step 0** | **~4.5-5.5 GiB** |
| External growth rate | ~10 MB/step |
| **Steps until l0_free exhausted** | **~450-550** |

This is much better than the default allocator (~28 steps), but it's not infinite. After
~500 steps, the same CCL external growth exhausts l0_free and the same crash pattern
occurs. For practical training (500 steps at 44s = ~6 hours), this may be sufficient.

### The real question: what IS the external growth?

Nobody has identified the source of the 10 MB/step CCL external growth. Candidates:

1. **IPC handle metadata accumulation** — CCL caches IPC handles for inter-tile
   communication. Even with stable VAs, it may accumulate metadata per-handle.
   `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536` prevents eviction but not accumulation.
2. **OFI completion queue entries** — each collective operation generates CQ entries.
   With `FI_MR_CACHE_MONITOR=disabled`, these may never be freed.
3. **Level Zero driver state** — `zeCommandListCreate`, `zeEventPoolCreate`, etc. may
   leak small amounts of device memory per collective.
4. **XCCL communicator internal buffers** — XCCL may allocate staging buffers that grow
   over time.

A diagnostic approach: run a minimal FSDP2 training loop (no GRPO, no generation,
just forward/backward/optimizer) with MEMPROBE logging, and plot `external` over 100+
steps. If external still grows at ~10 MB/step, the source is purely FSDP2+CCL. If
it's flat, the source is in the GRPO/generation/weight-sync path.

---

## Finding 3: Alternative Approaches Not Yet Explored

### A. Fix `FI_MR_CACHE_MONITOR` (upstream)

`FI_MR_CACHE_MONITOR=disabled` is required because Slingshot's default MR cache monitor
crashes on Aurora. But `disabled` prevents OFI from automatically deregistering stale
memory regions, which is the root cause of the OFI accumulation.

Has anyone tried `FI_MR_CACHE_MONITOR=userfaultfd` or `FI_MR_CACHE_MONITOR=memhooks`?
If either works, the entire allocator problem disappears — OFI would automatically
deregister freed VAs and re-register new ones.

### B. Wire `set_record_stream_fn` from Python (if it IS needed)

If it turns out `recordStream` is actually called somewhere we missed (e.g., in oneCCL's
internal PyTorch integration), we could wire it from Python without modifying PyTorch:

```python
import ctypes
allocator = torch._C._xpu_getAllocator()
# allocator is the _XPUPluggableAllocator instance
# If pybind11 exposes set_record_stream_fn, we could call it directly:
# allocator.set_record_stream_fn(record_stream_fn_ptr)
```

The C++ header (`XPUPluggableAllocator.h:53`) has `set_record_stream_fn` as a public
method. But the pybind11 registration (`Module.cpp:394-399`) registers the class WITHOUT
exposing `set_record_stream_fn`. This is fixable with a one-line pybind11 addition.

However, this requires modifying the PyTorch installation, which may not be viable
on a shared HPC system. An alternative: build a small `.so` that links against
`libtorch_python.so`, obtains the allocator pointer via `getCurrentAllocator()`, and
calls `set_record_stream_fn` directly from C++. This could be loaded at startup.

### C. Use PyTorch's default XPU allocator (not pluggable) + prevent GC

The default XPU caching allocator has `recordStream` implemented correctly (it's not
a pluggable allocator — it's the built-in one). The problem is that it triggers GC
under memory pressure, freeing blocks with CCL handles.

Could we prevent GC entirely? `gc_threshold:1.0` doesn't work because the allocator
still frees under OOM pressure. But what about:
- Patching `torch.xpu.memory._free_mutex` or similar to prevent the GC codepath
- Setting a very high `max_split_size_mb` to reduce fragmentation so GC is never triggered
- Pre-allocating a "ballast" tensor that pins memory so the allocator never needs to
  over-allocate and then GC

### D. Checkpoint-restart (pragmatic)

The current workaround (20-step segments) works. With 2-hop XCCL at ~44s/step:
- 20 steps = ~15 minutes per segment
- Process restart takes ~5 minutes (vLLM startup + model load + checkpoint resume)
- Effective throughput: 20 steps per 20 minutes = 1 step/minute
- For 500 steps: ~8.3 hours (within a 12h production queue allocation)

This is not elegant, but it works today with zero code changes.

---

## Recommendation

Before building another allocator variant, we should:

1. **Verify whether the step-1 crashes in custom allocators are from cross-stream
   races or OFI MR issues.** The `recordStream` explanation is plausible but unproven
   and contradicted by the finding that FSDP2 doesn't use `recordStream`. Run
   `delayfree` with `USM_ALLOC_DEBUG=1` to characterize the allocation pattern.

2. **Test `FI_MR_CACHE_MONITOR=userfaultfd`** — this is a one-line env var change that
   could eliminate the entire OFI accumulation problem if it works on Aurora's Slingshot.

3. **Identify the source of 10 MB/step external growth** — run a minimal FSDP2-only
   loop (no GRPO) for 100 steps and see if external grows. This determines whether the
   linear bucketing allocator would actually survive long runs.

4. **If linear bucketing is still the path**: The approach is sound but is a bandaid
   over three upstream bugs (CCL IPC stale handles, FI_MR_CACHE_MONITOR disabled,
   pluggable allocator recordStream unwired). It buys ~500 steps of headroom, which is
   probably sufficient for production, but the fragmentation math is tight (~4.5 GiB
   headroom). Variable-length allocations from GRPO (different generation lengths across
   steps) could erode this margin through bucket proliferation.

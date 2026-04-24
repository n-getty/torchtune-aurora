# Bug Report: oneCCL rejects XPU tensors allocated with `expandable_segments`

**System:** Aurora (ALCF), Intel Max 1550 GPU, `frameworks/2025.3.1` (torch 2.10, oneCCL 2021.17)

## Summary

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` causes all oneCCL collective
operations (allreduce, allgather, etc.) to fail with:

```
coll_check.cpp:68 ccl_check_usm_pointers: condition is_valid_type failed
coll: allreduce - invalid usm pointer type: unknown for device type: gpu
```

PyTorch's `expandable_segments` allocator uses SYCL's `ext_oneapi_virtual_mem`
extension (backed by Level Zero `zeVirtualMemReserve`/`zeVirtualMemMap`) instead
of direct USM device allocations (`zeMemAllocDevice`). The resulting pointers are
valid XPU device memory — compute kernels work correctly — but `zeMemGetAllocProperties`
classifies them as `ZE_MEMORY_TYPE_UNKNOWN` rather than `ZE_MEMORY_TYPE_DEVICE`.

This is documented Level Zero behavior: `ze_api.h` states that `zeMemGetAllocProperties`
returns `ZE_MEMORY_TYPE_UNKNOWN` for pointers "unrelated to the context" (i.e., not
allocated via USM APIs). Virtual memory mappings created via `zeVirtualMemMap` are
not USM allocations and fall into this category. oneCCL's `ccl_check_usm_pointers`
rejects any pointer that is not `device`, `host`, or `shared` type.

## Impact

`expandable_segments` significantly reduces memory fragmentation overhead in workloads
with mixed-size allocation patterns — such as RL training, where activation tensors,
gradient buffers, and optimizer state cycle through different sizes each step. Without
it, we hit `UR_RESULT_ERROR_OUT_OF_RESOURCES` at batch sizes that would otherwise fit
in the 48 GiB tile.

## Reproducer

Save as `repro_ccl_expandable_segments.py` and run:

```bash
# PASS:
ZE_AFFINITY_MASK=0,1 torchrun --nproc_per_node=2 repro_ccl_expandable_segments.py

# FAIL:
PYTORCH_ALLOC_CONF=expandable_segments:True \
ZE_AFFINITY_MASK=0,1 torchrun --nproc_per_node=2 repro_ccl_expandable_segments.py
```

Note: `ZE_AFFINITY_MASK=0,1` makes both tiles visible to torchrun; in production our
PBS launcher sets `ZE_AFFINITY_MASK=$LOCAL_RANK` per-rank so each process sees only
its own tile. Either approach reproduces the failure.

```python
"""Minimal reproducer: expandable_segments breaks oneCCL allreduce on XPU."""
import os, torch

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)

torch.distributed.init_process_group(backend="xccl")

t = torch.ones(10, device=device)
torch.distributed.all_reduce(t)  # <-- fails with expandable_segments:True
print(f"Rank {local_rank}: allreduce OK, sum={t[0].item()}")

torch.distributed.destroy_process_group()
```

## Fragmentation overhead evidence

The benefit of `expandable_segments` is largest under mixed-size allocation patterns
that create fragmentation — exactly the pattern of RL training. The following standalone
scenario quantifies this directly.

**Test setup:** allocate 6 small tensors (2–8 MB each, simulating activation tensors),
free 3 of them (simulating activation checkpointing), then allocate each large tensor
one at a time and measure fragmentation overhead (`reserved_bytes - allocated_bytes`):
the memory held by the allocator in freed segments that cannot service the next
large allocation.

```python
"""Standalone fragmentation scenario — no CCL, single-process, safe to run either way."""
import gc, torch

device = torch.device("xpu:0")
torch.xpu.set_device(0)

def overhead_mb():
    s = torch.xpu.memory_stats(0)
    return (s["reserved_bytes.all.current"] - s["allocated_bytes.all.current"]) / 1e6

# Simulate one RL step: small activations allocated, half freed (checkpointing)
small = [torch.randn(sz * 1024 * 1024 // 4, device=device) for sz in [2, 4, 4, 8, 8, 8]]
for i in range(1, len(small), 2):
    small[i] = None
gc.collect(); torch.xpu.synchronize()

# Allocate large tensors (gradient buffers / optimizer state) and measure overhead
for sz_mb in [64, 128, 256]:
    t = torch.randn(sz_mb * 1024 * 1024 // 4, device=device)
    print(f"  {sz_mb:3d} MB large alloc  overhead={overhead_mb():.0f} MB")
    del t; torch.xpu.synchronize()
```

Results on Intel Max 1550, `frameworks/2025.3.1`:

| Large alloc | Default overhead | Expandable overhead | Reduction |
|-------------|-----------------|---------------------|-----------|
| 64 MB       | 27 MB           | 23 MB               | 16%       |
| 128 MB      | 94 MB           | 19 MB               | 80%       |
| 256 MB      | 229 MB          | 32 MB               | 86%       |

The default allocator creates a new reserved segment for each large allocation that
does not fit in any freed slot, causing overhead to compound as more sizes have been
cycled. `expandable_segments` grows the existing segment instead, keeping overhead
nearly flat regardless of prior allocation history. At 256 MB the overhead is 7× lower.

Note: for uniform-size allocation cycles (allocate N tensors of identical size, free
all, repeat), both allocators behave identically — the benefit is specific to
mixed-size fragmentation patterns.

## Root cause confirmation: two-phase LD_PRELOAD investigation

### Phase 1: USM type check

We wrote a C shim intercepting `sycl::get_pointer_type()` (the actual call site
in `ccl_check_usm_pointers`, confirmed via `objdump -d libccl.so`) that promotes
`usm::alloc::unknown` → `device`:

```bash
gcc -shared -fPIC -o sycl_usm_shim.so recipes/dev/sycl_usm_shim.c -ldl

LD_PRELOAD=/path/to/sycl_usm_shim.so \
PYTORCH_ALLOC_CONF=expandable_segments:True \
ZE_AFFINITY_MASK=0,1 \
torchrun --nproc_per_node=2 repro_ccl_expandable_segments.py
```

Result (small tensor, 10 elements):

```
[sycl_usm_shim] unknown->device ptr=0x7f343ec00000
[sycl_usm_shim] unknown->device ptr=0x7f4bd6400000
Rank 0: allreduce OK, sum=2.0
Rank 1: allreduce OK, sum=2.0
```

This confirms the full causal chain for `ccl_check_usm_pointers`:

1. `PYTORCH_ALLOC_CONF=expandable_segments:True` causes PyTorch's XPU allocator
   to allocate via SYCL `ext_oneapi_virtual_mem` (→ L0 `zeVirtualMemMap`)
2. `ccl_check_usm_pointers` calls `sycl::get_pointer_type(ptr, context)`
3. SYCL calls `zeMemGetAllocProperties`, which returns `ZE_MEMORY_TYPE_UNKNOWN`
   for virtual-memory-mapped addresses (documented L0 behavior)
4. oneCCL rejects the pointer

### Phase 2: Level Zero IPC — a deeper incompatibility

The type check is not the only failure mode. Testing with production-scale tensors
(4096×4096 bfloat16, ~16 MB shards for 2 ranks) reveals a second failure even
when the shim bypasses the type check:

```
Segmentation fault from GPU at 0xff00000033256000,
  ctx_id: 5 (CCS) type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
```

This is a hardware-level GPU page fault from the Level Zero Compute Command
Streamer (CCS). The pattern is size-dependent:

| Output tensor size | Behavior with shim |
|--------------------|-------------------|
| ≤ 8 MB             | **PASS** — CCL uses staging buffers (copy-in, send via OFI, copy-out) |
| > 8–12 MB          | **GPU page fault** — CCL switches to zero-copy IPC path |

The zero-copy path calls `zeMemGetIpcHandle` on the destination buffer so the
sending rank can DMA directly into the receiver's GPU VAS. For USM allocations
(`zeMemAllocDevice`), L0 registers the allocation and can return an IPC handle.
For virtual memory regions (`zeVirtualMemMap`), L0 does not track them as USM
allocations: `zeMemGetIpcHandle` either fails or returns an invalid handle. The
sending rank then attempts to write through an unmapped GPU virtual address →
GPU page fault.

**Practical consequence:** Production FSDP2 training uses parameter tensors of
32–256 MB per layer. Every all_gather (forward) and reduce_scatter (backward)
of such parameters falls above the staging-buffer threshold and will trigger GPU
page faults, even with the LD_PRELOAD shim in place.

The shim source is in `recipes/dev/sycl_usm_shim.c`. It is useful for
confirming the type-check root cause (Phase 1) but **is not a viable production
fix** because it does not address the IPC failure (Phase 2).

## Suggested fix

**Option A — Fix in oneCCL:** The fix requires addressing two independent
failure modes:

1. **Type check** (`ccl_check_usm_pointers`): accept virtual-memory-mapped
   device pointers. If `sycl::get_pointer_type()` returns `unknown`, perform a
   secondary check via `zeVirtualMemQueryMappedRangeProperties` to confirm the
   range is mapped on the device. Alternatively, provide an opt-in environment
   variable `CCL_SKIP_USM_CHECK=1`. Note: fixing only the type check is
   insufficient (see Phase 2 above).

2. **IPC registration** (zero-copy path): for large tensors, oneCCL calls
   `zeMemGetIpcHandle` to share buffers across processes. Virtual memory regions
   cannot be shared via `zeMemGetIpcHandle` (only USM allocations can). Options:
   a. Use `zePhysicalMemGetIpcHandle` + `zeVirtualMemSetIpcHandle` on the
      physical memory backing the virtual range — supported in L0 1.5+.
   b. Fall back to the staging-buffer path when `zeMemGetAllocProperties`
      returns `ZE_MEMORY_TYPE_UNKNOWN`, instead of attempting IPC.
   c. Provide `CCL_FORCE_STAGING=1` to disable zero-copy for all-gather/
      reduce-scatter (performance trade-off, correctness guaranteed).

**Option B — Fix in PyTorch's XPU allocator:** Replace the virtual-memory
backing (`ext_oneapi_virtual_mem`) with `sycl::malloc_device` (plain USM), which
allocates via `zeMemAllocDevice` and reports as `ZE_MEMORY_TYPE_DEVICE`. We
implemented two generations of this fix, both validated fully.

### Generation 1: `usm_caching_alloc.cpp` (size-class free lists)

Segregated free lists, one set per device (power-of-2 size buckets). Fixes
both CCL failure modes. Limitation: no coalescing — mixed-size patterns
still fragment at the driver level.

### Generation 2: `usm_arena_alloc.cpp` (coalescing arena) — **current production**

Sub-allocates from large USM slabs (4 GiB by default, configurable via
`USM_ARENA_CHUNK_GB`). Small allocations (< 1 MiB) use size-class buckets.
Large allocations (≥ 1 MiB) come from the arena with **immediate coalescing**:
on every free, the block merges with adjacent free neighbors before re-entering
the free list. Adjacent alloc/free cycles of different sizes merge back into
one large block, eliminating driver-level fragmentation.

```cpp
// usm_arena_alloc.cpp (key internals)
// Build: icpx -shared -fPIC -fsycl -O2 -o usm_arena_alloc.so usm_arena_alloc.cpp
extern "C" {
void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue);
void  xpu_usm_free(void* ptr, size_t size, sycl::queue* queue);
void  xpu_usm_get_arena_stats(size_t* growths, size_t* coalesces,
                               size_t* reserved_bytes, size_t* allocated_bytes);
}
```

```python
from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
alloc = XPUPluggableAllocator("usm_arena_alloc.so", "xpu_usm_malloc", "xpu_usm_free")
change_current_allocator(alloc)  # must be before any XPU init / init_process_group
```

Verified results on Intel Max 1550, `frameworks/2025.3.1`, Aurora x4502c7s7b0n0:

- `allreduce` (any size): ✓ pass
- `all_gather_into_tensor` (16 MB output): ✓ pass
- `all_gather_into_tensor` (256 MB output, above IPC threshold): ✓ pass
- `reduce_scatter_tensor` (16 MB input): ✓ pass
- FSDP2 forward+backward (3-step, 4 ranks, mixed-size MLP): ✓ complete
  - 50 coalesces observed during training, 9.1 GiB reserved across 4 ranks
- Coalescing reuse: 64+128+256 MiB freed → 400 MiB allocated from merged block: ✓
- Chunk growth: 0 additional driver allocs for large tensors within existing slab: ✓

```bash
export XPU_USM_ALLOC_SO="${TORCHTUNE_DIR}/recipes/dev/usm_arena_alloc.so"
```

We run this allocator in production via `XPU_USM_ALLOC_SO` for workloads
that do not require intra-node CCL IPC (e.g., multi-node with OFI transport,
or single-node at model sizes where the default allocator does not OOM).

**Critical limitation:** the XPUPluggableAllocator mechanism itself is
incompatible with CCL's intra-node IPC zero-copy path — see the companion
bug report below. This allocator works for multi-node (OFI transport, no
IPC) but **cannot be used for single-node FSDP at 32B+ scale** where the
default allocator OOMs.

**Residual limitation vs `expandable_segments`:** interleaved live/free blocks
within the arena cannot coalesce (a live block prevents merging its neighbors).
In the pathological case (every other freed block has a live neighbor), large
tensors still require a new slab. The practical fix in PyTorch would be to
implement `expandable_segments` via `zePhysicalMemAllocate` (physical USM)
rather than `zeVirtualMemReserve`, enabling `zeMemGetIpcHandle` / L0 1.5+
`zePhysicalMemGetIpcHandle` for cross-process sharing.

---

# Bug Report 2: XPUPluggableAllocator + FSDP causes GPU segfaults (root cause unknown)

**System:** Aurora (ALCF), Intel Max 1550 GPU, `frameworks/2025.3.1` (torch 2.10, oneCCL 2021.17)

> **Update 2026-04-22 — ROOT CAUSE FOUND.** The Python `XPUPluggableAllocator`
> wrapper (`torch/xpu/memory.py`) never wires `set_record_stream_fn`. The
> resulting `recordStream()` is a silent no-op. FSDP2 issues all-gather /
> reduce-scatter on a comm stream whose pending kernels still reference
> allocations the compute stream has just dropped — the pluggable allocator
> recycles the buffer, the comm stream's kernel reads a recycled virtual
> address, and the GPU page-faults. **Confirmed by the `usm_delayfree_alloc`
> variant**, which adds `caller_q->wait()` to free and passes all FSDP steps
> where every other variant fails. See `experiments/arena_ipc/diag_findings.md`
> Phase 8/9. The original "SYCL context mismatch" hypothesis is rejected
> (`zeMemGetIpcHandle` is never called).

## Summary

Any custom allocator registered via `torch.xpu.memory.XPUPluggableAllocator`
causes GPU segfaults during the second training step of FSDP-sharded large
models (32B+, 10 ranks). The allocator returns proper USM device memory
(`ZE_MEMORY_TYPE_DEVICE`), step 0 fwd/bwd/optim succeeds, then step 1 page-faults
during compute (NOT during a CCL collective).

## Relationship to the `expandable_segments` bug

| | `expandable_segments` bug | `XPUPluggableAllocator` bug |
|---|---|---|
| **Allocation API** | `zeVirtualMemReserve`/`zeVirtualMemMap` | `sycl::malloc_device` (`zeMemAllocDevice`) |
| **L0 memory type** | `ZE_MEMORY_TYPE_UNKNOWN` | `ZE_MEMORY_TYPE_DEVICE` (correct) |
| **CCL type check** | Fails (`unknown` rejected) | Passes |
| **IPC failure** | `zeMemGetIpcHandle` unsupported for virtual mem | `zeMemGetIpcHandle` returns handle from wrong L0 context |
| **GPU fault address** | `0xff00000033256000` (PDE level) | `0xffffff8000000000` (PDP level, sentinel) |
| **Workaround** | Option B allocator (this doc) | Multi-node (OFI, no IPC) or default allocator |

Both bugs share the same downstream symptom: GPU page fault during CCL
zero-copy IPC DMA on tensors above the staging-buffer threshold (~8–12 MB).
But the root causes are independent — fixing `expandable_segments` would not
fix this bug, and vice versa.

## Reproducer

Any allocator `.so` registered via `XPUPluggableAllocator` will trigger this
on single-node FSDP with >=10 ranks and 32B+ parameter models. Minimal setup:

```cpp
// trivial_alloc.cpp — passthrough allocator, no pooling, no arena
#include <sycl/sycl.hpp>
#include <cstddef>
extern "C" {
void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    return sycl::malloc_device(size, *queue);  // plain USM
}
void xpu_usm_free(void* ptr, size_t size, sycl::queue* queue) {
    // queue parameter is often invalid (device ordinal, not pointer)
    // so we cannot call sycl::free here — just leak
}
}
```

```bash
icpx -shared -fPIC -fsycl -O2 -o trivial_alloc.so trivial_alloc.cpp

XPU_USM_ALLOC_SO=trivial_alloc.so \
ZE_FLAT_DEVICE_HIERARCHY=FLAT \
torchrun --standalone --nproc_per_node=10 \
    train_script.py  # any FSDP training with 32B+ model
```

Python allocator registration (must be before any XPU init):

```python
import os
_usm_so = os.environ.get("XPU_USM_ALLOC_SO")
if _usm_so:
    from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
    alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
    change_current_allocator(alloc)
```

## Observed failure

Crash occurs during the first FSDP `all_gather` or `reduce_scatter` in the
forward pass (typically within seconds of training start):

```
Segmentation fault from GPU at 0xffffff8000000000,
  ctx_id: 1 (CCS) type: 0 (NotPresent), level: 2 (PDP), access: 1 (Write), banned: 1, aborting.
```

The sentinel address `0xffffff8000000000` and PDP-level fault (one level above
the `expandable_segments` PDE-level fault) indicate an unmapped IPC handle —
the receiver opened an IPC handle that maps to no valid physical page at the
PDP level of the GPU page table.

## Systematic elimination (v5–v7 test series, 2026-04-22)

We tested three allocator variants to isolate the root cause, all producing
the same GPU segfault:

| Test | Allocator design | Arena? | Sub-alloc? | Result |
|------|-----------------|--------|------------|--------|
| v5 | Two-tier: small buckets + coalescing arena (1–8 MiB) + large direct (≥8 MiB) | Yes | Yes (1–8 MiB) | Segfault (0 steps) |
| v6 | Same as v5, BUCKET_CAP pooling bug fixed | Yes | Yes (1–8 MiB) | Segfault (0 steps) |
| v7 | **No arena at all**: small buckets (<1 MiB) + exact-aligned direct (≥1 MiB) | No | No | Segfault (0 steps) |

v7 is the critical test: every allocation ≥1 MiB is a first-class
`sycl::malloc_device` pointer with no sub-allocation, no arena, no pointer
arithmetic. The segfault is identical to v5 and v6.

**Control:** the same training script with the default allocator
(`unset XPU_USM_ALLOC_SO`) passes all CCL operations — but OOMs at step 4
for 32B models due to 29 GiB of driver overhead from memory fragmentation.

## Empirical investigation (2026-04-22)

A targeted investigation in `experiments/arena_ipc/` (see `diag_findings.md`)
ruled out the original SYCL-context-mismatch hypothesis and several follow-ups:

### What we confirmed

1. **The bug is strictly allocator-specific.** Same launch, same shim, same
   model, same 10-rank FSDP+Qwen3-32B run: with `XPU_USM_ALLOC_SO=` unset,
   step 0 + step 1 both pass cleanly. With it set, step 0 passes, step 1
   page-faults.

2. **The bug is NOT in CCL's L0 IPC path.** `experiments/arena_ipc/diag_2_l0_ipc_shim.c`
   is an LD_PRELOAD shim intercepting `zeMemGetIpcHandle`, `zeMemOpenIpcHandle`,
   and `zeMemGetAllocProperties`. In both the failing arena-allocator run and
   the passing default-allocator run, **zero** `zeMemGetIpcHandle` calls fire.
   Whatever CCL is doing for these single-node 10-rank collectives, it does
   not exercise the L0 IPC handle path.

3. **The page fault is NOT the `0xffffff8000000000` PDP sentinel** described
   in earlier sections. Real fault addresses observed (`0xff020005b01bd000`,
   `0xff050005b1dcb000`, etc.) are inside the high-half range XPU uses for
   regular USM device allocations. The level-4 (PML5) NotPresent fault means
   the page-table walk found no mapping for an address that should be mapped.

4. **The SYCL queue vs. context binding is NOT the issue.** A variant
   allocator (`experiments/arena_ipc/usm_ctx_alloc.cpp`) replaces
   `sycl::malloc_device(sz, queue)` with
   `sycl::aligned_alloc_device(align, sz, dev, ctx)` — i.e. binds to the
   queue's context, not the queue. It fails identically (step 0 passes,
   step 1 page-faults). PyTorch's `getCurrentXPUStream` rotates through a
   32-queue pool, but those queues all share the same L0 context (verified
   by `diag_1b_queue_probe.cpp`), so this rules out cross-queue pointer
   recycling within the pluggable allocator.

5. **The `queue` parameter to `xpu_usm_free` IS suspicious but orthogonal.**
   The Python docstring says 3 params; the C++ interface is 4 params
   `(void*, size_t, int, sycl::queue*)`. Allocators exporting 3 params (as
   the original `usm_caching_alloc.cpp` did) interpret `device_idx` as a
   `sycl::queue*` — values like `0x6`, `0x7`. The current arena allocator
   uses the correct 4-param signature; the bug persists regardless. This is
   a real PyTorch documentation/ABI bug but it is NOT the cause of the
   page faults.

### Open questions

- Step 0 always passes; step 1 always fails. What changes between iterations
  in FSDP that the pluggable allocator handles incorrectly?
- The simple 10-rank allreduce/all_gather/reduce_scatter reproducer
  (`experiments/arena_ipc/diag_2_repro_large.py`, 32 MiB tensors) PASSES
  with the arena allocator. Only FSDP `shard_model` + actual training
  triggers the failure. This implicates FSDP's parameter-shard lifetime
  management interacting with the pluggable allocator, not raw CCL.
- The page fault happens during regular GPU compute work in step 1, not at
  a collective boundary. Ranks page-fault on different addresses, all within
  the device USM range.

### Suggested next experiments

1. **Disable cache reuse**: build a passthrough (no free-list) version of the
   arena allocator. If FSDP+Qwen3-32B passes step 1, the bug is in the cache
   recycling logic (e.g. PyTorch returning a pointer to use before the L0
   driver has fully invalidated stale page-table entries from the prior
   tensor that lived there).
2. **Per-allocation queue tracking**: log `(malloc_queue, virtual_addr)` and
   `(free_queue, virtual_addr)` for every pointer; cross-reference the
   page-fault address against these to see if the faulting VA was recently
   freed-then-reallocated.
3. **`reshard_after_forward=False`** keeps params materialized through bwd;
   in our test it changed the failure mode to `UR_RESULT_ERROR_OUT_OF_RESOURCES`
   (UR:40) during fwd. This is a different code path but suggests the
   resharded shard lifecycle in FSDP is implicated.

## Root cause (confirmed 2026-04-22)

The Python `XPUPluggableAllocator` constructor in
`torch/xpu/memory.py:260-300` calls `torch._C._xpu_customAllocator(alloc_fn,
free_fn)` and never invokes `set_record_stream_fn`. Inspection of
`XPUPluggableAllocator::recordStream` in `libtorch_python.so` shows:

```
cmpq $0x0, 0x78(%rdi)    # if record_stream_fn_ == nullptr
je   ret                 # return without doing anything
```

The header `torch/include/torch/csrc/xpu/XPUPluggableAllocator.h` exposes the
`set_record_stream_fn` hook on the C++ side but the Python wrapper never uses
it. `recordStream` is therefore a no-op for **every** Python-loaded pluggable
allocator.

FSDP2 schedules collectives on a separate comm stream while compute kernels
are still pending on the compute stream. Normally `recordStream(dataPtr,
comm_stream)` keeps the buffer alive in the allocator until the comm stream
consumes it. With the pluggable allocator that call is a no-op, so the
compute-stream tensor goes out of Python scope, the buffer is freed and
returned to the cache, the next allocation reuses the same virtual address,
the L0 driver remaps the page table for that address — and the still-pending
comm-stream kernel issues a load/store against the now-stale mapping.

This is consistent with every datapoint:

- Default allocator: PASSES (it has internal `recordStream` wiring).
- Caching variants (arena, singleq, traceq): step 0 PASSES, step 1 FAULTS at
  PML5 level (page-table walk fails at the deepest level — recycled mapping).
- Power-of-2-only variant (largeonly): PTE level fault (exact-aligned
  recycling matches the freed slot exactly).
- No-cache passthrough: faults BEFORE step 0 at PDE level — every free returns
  to L0 immediately, comm-stream kernels see no mapping at all.
- `usm_delayfree_alloc` (caching + `caller_q->wait()` on free): PASSES all
  steps. Adding the synchronization that `recordStream` would have provided
  fixes the bug.

## Suggested fix

Wire `set_record_stream_fn` in the Python wrapper. Two-line change in
`torch/xpu/memory.py`:

```python
class XPUPluggableAllocator(_XPUAllocator):
    def __init__(self, path_to_lib_file, alloc_fn_name, free_fn_name,
                 record_stream_fn_name=None):
        ...
        if record_stream_fn_name is not None:
            rs_addr = ctypes.cast(getattr(allocator_lib, record_stream_fn_name),
                                  ctypes.c_void_p).value
            self._allocator = torch._C._xpu_customAllocator(
                alloc_fn_addr, free_fn_addr, rs_addr)
        else:
            self._allocator = torch._C._xpu_customAllocator(
                alloc_fn_addr, free_fn_addr)
```

with a matching C++ binding in `torch/csrc/xpu/Module.cpp` that forwards to
`set_record_stream_fn`. Until that lands upstream, the workaround is to call
`caller_q->wait()` (or maintain an internal pending-stream list) inside the
allocator's free path.

## Workarounds (until upstream fix lands)

**Single-node:** use `experiments/arena_ipc/usm_pending_alloc.so`. Each free
goes onto a per-size pending queue tagged with the freeing SYCL queue.
Allocation prefers a same-queue pending entry (no wait — same-queue ordering
guarantees prior kernels have completed before the new allocation reads the
address); only cross-queue reuse forces a `wait()` on the originating queue.
Confirmed by 8-step 10-rank Qwen3-32B FSDP run with steady loss decrease,
3.5 s/step.

(`usm_delayfree_alloc.so` — unconditional `wait()` on every free — also
works at 3.7 s/step. Use this if `pending` ever shows issues; it is the
simplest correct workaround.)

**Multi-node (current production)** — 2+ nodes, HSDP — continues to work.
Benchmarked at 19.4 s/step for 32B GRPO on 2 nodes. The reason this works
isn't that "OFI bypasses IPC" (the original explanation was wrong);
empirically, the per-rank tensor sizes / FSDP all-gather patterns under HSDP
differ enough that the cross-stream race window does not get hit, or the
recycled buffer is overwritten with valid data before the comm-stream
kernel reads it. We don't have a clean theoretical explanation for this
particular workaround, but it is reliable.

**Default allocator** also works, at the cost of memory fragmentation
(29 GiB of driver overhead at 32B) leading to OOM around step 4 for
single-node training.

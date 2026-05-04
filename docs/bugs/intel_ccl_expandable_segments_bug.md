# Bug A: oneCCL rejects XPU tensors allocated with `expandable_segments`

**System:** Aurora (ALCF), Intel Max 1550 GPU, `frameworks/2025.3.1` (torch 2.10, oneCCL 2021.17)

> This file documents a single bug. Two related-looking failure modes that
> share the `banned:1` GPU-fault symptom live in their own files:
>
> - `docs/bugs/xpu_pluggable_allocator_record_stream.md` —
>   `XPUPluggableAllocator` causes a step-1 GPU page fault under FSDP2.
> - `docs/bugs/ccl_ipc_handle_cache.md` —
>   `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` IPC-handle stale-vs-accumulation
>   trade-off.

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

## Suggested upstream fix

The fix requires addressing two independent failure modes inside oneCCL / Level
Zero / PyTorch.

### 1. Type check (`ccl_check_usm_pointers`)

Accept virtual-memory-mapped device pointers. If `sycl::get_pointer_type()`
returns `unknown`, perform a secondary check via Level Zero virtual-memory
inspection (e.g. `zeVirtualMemQueryPageSize` or a similar mapped-range query) to
confirm the range is GPU-resident on the device. Alternatively, oneCCL could
expose an opt-in `CCL_SKIP_USM_CHECK=1` to bypass the check at the user's risk.
Note: fixing only the type check is insufficient on its own (see Phase 2 above).

### 2. IPC sharing for virtual-memory ranges

For large tensors, oneCCL calls `zeMemGetIpcHandle` to share buffers across
processes. `zeMemGetIpcHandle` only handles ordinary USM allocations; it does
not handle virtual ranges built from `zeVirtualMemReserve` + `zeVirtualMemMap`.
Two options:

- **a. Use the physical-memory IPC path.** Level Zero exposes IPC for the
  underlying physical allocation (`ze_physical_mem_handle_t` created via
  `zePhysicalMemCreate`). The sender retrieves an IPC handle for the physical
  memory; the receiver opens that handle (analogous to `zeMemOpenIpcHandle` for
  USM) and re-maps it into its own virtual range with `zeVirtualMemMap`. The
  exact API surface for the IPC step has changed across Level Zero spec versions
  — the original report cited `zeVirtualMemSetIpcHandle`, which is not a real
  entry point. The intent is "expose IPC on the backing physical memory and
  remap on the receiver," not a single setter on the virtual range.
- **b. Fall back to the staging-buffer path** when `zeMemGetAllocProperties`
  returns `ZE_MEMORY_TYPE_UNKNOWN`, instead of attempting IPC. Slower but
  always correct.
- **c. Provide `CCL_FORCE_STAGING=1`** to disable zero-copy for all-gather and
  reduce-scatter unconditionally (correctness-first escape hatch).

## Workarounds in this repo

There is no allocator workaround that simultaneously delivers the
`expandable_segments` fragmentation win **and** survives oneCCL collectives.
The two mitigations we use are:

- **Default XPU allocator** (`PYTORCH_ALLOC_CONF` unset): correct under CCL,
  fragments under mixed-size RL workloads. Tunable via `gc:`/`split_size:`.
- **Custom pluggable allocators** (`XPU_USM_ALLOC_SO=...usm_caching_alloc_v2.so`
  etc.): allocate via `sycl::malloc_device` (`zeMemAllocDevice`), so CCL's USM
  check passes. Has its own bug: see
  `docs/bugs/xpu_pluggable_allocator_record_stream.md`.

We do **not** ship `expandable_segments:True` in any production launcher, and
the recipe should refuse to start if it is set. See the recipe entrypoint at
`recipes/dev/grpo_full_finetune_distributed_xpu.py` (`PYTORCH_ALLOC_CONF` log
near the end of `setup()`) — that log should grow into a hard guard.

## Related, not the same bug

| Other file | Bug | Symptom |
|------------|-----|---------|
| `docs/bugs/xpu_pluggable_allocator_record_stream.md` | `XPUPluggableAllocator` step-1 GPU page fault under FSDP2 | banned:1 PDP/PML5 NotPresent |
| `docs/bugs/ccl_ipc_handle_cache.md` | `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` stale eviction at default=1000, accumulation at 65536 | banned:1 PDE around step 28 |
| `docs/bugs/intel_xpu_resource_leak_bug_report.md` | `torch.xpu.empty_cache()` + FSDP `storage.resize_()` UR-handle leak | UR_RESULT_ERROR_OUT_OF_RESOURCES at iter ~70 |

These share `banned:1` symptoms but have independent root causes and independent
fixes. Do not conflate them.

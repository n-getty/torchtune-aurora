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

We are currently running this allocator in production via `XPU_USM_ALLOC_SO`.
It is a complete workaround that does not require any Intel changes.

**Residual limitation vs `expandable_segments`:** interleaved live/free blocks
within the arena cannot coalesce (a live block prevents merging its neighbors).
In the pathological case (every other freed block has a live neighbor), large
tensors still require a new slab. The practical fix in PyTorch would be to
implement `expandable_segments` via `zePhysicalMemAllocate` (physical USM)
rather than `zeVirtualMemReserve`, enabling `zeMemGetIpcHandle` / L0 1.5+
`zePhysicalMemGetIpcHandle` for cross-process sharing.

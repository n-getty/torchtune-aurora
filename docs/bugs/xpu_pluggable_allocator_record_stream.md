# Bug B: `XPUPluggableAllocator` + FSDP causes a step-1 GPU page fault

**System:** Aurora (ALCF), Intel Max 1550 GPU, `frameworks/2025.3.1` (torch 2.10, oneCCL 2021.17)

> Symptom note: this bug ends in a `banned:1` GPU fault, the same outward
> signature as `docs/bugs/intel_ccl_expandable_segments_bug.md` (Bug A) and
> `docs/bugs/ccl_ipc_handle_cache.md` (Bug C). The root causes are independent.
> See the cross-reference table at the bottom.

## Summary

Any custom allocator registered via `torch.xpu.memory.XPUPluggableAllocator`
causes GPU faults under FSDP2-sharded training of large models (32B+, 10 ranks).
The allocator returns proper USM device memory (`ZE_MEMORY_TYPE_DEVICE`), step 0
forward/backward/optim succeeds, then step 1 page-faults during compute (NOT
during a CCL collective). The mechanism is a cross-stream race between FSDP's
comm stream (where AllGather/ReduceScatter run) and the compute stream (where
free-then-realloc cycles recycle a virtual address) — but the precise piece of
PyTorch machinery that ought to prevent the race is still debated. See
"Mechanism — recordStream theory and its rebuttal" below.

The fault dump:

```
Segmentation fault from GPU at 0xff020005b01bd000,
  ctx_id: 1 (CCS) type: 0 (NotPresent), level: 4 (PML5), access: 1 (Write), banned: 1, aborting.
```

(Earlier reports cited the sentinel address `0xffffff8000000000` and a PDP-level
fault — that was a different code path in earlier sub-allocating arena variants.
Current allocators all hit PML5/PTE/PDE NotPresent depending on caching strategy
— see the variant matrix below.)

## What the bug does NOT do

The original framing said the bug was an L0 IPC mismatch. Direct LD_PRELOAD
tracing rules that out:

- `experiments/arena_ipc/diag_2_l0_ipc_shim.c` intercepts `zeMemGetIpcHandle`,
  `zeMemOpenIpcHandle`, `zeMemGetAllocProperties`. In both the failing
  arena-allocator run and the passing default-allocator run, **zero**
  `zeMemGetIpcHandle` calls fire. CCL's intra-node 10-rank collectives do not
  exercise the L0 IPC handle path here.
- The default allocator is byte-for-byte the same workload and passes.
- The SYCL queue vs. context binding is not the cause: `usm_ctx_alloc.so` binds
  to the L0 context (`sycl::aligned_alloc_device(align, sz, dev, ctx)`) and
  fails identically.

These were systematically eliminated in `experiments/arena_ipc/diag_findings.md`
(Phases 1b, 2, 5).

## Reproducer

Any allocator `.so` registered via `XPUPluggableAllocator` will trigger this
on single-node FSDP with ≥10 ranks and 32B+ parameter models. Minimal setup:

```cpp
// trivial_alloc.cpp — passthrough allocator, no pooling, no arena
#include <sycl/sycl.hpp>
#include <cstddef>
extern "C" {
void* xpu_usm_malloc(size_t size, int device, sycl::queue* queue) {
    return sycl::malloc_device(size, *queue);
}
void xpu_usm_free(void* ptr, size_t size, int device, sycl::queue* queue) {
    if (ptr && queue) sycl::free(ptr, *queue);
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

The simple multi-rank `allreduce` / `all_gather` / `reduce_scatter` reproducer
in `experiments/arena_ipc/diag_2_repro_large.py` (32 MiB tensors, 10 ranks)
**does NOT** trigger the bug. Only `FSDP.shard_model()` + an actual training
step does. That implicates FSDP's parameter-shard lifetime management, not raw
CCL.

## Allocator-variant matrix (DEFINITIVE)

Five variants under the same `fsdp_memory_stress.py` (10-rank
Qwen3-32B + FSDP `shard_model` + `reshard_after_forward=True`):

| Variant                    | Caching   | Behavior                                           | Fault level | Fault addr pattern   |
|----------------------------|-----------|----------------------------------------------------|-------------|----------------------|
| **default** (built-in)     | n/a       | All steps PASS                                     | —           | —                    |
| arena                      | yes       | step 0 OK → step 1 fault                           | PML5 (4)    | `0xff0X0005af...000` |
| singleq (pinned-queue)     | yes       | step 0 OK → step 1 fault                           | PML5 (4)    | `0xff040005af6e6000` |
| traceq (instrumented)      | yes       | step 0 OK → step 1 fault                           | PML5 (4)    | `0xff0100057d143000` |
| largeonly (no power-of-2)  | yes       | step 0 OK → fault BEFORE step 1                    | **PTE (0)** | `0xff0X000643f...000`|
| nocache (passthrough)      | no        | fault BEFORE step 0 finishes                       | **PDE (1)** | `0xff0X00bfae...000` |
| **delayfree** (`q->wait()` on free)  | yes  | All steps PASS                                | —           | —                    |
| **pending** (per-queue lazy drain)   | yes  | All steps PASS                                | —           | —                    |
| **caching / caching_v2** (`q->wait()` on cache-hit alloc) | yes | All steps PASS                          | —           | —                    |

Pattern: every cache strategy that does not explicitly synchronize the freeing
or requesting queue faults in step 1; every variant that adds a `queue->wait()`
on the alloc or free side passes.

## Mechanism — recordStream theory and its rebuttal

There are two on-record explanations for *why* `queue->wait()` fixes the bug.
Both predict the observed pass/fail pattern. We have not picked one
definitively; both reasonable researchers should be able to read this and
decide what next experiment to run.

### Theory 1 (`recordStream` no-op)

`torch/include/torch/csrc/xpu/XPUPluggableAllocator.h` exposes a
`set_record_stream_fn` hook. The Python wrapper
(`torch/xpu/memory.py:260-300`) calls
`torch._C._xpu_customAllocator(alloc_fn, free_fn)` and never invokes
`set_record_stream_fn`. Disassembly of `XPUPluggableAllocator::recordStream`
in `libtorch_python.so`:

```
cmpq $0x0, 0x78(%rdi)    # if record_stream_fn_ == nullptr
je   ret                 # return without doing anything
```

So `recordStream()` is a silent no-op for every Python-loaded pluggable
allocator. If FSDP relies on `recordStream` to keep the AllGather output buffer
alive on the comm stream while the compute stream lets it go out of scope, the
allocator will recycle the virtual address; the comm-stream kernel later issues
a load/store against the now-stale mapping and the GPU page-faults.

This theory explains the variant matrix: `delayfree` / `pending` /
`caching{,_v2}` all add the synchronization that `recordStream` would have
provided.

### Theory 2 (FSDP2 doesn't call `recordStream` at all)

A follow-up audit (`docs/reports/archive/allocator_deep_analysis_20260425.md`) of the
shipped FSDP2 source on Aurora found:

```
$ grep -rn 'record_stream' torch/distributed/fsdp/_fully_shard/*.py
(no results)

$ grep -n 'wait_stream\|wait_event' torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py
273:    all_gather_stream.wait_stream(all_gather_copy_in_stream)
361:    device_handle.current_stream().wait_event(all_gather_event)
531:    reduce_scatter_stream.wait_stream(current_stream)
```

FSDP2 synchronizes cross-stream access via `wait_stream` / `wait_event`, never
via `tensor.record_stream()`. Under this view the allocator-side `recordStream`
hook is irrelevant to the AllGather/ReduceScatter path, and the actual race is
either:

- a different cross-stream race that PyTorch's allocation scheduling can run
  ahead of the event insertion the allocator never sees, or
- OFI/libfabric memory-region (MR) accumulation: a fresh `sycl::malloc_device`
  produces a new VA, OFI registers it; once that VA is freed and another
  address is allocated, the old OFI registration points to invalid memory.
  `queue->wait()` in the allocator transitively forces pending DMA to
  complete before the address is recycled.

Theory 2 also fits the variant matrix: anything that defers reuse of a freshly
freed VA until the device is idle prevents the failure, regardless of whether
the underlying problem is `recordStream`-style cross-stream visibility or
OFI-style stale registrations.

### What we know without committing to a theory

- Bug is **strictly allocator-specific**: same launch, same `LD_PRELOAD`, same
  10-rank Qwen3-32B run; default allocator passes, every custom variant without
  synchronization fails at step 1.
- `zeMemGetIpcHandle` is **never called**. Bug is not in the L0 IPC path.
- Queue identity is **not the cause** (`singleq` and `usm_ctx_alloc` both fail).
- No-cache passthrough fails **earlier** than caching variants (PDE level vs
  PML5/PTE), consistent with "free-then-immediate-reuse" being the mechanism
  rather than "stale free-list entries."
- Adding `queue->wait()` on alloc-time cache-hit (`usm_caching_alloc.cpp`,
  `usm_caching_alloc_v2.cpp`) or on free (`usm_delayfree_alloc.cpp`) or
  per-queue lazy drain (`usm_pending_alloc.cpp`) **all** fix the failure,
  with overhead in roughly that order from cheapest to most expensive.

## Suggested upstream fixes

Two complementary targets, depending on which theory applies:

### 1. Wire `set_record_stream_fn` in PyTorch's Python wrapper

Two-line change in `torch/xpu/memory.py`:

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
`set_record_stream_fn`. This addresses Theory 1 directly.

### 2. Make OFI memory-region cache more robust on Aurora

`FI_MR_CACHE_MONITOR=disabled` (current default on Aurora because the Slingshot
`memhooks` monitor crashes) prevents OFI from auto-deregistering stale MRs.
Validating `FI_MR_CACHE_MONITOR=userfaultfd` or `memhooks` on the current
driver stack would address Theory 2 and would also help the unrelated
ccl-IPC-handle-cache bug. This is an upstream / driver investigation, not a
torchtune change.

## Workarounds shipped in this repo

We do not bind any of the unsafe variants by default. The allowed allocators
are the ones with built-in stream-safety hacks:

- `recipes/dev/usm_caching_alloc.cpp` (gen1) — power-of-2 buckets;
  `queue->wait()` at alloc time on cache hit. 4-param free signature.
- `recipes/dev/usm_caching_alloc_v2.cpp` (gen3) — exact-aligned large buckets +
  power-of-2 small buckets + `queue->wait()` at alloc time on cache hit + OOM
  retry releasing all cached blocks. **Preferred** for new pluggable runs.
- `experiments/arena_ipc/usm_pending_alloc.cpp` — per-queue lazy drain on
  alloc; only sync when reusing a cross-queue pointer. Fastest of the three
  in the multi-step Qwen3-32B test (3.5 s/step).
- `experiments/arena_ipc/usm_delayfree_alloc.cpp` — unconditional
  `queue->wait()` on every free; simplest correct variant (3.7 s/step).

The variants without the wait workaround are diagnostic only and should not
be used in production:

- `recipes/dev/usm_arena_alloc.cpp` (gen2) — exact alignment, no wait. **Unsafe
  with FSDP2.** Either retire it or port the alloc-time `queue->wait()` from
  gen1.
- `experiments/arena_ipc/usm_nocache_alloc.cpp`, `usm_largeonly_alloc.cpp`,
  `usm_singleq_alloc.cpp`, `usm_traceq_alloc.cpp`, `usm_ctx_alloc.cpp`,
  `usm_leak_alloc.cpp` — diagnostic; all fail.

### Current production stance

Production launchers explicitly **unset** `XPU_USM_ALLOC_SO`:

- `recipes/dev/run_qwen3_30b_ep4_vllm_2node.sh:54`
- `recipes/dev/run_qwen3_30b_ep4_vllm_3node.sh:51`
- `recipes/dev/run_qwen3_30b_ep8_vllm_2node.sh:49`
- `recipes/dev/run_qwen3_30b_ep8_vllm_3node.sh:56`
- `recipes/dev/run_qwen3_30b_ep16_vllm_2node.sh:59`

These rely on the default XPU allocator. The pluggable path is opt-in for
specific workloads (e.g. multi-node OFI-only runs that hit allocator
fragmentation). The recipe entrypoint
(`recipes/dev/grpo_full_finetune_distributed_xpu.py:231-235`) currently
accepts any `XPU_USM_ALLOC_SO` value without checking the basename — that
should be tightened to refuse known-unsafe variants unless an explicit
override env var is set. (Tracked separately.)

### Multi-node empirical workaround

2+ node HSDP runs continue to work in practice. The original explanation ("OFI
bypasses IPC") was wrong; empirically, the per-rank tensor sizes / FSDP
all-gather patterns under HSDP differ enough that the cross-stream race window
is rarely hit, or the recycled buffer is overwritten with valid data before the
comm-stream kernel reads it. We do not have a clean theoretical explanation,
but the configuration is reliable.

## Related, not the same bug

| Other file | Bug | Symptom |
|------------|-----|---------|
| `docs/bugs/intel_ccl_expandable_segments_bug.md` | `expandable_segments` virtual-mem pointers rejected by CCL | `invalid usm pointer type: unknown` |
| `docs/bugs/ccl_ipc_handle_cache.md` | `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` stale eviction at default=1000, accumulation at 65536 | banned:1 PDE around step 28 |
| `docs/bugs/intel_xpu_resource_leak_bug_report.md` | `torch.xpu.empty_cache()` + FSDP `storage.resize_()` UR-handle leak | UR_RESULT_ERROR_OUT_OF_RESOURCES at iter ~70 |

Investigation history (LD_PRELOAD shims, allocator-variant matrix, Phase 1b–9
findings) is in `experiments/arena_ipc/diag_findings.md`. Independent
re-analysis disputing the recordStream theory is in
`docs/reports/archive/allocator_deep_analysis_20260425.md`.

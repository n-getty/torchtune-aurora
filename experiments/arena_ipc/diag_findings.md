# Arena IPC Bug Investigation — Findings

## Summary

The "SYCL context mismatch" hypothesis in Bug Report 2 is **WRONG**. Empirical evidence:

### Phase 1b: Context comparison (PASSED)
- PyTorch's `getCurrentXPUStream` queue and a fresh sycl::queue share the **same L0 context**.
- Both default and pluggable allocations report `ZE_MEMORY_TYPE_DEVICE` in PyTorch's context.
- L0 context handle: `0x0000562d6945b118` for both → no mismatch.

### Phase 2: L0 IPC tracing (DEFINITIVE NEGATIVE)
- 10-rank FSDP+Qwen3-32B with arena allocator + LD_PRELOAD shim on `zeMemGetIpcHandle`/`zeMemOpenIpcHandle`.
- Bug reproduces (GPU page fault step 1, address `0xff020005b01bd000`, PML5 NotPresent, banned:1).
- **ZERO `[L0_IPC]` log lines** — neither `zeMemGetIpcHandle` nor `zeMemOpenIpcHandle` were called.
- Therefore the failure is NOT in the L0 IPC handle exchange path.

### Page fault address pattern
- This run: `0xff02...`, `0xff03...`, `0xff05...` — these look like real allocated USM pointers in the high-address range XPU uses for device memory.
- NOT the sentinel `0xffffff8000000000` from earlier reports — that was a different failure mode (likely sub-allocation in older arena versions).
- PML5 NotPresent at level 4 = page-table walk failed for an address that should be mapped.

### Simple 2-rank reproducer DOES NOT trigger the bug
- 2-rank `diag_2_repro_large.py` with 32 MiB tensors PASSES with arena allocator.
- 10-rank simple all_gather/reduce_scatter PASSES.
- Only **FSDP shard_model + actual training step** triggers the failure.

## Likely New Hypothesis

The failure is NOT in CCL's IPC path. Possibilities to investigate:
1. **FSDP all-gather of sharded parameters** — uses GPU kernels (sycl::copy in unshard) that page-fault on caching-allocator-recycled pointers under specific conditions.
2. **Reshard-after-forward** — FSDP frees a parameter shard during fwd; in step 1, the cached pointer gets handed back, but the kernel scheduled on it is targeting a stale virtual mapping.
3. **PyTorch's XPU stream pool (32 queues)** — the pluggable allocator's `getCurrentXPUStream(device)` returns from a pool; if the pluggable free-list returns a pointer that was malloc'd on queue A but is now used by a kernel on queue B, the L0 driver may not have properly synchronized the address-space view.

## Next investigation steps (after queue exhaustion)

1. Add a debug counter to arena allocator: log per-pointer (malloc_queue, free_queue) — look for cross-queue reuse.
2. Force allocator to use a **single queue per device** (cached at first allocation) instead of trusting the per-call queue → if this fixes it, root cause is multi-queue reuse.
3. Try `sycl::aligned_alloc_device(align, size, device, ctx)` instead of `sycl::malloc_device(size, queue)` — the former pins to a context, not a queue.
4. The "sub-allocator returns mid-slab pointer" hypothesis from the original bug report can also be ruled out for the current code — usm_arena_alloc.cpp is now exact-aligned direct sycl::malloc_device per allocation, no slab.

## Phase 2b: Default-allocator baseline (CONFIRMS)

Same 10-rank Qwen3-32B FSDP test, same LD_PRELOAD shim, but WITHOUT `XPU_USM_ALLOC_SO`:
- Step 0: loss=13.7705, 18.67 GiB allocated
- Step 1: loss=13.1466, 18.67 GiB allocated
- ALL STEPS PASSED.
- ZERO `[L0_IPC]` log lines (confirms shim doesn't break things and CCL doesn't take the IPC path here either).

Therefore:
1. The bug is **strictly allocator-specific** — same launch, only difference is `XPU_USM_ALLOC_SO`.
2. The bug is **NOT in CCL's IPC path** — `zeMemGetIpcHandle` is never called in either run.
3. The bug fires during **regular GPU kernel work** in step 1.

## Revised root-cause hypothesis

The most likely candidate is **per-call SYCL queue mismatch in the arena allocator**:

`usm_arena_alloc.cpp::xpu_usm_malloc` calls `sycl::malloc_device(sz, *queue)`. PyTorch passes the queue from `getCurrentXPUStream(device)`, which rotates through a 32-queue pool. Each `sycl::malloc_device(sz, queue)` call associates the allocation with that queue's L0 command-list/command-queue.

When the cached pointer is later returned by another `xpu_usm_malloc` call, the new caller may be using a *different* queue from the pool. The kernel scheduled on queue B reads/writes a USM pointer that was implicitly bound to queue A. Under L0, USM pointers are valid in any queue **within the same context**, so this *should* work — but the page fault suggests something about the L0 driver's address-space view doesn't propagate cleanly across queue switches when the pointer is recycled.

Alternative: the FSDP unshard path may issue an asynchronous `sycl::copy` or all-gather that depends on the parameter shard being owned by a specific queue.

### Test plan
1. Modify arena allocator to remember the queue per allocation. On cache hit, refuse to return the cached pointer if the requesting queue differs (force a fresh malloc) → if this fixes it, queue-binding is the root cause.
2. Alternative: switch to `sycl::aligned_alloc_device(ALIGNMENT, sz, device, ctx)` — this binds to **context** not **queue**, which is what PyTorch's default allocator does.

## Phase 5: Test queue→context binding fix (FAILED to fix)

Built `usm_ctx_alloc.so` — replaced `sycl::malloc_device(sz, *queue)` with
`sycl::aligned_alloc_device(ALIGNMENT, sz, dev, ctx)` (binds to context, not queue).

Result: identical failure pattern. Step 0 PASSES, step 1 page-faults at `0xff050005b1dcb000`.
Therefore the bug is **NOT queue vs. context binding**.

## Phase 6: Test reshard_after_forward=False

With original arena allocator + `reshard_after_forward=False`:
- Step 0 fwd starts but fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES` (UR:40).
- No GPU page fault — different failure mode.
- This MAY be just memory pressure (params not freed during fwd) OR may be the
  same root cause manifesting differently (UR resource exhaustion in non-default
  allocator path).

## Status

- Bug is **strictly allocator-specific** (default works, every custom variant fails).
- Bug is **NOT in CCL IPC** (zeMemGetIpcHandle never called).
- Bug is **NOT in queue-binding** (context-bound allocator fails identically).
- Step 0 always passes, step 1 always fails — implicates **second-iteration reuse** of cached pointers.

## Phase 7: Allocator-variant matrix (DEFINITIVE)

Built five variants and ran each against the same `fsdp_memory_stress.py` (10-rank
Qwen3-32B + FSDP `shard_model` + `reshard_after_forward=True`):

| Variant                    | Caching   | Behavior                                           | Fault level | Fault addr pattern   |
|----------------------------|-----------|----------------------------------------------------|-------------|----------------------|
| **default** (built-in)     | n/a       | All steps PASS                                     | —           | —                    |
| arena (caching)            | yes       | step 0 OK → step 1 fault                           | PML5 (4)    | `0xff0X0005af...000` |
| singleq (pinned-queue)     | yes       | step 0 OK → step 1 fault                           | PML5 (4)    | `0xff040005af6e6000` |
| traceq (instrumented)      | yes       | step 0 OK → step 1 fault                           | PML5 (4)    | `0xff0100057d143000` |
| largeonly (no power-of-2)  | yes       | step 0 OK → fault BEFORE step 1                    | **PTE (0)** | `0xff0X000643f...000`|
| **nocache** (passthrough)  | NO        | fault BEFORE step 0 finishes (right after shard)   | **PDE (1)** | `0xff0X00bfae...000` |
| default + ZE_AFFINITY_MASK | n/a       | All steps PASS                                     | —           | —                    |
| arena  + ZE_AFFINITY_MASK  | yes       | fault BEFORE step 0 (worse than no-affinity)       | PML5 (4)    | —                    |

### Cross-queue reuse (traceq results)

`USM_TRACE_LIMIT=10000`, 10-rank fsdp_memory_stress:
- Total cache hits logged: ~10000
- Hits where caller's queue ≠ malloc queue: 1770 (1.8%)
- Cross-queue queue-pointer pairs differ by exactly 0x06D0 (consistent layout offset)

So cross-queue reuse IS happening — but pinning the queue (singleq) does NOT fix the bug.
Therefore queue identity is not the cause.

### Faulting addresses

The page-fault address never matches a pointer present in the trace map. They are
all in the form `0xff0X..0..00X..000` — these look like FSDP all-gather/reduce-scatter
intermediate buffers, not application allocations.

Note: each allocator design produces a *different* fault level (PTE → PDE → PML5),
which means each variant breaks the L0 page tables in a different way.

## Conclusion

- Bug fires for **every** custom allocator we built, regardless of caching, queue,
  context, or bucket strategy.
- Disabling caching entirely makes the bug fire **sooner**, not later — confirms the
  cache-reuse hypothesis is wrong.
- The default allocator passes consistently. So the failure mode is something the
  default does that **no Python-side pluggable allocator can do**.

## Phase 8: ROOT CAUSE FOUND — `recordStream` is a no-op

Inspection of `torch/include/torch/csrc/xpu/XPUPluggableAllocator.h`:

```cpp
struct XPUPluggableAllocator {
  void recordStream(const c10::DataPtr&, c10::Stream stream) override;
  ...
  void set_record_stream_fn(
      std::function<void(void* ptr, sycl::queue* queue)> record_stream_fn) {
    record_stream_fn_ = std::move(record_stream_fn);
  }
  ...
  std::function<void(void* ptr, sycl::queue*)> record_stream_fn_;
};
```

Disassembly of `XPUPluggableAllocator::recordStream` in libtorch_python.so confirms:
```
cmpq $0x0, 0x78(%rdi)   # if (record_stream_fn_ == nullptr)
je   ret                # return immediately — NO-OP
```

The Python `torch.xpu.memory.XPUPluggableAllocator` constructor calls
`torch._C._xpu_customAllocator(alloc_fn_addr, free_fn_addr)` and **never wires
`set_record_stream_fn`**. So for every Python-loaded pluggable allocator,
`recordStream()` is silently a no-op.

### Why this is the bug

FSDP2 issues all-gather / reduce-scatter on a separate **communication stream**,
while the **compute stream** still has pending kernels referencing those
allocations. To keep the allocator from freeing a buffer that the comm stream is
still reading, PyTorch calls `recordStream(dataPtr, comm_stream)` — this tells
the caching allocator "do not return this pointer to the free list until the
comm stream has consumed it."

With the pluggable allocator, that call is a no-op. The compute stream releases
the tensor (Python refcount → 0 → `xpu_usm_free`); the cache happily hands the
pointer back to a fresh allocation; and the comm stream's still-pending
all-gather reads the now-reused virtual address. The reused virtual address has
been re-mapped (or the original L0 page-table entry has been invalidated for
recycling) → **GPU page fault**.

### Why the data fits perfectly

- **Default allocator works**: it has internal `recordStream` wiring.
- **Every custom variant fails**: all skip recordStream.
- **Caching delays the fault to step 1**: in step 0 the comm-stream kernel sees
  the original allocation; in step 1 the same virtual address has been recycled
  and remapped.
- **No-cache fails sooner**: every `xpu_usm_free` immediately returns the
  pointer to L0, which can recycle the address space immediately — the
  in-flight comm stream now reads an address with no L0 page-table entry at
  all. Hence the **PDE level** fault (page directory missing), not PML5/PTE.
- **Pinned-queue (singleq) doesn't help**: the bug isn't queue identity, it's
  early-free.
- **Largeonly produces PTE-level faults**: exact-aligned recycling matches the
  freed slot exactly, so the page-table entry walk reaches the leaf level
  before discovering the mapping is gone.
- **The faulting addresses don't appear in the trace map**: they are FSDP
  communication intermediates whose memory has been freed and rebound.
- **`reshard_after_forward=False` masks it**: parameter shards are never freed
  in step 1 → no recycle → no fault (it OOMs first instead).

### The fix (for whoever maintains the allocator)

Two options:

1. **Wire `set_record_stream_fn`** in the Python wrapper. The Python
   `XPUPluggableAllocator.__init__` should accept an optional
   `record_stream_fn_name` and pass its address to a C++ setter.

2. **Default `record_stream_fn_` to a callback that drops the cached
   pointer back to L0 only after queue completion**. The pluggable allocator
   could expose a built-in default (use `sycl::queue::ext_oneapi_submit_barrier`
   or store pointer→stream pairs and defer free until the next `emptyCache`).

Until either is fixed, our caching allocator can hack around this: on every
malloc, defer the actual return-to-pool of any "freed" pointer that has been
in the free list for less than a few seconds. But this is fragile and we
should NOT ship a workaround like that — the right answer is to upstream
the wiring fix to `torch/xpu/memory.py`.

## Phase 9: Empirical confirmation — `delayfree` allocator PASSES

Built `usm_delayfree_alloc.cpp`: same caching as arena, but `try_free()` calls
`caller_q->wait()` before returning the pointer to the free list. This forces the
freeing stream's pending kernels to complete before the address can be recycled —
i.e. it does in code what `recordStream` would have ensured if wired properly.

Same 10-rank Qwen3-32B `fsdp_memory_stress.py`, 3 steps:

```
POST-SHARD (19.8s): allocated=0.00 GiB
Step 0 complete: 7.9s (fwd=1.8s, bwd=4.7s) loss=13.7205
Step 1 complete: 3.9s (fwd=1.2s, bwd=2.4s) loss=13.2160
Step 2 complete: 3.9s (fwd=1.2s, bwd=2.4s) loss=12.9313
```

**ALL STEPS PASS.** Every other custom allocator faulted at step 0 or step 1.

8-step rerun also clean (loss decreasing steadily, suggesting actual training):

```
Step 0 complete: 7.2s loss=13.4316
Step 1 complete: 3.8s loss=13.2101
Step 2 complete: 3.7s loss=12.7356
Step 3 complete: 3.7s loss=12.7367
Step 4 complete: 3.7s loss=12.5530
Step 5 complete: 3.7s loss=12.5010
Step 6 complete: 3.7s loss=12.4381
Step 7 complete: 3.7s loss=12.2287
```

Adding `caller_q->wait()` on free is the only behavioral change vs the failing
arena allocator. This is **definitive evidence** that the bug is "freed buffers
recycled while another stream still has pending reads," exactly the scenario
`recordStream` is designed to prevent — and which the pluggable allocator's
no-op `recordStream` fails to prevent.

(Caveat: `caller_q->wait()` on every free is expensive — this allocator runs but
is not production-quality. The right fix is to either wire `set_record_stream_fn`
in the Python wrapper, or implement a per-pointer pending-stream list inside the
allocator's free path.)

### `usm_pending_alloc.so` — production-quality variant

`usm_pending_alloc.cpp` does the right thing: each freed pointer goes onto a
per-size pending queue paired with the freeing queue. On allocation, prefer a
pending pointer whose freeing queue matches the requesting queue (no wait —
in-order on a single queue guarantees the prior kernel has completed by the
time the allocation reads the address). Only when no same-queue pending entry
is available do we drain a pending entry on a different queue.

Same 8-step Qwen3-32B test:

```
Step 0 complete: 7.3s loss=13.6577
Step 1 complete: 3.5s loss=13.1240
Step 2 complete: 3.5s loss=12.8308
Step 3 complete: 3.5s loss=12.7103
Step 4 complete: 3.5s loss=12.5827
Step 5 complete: 3.5s loss=12.4589
Step 6 complete: 3.5s loss=12.3351
Step 7 complete: 3.5s loss=12.4059
```

ALL STEPS PASS, **slightly faster** than delayfree (3.5s vs 3.7s/step) — because
most allocs hit the same-queue fast path and avoid the synchronous drain.

**Use this allocator** in production until the upstream PyTorch
`set_record_stream_fn` wiring is fixed.

## Practical recommendation (today)

Do not use `XPUPluggableAllocator` with FSDP2 + cross-stream collectives on
Aurora's current PyTorch build. Use the default XPU allocator and tune via
`PYTORCH_XPU_ALLOC_CONF` / `expandable_segments=True` instead.

The original Bug Report 2 attribution to "SYCL context mismatch" is wrong — see
Phase 1b/Phase 2 above. The actual root cause is the missing
`set_record_stream_fn` wiring in the Python `XPUPluggableAllocator`
constructor: `recordStream()` is a silent no-op, so cross-stream FSDP
collectives free buffers that are still being read by the comm stream.

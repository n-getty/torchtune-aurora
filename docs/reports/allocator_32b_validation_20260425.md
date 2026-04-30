# 32B Allocator Validation Report — 2026-04-25

Job 8450884, 2-node debug queue (x4105c6s0b0n0 + x4105c6s1b0n0), 1h walltime.
Training: x4105c6s1b0n0 (12 tiles FSDP2), vLLM: x4105c6s0b0n0 (3 replicas TP=4).
Model: Qwen3-32B, G=16, fbs=4, max_gen=128, XCCL 2-hop weight sync.

## Summary

Pluggable allocators (usm_caching_alloc_v2.so) cannot work at 32B scale. The default
XPU allocator with `garbage_collection_threshold:0.95` is the correct approach —
validated 5/5 steps clean at 40.1s/step average. `FI_MR_CACHE_MONITOR=userfaultfd`
also validated clean on single-node (XeLink) with zero performance regression; needs
cross-node (Slingshot) validation before production use.

## Test matrix

| Run | Allocator | gc | FI_MR_CACHE_MONITOR | Result | Avg step (s) |
|-----|-----------|-----|---------------------|--------|-------------|
| run3 | gen3 v2.so (debug=2) | unset | disabled | UR:40 step 0 optimizer | — |
| run4 | gen3 v2.so + OOM retry (debug=1) | unset | disabled | banned:1 step 1 | — |
| gc95 | **default** | **0.95** | disabled | **5/5 clean** | **40.3** |
| ufd | **default** | **0.95** | **userfaultfd** | **5/5 clean** | **40.1** |

## Gen3 allocator failure analysis

### Run 3: UR:40 at step 0 optimizer (no OOM retry)

The gen3 allocator (`usm_caching_alloc_v2.cpp`) creates individual L0 allocations per
tensor via `sycl::malloc_device`. By the time the optimizer runs on step 0:

- Large pool: ~1,785 driver allocs per tile (7–22 MiB each, FSDP unshard buffers +
  parameter shards + gradient intermediates)
- Small pool: ~3,545 driver allocs per tile (<1 MiB each, metadata, scalars)
- Cache hit rate: 94.5% (1.09M hits / 64K new allocs across all devices)
- 206 large blocks (30.8 GiB) cached in free lists — forward/backward activations
  no longer needed but still occupying L0 VA space

AdamW `_multi_tensor_adam` called `torch._foreach_sqrt(device_exp_avg_sqs)`, which
tried to allocate output tensors. `sycl::malloc_device` threw
`UR_RESULT_ERROR_OUT_OF_RESOURCES` (error 40) on rank 6 — L0 had no remaining VA
space because all 64 GiB HBM was consumed by live + cached allocations.

Root cause: the gen3 allocator never returns memory to L0. Peak memory =
sum(all driver allocs ever made), not sum(live allocs). At 32B with FSDP2 on 12 tiles,
this exceeds 64 GiB per tile.

The log grew to 2.2M lines (145 MB) in 12 minutes with `USM_ALLOC_DEBUG=2`.

### Run 4: banned:1 at step 1 (with OOM retry)

Added OOM retry mechanism to the gen3 allocator:

```
When sycl::malloc_device fails:
  1. queue->wait() — drain pending work
  2. sycl::free() all cached blocks in both pools on the device
  3. Retry sycl::malloc_device
```

Step 0 completed successfully:
- OOM triggered on all 12 devices during optimizer step
- Each device released 206 cached blocks (30.8 GiB) + 61 small blocks (2.2 MiB)
- Retry succeeded — optimizer allocated exp_avg + exp_avg_sq
- Total step 0 time: 108.4s (opt=48.7s due to first-time AdamW state init + cache release)
- XCCL weight sync: 707 params, 61.02 GiB staged to CPU in 3.7s

Step 1 crashed immediately after generation:

```
Segmentation fault from GPU at 0xff00000755214000, ctx_id: 1 (CCS)
  type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
```

Root cause: the OOM retry called `sycl::free()` on cached blocks, which deallocated
their L0 virtual addresses. CCL's IPC handle cache still referenced those VAs from
step 0's AllGather operations. When step 1's FSDP2 AllGather tried to reuse a cached
IPC handle, L0 found the VA was freed → `banned:1`.

This is the same failure mode as PyTorch's internal GC (which caused the step-28
crash with gc:0.8), just triggered earlier because our OOM retry releases more
aggressively.

### Why pluggable allocators fundamentally cannot work at 32B

The default XPU allocator uses `expandable_segments` — large contiguous L0 allocations
(segments) from which individual tensors are suballocated. Key properties:

1. **Fewer L0 VAs**: ~20–50 segments vs ~5,000 individual allocs with our allocator
2. **GC reclaims suballocations without freeing segments**: when a suballocated block is
   freed internally, the segment VA stays alive. CCL's IPC handles reference segment VAs,
   which are never invalidated by GC.
3. **Suballocation eliminates per-tensor L0 overhead**: page table entries, VA metadata,
   IPC handle bookkeeping all scale with number of L0 allocs, not number of tensors.

Any pluggable allocator that creates per-tensor L0 allocs will hit the same problem:
either it pools everything and OOMs (run 3), or it releases on OOM and invalidates
CCL's handles (run 4). The only fix would be implementing suballocation within the
pluggable allocator — essentially reimplementing PyTorch's CachingAllocator.

## Default allocator with gc:0.95

`PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95`, no `XPU_USM_ALLOC_SO`.

### Step timings

| Step | total (s) | gen (s) | grpo (s) | opt (s) | other (s) |
|------|-----------|---------|----------|---------|-----------|
| 0 | 40.1 | 20.3 | 14.6 | 1.4 | 3.6 |
| 1 | 41.7 | 20.1 | 16.8 | 0.1 | 4.4 |
| 2 | 40.4 | 19.2 | 16.0 | 0.1 | 4.8 |
| 3 | 37.8 | 18.1 | 14.6 | 0.1 | 4.7 |
| 4 | 41.4 | 19.4 | 16.9 | 0.1 | 4.7 |

Average: 40.3s/step. Optimizer: 1.4s on step 0 (AdamW state init), 0.1s thereafter.

### Memory profile (rank 0, PRE-BWD)

| Step | l0_free (GiB) | torch_resv (GiB) | external (GiB) | torch_alloc (GiB) |
|------|--------------|------------------|----------------|-------------------|
| 0 | 28.52 | 33.87 | 1.60 | 19.74 |
| 1 | 11.91 | 50.35 | 1.72 | 36.78 |
| 2 | 2.58 | 59.64 | 1.76 | 36.71 |
| 3 | 2.56 | 59.64 | 1.79 | 36.43 |
| 4 | 2.53 | 59.64 | 1.81 | 36.83 |

Between-step memory (all ranks): allocated=28.45 GiB, reserved=59.13 GiB, gap=30.68 GiB.

Key observations:
- **torch_resv stabilizes at 59.64 GiB from step 2 onward** — the allocator's
  expandable segments have fully expanded and don't grow further
- **l0_free stabilizes at ~2.5 GiB** — tight but constant
- **external grows at ~30 MiB/step** (1.60 → 1.81 over 5 steps)
- **GC never fired** — the pool satisfies all requests from cache after step 1.
  gc threshold is 0.95 × 63.98 = 60.78 GiB; torch_resv is 59.64, just below it.
  But even if torch_resv exceeded the threshold, GC only triggers on allocation
  failure, and the pool has 30+ GiB of cached free blocks available.

### Why gc threshold doesn't matter (in practice)

GC triggers when: (1) an allocation fails (can't find a block in the pool AND
`sycl::malloc_device` fails), AND (2) `reserved > threshold × total_device_memory`.

After step 1, FSDP2 training has consistent allocation patterns — the same tensor
sizes repeat each step. The pool warms up during steps 0–1 and then satisfies 100%
of requests from cache. No allocation failures → no GC → threshold is irrelevant.

Both gc:0.6 (24 steps, job 8450367) and gc:0.95 (5 steps, this test) show identical
behavior: GC never fires, memory is flat. The threshold only matters if an unexpected
allocation pattern occurs (new tensor size not in the pool).

### Long-run projection

- torch_resv: FLAT at 59.64 GiB (no growth)
- external: **+30 MiB/step** (3× the earlier 10 MiB/step estimate from single-node tests)
- l0_free starts at ~2.5 GiB → **reaches 0 at ~step 85**
- At l0_free=0, training allocations still come from the pool's 30+ GiB of cached
  blocks — so the training loop itself continues. The failure point is whether CCL
  needs fresh L0 allocations during a collective (AllGather, ReduceScatter). If it
  does, the crash returns as UR:40.
- **Estimated safe run length: ~80 steps.** Beyond that is uncertain — CCL's internal
  allocation patterns determine whether it can operate with l0_free=0. Production
  runs should use checkpoint-restart with `save_every_n_steps=20`.
- **userfaultfd could eliminate this constraint entirely**: if OFI auto-deregisters
  stale MR entries on Slingshot, external growth may flatten to zero, making runs
  unlimited. This is the highest-leverage experiment remaining (see below).

## FI_MR_CACHE_MONITOR=userfaultfd validation

Same configuration as gc:0.95 test, but with `FI_MR_CACHE_MONITOR=userfaultfd`
instead of `disabled`.

### Step timings

| Step | total (s) | gen (s) | grpo (s) | opt (s) | other (s) |
|------|-----------|---------|----------|---------|-----------|
| 0 | 40.1 | 20.5 | 14.6 | 1.4 | 3.5 |
| 1 | 41.0 | 20.0 | 16.3 | 0.1 | 4.4 |
| 2 | 40.1 | 19.2 | 15.9 | 0.1 | 4.6 |
| 3 | 38.6 | 19.0 | 14.2 | 0.1 | 4.9 |
| 4 | 40.7 | 19.5 | 16.1 | 0.1 | 4.8 |

Average: 40.1s/step — zero performance regression vs disabled.

### Memory comparison (rank 0, PRE-BWD)

| Step | ufd external (GiB) | disabled external (GiB) | ufd l0_free (GiB) | disabled l0_free (GiB) |
|------|--------------------|-----------------------|-------------------|-----------------------|
| 0 | 1.75 | 1.60 | 28.36 | 28.52 |
| 1 | 1.87 | 1.72 | 11.77 | 11.91 |
| 2 | 1.89 | 1.76 | 2.45 | 2.58 |
| 3 | 1.90 | 1.79 | 2.44 | 2.56 |

External baseline is ~0.13 GiB higher with userfaultfd (the uffd mechanism itself
allocates tracking structures). Growth rate is similar (~20–30 MiB/step in both).
torch_resv is identical (59.64 GiB) in both.

### Significance

`userfaultfd` makes the OFI layer automatically invalidate MR cache entries when a
VA is freed (via mmap/munmap tracking). If this works cross-node (Slingshot), it
would prevent the fundamental `banned:1` failure mode — even if GC frees a segment,
OFI would notice the VA change and create fresh MR entries instead of using stale ones.

**Caveat**: this test used `torch.distributed.run --standalone` (single-node, XeLink
only). The concern about userfaultfd was specifically about the CXI provider
(Slingshot cross-node fabric). Validating cross-node requires `mpiexec --pmi=pmix`
in a multi-node production launch.

## userfaultfd cross-node validation (Test CL)

Job 8450921, 2-node debug queue, `FI_MR_CACHE_MONITOR=userfaultfd`, gc:0.95,
`torch.distributed.run --standalone` on training node (SSH launch, not mpiexec).
50 steps requested; crashed at step 8.

### Result: userfaultfd works cross-node but does NOT prevent banned:1

**Good news**: userfaultfd is fully compatible with CXI/Slingshot cross-node communication.
7 clean training steps completed with XCCL 2-hop weight sync over Slingshot. Zero CXI
compatibility issues.

**Bad news**: Crashed at step 8 with `banned:1` — the same fundamental failure mode as
`FI_MR_CACHE_MONITOR=disabled`. userfaultfd operates at the wrong layer.

### Memory timeline (rank 0, PRE-BWD)

| Step | external (GiB) | delta (MiB) | l0_free (GiB) | torch_resv (GiB) |
|------|---------------|-------------|---------------|-------------------|
| 0 | 2.38 | — | 27.74 | 33.87 |
| 1 | 2.50 | +123 | 11.14 | 50.35 |
| 2 | 2.52 | +20 | 1.83 | 59.64 |
| 3 | 2.53 | +10 | 1.81 | 59.64 |
| 4 | 2.54 | +10 | 1.80 | 59.64 |
| 5 | 2.56 | +20 | 1.79 | 59.64 |
| 6 | 2.58 | +20 | 1.77 | 59.64 |

Steady-state external growth: ~10-20 MiB/step (vs ~30 MiB/step with `disabled`).
Reduced but NOT eliminated.

### Crash sequence

Step 6 POST-BWD: rank 0's torch_resv expanded to 62.54 GiB (other ranks stayed at 59.64).
This exceeded the gc:0.95 threshold of 60.78 GiB. At step 7, GC fired — freed segments,
released VAs. External spiked from 2.58 → 4.84 GiB as CCL re-registered OFI MR entries.
XCCL weight sync slowed 2.4× (MR re-registration overhead). Step 8: `banned:1`.

### Why userfaultfd doesn't help

userfaultfd manages the **OFI MR cache** — RDMA memory registrations used by the CXI
provider for Slingshot data transfers. When a VA is freed (munmap), userfaultfd
auto-invalidates the corresponding MR entries.

But `banned:1` comes from **CCL's IPC handle cache** — L0 driver-level handles for
XeLink peer-to-peer access (intra-node AllGather). These are a completely separate
mechanism:

- OFI MR: managed by libfabric, lives in user-space, userfaultfd can invalidate
- L0 IPC handles: managed by the L0 driver, lives in kernel-space, no user-space hook

When GC frees expandable_segments VAs:
1. userfaultfd correctly invalidates OFI MR entries → Slingshot recovers
2. But CCL's IPC handle cache still holds stale L0 handles → GPU segfault

The crash at step 8 (after GC at step 7) confirms this: userfaultfd cleaned up
OFI handles fine, but the L0 IPC handle was the one that killed the GPU.

### Why GC fired earlier with userfaultfd

With `disabled`, rank 0's torch_resv stayed at 59.64 GiB through all 5 steps (below
the 60.78 GiB threshold). With userfaultfd, rank 0 expanded to 62.54 GiB at step 6
POST-BWD. This is likely due to userfaultfd's MR tracking overhead (uffd page faults,
mmap intercepting) creating additional pressure that forced the allocator to expand.

## Recommendations

### Immediate (production launcher)

```bash
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95
unset XPU_USM_ALLOC_SO
export FI_MR_CACHE_MONITOR=disabled
```

Updated in `experiments/multinode_32b/run_32b_2hop_production.sh`.

`FI_MR_CACHE_MONITOR=disabled` is the correct production setting because:
- userfaultfd does NOT prevent the `banned:1` crash
- userfaultfd actually causes rank 0 to expand earlier (62.54 GiB vs 59.64 GiB),
  triggering GC at step 7 instead of never
- userfaultfd reduces external growth rate (10-20 vs 30 MiB/step) but this benefit
  is irrelevant if GC fires earlier due to torch_resv expansion
- External MR tracking overhead provides no value when the crash vector is L0 IPC handles

### Production constraint

Safe run length with gc:0.95 + disabled: **~80 steps** (l0_free depletes at ~30 MiB/step
from external growth). Beyond that, CCL's behavior at l0_free=0 is uncertain.

Mitigation: `save_every_n_steps=20` with manual checkpoint-restart.

### Remaining work

Identify the specific GRPO-specific source of external growth. Candidates:
1. XCCL 2-hop weight sync broadcast (~61 GiB staged per step)
2. clip_grad_norm DTensor AllReduce
3. XCCL process group management (new PGs created during training)

A targeted diagnostic adding ONLY the weight sync broadcast to the FSDP2 loop would
isolate whether that's the 30 MiB/step source.

## FSDP2 external growth diagnostic

### Question

Is the 30 MiB/step external memory growth from FSDP2+CCL base collectives
(AllGather/ReduceScatter) or from GRPO-specific operations?

### Setup

Job 8450943, single-node (x4218c3s4b0n0), 12 tiles FSDP2, `--standalone`.
Minimal forward/backward loop on Qwen3-32B with FSDP2 sharding and activation
checkpointing matching production. **No GRPO, no generation, no weight sync,
no optimizer, no ref model.** Real checkpoint weights (not random).

Config:
- `FI_MR_CACHE_MONITOR=disabled`
- `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95`
- `--seq-len 128`, `--batch-size 1`, `--no-optimizer` (skips AdamW entirely)
- `--num-steps 100`

### Result: external is FLAT — zero growth

| Step | external (GiB) | l0_free (GiB) | torch_resv (GiB) | torch_alloc (GiB) | time (s) |
|------|---------------|---------------|-------------------|-------------------|----------|
| INIT | 0.90 | 52.69 | 10.40 | 10.33 | — |
| 0 | 1.68 | 10.82 | 51.48 | 20.64 | 9.6 |
| 1 | 1.68 | 10.18 | 52.12 | 20.64 | 6.7 |
| 5 | 1.68 | 10.18 | 52.12 | 20.64 | 6.8 |
| 10 | 1.68 | 10.18 | 52.12 | 20.64 | 6.8 |
| 15 | 1.68 | 10.18 | 52.12 | 20.64 | 6.8 |
| 20 | 1.68 | 10.18 | 52.12 | 20.64 | 6.8 |
| 25 | 1.68 | 10.18 | 52.12 | 20.64 | 6.8 |

| 50 | 1.69 | 10.18 | 52.12 | 20.64 | 6.8 |
| 75 | 1.69 | 10.18 | 52.12 | 20.64 | 6.8 |
| 99 | 1.69 | 10.18 | 52.12 | 20.64 | 6.8 |
| FINAL | 1.69 | 10.18 | 52.12 | 20.57 | — |

External grew **10 MiB total** over 100 steps (1.68→1.69 GiB). For comparison,
production GRPO grows **~3000 MiB** over 100 steps (30 MiB/step). All other metrics
(l0_free, torch_resv, torch_alloc) perfectly stable throughout.

### Conclusion

**The 30 MiB/step external growth in production is NOT from FSDP2 base collectives.**
FSDP2's AllGather (forward unshard) and ReduceScatter (backward gradient reduction)
use XCCL under the hood, and their CCL internal allocations are fully stable across
100 steps (0.1 MiB/step growth, 300× less than production).

The growth source is one or more GRPO-specific operations:
1. **XCCL 2-hop weight sync broadcast** (most likely — 61 GiB of parameters staged
   to CPU and broadcast every step, creating new CCL internal state each time)
2. **clip_grad_norm AllReduce** (DTensor-based, separate from FSDP2 collectives)
3. **XCCL process group lifecycle** (new PGs for weight sync vs training)

This changes the production constraint analysis: if the weight sync broadcast is
the source, the growth could potentially be mitigated by reusing CCL buffers or
increasing the weight sync interval. The 80-step safe run length is not a
fundamental FSDP2 limitation — it's a GRPO architecture limitation.

### Allocator files

| File | Status | Notes |
|------|--------|-------|
| `recipes/dev/usm_caching_alloc.so` | Production (3B) | Gen1, power-of-2, 130 steps validated |
| `recipes/dev/usm_caching_alloc_v2.cpp` | Research only | Gen3, exact-align + OOM retry. Works for 3B, fails at 32B |
| `recipes/dev/usm_arena_alloc.cpp` | Deprecated | Gen2, no cross-stream safety |

## Appendix: gen3 allocator OOM retry implementation

Added to `usm_caching_alloc_v2.cpp` during this session:

```cpp
// In CachingPool::alloc():
void* ptr = try_alloc_device(sz, q);  // wraps sycl::malloc_device in try/catch
if (ptr) return ptr;
// Returns nullptr to xpu_usm_malloc for device-level OOM handling

// In xpu_usm_malloc():
void* ptr = target.alloc(size, queue);
if (ptr) return ptr;
// OOM — release cached blocks from BOTH pools on this device
size_t freed = pool.small_pool.release_cached(queue);
freed += pool.large_pool.release_cached(queue);
// Retry
ptr = target.alloc(size, queue);

// release_cached():
// 1. queue->wait() — drain pending work for cross-stream safety
// 2. sycl::free(ptr, *q) for each block in free_lists
// 3. Remove from alloc_sizes tracking
// 4. Clear free_lists
```

Debug levels: `USM_ALLOC_DEBUG=1` logs OOM/release events only (manageable log size);
`USM_ALLOC_DEBUG=2` logs every alloc/free (2.2M lines in 12 min at 32B — use sparingly).

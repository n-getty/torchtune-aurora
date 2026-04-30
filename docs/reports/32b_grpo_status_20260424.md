# 32B GRPO on Aurora — Status Report (2026-04-24)

## System Architecture

**Hardware**: 2 Aurora nodes (Intel Max Series GPU, 12 tiles/node, 64 GiB HBM/tile, Slingshot 11 interconnect).

**Training node**: 12-tile FSDP2, Qwen3-32B in BF16. `--standalone` torch.distributed, `CCL_ATL_TRANSPORT=ofi`, `FI_PROVIDER=cxi`. Training runs per-layer AllGather: each layer ~1 GiB BF16 output on 12 tiles.

**Inference node**: 3 vLLM replicas × TP=4 (12 tiles total). `distributed-executor-backend=mp`, `gpu-memory-utilization=0.80`. Generates completions for GRPO rollouts.

**Step timing (nominal)**: ~40–48s/step, breakdown: `gen≈19s + grpo≈14s + other≈9s`. `other` is the XCCL weight sync. `grpo` includes FSDP2 forward+backward+optimizer.

---

## Weight Sync: 2-Hop XCCL Broadcast

**Problem**: Flat broadcast from training to all 12 vLLM ranks crosses Slingshot for each
rank — 13-rank broadcast bandwidth was 1.7 GB/s vs 10.1 GB/s for a 2-rank transfer. The
bottleneck is Slingshot being shared across all 13 ranks simultaneously.

**Solution implemented**: 2-rank process group (training rank 0 + vLLM rank 1) for the
cross-node hop via Slingshot, then a 12-rank intra-node XeLink broadcast for distribution
within the inference node. vLLM rank 1 coordinates both hops.

**Result**: sync time 38s → **9.2–9.6s** (4× improvement). Validated 2026-04-24, confirmed
stable through 28 steps.

**Temporary buffers created**: The broadcast gathers ~6 GiB of parameter tensors per tile.
These are live during the sync and freed afterward. This is the key trigger for the crash
described below.

---

## Crash History and Root Causes

### Crash A: Step-2 `banned:1` (early runs)

**Symptom**: GPU CCS page fault at step 2 during FSDP AllGather.

**Root cause**: `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` defaults to 1000. oneCCL caches
IPC handle lookups; when the cache fills and evicts, subsequent accesses to those handles
cause Level Zero to fault.

**Fix**: `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536`. **Confirmed fixed**, present in
all launchers.

---

### Crash B: Step-7/8 `banned:1` (default allocator, no gc threshold)

**Symptom**: Step 7 `grpo` spikes to 50–73s (was ~14s), step 8 crashes with GPU write
fault. Asymmetric across tiles: tiles 1, 5, 6, 7 (receive-side in the AllGather ring) hit
`l0_free=0.01 GiB` while tiles 0, 2, 3, 4 remain healthy at `l0_free=28 GiB`.

**Root cause — OFI Fabric MR accumulation**:

- `FI_MR_CACHE_MONITOR=disabled` (required for Slingshot on Aurora) disables automatic
  deregistration of OFI memory regions.
- PyTorch's default XPU allocator returns a **fresh virtual address** for each FSDP
  AllGather output buffer every step.
- Each new VA triggers a new OFI/libfabric DMA registration. Old registrations are never
  deregistered.
- After N steps: N × ~6 GiB registrations accumulate on receive-side tiles → HBM exhausted
  → 50s PyTorch GC stall → comm stream reads a recycled VA while GC is in progress →
  `banned:1`.

**Partial mitigation**: `PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6`.
This keeps PyTorch's pool bloated (`torch_resv=62 GiB`), preventing the allocator from
returning blocks via `sycl::free`. No VA recycling → OFI keeps existing registrations valid
→ no accumulation.

**Confirmed at 32B**: 15-step stability test showed `external` memory FLAT at 1.83–2.70 GiB
all 15 steps; step 7 `grpo=17.4s` (vs 73s without gc:0.6). `l0_free` settles at 0.06–0.07
GiB from step 7 onward — tight, but stable so long as nothing triggers GC.

---

### Crash C: Step-29 `banned:1` (gc:0.6 + XCCL weight sync)

**Symptom**: Steps 0–27 fully clean at 40–48s/step. At step 28 PRE-BWD:

```
step 27: torch_resv=62.04 GiB  l0_free=0.06 GiB  external=1.89 GiB  grpo=14.6s
step 28: torch_resv=46.05 GiB  l0_free=14.09 GiB  external=3.85 GiB  grpo=97.7s  ← GC fired
step 29: Segmentation fault from GPU, ctx_id:1 (CCS), type:0 (NotPresent),
         level:1 (PDE), access:1 (Write), banned:1
```

**Root cause**: gc:0.6 keeps the pool bloated only as long as no large block is freed. The
XCCL weight sync (which runs every step) allocates and then frees ~6 GiB of parameter
buffers per tile. By step 28, the cumulative freed memory pushed the pool's free ratio above
0.6 → PyTorch called `sycl::free` on ~16 GiB of cached blocks → OFI registrations for
those VAs were invalidated → step 29's FSDP AllGather wrote to a VA whose OFI registration
had just been torn down → GPU write fault.

**Conclusion**: gc:0.6 deferred the crash from step 7 to step 28. It is not a sufficient
fix for runs with XCCL weight sync.

---

## Allocator Investigation

The fundamental requirement is: **never call `sycl::free` on any buffer touched by FSDP
AllGather or weight sync**, so OFI's DMA registrations remain valid indefinitely.

| Approach | Outcome | Reason |
|---|---|---|
| `expandable_segments:True` | Blocked | Virtual memory (`zeVirtualMemMap`) returns `ZE_MEMORY_TYPE_UNKNOWN`; oneCCL rejects pointer as non-USM. Second failure: no `zeMemGetIpcHandle` for virtual regions → GPU page fault on large tensors via CCL IPC zero-copy. |
| Arena allocator (`usm_arena_alloc.cpp`) | Blocked at 32B | Sub-allocates from 4 GiB `sycl::malloc_device` slabs. CCL's IPC zero-copy path calls `zeMemGetIpcHandle` on the sub-pointer; Level Zero returns handle for the slab base, not the sub-offset → GPU page fault. Works at 3B (50 MB AllGather is below CCL's IPC threshold; staging path used instead). |
| `gc:0.6` (default allocator) | Partial | Defers crash from step 7 to step 28. Insufficient with XCCL weight sync (see Crash C). |
| `usm_caching_alloc.so` (gene3b, 3B) | **Confirmed** | Individual `sycl::malloc_device()` per pooled block — no sub-allocation. Power-of-2 free lists, `kBucketCap=8 GiB`. Freed blocks return to pool; `sycl::free` never called for pooled blocks. Alloc-time `queue->wait()` provides cross-stream safety (covers the `recordStream` no-op bug in `XPUPluggableAllocator`). Confirmed clean for 15 steps on gene3b, including step 7 at `grpo=7.0s` with no OFI accumulation. |
| `usm_caching_alloc.so` (32B) | **Untested** | See Proposed Fix below. |

---

## Proposed Fix for 32B: `usm_caching_alloc.so`

**Why it should work at 32B**: The allocator's pooled blocks are individual
`sycl::malloc_device()` allocations — the returned pointer is the base USM allocation.
`zeMemGetIpcHandle` is called on the base pointer → valid IPC handle → CCL zero-copy path
works at any AllGather size. There is no sub-allocation; the confusion with the arena
allocator was a false equivalence written during investigation and not subsequently
validated.

**Coverage of the weight sync trigger**: The XCCL weight sync creates ~6 GiB buffers per
tile. `bucket_size(6 GiB)` rounds up to 8 GiB = exactly `kBucketCap`. This buffer is
pooled, never `sycl::free`'d → the step-28 GC trigger is eliminated entirely.

**Pool steady state at 32B**: FSDP2 prefetches 1–2 layers ahead, so the working set is
~2–4 concurrent AllGather buffers. Per-layer sizes at 32B vary (attention, MLP) but all
round to power-of-2 buckets ≤ 2 GiB. Weight sync adds one 8 GiB bucket. Estimated total
pool: **15–25 GiB per tile** — well within the 64 GiB HBM budget.

**Risk**: Any single allocation > 8 GiB would fall through to `sycl::free` (the oversized
path in the allocator). Based on current knowledge this shouldn't occur, but if it does, it
would manifest as a late-step crash similar to Crash C.

---

## Current Status

| Component | Status |
|---|---|
| 2-hop XCCL weight sync | Confirmed working — 9.2s/step, stable through 28 steps |
| 32B FSDP2 training (no weight sync) | Confirmed stable — 15 steps clean with gc:0.6 |
| 32B with gc:0.6 + XCCL weight sync | Crashes at step 29 (Crash C above) |
| Checkpoints saved | Steps 10 and 20 at `outputs/32b_2hop_prod_20260424_200908/ref/epoch_0` |
| Gene3b production | Running (job 8450268, debug queue, `usm_caching_alloc.so`, logging to `/lus`) |

---

## Next Steps

1. **Test `usm_caching_alloc.so` at 32B scale** (priority 1). Update
   `run_32b_2hop_production.sh`: replace `PYTORCH_ALLOC_CONF=gc:0.6` with
   `XPU_USM_ALLOC_SO=.../usm_caching_alloc.so`, unset `PYTORCH_ALLOC_CONF`. Run a
   15-step held-node stability test watching step 7 and the `external` memory trajectory.
   If `external` stays flat and step 7 doesn't spike, the fix generalizes to 32B.

2. **If pool exceeds HBM budget**: Adjust `kBucketCap` or add an LRU eviction limit per
   device. Evicted blocks must be tracked carefully — any OFI-registered VA that is
   `sycl::free`'d while FI_MR_CACHE_MONITOR=disabled will leave a stale DMA registration.

3. **If CCL IPC still fails at 32B** (unexpected): Force CCL onto the staging-buffer path
   for all AllGather operations via CCL algorithm configuration, bypassing IPC zero-copy
   entirely. This would be a CCL configuration change rather than an allocator change,
   at the cost of additional HBM-to-HBM copy overhead.

4. **Gene3b 500-step production run**: Monitor job 8450268 for stability past step 15. If
   clean through the 1h debug-queue wall, submit the full 7h production run with
   `usm_caching_alloc.so`.

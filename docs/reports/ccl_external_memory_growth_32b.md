# CCL External Memory Growth: Step ~28 Crash in 32B FSDP2 Training

**Date:** 2026-04-25
**Severity:** Blocks production runs > 25 steps
**Affected:** 32B Qwen3-32B GRPO, 2-node Aurora, 12-tile FSDP2 + 3×TP=4 dedicated vLLM
**Status:** Root cause identified, no fix. Workaround: checkpoint-restart every 20 steps.

## Summary

After ~28 steps of 32B FSDP2 training, external (non-PyTorch) GPU memory grows enough
to exhaust L0 free memory. A transient allocation spike triggers garbage collection,
which frees PyTorch cached blocks. Some freed blocks have oneCCL IPC handles registered
against them. When CCL subsequently accesses those handles, the GPU faults with
`banned:1` (CCS NotPresent page fault).

This bug is **distinct** from the three bugs documented in `docs/bugs/`:
- **Not expandable_segments** — that bug is about USM pointer type rejection
- **Not empty_cache UR leak** — no explicit `empty_cache()` is called
- **Not the step-2 IPC cache eviction** — that was fixed by `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536`

## Crash Signature

```
Segmentation fault from GPU at 0xff000007fe3ee000, ctx_id: 1 (CCS)
type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
```

Preceded by a massive GRPO slowdown (7-10x normal) at the crash step:
```
TIMING step=28  total=122.7s  gen=21.9s  grpo=97.3s  ...  # normal grpo ~14s
```

## Reproduction

Reproduced in 3 independent runs on 3 different node pairs:

| Run | Job | GC Threshold | Weight Sync | Crash Step | Notes |
|-----|-----|-------------|-------------|------------|-------|
| 1 | 8450445 | gc:0.6 | synchronous XCCL | 29 | 28 steps clean, GC at step 29 |
| 2 | 8450499 | gc:0.8 | deferred async XCCL | 28 | 27 steps clean, GC at step 28 |
| 3 | 8450493 | gc:0.8 | sync (broken, UnboundLocalError) | 28 | 27 steps clean, GC at step 28 |

The crash step (~28) is consistent regardless of GC threshold, weight sync method,
and node assignment. This rules out the async optimization as a cause — the bug is in
the base FSDP2 + CCL interaction.

## Root Cause Analysis

### Phase 1: Slow external memory growth (steps 0-27)

External memory (L0 allocations outside PyTorch's caching allocator — primarily CCL
communication buffers and OFI fabric memory registrations) grows linearly at
~10 MB/step. This growth is visible across ranks and configurations:

**Rank 0, POST-BWD external memory (async test, job 8450499):**
```
step=10  external=1.050 GiB  l0_free=0.39 GiB
step=15  external=1.150 GiB  l0_free=0.30 GiB
step=20  external=1.200 GiB  l0_free=0.25 GiB
step=25  external=1.220 GiB  l0_free=0.21 GiB
step=27  external=1.230 GiB  l0_free=0.20 GiB
```

**Rank 0, POST-BWD external memory (baseline longrun, job 8450493):**
```
step=10  external=1.640 GiB  l0_free=0.31 GiB
step=15  external=1.690 GiB  l0_free=0.26 GiB
step=20  external=1.720 GiB  l0_free=0.23 GiB
step=25  external=1.750 GiB  l0_free=0.20 GiB
step=27  external=1.770 GiB  l0_free=0.18 GiB
```

Both runs show the same pattern: external grows, `l0_free` shrinks. The starting
external differs (~1.0 vs ~1.6 GiB) due to different XCCL PG configurations, but
the growth rate is consistent at ~10 MB/step.

Meanwhile, `torch_resv` is stable at 62.04 GiB from step ~7 onward (the PyTorch
caching allocator has claimed nearly all device memory). The budget is:

```
l0_total  = 63.98 GiB   (Intel Max 1550 per tile)
torch_resv = 62.04 GiB   (PyTorch cached, stable)
external  ≈  1.7  GiB   (CCL + OFI, growing)
l0_free   ≈  0.2  GiB   (shrinking)
```

### Phase 2: GC trigger (step 28)

At step 28, during the GRPO forward pass (which includes FSDP2 AllGather operations
that temporarily expand sharded parameters to full size), a transient allocation spike
exceeds available `l0_free`. This forces the PyTorch caching allocator to reclaim
cached blocks via garbage collection.

GC frees reserved blocks, dropping `torch_resv` from 62.04 to 46.05 GiB — releasing
~16 GiB back to the L0 driver.

### Phase 3: Asymmetric CCL memory grab (step 28, during backward)

After GC frees those 16 GiB, CCL immediately claims a large portion on **some** ranks.
The rank-level snapshot at PRE-BWD step 28 shows a stark asymmetry:

| Rank | l0_free | torch_resv | external | Status |
|------|---------|------------|----------|--------|
| 0 | 13.58 | 46.84 | 3.56 | Healthy |
| **1** | **0.03** | **46.05** | **17.91** | **Exhausted** |
| 2 | 14.44 | 46.05 | 3.50 | Healthy |
| **3** | **0.01** | **46.05** | **17.92** | **Exhausted** |
| 4 | 14.44 | 46.05 | 3.50 | Healthy |
| 5 | 14.43 | 46.05 | 3.51 | Healthy |
| 6 | 14.44 | 46.05 | 3.50 | Healthy |
| 7 | 14.43 | 46.05 | 3.51 | Healthy |
| 8 | 14.44 | 46.05 | 3.50 | Healthy |
| **9** | **0.02** | **46.05** | **17.92** | **Exhausted** |
| 10 | 14.36 | 46.05 | 3.58 | Healthy |
| **11** | **0.02** | **45.50** | **18.47** | **Exhausted** |

Ranks 1, 3, 9, 11 have **17.9 GiB external** (vs 3.5 GiB on healthy ranks).
All the freed memory was consumed by CCL on these ranks, leaving `l0_free ≈ 0.02 GiB`.

The affected ranks {1, 3, 9, 11} are likely the receive-side ranks in the FSDP2
AllGather ring topology. When CCL's staging/IPC buffers for these ranks had stale
handles pointing to the freed torch pages, CCL allocated large replacement buffers
from the freed pool, exhausting device memory on those ranks.

### Phase 4: Crash (step 28-29)

The backward pass on the crash step takes 7-10x longer than normal (grpo = 57-97s
vs ~14s) as CCL stalls waiting for memory or re-registering OFI fabric entries.
After the step completes, the next step's FSDP2 AllGather triggers CCL to access
stale IPC handles on the freed pages, causing the GPU page fault.

## Why the GC threshold doesn't matter

The GC threshold determines *when* the allocator proactively frees cached blocks:

- gc:0.6 → trigger when `torch_alloc / torch_resv > 0.6`
- gc:0.8 → trigger when `torch_alloc / torch_resv > 0.8`

But the crash isn't caused by proactive GC. It's caused by **reactive memory pressure**:
when `l0_free` drops below what CCL needs for a collective operation, the PyTorch
allocator must free blocks to make room — regardless of the GC threshold. This reactive
path fires at step ~28 for both gc:0.6 and gc:0.8 because the trigger is l0_free
exhaustion (determined by external growth rate), not the GC ratio.

Setting gc:1.0 (effectively disabling GC) would not help — the allocator still frees
blocks under OOM pressure.

## Relationship to other documented bugs

### vs. OFI MR accumulation (memory file: `project_ccl_ipc_handle_cache.md`)

The OFI MR accumulation bug documented for gene3b (fixed by `usm_caching_alloc.so`)
was caused by **new VA addresses** being allocated each step — each new VA required a
new OFI fabric memory registration, accumulating ~6 GiB/step.

This 32B bug is different: the PyTorch caching allocator maintains stable VAs
(`torch_resv` flat at 62.04 GiB), so OFI MR registrations should be stable. The
~10 MB/step external growth comes from a different source — likely CCL internal
bookkeeping, IPC handle metadata, or Level Zero driver state associated with the
collective operations themselves.

### vs. expandable_segments bug (`docs/bugs/intel_ccl_expandable_segments_bug.md`)

Completely different. That bug is about `zeVirtualMemMap` pointers returning
`ZE_MEMORY_TYPE_UNKNOWN`, causing CCL to reject tensors. This bug uses standard
USM device allocations — CCL accepts the pointers fine. The crash comes from
accessing handles to memory that was freed by GC.

### vs. empty_cache UR leak (`docs/bugs/intel_xpu_resource_leak_bug_report.md`)

Related mechanism (both involve freeing L0 memory with outstanding handles), but
different trigger. The UR leak requires explicit `empty_cache()` calls. This bug
triggers through normal GC operation — no `empty_cache()` is called anywhere in the
training loop.

## Impact

- **32B GRPO training limited to ~25 steps per process lifetime**
- Production 70-step runs require checkpoint-restart (save every 20 steps, restart
  process to clear accumulated CCL state)
- The deferred async weight sync optimization (13.4% throughput improvement) is
  validated but can't be used for long runs without the restart workaround
- The 3B gene recall training is NOT affected (uses `usm_caching_alloc.so` which
  pools buffers at stable VAs, preventing external growth)

## Workarounds

### Checkpoint-restart (recommended)

Run training in 20-step segments. Each segment starts with fresh CCL state
(external ≈ 1.0-1.5 GiB, l0_free ≈ 0.4+ GiB). The `save_every_n_steps` parameter
and resume-from-checkpoint support this workflow.

Budget: 20 steps × 37s = 740s = 12.4 min per segment. A 70-step run needs 4
segments, each fitting comfortably within the 1-hour debug queue.

### Reduce torch_resv headroom (untested)

If `torch_resv` could stabilize at 60 GiB instead of 62 GiB, `l0_free` would start
at ~2 GiB — providing ~200 steps of headroom at 10 MB/step growth. However, the
PyTorch caching allocator doesn't provide a `max_reserved_bytes` control on XPU,
and reducing `max_split_size_mb` might increase fragmentation rather than reduce
reserved memory.

### CCL IPC handle cache flush (untested)

If CCL provides an API to flush its IPC handle cache (releasing the external memory
registrations), calling it periodically (e.g., every 20 steps) might reset the
external growth. No such API has been identified.

## Data Files

- Async test log: `experiments/multinode_32b/run_32b_async_wsync_live.log`
- Baseline longrun log: `experiments/multinode_32b/run_32b_longrun.log`
- gc:0.6 production crash: documented in memory file `bugs/project_ccl_ipc_handle_cache.md`
- MEMPROBE instrumentation: `recipes/dev/grpo_full_finetune_distributed_xpu.py` (search `MEMPROBE`)

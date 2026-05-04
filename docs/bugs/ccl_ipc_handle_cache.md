# Bug C: oneCCL IPC handle cache — stale-eviction at default, accumulation at high threshold

**System:** Aurora (ALCF), Intel Max 1550 GPU, `frameworks/2025.3.1` (torch 2.10, oneCCL 2021.17)

> Symptom note: this bug ends in a `banned:1` GPU fault, the same outward
> signature as `docs/bugs/intel_ccl_expandable_segments_bug.md` (Bug A) and
> `docs/bugs/xpu_pluggable_allocator_record_stream.md` (Bug B). The root causes
> are independent. See the cross-reference table at the bottom.

## Summary

oneCCL caches Level Zero IPC handles for GPU virtual addresses participating
in collectives, controlled by `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD`. Both
the default and the high-threshold settings have failure modes:

- **Default (threshold ≈ 1000):** the LRU eviction kicks in around step ~28 of
  32B FSDP2 training. A subsequent AllGather opens a fresh IPC handle for a
  newly registered VA but a sibling rank still holds an evicted handle — the
  next collective issues DMA against a no-longer-mapped VA and the GPU
  page-faults with `banned:1`.
- **High (threshold = 65536):** evictions are suppressed, but handle metadata
  accumulates. On a 10-tile FSDP2 backbone, ~10.85 GiB of IPC handle memory
  builds up by the end of step 1's backward, and step 2 OOMs the L0 device.
  This is the failure mode that makes single-node 32B FSDP unviable; we use
  2-node HSDP instead.
- **Zero (threshold = 0, WS12 candidate fix):** handles are closed immediately
  after each AllGather. No stale handles can exist when VAs get recycled, and
  no accumulation builds up. **Status: queued for validation, not yet
  confirmed.** The recipe envelope is in `recipes/dev/run_qwen3_30b_ep8_vllm_2node.sh:44`
  (`${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-65536}`).

The issue surfaces in two distinct flavors that share the same root cause:

1. **Step ~28 dense 32B training crash** (across-step VA churn under FSDP2
   AllGather). Documented in
   `docs/reports/ccl_external_memory_growth_32b.md`.
2. **Within-step EP=8 chunked-loss crash** (across-chunk VA churn when
   `_ep_release_fsdp_unsharded_grads()` calls `free_unsharded_param()` between
   chunks; the next chunk's re-AllGather recycles the freed VA and CCL accesses
   it via a stale IPC handle). This is the WS11 → WS12 chain in
   `docs/status.md` for Qwen3-30B-A3B EP=8.

## Crash signature

```
Segmentation fault from GPU at 0xff010002e27f3000, ctx_id: 1 (CCS)
type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
```

Often preceded by a 5–10× slowdown at the offending step or chunk as CCL
internal bookkeeping thrashes. Example dense-32B step 28:

```
TIMING step=28  total=122.7s  gen=21.9s  grpo=97.3s  ...  # normal grpo ~14s
```

Example EP=8 WS11 chunked crash (job 8465485):

- chunk[0:1] clean (BATCH_REWARD step=0 reward_mean=0.5020 success=0.75).
- chunk[0:1] backward complete (per-rank 46.6–76.9s).
- `_ep_release_fsdp_unsharded_grads()` releases the backbone unsharded params.
- chunk[1:2] forward starts → banned:1 PDE at `0xff010002e27f3000`.

## Reproduction

Both flavors reproduce deterministically:

### Dense 32B step ~28 crash

3 independent runs on 3 different node pairs (see
`docs/reports/ccl_external_memory_growth_32b.md` for details):

| Run | Job | GC threshold | Weight sync | Crash step |
|-----|-----|--------------|-------------|------------|
| 1 | 8450445 | gc:0.6 | synchronous XCCL | 29 |
| 2 | 8450499 | gc:0.8 | deferred async XCCL | 28 |
| 3 | 8450493 | gc:0.8 | sync (broken, UnboundLocalError) | 28 |

Crash step is consistent across GC threshold, weight sync method, and node
assignment. Rules out async/sync sync method as the cause; the bug is in the
base FSDP2 + CCL interaction.

### EP=8 within-step chunked crash

WS9 (MAX_GEN=512) clean; WS11 (MAX_GEN=768) crashes at chunk[1:2] forward
with the signature above. Larger MAX_GEN → larger activations → more allocator
pressure → freed backbone VAs get recycled across chunks → CCL stale handles
fire.

## Mechanism

CCL's intra-node collective path opens IPC handles for the GPU VAs of
collective input/output buffers via `zeMemGetIpcHandle`. To avoid paying for
the handle exchange every collective, oneCCL caches open handles, keyed by VA.
`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` is the LRU cap.

- **Eviction (low threshold).** When the cache exceeds the cap, oneCCL closes
  the least-recently-used handle. If that handle is still referenced by a
  pending or upcoming collective on a peer rank, the peer's next access faults
  with `NotPresent` (the handle's mapping was torn down on the local side).
- **Accumulation (high threshold).** With the cap raised to 65536, handles are
  effectively never evicted. Each unique VA that has ever participated in a
  collective accumulates O(KB)–O(MB) of L0/CCL bookkeeping. With FSDP2's
  AllGather output buffers churning (different sizes / fresh VAs across steps
  or chunks), the cache fills the device. Measured: ~10.85 GiB total IPC
  handle memory by end of step 1 backward on 10-tile FSDP2 → step 2 OOM.
- **Zero (immediate close).** Handles are released after each collective. No
  stale handles, no accumulation. Cost: a fresh IPC open per collective. We
  do not yet have a measured throughput hit; that's part of WS12 validation.

This is **distinct** from the OFI memory-region (MR) cache. OFI MR is
user-space libfabric state on Slingshot for cross-node RDMA. CCL IPC handles
are kernel-space Level Zero state for intra-node XeLink peer access.
Validated 2026-04-25: `FI_MR_CACHE_MONITOR=userfaultfd` does NOT prevent
the `banned:1` crash because it manages a different cache (see
`docs/status.md:1263-1278`).

## Workarounds in this repo

### Multi-node 32B (current production)

`recipes/dev/run_qwen3_30b_ep8_vllm_2node.sh:44`:

```bash
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-65536}
```

Threshold=65536 is the current default. Survives ~28 steps of dense 32B FSDP2.
Mitigation for runs longer than 25 steps is **checkpoint-restart every 20
steps** — a 5-minute restart amortizes well over a 12 h queue allocation
(see `docs/reports/ccl_external_memory_growth_32b.md`).

For EP=8 chunked-loss runs, threshold=65536 is the WS11 setting that crashes
inside step 1. WS12 sets threshold=0 before sourcing the same launcher.

### Single-node 32B FSDP — blocked

The 65536 setting accumulates ~10.85 GiB of IPC handle memory by end of step 1
backward on a 10-tile FSDP2 backbone, leaving no room to even reach step 2.
We use 2-node HSDP instead. See CLAUDE.md "Critical Platform Constraints" and
`memory/project_xccl_broadcast_test.md` for the full numbers.

### EP=8 chunked loss (WS11 → WS12)

WS11 (MAX_GEN=768, threshold=65536): crashes between chunks because
`_ep_release_fsdp_unsharded_grads()` frees backbone VAs, the next chunk's
AllGather recycles them, and CCL accesses them via stale IPC handles.

WS12 candidate (threshold=0): close handles immediately after each AllGather.
**Status: queued for validation as of 2026-05-02. Not yet proven on the EP=8
chain.** If WS12 passes, threshold=0 becomes the recommended setting for
chunked-loss EP runs.

## Suggested upstream fix

Two complementary directions for oneCCL:

1. **Reference-count IPC handles by ongoing collective.** Don't close a handle
   while any in-flight or queued collective on any rank still references it.
   Eliminates the eviction race at low threshold without unbounded growth.
2. **Track handle lifetime against allocator events.** Expose a hook so the
   allocator can notify CCL when a VA is freed or rebound; CCL closes its
   handle immediately. This solves both eviction and accumulation: the cache
   only holds handles for currently-live VAs, so high thresholds become
   harmless.

Either fix would let us drop the threshold tuning entirely.

## Diagnostic notes

- The "10 MB/step external growth" reported in
  `docs/reports/ccl_external_memory_growth_32b.md` is the visible signature of
  this bug. Even if the PyTorch caching allocator is configured to keep VAs
  stable, IPC-handle metadata still accumulates inside CCL and is invisible
  to PyTorch's `memory_stats()`. The growth is allocator-stable; the leak is
  CCL-internal.
- A minimal FSDP2-only loop (no GRPO, no generation, no weight sync) would
  isolate this — if external memory still grows at ~10 MB/step there, the
  source is purely FSDP2 + CCL. Not yet measured.
- `FI_MR_CACHE_MONITOR=userfaultfd` (Aurora userfaultfd MR monitor) is not a
  workaround. It does not touch L0 IPC handles at all; tested on cross-node
  Slingshot, still crashes at step 8.

## Related, not the same bug

| Other file | Bug | Symptom |
|------------|-----|---------|
| `docs/bugs/intel_ccl_expandable_segments_bug.md` | `expandable_segments` virtual-mem pointers rejected by CCL | `invalid usm pointer type: unknown` |
| `docs/bugs/xpu_pluggable_allocator_record_stream.md` | `XPUPluggableAllocator` step-1 GPU page fault under FSDP2 | banned:1 PDP/PML5 NotPresent |
| `docs/bugs/intel_xpu_resource_leak_bug_report.md` | `torch.xpu.empty_cache()` + FSDP `storage.resize_()` UR-handle leak | UR_RESULT_ERROR_OUT_OF_RESOURCES at iter ~70 |

Investigation history: `docs/reports/ccl_external_memory_growth_32b.md`,
`docs/reports/allocator_deep_analysis_20260425.md`, `docs/status.md` (WS9–WS12
EP=8 chain).

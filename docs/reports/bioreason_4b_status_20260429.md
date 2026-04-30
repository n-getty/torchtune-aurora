# BioReason 4B GRPO — Status & Barriers (2026-04-29)

## TL;DR (RESOLVED)

**Run 41 cleared 6/6 steps cleanly with exit=0 and checkpoint saved.** The fix combination on a single node:
1. **Drop `_multimodal` gate from chunked-loss branch** (`recipes/dev/grpo_full_finetune_distributed_xpu.py:5484`) — enables single fwd+bwd path, removes per-chunk graph retention.
2. **Persistent wsync chunk buffer** (`grpo_full_finetune_distributed_xpu.py:3450/3478/3517`) — avoids per-step `torch.empty()` returning fresh L0 pages that CCL accumulates as IPC handles.
3. **`grpo_samples=4`, `forward_batch_size=4`** (was 8/8) — halves activation peak. POST-BWD `torch_resv` flat at 51.42 GiB instead of 60.83 GiB. l0_free stays at 6-9 GiB instead of 0.49 GiB. wsync timing flat at 22-24s instead of climbing 31→30→50s.
4. **Launcher hygiene**: `TORCHTUNE_USE_CHUNKED_LOSS=1`, `PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.95`, vLLM pkill cleanup pre-launch.

Step time ~43s steady (40.9-49.1s observed). Throughput: 4 rollouts/prompt × 11 ranks = 44 rollouts/step.

**Run 42 (20-step validation): 20/20 clean, exit=0, 14:45 wall, two checkpoints saved.** Step times trended down (early ~45s, late ~42s) as caches warmed. wsync FLAT 20.6-23.6s with no growth. Confirms long-horizon stability.

## Original problem (kept for reference)

The BioReason 4B multimodal GRPO recipe (ESM3 protein encoder + GO graph encoder + Qwen3-4B backbone, 11+1 dedicated-vLLM, FSDP1 ZeRO-2) cannot get past **step 2 PRE-STEP** on a single Aurora node. Every failure is the same `banned:1` SIGABRT — CCL collective hits a stale Level-Zero IPC handle. The mechanism is now well-characterized: **peak HBM during step-N backward pushes torch_resv to ~60 GiB, the PyTorch caching allocator's automatic GC fires, returns L0 pages whose addresses CCL has cached as IPC handles, and the next collective dies.**

We have already eliminated several plausible root causes (XCCL teardown, rank0_only=False OOM, fbs=4 chunk doubling, allocator V2, explicit `empty_cache`). The remaining mechanism is **automatic GC under memory pressure**, and the underlying pressure comes from the FSDP1 ZeRO-2 + multimodal-gated chunked-loss path — a configuration that no other working recipe in this fork uses.

The paradox the user flagged is real and worth restating: a 4B model cannot fit on one node while a 32B model can. The reason has nothing to do with parameter count. It is an **architectural choice mismatch**: BioReason runs ZeRO-2 (params replicated on every tile) on a node with no headroom to spare, while 32B runs FULL_SHARD across 24 tiles on three nodes with ~25 GiB free per tile.

## Run history (relevant)

| Run | Change | Step reached | Outcome |
|---|---|---|---|
| 27 | FSDP1 SHARD_GRAD_OP replacing DDP | step 0 OK, step 1 weight sync OOM | architecture validated, weight sync needed work |
| 30 | Chunked weight sync (37 broadcasts) | step 0 clean, teardown SIGSEGV | XCCL teardown bug — fixed via os._exit |
| 31 | rank0_only=False removed | step 0 → IGC compiler crash | reverted to rank0_only=True |
| 32-33 | gc:0.99 allocator config | banned:1 at step 1 | GC threshold not the lever |
| 34-35 | usm_caching_alloc_v2.so | banned:1 at step 1 | OOM-retry releases blocks → IPC stale |
| 36 | fbs=4 (no_sync 2 chunks) | banned:1 at step 1 | FSDP1 AllGather repeats per chunk → worse |
| 37 | pre-warm summon_full_params + fbs=8 | banned:1 at step 1 | pre-warm did not change peak BWD pool |
| **38** | **removed rogue `torch.xpu.empty_cache()` in ref-offload** | **step 0 + step 1 OK; banned:1 at step 2 PRE-STEP** | **defers crash by one step; same mechanism** |
| 39 | dropped `_multimodal` gate from chunked-loss branch (single-bwd path) | step 0/1/2 OK; banned:1 at step 3 PRE-STEP | wsync climbed 31→30→50s → IPC accumulation theory |
| 40 | + persistent wsync chunk buffer (avoid per-step `torch.empty`) | step 0/1/2 OK; banned:1 at step 3 PRE-STEP | persistent buffer alone insufficient; l0_free hit 0.49 GiB at POST-BWD step 1 |
| **41** | **+ `grpo_samples=4`, `forward_batch_size=4` (G=4 instead of G=8)** | **6/6 steps clean; exit=0; checkpoint saved** | **THIS IS THE FIX. POST-BWD peak FLAT at 51.42 GiB (vs 60.83 in run 40). l0_free FLAT 6-9 GiB. wsync FLAT 22-24s. external CCL FLAT.** |

## Run 38 — what actually happened

Step 0:
- 60s total (vs 92-96s in runs 31-37 — a 35% speedup, attributable purely to removing the rogue `empty_cache`).
- POST-BWD: torch_resv 53-58 GiB across all ranks; l0_free 2.8-6.8 GiB. **Healthy.**

Step 1 (the new failure point):
- POST-BWD on **6 of 11 ranks** (0,1,4,5,6,7) shows the smoking gun:
  - `torch_resv` collapsed from peak ~60 GiB to **29 GiB** → automatic GC fired during BWD
  - `external` (CCL-held memory) jumped from ~3 GiB to **15-16 GiB** → CCL re-mapped pages because IPC handles became stale
- Other 5 ranks (2,3,8,9,10) didn't GC — they remained pinned at torch_resv 61-63 GiB / l0_free < 0.15 GiB.
- Rank 5 PRE-BWD step=1 hit `l0_free=0.00 GiB l0_used=63.98 GiB` — L0 driver completely exhausted.
- TIMING step=1 = 124s (vs 60s for step 0 and ~92s in earlier runs); the extra 64s is allocator thrash + CCL re-IPC cost.

Step 2 PRE-STEP:
- SIGABRT (exit -6) on rank 2; ranks 3, 4 also hit SIGABRT; rest got SIGTERM from the launcher.
- This is the documented `banned:1` signature from `bugs/project_xpu_emptycache_revalidated.md` — `empty_cache` is one trigger, **automatic allocator GC under pressure is another path to the same end-state.**

Also notable: step-1 `grad_norm = 43520`. Whether this is a real loss-landscape issue or an artifact of the GC mid-backward (some grads computed against pre-GC tensors, others post-GC) is unclear. Either way it indicates the run is not in a healthy state at step 1, regardless of whether step 2 crashed.

## Why a 4B model uses more HBM than a 32B model on this node

Per-tile static memory footprint (model weights + optim state, before activations):

| Model | Sharding | Tiles | Params/tile | Grads/tile | Optim/tile | Encoders | Total static |
|---|---|---|---|---|---|---|---|
| Qwen3 32B | FULL_SHARD | 24 (3 nodes) | 64B params × 2B / 24 = 2.7 GiB | 2.7 GiB | 5.3 GiB | — | **~10.7 GiB** |
| BioReason 4B | SHARD_GRAD_OP (ZeRO-2) | 11 (1 node, 11+1 vLLM) | 4B × 2B = 8 GiB (replicated) | 8/11 = 0.73 GiB | 16/11 = 1.45 GiB | ESM3 (~2 GiB fp32) + GO + ref model (~8 GiB on alternating ranks) | **~20 GiB baseline + 8 GiB ref when on-device** |

So the 4B BioReason carries **~21 GiB static / tile** while 32B carries **~10.7 GiB static / tile**. With ~64 GiB L0 total per tile, 32B has ~50 GiB of working room for activations and CCL buffers; BioReason has ~40 GiB. That working room then gets squeezed by:
- ESM3 fp32 weights (kept fp32 because it's a frozen encoder)
- Ref model swapped on/off device per step
- vLLM dedicated rank using one tile (constrains us to 11 training tiles)
- summon_full_params AllGather buffer (7.49 GiB on every rank, even with rank0_only=True the AllGather collective requires it)

This is why BWD step 1 pushes 6 ranks past the GC threshold: the static floor is so high that even with pre-warmed AllGather and fbs=8, the BWD peak crosses 60 GiB.

## Eliminated hypotheses

These have all been tested and ruled out:

1. **XCCL teardown bug** — fixed in `cleanup()` with os._exit guard (run 30).
2. **rank0_only=False OOM** — confirmed; reverted to rank0_only=True (run 31).
3. **fbs=4 to halve activations** — fbs reduction with FSDP1 SHARD_GRAD_OP creates extra no_sync chunks, each requiring a full 7.49 GiB AllGather. POST-BWD goes UP, not down (run 36).
4. **gc:0.99 allocator threshold** — the threshold is hit by step-1 BWD natural pressure, not by spurious GC; raising it doesn't help (run 33).
5. **usm_caching_alloc_v2.so** — pluggable allocator's OOM-retry path releases cached blocks, invalidating CCL IPC handles even more aggressively (runs 34-35).
6. **pre-warm summon_full_params** — caches the AllGather buffer, but the BWD peak is dominated by activations, not that buffer (run 37).
7. **Rogue `empty_cache` in ref-offload** — REAL bug, was fixed (run 38). It bought a 35% step-0 speedup and one extra clean step, but the remaining peak still trips the same mechanism via auto-GC.

## What's actually different from the working 32B recipe

| Lever | 32B (works) | BioReason (fails) | Why it matters |
|---|---|---|---|
| Sharding | FULL_SHARD across 24 tiles | SHARD_GRAD_OP across 11 tiles | 32B has ~10× less per-tile param footprint |
| `TORCHTUNE_USE_CHUNKED_LOSS` | `=1` | unset | 32B uses single fwd+bwd; BioReason falls into chunked-bwd |
| `PYTORCH_ALLOC_CONF` | `max_split_size_mb:512, gc:0.6` | unset (defaults) | unclear which is healthier here — see below |
| Multimodal chunked-loss gate (recipe line 5486) | not hit | hits `_multimodal=True` and falls through to per-chunk fwd path | per-chunk fwd retains intermediate graphs → pushes peak alloc up |
| Topology | 3 nodes / 24 tiles | 1 node / 11 tiles + 1 vLLM | no horizontal escape valve |

Both `TORCHTUNE_USE_CHUNKED_LOSS=1` and the multimodal gate at line 5486 are gates the 32B recipe sails through. BioReason's `_multimodal=True` flag forces the chunked path even with the env var set. **This is the one configuration BioReason exercises that no other working recipe in this fork ever exercises.**

## Allocator config — why "match 32B" is not obviously right

32B uses `garbage_collection_threshold:0.6`. That sounds aggressive — it forces the allocator to GC sooner. On 32B with ~50 GiB working room per tile this is fine: GC fires below the IPC-handle danger zone, before reserved gets near total. On BioReason with ~40 GiB working room and step-1 BWD reaching 60 GiB reserved, **gc:0.6 would fire GC even more often** during BWD — exactly the trigger we are trying to avoid. The healthy direction here is the opposite: push GC threshold to ~0.99 *and* reduce the actual peak so it never crosses that threshold organically. Run 32-33 already tested gc:0.99 in isolation — it didn't help because the natural pool growth crossed the threshold anyway. So the allocator knob is contingent on first lowering peak BWD pressure.

## Three concrete options going forward

The remaining knobs all aim at the same target: lower BWD peak below the GC-fire threshold.

### Option A — drop the multimodal chunked-loss gate (smallest change)

At `recipes/dev/grpo_full_finetune_distributed_xpu.py:5486`, the condition is:
```python
elif (
    os.environ.get("TORCHTUNE_USE_CHUNKED_LOSS") == "1"
    and not _multimodal       # ← this gate
    and self._expert_parallel_degree <= 1
):
    # single fwd + single bwd path
```

Removing `and not _multimodal` lets BioReason use the same single fwd+bwd path 32B uses. The fall-through chunked path retains per-chunk forward graphs simultaneously, which is what's pushing peak past 60 GiB.

- Risk: the multimodal path was originally gated out for a reason — likely "single-fwd of all 8 sequences at once would OOM with multimodal embed buffers." Need to verify: does build_full_embeds for 8 seqs of 1024 tokens produce a tensor large enough to OOM on its own? (Embed dim 2048 × bf16 × 8 × 1024 = 32 MiB — trivially fine. So the gate may simply be conservative and removable.)
- Pair with: `TORCHTUNE_USE_CHUNKED_LOSS=1` in the launcher.
- Expected: peak BWD drops materially because activations for non-current chunks aren't held.

### Option B — switch FSDP1 SHARD_GRAD_OP → FULL_SHARD

At `recipes/dev/grpo_full_finetune_distributed_xpu.py:2880`, change `ShardingStrategy.SHARD_GRAD_OP` → `ShardingStrategy.FULL_SHARD`. Recovers 8 - 8/11 = ~7.3 GiB/tile of replicated parameter memory.

- Risk: FULL_SHARD with FSDP1 means an extra AllGather per layer per forward AND per layer per backward (vs SHARD_GRAD_OP which only AllGathers in fwd). On Aurora's per-layer XPU AllGather costs this could be 2-3× step time. The fork explicitly avoided FSDP2 fully_shard for this reason (per-layer comms deadlock with oneCCL); FSDP1 FULL_SHARD has the same comm pattern but uses XCCL synchronously. Worth measuring step time impact, not just memory.
- Pair with: keep the rest of the config identical to run 38.

### Option C — multi-node BioReason (capacity escape)

The 32B recipe works because it has 24 tiles. BioReason on 2 nodes with 22+2 (22 training, 2 vLLM) would:
- Halve params/tile to ~4 GiB (FULL_SHARD across 22 tiles).
- Still need cross-node weight sync (the static XCCL buffer fix already validated in `project_static_xccl_buffer_fix.md` covers this).
- Eliminate the per-tile pressure problem entirely.

- Risk: BioReason hasn't been tested on multi-node. Possible new bugs in encoder distribution, gloo cross-PG setup, etc. But the recipe path it would use (HSDP / multi-node FSDP1) is the path 32B already validated.
- Cost: needs a 2-node debug-scaling slot.

## Recommendation order for next iteration

1. **A first** — single-line code change, lowest risk, can be tested on the held node within minutes. If peak BWD drops below ~55 GiB this is the entire fix.
2. **A + allocator gc:0.99 + max_split:512** — if A alone leaves headroom thin, raise GC threshold and tune split to reduce fragmentation. Still single-node, single-job.
3. **B if A insufficient** — accept the per-layer AllGather slowdown for guaranteed memory headroom. Time it; if step time exceeds ~150s the recipe is academically interesting only.
4. **C if A and B both fail** — this is the architecturally clean answer and what the codebase already proves works (32B did exactly this), but requires multi-node scheduling.

Notably absent from the list: any further allocator-only tweaks (gc threshold variations, expandable_segments, pluggable allocs). All have been tested in isolation; none address the underlying fact that BWD peak is too high.

## Open questions for the user

- Is the multimodal gate at line 5486 there for a reason we should preserve, or is it a conservative leftover? The git blame should show whether it predates the `TORCHTUNE_USE_CHUNKED_LOSS` env var — if so, it's likely vestigial.
- Is single-node a hard constraint for BioReason, or can we use 2 nodes if needed? The user's framing was "we can fit a 32B on a single node, why not 4B" — the architectural answer (FULL_SHARD across more tiles) suggests the right answer for BioReason might be 2-node FULL_SHARD even though it's "only" 4B.
- The grad_norm=43520 at step 1 is a separate concern. It might just be a transient on a barely-converging RL run with mostly-zero rewards (every reward in run 38 was 0.0). Worth keeping an eye on once memory is stable.

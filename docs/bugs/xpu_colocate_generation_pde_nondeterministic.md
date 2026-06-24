# Non-deterministic CCS/PDE GPU page fault in per-tile in-process colocate generation (max_gen > ~256)

**Status**: ROOT-CAUSE CHARACTERIZED as **non-deterministic** (2026-06-24). NOT fixable by a
recipe config change. Distinct from the Ray TP=8 co-tenancy fault
([`xpu_l0_event_pool_co_tenancy.md`](xpu_l0_event_pool_co_tenancy.md)) — this is the
**single-process, per-tile TP=1 in-process** colocate path (AGPT-2B / Qwen3-4B LoRA colocate),
and it fires during **vLLM generation**, not at the first backward.

## Symptom

```
Segmentation fault from GPU at 0xff00ffff........, ctx_id: 1 (CCS) type: 0 (NotPresent),
level: 1 (PDE) [or level: 0 (PTE)], access: 0 (Read), banned: 1, aborting.
```

- Fires during the **vLLM generation phase** of a training step (not warmup-gen, not the
  trainer backward).
- `ctx_id 1 (CCS)` = compute command streamer; `NotPresent` at page-directory (PDE) or
  page-table (PTE) level = the GPU referenced an **unmapped page** — a freed/remapped KV/page
  region inside vLLM's paged-attention allocator.
- **~24-28 GiB HBM free at the crash** — NOT an OOM.
- Single OS process per tile (in-process vLLM + trainer) — NOT the two-process Ray co-tenancy
  trigger.

## It is NON-DETERMINISTIC (the key finding)

Decisive A/B, job 8559162, same `max_gen=512`, same node, back-to-back:

| leg | config | outcome |
|-----|--------|---------|
| A | `publish_every_steps=999` (no weight sync after step 0) | **crash @ step 1** |
| B | `publish_every_steps=1` (sync every step, 20 publishes, gens up to 3636 tok) | **survived 9 steps clean** |

Same envelope, opposite outcomes; leg B did *more* work and lived. Independently corroborated
by 2026-06-18 data (predating any 2026-06-24 change): two `max_gen=384` LoRA colocate runs —
`lora_colocate_20260618_051008` survived 10 steps, `lora_colocate_20260618_070035` crashed.

**Every apparent deterministic explanation was noise fit to coin-flips** and is REFUTED:
- NOT a max_gen / per-rank token ceiling — warmup-at-max runs a full `seq_len=1536` generation
  and survives, then a *shorter* step-1 gen faults; mg512 both crashes and survives.
- NOT weight sync (`load_weights`) — crashes with sync OFF, survives with sync ON.
- NOT memory/OOM — 24-28 GiB free at every crash, no creep.
- NOT vLLM KV config — `max_model_len=1536`, `GPU KV 54,016 tokens`, `num_gpu_blocks` are
  byte-identical between surviving (mg256) and crashing (mg512) runs (`max_generated_tokens` is
  a sampling param capped by `max_model_len`; it does not change the engine).
- NOT our `LinearGRPOLoss` / varlen changes — reproduced on default `GRPOSimpleLoss`; predates
  them.

`P(crash per generation)` **rises with cumulative generation volume**: max_gen=64 (AGPT-2B
production) effectively never hits it; mg256 usually survives 9+ steps; mg384/512 are flaky
(crash some fraction of runs); warmup-forced max-len gen → crash at step 0-1.

## Likely mechanism (best supported, not pinned to a line)

A probabilistic page-table corruption in **vLLM-on-XPU paged-attention KV block recycling**
under repeated generate cycles — same XPU Level-Zero driver-instability family as the
documented `empty_cache`/UR-handle leak ([`intel_xpu_resource_leak_bug_report.md`](intel_xpu_resource_leak_bug_report.md))
and the oneCCL / torch-xpu-ops#3744 leaks ([`ccl_ipc_handle_cache.md`](ccl_ipc_handle_cache.md)).
Driver/runtime-level, not a torchtune logic bug.

## Practical guidance

- **max_gen ≤ 64**: production-safe (AGPT-2B colocate runs here, never observed faulting).
- **max_gen ≤ 256**: low crash probability; usually clears a normal step budget. Treat as the
  practical colocate ceiling, **NOT a guaranteed-safe hard line**.
- **max_gen ≥ 384**: usable but flaky — will fail a fraction of runs. **Not production-reliable.**
- The paper/BioReason `max_gen=1024` colocate envelope is **blocked** by this until mitigated.

## Mitigations (both heavyweight; no recipe-side deterministic fix exists)

1. **Periodic vLLM engine teardown + respawn** (W17/W19 pattern from the Ray TP=8 work,
   [`xpu_l0_event_pool_co_tenancy.md`](xpu_l0_event_pool_co_tenancy.md)) — resets the L0/KV
   state and bounds accumulation. Adapt to the per-tile in-process path (no Ray): destroy +
   rebuild each rank's in-process engine every N steps, reload the adapter. Multi-session build.
2. **Checkpoint + auto-resume** — accept the flakiness, save every N steps, auto-restart on
   `banned:1`. Operationally simpler, wastes the partial step.
3. **Server / dedicated vLLM mode** instead of colocate — moves vLLM to its own process/tiles
   (no in-process co-residence); the GSM8K production runs already use `vllm_mode=server` at
   max_gen=512 and do not hit this.

## Reproduction

```bash
# 1-node, deterministic-ish trigger via warmup-forced max-len gen at larger max_gen:
env MAX_GEN=512 TORCHTUNE_COLOCATE_NO_FSDP=1 TORCHTUNE_COLOCATE_WARMUP_AT_MAX=1 \
    TORCHTUNE_COLOCATE_MEM_PROBE=1 NSTEPS=10 \
  bash experiments/lora_grpo/run_lora_colocate.sh
# Crashes within the first ~1-2 steps SOME fraction of runs (non-deterministic);
# run 2-3x to observe both crash and survive. max_gen=256 clears 9+ steps.
```
Sweep + isolation jobs: 8559141 (mg 256/512/768), 8559162 (mg512 sync on/off A/B).

## Cross-refs
`memory/project_colocate_mg1024_ccs_pagefault_20260624`,
`memory/project_linear_grpo_loss_chunked_vocab_20260624`,
`docs/reports/colocate_memory_varlen_and_chunked_vocab_20260624.md`.

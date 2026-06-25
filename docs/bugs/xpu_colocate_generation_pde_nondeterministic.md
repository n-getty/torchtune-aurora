# CCS/PDE GPU page fault in per-tile in-process colocate — vLLM weight-publish × XCCL co-residence

**Status**: ROOT CAUSE ISOLATED (2026-06-25) as a **2-factor L0 co-residence interaction**, NOT
fixable by any recipe-config value. Reproduced byte-identically in a **torchtune-free,
frameworks-module-only** script. Supersedes the earlier "non-deterministic, unanalyzable" reading
(that was the low-probability mg512 regime; at mg768 the in-framework recipe faults 8/8). Full
methodology + crash-rate tables: [`docs/reports/colocate_pagefault_investigation_20260625.md`](../reports/colocate_pagefault_investigation_20260625.md).
Same broad family as the Ray TP=8 co-tenancy fault
([`xpu_l0_event_pool_co_tenancy.md`](xpu_l0_event_pool_co_tenancy.md)) — this is its
**single-OS-process, in-process TP=1** analogue, firing during/around the colocate weight publish.

## Symptom

```
Segmentation fault from GPU at 0xff00ffff........, ctx_id: 1 (CCS) type: 0 (NotPresent),
level: 1 (PDE) [or level: 0 (PTE)], access: 0 (Read), banned: 1, aborting.
```
followed by a NEO `drm_neo.cpp:288` abort (SIGABRT). **~24-55 GiB HBM free at the crash — NOT an
OOM.** Single OS process per tile (in-process vLLM + trainer).

## ROOT CAUSE — a 2-factor interaction (HW-isolated, the key finding)

The fault requires **BOTH** of the following to be true at once; **neither alone triggers it**:

1. a real vLLM **`load_weights()`** weight-publish into the **live in-process engine** (the
   colocate adapter publish — `copy_` into the engine's resident device tensors), AND
2. concurrent **cross-rank XCCL collectives** on the same tile (the FSDP / adapter-all-reduce
   training world).

Decisive isolation (standalone, torchtune-free; jobs 8559640/8559650/8559661, 2026-06-25):

| config | real `load_weights` ran? | concurrent multi-rank XCCL? | result |
|--------|--------------------------|-----------------------------|--------|
| single-tile vLLM + `load_weights` (R-LW) | yes | no | **clean 0/12** |
| 12-tile, `load_weights` skipped (KeyError) | no | yes | **clean 12/12** |
| 12-tile + real `load_weights` (v6) | yes | yes | **CRASH (N=2, byte-identical PDE)** |

Mechanism: `load_weights` mutates vLLM's resident device weights while the **trainer's XCCL/L0
driver context is co-resident on the same tile**; that combination corrupts the per-tile page
tables → CCS NotPresent PDE. **It is the resident co-tenancy, not instantaneous concurrency** — a
`barrier()`+sync that removes any *in-flight* collective around the publish does NOT help (quiesce
A/B below, 6/6 crash). This matches the Ray TP=8 W10 finding (zero outstanding submissions still
wedged → the static two-L0-client footprint is the cause). `vllm_mode=server` is immune (engine in
a separate process, no shared L0 context); ezpz is immune (its default rollout is in-trainer HF
`.generate()`, no co-resident vLLM — its vLLM is off-by-default, commented "XPU vLLM is fragile").

**Every recipe-config lever was eliminated** by in-framework A/B (Qwen3-4B, mg768, N≥4/cell, all
crashed; baseline 8/8): `reset_prefix_cache` skip, publish cadence (`publish_every=999`), no-FSDP,
KV-block headroom, `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD` {2048,8192}, and `empty_cache`
cadence. (The earlier "NOT weight sync" A/B was invalid: `publish_every=999` still publishes at
step 0, so its "sync-off" leg still did one `load_weights` — exactly factor 1.)

## Practical guidance

- **max_gen ≤ 64**: production-safe (AGPT-2B colocate runs here, never observed faulting).
- **max_gen ≤ 256**: low crash probability; usually clears a normal step budget. Treat as the
  practical colocate ceiling, **NOT a guaranteed-safe hard line**.
- **max_gen ≥ 384**: usable but flaky — will fail a fraction of runs. **Not production-reliable.**
- The paper/BioReason `max_gen=1024` colocate envelope is **blocked** by this until mitigated.

## Mitigations

1. **Server / dedicated vLLM mode** (RECOMMENDED) — `vllm_mode=server`/`dedicated` moves vLLM to
   its own process/tiles, eliminating factor (b)'s shared L0 context. GSM8K production already runs
   this at max_gen=512 with no fault. This is the validated path for mg≥384.
2. **Quiesce-wsync** (`TORCHTUNE_COLOCATE_QUIESCE_WSYNC=1`) — barrier+sync around the publish.
   **REFUTED 2026-06-25** (A/B job 8559693: 6/6 crash, barrier confirmed engaged). A barrier only
   syncs rank *arrival*; it does not remove the resident co-tenant L0 context, which is the actual
   cause. Kept as a documented dead-end (do not use). Implies any *serialization*-based recipe fix
   is futile — the fix must REMOVE the second L0 client.
3. **Engine sleep/wake around the publish** (`enable_sleep_mode` → `.sleep()` releases vLLM's
   device context, publish, `.wake_up()`) — UNTESTED; the most promising recipe-side lever since it
   actually removes the co-resident L0 client during the mutation. Note vLLM-XPU sleep has its own
   fragility history (`project_colocate_sleep_8b_validated` — 3-step L0 UR leak); validate before relying.
4. **Periodic vLLM engine teardown + respawn** (W17/W19 from the Ray TP=8 work) — heavyweight.
5. **Checkpoint + auto-resume** on `banned:1` — operationally simple, wastes the partial step.

## Practical envelope (colocate, until a mitigation is validated)
max_gen ≤ 64 safe (AGPT-2B production); ≤ 256 low-risk; ≥ 384 flaky; 1024 blocked. **For mg≥384,
use server mode** — it is unaffected.

## Reproduction

**Faithful, torchtune-free (frameworks module only — torch + vllm + ipex + mpi4py):**
```bash
# 12-rank in-process vLLM + XCCL FSDP, with the real Qwen3-4B load_weights publish each step.
qsub -v STEPS=12,LOAD_REAL=1 experiments/colocate/pbs_repro_multitile.sh
# Crashes at step 0 with the byte-identical CCS NotPresent PDE banned:1 (N=2). The single-tile
# control (no XCCL) and the load_weights-skipped run are both CLEAN — see the 2-factor table above.
```
In-framework (the real recipe, mg768 → 4/4 by step ~3-4):
```bash
CELL=baseline M=4 MAX_GEN=768 bash experiments/colocate/run_colocate_ab.sh   # under a PBS job
```
Jobs: standalone 8559640/8559661 (crash), 8559650 R-LW control (clean); in-framework A/B
8559411/8559507/8559508/8559513 (all cells crash). Earlier (superseded) sweeps: 8559141, 8559162.

## Cross-refs
- [`docs/reports/colocate_pagefault_investigation_20260625.md`](../reports/colocate_pagefault_investigation_20260625.md) — full investigation, crash-rate tables, ezpz consistency check.
- [`xpu_l0_event_pool_co_tenancy.md`](xpu_l0_event_pool_co_tenancy.md) — the Ray TP=8 two-process analogue (same L0 co-residence family).
- `memory/project_colocate_pagefault_investigation_20260624`,
  `memory/project_colocate_mg1024_ccs_pagefault_20260624` (investigation history).

# CCS/PDE GPU page fault in per-tile in-process colocate — vLLM weight-publish × XCCL co-residence

**Status**: ROOT CAUSE ISOLATED (2026-06-25) as a **2-factor L0 co-residence interaction**, NOT
fixable by any recipe-config value or any in-process mitigation (quiesce / PG-teardown / vLLM
sleep all REFUTED — see Mitigations). Reproduced byte-identically in a **torchtune-free,
frameworks-module-only** script. Supersedes the earlier "non-deterministic, unanalyzable" reading
(that was the low-probability mg512 regime; at mg768 the in-framework recipe faults 8/8). Full
methodology + crash-rate tables: [`docs/reports/colocate_pagefault_investigation_20260625.md`](../reports/colocate_pagefault_investigation_20260625.md).
Same broad family as the Ray TP=8 co-tenancy fault
([`xpu_l0_event_pool_co_tenancy.md`](xpu_l0_event_pool_co_tenancy.md)) — this is its
**single-OS-process, in-process TP=1** analogue, firing during/around the colocate weight publish.
**Server mode avoids THIS fault (separate process), but is not a blanket escape** — a separate
trainer-side step-6 `banned:1` was found in the Qwen3-4B LoRA 2-node *server* config (Mitigations §1).

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

**No in-process recipe-side fix exists — every lever below that keeps vLLM in-process was tested
and REFUTED (2026-06-25).** Both L0 driver clients (the vLLM engine and the trainer's XCCL
context) live in one OS process, and nothing short of process exit evicts either. Only running
vLLM in a *separate process* (server/dedicated mode) removes a client — but see the server-mode
caveat below.

1. **Server / dedicated vLLM mode** — `vllm_mode=server`/`dedicated` moves vLLM to its own
   process/tiles, eliminating factor (1)'s shared L0 context, so the **colocate** fault does not
   occur. The AGPT-2B GSM8K server production envelope (full-FT, 2B) runs clean for hours.
   **CAVEAT (2026-06-25):** server mode is NOT a blanket clean path. A first-ever multi-step soak
   of **Qwen3-4B LoRA 2-node server** faults **trainer-side** (node-0 grpo_step) at **step ~5-6**
   with the same `banned:1` — reproduced 4× across nodes, **both `delta` and `merged`** publish
   (jobs 8559836/8559857/8559882). That is a **separate** fault (vLLM is on another node; the
   publish succeeds), still under diagnosis — see [the investigation report](../reports/colocate_pagefault_investigation_20260625.md)
   §"NEW separate issue". So: server mode dodges the *colocate* bug, but the 4B-LoRA-2N server
   config has its own step-6 trainer-side stability fault — **do not assume server+LoRA at 4B/2N is
   production-ready until that soak clears.** The 2B full-FT server envelope is the known-good one.
2. **Quiesce-wsync** (`TORCHTUNE_COLOCATE_QUIESCE_WSYNC=1`) — barrier+sync around the publish.
   **REFUTED** (job 8559693: 6/6 crash, barrier confirmed engaged). A barrier only syncs rank
   *arrival*; it does not remove the resident co-tenant L0 context. Any *serialization*-based fix
   is futile. Kept as a documented dead-end (do not use).
3. **PG teardown around the publish** (`destroy_process_group` before `load_weights`, recreate
   after — reproducer `--fix cedrain`) — **REFUTED** (job 8559753: still crashes). It only drops
   the PyTorch PG *wrapper*; the underlying L0 driver context + FSDP device memory persist, so the
   co-tenancy is unchanged.
4. **Engine sleep/wake around the publish** (`TORCHTUNE_COLOCATE_SLEEP_WSYNC=1` / `--fix sleep`,
   `enable_sleep_mode` → `.sleep()` → publish → `.wake_up()`) — **REFUTED** at both
   `sleep(level=10)` (KV-only; weights stay resident, mutated in place — job 8559763) and
   `sleep(level=2)` (discard weights — job 8559771). Both crash byte-identically. VLLM sleep frees
   *vLLM's* device state but not the *trainer's* co-resident L0 context.
5. **Periodic vLLM engine teardown + respawn** (W17/W19 from the Ray TP=8 work) — **not viable
   in-process**: `_init_vllm_early` calls `destroy_process_group`, which would also tear down the
   live XCCL training PG. The Ray version killed a *separate* vLLM actor process; there is no
   in-process analogue.
6. **Checkpoint + auto-resume** on `banned:1` — operationally simple, wastes the partial step.

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
8559411/8559507/8559508/8559513 (all cells crash). Refuted in-process fixes: 8559693 (quiesce),
8559753 (cedrain PG-teardown), 8559763 (sleep L10), 8559771 (sleep L2) — all crash. Server-mode
4B-LoRA-2N step-6 trainer fault: 8559836/8559857 (delta), 8559882 (merged). Earlier (superseded)
sweeps: 8559141, 8559162.

## Cross-refs
- [`docs/reports/colocate_pagefault_investigation_20260625.md`](../reports/colocate_pagefault_investigation_20260625.md) — full investigation, crash-rate tables, ezpz consistency check.
- [`xpu_l0_event_pool_co_tenancy.md`](xpu_l0_event_pool_co_tenancy.md) — the Ray TP=8 two-process analogue (same L0 co-residence family).
- `memory/project_colocate_pagefault_investigation_20260624`,
  `memory/project_colocate_mg1024_ccs_pagefault_20260624` (investigation history).

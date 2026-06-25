# Colocate vLLM generation page fault — systematic investigation (2026-06-25)

Follow-up to `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md`. That doc concluded the
fault was "probabilistic, no recipe-side fix." This investigation tested that conclusion with a
crash-rate methodology (N≥4/cell, genuine-fault-verified) and a dependency-limited standalone
reproducer. **Both halves of the original conclusion are refined: the fault is near-deterministic
at mg768, and "no recipe-side fix" is now established by elimination rather than assumed.**

> STATUS: complete. Headline result — the fault is a **2-factor interaction** reproduced in a
> **torchtune-free, frameworks-module-only** script: it requires BOTH (a) a real vLLM
> `load_weights()` weight-publish into the live in-process engine AND (b) concurrent cross-rank
> XCCL collectives on the same tiles. Neither factor alone faults.

## Method

- **In-framework A/B** (`experiments/colocate/run_colocate_ab.sh` + `pbs_colocate_ab.sh`): the real
  12-rank LoRA-GRPO colocate recipe, Qwen3-4B, mg768, one independent variable per cell, M=4 runs
  each, back-to-back on one node. Crash = genuine `CCS NotPresent PDE banned:1` in the run log
  (verified, not just nonzero exit).
- **Standalone reproducer** (`scratch/repro_colocate_pagefault*.py`): torch + vLLM only, ZERO
  torchtune. Single-tile ladder (R-A..R-E) and a 12-rank multi-tile variant (in-process vLLM +
  XCCL FSDP per rank). Confirmed faithful to the recipe's vLLM init (identical KV geometry).

## Result 1 — the fault is near-DETERMINISTIC at mg768, not stochastic

Baseline (real recipe) crashes **4/4** at **step ~3-4**, after 3 clean steps with real reward.
The precursor is striking and reproducible across **5/5** runs: steps 0-2 run ~23s with the
adapter all-reduce (a 66 MB XCCL collective) at <2s; **at step 3 that all-reduce explodes to
39-41s**, then the GPU page fault fires at ~55 GiB free (not OOM). The original "stochastic"
reading was the lower-probability mg512 regime; at mg768 it is essentially deterministic.

## Result 2 — every recipe-side lever ELIMINATED

All cells crashed 4/4 with the genuine fault (mg768):

| Cell | varies | crashed/N | conclusion |
|------|--------|-----------|------------|
| baseline | — | **8/8** | reliable repro |
| noreset | `SKIP_RESET_PREFIX=1` (reset_prefix_cache skipped, verified) | 4/4 | reset_prefix NOT trigger |
| pub999 | `publish_every=999` (one step-0 publish only) | 4/4 | publish *cadence* NOT trigger (still publishes at step 0) |
| nofsdp | `NO_FSDP=1` (no FSDP wrap/collectives) | 4/4 | FSDP wrap NOT required (crashes slightly earlier) |
| bigkv | 4000 KV blocks, gpu_mem 0.55 | 4/4 | KV headroom does not help |
| ccl_mid / ccl_low | IPC-handle cache threshold 8192 / 2048 | 4/4 / 4/4 | IPC-handle threshold does not help |
| noreclaim | `NO_FSDP` + ~no empty_cache | 4/4 | empty_cache cadence does not help |

Every application-config lever fails. This matches the Ray TP=8 co-tenancy W-probe history
(`xpu_l0_event_pool_co_tenancy.md`: W4-W16 all failed; only process isolation worked). **The fault
is a driver-level L0 co-residence interaction, not addressable by recipe config.** The
architectural fix is `vllm_mode=server`/`dedicated` (separate process) — already the validated
production path at mg512+. (Note: pub999 still crashes because it *does* publish at step 0 — see
the standalone isolation below, which proves the weight-publish itself, not its cadence, is one of
the two required factors.)

## Result 3 — standalone single-tile ladder: clean (rules out single-tile causes)

| Rung | adds | crashed/N |
|------|------|-----------|
| R-A | vLLM generate only | 0/12 |
| R-B | + resident model | 0/12 |
| R-D | + trainer compute + empty_cache churn | 0/12 |

(R-E `--fsdp on` was a harness error — world=1 FSDP all_reduce on gloo — not a GPU fault.)
The single-tile harness cannot reproduce the fault → it requires multi-rank co-residence.

## Result 4 — standalone MULTI-TILE reproducer: clean across iterations (so far)

12-rank in-process vLLM + XCCL FSDP, torchtune-free (`frameworks` module only: torch + vllm +
ipex + mpi4py). Iterated to isolate the trigger:
- v1 trivial trainer: 12/12 clean. v2 + activation pressure: clean. v3 + real SDPA attention
  (driven to 20.8 GiB free ≈ real crash envelope): clean. v4 40-step volume: clean (AR ≤0.15s)
  → **volume is not the trigger**.
- v5 per-step `load_weights` (self-named round-trip): clean — but `load_weights` threw
  `KeyError(qkv_proj)` (vLLM expects HF checkpoint names), so it never executed. Inconclusive.
- **v6** per-step `load_weights` of the **real Qwen3-4B HF safetensors** into the live engine:
  **CRASH — byte-identical `CCS NotPresent PDE banned:1` + NEO `drm_neo.cpp:288`**, at step 0
  right after all 12 ranks ran load_weights (0 load failures, 398 HF params/3 shards). Reproduced
  N=2 (jobs 8559640, 8559661).
- **R-LW** (single-tile control): vLLM + the *same* real `load_weights()` each step, but NO
  XCCL multi-rank trainer → **0/12 clean** (load_weights ran, 12 iters). load_weights ALONE does
  not fault.

**This isolates a 2-factor interaction:**

| config | real `load_weights` ran? | concurrent multi-rank XCCL? | faults? |
|--------|--------------------------|-----------------------------|---------|
| R-LW (1 tile) | yes | no | clean 0/12 |
| v5 (12 tiles) | no (KeyError) | yes | clean 12/12 |
| v6 (12 tiles) | yes | yes | **CRASH** |

The fault requires **both** a real weight-publish into the live in-process vLLM engine **and**
concurrent cross-rank XCCL collectives on the same tiles. Neither alone suffices. Mechanistically:
`load_weights` mutates vLLM's resident device tensors (`copy_` into model weights) while XCCL holds
L0 driver state on the same tile — the combination corrupts the per-tile page tables → CCS
NotPresent PDE. `vllm_mode=server` is immune because the engine is a separate process (no shared
L0 context); ezpz is immune because its default rollout is in-trainer HF `.generate()` with no
second co-resident engine (its vLLM is off-by-default — "XPU vLLM is fragile").

**`scratch/repro_colocate_pagefault_multitile.py` (run with `--load-real-weights`) is the minimal
faithful, torchtune-free reproducer** — the Intel/vLLM handoff artifact. It cannot be reduced to a
single process (R-LW proves the XCCL co-residence factor is required).

## Conclusion

1. **Mechanism (refined):** a **2-factor L0 co-residence interaction** — a real vLLM
   `load_weights()` weight-publish into the live in-process engine, concurrent with cross-rank
   XCCL collectives on the same tile, corrupts the per-tile GPU page tables → `CCS NotPresent PDE
   banned:1`. Same family as the Ray TP=8 UR40 co-tenancy wedge; this is the single-OS-process
   in-process analogue. The original "stochastic KV-paging" reading was the low-probability mg512
   regime; at mg768 it is near-deterministic (baseline 8/8).
2. **No recipe-config fix exists** — established by eliminating *every* application lever
   (reset_prefix, publish cadence, FSDP, KV headroom, IPC-handle threshold, empty_cache cadence),
   not assumed. Use `vllm_mode=server`/`dedicated` for mg≥384 (the validated production path).
   Colocate is safe only at small mg (≤256 low-risk; ≤64 AGPT-2B production).
3. **Handoff:** `scratch/repro_colocate_pagefault_multitile.py --load-real-weights` is a faithful,
   **torchtune-free** (frameworks-module-only) reproducer — crashes N=2 with the byte-identical
   driver signature. This is the clean artifact to file with Intel/vLLM. It needs the multi-tile
   XCCL world (single-process R-LW control is clean), so it cannot be reduced below 12 ranks.
4. **Mitigations:**
   - **Quiesce XCCL around the publish** (`TORCHTUNE_COLOCATE_QUIESCE_WSYNC=1`) — **REFUTED**
     (job 8559693: 6/6 crash, barrier confirmed engaged). A barrier syncs rank *arrival* but does
     not remove the resident co-tenant L0 context — and *that residency*, not instantaneous
     collective overlap, is the cause (matches Ray TP=8 W10: zero outstanding submissions still
     wedged). **Any serialization-based recipe fix is futile;** the fix must remove the 2nd L0 client.
   - **PG teardown around the publish** (`--fix cedrain`: `destroy_process_group` before
     load_weights, recreate after) — **REFUTED** (job 8559753: still crashes). `destroy_process_group`
     only drops the PyTorch PG *wrapper*; the underlying L0 driver context + FSDP device memory
     persist, so the co-tenancy is unchanged.
   - **vLLM engine sleep/wake around the publish** (`TORCHTUNE_COLOCATE_SLEEP_WSYNC=1` /
     `--fix sleep`) — **REFUTED** at both `sleep(level=10)` (KV-only — keeps weights resident, so
     load_weights mutates the same entangled tensors; job 8559763) and `sleep(level=2)` (discard
     weights; job 8559771). Both crash byte-identically.
   - **Server/dedicated mode** — the validated, RECOMMENDED path. Removes a whole L0 client by
     running vLLM in a separate process. With the `delta` publish it is also *fast*: ~26.7s/step
     (2× faster than `merged`) at G=8/max_gen=384, HW-validated bit-exact
     (`docs/reports/lora_delta_publish_path_20260617.md`). **This is the performant working RL path.**
   - Engine teardown+respawn (W17/W19) — not viable in-process: `_init_vllm_early` calls
     `destroy_process_group`, which would also kill the live XCCL training PG. The Ray version
     killed a *separate* vLLM actor process; there is no in-process analogue.

   **Why no in-process fix exists (the definitive finding):** both L0 driver clients (the vLLM
   engine and the trainer's XCCL context) live in **one OS process**. The corruption from
   `load_weights`-under-co-residence cannot be avoided by serialization (barrier), PG teardown
   (`destroy_process_group` doesn't free the L0 context), or vLLM sleep (doesn't free the
   *trainer's* L0 context). Only removing a whole L0 client works — i.e. a separate process —
   which is exactly why the Ray TP=8 fix had to **kill the vLLM process** (W17). In-process
   colocate at mg≥384 is therefore **driver-blocked**; the supported performant path is
   server/dedicated mode (recommended: `+ delta` publish).

## NEW separate issue — server-mode 4B-LoRA-2N step-6 trainer fault is a MEMORY LEAK, not L0 co-residence

A multi-step soak of **Qwen3-4B LoRA 2-node *server* mode** faults trainer-side at **step ~6**
with `banned:1` (jobs 8559836/8559857/8559882; both `delta` and `merged`). It was initially
filed alongside the colocate fault; **diagnosis 2026-06-25 from
`experiments/lora_grpo/run_lora_grpo_2node_20260625_051455.log` shows it is a DIFFERENT bug —
a trainer-side per-step memory creep → OOM at the tile ceiling.**

Evidence it is **not** the 2-factor L0 co-residence fault:
- **vLLM is on a separate node** — trainer `x4412c6s2b0n0`, vLLM `x4412c6s4b0n0`. No co-resident
  L0 engine client on the trainer tiles, so the 2-factor mechanism cannot apply.
- The publish **succeeds** (`publish join 13.18s`) and vLLM generation **succeeds** (64 seqs in
  9.0s); the fault fires afterward in **`ref_fwd`** on trainer ranks 0 & 2.
- Fault is `access: 1 (Write)`; the colocate fault is `access: 0 (Read)`.

Evidence it is a **live memory leak** (not allocator caching, not the bounded varlen cache):

| step | active GiB | reserved GiB |
|------|-----------|--------------|
| 1    | 39.17     | 45.59        |
| 2    | 44.10     | 55.75        |
| 4    | 49.07     | 56.74        |
| 5    | 50.59     | 59.47        |
| 6    | 52.86     | **60.34** → fault in ref_fwd |

- ~2.7 GiB/step; `active` and `reserved` rise together → genuine reference retention, not caching.
- Larger than the known `_varlen_out_cache` variable-seqlen leak (capped at
  `TORCHTUNE_VARLEN_CACHE_MAX=8` ≈ bounded; `varlen=engaged` confirmed in this run), so it is a
  **separate** retention.
- Suspects (under bisection): per-step merged-publish staging buffers, master-fp32 copies, or
  ungated gather buffers in the server publish path.

**Action:** hold-node bisect with a per-step component memory probe + A/B
(`TORCHTUNE_USE_CHUNKED_LOSS`, varlen off, publish-path probes). Until the leak is fixed,
4B-LoRA-2N server is **not** production-ready; the known-good server envelope is AGPT-2B full-FT.

## Artifacts

- `scratch/repro_colocate_pagefault.py` (single-tile ladder), `..._multitile.py` (12-rank).
- `experiments/colocate/{run_repro_ladder,run_colocate_ab,pbs_colocate_ab,pbs_repro_multitile}.sh`.
- `experiments/colocate/{repro_results,ab_results}.tsv` (raw crash-rate data).
- `experiments/colocate/README_repro.md` (Intel handoff package).
- Recipe: env-gated `TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX` guard +
  `tests/torchtune/dev/rl/test_colocate_skip_reset_prefix.py` (CPU, green).

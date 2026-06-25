# Session summary: colocate vLLM page-fault — root-caused + performant path (2026-06-25)

Overnight investigation of the LoRA-GRPO `vllm_mode=colocate` GPU page fault, with the goal of a
functional, performant colocate RL path. Branch: `investigate/colocate-pagefault-20260625`.

## TL;DR

- **In-process colocate (vLLM on the same tile as the trainer) is BLOCKED at the Intel driver
  level** — a single-OS-process, two-L0-client page-table corruption. Root-caused precisely and
  reproduced in a **torchtune-free** script for an Intel/vLLM filing. *(This is the solid result.)*
- **Every in-process recipe-side fix was implemented and refuted** (not assumed).
- **Server mode (vLLM in a separate process) avoids the colocate co-residence fault**, and `delta`
  publish gives a ~26.8s/step rate (2× faster than `merged`). **HOWEVER** — a multi-step soak run
  overnight (jobs 8559836, 8559857) uncovered a **separate, reproducible trainer-side
  `banned:1` at step ~6** in the 4B-LoRA-2N config (both `delta`; `merged` test in flight). So
  server+delta is **performance-confirmed but NOT yet stability-validated** at this envelope — the
  earlier "validated" claim rested on a step-time measurement, not a soak (the soak was never run
  until now). Diagnosis in progress; see below. Known-good reference: AGPT-2B GSM8K server-mode
  production runs (different recipe/config) run clean.

## Root cause (definitive)

The fault is a **2-factor L0 co-residence interaction**, fires as
`CCS NotPresent / PDE banned:1` (NEO `drm_neo.cpp:288` abort), ~24-55 GiB HBM free (not OOM):

1. a real vLLM `load_weights()` weight-publish into the **live in-process engine**, AND
2. a **resident trainer XCCL/L0 context** on the same tile.

Neither alone faults — proven by a 3-way standalone isolation (single-tile load_weights = clean;
multi-tile without load_weights = clean; both = crash, N=2, byte-identical). At mg768 the
in-framework recipe crashes near-deterministically (baseline 8/8, ~step 3). The earlier
"non-deterministic" reading was the lower-probability mg512 regime.

## Why there is no in-process fix (all refuted by direct experiment)

| Lever | Why it should have worked | Result |
|-------|---------------------------|--------|
| reset_prefix / publish cadence / FSDP / KV headroom / IPC-handle threshold / empty_cache cadence | recipe-config knobs | all crash (A/B, N≥4 each) |
| QUIESCE (barrier+sync around publish) | remove in-flight collective overlap | crash — it is *resident context*, not timing |
| cedrain (`destroy_process_group` around publish) | remove the XCCL PG | crash — only drops the PG *wrapper*, not the L0 context |
| vLLM `sleep(level=10)` (KV-only) | release vLLM KV/L0 | crash — weights stay resident, mutated in place |
| vLLM `sleep(level=2)` (discard weights) | load into fresh tensors | crash |

Both L0 driver clients live in **one OS process**; nothing short of process exit evicts either.
This is the in-process analogue of the Ray TP=8 UR40 co-tenancy bug — whose only fix was to **kill
the vLLM process** (W17). There is no in-process equivalent (`_init_vllm_early` tears down the
XCCL PG), so in-process colocate at mg≥384 is driver-blocked.

## The performant working path: server + delta

`vllm_mode=server` runs vLLM in a separate process (no shared L0 context → immune). With the
`delta` publish (ship base once, then ~66 MB lora_a/b per step, merge on the vLLM worker):
- **~26.7s/step at G=8/max_gen=384** (2× faster than `merged` ~52s), bit-exact, GREEN
  (`docs/reports/lora_delta_publish_path_20260617.md`).
- Launch: `experiments/lora_grpo/run_qwen3_4b_lora_2node.sh` with `LORA_PUBLISH_MODE=delta`, or the
  self-terminating `batch_qwen3_4b_lora_2node.sh -v LORA_PUBLISH_MODE=delta`.
- **Soak validation:** steps 0-4 ran clean at **~26.8s/step** (matches the validated 26.7s), delta
  publish join ~0.49s. First 20-step attempt (8559836) hit a **trainer-side** (node 0) FSDP/XCCL
  `banned:1` at step 6 — NOT vLLM co-residence (vLLM is on node 1); suspected transient node/tile
  degradation (the separate documented FSDP+XCCL banned:1 class, not the colocate 2-factor bug).
  Re-running on a fresh allocation (8559857) to confirm — N=1 banned:1 is not conclusive (node
  variance / degraded-tile discipline). _Re-run result pending._

## Deliverables (branch `investigate/colocate-pagefault-20260625`)

- **Intel handoff:** `scratch/repro_colocate_pagefault_multitile.py --load-real-weights` —
  torchtune-free (torch+vllm+ipex+mpi4py), reproduces the byte-identical fault (N=2). Single-tile
  ladder `scratch/repro_colocate_pagefault.py` isolates the factors.
- **A/B harness + data:** `experiments/colocate/{run_colocate_ab,pbs_colocate_ab,run_repro_ladder,
  pbs_repro_multitile}.sh`, `ab_results.tsv` / `repro_results.tsv`.
- **Docs:** `docs/bugs/xpu_colocate_generation_pde_nondeterministic.md` (rewritten to the 2-factor
  root cause), `docs/reports/colocate_pagefault_investigation_20260625.md` (full methodology),
  `experiments/colocate/README_repro.md` (handoff package: versions, crash tables).
- **Recipe:** env-gated `TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX` (verified not the trigger),
  `QUIESCE_WSYNC` (refuted), `SLEEP_WSYNC` (refuted) — all default-off, byte-identical off-path;
  CLAUDE.md table rows + `tests/torchtune/dev/rl/test_colocate_skip_reset_prefix.py` (green).

## NEW separate issue uncovered by the soak (server-mode 4B-LoRA-2N step-6 trainer fault)

The 4B-LoRA **2-node server** run faults **trainer-side** (node 0 ranks) at **step ~5-6**, entering
`grpo_step` after generation — `CCS NotPresent PDE banned:1 access=Write`. **Reproduced 4×** across
multiple nodes, **both `delta` AND `merged`** publish (8559836, 8559857, 8559882). Pattern: steps
0-4 clean, step 5 `grpo_step` spikes (45-181s), step 6 faults. Findings:
- **Not delta-specific** (merged faults too), **not vLLM co-residence** (vLLM is on node 1),
  **not node variance** (3+ nodes), **not the publish call** (it succeeds each step).
- **Not the CCL IPC-handle threshold**: the known-good AGPT-2B 2N GSM8K *server* run uses the same
  `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536` and runs clean for hours. Ruled out.
- The discriminator vs known-good is **model size (4B vs 2B) / LoRA + adapter-publish / FSDP-at-2N**.
  Suspect: the LoRA adapter-publish path or 4B FSDP at 2N. **Not chased further overnight** — it is
  a second, separate investigation beyond the colocate scope; flagged here for a daytime follow-up.
- **Possible connection:** the step-6 `grpo_step`-slowdown→`banned:1` precursor *resembles* the
  colocate precursor. It is worth checking whether the colocate "factor 2" (resident trainer
  XCCL/L0) and this 2N-server trainer fault share a root cause — i.e. whether the trainer-side
  FSDP/XCCL path itself has an L0-accumulation fault that colocate co-residence merely accelerates.
  (The standalone isolation — XCCL+FSDP without load_weights ran clean at step 0 single-node —
  argues they may differ, but this was not tested at 2-node/step-6.)

## Recommendation (interim — pending the diagnosis above)

1. **In-process colocate: do not pursue** without an Intel driver fix — file the torchtune-free
   reproducer upstream (vLLM-XPU / Intel compute-runtime). This conclusion is solid.
2. **Server mode is the right architecture** (separate process avoids the co-residence fault), and
   `delta` gives the 2× step-time win — but **run a clean multi-step soak before declaring it
   production-ready**; the overnight soak found a step-6 trainer fault still being diagnosed.
   Known-good today: the AGPT-2B GSM8K server-mode production envelope.
3. Software stack for the filing: frameworks/2025.3.1, torch 2.10.0a0, ipex 2.10.10, vllm 0.15.0,
   i915 25.2.29, NEO 25.18.33578, ze_loader 1.24.0.

# Session summary: colocate vLLM page-fault — root-caused + performant path (2026-06-25)

Overnight investigation of the LoRA-GRPO `vllm_mode=colocate` GPU page fault, with the goal of a
functional, performant colocate RL path. Branch: `investigate/colocate-pagefault-20260625`.

## TL;DR

- **In-process colocate (vLLM on the same tile as the trainer) is BLOCKED at the Intel driver
  level** — a single-OS-process, two-L0-client page-table corruption. Root-caused precisely and
  reproduced in a **torchtune-free** script for an Intel/vLLM filing.
- **Every in-process recipe-side fix was implemented and refuted** (not assumed).
- **The performant, working RL path is `vllm_mode=server` + `delta` publish** — vLLM in a separate
  process, ~26.7s/step (2× faster than `merged`), HW-validated bit-exact. Use this for production.

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
- **Soak validation:** _job 8559836 (20-step delta) — result pending; fill on completion._

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

## Recommendation

1. **Production RL: use server + delta** (`LORA_PUBLISH_MODE=delta`). Consider making `delta` the
   YAML default once the soak (8559836) clears 20+ steps GREEN.
2. **Do not pursue in-process colocate further** without an Intel driver fix — file the standalone
   reproducer upstream (vLLM-XPU / Intel compute-runtime).
3. Software stack for the filing: frameworks/2025.3.1, torch 2.10.0a0, ipex 2.10.10, vllm 0.15.0,
   i915 25.2.29, NEO 25.18.33578, ze_loader 1.24.0.

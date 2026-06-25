#!/bin/bash
# Bisect the 4B-LoRA-2N server-mode step-6 trainer memory leak (banned:1 = OOM).
# Diagnosis: ~2.7 GiB/step active creep -> 64 GiB tile ceiling at step ~6.
# See docs/bugs/xpu_colocate_generation_pde_nondeterministic.md "NEW separate issue".
#
# Requires a free 2-node debug (or debug-scaling) slot. Do NOT submit while another
# of your jobs occupies the target queue (1 job/user/queue).
#
# Usage:
#   qsub -q debug -l walltime=00:50:00 experiments/lora_grpo/bisect_step6_leak.sh
#
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -N bisect_step6_leak
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/bisect_step6_leak.out

set -eo pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "${TT}"

# ── Bisect legs (run one at a time; flip the env, resubmit) ──────────────────
# All legs run NSTEPS=8 (crash is step ~6) with the per-phase probe ON so
# COLOCATE_PHASEPROBE lines attribute the per-step active-GiB growth to
# gen / grpo_step / sync. Localize the leaking phase first, then the buffer.
#
#   LEG=baseline  : probe on, everything else default (chunked-loss=1, varlen=1).
#                   Expect crash ~step6; read which phase's ACTIVE delta is +ve/step.
#   LEG=novarlen  : + TORCHTUNE_USE_IPEX_VARLEN=0 TORCHTUNE_MASKFREE_CAUSAL=0.
#                   Rules in/out the varlen no-grad output cache as the leak.
#   LEG=single_bwd: + TORCHTUNE_USE_CHUNKED_LOSS=0 (chunked fwd+bwd).
#                   Rules in/out per-chunk retention in grpo_step.
#   LEG=full_shard: + LORA_FSDP_FULL_SHARD=1 (ZeRO-3, reshards after forward).
#                   PRIMARY FIX CANDIDATE: SHARD_GRAD_OP (ZeRO-2, the default)
#                   keeps unsharded params resident after forward; the no-grad
#                   ref_fwd under disable_adapter re-gathers them each step with
#                   no backward to free them -> ~2.5 GiB/step floor creep (baseline
#                   leg, job 8560856). FULL_SHARD reshards, so the floor should be
#                   FLAT. Baseline-leg floor: 3.97->7.01->9.46->12.11->14.67->17.19.
LEG=${LEG:-baseline}

export MEM_PROBE=1            # -> TORCHTUNE_COLOCATE_MEM_PROBE=1 (per-phase probe)
export NSTEPS=8              # reach the step-6 fault with 2 steps of margin

case "${LEG}" in
  baseline)   ;;
  novarlen)   export TORCHTUNE_USE_IPEX_VARLEN=0; export TORCHTUNE_MASKFREE_CAUSAL=0 ;;
  single_bwd) export TORCHTUNE_USE_CHUNKED_LOSS=0 ;;
  full_shard) export LORA_FSDP_FULL_SHARD=1 ;;
  *) echo "unknown LEG=${LEG}"; exit 2 ;;
esac

echo "=== bisect_step6_leak LEG=${LEG} NSTEPS=${NSTEPS} MEM_PROBE=${MEM_PROBE} ==="
bash "${TT}/experiments/lora_grpo/run_qwen3_4b_lora_2node.sh"

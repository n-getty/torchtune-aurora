#!/bin/bash
#PBS -N br_s1nex_4n
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=4
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_stage1norm_exhaustive.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_stage1norm_exhaustive.err
#
# BioReason Qwen3-32B — EXP 1: stage1norm + EXHAUSTIVE TARGET.
#
# ★ Identical to pbs_4n_sft_stage1norm.sh (frozen backbone, no LoRA, LayerNorm projector,
# layer-37 ESM3) EXCEPT the config appends the FULL GT GO-term list to the SFT target so the
# model learns to ENUMERATE the full answer (CAFA F_max is ancestor-propagated -> rewards
# breadth). Baselines: echo-go_pred 0.69, our terse stage1norm 0.6008. Target is GT (not
# go_pred) so it can in principle EXCEED echo. Watch generation stays coherent (Run-1 risk).
#
# Config: recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_exhaustive_xpu.yaml
# Output: experiments/bioreason/runs/sft_qwen3_32b_stage1norm_exhaustive
#
# Thin wrapper over pbs_4n_sft_full.sh. Resume next segment with BIOREASON_RESUME=1.

export CONFIG=recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_exhaustive_xpu.yaml
export OUTDIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_stage1norm_exhaustive
export EPOCHS=1
export SAVE_EVERY=${SAVE_EVERY:-20}
export BIOREASON_RESUME=${BIOREASON_RESUME:-0}

exec bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/pbs_4n_sft_full.sh

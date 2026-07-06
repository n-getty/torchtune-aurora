#!/bin/bash
#PBS -N br_s1nld_4n
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=4
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_stage1norm_leakdrop.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_stage1norm_leakdrop.err
#
# BioReason Qwen3-32B — RUN 1: stage1norm + go_pred LEAK-DROPOUT (0.5).
#
# ★ Identical to pbs_4n_sft_stage1norm.sh (frozen backbone, no LoRA, LayerNorm projector,
# layer-37 ESM3) EXCEPT the config drops go_pred on 50% of TRAINING samples so the protein/GO
# features must carry the GO signal instead of the prompt-copy shortcut (loss->0 via the leak).
# Eval always injects go_pred. Baseline to beat: stage1norm F_max 0.6008.
#
# Config: recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_leakdrop_xpu.yaml
# Output: experiments/bioreason/runs/sft_qwen3_32b_stage1norm_leakdrop
#
# Thin wrapper over pbs_4n_sft_full.sh. Resume next segment with BIOREASON_RESUME=1.
# Frozen backbone -> tiny backward; step time ~290s (frozen-32B forward dominates), 8h ~= 100 steps.

export CONFIG=recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_leakdrop_xpu.yaml
export OUTDIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_stage1norm_leakdrop
export EPOCHS=1
export SAVE_EVERY=${SAVE_EVERY:-20}
export BIOREASON_RESUME=${BIOREASON_RESUME:-0}

exec bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/pbs_4n_sft_full.sh

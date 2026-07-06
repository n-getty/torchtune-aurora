#!/bin/bash
#PBS -N br_ex_noemb_4n
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=4
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_exhaustive_noembed.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_exhaustive_noembed.err
#
# BioReason Qwen3-32B — EXP 2b: EXHAUSTIVE target + PROTEIN-EMBED OFF (the embed-off leg).
#
# ★ The embed-on/off ablation the paper skipped, as a clean single-variable A/B. Embed-ON leg =
# pbs_4n_sft_stage1norm_exhaustive.sh (job 8640188). This leg is IDENTICAL except
# model.disable_protein_splice=true (placeholder tokens kept, ESM3 features NOT written), trained
# from scratch (NOT eval-time yanked -> not OOD). Compare final F_max: embed-OFF == embed-ON =>
# ESM3 embedding redundant with go_pred; embed-OFF < embed-ON => embedding genuinely contributes.
#
# Config: recipes/configs/dev/production/sft_bioreason_qwen3_32B_exhaustive_noembed_xpu.yaml
# Output: experiments/bioreason/runs/sft_qwen3_32b_exhaustive_noembed
#
# Thin wrapper over pbs_4n_sft_full.sh. Resume next segment with BIOREASON_RESUME=1.

export CONFIG=recipes/configs/dev/production/sft_bioreason_qwen3_32B_exhaustive_noembed_xpu.yaml
export OUTDIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_exhaustive_noembed
export EPOCHS=1
export SAVE_EVERY=${SAVE_EVERY:-20}
export BIOREASON_RESUME=${BIOREASON_RESUME:-0}

exec bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/pbs_4n_sft_full.sh

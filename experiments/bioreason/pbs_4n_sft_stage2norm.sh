#!/bin/bash
#PBS -N br_s2norm_4n
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=4
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_stage2norm.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_4n_sft_stage2norm.err
#
# BioReason Qwen3-32B — RUN 2: gated Stage-2 backbone (LoRA @ lr 2e-5) on the CLEAN projector.
#
# ★ LoRA-finetunes the backbone on top of the validated clean Stage-1 (frozen+LayerNorm) projector.
# The published recipe's Stage 2; the 0.6686 anchor is a post-Stage-2 number. GATED: the prior
# Stage-2 collapsed to ':' but on the OLD unbounded projector — this uses the LayerNorm-bounded
# projector + lr lowered 1e-4->2e-5. SAVE_EVERY=10 so the FIRST checkpoint lands early for the
# probe gate: run probe_collapse.py against runs/sft_qwen3_32b_stage2norm/epoch_0 BEFORE letting
# it run long. If ':' reappears -> qdel + fall back to Run-1 projector-only. If coherent -> let run.
#
# Config: recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage2norm_xpu.yaml
#   (verify stage1_proj_dir points at the Run-1-chosen best clean projector before submit)
# Output: experiments/bioreason/runs/sft_qwen3_32b_stage2norm
#
# Thin wrapper over pbs_4n_sft_full.sh. Resume next segment with BIOREASON_RESUME=1.

export CONFIG=recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage2norm_xpu.yaml
export OUTDIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_stage2norm
export EPOCHS=1
export SAVE_EVERY=${SAVE_EVERY:-10}
export BIOREASON_RESUME=${BIOREASON_RESUME:-0}

exec bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/pbs_4n_sft_full.sh

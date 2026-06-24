#!/bin/bash
# BioReason F_max eval over LOCAL parquet (no gated data) at FAITHFUL inputs.
# Generates per-aspect prediction JSONs for a checkpoint, then scores with the
# paper's UNMODIFIED cafa_evals.py (reasoning_mode). One XPU tile.
#
# Usage (on a hold node):
#   CKPT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft TAG=sft \
#     bash experiments/bioreason/run_eval_fmax.sh
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
cd $TT

CKPT=${CKPT:-/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft}
TAG=${TAG:-sft}
CACHE=${CACHE:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt}
PARQUET=${PARQUET:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl}
MAXSAMPLES=${MAXSAMPLES:--1}
MAXGEN=${MAXGEN:-2048}
OUT=$TT/experiments/bioreason/eval_out/$TAG

echo "=== eval start $(date) ckpt=$CKPT tag=$TAG tile=$ZE_AFFINITY_MASK ==="
python experiments/bioreason/eval_cafa_fmax.py \
  --ckpt_dir $CKPT \
  --local_parquet $PARQUET \
  --esm3_cache_path $CACHE \
  --out $OUT \
  --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens $MAXGEN \
  --max_samples $MAXSAMPLES
RC=$?
# Score with the paper's processing verbatim (reasoning_mode), but via our thin
# wrapper that reads the UNWEIGHTED best-F ('f') so no IA.txt is required. If IA_FILE
# is set + exists, the wrapper passes it through for the weighted number too.
echo "=== generation rc=$RC; scoring (paper processing, reasoning_mode) ==="
if [ $RC -eq 0 ]; then
  IA_ARG=""
  [ -n "$IA_FILE" ] && [ -f "$IA_FILE" ] && IA_ARG="--ia_file $IA_FILE"
  python experiments/bioreason/score_fmax_unweighted.py \
    --input_dir $OUT \
    --ontology $BIOREASON_SRC/bioreason2/dataset/go-basic.obo \
    --reasoning_mode True $IA_ARG 2>&1 | tail -40
fi
echo "=== eval end rc=$RC $(date) ==="
exit $RC

#!/bin/bash
# Full BioReason F_max eval, sharded across all 12 tiles of one node (~12x speedup).
# Each tile runs eval_cafa_fmax.py over a strided protein subset, all writing to the
# SAME --out dir (unique {pid}_{ASPECT}_k00.json names). After all shards finish, one
# scoring pass over the union. ~9197 proteins / 12 ≈ 766 proteins/tile.
#
# Usage (on a hold node, run via nohup):
#   CKPT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft TAG=sft \
#     bash experiments/bioreason/run_eval_fmax_sharded.sh
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
cd $TT

CKPT=${CKPT:-/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft}
TAG=${TAG:-sft}
CACHE=${CACHE:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt}
PARQUET=${PARQUET:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl}
NSHARDS=${NSHARDS:-12}
MAXSAMPLES=${MAXSAMPLES:--1}
MAXGEN=${MAXGEN:-2048}
OUT=$TT/experiments/bioreason/eval_out/$TAG
LOGDIR=$TT/experiments/bioreason/eval_logs/$TAG
mkdir -p $OUT $LOGDIR

echo "=== sharded eval start $(date) ckpt=$CKPT tag=$TAG nshards=$NSHARDS ==="
PIDS=()
for SID in $(seq 0 $((NSHARDS-1))); do
  ZE_AFFINITY_MASK=$SID nohup python experiments/bioreason/eval_cafa_fmax.py \
    --ckpt_dir $CKPT \
    --local_parquet $PARQUET \
    --esm3_cache_path $CACHE \
    --out $OUT \
    --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens $MAXGEN \
    --max_samples $MAXSAMPLES \
    --num_shards $NSHARDS --shard_id $SID \
    > $LOGDIR/shard_${SID}.log 2>&1 &
  PIDS+=($!)
  sleep 3   # stagger vLLM inits to avoid simultaneous SPIR-V/cache contention
done
echo "launched ${#PIDS[@]} shards: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
echo "=== all shards done (fail=$FAIL) $(date); JSONs: $(ls $OUT/*.json 2>/dev/null | wc -l) ==="

# Single scoring pass over the union.
IA_ARG=""
[ -n "$IA_FILE" ] && [ -f "$IA_FILE" ] && IA_ARG="--ia_file $IA_FILE"
python experiments/bioreason/score_fmax_unweighted.py \
  --input_dir $OUT \
  --ontology $BIOREASON_SRC/bioreason2/dataset/go-basic.obo \
  --reasoning_mode True $IA_ARG 2>&1 | tail -40
echo "=== sharded eval end $(date) ==="

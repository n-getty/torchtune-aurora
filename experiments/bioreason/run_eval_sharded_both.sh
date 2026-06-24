#!/bin/bash
# Bigger-N baseline: 12-tile sharded eval for BOTH ckpts (SFT + RL) at faithful inputs,
# same N proteins (same --seed), then score each. Settles RL-vs-SFT beyond N=30 noise.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
cd $TT

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
RL=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-rl
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt
PARQUET=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
N=${N:-600}
NSHARDS=${NSHARDS:-12}

run_ckpt () {
  local tag=$1 ckpt=$2
  local out=$TT/experiments/bioreason/eval_out/big_${tag}
  local logdir=$TT/experiments/bioreason/eval_logs/big_${tag}
  rm -rf "$out"; mkdir -p "$out" "$logdir"
  echo "===== CKPT=$tag N=$N nshards=$NSHARDS $(date) ====="
  local pids=()
  for SID in $(seq 0 $((NSHARDS-1))); do
    # Unique MASTER_PORT + VLLM_PORT per shard: co-located vLLM engines each open a
    # TCP rendezvous store; without distinct ports they collide (EADDRINUSE on the
    # shared node). Space by 8 to avoid vLLM's internal port+offset reuse.
    ZE_AFFINITY_MASK=$SID \
    MASTER_PORT=$((29600 + SID*8)) VLLM_PORT=$((34000 + SID*8)) \
    nohup python experiments/bioreason/eval_cafa_fmax.py \
      --ckpt_dir $ckpt --local_parquet $PARQUET --esm3_cache_path $CACHE \
      --out $out --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
      --max_samples $N --num_shards $NSHARDS --shard_id $SID \
      > $logdir/shard_${SID}.log 2>&1 &
    pids+=($!); sleep 3
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "----- SCORE $tag ($(ls $out/*.json 2>/dev/null | grep -v _cafa_pred | wc -l) jsons) -----"
  python experiments/bioreason/score_fmax_unweighted.py \
    --input_dir $out --ontology $OBO --reasoning_mode True 2>&1 | \
    grep -E "biological_process|molecular_function|cellular_component|OVERALL|proteins_with"
}

echo "=== big-N eval start $(date) ==="
run_ckpt sft $SFT
run_ckpt rl  $RL
echo "=== big-N eval end $(date) ==="

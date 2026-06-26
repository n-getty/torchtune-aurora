#!/bin/bash
# go_pred uplift eval: SFT base vs OUR trained LoRA adapter, on the HELD-OUT TEST SET,
# with --inject_go_pred + IA.txt — the SAME harness as the published-RL-vs-SFT verdict
# (job 8564498: SFT 0.6686, published RL 0.6866). So our adapter's number is directly
# comparable to those anchors. TARGET: our_rl > sft_base by ~+0.018 (match published RL).
#
# Differs from run_eval_adapter_vs_sft.sh (which used the RL TRAIN set + cold prompt):
#   - held-out bioreason_pro_test (+ its test ESM3 cache, keyed over test seqs)
#   - --inject_go_pred (the train/SFT/eval-matched prompt)
#
# Usage (1-node PBS batch or held node):
#   ADAPTER_EPOCH=/lus/.../outputs/<run>/epoch_0 N=280 bash run_eval_adapter_gopred_testset.sh
#
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug
#PBS -N br_eval_gopred_uplift
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_eval_adapter_gopred_testset.out
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 RUST_BACKTRACE=0
cd $TT

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
# HELD-OUT test set + its OWN ESM3 cache (keyed over TEST seqs; the train cache KeyErrors).
TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt
IA=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
N=${N:-280}
NSHARDS=${NSHARDS:-12}
ADAPTER_EPOCH=${ADAPTER_EPOCH:?set ADAPTER_EPOCH=<outputs/.../epoch_N dir>}
ADAPTER_DIR=${ADAPTER_DIR:-${ADAPTER_EPOCH}/adapter}
[ -f "${ADAPTER_DIR}/adapter_model.safetensors" ] || { echo "ERROR: no adapter at ${ADAPTER_DIR}"; exit 2; }
EVAL_TAG=${EVAL_TAG:-$(basename "$(dirname "${ADAPTER_EPOCH}")")_$(basename "${ADAPTER_EPOCH}")}
TRAJ=$TT/experiments/bioreason/eval_out/gopred_uplift_trajectory.tsv

run_ckpt () {
  local tag=$1; shift; local extra="$*"
  local out=$TT/experiments/bioreason/eval_out/gopu_${EVAL_TAG}_${tag}
  local logdir=$TT/experiments/bioreason/eval_logs/gopu_${EVAL_TAG}_${tag}
  rm -rf "$out"; mkdir -p "$out" "$logdir"
  echo "===== CKPT=$tag N=$N extra='${extra}' $(date) ====="
  local pids=()
  for SID in $(seq 0 $((NSHARDS-1))); do
    ZE_AFFINITY_MASK=$SID MASTER_PORT=$((29600+SID*8)) VLLM_PORT=$((34000+SID*8)) \
    nohup python experiments/bioreason/eval_cafa_fmax.py \
      --ckpt_dir $SFT --local_parquet $TEST --esm3_cache_path $CACHE \
      --inject_go_pred \
      --out $out --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
      --max_samples $N --num_shards $NSHARDS --shard_id $SID $extra \
      > $logdir/shard_${SID}.log 2>&1 &
    pids+=($!); sleep 8
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "----- SCORE ${EVAL_TAG}/$tag ($(ls $out/*.json 2>/dev/null | grep -v _cafa_pred | wc -l) jsons) -----"
  python experiments/bioreason/score_fmax_unweighted.py \
    --input_dir $out --ontology $OBO --ia_file $IA --reasoning_mode True --final_answer_only False 2>&1 | \
    grep -E "biological_process|molecular_function|cellular_component|OVERALL|proteins_with" | \
    tee >(grep "OVERALL" | sed "s|^|$(date +%H:%M:%S)\t${EVAL_TAG}\t${tag}\tN=${N}\t|" >> "$TRAJ")
}

echo "=== go_pred uplift eval start $(date) ADAPTER_EPOCH=${ADAPTER_EPOCH} EVAL_TAG=${EVAL_TAG} ==="
run_ckpt sft_base
run_ckpt our_rl --adapter_path "${ADAPTER_DIR}" --proj_dir "${ADAPTER_EPOCH}"
echo "=== go_pred uplift eval end $(date) ==="
echo "Compare OVERALL F_max sft_base vs our_rl (anchors: SFT 0.669, published RL 0.687). Trajectory: $TRAJ"

#!/bin/bash
# One-shot PBS batch: REAL RL uplift — eval the prod RL adapter vs SFT base on the held-out
# test set, BOTH with the go_pred prompt fix + test ESM3 cache (the validated path). Same
# harness/seed/N so the delta is internally controlled. SFT baseline = ~0.65 weighted
# (job 8557398). This measures whether our GRPO actually beats SFT on the correct footing.
#
# ADAPTER_EPOCH defaults to the prod run's latest checkpoint.
#
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug
#PBS -N br_rl_uplift
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_eval_rl_uplift.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
cd $TT
echo "=== RL uplift eval start $(date) job=${PBS_JOBID} ==="

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
IA=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt
N=${N:-240}
NSHARDS=${NSHARDS:-12}
ADAPTER_EPOCH=${ADAPTER_EPOCH:-$TT/outputs/bioreason_prod_rl_4n_20260623/epoch_0}
ADAPTER_DIR=${ADAPTER_DIR:-${ADAPTER_EPOCH}/adapter}
TRAJ=$TT/experiments/bioreason/eval_out/fmax_trajectory_fixed.tsv

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
  echo "ERROR: ${ADAPTER_DIR}/adapter_model.safetensors not found"; exit 2
fi

# $1=tag  $2..=extra eval args (adapter overlay for our_rl; none for sft_base)
run_leg () {
  local tag=$1; shift; local extra="$*"
  local OUT=$TT/experiments/bioreason/eval_out/uplift_fixed_${tag}
  local LOGD=$TT/experiments/bioreason/eval_logs/uplift_fixed_${tag}
  rm -rf "$OUT" "$LOGD"; mkdir -p "$OUT" "$LOGD"
  echo "===== leg=$tag N=$N extra='${extra}' $(date) ====="
  local pids=()
  for SID in $(seq 0 $((NSHARDS-1))); do
    ZE_AFFINITY_MASK=$SID MASTER_PORT=$((29600+SID*8)) VLLM_PORT=$((34000+SID*8)) \
    nohup python experiments/bioreason/eval_cafa_fmax.py \
      --ckpt_dir $SFT --local_parquet $TEST --esm3_cache_path $CACHE --inject_go_pred \
      --out $OUT --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
      --max_samples $N --num_shards $NSHARDS --shard_id $SID $extra \
      > $LOGD/shard_${SID}.log 2>&1 &
    pids+=($!); sleep 8
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "----- SCORE $tag ($(ls $OUT/*.json 2>/dev/null | grep -v _cafa_pred | wc -l) jsons) -----"
  cd $BIOREASON_SRC/evals
  python cafa_evals.py --input_dir $OUT --ontology $OBO --ia_file $IA \
    --output_dir /tmp/uplift_${tag}_out --reasoning_mode True --final_answer_only False --threads 0 2>&1 | \
    grep -iE "Overall mean|OVERALL AVERAGE" | tee >(sed "s|^|$(date +%H:%M:%S)\t${tag}\tN=${N}\t|" >> "$TRAJ")
  cd $TT
}

echo "ADAPTER_EPOCH=${ADAPTER_EPOCH}"
run_leg sft_base
run_leg our_rl --adapter_path "${ADAPTER_DIR}" --proj_dir "${ADAPTER_EPOCH}"
echo "=== RL uplift eval end $(date) ==="
echo "UPLIFT = our_rl - sft_base (both go_pred-fixed, test set, same N). SFT anchor ~0.65 weighted."

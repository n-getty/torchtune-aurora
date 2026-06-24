#!/bin/bash
# OUR-RL-uplift eval: 12-tile sharded F_max for the SFT base vs OUR trained LoRA
# adapter (+ trained projections), same N proteins / same seed, then score both.
# This is the measurement that answers "did our GRPO improve F_max over SFT 0.414?".
#
# Unlike run_eval_sharded_both.sh (which compares SFT vs the PUBLISHED RL ckpt), this
# loads our adapter via the new eval-driver --adapter_path/--proj_dir flags: frozen
# backbone from $SFT + adapter_model.safetensors + protein/go_projection.pt from our
# saved epoch dir.
#
# Usage (on a single held node / 1-node PBS batch):
#   ADAPTER_EPOCH=/lus/.../outputs/<run>/epoch_0 N=600 bash run_eval_adapter_vs_sft.sh
#   (ADAPTER_EPOCH must contain adapter/adapter_model.safetensors + *_projection.pt)
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
# Thread-pool caps: 12 sharded vLLM+tokenizer+ESM3 processes on ONE node exhaust the
# per-user thread/process limit -> the HF fast-tokenizer's rayon pool fails to spawn
# ("Resource temporarily unavailable", errno 11 EAGAIN -> PanicException, ~3/12 shards
# crashed on 2026-06-23 job 8556949). Cap every per-process thread pool to 1.
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export RUST_BACKTRACE=0
cd $TT

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt
PARQUET=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
N=${N:-600}
NSHARDS=${NSHARDS:-12}
ADAPTER_EPOCH=${ADAPTER_EPOCH:?set ADAPTER_EPOCH=<outputs/.../epoch_N dir>}
ADAPTER_DIR=${ADAPTER_DIR:-${ADAPTER_EPOCH}/adapter}

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
  echo "ERROR: ${ADAPTER_DIR}/adapter_model.safetensors not found"; exit 2
fi

# EVAL_TAG: per-checkpoint label so sequential checkpoint evals do NOT clobber each
# other's prediction JSONs / scores (the F_max-vs-steps trajectory must be preserved).
# Defaults to the adapter epoch dir name (e.g. epoch_0). Override via EVAL_TAG env.
EVAL_TAG=${EVAL_TAG:-$(basename "${ADAPTER_EPOCH}")}

# $1=tag $2=extra eval args (adapter overlay or empty for base SFT)
run_ckpt () {
  local tag=$1; shift
  local extra="$*"
  local out=$TT/experiments/bioreason/eval_out/uplift_${EVAL_TAG}_${tag}
  local logdir=$TT/experiments/bioreason/eval_logs/uplift_${EVAL_TAG}_${tag}
  rm -rf "$out"; mkdir -p "$out" "$logdir"
  echo "===== CKPT=$tag N=$N nshards=$NSHARDS extra='${extra}' $(date) ====="
  local pids=()
  for SID in $(seq 0 $((NSHARDS-1))); do
    ZE_AFFINITY_MASK=$SID \
    MASTER_PORT=$((29600 + SID*8)) VLLM_PORT=$((34000 + SID*8)) \
    nohup python experiments/bioreason/eval_cafa_fmax.py \
      --ckpt_dir $SFT --local_parquet $PARQUET --esm3_cache_path $CACHE \
      --out $out --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
      --max_samples $N --num_shards $NSHARDS --shard_id $SID $extra \
      > $logdir/shard_${SID}.log 2>&1 &
    pids+=($!); sleep 8   # stagger init to avoid 12-way thread-pool spawn contention
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "----- SCORE ${EVAL_TAG}/$tag ($(ls $out/*.json 2>/dev/null | grep -v _cafa_pred | wc -l) jsons) -----"
  # Tee the score into a CUMULATIVE trajectory file (survives across checkpoint evals;
  # the per-job PBS .out is overwritten each run). One row per (eval_tag, ckpt).
  local _traj=$TT/experiments/bioreason/eval_out/fmax_trajectory.tsv
  python experiments/bioreason/score_fmax_unweighted.py \
    --input_dir $out --ontology $OBO --reasoning_mode True 2>&1 | \
    grep -E "biological_process|molecular_function|cellular_component|OVERALL|proteins_with" | \
    tee >(grep "OVERALL" | sed "s|^|$(date +%H:%M:%S)\t${EVAL_TAG}\t${tag}\tN=${N}\t|" >> "$_traj")
}

echo "=== uplift eval start $(date) ADAPTER_EPOCH=${ADAPTER_EPOCH} EVAL_TAG=${EVAL_TAG} ==="
# Baseline: SFT alone (same harness, same seed/N) — re-measure so the comparison is
# on identical proteins (don't rely on a stale 0.414 from a different N).
run_ckpt sft_base
# Our trained adapter + projections overlaid on the SFT base.
run_ckpt our_rl --adapter_path "${ADAPTER_DIR}" --proj_dir "${ADAPTER_EPOCH}"
echo "=== uplift eval end $(date) ==="
echo "Compare OVERALL MEAN F_max: sft_base vs our_rl above (uplift if our_rl > sft_base)."

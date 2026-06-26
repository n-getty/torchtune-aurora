#!/bin/bash
# One-shot PBS batch: PUBLISHED RL vs PUBLISHED SFT, BOTH with the go_pred prompt fix +
# weighted F_max (IA.txt), on the held-out bioreason_pro_test split, SAME node, SAME N.
#
# THE DECISIVE TEST: does the OFFICIAL bioreason-pro-rl ckpt actually beat its own SFT base
# on F_max in our (now-fixed) harness?
#   - If YES -> F_max is the right axis; our flat RL result points at the training loss
#     (KL explosion) -> fix kl/skip-zero-adv and rerun.
#   - If NO  -> even the paper's RL barely moves F_max; the RL gain lives on a different axis
#     (LLM-judge prose). Stop chasing F_max uplift; change the metric.
# Prior numbers for reference: cold-prompt SFT=0.41, GO-GPT floor=0.54, go_pred-fixed SFT~0.65u/0.70w,
# paper full=0.736 (weighted, temporal split).
#
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_eval_pub_rl_vs_sft
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_eval_published_rl_vs_sft.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
cd $TT
echo "=== published RL-vs-SFT go_pred-fix eval start $(date) job=${PBS_JOBID} ==="
echo "node: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
RL=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-rl
TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
IA=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt
N=${N:-280}
NSHARDS=${NSHARDS:-12}
TRAJ=$TT/experiments/bioreason/eval_out/pub_rl_vs_sft_trajectory.tsv
> "$TRAJ"

run_leg() {
  local tag="$1"; shift
  local OUT=$TT/experiments/bioreason/eval_out/pub_${tag}
  local LOGD=$TT/experiments/bioreason/eval_logs/pub_${tag}
  rm -rf "$OUT" "$LOGD"; mkdir -p "$OUT" "$LOGD"
  echo "===== leg=$tag N=$N args='$*' $(date) ====="
  local pids=()
  for SID in $(seq 0 $((NSHARDS-1))); do
    ZE_AFFINITY_MASK=$SID MASTER_PORT=$((29600+SID*8)) VLLM_PORT=$((34000+SID*8)) \
    nohup python experiments/bioreason/eval_cafa_fmax.py \
      "$@" --local_parquet $TEST --esm3_cache_path $CACHE \
      --inject_go_pred \
      --out $OUT --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
      --max_samples $N --num_shards $NSHARDS --shard_id $SID \
      > $LOGD/shard_${SID}.log 2>&1 &
    pids+=($!); sleep 8
  done
  for p in "${pids[@]}"; do wait $p; done
  local njson=$(ls $OUT/*.json 2>/dev/null | grep -v _cafa_pred | wc -l)
  echo "----- SCORE $tag ($njson jsons) weighted+unweighted -----"
  python experiments/bioreason/score_fmax_unweighted.py --input_dir $OUT \
    --ontology $OBO --ia_file $IA --reasoning_mode True --final_answer_only False 2>&1 | \
    tee >(grep -iE "OVERALL|namespace|: 0\.|weighted" | sed "s|^|$(date +%H:%M:%S)\t${tag}\tN=${N}\t|" >> "$TRAJ") | \
    grep -iE "F_max|OVERALL|biological_process|molecular_function|cellular_component|weighted"
}

# SFT base (sanity re-anchor; should match the prior gopredfix ~0.65u/0.70w)
run_leg sft_base --ckpt_dir $SFT
# Published RL (the question) — full model, ships its own projections in $RL
run_leg rl_published --ckpt_dir $RL

echo "=== eval end $(date) ==="
echo "VERDICT: compare pub_rl_published vs pub_sft_base in $TRAJ (weighted = paper metric)"
cat "$TRAJ"

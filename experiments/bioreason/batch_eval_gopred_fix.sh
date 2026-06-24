#!/bin/bash
# One-shot PBS batch: re-eval the PUBLISHED SFT checkpoint with the go_pred PROMPT FIX.
#
# THE TEST: our eval was feeding a cold prompt MISSING the GO-GPT predictions (go_pred) the
# model was trained to refine -> scored 0.41, BELOW GO-GPT's own 0.54. eval_cafa_fmax.py now
# injects go_pred via the paper's _format_reasoning_prompt (--inject_go_pred default ON).
# If the fix is right, the PUBLISHED SFT ckpt should jump from 0.41 toward ~0.7 (its real
# number). This validates the whole pipeline before trusting any RL uplift.
#
# Runs on the held-out bioreason_pro_test set (8630 proteins, public). 12-tile sharded,
# thread-pool capped. N small for fast turnaround; bump for the final number.
#
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_eval_gopred
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_eval_gopred_fix.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
# thread-pool caps (12 sharded vLLM+tokenizer procs on one node) — see run_eval_adapter_vs_sft
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
cd $TT
echo "=== go_pred-fix eval start $(date) job=${PBS_JOBID} ==="
echo "node: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
# TEST-set ESM3 cache (built by batch_precompute_test_esm3.sh, 8358 seqs @ 2048). Using
# the cache frees ~5.5 GiB/tile vs live ESM3 -> all 12 shards survive (live ESM3 OOM'd
# 6/12 with UR:40). The train-only esm3_cache_2048 would KeyError on test proteins; THIS
# one is keyed over the test sequences.
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
IA=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt
N=${N:-240}
NSHARDS=${NSHARDS:-12}
OUT=$TT/experiments/bioreason/eval_out/gopredfix_sft
LOGD=$TT/experiments/bioreason/eval_logs/gopredfix_sft
rm -rf "$OUT" "$LOGD"; mkdir -p "$OUT" "$LOGD"

pids=()
for SID in $(seq 0 $((NSHARDS-1))); do
  ZE_AFFINITY_MASK=$SID MASTER_PORT=$((29600+SID*8)) VLLM_PORT=$((34000+SID*8)) \
  nohup python experiments/bioreason/eval_cafa_fmax.py \
    --ckpt_dir $SFT --local_parquet $TEST --esm3_cache_path $CACHE \
    --inject_go_pred \
    --out $OUT --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
    --max_samples $N --num_shards $NSHARDS --shard_id $SID \
    > $LOGD/shard_${SID}.log 2>&1 &
  pids+=($!); sleep 8
done
for p in "${pids[@]}"; do wait $p; done

echo "----- SCORE gopredfix_sft ($(ls $OUT/*.json 2>/dev/null | grep -v _cafa_pred | wc -l) jsons) -----"
cd $BIOREASON_SRC/evals
python cafa_evals.py --input_dir $OUT --ontology $OBO --ia_file $IA \
  --output_dir /tmp/gopredfix_out --reasoning_mode True --final_answer_only False --threads 0 2>&1 | \
  grep -iE "Overall mean|OVERALL AVERAGE|biological_process|molecular_function|cellular_component"
echo "=== go_pred-fix eval end $(date) ==="
echo "COMPARE: old cold-prompt SFT=0.41 ; GO-GPT floor=0.54 ; paper full=0.736"

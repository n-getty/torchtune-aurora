#!/bin/bash
# Input-fidelity A/B: SAME SFT model, SAME proteins, two input configs.
#   STARVED  : GO 50  + protein 128  (esm3_cache.pt)       <- our old bring-up defaults
#   FAITHFUL : GO 200 + protein 2048 (esm3_cache_2048.pt)  <- matches the SFT ckpt
# Isolates whether input starvation (not the RL loop) capped reward. One tile, greedy.
#
# Usage:  N=22 bash experiments/bioreason/run_fidelity_ab.sh
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
cd $TT

CKPT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
PARQUET=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
N=${N:-22}
# Note: --seed 23 + same parquet => identical protein selection across both legs
# (the cache/GO/protein-len differ, the protein ROWS do not).

run_leg () {
  local tag=$1 cache=$2 mpl=$3 ngo=$4
  local out=$TT/experiments/bioreason/eval_out/ab_${tag}
  echo "===== LEG=$tag  cache=$(basename $cache) protein_len=$mpl go=$ngo  N=$N ====="
  python experiments/bioreason/eval_cafa_fmax.py \
    --ckpt_dir $CKPT --local_parquet $PARQUET --esm3_cache_path $cache \
    --out $out --max_protein_len $mpl --num_go_tokens $ngo \
    --max_new_tokens 2048 --max_samples $N
  echo "----- SCORE $tag -----"
  python experiments/bioreason/score_fmax_unweighted.py \
    --input_dir $out --ontology $OBO --reasoning_mode True 2>&1 | \
    grep -E "F_max|biological_process|molecular_function|cellular_component|OVERALL|proteins_with"
}

echo "=== fidelity A/B start $(date) ==="
run_leg starved  $PARQUET/esm3_cache.pt      128  50
run_leg faithful $PARQUET/esm3_cache_2048.pt 2048 200
echo "=== fidelity A/B end $(date) ==="

#!/bin/bash
# Two diagnostics back-to-back on one tile (after the fidelity A/B disproved input-starvation):
#   1) TEXT-ABLATION (SFT): faithful embeddings (GO200/prot2048) but STRIP InterPro/PPI/function
#      text from the prompt. If F_max collapses vs the 0.42 faithful baseline => model is
#      text-driven, the multimodal embeddings are near-dead weight.
#   2) RL BASELINE: published bioreason-pro-rl ckpt at full faithful inputs. If RL ~= SFT ~= 0.42,
#      GRPO isn't moving eval F_max either.
# Same N + seed => identical protein set across all legs. Greedy.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
cd $TT

SFT=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
RL=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-rl
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt
PARQUET=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl
OBO=$BIOREASON_SRC/bioreason2/dataset/go-basic.obo
N=${N:-30}

run () {
  local tag=$1 ckpt=$2; shift 2
  local out=$TT/experiments/bioreason/eval_out/diag_${tag}
  echo "===== LEG=$tag ckpt=$(basename $ckpt) extra='$*' N=$N ====="
  python experiments/bioreason/eval_cafa_fmax.py \
    --ckpt_dir $ckpt --local_parquet $PARQUET --esm3_cache_path $CACHE \
    --out $out --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048 \
    --max_samples $N "$@"
  echo "----- SCORE $tag -----"
  python experiments/bioreason/score_fmax_unweighted.py \
    --input_dir $out --ontology $OBO --reasoning_mode True 2>&1 | \
    grep -E "F_max|biological_process|molecular_function|cellular_component|OVERALL|proteins_with"
}

echo "=== diag start $(date) ==="
# 0) MANDATORY generation-health gate — fail fast, before spending real eval compute,
# if either checkpoint has degenerated (see probe_generation_health.py's docstring for
# why: two ablations in this same investigation silently broke generation and were only
# caught after the fact via the F_max empty-response rate).
echo "----- PROBE (generation health) sft_textablate -----"
python experiments/bioreason/probe_generation_health.py \
    --ckpt_dir $SFT --local_parquet $PARQUET --esm3_cache_path $CACHE \
    --max_protein_len 2048 --num_go_tokens 200 --num_proteins 3 \
    --no-interpro_in_prompt --no-ppi_in_prompt --no-include_protein_function_summary \
    --out $TT/experiments/bioreason/eval_out/_probe_diag_sft_textablate
probe_sft_rc=$?
echo "----- PROBE (generation health) rl_faithful -----"
python experiments/bioreason/probe_generation_health.py \
    --ckpt_dir $RL --local_parquet $PARQUET --esm3_cache_path $CACHE \
    --max_protein_len 2048 --num_go_tokens 200 --num_proteins 3 \
    --out $TT/experiments/bioreason/eval_out/_probe_diag_rl_faithful
probe_rl_rc=$?
if [ $probe_sft_rc -ne 0 ] || [ $probe_rl_rc -ne 0 ]; then
    echo "=== ABORTING: generation-health probe failed (sft_rc=$probe_sft_rc rl_rc=$probe_rl_rc) — do NOT trust an F_max eval on a collapsed checkpoint. ==="
    exit 1
fi
# 1) text-ablation: faithful embeddings, NO interpro/ppi/function text
run sft_textablate $SFT --no-interpro_in_prompt --no-ppi_in_prompt --no-include_protein_function_summary
# 2) RL baseline: full faithful (text ON, as the paper evaluates)
run rl_faithful $RL
echo "=== diag end $(date) ==="

#!/bin/bash -l
# F3: Qwen3-4B TP=2 NO LoRA on torch211.
# Pinpoints whether the T4 failure is LoRA-specific (vllm-xpu-kernels punica)
# or a TP=2 bug in the new stack itself.

TS=$(date +%Y%m%d_%H%M%S)
ROOT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey
TEST_NAME=F3_tp2_dense
OUT_DIR=${ROOT}/${TEST_NAME}_${TS}
mkdir -p "${OUT_DIR}"
LOG=${OUT_DIR}/test.log
VLLM_LOG=${OUT_DIR}/vllm.log
exec > "${LOG}" 2>&1

source ${ROOT}/_common.sh

PORT=9932
MODEL=${MODEL_4B}
PROMPTS_VAR=PROMPTS_7

clean_orphans
setup_env
export VLLM_XPU_ENABLE_XPU_GRAPH=1

# TP=2, NO LoRA, recommended defaults
FLAGS="--tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 1536 --max-num-seqs 64"
if ! start_vllm "0,1" ${FLAGS}; then
    echo "[${TEST_NAME}] FAIL: vllm start"
    echo "${TEST_NAME}: FAIL_START"
    exit 1
fi

run_curls 7 24 512 main
finish_vllm

BC=${F3_tp2_dense_BANNED:-?}
OK=${main_OK:-?}
WALL=${main_WALL:-?}
TOK=${main_OUT_TOK:-?}
TPS=$(awk -v t=${WALL} -v o=${TOK} 'BEGIN{ if (t>0) printf "%.2f", o/t; else print "0"}')

echo ""
echo "=================================================================="
echo "=== F3 SUMMARY ==="
echo "=================================================================="
if [ "${OK}" = "7" ] && [ "${BC}" = "0" ]; then
    echo "${TEST_NAME}: PASS  ok=${OK}/7  banned1=${BC}  wall=${WALL}s  out_tok=${TOK}  tok_per_s=${TPS}"
    echo "  → CONFIRMS: T4 failure is LoRA-specific (vllm-xpu-kernels punica TP slicing bug, not TP itself)"
else
    echo "${TEST_NAME}: FAIL  ok=${OK}/7  banned1=${BC}  wall=${WALL}s"
    echo "  → TP=2 itself is broken on torch211 stack (NOT just LoRA)"
fi

#!/bin/bash -l
# F4: long-context stress on torch211 stack.
# max_model_len=4096, max_gen=1024 — closer to LoRA-GRPO production envelope.
# Validates KV cache + paged attention at depth.

TS=$(date +%Y%m%d_%H%M%S)
ROOT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey
TEST_NAME=F4_long_ctx
OUT_DIR=${ROOT}/${TEST_NAME}_${TS}
mkdir -p "${OUT_DIR}"
LOG=${OUT_DIR}/test.log
VLLM_LOG=${OUT_DIR}/vllm.log
exec > "${LOG}" 2>&1

source ${ROOT}/_common.sh

PORT=9933
MODEL=${MODEL_4B}
PROMPTS_VAR=PROMPTS_7

clean_orphans
setup_env
export VLLM_XPU_ENABLE_XPU_GRAPH=1

# Long context + LoRA + recommended defaults
FLAGS="--tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 32 --enable-lora --max-lora-rank 16 --max-loras 2"
if ! start_vllm "0" ${FLAGS}; then
    echo "[${TEST_NAME}] FAIL: vllm start"
    echo "${TEST_NAME}: FAIL_START"
    exit 1
fi

# 7 prompts × n=8 × max_tokens=1024 (smaller batch fan-out so each gen is longer)
run_curls 7 8 1024 main
finish_vllm

BC=${F4_long_ctx_BANNED:-?}
OK=${main_OK:-?}
WALL=${main_WALL:-?}
TOK=${main_OUT_TOK:-?}
TPS=$(awk -v t=${WALL} -v o=${TOK} 'BEGIN{ if (t>0) printf "%.2f", o/t; else print "0"}')

echo ""
echo "=================================================================="
echo "=== F4 SUMMARY (long-context stress) ==="
echo "=================================================================="
if [ "${OK}" = "7" ] && [ "${BC}" = "0" ]; then
    echo "${TEST_NAME}: PASS  ok=${OK}/7  banned1=${BC}  wall=${WALL}s  out_tok=${TOK}  tok_per_s=${TPS}"
else
    echo "${TEST_NAME}: FAIL  ok=${OK}/7  banned1=${BC}  wall=${WALL}s"
fi

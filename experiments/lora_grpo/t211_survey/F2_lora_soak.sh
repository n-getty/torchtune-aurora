#!/bin/bash -l
# F2: 5-min continuous LoRA traffic on torch211 stack.
# Looks for late banned:1, KV fragmentation, or perf drift across rounds.

TS=$(date +%Y%m%d_%H%M%S)
ROOT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey
TEST_NAME=F2_lora_soak
OUT_DIR=${ROOT}/${TEST_NAME}_${TS}
mkdir -p "${OUT_DIR}"
LOG=${OUT_DIR}/test.log
VLLM_LOG=${OUT_DIR}/vllm.log
exec > "${LOG}" 2>&1

source ${ROOT}/_common.sh

PORT=9931
MODEL=${MODEL_4B}
PROMPTS_VAR=PROMPTS_7

clean_orphans
setup_env
export VLLM_XPU_ENABLE_XPU_GRAPH=1

# Recommended defaults: NO eager, async ON, LoRA on
FLAGS="--tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 1536 --max-num-seqs 64 --enable-lora --max-lora-rank 16 --max-loras 2"
if ! start_vllm "0" ${FLAGS}; then
    echo "[${TEST_NAME}] FAIL: vllm start"
    echo "${TEST_NAME}: FAIL_START"
    exit 1
fi

# 5-min loop: ~10 rounds × 7 curls × n=24 × max_tokens=256
# (256 not 512 to keep round duration ~25-30s and get more rounds in 5 min)
SOAK_END=$(( $(date +%s) + 300 ))
ROUND=0
TOTAL_OK=0
TOTAL_FAIL=0
TOTAL_OUT_TOK=0
TOTAL_WALL=0
declare -a ROUND_TPS

while [ $(date +%s) -lt ${SOAK_END} ]; do
    ROUND=$((ROUND + 1))
    TAG="rnd${ROUND}"
    echo ""
    echo "=== ROUND ${ROUND} t=$(($(date +%s) - SOAK_END + 300))s ==="
    run_curls 7 24 256 ${TAG}
    OK_VAR="${TAG}_OK"; WALL_VAR="${TAG}_WALL"; TOK_VAR="${TAG}_OUT_TOK"
    OK=${!OK_VAR}; WALL=${!WALL_VAR}; TOK=${!TOK_VAR}
    TOTAL_OK=$((TOTAL_OK + OK))
    TOTAL_FAIL=$((TOTAL_FAIL + (7 - OK)))
    TOTAL_OUT_TOK=$((TOTAL_OUT_TOK + TOK))
    TOTAL_WALL=$((TOTAL_WALL + WALL))
    R_TPS=$(awk -v t=${WALL} -v o=${TOK} 'BEGIN{ if (t>0) printf "%.2f", o/t; else print "0"}')
    ROUND_TPS+=("${R_TPS}")
    BC=$(grep -c "banned:1" "${VLLM_LOG}" 2>/dev/null || echo 0)
    echo "[ROUND ${ROUND}] tps=${R_TPS} ok=${OK}/7 banned1_running=${BC}"
    # Bail if banned:1 fires
    if [ "${BC}" -gt 0 ]; then
        echo "[${TEST_NAME}] BANNED:1 detected at round ${ROUND}, breaking soak"
        break
    fi
done

finish_vllm

BC_FINAL=${F2_lora_soak_BANNED:-?}
AVG_TPS=$(awk -v t=${TOTAL_WALL} -v o=${TOTAL_OUT_TOK} 'BEGIN{ if (t>0) printf "%.2f", o/t; else print "0"}')
FIRST_TPS=${ROUND_TPS[0]:-0}
LAST_TPS=${ROUND_TPS[-1]:-0}
DRIFT=$(awk -v f=${FIRST_TPS} -v l=${LAST_TPS} 'BEGIN{ if (f>0) printf "%.3f", l/f; else print "n/a"}')

echo ""
echo "=================================================================="
echo "=== F2 SUMMARY (5-min LoRA soak) ==="
echo "=================================================================="
echo "  rounds=${ROUND}"
echo "  per-round tps: ${ROUND_TPS[@]}"
echo "  first→last drift: ${FIRST_TPS} → ${LAST_TPS} = ${DRIFT} (1.00 = stable, <0.9 = degrading)"
echo "  total: ok=${TOTAL_OK}/$((ROUND * 7)) fail=${TOTAL_FAIL} avg_tok_per_s=${AVG_TPS} banned:1=${BC_FINAL}"
if [ "${TOTAL_FAIL}" = "0" ] && [ "${BC_FINAL}" = "0" ]; then
    echo "${TEST_NAME}: PASS"
else
    echo "${TEST_NAME}: FAIL ok=${TOTAL_OK}/$((ROUND * 7)) banned1=${BC_FINAL}"
fi

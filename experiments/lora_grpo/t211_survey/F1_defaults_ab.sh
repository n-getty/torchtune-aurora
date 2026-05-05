#!/bin/bash -l
# F1: real promotion A/B — both arms with RECOMMENDED defaults.
# Legacy: drop --enforce-eager + --no-async-scheduling.
# torch211: drop both + VLLM_XPU_ENABLE_XPU_GRAPH=1.
# T6 measured both arms with eager+sync; this measures both as actually-deployed.

TS=$(date +%Y%m%d_%H%M%S)
ROOT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey
TEST_NAME=F1_defaults_ab
OUT_DIR=${ROOT}/${TEST_NAME}_${TS}
mkdir -p "${OUT_DIR}"
SUMMARY=${OUT_DIR}/summary.log
exec > "${SUMMARY}" 2>&1

source ${ROOT}/_common.sh

PORT=9930
MODEL=${MODEL_4B}
PROMPTS_VAR=PROMPTS_7

#####################
# A) Legacy + recommended defaults
#####################
echo ""
echo "=================================================================="
echo "=== F1_A  legacy + recommended defaults (no eager, async on) ==="
echo "=================================================================="

clean_orphans
TEST_NAME=F1_A_legacy
OUT_DIR_A=${OUT_DIR}/A_legacy
mkdir -p "${OUT_DIR_A}"
LOG=${OUT_DIR_A}/test.log
VLLM_LOG=${OUT_DIR_A}/vllm.log

setup_legacy_env() {
    echo "[${TEST_NAME}] start $(date) host=$(hostname)"
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV PYTHONPATH 2>/dev/null
    module purge 2>&1 | head -2
    module load frameworks/2025.3.1 2>&1 | head -2
    unset PYTHONNOUSERSITE 2>/dev/null
    PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3
    export PATH=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin:${PATH}
    echo "[${TEST_NAME}] PY=${PY}"
    ${PY} -c "import torch; print('[${TEST_NAME}] torch:', torch.__version__)" 2>&1
    ${PY} -c "import vllm; print('[${TEST_NAME}] vllm:', vllm.__version__)" 2>&1
    ${PY} -c "
try:
    import intel_extension_for_pytorch as ipex
    print('[${TEST_NAME}] ipex:', ipex.__version__, '(legacy expected)')
except ImportError:
    print('[${TEST_NAME}] ipex NOT importable (unexpected for legacy!)')
" 2>&1
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy 2>/dev/null
    export no_proxy='*' NO_PROXY='*'
    unset VLLM_SERVER_DEV_MODE VLLM_ALLOW_RUNTIME_LORA_UPDATING 2>/dev/null
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export TORCH_COMPILE_DISABLE=1
    export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
    export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=cxi CCL_KVS_IFACE=lo
    export RAYON_NUM_THREADS=1 TOKIO_WORKER_THREADS=1
    # Recommended defaults: NO eager, async ON
    unset VLLM_XPU_ENABLE_XPU_GRAPH 2>/dev/null
}

setup_legacy_env

# NO --enforce-eager, NO --no-async-scheduling
FLAGS_A="--tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 1536 --max-num-seqs 64"
if ! start_vllm "0" ${FLAGS_A}; then
    echo "[${TEST_NAME}] FAIL: vllm start"
    A_OK=0; A_WALL=0; A_TOK=0; A_BC=?
else
    run_curls 7 24 512 main_a
    finish_vllm
    A_OK=${main_a_OK:-?}; A_WALL=${main_a_WALL:-?}; A_TOK=${main_a_OUT_TOK:-?}; A_BC=${F1_A_legacy_BANNED:-?}
fi

A_TPS=$(awk -v t=${A_WALL} -v o=${A_TOK} 'BEGIN{ if (t>0) printf "%.2f", o/t; else print "0"}')
echo "[F1_A] ok=${A_OK}/7 banned1=${A_BC} wall=${A_WALL}s out_tok=${A_TOK} tok_per_s=${A_TPS}"

#####################
# B) torch211 + recommended defaults (XPUGraph ON)
#####################
echo ""
echo "=================================================================="
echo "=== F1_B  torch 2.11 + recommended defaults (XPUGraph + async) ==="
echo "=================================================================="

clean_orphans
TEST_NAME=F1_B_t211
OUT_DIR_B=${OUT_DIR}/B_t211
mkdir -p "${OUT_DIR_B}"
LOG=${OUT_DIR_B}/test.log
VLLM_LOG=${OUT_DIR_B}/vllm.log

setup_env  # from _common.sh
export VLLM_XPU_ENABLE_XPU_GRAPH=1

# NO --enforce-eager, NO --no-async-scheduling
FLAGS_B="--tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 1536 --max-num-seqs 64"
if ! start_vllm "0" ${FLAGS_B}; then
    echo "[${TEST_NAME}] FAIL: vllm start"
    B_OK=0; B_WALL=0; B_TOK=0; B_BC=?
else
    run_curls 7 24 512 main_b
    finish_vllm
    B_OK=${main_b_OK:-?}; B_WALL=${main_b_WALL:-?}; B_TOK=${main_b_OUT_TOK:-?}; B_BC=${F1_B_t211_BANNED:-?}
fi

B_TPS=$(awk -v t=${B_WALL} -v o=${B_TOK} 'BEGIN{ if (t>0) printf "%.2f", o/t; else print "0"}')
RATIO=$(awk -v a=${A_TPS} -v b=${B_TPS} 'BEGIN{ if (a>0) printf "%.3f", b/a; else print "n/a"}')

echo ""
echo "=================================================================="
echo "=== F1 SUMMARY (defaults A/B) ==="
echo "=================================================================="
echo "  A_legacy + defaults : ok=${A_OK}/7 banned1=${A_BC} wall=${A_WALL}s out_tok=${A_TOK} tok_per_s=${A_TPS}"
echo "  B_t211   + XPUGraph : ok=${B_OK}/7 banned1=${B_BC} wall=${B_WALL}s out_tok=${B_TOK} tok_per_s=${B_TPS}"
echo "  ratio (B/A) = ${RATIO}  (≥0.85 = acceptable; >1.0 = strict win)"
PASS_A=0; PASS_B=0
[ "${A_OK}" = "7" ] && [ "${A_BC}" = "0" ] && PASS_A=1
[ "${B_OK}" = "7" ] && [ "${B_BC}" = "0" ] && PASS_B=1
if [ "${PASS_A}" = "1" ] && [ "${PASS_B}" = "1" ]; then
    echo "F1 OVERALL: PASS  (both arms healthy)"
else
    echo "F1 OVERALL: FAIL  PASS_A=${PASS_A} PASS_B=${PASS_B}"
fi

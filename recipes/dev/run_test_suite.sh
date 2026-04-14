#!/bin/bash
# Automated test suite for GRPO optimization on Aurora XPU
# Runs all tests sequentially, logs results to /tmp/torchtune/test_results/
#
# Submit: qsub recipes/dev/run_test_suite.sh
# Monitor: tail -f /tmp/torchtune/test_results/summary.log
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -o logs/test_suite.out
#PBS -e logs/test_suite.err
#PBS -N grpo_test_suite

# Do NOT use set -e — individual test failures should not abort the suite.
# The run_test function captures exit codes.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

# Load environment
module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

RESULTS=/tmp/torchtune/test_results
mkdir -p ${RESULTS}
LOG=${RESULTS}/summary.log

echo "=== GRPO Test Suite ===" | tee ${LOG}
echo "Node: $(hostname), Date: $(date)" | tee -a ${LOG}
echo "Python: $(which python3)" | tee -a ${LOG}
echo "" | tee -a ${LOG}

# --- Stage model to /tmp ---
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B
LOCAL_MODEL=/tmp/torchtune/Qwen2.5-3B
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..." | tee -a ${LOG}
    t0=$SECONDS
    cp -r "${MODEL_SRC}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s" | tee -a ${LOG}
else
    echo "Model already staged at ${LOCAL_MODEL}" | tee -a ${LOG}
fi

# Helper: run a test and capture output
run_test() {
    local name="$1"
    local cmd="$2"
    local test_log="${RESULTS}/${name}.log"

    echo "" | tee -a ${LOG}
    echo ">>> TEST: ${name}" | tee -a ${LOG}
    echo "    CMD: ${cmd}" | tee -a ${LOG}
    echo "    Start: $(date)" | tee -a ${LOG}
    t0=$SECONDS

    if eval "${cmd}" > "${test_log}" 2>&1; then
        elapsed=$((SECONDS - t0))
        echo "    PASS (${elapsed}s)" | tee -a ${LOG}
        # Extract key metrics from log
        grep -E "(Step|step_time|vLLM generation|Colocated vLLM|peak_memory|rewards)" "${test_log}" | tail -20 >> ${LOG}
    else
        elapsed=$((SECONDS - t0))
        echo "    FAIL (${elapsed}s)" | tee -a ${LOG}
        echo "    Last 20 lines:" >> ${LOG}
        tail -20 "${test_log}" >> ${LOG}
    fi
    echo "" | tee -a ${LOG}
}

# --- Test 1: Colocated vLLM, 2 tiles, Config A (5 steps) ---
# Tests if colocated mode works with no-mask approach
run_test "colocate_2tile_cfgA" \
    "timeout 180 bash recipes/dev/run_grpo_colocate_xpu.sh 2 ${LOCAL_MODEL} 5"

# --- Test 2: Colocated vLLM, 6 tiles, Config A (5 steps) ---
# If 2-tile works, scale up
run_test "colocate_6tile_cfgA" \
    "timeout 180 bash recipes/dev/run_grpo_colocate_xpu.sh 6 ${LOCAL_MODEL} 5"

# --- Test 3: Colocated vLLM, 12 tiles, Config A (5 steps) ---
# Full node colocated — the ideal config
run_test "colocate_12tile_cfgA" \
    "timeout 180 bash recipes/dev/run_grpo_colocate_xpu.sh 12 ${LOCAL_MODEL} 5"

# --- Test 4: Server mode, 1+6 tiles, grpo_samples=8 (5 steps) ---
# Higher batch size for better vLLM utilization
run_test "server_6tile_gs8" \
    "timeout 300 bash recipes/dev/run_grpo_vllm_xpu.sh 1 6 ${LOCAL_MODEL} 5 grpo_samples=8 vllm_weight_sync=true vllm_weight_sync_interval=5"

# --- Test 5: Server mode, 1+6 tiles, grpo_samples=16, max_gen=512 (Config B, 5 steps) ---
# Matching A100 baseline config — THE comparison test
# max_model_len=2048 needed for prompt+512 gen tokens
run_test "server_6tile_cfgB" \
    "VLLM_MAX_MODEL_LEN=2048 timeout 600 bash recipes/dev/run_grpo_vllm_xpu.sh 1 6 ${LOCAL_MODEL} 5 grpo_samples=16 max_generated_tokens=512 vllm_weight_sync=true vllm_weight_sync_interval=5"

# --- Test 6: Server mode, 1+4 tiles, Config A (5 steps) ---
# Map scatter_gather bug boundary
run_test "server_4tile_cfgA" \
    "timeout 180 bash recipes/dev/run_grpo_vllm_xpu.sh 1 4 ${LOCAL_MODEL} 5 vllm_weight_sync=true vllm_weight_sync_interval=5"

echo "" | tee -a ${LOG}
echo "=== All tests complete ===" | tee -a ${LOG}
echo "End: $(date)" | tee -a ${LOG}

# Copy results to persistent storage
cp -r ${RESULTS} "${TORCHTUNE_DIR}/test_results_$(date +%Y%m%d_%H%M%S)"
echo "Results saved to ${TORCHTUNE_DIR}/test_results_*"

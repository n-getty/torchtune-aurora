#!/bin/bash
#
# Test 72B GRPO optimizations: fbs=4 and Adafactor
#
# Runs three configurations sequentially on a 3-node PBS allocation:
#   1. Baseline:  fbs=1, AdamW, CPU offload (reproduce 84.6s)
#   2. fbs=4:     fbs=4, AdamW, CPU offload (expect ~57-60s)
#   3. Adafactor: fbs=4, Adafactor, CPU offload (expect ~51-54s)
#
# Usage (on a held 3-node PBS job):
#   export PBS_JOBID=<jobid>
#   export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
#   bash recipes/dev/test_72b_optimizations.sh [baseline|fbs4|adafactor|all]
#
# Default: runs all three tests sequentially.

set -e

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

TEST=${1:-all}

LAUNCH_SCRIPT="recipes/dev/aurora_grpo_72b_dedicated_vllm.sh"

# Configs
BASELINE_CONFIG="recipes/configs/dev/experimental/qwen72B_grpo_dedicated_vllm_xpu.yaml"
FBS4_CONFIG="recipes/configs/dev/experimental/qwen72B_grpo_fbs4.yaml"
ADAFACTOR_CONFIG="recipes/configs/dev/experimental/qwen72B_grpo_adafactor.yaml"

NSTEPS=3
GRPO_SAMPLES=4

run_test() {
    local name=$1
    local config=$2
    shift 2
    local extra_args=("$@")

    echo ""
    echo "============================================================"
    echo "  TEST: ${name}"
    echo "  Config: ${config}"
    echo "  Steps: ${NSTEPS}"
    echo "============================================================"
    echo ""

    CONFIG="${config}" \
    NSTEPS="${NSTEPS}" \
    GRPO_SAMPLES="${GRPO_SAMPLES}" \
    bash "${LAUNCH_SCRIPT}" "${extra_args[@]}" 2>&1 | tee "/tmp/torchtune/test_72b_${name}.log"

    echo ""
    echo "=== ${name} complete ==="
    echo "Log: /tmp/torchtune/test_72b_${name}.log"

    # Extract step times from log
    echo "Step times:"
    grep -E "step_time|Step [0-9]+ took" "/tmp/torchtune/test_72b_${name}.log" || \
    grep -E "Time for step" "/tmp/torchtune/test_72b_${name}.log" || \
    echo "  (parse step times from log manually)"
    echo ""
}

if [[ "${TEST}" == "baseline" ]] || [[ "${TEST}" == "all" ]]; then
    run_test "baseline_fbs1_adamw" "${BASELINE_CONFIG}"
fi

if [[ "${TEST}" == "fbs4" ]] || [[ "${TEST}" == "all" ]]; then
    run_test "fbs4_adamw" "${FBS4_CONFIG}"
fi

if [[ "${TEST}" == "adafactor" ]] || [[ "${TEST}" == "all" ]]; then
    run_test "fbs4_adafactor" "${ADAFACTOR_CONFIG}"
fi

echo ""
echo "============================================================"
echo "  ALL TESTS COMPLETE"
echo "  Logs in /tmp/torchtune/test_72b_*.log"
echo "============================================================"
echo ""
echo "Compare results:"
echo "  grep -E 'step_time|peak_memory' /tmp/torchtune/test_72b_*.log"

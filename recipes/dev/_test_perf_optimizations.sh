#!/bin/bash
# Performance optimization tests for 32B GRPO on Aurora XPU.
#
# Tests three optimizations that were disabled under broken CCL config:
#   1. sdpa            — Re-enable flash + mem_efficient SDPA
#   2a. compile-singlenode — torch.compile on single-node
#   2b. compile-multinode  — torch.compile on multi-node (needs 2-node hold job)
#   3. overlap          — AllGather-compute overlap (reshard_after_forward=None)
#   4. combined         — All optimizations together
#   baseline            — Production config (control)
#
# Usage:
#   bash recipes/dev/_test_perf_optimizations.sh <test_name>
#
# Single-node tests (baseline, sdpa, compile-singlenode, overlap, combined):
#   Requires a 1-node hold job. SSH to the compute node, then run.
#
# Multi-node tests (compile-multinode):
#   Requires a 2-node hold job. SSH to first node, then run.
#   Uses _launch_32b_test.sh pattern to source PBS env.
#
# Output: logs/perf_test_<name>_<timestamp>.log
# Look for TIMING lines to compare step times.

set -e
cd /lus/flare/projects/ModCon/ngetty/torchtune

TEST_NAME=${1:?Usage: $0 <baseline|sdpa|compile-singlenode|compile-multinode|overlap|combined>}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/perf_test_${TEST_NAME}_${TIMESTAMP}.log"

# Config mapping
case "${TEST_NAME}" in
    baseline)
        CONFIG=recipes/configs/dev/production/qwen32B_grpo_server_xpu.yaml
        MODE=singlenode
        EXTRA="num_steps=5"
        ;;
    sdpa)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_sdpa.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    compile-singlenode)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_compile_singlenode.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    compile-multinode)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_compile_multinode.yaml
        MODE=multinode
        EXTRA=""
        ;;
    overlap)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_overlap.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    combined)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_combined.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    compile-dynamic)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_compile_dynamic.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    optimized)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_optimized.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    G32)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_G32.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    chunked-loss)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_chunked_loss.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    G32-chunked)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_G32_chunked.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    packing)
        CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_test_packing.yaml
        MODE=singlenode
        EXTRA=""
        ;;
    *)
        echo "Unknown test: ${TEST_NAME}"
        echo "Valid: baseline, sdpa, compile-singlenode, compile-multinode, overlap, combined, compile-dynamic, optimized, G32, chunked-loss, G32-chunked, packing"
        exit 1
        ;;
esac

echo "=== Performance Test: ${TEST_NAME} ==="
echo "Config: ${CONFIG}"
echo "Mode: ${MODE}"
echo "Log: ${LOGFILE}"
echo "Date: $(date)"
echo ""

if [ "${MODE}" = "singlenode" ]; then
    # Single-node: use run_grpo_vllm_xpu.sh with 2 vLLM + 10 training tiles
    echo "Launching single-node 32B test (2 vLLM + 10 training tiles)..."
    bash recipes/dev/run_grpo_vllm_xpu.sh \
        2 10 \
        /lus/flare/projects/ModCon/ngetty/models/Qwen3-32B \
        5 \
        "${CONFIG}" \
        ${EXTRA} \
        2>&1 | tee "${LOGFILE}"

elif [ "${MODE}" = "multinode" ]; then
    # Multi-node: source PBS env from hold job, launch via HSDP script
    HOLD_PID=$(pgrep -u $(whoami) -f "sleep 3600" | head -1)
    if [[ -z "$HOLD_PID" ]]; then
        echo "ERROR: No hold job found (sleep 3600). Submit a 2-node hold job first:"
        echo "  qsub recipes/dev/hold_2nodes.sh"
        exit 1
    fi
    eval $(cat /proc/${HOLD_PID}/environ 2>/dev/null | tr '\0' '\n' | grep -E '^PBS_|^PALS_' | sed 's/^/export /')
    if [[ -z "$PBS_JOBID" ]]; then
        echo "ERROR: Could not extract PBS_JOBID from hold process"
        exit 1
    fi
    export PBS_NODEFILE="/var/spool/pbs/aux/${PBS_JOBID}"
    NNODES=$(wc -l < "${PBS_NODEFILE}")
    if [ "${NNODES}" -lt 2 ]; then
        echo "ERROR: compile-multinode requires 2+ nodes, found ${NNODES}"
        exit 1
    fi
    export USE_AFFINITY_MASK=training
    export CONFIG
    export NSTEPS=5
    export VLLM_MAX_MODEL_LEN=1024
    export MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B

    echo "PBS_JOBID=${PBS_JOBID}, nodes=${NNODES}"
    echo "Launching multi-node 32B HSDP test..."
    bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh 2>&1 | tee "${LOGFILE}"
fi

echo ""
echo "=== Test ${TEST_NAME} complete ==="
echo "Log: ${LOGFILE}"
echo ""
echo "=== TIMING summary ==="
grep "TIMING\|step_time\|total=" "${LOGFILE}" 2>/dev/null || echo "(no TIMING lines found)"

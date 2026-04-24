#!/bin/bash
#
# Test 32B (2 nodes) and 72B no-offload (4 nodes) with dedicated vLLM.
#
# Run on a held 4-node PBS job. Runs 32B first (2 nodes), then 72B (4 nodes).
#
# Usage:
#   ssh <node0> "bash /lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/test_no_offload.sh"
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune

# Auto-detect PBS job
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    # Try to find it from PBS_JOBID
    for f in /var/spool/pbs/aux/*; do
        if [ -f "$f" ]; then
            export PBS_NODEFILE="$f"
            break
        fi
    done
fi

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: Cannot find PBS_NODEFILE"
    exit 1
fi

echo "PBS_NODEFILE: ${PBS_NODEFILE}"
echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo ""

GENERIC_LAUNCHER="recipes/dev/aurora_grpo_dedicated_vllm_generic.sh"
mkdir -p /tmp/torchtune

# ============================================================
# Test 1: 32B on 2 nodes (1 vLLM + 1 training = 12 tiles)
# ============================================================
echo "============================================================"
echo "  TEST 1: 32B dedicated vLLM (2 nodes, 12 training tiles)"
echo "  Baseline comparison: 18.2s/step (10+2 colocated, single node)"
echo "============================================================"
echo ""

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B \
CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_dedicated_vllm_xpu.yaml \
MIN_NODES=2 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=300 \
NSTEPS=5 \
GRPO_SAMPLES=4 \
bash "${GENERIC_LAUNCHER}" 2>&1 | tee /tmp/torchtune/test_32b_dedicated_vllm.log

echo ""
echo "=== 32B test complete ==="
echo ""

# Clean up between tests
sleep 10

# ============================================================
# Test 2: 72B no-offload on 4 nodes (1 vLLM + 3 training = 36 tiles)
# ============================================================
echo "============================================================"
echo "  TEST 2: 72B no-offload (4 nodes, 36 training tiles, fbs=4)"
echo "  Baseline comparison: 84.6s/step (CPU offload, fbs=1)"
echo "============================================================"
echo ""

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload.yaml \
MIN_NODES=4 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=900 \
NSTEPS=3 \
GRPO_SAMPLES=4 \
bash "${GENERIC_LAUNCHER}" 2>&1 | tee /tmp/torchtune/test_72b_no_offload.log

echo ""
echo "============================================================"
echo "  ALL TESTS COMPLETE"
echo "  Logs: /tmp/torchtune/test_32b_dedicated_vllm.log"
echo "        /tmp/torchtune/test_72b_no_offload.log"
echo "============================================================"
echo ""
echo "Step times:"
grep "^TIMING" /tmp/torchtune/test_32b_dedicated_vllm.log 2>/dev/null || true
echo "---"
grep "^TIMING" /tmp/torchtune/test_72b_no_offload.log 2>/dev/null || true

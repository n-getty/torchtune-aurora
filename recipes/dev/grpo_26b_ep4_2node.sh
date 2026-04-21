#!/bin/bash
# 2-Node EP=4/DP=3 GRPO for Gemma4 26B-A4B MoE on Aurora XPU.
#
# Architecture:
#   Node 0 (vLLM):    12 tiles, vLLM TP=4 serving generation on port 8001
#   Node 1 (training): 12 tiles, EP=4/DP=3 torchtune GRPO
#
# Submit:
#   qsub recipes/dev/grpo_26b_ep4_2node.sh
#
# Or with overrides:
#   NSTEPS=10 qsub recipes/dev/grpo_26b_ep4_2node.sh
#
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A ModCon
#PBS -N grpo_26b_ep4_2node
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/grpo_26b_ep4_2node.out

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "${PROJDIR}"

NSTEPS=${NSTEPS:-5}
VLLM_PORT=8001
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
VLLM_LOG=/tmp/torchtune/vllm_26b_ep4.log

echo "=== Gemma4 26B-A4B EP=4/DP=3 GRPO — 2 Node ==="
echo "Date: $(date)"
echo "Job: ${PBS_JOBID:-local}"
echo "NSTEPS: ${NSTEPS}"

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
if [ ${#UNIQUE_NODES[@]} -lt 2 ]; then
    echo "ERROR: Need 2 nodes, got: ${UNIQUE_NODES[*]}"
    exit 1
fi

VLLM_NODE="${UNIQUE_NODES[0]}"
TRAIN_NODE="${UNIQUE_NODES[1]}"

# Get IP for vLLM to bypass Aurora's Squid HTTP proxy
VLLM_NODE_IP=$(ssh -o StrictHostKeyChecking=no "${VLLM_NODE}" "hostname -i" 2>/dev/null | awk '{print $1}')
if [[ -z "${VLLM_NODE_IP}" ]]; then
    echo "WARNING: Could not resolve IP for ${VLLM_NODE}, falling back to hostname"
    VLLM_NODE_IP="${VLLM_NODE}"
fi

# Bypass proxy for all intra-cluster traffic
export no_proxy="*"
export NO_PROXY="*"

echo "vLLM node:    ${VLLM_NODE} (IP=${VLLM_NODE_IP})"
echo "Training node: ${TRAIN_NODE}"

# ============================================================
# Stage model to both nodes in parallel
# ============================================================
LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
echo "Staging model to both nodes..."
for node in "${UNIQUE_NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no "${node}" "
        if [ ! -f '${LOCAL_MODEL}/config.json' ]; then
            echo 'Staging model on ${node}...'
            mkdir -p /tmp/torchtune
            t0=\$SECONDS
            cp -r '${MODEL_PATH}' '${LOCAL_MODEL}'
            echo \"Staged on ${node} in \$((SECONDS - t0))s\"
        else
            echo 'Model already staged on ${node}'
        fi
    " &
done
wait
echo "Model staging complete on all nodes."

# ============================================================
# Cleanup handler
# ============================================================
VLLM_SSH_PID=""
cleanup() {
    echo "Cleanup: stopping vLLM on ${VLLM_NODE}..."
    if [[ -n "${VLLM_SSH_PID}" ]]; then
        kill "${VLLM_SSH_PID}" 2>/dev/null || true
    fi
    # Kill vLLM processes on the vLLM node
    ssh -o StrictHostKeyChecking=no "${VLLM_NODE}" \
        "pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# ============================================================
# Launch vLLM server on Node 0 (background)
# ============================================================
echo "Starting vLLM on ${VLLM_NODE} (TP=4)..."
mkdir -p "$(dirname ${VLLM_LOG})" || true
ssh -o StrictHostKeyChecking=no "${VLLM_NODE}" \
    "bash ${PROJDIR}/recipes/dev/run_vllm_server_26b.sh" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_SSH_PID=$!
echo "vLLM SSH PID: ${VLLM_SSH_PID}"
echo "vLLM log: ${VLLM_LOG} (on ${VLLM_NODE})"

# ============================================================
# Launch EP=4 training on Node 1 (foreground; waits for vLLM internally)
# ============================================================
echo "Starting EP=4/DP=3 training on ${TRAIN_NODE}..."
echo "vLLM URL will be: http://${VLLM_NODE_IP}:${VLLM_PORT}"
ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" \
    "VLLM_NODE=${VLLM_NODE} VLLM_ADDR=${VLLM_NODE_IP} \
     bash ${PROJDIR}/recipes/dev/run_ep4_vllm_2node.sh ${NSTEPS}"
TRAIN_EXIT=$?

echo "=== Training complete (exit=${TRAIN_EXIT}) ==="

# cleanup trap will fire on exit
exit ${TRAIN_EXIT}

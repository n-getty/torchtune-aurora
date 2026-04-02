#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -q debug
#PBS -N grpo_multinode
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/grpo_multinode.out
#
# 2-node GRPO with vLLM server on Aurora XPU.
#
# Layout:
#   Node 0: tiles 0-9 training (10 ranks) + tiles 10-11 vLLM server (TP=2)
#   Node 1: tiles 0-9 training (10 ranks)
#   Total: 20 training ranks, 2 vLLM tiles
#
# Usage:
#   qsub recipes/dev/aurora_grpo_vllm_multinode.sh
#   Or interactive (on 2 allocated nodes):
#     bash recipes/dev/aurora_grpo_vllm_multinode.sh
#
set -e

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

# ============================================================
# Configuration
# ============================================================
NGPUS_PER_NODE=10  # Training tiles per node (uniform ppn, wastes 2 on node 1)
VLLM_TILES=1       # vLLM TP=1 on node 0 (avoids CCL conflict with training)
VLLM_PORT=8001
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
MODEL_PATH=${MODEL_PATH:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B}
NSTEPS=${NSTEPS:-5}
CONFIG=${CONFIG:-recipes/configs/dev/experimental/qwen3B_grpo_vllm_xpu.yaml}
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_grpo_vllm_wrapper.sh"

# ============================================================
# Environment setup
# ============================================================
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment — multi-node requires ring algorithms
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1  # was 4; 4 causes 48x AllGather regression
export CCL_ALLREDUCE=ring
# CCL_REDUCE_SCATTER=ring causes 63x regression on multi-node. Do NOT set.
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# vLLM / HuggingFace
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
export PYTHONPATH="${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ============================================================
# Node discovery
# ============================================================
if [[ -n "${PBS_NODEFILE:-}" ]]; then
    # PBS_NODEFILE has one line per core (208 per node on Aurora).
    # Preserve order from hostfile (rank 0 goes to first node listed).
    # awk '!seen[$0]++' preserves order while deduplicating.
    UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
    NODE0="${UNIQUE_NODES[0]}"
    NODE1="${UNIQUE_NODES[1]:-${UNIQUE_NODES[0]}}"
    NUM_NODES=${#UNIQUE_NODES[@]}
    echo "PBS_NODEFILE unique nodes (order-preserved): ${UNIQUE_NODES[*]}"
else
    NODE0=$(hostname | cut -d'.' -f1)
    NODE1=${NODE1:-$NODE0}
    NUM_NODES=${NUM_NODES:-1}
fi

# Master address: use HSN (high-speed network) for inter-node bandwidth
export MASTER_ADDR="${NODE0}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES
export NGPUS_PER_NODE

TOTAL_RANKS=$((NUM_NODES * NGPUS_PER_NODE))

echo "=== GRPO Multi-Node + vLLM XPU ==="
echo "Nodes:        ${NUM_NODES} (${NODE0}, ${NODE1})"
echo "Training:     ${TOTAL_RANKS} ranks (${NGPUS_PER_NODE}/node)"
echo "vLLM:         TP=${VLLM_TILES} on ${NODE0} tiles 10-11"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:       ${CONFIG}"
echo "Model:        ${MODEL_PATH}"
echo "Steps:        ${NSTEPS}"
echo "=================================="

mkdir -p /tmp/torchtune
mkdir -p "${TORCHTUNE_DIR}/logs"

# ============================================================
# Stage model to /tmp on BOTH nodes
# ============================================================
LOCAL_MODEL="/tmp/torchtune/$(basename ${MODEL_PATH})"

stage_model() {
    local node=$1
    if ssh "${node}" "test -f '${LOCAL_MODEL}/config.json'" 2>/dev/null; then
        echo "Model already staged on ${node}"
    else
        echo "Staging model to ${node}:${LOCAL_MODEL}..."
        ssh "${node}" "mkdir -p /tmp/torchtune && cp -r '${MODEL_PATH}' '${LOCAL_MODEL}'" 2>/dev/null
        echo "Staged on ${node}"
    fi
}

# Stage in parallel
stage_model "${NODE0}" &
PID_STAGE0=$!
if [[ "${NODE0}" != "${NODE1}" ]]; then
    stage_model "${NODE1}" &
    PID_STAGE1=$!
fi
wait $PID_STAGE0
[[ "${NODE0}" != "${NODE1}" ]] && wait $PID_STAGE1
MODEL_PATH="${LOCAL_MODEL}"

# ============================================================
# Warm vLLM model info cache on node 0
# ============================================================
VLLM_TILE_START=$((12 - VLLM_TILES))
VLLM_MASK=$(seq -s, ${VLLM_TILE_START} 11)

# vLLM placement: NODE0 for single-node vLLM, NODE1 for cross-node test
VLLM_NODE="${NODE0}"
echo "Warming vLLM cache on ${VLLM_NODE} tile ${VLLM_TILE_START}..."
ssh "${VLLM_NODE}" "module load frameworks 2>/dev/null; \
    ZE_AFFINITY_MASK=${VLLM_TILE_START} \
    ZE_FLAT_DEVICE_HIERARCHY=FLAT \
    PYTHONPATH='${PYTHONPATH}' \
    python3 -c \"
from vllm.config import ModelConfig
ModelConfig(model='${MODEL_PATH}', tokenizer='${MODEL_PATH}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
\"" 2>&1 | tail -3
echo ""

# ============================================================
# Launch vLLM server on node 0 (tiles 10-11)
# ============================================================
echo "Starting vLLM server on ${VLLM_NODE} tiles ${VLLM_MASK} (TP=${VLLM_TILES})..."
VLLM_LOG="/tmp/torchtune/vllm_server_multinode.log"

ssh "${VLLM_NODE}" "module load frameworks 2>/dev/null; \
    export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//'); \
    export ZE_AFFINITY_MASK=${VLLM_MASK}; \
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT; \
    export VLLM_WORKER_MULTIPROC_METHOD=spawn; \
    export TORCH_COMPILE_DISABLE=1; \
    export PYTHONPATH='${PYTHONPATH}'; \
    export HF_DATASETS_OFFLINE=1; \
    export HF_HUB_OFFLINE=1; \
    unset FI_PROVIDER; \
    unset CCL_ATL_TRANSPORT; \
    unset CCL_KVS_IFACE; \
    unset FI_CXI_RX_MATCH_MODE; \
    unset FI_CXI_OFLOW_BUF_SIZE; \
    unset FI_CXI_DEFAULT_CQ_SIZE; \
    export FI_CXI_DISABLE=1; \
    export CCL_PROCESS_LAUNCHER=none; \
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model '${MODEL_PATH}' \
        --tensor-parallel-size ${VLLM_TILES} \
        --host 0.0.0.0 \
        --port ${VLLM_PORT} \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        > '${VLLM_LOG}' 2>&1 &
    echo \$!" &

# Give SSH a moment, then get vLLM PID
sleep 3
VLLM_PID=$(ssh "${VLLM_NODE}" "pgrep -f 'vllm.entrypoints.openai.api_server' | head -1" 2>/dev/null || echo "")
echo "vLLM PID: ${VLLM_PID:-unknown}"

# Cleanup on exit
cleanup() {
    echo "Cleaning up vLLM server on ${VLLM_NODE}..."
    ssh "${VLLM_NODE}" "pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null; \
        pkill -f 'vllm.v1' 2>/dev/null" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT

# ============================================================
# Wait for vLLM health check
# ============================================================
echo "Waiting for vLLM health check on ${VLLM_NODE}:${VLLM_PORT}..."
VLLM_TIMEOUT=600
ELAPSED=0
while ! ssh "${VLLM_NODE}" "curl -s http://localhost:${VLLM_PORT}/health/ > /dev/null 2>&1" 2>/dev/null; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ ${ELAPSED} -ge ${VLLM_TIMEOUT} ]; then
        echo "ERROR: vLLM did not start within ${VLLM_TIMEOUT}s"
        ssh "${VLLM_NODE}" "tail -50 '${VLLM_LOG}'" 2>/dev/null || true
        exit 1
    fi
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  Still waiting... (${ELAPSED}s)"
    fi
done
echo "vLLM server ready (took ${ELAPSED}s)"

# ============================================================
# Launch training via mpiexec
# ============================================================
echo "Starting multi-node GRPO training (${TOTAL_RANKS} ranks across ${NUM_NODES} nodes)..."

# Export for wrapper
export MODEL_PATH
export NSTEPS
export CONFIG

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS_PER_NODE}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" \
    "dev/grpo_full_finetune_distributed_xpu" \
    "${CONFIG}" \
    "base_model_path=${MODEL_PATH}" \
    "num_steps=${NSTEPS}" \
    "vllm_url=http://localhost:${VLLM_PORT}"

echo "=== Multi-node GRPO training complete ==="

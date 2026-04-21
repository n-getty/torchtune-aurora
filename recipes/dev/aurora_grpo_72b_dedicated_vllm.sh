#!/bin/bash
#
# Dedicated vLLM node + multi-node training for 72B GRPO on Aurora XPU.
#
# Architecture (true FSDP, cross-node sharding):
#   Node 0 (vLLM only):    VLLM_DP replicas, each with VLLM_TP tiles
#   Nodes 1-N (training):  12 tiles/node, true FSDP across all training tiles
#
# 72B requires cross-node FSDP because model + optimizer (~720 GiB) exceeds
# single-node capacity (~576 GiB). With 24 tiles (2 training nodes), each tile
# holds ~32-36 GiB (6 GiB weights + 6 GiB ref + 12 GiB optimizer + activations).
#
# vLLM TP constraints for Qwen2.5-72B (64 heads, 8 KV heads):
#   TP=4, DP=3  →  3 replicas × 4 tiles = 12 tiles (36 GiB/tile model, ~12 GiB KV cache)
#   TP=8, DP=1  →  1 replica × 8 tiles (18 GiB/tile model, ~30 GiB KV cache, 4 tiles idle)
#   TP=2: 72 GiB/tile — exceeds 48 GiB tile limit, NOT viable for 72B
#
# Examples:
#   VLLM_TP=4 VLLM_DP=3 bash recipes/dev/aurora_grpo_72b_dedicated_vllm.sh
#   VLLM_TP=8 VLLM_DP=1 VLLM_MAX_MODEL_LEN=4096 bash recipes/dev/aurora_grpo_72b_dedicated_vllm.sh
#
# Usage (interactive on held 3-node PBS job):
#   export PBS_JOBID=<jobid>
#   export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
#   VLLM_TP=4 VLLM_DP=3 bash recipes/dev/aurora_grpo_72b_dedicated_vllm.sh
#
# Usage (PBS submission):
#   qsub recipes/dev/aurora_grpo_72b_dedicated_vllm.sh
#
#PBS -l select=3:system=aurora
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -o logs/grpo_72b_dedicated_vllm.out
#PBS -e logs/grpo_72b_dedicated_vllm.err
#PBS -N grpo_72b_dedicated_vllm
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

# ============================================================
# Configuration
# ============================================================
VLLM_TP=${VLLM_TP:-4}                 # Tensor parallelism per replica
VLLM_DP=${VLLM_DP:-3}                 # Number of vLLM replicas
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.80}
VLLM_TIMEOUT=${VLLM_TIMEOUT:-900}     # 72B takes longer to load
TRAIN_TILES_PER_NODE=${TRAIN_TILES_PER_NODE:-12}
MODEL_SRC=${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/$(basename ${MODEL_SRC})}
NSTEPS=${NSTEPS:-3}
GRPO_SAMPLES=${GRPO_SAMPLES:-8}
CONFIG=${CONFIG:-recipes/configs/dev/experimental/qwen72B_grpo_dedicated_vllm_xpu.yaml}
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_grpo_vllm_wrapper.sh"

# Validate: TP * DP must equal 12
TOTAL_VLLM_TILES=$((VLLM_TP * VLLM_DP))
if [ ${TOTAL_VLLM_TILES} -ne 12 ] && [ ${VLLM_DP} -gt 1 ]; then
    echo "ERROR: VLLM_TP(${VLLM_TP}) * VLLM_DP(${VLLM_DP}) = ${TOTAL_VLLM_TILES}, must be 12"
    exit 1
fi

# ============================================================
# Environment setup
# ============================================================
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
# NOTE: Do NOT set CCL_REDUCE_SCATTER=ring — causes 63x regression
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# Paths
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
VLLM_PYTHONPATH="$(aurora_pythonpath "${TORCHTUNE_DIR}" "${TRL_DIR}" "${VLLM_CUSTOMIZATION}")"
export PYTHONPATH="${VLLM_PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ============================================================
# Node discovery from PBS_NODEFILE
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Set it first:"
    echo "  export PBS_JOBID=<jobid>"
    echo "  export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
if [ ${#UNIQUE_NODES[@]} -lt 3 ]; then
    echo "ERROR: Need at least 3 nodes (1 vLLM + 2 training). Got: ${UNIQUE_NODES[*]}"
    exit 1
fi

VLLM_NODE="${UNIQUE_NODES[0]}"
TRAIN_NODES=("${UNIQUE_NODES[@]:1}")  # All nodes after the first
NUM_TRAIN_NODES=${#TRAIN_NODES[@]}
TOTAL_TRAIN_TILES=$((NUM_TRAIN_NODES * TRAIN_TILES_PER_NODE))

VLLM_NODE_HSN="${VLLM_NODE}.hsn.cm.aurora.alcf.anl.gov"
TRAIN_NODE_0_HSN="${TRAIN_NODES[0]}.hsn.cm.aurora.alcf.anl.gov"

# Get vLLM node IP for direct HTTP (bypasses Squid proxy on Aurora)
VLLM_NODE_IP=$(ssh "${VLLM_NODE}" "hostname -i" 2>/dev/null | head -1)
if [[ -z "${VLLM_NODE_IP}" ]]; then
    echo "ERROR: Could not resolve IP for ${VLLM_NODE}"
    exit 1
fi

# Bypass HTTP proxy for internal cluster traffic
export no_proxy="*"
export NO_PROXY="*"

export MASTER_ADDR="${TRAIN_NODE_0_HSN}"
export MASTER_PORT=$((20000 + RANDOM % 20000))

echo "=== 72B Dedicated vLLM Node GRPO Test ==="
echo "vLLM node:      ${VLLM_NODE} (IP=${VLLM_NODE_IP}, TP=${VLLM_TP}, DP=${VLLM_DP})"
echo "Train nodes:    ${TRAIN_NODES[*]} (${NUM_TRAIN_NODES} nodes × ${TRAIN_TILES_PER_NODE} tiles = ${TOTAL_TRAIN_TILES} tiles)"
echo "FSDP:           True FSDP (dp_replicate=1, ${TOTAL_TRAIN_TILES}-way sharding)"
echo "Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:         ${CONFIG}"
echo "Model:          ${MODEL_PATH}"
echo "GRPO samples:   ${GRPO_SAMPLES}"
echo "Steps:          ${NSTEPS}"
echo "========================================"

# ============================================================
# Stage model to /tmp on ALL nodes
# ============================================================
echo "Staging model to all nodes..."
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "  Copying model to ${node}:${MODEL_PATH} (~144 GiB, may take several minutes)..."
        ssh "${node}" "mkdir -p $(dirname ${MODEL_PATH}) && cp -r ${MODEL_SRC} ${MODEL_PATH}" &
    else
        echo "  Model already staged on ${node}"
    fi
done
wait
echo "Model staging complete."

# ============================================================
# Warm vLLM cache on vLLM node
# ============================================================
echo "Warming vLLM cache on ${VLLM_NODE}..."
ssh "${VLLM_NODE}" "
cd ${TORCHTUNE_DIR}
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python3 -c \"
from vllm.config import ModelConfig
ModelConfig(
    model='${MODEL_PATH}',
    tokenizer='${MODEL_PATH}',
    dtype='bfloat16',
    enforce_eager=True,
)
print('Cache warmed for 72B model')
\" 2>&1 | tail -1
"

# ============================================================
# Launch vLLM replicas on Node 0
# ============================================================
VLLM_PIDS=()
VLLM_URLS=""

echo "Starting ${VLLM_DP} vLLM replicas on ${VLLM_NODE} (TP=${VLLM_TP})..."
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    TILE_START=$((r * VLLM_TP))
    TILE_END=$((TILE_START + VLLM_TP - 1))
    TILE_MASK=$(seq -s, ${TILE_START} ${TILE_END})
    VLLM_LOG="/tmp/torchtune/vllm_72b_replica_${r}.log"

    # Build comma-separated URL list (use IP to bypass Squid proxy)
    if [ -n "${VLLM_URLS}" ]; then
        VLLM_URLS="${VLLM_URLS},http://${VLLM_NODE_IP}:${PORT}"
    else
        VLLM_URLS="http://${VLLM_NODE_IP}:${PORT}"
    fi

    ssh "${VLLM_NODE}" "
cd ${TORCHTUNE_DIR}
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${TILE_MASK}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH='${VLLM_PYTHONPATH}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
mkdir -p /tmp/torchtune
python3 -m vllm.entrypoints.openai.api_server \
    --model '${MODEL_PATH}' \
    --tensor-parallel-size ${VLLM_TP} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL} \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --distributed-executor-backend mp \
    > '${VLLM_LOG}' 2>&1
" &
    VLLM_PIDS+=($!)
    echo "  Replica ${r}: tiles ${TILE_MASK}, port ${PORT} (PID $!)"
done

echo "vLLM URLs: ${VLLM_URLS}"

# ============================================================
# Cleanup: kill vLLM on exit
# ============================================================
cleanup() {
    echo "Cleaning up vLLM on ${VLLM_NODE}..."
    ssh "${VLLM_NODE}" "
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null
pkill -f 'vllm.v1.engine' 2>/dev/null
pkill -f 'from multiprocessing' 2>/dev/null
" 2>/dev/null || true
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ============================================================
# Wait for all vLLM replicas to become healthy
# ============================================================
echo "Waiting for vLLM replicas to become healthy (timeout=${VLLM_TIMEOUT}s)..."
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    ELAPSED=0
    while ! ssh "${VLLM_NODE}" "curl --noproxy '*' -s http://localhost:${PORT}/health/ > /dev/null 2>&1" 2>/dev/null; do
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge ${VLLM_TIMEOUT} ]; then
            echo "ERROR: vLLM replica ${r} on port ${PORT} did not start within ${VLLM_TIMEOUT}s"
            ssh "${VLLM_NODE}" "tail -50 /tmp/torchtune/vllm_72b_replica_${r}.log" 2>/dev/null || true
            exit 1
        fi
        if [ $((ELAPSED % 60)) -eq 0 ]; then
            echo "  Still waiting for replica ${r} on port ${PORT} (${ELAPSED}s)..."
        fi
    done
    echo "  Replica ${r} healthy on port ${PORT} (${ELAPSED}s)"
done
echo "All ${VLLM_DP} vLLM replicas ready."

# ============================================================
# Create hostfile for training nodes
# ============================================================
TRAIN_HOSTFILE=$(mktemp /tmp/train_hostfile.XXXXXX)
for train_node in "${TRAIN_NODES[@]}"; do
    for ((i=0; i<TRAIN_TILES_PER_NODE; i++)); do
        echo "${train_node}" >> "${TRAIN_HOSTFILE}"
    done
done
echo "Training hostfile: ${TRAIN_HOSTFILE} (${TOTAL_TRAIN_TILES} entries across ${NUM_TRAIN_NODES} nodes)"

# ============================================================
# Launch training on Nodes 1-N
# ============================================================
# True FSDP: shard across all training tiles (dp_replicate=1, default).
# No HSDP mesh — all tiles collectively hold one model copy.
# Unset ZE_AFFINITY_MASK so CCL can discover all device UUIDs for topology routing.
export USE_AFFINITY_MASK=training

# Export WORLD_SIZE/NUM_NODES for the wrapper (PALS_NRANKS/PMI_SIZE may not be set with custom hostfile).
export NUM_NODES=${NUM_TRAIN_NODES}
export WORLD_SIZE=${TOTAL_TRAIN_TILES}
export NGPUS_PER_NODE=${TRAIN_TILES_PER_NODE}
# Bypass HTTP proxy for vLLM connections
export no_proxy="*"
export NO_PROXY="*"

echo ""
echo "Starting training on ${NUM_TRAIN_NODES} nodes (${TOTAL_TRAIN_TILES} tiles, true FSDP)..."
echo "vLLM URLs: ${VLLM_URLS}"
mpiexec \
    --pmi=pmix \
    --hostfile "${TRAIN_HOSTFILE}" \
    -n "${TOTAL_TRAIN_TILES}" \
    -ppn "${TRAIN_TILES_PER_NODE}" \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" \
    "dev/grpo_full_finetune_distributed_xpu" \
    "${CONFIG}" \
    "base_model_path=${MODEL_PATH}" \
    "num_steps=${NSTEPS}" \
    "grpo_samples=${GRPO_SAMPLES}" \
    "vllm_url=${VLLM_URLS}" \
    "vllm_weight_sync=false"

rm -f "${TRAIN_HOSTFILE}"
echo "=== 72B Training complete ==="

#!/bin/bash
#
# Generalized dedicated vLLM node + multi-node training for GRPO on Aurora XPU.
#
# Architecture (1 vLLM node + N training nodes):
#   Node 0 (vLLM only):    VLLM_DP replicas, each with VLLM_TP tiles
#   Nodes 1-N (training):  12 tiles/node, true FSDP across all training tiles
#
# Works for any model size. Set MODEL_SRC and CONFIG via env vars.
#
# Examples:
#   # 32B on 2 nodes (1 vLLM + 1 training = 12 tiles)
#   MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B \
#   CONFIG=recipes/configs/dev/experimental/qwen32B_grpo_dedicated_vllm_xpu.yaml \
#   MIN_NODES=2 VLLM_TP=4 VLLM_DP=3 \
#   bash recipes/dev/aurora_grpo_dedicated_vllm_generic.sh
#
#   # 72B on 4 nodes (1 vLLM + 3 training = 36 tiles), no CPU offload
#   MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
#   CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload.yaml \
#   MIN_NODES=4 VLLM_TP=4 VLLM_DP=3 VLLM_TIMEOUT=900 \
#   bash recipes/dev/aurora_grpo_dedicated_vllm_generic.sh
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

# ============================================================
# Configuration (all overridable via env vars)
# ============================================================
VLLM_TP=${VLLM_TP:-4}
VLLM_DP=${VLLM_DP:-3}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.80}
VLLM_TIMEOUT=${VLLM_TIMEOUT:-600}
TRAIN_TILES_PER_NODE=${TRAIN_TILES_PER_NODE:-12}
MODEL_SRC=${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/$(basename ${MODEL_SRC})}
NSTEPS=${NSTEPS:-3}
GRPO_SAMPLES=${GRPO_SAMPLES:-4}
CONFIG=${CONFIG:-recipes/configs/dev/experimental/qwen72B_grpo_dedicated_vllm_xpu.yaml}
MIN_NODES=${MIN_NODES:-2}
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_grpo_vllm_wrapper.sh"

# ============================================================
# Environment setup
# ============================================================
module load frameworks 2>/dev/null || true

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
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# NOTE: expandable_segments:True is INCOMPATIBLE with oneCCL RDMA (CXI fabric).
# max_split_size_mb prevents splintering of FSDP AllGather blocks by small requests.
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
export TORCH_COMPILE_DISABLE=1

VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
VLLM_PYTHONPATH="$(aurora_pythonpath "${TORCHTUNE_DIR}" "${TRL_DIR}" "${VLLM_CUSTOMIZATION}")"
export PYTHONPATH="${VLLM_PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set."
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
if [ ${#UNIQUE_NODES[@]} -lt ${MIN_NODES} ]; then
    echo "ERROR: Need at least ${MIN_NODES} nodes. Got ${#UNIQUE_NODES[@]}: ${UNIQUE_NODES[*]}"
    exit 1
fi

USED_NODES=("${UNIQUE_NODES[@]:0:${MIN_NODES}}")
VLLM_NODE="${USED_NODES[0]}"
TRAIN_NODES=("${USED_NODES[@]:1}")
NUM_TRAIN_NODES=${#TRAIN_NODES[@]}
TOTAL_TRAIN_TILES=$((NUM_TRAIN_NODES * TRAIN_TILES_PER_NODE))

TRAIN_NODE_0_HSN="${TRAIN_NODES[0]}.hsn.cm.aurora.alcf.anl.gov"

VLLM_NODE_IP=$(ssh "${VLLM_NODE}" "hostname -i" 2>/dev/null | head -1)
if [[ -z "${VLLM_NODE_IP}" ]]; then
    echo "ERROR: Could not resolve IP for ${VLLM_NODE}"
    exit 1
fi

export no_proxy="*"
export NO_PROXY="*"
export MASTER_ADDR="${TRAIN_NODE_0_HSN}"
export MASTER_PORT=$((20000 + RANDOM % 20000))

echo "=== Dedicated vLLM GRPO Test ==="
echo "Model:          $(basename ${MODEL_SRC})"
echo "vLLM node:      ${VLLM_NODE} (IP=${VLLM_NODE_IP}, TP=${VLLM_TP}, DP=${VLLM_DP})"
echo "Train nodes:    ${TRAIN_NODES[*]} (${NUM_TRAIN_NODES} x ${TRAIN_TILES_PER_NODE} = ${TOTAL_TRAIN_TILES} tiles)"
echo "Config:         ${CONFIG}"
echo "Steps:          ${NSTEPS}, G=${GRPO_SAMPLES}"
echo "================================="

# ============================================================
# Stage model
# ============================================================
echo "Staging model to all nodes..."
for node in "${USED_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "  Copying to ${node}:${MODEL_PATH}..."
        ssh "${node}" "mkdir -p $(dirname ${MODEL_PATH}) && cp -r ${MODEL_SRC} ${MODEL_PATH}" &
    else
        echo "  Already staged on ${node}"
    fi
done
wait
echo "Model staging complete."

# ============================================================
# Warm vLLM cache
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
ModelConfig(model='${MODEL_PATH}', tokenizer='${MODEL_PATH}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
\" 2>&1 | tail -1
"

# ============================================================
# Launch vLLM replicas
# ============================================================
VLLM_PIDS=()
VLLM_URLS=""

echo "Starting ${VLLM_DP} vLLM replicas on ${VLLM_NODE} (TP=${VLLM_TP})..."
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    TILE_START=$((r * VLLM_TP))
    TILE_END=$((TILE_START + VLLM_TP - 1))
    TILE_MASK=$(seq -s, ${TILE_START} ${TILE_END})
    VLLM_LOG="/tmp/torchtune/vllm_replica_${r}.log"

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
# NOTE: Do NOT set PYTORCH_ALLOC_CONF=expandable_segments:True for vLLM.
# Expandable segments use virtual memory mapping that produces non-standard
# USM pointer types, causing oneCCL "invalid usm pointer type: unknown" errors.
unset PYTORCH_ALLOC_CONF
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
# Cleanup
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
# Wait for health
# ============================================================
echo "Waiting for vLLM (timeout=${VLLM_TIMEOUT}s)..."
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    ELAPSED=0
    while ! ssh "${VLLM_NODE}" "curl --noproxy '*' -s http://localhost:${PORT}/health/ > /dev/null 2>&1" 2>/dev/null; do
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge ${VLLM_TIMEOUT} ]; then
            echo "ERROR: vLLM replica ${r} did not start within ${VLLM_TIMEOUT}s"
            ssh "${VLLM_NODE}" "tail -50 /tmp/torchtune/vllm_replica_${r}.log" 2>/dev/null || true
            exit 1
        fi
        if [ $((ELAPSED % 60)) -eq 0 ]; then
            echo "  Waiting for replica ${r} (${ELAPSED}s)..."
        fi
    done
    echo "  Replica ${r} healthy (${ELAPSED}s)"
done
echo "All ${VLLM_DP} vLLM replicas ready."

# ============================================================
# Launch training
# ============================================================
TRAIN_HOSTFILE=$(mktemp /tmp/train_hostfile.XXXXXX)
for train_node in "${TRAIN_NODES[@]}"; do
    for ((i=0; i<TRAIN_TILES_PER_NODE; i++)); do
        echo "${train_node}" >> "${TRAIN_HOSTFILE}"
    done
done

export USE_AFFINITY_MASK=training
export NUM_NODES=${NUM_TRAIN_NODES}
export WORLD_SIZE=${TOTAL_TRAIN_TILES}
export NGPUS_PER_NODE=${TRAIN_TILES_PER_NODE}

echo ""
echo "Starting training on ${NUM_TRAIN_NODES} nodes (${TOTAL_TRAIN_TILES} tiles)..."
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
echo "=== Training complete ==="

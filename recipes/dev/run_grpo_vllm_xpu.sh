#!/bin/bash
# GRPO with vLLM-accelerated generation on Aurora XPU
#
# Usage:
#   bash recipes/dev/run_grpo_vllm_xpu.sh [vllm_tiles] [train_tiles] [model_path] [num_steps] [config_file] [extra_args...]
#
# Tile layout: training on tiles 0..TRAIN_TILES-1, vLLM on tiles (12-VLLM_TILES)..11
# Training ranks do NOT use ZE_AFFINITY_MASK (CCL needs full device visibility).
# Valid train_tiles: 2, 4, 6, 10, 12 (must align with CCL topology).
# Default: 1 vLLM tile (tile 11), 10 training tiles (tiles 0-9)
#
# vLLM Data Parallelism:
#   VLLM_DP=2 bash recipes/dev/run_grpo_vllm_xpu.sh 2 10 ...
#   Launches VLLM_DP independent vLLM instances, each with TP=VLLM_TILES/VLLM_DP.
#   Nearly doubles generation throughput for small models (3B).
set -e

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

# Load Aurora frameworks module — MUST use 2025.2.0 (2025.3.1 has broken XCCL allreduce)
module load frameworks/2025.2.0 2>/dev/null || true

# Remove user virtualenv from PATH so frameworks python is used
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
# NOTE: Do NOT set PYTHONNOUSERSITE=1 — math_verify is only in ~/.local

# CCL / XPU environment
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
# FLAT mode: Each tile is an independent device (12 tiles/node).
# Training does NOT set ZE_AFFINITY_MASK — CCL needs to see all device UUIDs
# for multi-rank allreduce. Each rank targets the correct tile via
# device_id=xpu:{LOCAL_RANK} in init_process_group.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# CXI fabric tuning (Slingshot 11) — from PRISM production config
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
# XPU memory allocator
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Increase CCL IPC handle cache to suppress "mem handle cache limit reached" warnings
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
# NOTE: Do NOT set CCL_ALLREDUCE=ring / CCL_REDUCE_SCATTER=ring for
# single-node FSDP2 — they force the scheduler path which doesn't support
# ReduceOp.AVG. Only needed for multi-node with large tensors.
# usercustomize patch: disable transformers version check (hf-hub 1.7 vs <1.0)
VLLM_CUSTOMIZATION=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/_usercustomize_vllm
VLLM_GEMMA4_OVERLAY=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay
# If VLLM_GEMMA4=1, use Gemma4 overlay usercustomize (registers gemma4 arch with vLLM)
# instead of the standard one. The overlay's usercustomize.py includes all Aurora patches.
if [ "${VLLM_GEMMA4:-0}" = "1" ]; then
    VLLM_CUSTOMIZATION=${VLLM_GEMMA4_OVERLAY}
fi
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

VLLM_TILES=${1:-1}
TRAIN_TILES=${2:-10}
MODEL_PATH=${3:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B}
NSTEPS=${4:-10}
CONFIG=${5:-recipes/configs/dev/experimental/qwen3B_grpo_vllm_xpu.yaml}
shift 5 2>/dev/null || true
EXTRA_ARGS="$@"  # additional config overrides (e.g., vllm_weight_sync=true)
VLLM_BASE_PORT=8001
VLLM_DP=${VLLM_DP:-1}
VLLM_TP=$((VLLM_TILES / VLLM_DP))
VLLM_LOG=/tmp/torchtune/vllm_server.log
# vLLM max_model_len: must be >= max prompt length + max_generated_tokens.
# Default 2048 covers Config B (max_gen=512 + prompt ~300).
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}

echo "=== GRPO + vLLM XPU: ${VLLM_TILES} vLLM tile(s) (TP=${VLLM_TP} × DP=${VLLM_DP}), ${TRAIN_TILES} training tiles, ${NSTEPS} steps, config=${CONFIG} ==="
echo "Node: $(hostname), Date: $(date)"
echo "Python: $(which python3)"
echo "Model: ${MODEL_PATH}"

mkdir -p /tmp/torchtune
# Weight sync uses /dev/shm (RAM tmpfs, 504 GB on Aurora) for fast I/O.
# Pre-create the directory so training ranks don't race to create it.
mkdir -p /dev/shm/torchtune

# --- Stage model to /tmp for fast loading ---
LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi
MODEL_PATH=${LOCAL_MODEL}

# vLLM goes on the LAST tiles (e.g., tile 11 for 1 vLLM tile, tiles 10-11 for 2).
# Training uses tiles 0..TRAIN_TILES-1 with NO ZE_AFFINITY_MASK (CCL needs full
# device visibility for ReduceOp.AVG). vLLM is isolated via ZE_AFFINITY_MASK.
VLLM_TILE_START=$((12 - VLLM_TILES))
VLLM_MASK=$(seq -s, ${VLLM_TILE_START} 11)

# --- 0. Warm vLLM model info cache ---
# vLLM 0.15 inspects model architectures via subprocess, which segfaults on XPU.
# Our usercustomize.py patches it to run in-process (fallback), but that consumes
# GPU memory on the same tile. Pre-warming the cache on a scratch tile avoids this.
# Use the vLLM tile itself for cache warm (it will be used for vLLM anyway)
SCRATCH_TILE=${VLLM_TILE_START}
echo "Warming vLLM model info cache on tile ${SCRATCH_TILE}..."
ZE_AFFINITY_MASK=${SCRATCH_TILE} python3 -c "
from vllm.config import ModelConfig
ModelConfig(
    model='${MODEL_PATH}',
    tokenizer='${MODEL_PATH}',
    dtype='bfloat16',
    enforce_eager=True,
)
print('Cache warmed successfully')
" 2>&1 | tail -3
echo ""

# --- 1. Launch vLLM server(s) ---
# For TP=1: override CCL to MPI transport to isolate from training's OFI/CXI.
# For TP>1: vLLM workers need CCL for allreduce, so use OFI transport (same as
# training). ZE_AFFINITY_MASK isolates vLLM to its own tiles. Training doesn't
# start until vLLM is ready, and they use separate process groups.
#
# DP>1: launch multiple independent vLLM instances on separate tiles/ports.
# Each replica gets TP tiles. Round-robin dispatch in the training recipe.

VLLM_PIDS=()
VLLM_URLS=""

launch_vllm_replica() {
    local REPLICA=$1
    local TILE_MASK=$2
    local PORT=$3
    local LOG="${VLLM_LOG}.${REPLICA}"

    echo "  Replica ${REPLICA}: tiles=${TILE_MASK}, port=${PORT}, log=${LOG}"

    # Use standard vLLM OpenAI API server with WeightSyncFromFileExtension.
    # The extension adds load_weights_from_path() callable via /collective_rpc,
    # which the training recipe calls after saving weights to /tmp as safetensors.
    # This avoids XCCL communicator setup which SIGABRTs on XPU.
    #
    # For TP>1: needs OFI transport and --distributed-executor-backend mp.
    # For TP=1: use MPI transport to isolate from training's OFI/CXI fabric.
    local EXTRA_VLLM_ARGS=""
    local VLLM_CCL_TRANSPORT="mpi"
    local VLLM_CCL_LAUNCHER="None"
    local VLLM_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
    if [ ${VLLM_TP} -gt 1 ]; then
        EXTRA_VLLM_ARGS="--distributed-executor-backend mp"
        VLLM_CCL_TRANSPORT="ofi"
        VLLM_CCL_LAUNCHER="none"
        VLLM_ALLOC_CONF=""  # expandable_segments breaks CCL RDMA for TP>1
    fi

    ZE_AFFINITY_MASK=${TILE_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        PYTORCH_ALLOC_CONF=${VLLM_ALLOC_CONF} \
        CCL_PROCESS_LAUNCHER=${VLLM_CCL_LAUNCHER} \
        CCL_ATL_TRANSPORT=${VLLM_CCL_TRANSPORT} \
        FI_PROVIDER=cxi \
        CCL_KVS_IFACE=lo \
        VLLM_SERVER_DEV_MODE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --tensor-parallel-size ${VLLM_TP} \
        --port ${PORT} \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \
        ${EXTRA_VLLM_ARGS} \
        > "${LOG}" 2>&1 &
    VLLM_PIDS+=($!)
}

echo "Starting ${VLLM_DP} vLLM replica(s) (TP=${VLLM_TP} each) on tile(s) ${VLLM_MASK}..."
for ((r=0; r<VLLM_DP; r++)); do
    REPLICA_TILE_START=$((VLLM_TILE_START + r * VLLM_TP))
    REPLICA_TILE_END=$((REPLICA_TILE_START + VLLM_TP - 1))
    REPLICA_MASK=$(seq -s, ${REPLICA_TILE_START} ${REPLICA_TILE_END})
    REPLICA_PORT=$((VLLM_BASE_PORT + r))

    launch_vllm_replica ${r} ${REPLICA_MASK} ${REPLICA_PORT}

    if [ -n "${VLLM_URLS}" ]; then
        VLLM_URLS="${VLLM_URLS},http://localhost:${REPLICA_PORT}"
    else
        VLLM_URLS="http://localhost:${REPLICA_PORT}"
    fi
done

# Cleanup on exit — kill all vLLM process trees
cleanup() {
    echo "Cleaning up ${#VLLM_PIDS[@]} vLLM server(s)..."
    for PID in "${VLLM_PIDS[@]}"; do
        kill -- -${PID} 2>/dev/null || true
        pkill -P ${PID} 2>/dev/null || true
        kill ${PID} 2>/dev/null || true
        wait ${PID} 2>/dev/null || true
    done
    # Remove weight sync file from /dev/shm to free RAM (can be 60+ GB for 31B).
    rm -f /dev/shm/torchtune/weight_update.safetensors 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT

# --- 2. Wait for all vLLM replicas to be ready ---
VLLM_TIMEOUT=600
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    echo "Waiting for vLLM replica ${r} health check on port ${PORT}..."
    ELAPSED=0
    while ! curl -s http://localhost:${PORT}/health/ > /dev/null 2>&1; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [ ${ELAPSED} -ge ${VLLM_TIMEOUT} ]; then
            echo "ERROR: vLLM replica ${r} did not start within ${VLLM_TIMEOUT}s"
            echo "=== vLLM log (replica ${r}) ==="
            tail -50 "${VLLM_LOG}.${r}"
            exit 1
        fi
    done
    echo "vLLM replica ${r} ready (took ${ELAPSED}s)"
done
echo "All ${VLLM_DP} vLLM replica(s) ready"

# --- 3. Launch training ---
# Training uses tiles 0..TRAIN_TILES-1 with NO ZE_AFFINITY_MASK.
# CCL needs to see all 12 device UUIDs for ReduceOp.AVG; each rank
# targets its tile via device_id=xpu:LOCAL_RANK.
echo "Starting training on ${TRAIN_TILES} tiles (xpu:0 through xpu:$((TRAIN_TILES-1)))..."
# Pass vLLM URL(s) to training — comma-separated for DP>1
VLLM_URL_OVERRIDE="vllm_url=${VLLM_URLS}"
echo "vLLM URLs: ${VLLM_URLS}"
python3 -m torch.distributed.run --standalone --nproc_per_node=${TRAIN_TILES} \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} ${VLLM_URL_OVERRIDE} ${EXTRA_ARGS}

echo "=== Training complete ==="

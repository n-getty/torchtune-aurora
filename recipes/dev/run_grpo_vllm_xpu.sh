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
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune

# Load Aurora frameworks module (provides XPU-enabled PyTorch + python 3.12)
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH so frameworks python is used
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

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
# NOTE: Do NOT set CCL_ALLREDUCE=ring / CCL_REDUCE_SCATTER=ring for
# single-node FSDP2 — they force the scheduler path which doesn't support
# ReduceOp.AVG. Only needed for multi-node with large tensors.
# usercustomize patch: disable transformers version check (hf-hub 1.7 vs <1.0)
VLLM_CUSTOMIZATION=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/_usercustomize_vllm
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
VLLM_PORT=8001
VLLM_LOG=/tmp/torchtune/vllm_server.log
# vLLM max_model_len: must be >= max prompt length + max_generated_tokens.
# Default 2048 covers Config B (max_gen=512 + prompt ~300).
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}

echo "=== GRPO + vLLM XPU: ${VLLM_TILES} vLLM tile(s), ${TRAIN_TILES} training tiles, ${NSTEPS} steps, config=${CONFIG} ==="
echo "Node: $(hostname), Date: $(date)"
echo "Python: $(which python3)"
echo "Model: ${MODEL_PATH}"

mkdir -p /tmp/torchtune

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

# --- 1. Launch vLLM server ---
# For TP=1: override CCL to MPI transport to isolate from training's OFI/CXI.
# For TP>1: vLLM workers need CCL for allreduce, so use OFI transport (same as
# training). ZE_AFFINITY_MASK isolates vLLM to its own tiles. Training doesn't
# start until vLLM is ready, and they use separate process groups.
echo "Starting vLLM server on tile(s) ${VLLM_MASK}..."
# Choose server launch: TRL's vllm_serve for TP=1, vllm CLI serve for TP>1
# TRL's vllm_serve uses the LLM API which fails with V1 TP>1 on XPU.
# The vllm CLI serve uses the AsyncLLMEngine path which works.
if [ ${VLLM_TILES} -gt 1 ]; then
    echo "Using vllm serve CLI (TP=${VLLM_TILES})"
    ZE_AFFINITY_MASK=${VLLM_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --tensor-parallel-size ${VLLM_TILES} \
        --port ${VLLM_PORT} \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --distributed-executor-backend mp \
        > "${VLLM_LOG}" 2>&1 &
else
    echo "Using TRL vllm_serve (TP=1)"
    ZE_AFFINITY_MASK=${VLLM_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        CCL_ATL_TRANSPORT=mpi \
        CCL_PROCESS_LAUNCHER=None \
        python3 recipes/dev/vllm_serve_xpu.py \
        --model "${MODEL_PATH}" \
        --tensor_parallel_size ${VLLM_TILES} \
        --port ${VLLM_PORT} \
        --enforce_eager \
        --dtype bfloat16 \
        --gpu_memory_utilization 0.80 \
        --max_model_len ${VLLM_MAX_MODEL_LEN} \
        > "${VLLM_LOG}" 2>&1 &
fi
VLLM_PID=$!

# Cleanup on exit — kill entire process tree (vLLM spawns EngineCore + workers)
cleanup() {
    echo "Cleaning up vLLM server (PID ${VLLM_PID}) and child processes..."
    # Kill the whole process group, then any stragglers
    kill -- -${VLLM_PID} 2>/dev/null || true
    pkill -P ${VLLM_PID} 2>/dev/null || true
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    # Give L0 driver a moment to release contexts
    sleep 1
}
trap cleanup EXIT

# --- 2. Wait for vLLM to be ready ---
echo "Waiting for vLLM health check on port ${VLLM_PORT}..."
VLLM_TIMEOUT=600
ELAPSED=0
while ! curl -s http://localhost:${VLLM_PORT}/health/ > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ ${ELAPSED} -ge ${VLLM_TIMEOUT} ]; then
        echo "ERROR: vLLM server did not start within ${VLLM_TIMEOUT}s"
        echo "=== vLLM log ==="
        tail -50 "${VLLM_LOG}"
        exit 1
    fi
done
echo "vLLM server ready (took ${ELAPSED}s)"

# --- 3. Launch training ---
# Training uses tiles 0..TRAIN_TILES-1 with NO ZE_AFFINITY_MASK.
# CCL needs to see all 12 device UUIDs for ReduceOp.AVG; each rank
# targets its tile via device_id=xpu:LOCAL_RANK.
echo "Starting training on ${TRAIN_TILES} tiles (xpu:0 through xpu:$((TRAIN_TILES-1)))..."
python3 -m torch.distributed.run --standalone --nproc_per_node=${TRAIN_TILES} \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} ${EXTRA_ARGS}

echo "=== Training complete ==="

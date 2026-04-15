#!/bin/bash
# GRPO with vLLM-accelerated generation on Aurora XPU — Gemma 4 31B
#
# Uses the vllm_gemma4_overlay to add Gemma4 model support to vLLM 0.10.1.
# MUST use frameworks/2025.2.0 (2025.3.1 has broken XCCL allreduce).
#
# Usage:
#   bash recipes/dev/run_gemma4_grpo_vllm.sh [vllm_tiles] [train_tiles] [num_steps] [extra_args...]
#
# Tile layout: training on tiles 0..TRAIN_TILES-1, vLLM on tiles (12-VLLM_TILES)..11
# Default: 2 vLLM tiles (TP=2), 10 training tiles, 5 steps
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

PROJDIR="${TORCHTUNE_DIR}"
cd ${PROJDIR}

# Use 2025.2.0 — 2025.3.1 has broken XCCL allreduce (USM pointer validation)
module load frameworks/2025.2.0 2>/dev/null || true

# Remove user virtualenv from PATH so frameworks python is used
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# CXI fabric tuning (Slingshot 11)
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
# NOTE: XPU allocator ignores PYTORCH_ALLOC_CONF; expandable_segments
# incompatible with CCL. Leave unset.
unset PYTORCH_ALLOC_CONF

# Gemma4 vLLM overlay — provides usercustomize.py + model files
# IMPORTANT: overlay dir must come FIRST so its usercustomize.py is loaded
# (it includes all patches from _usercustomize_vllm plus Gemma4 registration)
GEMMA4_OVERLAY=${PROJDIR}/recipes/dev/vllm_gemma4_overlay
aurora_export_pythonpath "${GEMMA4_OVERLAY}" "${PROJDIR}" "${TRL_DIR}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

VLLM_TILES=${1:-2}
TRAIN_TILES=${2:-10}
NSTEPS=${3:-5}
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"
VLLM_PORT=8001
VLLM_LOG=/tmp/torchtune/vllm_gemma4_server.log
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/gemma-4-31B
CONFIG=recipes/configs/dev/production/gemma4_31B_grpo_server_xpu.yaml

echo "=== GRPO + vLLM (Gemma4) XPU: ${VLLM_TILES} vLLM tile(s), ${TRAIN_TILES} training tiles, ${NSTEPS} steps ==="
echo "Node: $(hostname), Date: $(date)"
echo "Python: $(which python3)"
echo "Model: ${MODEL_PATH}"
echo "Overlay: ${GEMMA4_OVERLAY}"

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

# Fix tokenizer for older transformers (4.57.x): extra_special_tokens list→dict
python3 ${PROJDIR}/recipes/dev/_fix_gemma4_tokenizer.py ${MODEL_PATH}/tokenizer_config.json

# vLLM tiles: last N tiles
VLLM_TILE_START=$((12 - VLLM_TILES))
VLLM_MASK=$(seq -s, ${VLLM_TILE_START} 11)

# --- 0. Warm vLLM model info cache ---
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
" 2>&1 | tail -5
echo ""

# --- 1. Launch vLLM server ---
echo "Starting vLLM server on tile(s) ${VLLM_MASK} (TP=${VLLM_TILES})..."
if [ ${VLLM_TILES} -gt 1 ]; then
    echo "Using vllm serve CLI (TP=${VLLM_TILES})"
    ZE_AFFINITY_MASK=${VLLM_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        PYTORCH_ALLOC_CONF= \
        CCL_PROCESS_LAUNCHER=none \
        CCL_ATL_TRANSPORT=ofi \
        FI_PROVIDER=cxi \
        CCL_KVS_IFACE=lo \
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
    echo "Using vllm serve CLI (TP=1)"
    ZE_AFFINITY_MASK=${VLLM_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        CCL_ATL_TRANSPORT=mpi \
        CCL_PROCESS_LAUNCHER=None \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --tensor-parallel-size ${VLLM_TILES} \
        --port ${VLLM_PORT} \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        > "${VLLM_LOG}" 2>&1 &
fi
VLLM_PID=$!

# Cleanup on exit
cleanup() {
    echo "Cleaning up vLLM server (PID ${VLLM_PID}) and child processes..."
    kill -- -${VLLM_PID} 2>/dev/null || true
    pkill -P ${VLLM_PID} 2>/dev/null || true
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
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
        echo "=== vLLM log (last 80 lines) ==="
        tail -80 "${VLLM_LOG}"
        exit 1
    fi
    # Show progress every 30s
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  Still waiting... (${ELAPSED}s)"
        tail -3 "${VLLM_LOG}" 2>/dev/null
    fi
done
echo "vLLM server ready (took ${ELAPSED}s)"

# Quick verification
echo "Testing vLLM generation..."
curl -s http://localhost:${VLLM_PORT}/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Model: {d[\"data\"][0][\"id\"]}')" 2>/dev/null || echo "(model list failed, proceeding anyway)"

# --- 3. Launch training ---
echo "Starting training on ${TRAIN_TILES} tiles (xpu:0 through xpu:$((TRAIN_TILES-1)))..."
python3 -m torch.distributed.run --standalone --nproc_per_node=${TRAIN_TILES} \
    ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${PROJDIR}/${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} ${EXTRA_ARGS}

echo "=== Training complete ==="

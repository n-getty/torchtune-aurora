#!/bin/bash
# Interactive test for Gemma4 26B-A4B vLLM GRPO with bmm_scatter fix in overlay.
# Run from a hold-node SSH session (not qsub — node is already allocated).
#
# Usage: bash recipes/dev/test_26b_vllm_bmm.sh [vllm_tiles] [train_tiles] [nsteps]
#   Defaults: vllm_tiles=2, train_tiles=10, nsteps=4

set -e
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

VLLM_TILES=${1:-2}
TRAIN_TILES=${2:-10}
NSTEPS=${3:-4}
VLLM_PORT=8001
VLLM_LOG=/lus/flare/projects/ModCon/ngetty/torchtune/vllm_26b_server_${VLLM_TILES}tp.log
VLLM_MAX_MODEL_LEN=2048
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
CONFIG=${PROJDIR}/recipes/configs/dev/production/gemma4_26b_a4b_grpo_server_xpu.yaml
GEMMA4_OVERLAY=${PROJDIR}/recipes/dev/vllm_gemma4_overlay

echo "=== 26B-A4B vLLM GRPO test (bmm fix) ==="
echo "vLLM tiles: ${VLLM_TILES} (TP=${VLLM_TILES}), training tiles: ${TRAIN_TILES}, steps: ${NSTEPS}"
echo "Node: $(hostname), Date: $(date)"

module load frameworks/2025.2.0 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
unset PYTORCH_ALLOC_CONF

export PYTHONPATH=${GEMMA4_OVERLAY}:${PROJDIR}:/flare/ModCon/ngetty/trl:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p /tmp/torchtune

# Stage model to /tmp if not already there
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

# Fix tokenizer
python3 ${PROJDIR}/recipes/dev/_fix_gemma4_tokenizer.py ${MODEL_PATH}/tokenizer_config.json

# vLLM on last VLLM_TILES tiles
TOTAL_TILES=12
VLLM_TILE_START=$((TOTAL_TILES - VLLM_TILES))
VLLM_MASK=$(seq -s, ${VLLM_TILE_START} $((TOTAL_TILES - 1)))

echo "Warming vLLM model info cache on tile ${VLLM_TILE_START}..."
ZE_AFFINITY_MASK=${VLLM_TILE_START} python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${MODEL_PATH}', tokenizer='${MODEL_PATH}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
" 2>&1 | tail -3

echo "Starting vLLM server on tiles ${VLLM_MASK} (TP=${VLLM_TILES})..."
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
    --gpu-memory-utilization 0.85 \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --distributed-executor-backend mp \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Cleaning up vLLM (PID ${VLLM_PID})..."
    kill -- -${VLLM_PID} 2>/dev/null || true
    pkill -P ${VLLM_PID} 2>/dev/null || true
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM on port ${VLLM_PORT}..."
ELAPSED=0
while ! curl -s http://localhost:${VLLM_PORT}/health/ > /dev/null 2>&1; do
    sleep 5; ELAPSED=$((ELAPSED + 5))
    if [ ${ELAPSED} -ge 600 ]; then
        echo "ERROR: vLLM did not start within 600s"; tail -50 "${VLLM_LOG}"; exit 1
    fi
    [ $((ELAPSED % 30)) -eq 0 ] && echo "  Waiting... ${ELAPSED}s" && tail -3 "${VLLM_LOG}" 2>/dev/null
done
echo "vLLM ready (${ELAPSED}s)"

echo "Starting training on ${TRAIN_TILES} tiles..."
ZE_AFFINITY_MASK=$(seq -s, 0 $((TRAIN_TILES - 1))) \
python3 -m torch.distributed.run --standalone --nproc_per_node=${TRAIN_TILES} \
    ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} \
    vllm_url=http://localhost:${VLLM_PORT}

echo "=== Test complete ==="

#!/bin/bash
# EP=4 / DP=3 GRPO benchmark on 12 tiles (Gemma4 26B-A4B)
# Runs on a hold node via SSH. vLLM must be running separately.
# Usage: bash recipes/dev/run_ep4_grpo_xpu.sh [num_steps] [vllm_node]

set -e
NSTEPS=${1:-5}
VLLM_NODE=${2:-localhost}
REPO=/lus/flare/projects/ModCon/ngetty/torchtune
TRAIN_TILES=12

cd $REPO

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')

GEMMA4_OVERLAY=${REPO}/recipes/dev/vllm_gemma4_overlay
export PYTHONPATH=${GEMMA4_OVERLAY}:${REPO}:/flare/ModCon/ngetty/trl:${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*"

export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export FI_CXI_RX_MATCH_MODE=hybrid
unset PYTORCH_ALLOC_CONF

MODEL_PATH=/tmp/torchtune/gemma-4-26B-A4B
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Staging model to ${MODEL_PATH}..."
    mkdir -p /tmp/torchtune
    cp -r /lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B "${MODEL_PATH}"
fi

CONFIG=${REPO}/recipes/configs/dev/experimental/gemma4_26b_grpo_ep4_xpu.yaml

echo "=== EP=4 DP=3 GRPO benchmark: ${TRAIN_TILES} tiles, ${NSTEPS} steps ==="
echo "Node: $(hostname), Date: $(date)"

MASTER_PORT=$((29800 + RANDOM % 200))
torchrun \
    --nproc_per_node=${TRAIN_TILES} \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    ${REPO}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} \
    vllm_url=http://${VLLM_NODE}:8001 \
    2>&1

echo "=== Done ==="

#!/bin/bash
# Quick test launcher for Gemma4 26B-A4B MoE GRPO on single node (12 tiles)
set -e
cd /lus/flare/projects/ModCon/ngetty/torchtune

module load frameworks/2025.3.1
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC=${1:-12}
NSTEPS=${2:-2}
MODEL_DIR=${3:-/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B}
CONFIG=${4:-recipes/configs/dev/production/gemma4_26b_a4b_grpo_novllm_xpu.yaml}

echo "=== Gemma4 26B-A4B MoE GRPO test: ${NPROC} tiles, ${NSTEPS} steps ==="
echo "Node: $(hostname), Date: $(date)"

python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config $CONFIG \
    num_steps=$NSTEPS \
    base_model_path=$MODEL_DIR

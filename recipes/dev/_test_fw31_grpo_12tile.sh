#!/bin/bash
# Full-node GRPO test on frameworks/2025.3.1
# 12 tiles, 2 steps — validates FSDP2+XCCL at production scale
set -e
cd /lus/flare/projects/ModCon/ngetty/torchtune

module load frameworks/2025.3.1 2>/dev/null || true
unset PYTORCH_ALLOC_CONF
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
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC=${1:-12}
NSTEPS=${2:-2}

echo "=== GRPO ${NPROC}-tile test: frameworks/2025.3.1 ==="
echo "Node: $(hostname), Date: $(date)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "PYTORCH_ALLOC_CONF: ${PYTORCH_ALLOC_CONF:-unset}"

python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config recipes/configs/dev/baseline/qwen3B_grpo_xpu_baseline.yaml \
    num_steps=$NSTEPS

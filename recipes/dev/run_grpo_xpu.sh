#!/bin/bash
# Quick launcher for GRPO XPU baseline test
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune

# Load Aurora frameworks module (provides XPU-enabled PyTorch 2.10 + python 3.12)
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH so frameworks python is used
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
# Also unset VIRTUAL_ENV so pip/site-packages don't pollute
unset VIRTUAL_ENV

# CCL / XPU environment
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# CXI fabric tuning (Slingshot 11) — from PRISM production config
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
# XPU memory allocator
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
# NOTE: CCL_ALLREDUCE=ring / CCL_REDUCE_SCATTER=ring are NOT set for
# single-node FSDP2 — they force the scheduler path which doesn't support
# ReduceOp.AVG. These are only needed for multi-node with large tensors.
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC=${1:-2}
NSTEPS=${2:-3}
TILE_START=${3:-0}

echo "=== GRPO XPU baseline: ${NPROC} tiles, ${NSTEPS} steps (tile_start=${TILE_START}) ==="
echo "Node: $(hostname), Date: $(date)"
echo "Python: $(which python3)"

# Offset tile assignment so training ranks start at TILE_START
export VLLM_TILE_OFFSET=${TILE_START}

python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config recipes/configs/dev/baseline/qwen3B_grpo_xpu_baseline.yaml \
    num_steps=$NSTEPS

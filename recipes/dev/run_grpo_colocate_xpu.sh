#!/bin/bash
# GRPO with colocated vLLM on Aurora XPU
# Each rank runs its own in-process vLLM engine — no separate server needed.
# All tiles used for both training and generation.
#
# Usage:
#   bash recipes/dev/run_grpo_colocate_xpu.sh [num_tiles] [model_path] [num_steps] [config_file]
#
# Default: 12 tiles, Qwen2.5-3B, 10 steps, Config A
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
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# Increase CCL handle cache for large models (32B has 707 params × N all-gathers)
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=8192
# CXI fabric tuning (Slingshot 11) — from PRISM production config
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
# XPU memory allocator
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
# Disable torch.compile globally (vLLM not viable on XPU with compile).
# The recipe will temporarily unset this when it needs to compile the training model.
export TORCH_COMPILE_DISABLE=1
# vLLM V1 multiprocessing: set inside recipe on rank 0 only.
# Do NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0 globally — it's only needed
# for the rank that creates the vLLM engine.
# Spawn method for vLLM workers (if TP > 1)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# usercustomize patch: disable transformers version check (hf-hub 1.7 vs <1.0)
VLLM_CUSTOMIZATION=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/_usercustomize_vllm
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC=${1:-12}
MODEL_PATH=${2:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B}
NSTEPS=${3:-10}
CONFIG=${4:-recipes/configs/dev/production/qwen3B_grpo_colocate_xpu.yaml}
# Shift past positional args to get EXTRA_ARGS
_nargs=$#
if [ $_nargs -ge 4 ]; then shift 4; else shift $_nargs; fi
EXTRA_ARGS="$@"  # additional config overrides

echo "=== GRPO + Colocated vLLM XPU: ${NPROC} tiles, ${NSTEPS} steps, config=${CONFIG} ==="
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

# --- 0. Warm vLLM model info cache ---
# vLLM 0.15 inspects model architectures via subprocess, which segfaults on XPU.
# Pre-warming the cache on a scratch tile avoids this.
SCRATCH_TILE=$((NPROC - 1))
echo "Warming vLLM model info cache on scratch tile ${SCRATCH_TILE}..."
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

# --- 1. Launch training (each rank creates its own colocated vLLM engine) ---
# Training uses device_id=xpu:{LOCAL_RANK} with NO ZE_AFFINITY_MASK.
# CCL needs full device visibility for allreduce. Each rank's vLLM engine
# uses the device set by torch.xpu.set_device(LOCAL_RANK).
echo "Starting colocated GRPO training on ${NPROC} tiles..."
python3 -m torch.distributed.run --standalone --nproc_per_node=${NPROC} \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} ${EXTRA_ARGS}

echo "=== Training complete ==="

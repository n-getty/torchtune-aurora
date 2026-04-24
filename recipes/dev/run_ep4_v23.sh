#!/bin/bash
# EP=4/DP=3 GRPO no-vLLM correctness test — v23
# Fix: expert_cpu_offload=True — keep EP-sharded expert param shards and
#      grad shards on CPU. Materialized on XPU only during FSDP2 AllGather.
#      Saves ~2.86 GiB XPU HBM (accumulated expert grad buffers during backward).
#      Fixes UR:40 OOM at experts.py:82 during activation checkpoint recompute.
#
# empty_cache() placement: INSIDE grpo_step, before first backward chunk,
#      after all training forward chunks. Releases inference + training-fwd
#      reserved-but-freed blocks back to L0 in a single call.

set -eo pipefail

module load frameworks/2025.2.0

REPO=/lus/flare/projects/ModCon/ngetty/torchtune
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.2.0/lib/python3.10/site-packages
FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages

# PYTHONNOUSERSITE=1 prevents ~/.local/lib/python3.10/site-packages (plain)
# from prepending torchao-0.16.0. Explicit PYTHONPATH: FW_SITE first so
# torchao-0.12.0 wins; LOCAL_SITE (frameworks-specific) restores math_verify.
export PYTHONNOUSERSITE=1
export PYTHONPATH=${REPO}:${FW_SITE}:${LOCAL_SITE}:${PYTHONPATH:-}

# CCL / transport config
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_OP_SYNC=1
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring

# ZE / Level Zero
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

CONFIG=${REPO}/recipes/configs/dev/experimental/gemma4_26b_grpo_ep4_nollm_xpu.yaml
NNODES=${NNODES:-1}
NPROC=12

echo "=== EP4 GRPO v23 — expert_cpu_offload=True ==="
echo "Config: ${CONFIG}"
echo "Nodes: ${NNODES}, Procs: ${NPROC}"
python3 -c "import torchao; print('torchao:', torchao.__version__)"
python3 -c "import math_verify; print('math_verify:', math_verify.__file__)"

# Stage model to /tmp if not already there
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
MODEL_DST=/tmp/torchtune/gemma-4-26B-A4B
if [ ! -f "${MODEL_DST}/model-00001-of-00002.safetensors" ]; then
    echo "Staging model from ${MODEL_SRC} to ${MODEL_DST}..."
    mkdir -p ${MODEL_DST}
    cp ${MODEL_SRC}/model-00001-of-00002.safetensors ${MODEL_DST}/
    cp ${MODEL_SRC}/model-00002-of-00002.safetensors ${MODEL_DST}/
    cp ${MODEL_SRC}/config.json ${MODEL_DST}/
    echo "Model staged."
else
    echo "Model already staged at ${MODEL_DST}."
fi

torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=${NNODES} \
    --master_addr=localhost \
    --master_port=29500 \
    ${REPO}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    2>&1

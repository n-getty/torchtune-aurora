#!/bin/bash
# EP=1 / DP=12 GRPO baseline — no expert parallelism
# Use to compare grpo_time against EP=4's 74s.

set -eo pipefail

module load frameworks/2025.2.0

REPO=/lus/flare/projects/ModCon/ngetty/torchtune
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.2.0/lib/python3.10/site-packages
FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages

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

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

CONFIG=${REPO}/recipes/configs/dev/experimental/gemma4_26b_grpo_ep1_nollm_xpu.yaml
NNODES=${NNODES:-1}
NPROC=12

echo "=== EP1 GRPO baseline ==="
echo "Config: ${CONFIG}"
python3 -c "import torchao; print('torchao:', torchao.__version__)"

# Model already staged to /tmp from EP=4 runs
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
MODEL_DST=/tmp/torchtune/gemma-4-26B-A4B
if [ ! -f "${MODEL_DST}/model-00001-of-00002.safetensors" ]; then
    echo "Staging model..."
    mkdir -p ${MODEL_DST}
    cp ${MODEL_SRC}/model-00001-of-00002.safetensors ${MODEL_DST}/
    cp ${MODEL_SRC}/model-00002-of-00002.safetensors ${MODEL_DST}/
    cp ${MODEL_SRC}/config.json ${MODEL_DST}/
fi

torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=${NNODES} \
    --master_addr=localhost \
    --master_port=29501 \
    ${REPO}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    2>&1

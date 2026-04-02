#!/bin/bash
# vLLM server launched as an mpiexec rank.
# Inherits CXI service from PBS job allocation.
#
# Usage: launched via mpiexec MPMD colon syntax
#   mpiexec ... -n 20 -ppn 10 train_wrapper.sh : -n 1 -ppn 1 vllm_mpiexec_rank.sh
#

set -e

module load frameworks 2>/dev/null || true

# Strip myenv, re-set paths
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV 2>/dev/null || true

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
export PYTHONPATH="${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# GPU: use tile 11 (last tile on node 0, not used by training)
export ZE_AFFINITY_MASK=${VLLM_TILE:-11}
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1

# CCL: force vLLM to use MPI transport (not OFI/CXI) to avoid
# CXI fabric contention with training ranks.
# vLLM TP=1 only needs a local PG, so MPI transport is sufficient.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=mpi
unset FI_PROVIDER
unset CCL_KVS_IFACE

MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/Qwen2.5-3B}
VLLM_PORT=${VLLM_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}

echo "[vLLM rank] Starting vLLM server on $(hostname) tile ${VLLM_TILE:-11} port ${VLLM_PORT}"
echo "[vLLM rank] PMI_RANK=${PMI_RANK:-<unset>} PMI_SIZE=${PMI_SIZE:-<unset>}"
echo "[vLLM rank] CXI env: $(env | grep CXI_SERVICE 2>/dev/null || echo 'no CXI_SERVICE_ID')"

# Warm model info cache
python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${MODEL_PATH}', tokenizer='${MODEL_PATH}', dtype='bfloat16', enforce_eager=True)
print('[vLLM rank] Cache warmed')
" 2>&1

# Start server (blocking — runs until killed)
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}"

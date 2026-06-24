#!/bin/bash
# Launch vanilla vLLM HTTP API server on VLLM_NODE for BioReason prompt_embeds prototype.
# Run via: ssh ${VLLM_NODE} "bash <this-script>"
set -o pipefail

FW_BIN=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin
export PATH=${FW_BIN}:$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo

MODEL=/tmp/torchtune/bioreason-pro-sft
PORT=${PORT:-8001}
TP=${TP:-1}
LOG=/tmp/torchtune/vllm_http_proto.log

mkdir -p /tmp/torchtune

# Pre-launch cleanup
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
sleep 2
rm -f /dev/shm/vllm* 2>/dev/null || true

echo "=== Launching vLLM HTTP server: model=${MODEL} TP=${TP} PORT=${PORT} ==="
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size ${TP} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 2048 \
    --enable-prompt-embeds \
    --distributed-executor-backend mp \
    > "${LOG}" 2>&1

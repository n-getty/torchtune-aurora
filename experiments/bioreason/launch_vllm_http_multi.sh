#!/bin/bash
# Spawn N vLLM HTTP servers on this node, one per tile (TILE 0..N-1, ports 8001..8001+N-1).
# Used for Phase 2 multi-replica DP validation.
#
# Usage: bash launch_vllm_http_multi.sh <N>
set -o pipefail

N=${1:-2}

FW_BIN=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin
export PATH=${FW_BIN}:$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
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

mkdir -p /tmp/torchtune

# Pre-launch cleanup
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
sleep 2
rm -f /dev/shm/vllm* 2>/dev/null || true

PIDS=()
for i in $(seq 0 $((N-1))); do
    PORT=$((8001 + i))
    LOG=/tmp/torchtune/vllm_http_tile${i}.log
    echo "=== Launching vLLM tile ${i} on port ${PORT} ==="
    ZE_AFFINITY_MASK=${i} python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size 1 \
        --port ${PORT} \
        --host 0.0.0.0 \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.70 \
        --max-model-len 2048 \
        --enable-prompt-embeds \
        --distributed-executor-backend mp \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Logs: /tmp/torchtune/vllm_http_tile{0..$((N-1))}.log"

# Wait for any to exit
wait

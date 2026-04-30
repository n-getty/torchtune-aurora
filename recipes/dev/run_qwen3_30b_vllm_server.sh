#!/bin/bash
# Dedicated vLLM server for Qwen3-30B-A3B MoE — runs on a separate node.
# Mirrors run_vllm_server_26b.sh but: (1) uses Qwen3 model, (2) no Gemma4 tokenizer
# fixup, (3) frameworks/2025.3.1 (matches qwen3_moe/run_grpo_e2e.sh which validates
# vLLM-native Qwen3-30B-A3B serving on XPU).
#
# 3 independent TP=4 servers on tile groups (0-3, 4-7, 8-11), one port each.
# Recipe dispatches across all 3 URLs via ThreadPoolExecutor.
#
# Run this FIRST on the vLLM node, then run_qwen3_30b_ep4_vllm_2node.sh on training node.
#
# Usage: bash recipes/dev/run_qwen3_30b_vllm_server.sh [base_port]

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset PYTORCH_ALLOC_CONF

# Avoid user-site shadowing framework transformers/huggingface-hub
# (frameworks/2025.3.1 transformers requires huggingface-hub<1.0; user-site has 1.7.1)
FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

VLLM_BASE_PORT=${1:-8001}
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
VLLM_MAX_MODEL_LEN=2048
TP_SIZE=4
N_REPLICAS=3

echo "=== vLLM Server (Qwen3-30B-A3B MoE) — ${N_REPLICAS}×TP=${TP_SIZE} ==="
echo "Node: $(hostname), Date: $(date)"
echo "Model: ${MODEL_PATH}"
echo "Ports: ${VLLM_BASE_PORT} to $((VLLM_BASE_PORT + N_REPLICAS - 1))"

mkdir -p /tmp/torchtune

LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi

# Warm model info cache once
echo "Warming vLLM model info cache..."
ZE_AFFINITY_MASK=0 python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${LOCAL_MODEL}', tokenizer='${LOCAL_MODEL}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
" 2>&1 | tail -2

VLLM_PIDS=()
for r in $(seq 0 $((N_REPLICAS - 1))); do
    PORT=$((VLLM_BASE_PORT + r))
    TILE_START=$((r * TP_SIZE))
    TILE_END=$((TILE_START + TP_SIZE - 1))
    TILE_MASK=$(seq -s, ${TILE_START} ${TILE_END})
    VLLM_LOG=/tmp/torchtune/vllm_qwen3_30b_replica_${r}.log

    echo "Starting replica ${r}: tiles ${TILE_MASK}, port ${PORT}, log ${VLLM_LOG}"
    ZE_AFFINITY_MASK=${TILE_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        PYTORCH_ALLOC_CONF= \
        CCL_PROCESS_LAUNCHER=none \
        CCL_ATL_TRANSPORT=ofi \
        FI_PROVIDER=cxi \
        CCL_KVS_IFACE=lo \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "${LOCAL_MODEL}" \
        --tensor-parallel-size ${TP_SIZE} \
        --port ${PORT} \
        --host 0.0.0.0 \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.80 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --distributed-executor-backend mp \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PIDS+=($!)
    echo "  Replica ${r} PID: ${VLLM_PIDS[-1]}"
done

echo "All ${N_REPLICAS} replicas launched. Waiting for health checks..."

for r in $(seq 0 $((N_REPLICAS - 1))); do
    PORT=$((VLLM_BASE_PORT + r))
    ELAPSED=0
    while ! curl -s http://localhost:${PORT}/health/ > /dev/null 2>&1; do
        sleep 10; ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge 600 ]; then
            echo "ERROR: vLLM replica ${r} on port ${PORT} did not respond within 600s"
            tail -20 /tmp/torchtune/vllm_qwen3_30b_replica_${r}.log
            exit 1
        fi
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  Waiting for replica ${r}... ${ELAPSED}s"
    done
    echo "Replica ${r} ready on port ${PORT} (waited ${ELAPSED}s)"
done

echo "=== All vLLM replicas ready. Ports: ${VLLM_BASE_PORT}-$((VLLM_BASE_PORT + N_REPLICAS - 1)) ==="
echo "vLLM server running. Ctrl+C or job exit to stop."

wait "${VLLM_PIDS[@]}"
echo "vLLM server exited."

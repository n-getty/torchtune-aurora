#!/bin/bash
# Dedicated vLLM server for 26B-A4B MoE — runs on a separate node.
# Launches 3 separate TP=4 processes on 3 tile groups (0-3, 4-7, 8-11),
# each on its own port (VLLM_BASE_PORT, +1, +2).
#
# This mirrors the Aurora reference: debug_dp_inter_gpt.sh (3× DP replicas,
# each a separate vllm serve process with --data-parallel-size 3 --data-parallel-rank N)
#
# Recipe uses all 3 URLs (comma-separated) via ThreadPoolExecutor for parallel generation.
# 4 grpo_samples split across 3 servers → ~3× faster generation vs single TP=4 server.
#
# Run this FIRST on the vLLM node, then run_ep4_vllm_2node.sh on the training node.
# Leave running until training is complete.
#
# Usage: bash recipes/dev/run_vllm_server_26b.sh [base_port]
#   base_port: first vLLM port (default 8001); replicas use base_port+0, +1, +2

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset PYTORCH_ALLOC_CONF

GEMMA4_OVERLAY=${PROJDIR}/recipes/dev/vllm_gemma4_overlay
export PYTHONPATH=${GEMMA4_OVERLAY}:${PROJDIR}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

VLLM_BASE_PORT=${1:-8001}
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
VLLM_MAX_MODEL_LEN=1024
TP_SIZE=4
N_REPLICAS=3   # 3 independent TP=4 servers, each on 4 tiles (tiles 0-3, 4-7, 8-11)

echo "=== vLLM Server (26B-A4B MoE) — ${N_REPLICAS}×TP=${TP_SIZE} ==="
echo "Node: $(hostname), Date: $(date)"
echo "Model: ${MODEL_PATH}"
echo "Ports: ${VLLM_BASE_PORT} to $((VLLM_BASE_PORT + N_REPLICAS - 1))"

mkdir -p /tmp/torchtune

# Stage model to /tmp
LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi

# Fix tokenizer
python3 ${PROJDIR}/recipes/dev/_fix_gemma4_tokenizer.py ${LOCAL_MODEL}/tokenizer_config.json

# Warm model info cache once (on tile 0)
echo "Warming vLLM model info cache..."
ZE_AFFINITY_MASK=0 python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${LOCAL_MODEL}', tokenizer='${LOCAL_MODEL}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
" 2>&1 | tail -2

# Launch N_REPLICAS independent TP=4 vLLM servers on tile groups 0-3, 4-7, 8-11.
# Each is a fully independent server (no --data-parallel-* coordination flags).
# The training recipe dispatches prompts across all 3 servers via ThreadPoolExecutor.
VLLM_PIDS=()
for r in $(seq 0 $((N_REPLICAS - 1))); do
    PORT=$((VLLM_BASE_PORT + r))
    TILE_START=$((r * TP_SIZE))
    TILE_END=$((TILE_START + TP_SIZE - 1))
    TILE_MASK=$(seq -s, ${TILE_START} ${TILE_END})
    VLLM_LOG=/tmp/torchtune/vllm_26b_replica_${r}.log

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
        --gpu-memory-utilization 0.6 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --distributed-executor-backend mp \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PIDS+=($!)
    echo "  Replica ${r} PID: ${VLLM_PIDS[-1]}"
done

echo "All ${N_REPLICAS} replicas launched. Waiting for health checks..."

# Wait for all replicas to be healthy
for r in $(seq 0 $((N_REPLICAS - 1))); do
    PORT=$((VLLM_BASE_PORT + r))
    ELAPSED=0
    while ! curl -s http://localhost:${PORT}/health/ > /dev/null 2>&1; do
        sleep 10; ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge 600 ]; then
            echo "ERROR: vLLM replica ${r} on port ${PORT} did not respond within 600s"
            tail -20 /tmp/torchtune/vllm_26b_replica_${r}.log
            exit 1
        fi
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  Waiting for replica ${r}... ${ELAPSED}s"
    done
    echo "Replica ${r} ready on port ${PORT} (waited ${ELAPSED}s)"
done

echo "=== All vLLM replicas ready. Ports: ${VLLM_BASE_PORT}-$((VLLM_BASE_PORT + N_REPLICAS - 1)) ==="
echo "vLLM server running. Ctrl+C or job exit to stop."

# Wait for all replica processes
wait "${VLLM_PIDS[@]}"
echo "vLLM server exited."

#!/bin/bash
# Test vLLM Gemma4 overlay server startup
set -e

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}
module load frameworks/2025.2.0 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

GEMMA4_OVERLAY=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay
export PYTHONPATH=${GEMMA4_OVERLAY}:/lus/flare/projects/ModCon/ngetty/torchtune:$PYTHONPATH
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

MODEL_PATH=/tmp/torchtune/gemma-4-31B
VLLM_PORT=8001
VLLM_LOG=/tmp/torchtune/vllm_gemma4_server.log

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Fix tokenizer for older transformers
python3 ${PROJDIR}/recipes/dev/_fix_gemma4_tokenizer.py ${MODEL_PATH}/tokenizer_config.json 2>/dev/null || true

echo "Launching vLLM server with Gemma4 overlay (TP=2, tiles 10-11)..."
ZE_AFFINITY_MASK=10,11 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_ALLOC_CONF= \
    CCL_KVS_IFACE=lo \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size 2 \
    --port ${VLLM_PORT} \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --distributed-executor-backend mp \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for server startup..."

# Wait up to 5 minutes for health check
ELAPSED=0
while ! curl -s http://localhost:${VLLM_PORT}/health/ > /dev/null 2>&1; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ ${ELAPSED} -ge 300 ]; then
        echo "ERROR: vLLM did not start within 300s"
        echo "=== Last 80 lines of log ==="
        tail -80 "${VLLM_LOG}"
        kill ${VLLM_PID} 2>/dev/null || true
        exit 1
    fi
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  Still waiting... (${ELAPSED}s)"
        tail -3 "${VLLM_LOG}" 2>/dev/null
    fi
done
echo "vLLM server ready (took ${ELAPSED}s)"

# Test model listing
echo "=== Model list ==="
curl -s http://localhost:${VLLM_PORT}/v1/models | python3 -m json.tool 2>/dev/null || echo "(model list failed)"

# Test generation
echo ""
echo "=== Test generation ==="
RESP=$(curl -s -X POST http://localhost:${VLLM_PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/tmp/torchtune/gemma-4-31B","prompt":"The capital of France is","max_tokens":16,"temperature":0.7}')
echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "Raw: ${RESP}"

echo ""
echo "=== Server running, PID=${VLLM_PID} ==="
echo "To stop: kill ${VLLM_PID}"

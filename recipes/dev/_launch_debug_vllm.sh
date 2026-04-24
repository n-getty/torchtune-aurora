#!/bin/bash
# Launch vLLM Gemma4 server for debugging
module load frameworks/2025.2.0 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
export PYTHONPATH=${PROJDIR}/recipes/dev/vllm_gemma4_overlay:${PROJDIR}:$PYTHONPATH
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_CXI_RX_MATCH_MODE=hybrid

pkill -f "vllm.entrypoints" 2>/dev/null
sleep 1

TILES=${1:-"4,5"}
PORT=${2:-8001}
LOG=/tmp/torchtune/vllm_debug_${TILES//,/_}_p${PORT}.log

echo "Starting vLLM on tiles ${TILES}, port ${PORT}"
echo "Log: ${LOG}"

ZE_AFFINITY_MASK=${TILES} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_ALLOC_CONF= \
    CCL_KVS_IFACE=lo \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /tmp/torchtune/gemma-4-31B \
    --tensor-parallel-size 2 \
    --port ${PORT} \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 2048 \
    --distributed-executor-backend mp \
    > "${LOG}" 2>&1 &
PID=$!
echo "Server PID: ${PID}"

# Wait for health check
ELAPSED=0
while ! curl -s http://localhost:${PORT}/health/ > /dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ ${ELAPSED} -ge 300 ]; then
        echo "ERROR: Server did not start in 300s"
        tail -50 "${LOG}"
        exit 1
    fi
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  Waiting... (${ELAPSED}s)"
    fi
done
echo "Server ready (${ELAPSED}s)"

# Test generation
echo ""
echo "=== Test generation ==="
RESP=$(curl -s -X POST http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/tmp/torchtune/gemma-4-31B","prompt":"The capital of France is","max_tokens":32,"temperature":0}')
echo "${RESP}" | python3 -m json.tool 2>/dev/null || echo "Raw: ${RESP}"

echo ""
echo "=== Debug output from model ==="
grep -E "\[layer=|\[local_attn\]|\[k_eq_v" "${LOG}" | head -80

echo ""
echo "=== Server running, PID=${PID} ==="
echo "Kill: kill ${PID}; pkill -P ${PID}"

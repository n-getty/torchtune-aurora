#!/bin/bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
echo "=== Worker TP0 errors ==="
grep "VllmWorker TP0" /tmp/torchtune/vllm_gemma4_server.log | grep "ERROR" | tail -25
echo ""
echo "=== Worker TP1 errors ==="
grep "VllmWorker TP1" /tmp/torchtune/vllm_gemma4_server.log | grep "ERROR" | tail -25
echo ""
echo "=== EngineCore errors ==="
grep "EngineCore_0" /tmp/torchtune/vllm_gemma4_server.log | grep "ERROR" | tail -5

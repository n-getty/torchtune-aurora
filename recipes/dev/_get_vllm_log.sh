#!/bin/bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
echo "=== LOG LINES: $(wc -l < /tmp/torchtune/vllm_gemma4_server.log) ==="
# Get the EngineCore error section
grep -n "ERROR\|Traceback\|ValueError\|RuntimeError\|ImportError\|AttributeError\|TypeError\|KeyError" /tmp/torchtune/vllm_gemma4_server.log | tail -30
echo ""
echo "=== WORKER ERRORS ==="
grep "VllmWorker\|EngineCore" /tmp/torchtune/vllm_gemma4_server.log | grep -i "error\|exception\|traceback\|value\|type\|key\|import\|attribute" | tail -30

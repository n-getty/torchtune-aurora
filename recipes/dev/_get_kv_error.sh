#!/bin/bash
pkill -f "vllm.entrypoints" 2>/dev/null || true
grep "EngineCore_0" /tmp/torchtune/vllm_gemma4_server.log | grep "ERROR" | head -40

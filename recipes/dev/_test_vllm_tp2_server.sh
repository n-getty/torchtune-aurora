#!/bin/bash
# Test vLLM TP=2 as a server (matching working Aurora-Inferencing pattern)
set -e

module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# Environment from working scripts
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${1:-10,11}
export CCL_PROCESS_LAUNCHER=None
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FI_MR_CACHE_MONITOR=userfaultfd
export OMP_NUM_THREADS=16
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/_usercustomize_vllm:$PYTHONPATH

TP=${2:-2}
MODEL=${3:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B}
PORT=8099

echo "=== vLLM TP=${TP} server test on tiles ${ZE_AFFINITY_MASK} ==="

# Warm cache
python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${MODEL}', tokenizer='${MODEL}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
" 2>&1 | tail -2

# Start server
echo "Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size ${TP} \
    --port ${PORT} \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --distributed-executor-backend mp \
    &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# Wait for ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${PORT}/health/ > /dev/null 2>&1; then
        echo "Server ready in ${i}s"
        break
    fi
    sleep 2
    if [ $i -eq 60 ]; then
        echo "TIMEOUT waiting for server"
        exit 1
    fi
done

# Benchmark: 16 prompts (simulating GRPO Config B batch)
echo ""
echo "--- Benchmark: 16 prompts, max_tokens=256 ---"
python3 -c "
import requests, time, json

prompts = [
    'Solve: What is 15 * 23? Show your work step by step.',
    'Calculate the sum of all prime numbers less than 50.',
    'If a train travels 120 miles in 2 hours, what is its average speed?',
    'What is the derivative of x^3 + 2x^2 - 5x + 3?',
] * 4  # 16 prompts

t0 = time.time()
total_tokens = 0
for p in prompts:
    r = requests.post('http://localhost:${PORT}/v1/completions', json={
        'model': '${MODEL}',
        'prompt': p,
        'max_tokens': 256,
        'temperature': 0.7,
    })
    data = r.json()
    tokens = data['usage']['completion_tokens']
    total_tokens += tokens

elapsed = time.time() - t0
print(f'Total tokens: {total_tokens}')
print(f'Time: {elapsed:.2f}s')
print(f'Throughput: {total_tokens/elapsed:.1f} tok/s')
print(f'Avg response: {total_tokens/16:.0f} tokens')
"

echo ""
echo "--- Benchmark: 16 concurrent prompts, max_tokens=512 ---"
python3 -c "
import requests, time, json
from concurrent.futures import ThreadPoolExecutor

prompts = [
    'Solve: What is 15 * 23? Show your work step by step.',
    'Calculate the sum of all prime numbers less than 50.',
    'If a train travels 120 miles in 2 hours, what is its average speed?',
    'What is the derivative of x^3 + 2x^2 - 5x + 3?',
] * 4  # 16 prompts

def gen(p):
    r = requests.post('http://localhost:${PORT}/v1/completions', json={
        'model': '${MODEL}',
        'prompt': p,
        'max_tokens': 512,
        'temperature': 0.7,
    })
    return r.json()['usage']['completion_tokens']

t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    tokens = list(ex.map(gen, prompts))
elapsed = time.time() - t0
total = sum(tokens)
print(f'Total tokens: {total}')
print(f'Time: {elapsed:.2f}s')
print(f'Throughput: {total/elapsed:.1f} tok/s (concurrent)')
print(f'Avg response: {total/16:.0f} tokens')
"

echo ""
echo "=== TP=${TP} server test complete ==="

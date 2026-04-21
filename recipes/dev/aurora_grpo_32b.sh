#!/bin/bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=flare
#PBS -A ModCon
#PBS -N grpo_32b_opt
#PBS -j oe
#PBS -o logs/grpo_32b_opt.out

# 32B GRPO optimization: test TP=2 and TP=4 with batched generation
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

echo "=============================================="
echo "Test 1: TP=2 vLLM (2 tiles) + 10 training tiles"
echo "=============================================="
bash recipes/dev/run_grpo_vllm_xpu.sh 2 10 \
    /lus/flare/projects/ModCon/ngetty/models/Qwen3-32B 5 \
    recipes/configs/dev/production/qwen32B_grpo_server_xpu.yaml

# Clean up between runs
sleep 5
bash recipes/dev/clean_tiles.sh 2>/dev/null | tail -1

echo ""
echo "=============================================="
echo "Test 2: TP=4 vLLM (4 tiles) + 8 training tiles"
echo "=============================================="
bash recipes/dev/run_grpo_vllm_xpu.sh 4 8 \
    /lus/flare/projects/ModCon/ngetty/models/Qwen3-32B 5 \
    recipes/configs/dev/production/qwen32B_grpo_server_xpu.yaml

echo "=== All 32B optimization tests complete ==="

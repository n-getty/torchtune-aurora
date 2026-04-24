#!/bin/bash
#PBS -l select=5
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/test_72b_5node.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/test_72b_5node.err
#PBS -N test_72b_5n
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune
mkdir -p logs /tmp/torchtune

echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo "Start: $(date)"

# ============================================================
# 72B no-offload on 5 nodes (1 vLLM + 4 training = 48 tiles)
# 36 tiles OOM (50.5 GiB peak vs 48 GiB). 48 tiles = 25% less per tile.
# ============================================================
echo "============================================================"
echo "  72B no-offload fbs=2, 5 nodes (48 training tiles)"
echo "  36 tiles: OOM (50.5 GiB peak). 48 tiles: ~25 GiB estimated."
echo "  Baseline: 84.6s (24 tiles, CPU offload, fbs=1)"
echo "============================================================"

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload_5node.yaml \
MIN_NODES=5 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=600 \
NSTEPS=5 \
GRPO_SAMPLES=4 \
bash recipes/dev/aurora_grpo_dedicated_vllm_generic.sh 2>&1

echo ""
echo "============================================================"
echo "  TEST COMPLETE"
echo "============================================================"
echo "End: $(date)"

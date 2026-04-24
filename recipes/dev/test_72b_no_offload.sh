#!/bin/bash
#PBS -l select=4
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/test_72b_no_offload.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/test_72b_no_offload.err
#PBS -N test_72b_nooff
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune
mkdir -p logs /tmp/torchtune

echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo "Start: $(date)"

# ============================================================
# 72B no-offload fbs=2 with aggressive empty_cache
# Previous: OOM because no-op empty_cache left 28+ GiB cached.
# Fix: Real empty_cache between chunks, between forward phases,
# and between steps. FSDP is idle at these boundaries.
# ============================================================
echo "============================================================"
echo "  72B no-offload fbs=2 (aggressive empty_cache, 36 tiles)"
echo "  Baseline: 84.6s (CPU offload)"
echo "============================================================"

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload_fbs2.yaml \
MIN_NODES=4 \
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

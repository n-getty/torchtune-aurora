#!/bin/bash
#PBS -l select=4
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/test_72b_fbs2.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/test_72b_fbs2.err
#PBS -N test_72b_fbs2
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune
mkdir -p logs

echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo "Start: $(date)"

# 72B no-offload with fbs=2 (3 training nodes + 1 vLLM node)
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload_fbs2.yaml \
MIN_NODES=4 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=600 \
NSTEPS=3 \
GRPO_SAMPLES=4 \
bash recipes/dev/aurora_grpo_dedicated_vllm_generic.sh 2>&1

echo "End: $(date)"

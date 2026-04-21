#!/bin/bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -N grpo_qwen3b_learn
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/grpo_qwen3b_learning_run.out

# GRPO learning run — Qwen2.5-3B on GSM8K with periodic eval
# Fast iteration: G=16, fbs=16, max_gen=512, vLLM DP=2
# ~5s/step, 200 steps + 10 evals ≈ 35 min training + 10 min startup

cd /lus/flare/projects/ModCon/ngetty/torchtune

mkdir -p logs results/qwen3b_learning_run

VLLM_DP=2 bash recipes/dev/run_grpo_vllm_xpu.sh 2 10 \
    /lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B 200 \
    recipes/configs/dev/production/qwen3B_grpo_learning_run.yaml

echo "=== Qwen2.5-3B learning run complete ==="
echo "Results at: results/qwen3b_learning_run/logs/"

#!/bin/bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -N grpo_qwen32b_learn
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/grpo_qwen32b_learning_run.out

# GRPO learning run — Qwen3-32B on GSM8K with periodic eval
# ~31s/step (max_gen=256, G=8), 60 steps + 3 evals ≈ 45 min training
# Plus ~10 min startup = ~55 min total

cd /lus/flare/projects/ModCon/ngetty/torchtune

mkdir -p logs results/qwen32b_learning_run

bash recipes/dev/run_grpo_vllm_xpu.sh 2 10 \
    /lus/flare/projects/ModCon/ngetty/models/Qwen3-32B 60 \
    recipes/configs/dev/production/qwen32B_grpo_learning_run.yaml

echo "=== Qwen3-32B learning run complete ==="
echo "Results at: results/qwen32b_learning_run/logs/"

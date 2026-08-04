#!/usr/bin/env bash
#PBS -N kimi_k3_stage_only
#PBS -l walltime=00:10:00
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/stage_only_capacity.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/stage_only_capacity.err

set -euo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/torchtune
MODEL=${MODEL:-/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B}
LOG_DIR=${LOG_DIR:-$ROOT/experiments/kimi_k3_serving/logs/stage_only_${PBS_JOBID}}
STAGE_ROOT=${STAGE_ROOT:-/tmp/kimi_k3_stage_${PBS_JOBID}}

mkdir -p "$LOG_DIR"
echo "job=$PBS_JOBID nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')" | tee "$LOG_DIR/metadata"
LOG_DIR="$LOG_DIR" STAGE_ROOT="$STAGE_ROOT" \
    exec "$ROOT/experiments/kimi_k3_serving/serve_k3.sh" \
        --model "$MODEL" --stage-only --stage-model

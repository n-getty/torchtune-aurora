#!/usr/bin/env bash
#PBS -N kimi_k3_hold
#PBS -l walltime=01:00:00
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold.err

set -euo pipefail
echo "job=$PBS_JOBID nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
sleep 3600

#!/usr/bin/env bash
#PBS -N kimi_k3_hold_2n
#PBS -l walltime=01:00:00
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_2n.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_2n.err

set -euo pipefail
echo "job=$PBS_JOBID nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
sleep 3600

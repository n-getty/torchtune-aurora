#!/bin/bash
#PBS -N br_native_sft_hold
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/hold_native_sft_smoke.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/hold_native_sft_smoke.err
# 1-node hold for the BioReason native-Gemma4 SFT smoke. After it starts (state R),
# read the assigned node from `qstat -f <jobid>` exec_host, SSH in, and run:
#   bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_native_gemma4_sft_smoke.sh
echo "=== hold start $(date) host=$(hostname) ==="
echo "PBS_JOBID=$PBS_JOBID"
cat $PBS_NODEFILE
sleep 3500
echo "=== hold end $(date) ==="

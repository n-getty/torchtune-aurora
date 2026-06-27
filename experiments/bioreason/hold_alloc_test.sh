#!/bin/bash
#PBS -N br_alloc_test
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/hold_alloc_test.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/hold_alloc_test.err
echo "=== hold start $(date) ==="; echo "PBS_JOBID=$PBS_JOBID"; cat $PBS_NODEFILE; sleep 3500; echo "=== hold end $(date) ==="

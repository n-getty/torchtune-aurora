#!/bin/bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -N hold_bioreason_debug
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_debug.out

# Hold node on debug-scaling for BioReason GRPO empty_cache fix validation.
# SSH in and run:
#   cd /lus/flare/projects/ModCon/ngetty/torchtune
#   bash experiments/bioreason/run_bioreason_dedicated.sh 5
echo "=== Node held: $(hostname) ==="
echo "=== SSH: ssh $(hostname) ==="
sleep infinity

#!/bin/bash
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A ModCon
#PBS -N hold_bioreason_node
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_node.out

# Hold node for BioReason GRPO interactive debugging.
# SSH in and run from PROJDIR:
#   cd /lus/flare/projects/ModCon/ngetty/torchtune
#   bash experiments/bioreason/run_bioreason_interactive.sh

echo "=== Node held: $(hostname) ==="
echo "=== SSH: ssh $(hostname) ==="
sleep infinity

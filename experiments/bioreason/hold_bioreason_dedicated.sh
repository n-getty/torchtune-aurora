#!/bin/bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A ModCon
#PBS -N hold_bioreason_dedicated
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_dedicated.out

echo "=== Node held (12-tile dedicated vLLM test): $(hostname) ==="
echo "=== SSH: ssh $(hostname) ==="
sleep infinity

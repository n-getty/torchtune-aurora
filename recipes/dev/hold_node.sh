#!/bin/bash
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -o logs/hold_node.out
#PBS -e logs/hold_node.err
#PBS -N hold_node

echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Holding node for 60 minutes. SSH in to run tests."
sleep 3600
echo "End: $(date)"

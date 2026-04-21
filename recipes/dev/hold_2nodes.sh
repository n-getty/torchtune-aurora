#!/bin/bash
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -o logs/hold_2nodes.out
#PBS -e logs/hold_2nodes.err
#PBS -N hold_2nodes

echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo "Start: $(date)"
echo "PBS_NODEFILE: $PBS_NODEFILE"
cat $PBS_NODEFILE | sort -u
echo "Holding 2 nodes for 60 minutes. SSH in to run tests."
sleep 3600
echo "End: $(date)"

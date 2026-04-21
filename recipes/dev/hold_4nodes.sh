#!/bin/bash
#PBS -l select=4
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/hold_4nodes.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/hold_4nodes.err
#PBS -N hold_4nodes

echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo "Start: $(date)"
echo "PBS_NODEFILE: $PBS_NODEFILE"
cat $PBS_NODEFILE | sort -u
echo "Holding 4 nodes for 60 minutes. SSH in to run tests."
sleep 3600
echo "End: $(date)"

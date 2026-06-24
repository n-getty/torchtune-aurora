#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -N hold_bior2n_dbgs
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_2node_dbgs.out

# 2-node, 1h hold on debug-scaling for BioReason XCCL wsync NSTEPS≥5 validation.
echo "=== 2-node 1h hold (debug-scaling): $(hostname) ==="
echo "=== Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ') ==="
sleep infinity

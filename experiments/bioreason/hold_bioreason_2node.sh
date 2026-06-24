#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -N hold_bioreason2n
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_2node.out

# Hold 2 nodes on debug-scaling for BioReason 2-node split test.
# Node 0: 11 train ranks (FSDP1 ZeRO-2)
# Node 1: 1 vLLM rank (rank 11) — could be expanded to TP>1 later
# Wsync: gloo PG over Slingshot
echo "=== 2-node hold: $(hostname) ==="
echo "=== Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ') ==="
sleep infinity

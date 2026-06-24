#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -N hold_bior2n_dbg
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_2node_debug.out

# 2-node, 1h hold for BioReason Phase 2 testing on debug-scaling.
# Use while waiting for the 4h capacity hold (8456716) to start.
# Topology when used:
#   Node 0 (TRAIN_NODE): 11 train ranks (FSDP1 ZeRO-2)
#   Node 1 (VLLM_NODE):  12 vLLM HTTP servers (DP=12, ports 8001-8012)
echo "=== 2-node 1h hold (debug-scaling): $(hostname) ==="
echo "=== Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ') ==="
sleep infinity

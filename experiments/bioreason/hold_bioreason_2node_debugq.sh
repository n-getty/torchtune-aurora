#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A ModCon
#PBS -N hold_bior2n_dbg
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_2node_debugq.out

# 2-node, 1h hold for BioReason Phase 2 G-bisect (varlen G=12/16 ceiling test).
# Topology when used:
#   Node 0 (TRAIN_NODE): 11 train ranks (FSDP1 ZeRO-2)
#   Node 1 (VLLM_NODE):  12 vLLM HTTP servers (DP=12, ports 8001-8012)
echo "=== 2-node 1h hold (debug): $(hostname) ==="
echo "=== Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ') ==="
sleep infinity

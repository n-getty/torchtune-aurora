#!/bin/bash
#PBS -l select=2
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A ModCon
#PBS -N hold_bior2n_4h
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_2node_4h.out

# 2-node, 4h hold for BioReason Phase 2 implementation:
# - Node 0 (TRAIN_NODE): 11 train ranks (FSDP1 ZeRO-2)
# - Node 1 (VLLM_NODE): 12 vLLM HTTP servers (DP=12, ports 8001-8012)
# - Wsync TBD: either restart-per-step OR XCCL worker_extension port
echo "=== 2-node 4h hold (capacity): $(hostname) ==="
echo "=== Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ') ==="
sleep infinity

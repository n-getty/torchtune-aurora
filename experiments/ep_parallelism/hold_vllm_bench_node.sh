#!/bin/bash
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A ModCon
#PBS -N vllm_bench_node
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/ep_parallelism/vllm_bench_node.out

# Hold node for interactive vLLM MoE benchmark debugging.
# SSH in and run experiments/ep_parallelism/hold_vllm_moe_bench.sh manually
# or step through configs A-J one at a time.
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job: ${PBS_JOBID}"
cat "$PBS_NODEFILE"
sleep 3600

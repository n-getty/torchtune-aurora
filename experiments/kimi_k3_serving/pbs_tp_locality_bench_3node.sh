#!/usr/bin/env bash
#PBS -N k3_tploc
#PBS -l walltime=00:20:00
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=3
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/tploc.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/tploc.err

# Decide the TP/PP re-topology lever WITHOUT spending K3 capacity hold time.
# See bench_tp_group_locality.py for the question and the decision rule.
#
# No model, no DAOS, no vLLM -- pure XCCL, so it fits the free debug-scaling
# queue and runs in ~3 minutes. This follows the standing rule: no K3
# capacity load for a hypothesis a cheaper run can falsify.
#
# 36 ranks (12/node) not 32: the whole point is a group that fits inside one
# node's 12 tiles, so every node must contribute all 12.

set -o pipefail   # NOT set -e: `module load frameworks` exits non-zero on
                  # Aurora and would silently kill the script (see
                  # memory/feedback_aurora_set_e_kills_module_load.md).

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
RUN_DIR=$EXP/logs/tploc_${PBS_JOBID%%.*}
mkdir -p "$RUN_DIR"

echo "job=$PBS_JOBID nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"

# Production multi-node row of the CLAUDE.md launcher decision table --
# IDENTICAL to pbs_collective_bench_3node.sh so the tp32_3node column here is
# directly comparable to the 0.557 ms already measured on job 8748815.
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export BENCH_OUT="$RUN_DIR/results.json"
export BENCH_ITERS=${BENCH_ITERS:-200}

echo "phase=run ranks=36 ppn=12 time=$(date -Is)"
mpiexec -n 36 -ppn 12 --pmi=pmix \
    --hostfile "$PBS_NODEFILE" \
    "$PYTHON" "$EXP/bench_tp_group_locality.py" 2>&1 | tee "$RUN_DIR/bench.log"
echo "phase=done rc=$? time=$(date -Is)"

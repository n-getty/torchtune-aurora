#!/usr/bin/env bash
#PBS -N k3_contend
#PBS -l walltime=00:30:00
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/contend.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/contend.err

# Measure the RENDEZVOUS cost of interleaved collectives -- the ~715 ms/token
# left unexplained after compute+dispatch+isolated-collectives. See the .py.
#
# Submitted as a JOB, not run against a hold: mpiexec --pmi=pmix needs the
# PALS job process tree, and a held `sleep` cannot provide it -- launching
# from a login node against a hold fails with "Couldn't send RPC launch(...)"
# (see memory/feedback_mpiexec_pals_ssh.md, hit again on 2026-08-11).
#
# No model, no DAOS, no vLLM -- this is pure XCCL, so it fits the free
# debug-scaling queue and runs in ~2 minutes.

set -o pipefail   # NOT set -e: `module load frameworks` exits non-zero on
                  # Aurora and would silently kill the script (see
                  # memory/feedback_aurora_set_e_kills_module_load.md).

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
RUN_DIR=$EXP/logs/contend_${PBS_JOBID%%.*}
mkdir -p "$RUN_DIR"

echo "job=$PBS_JOBID nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"

# Production multi-node row of the CLAUDE.md launcher decision table.
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

echo "phase=run ranks=32 ppn=11 time=$(date -Is)"
mpiexec -n 12 -ppn 12 --pmi=pmix \
    --hostfile "$PBS_NODEFILE" \
    "$PYTHON" "$EXP/probe_launch_contended.py" 2>&1 | tee "$RUN_DIR/bench.log"
echo "phase=done rc=$? time=$(date -Is)"

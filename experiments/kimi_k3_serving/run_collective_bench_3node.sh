#!/usr/bin/env bash
# Launch bench_collective_latency_3node.py on a running 3-node hold.
#
# Uses the "Production multi-node" row of the CLAUDE.md launcher decision
# table (mpiexec --pmi=pmix / CCL_PROCESS_LAUNCHER=pmix / CCL_ATL_TRANSPORT=mpi
# / CCL_KVS_MODE=mpi), NOT the interactive row -- mixing those silently breaks
# CCL init.
#
# Usage:  bash run_collective_bench_3node.sh <FULL_PBS_JOB_ID>
set -uo pipefail

JOB_ID=${1:?Usage: $0 FULL_PBS_JOB_ID}
case "$JOB_ID" in
    *.aurora-pbs-*) ;;
    *) echo "ERROR: need the FULL PBS job id" >&2; exit 2 ;;
esac

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
RUN_DIR=$EXP/logs/collbench_${JOB_ID%%.*}
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/bench.log") 2>&1

echo "phase=start job=$JOB_ID time=$(date -Is)"

NODEFILE=$RUN_DIR/nodefile
qstat -f "$JOB_ID" | tr -d '\n\t ' \
    | grep -oP 'exec_host=\K.*?(?=exec_vnode)' \
    | tr '+' '\n' | sed 's#/.*##' | grep -oE '^x[0-9a-z]+' | sort -u > "$NODEFILE"
mapfile -t NODES < "$NODEFILE"
echo "nodes=${NODES[*]} count=${#NODES[@]}"
[[ ${#NODES[@]} -eq 3 ]] || { echo "ERROR: need 3 nodes, got ${#NODES[@]}"; exit 2; }
export PBS_NODEFILE="$NODEFILE"

# 32 ranks over 3 nodes to match K3's TP=32 exactly. 11+11+10 rather than
# 12/12/8: the point is to reproduce K3's rank->tile placement pressure, and
# --ppn 11 with -n 32 fills 11/11/10.
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

echo "phase=run ranks=32 ppn=11"
timeout 1200 mpiexec -n 32 -ppn 11 --pmi=pmix \
    --hostfile "$NODEFILE" \
    "$PYTHON" "$EXP/bench_collective_latency_3node.py"
rc=$?
echo "phase=done rc=$rc time=$(date -Is)"
[[ -f "$BENCH_OUT" ]] && echo "results: $BENCH_OUT"
exit $rc

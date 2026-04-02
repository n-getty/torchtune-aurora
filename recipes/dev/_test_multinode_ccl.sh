#!/bin/bash
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -q debug
#PBS -N ccl_test
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/ccl_multinode_test.out
set -e

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

module load frameworks 2>/dev/null || true

# Remove user virtualenv
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# Node discovery
UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
NODE0="${UNIQUE_NODES[0]}"
NODE1="${UNIQUE_NODES[1]:-${UNIQUE_NODES[0]}}"
NUM_NODES=${#UNIQUE_NODES[@]}

export MASTER_ADDR="${NODE0}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES

echo "=== CCL Multi-Node Test ==="
echo "Nodes: ${UNIQUE_NODES[*]}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================="

# CCL config
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1  # was 4; 4 causes 48x AllGather regression
export CCL_ALLREDUCE=ring
export CCL_REDUCE_SCATTER=ring
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCH_COMPILE_DISABLE=1

# Test with 10 ranks per node (matches GRPO training)
PPNS=10
TOTAL=$((NUM_NODES * PPNS))
echo "Testing with ${TOTAL} ranks (${PPNS}/node)..."

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL}" \
    -ppn "${PPNS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash -c '
export RANK="${PALS_RANKID:-0}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
export LOCAL_WORLD_SIZE="${PALS_LOCAL_SIZE:-'"${PPNS}"'}"
export WORLD_SIZE='"${TOTAL}"'
export ZE_AFFINITY_MASK="${LOCAL_RANK}"
export MASTER_ADDR='"${MASTER_ADDR}"'
export MASTER_PORT='"${MASTER_PORT}"'
module load frameworks >/dev/null 2>&1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1  # was 4; 4 causes 48x AllGather regression
export CCL_ALLREDUCE=ring
export CCL_REDUCE_SCATTER=ring
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK="${LOCAL_RANK}"
export MPI_LOCALRANKID="${LOCAL_RANK}"
export MPI_LOCALNRANKS="${LOCAL_WORLD_SIZE}"
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v myenv | tr "\n" ":" | sed "s/:$//")
python -u '"${TORCHTUNE_DIR}"'/recipes/dev/_test_multinode_ccl.py
'

echo "=== CCL Multi-Node Test Complete ==="

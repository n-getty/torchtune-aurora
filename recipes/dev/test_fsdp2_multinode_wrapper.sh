#!/bin/bash
# Minimal per-rank wrapper for multi-node FSDP2 benchmark.
# No vLLM, no torchtune dependencies — just CCL + FSDP2.
#
# Usage:
#   mpiexec -n 20 -ppn 10 --hostfile $PBS_NODEFILE \
#       --no-vni --cpu-bind depth --depth 8 \
#       bash recipes/dev/test_fsdp2_multinode_wrapper.sh \
#       recipes/dev/test_fsdp2_multinode_minimal.py
#
# Environment variables (optional):
#   AFFINITY_MODE: "single" (ZE_AFFINITY_MASK=$LOCAL_RANK),
#                  "training" (mask=0..ppn-1),
#                  "none" (unset mask)
#                  Default: "training"
#   CCL_TRANSPORT: "ofi" (direct CXI), "mpi" (MPI-based, official Aurora),
#                  "mpi-nolauncher" (MPI transport without PMIx launcher)
#                  Default: "ofi"
#   CCL_ALGO:      "ring", "topo" (topology-aware), "0" (CCL defaults)
#                  Default: "ring"
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

SCRIPT="${1:?Usage: test_fsdp2_multinode_wrapper.sh <script.py> [args...]}"
shift

# ============================================================
# Rank environment from PALS/PMI
# ============================================================
export RANK="${PMI_RANK:-${PALS_RANKID:-${RANK:-0}}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-${LOCAL_RANK:-0}}}"
export LOCAL_WORLD_SIZE="${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-${LOCAL_WORLD_SIZE:-10}}}"

if [[ -n "${PMI_SIZE:-}" ]]; then
    export WORLD_SIZE="${PMI_SIZE}"
elif [[ -n "${NUM_NODES:-}" ]]; then
    export WORLD_SIZE=$((NUM_NODES * LOCAL_WORLD_SIZE))
else
    export WORLD_SIZE="${WORLD_SIZE:-1}"
fi

export MPI_LOCALRANKID="${LOCAL_RANK}"
export MPI_LOCALNRANKS="${LOCAL_WORLD_SIZE}"

# ============================================================
# Module + CCL environment
# ============================================================
module load frameworks >/dev/null 2>&1 || true

# CCL transport mode:
#   "ofi"  — direct OFI/CXI (our current config, bypasses MPI)
#   "mpi"  — MPI-based transport (official Aurora recommendation, topology-aware)
CCL_TRANSPORT="${CCL_TRANSPORT:-ofi}"

if [[ "${CCL_TRANSPORT}" == "mpi" ]]; then
    # Official Aurora recommended settings (tested up to 1024 nodes)
    # Uses MPI's topology-aware transport — sub-communicators should get
    # intra-node shared-memory paths automatically.
    # Requires: mpiexec --pmi=pmix AND mpi4py pre-init in the Python script.
    export CCL_PROCESS_LAUNCHER=pmix
    export CCL_ATL_TRANSPORT=mpi
    export CCL_KVS_MODE=mpi
    export CCL_KVS_USE_MPI_RANKS=1
    export CCL_CONFIGURATION_PATH=""
    export CCL_CONFIGURATION=cpu_gpu_dpcpp
    export CCL_KVS_CONNECTION_TIMEOUT=600
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
    export FI_PROVIDER=cxi
elif [[ "${CCL_TRANSPORT}" == "mpi-nolauncher" ]]; then
    # Hybrid: MPI data transport but without PMIx process launcher.
    # Fallback if CCL_PROCESS_LAUNCHER=pmix causes "Fatal error in
    # internal_Init_thread". Relies on mpi4py pre-init in the Python script
    # to make MPI available without CCL doing its own launcher init.
    export CCL_PROCESS_LAUNCHER=none
    export CCL_ATL_TRANSPORT=mpi
    export CCL_KVS_MODE=mpi
    export CCL_KVS_USE_MPI_RANKS=1
    export CCL_CONFIGURATION_PATH=""
    export CCL_CONFIGURATION=cpu_gpu_dpcpp
    export CCL_KVS_CONNECTION_TIMEOUT=600
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
    export FI_PROVIDER=cxi
    export CCL_KVS_IFACE=hsn0
else
    # Direct OFI transport (our current config)
    export CCL_PROCESS_LAUNCHER=none
    export CCL_ATL_TRANSPORT=ofi
    export FI_PROVIDER=cxi
    export CCL_KVS_IFACE=hsn0
fi

export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1}
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# CCL algorithm selection
#   "ring"  — ring (our current config)
#   "topo"  — topology-aware (official Aurora recommendation)
#   "0"     — let CCL choose defaults
CCL_ALGO="${CCL_ALGO:-ring}"
if [[ "${CCL_ALGO}" == "ring" ]]; then
    export CCL_ALLREDUCE=ring
    export CCL_REDUCE_SCATTER=ring
elif [[ "${CCL_ALGO}" == "topo" ]]; then
    export CCL_ALLREDUCE=topo
    export CCL_ALLREDUCE_SCALEOUT=rabenseifner
    unset CCL_REDUCE_SCATTER 2>/dev/null || true
else
    unset CCL_ALLREDUCE 2>/dev/null || true
    unset CCL_REDUCE_SCATTER 2>/dev/null || true
fi

# GPU affinity
AFFINITY_MODE="${AFFINITY_MODE:-training}"
if [[ "${AFFINITY_MODE}" == "single" ]]; then
    export ZE_AFFINITY_MASK="${LOCAL_RANK}"
elif [[ "${AFFINITY_MODE}" == "training" ]]; then
    export ZE_AFFINITY_MASK=$(seq -s, 0 $((LOCAL_WORLD_SIZE - 1)))
elif [[ "${AFFINITY_MODE}" == "all" ]]; then
    export ZE_AFFINITY_MASK=$(seq -s, 0 11)
else
    unset ZE_AFFINITY_MASK 2>/dev/null || true
fi

# Remove user virtualenv
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV 2>/dev/null; true
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Master address
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# Log on rank 0
if [[ "${RANK}" == "0" ]]; then
    echo "[Rank 0] node=$(hostname) WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "[Rank 0] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    echo "[Rank 0] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
    echo "[Rank 0] CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT} CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER}"
    echo "[Rank 0] CCL_ALLREDUCE=${CCL_ALLREDUCE:-<default>} CCL_REDUCE_SCATTER=${CCL_REDUCE_SCATTER:-<default>}"
    echo "[Rank 0] CCL_KVS_MODE=${CCL_KVS_MODE:-<unset>} AFFINITY_MODE=${AFFINITY_MODE}"
fi

# ============================================================
# Run
# ============================================================
cd "${TORCHTUNE_DIR}"
python -u "${SCRIPT}" "$@"

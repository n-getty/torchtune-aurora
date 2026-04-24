#!/bin/bash
# Per-rank wrapper for multi-node GRPO training on Aurora XPU.
# Called by aurora_grpo_vllm_multinode.sh via mpiexec.
#
# Sets ZE_AFFINITY_MASK=$LOCAL_RANK so each rank sees only its tile as xpu:0.
# Multi-node requires this for proper CXI fabric authentication.
#
# Usage: bash aurora_grpo_vllm_wrapper.sh <RECIPE> <CONFIG> [extra tune args...]

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

RECIPE="${1:?Usage: aurora_grpo_vllm_wrapper.sh <RECIPE> <CONFIG> [args...]}"
CONFIG="${2:?Usage: aurora_grpo_vllm_wrapper.sh <RECIPE> <CONFIG> [args...]}"
shift 2

# ============================================================
# Rank environment (from PALS/PMI — set by mpiexec)
# ============================================================
# PALS always sets: PALS_RANKID, PALS_LOCAL_RANKID, PALS_LOCAL_SIZE, PALS_NODEID
# Cray PMI additionally sets: PMI_RANK, PMI_LOCAL_RANK, PMI_SIZE, PMI_LOCAL_SIZE
# PALS_NRANKS is an INPUT var (not set per-rank) — do NOT rely on it for WORLD_SIZE.
#
# CRITICAL: Read PMI/PALS vars FIRST, fall back to pre-existing env vars LAST.
# module load / login profiles may pre-set WORLD_SIZE=1.

export RANK="${PMI_RANK:-${PALS_RANKID:-${RANK:-0}}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-${LOCAL_RANK:-0}}}"
export LOCAL_WORLD_SIZE="${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-${LOCAL_WORLD_SIZE:-${NGPUS_PER_NODE:-10}}}}"

# WORLD_SIZE: PMI_SIZE is the only reliable per-rank source.
# If missing, compute from NUM_NODES * LOCAL_WORLD_SIZE (exported by launcher).
if [[ -n "${PMI_SIZE:-}" ]]; then
    export WORLD_SIZE="${PMI_SIZE}"
elif [[ -n "${NUM_NODES:-}" ]] && [[ -n "${LOCAL_WORLD_SIZE}" ]]; then
    export WORLD_SIZE=$((NUM_NODES * LOCAL_WORLD_SIZE))
else
    export WORLD_SIZE="${WORLD_SIZE:-1}"
fi

# Debug: log all PMI/PALS vars on every rank (just first 2) to diagnose env propagation
if [[ "${RANK}" == "0" ]] || [[ "${RANK}" == "1" ]]; then
    echo "[DEBUG rank${RANK}] PMI_RANK=${PMI_RANK:-<unset>} PALS_RANKID=${PALS_RANKID:-<unset>}"
    echo "[DEBUG rank${RANK}] PMI_SIZE=${PMI_SIZE:-<unset>} PALS_NRANKS=${PALS_NRANKS:-<unset>} NUM_NODES=${NUM_NODES:-<unset>}"
    echo "[DEBUG rank${RANK}] PMI_LOCAL_RANK=${PMI_LOCAL_RANK:-<unset>} PMI_LOCAL_SIZE=${PMI_LOCAL_SIZE:-<unset>}"
    echo "[DEBUG rank${RANK}] PALS_LOCAL_RANKID=${PALS_LOCAL_RANKID:-<unset>} PALS_LOCAL_SIZE=${PALS_LOCAL_SIZE:-<unset>}"
    echo "[DEBUG rank${RANK}] => RANK=${RANK} LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
fi

# CCL local topology hint (suppresses "could not get local_idx/count" warning)
export MPI_LOCALRANKID="${LOCAL_RANK}"
export MPI_LOCALNRANKS="${LOCAL_WORLD_SIZE}"

# GPU affinity: CCL needs visibility of all training tiles for proper
# UUID-based topology routing. Setting ZE_AFFINITY_MASK to a single tile
# causes "narrow device affinity mask" and 12x AllGather regression.
# Setting it to ALL 12 tiles conflicts with vLLM on tiles 10-11 (backward hangs).
# Solution: set it to exactly the training tiles (0..NGPUS_PER_NODE-1).
if [[ "${USE_AFFINITY_MASK:-0}" == "1" ]]; then
    export ZE_AFFINITY_MASK="${LOCAL_RANK}"
elif [[ "${USE_AFFINITY_MASK:-0}" == "training" ]]; then
    # Show all training tiles to each rank — CCL sees all 10 UUIDs
    TRAIN_TILES_MASK=$(seq -s, 0 $((LOCAL_WORLD_SIZE - 1)))
    export ZE_AFFINITY_MASK="${TRAIN_TILES_MASK}"
else
    unset ZE_AFFINITY_MASK
fi

# Master address (set by launcher)
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# ============================================================
# Module and environment
# ============================================================
module load frameworks/2025.3.1 >/dev/null 2>&1 || true

# Ensure MPI library is in LD_LIBRARY_PATH (CCL MPI transport needs libmpi.so.12).
# Only do this when MPI transport is actually selected — the 25.190.0 oneapi tree
# also contains libsycl.so.8 with an older ABI that breaks framework 2025.3.1's
# libtorch-xpu-ops-sycltla-mha_fwd.so. With OFI transport, MPI lib injection is
# unnecessary and actively harmful (poisons libsycl resolution).
MPI_LIB_DIR="/opt/aurora/25.190.0/oneapi/2025.2/lib"
if [[ "${CCL_TRANSPORT_OVERRIDE:-mpi}" == "mpi" ]]; then
    if [[ -d "${MPI_LIB_DIR}" ]] && [[ ":${LD_LIBRARY_PATH:-}:" != *":${MPI_LIB_DIR}:"* ]]; then
        export LD_LIBRARY_PATH="${MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
else
    # Defensive: strip any inherited reference to the old oneapi tree to keep
    # libsycl resolution on the 26.26.0/2025.3.1 path.
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/opt/aurora/25\.' | tr '\n' ':' | sed 's/:$//')
    fi
fi

# CRITICAL: Re-export CCL env vars AFTER module load.
# `module load frameworks` resets CCL_PROCESS_LAUNCHER to "pmix".
#
# Transport selection (set CCL_TRANSPORT_OVERRIDE in launcher to override):
#   mpi: topology-aware sub-communicator routing, ~4.5 GiB/s AllGather intra-node
#        Requires: mpiexec --pmi=pmix AND mpi4py pre-init in Python
#        BUG: deadlocks on multi-node broadcast (XCCL communicator creation fails)
#   ofi: libfabric/CXI direct, ~2.4 GiB/s AllGather intra-node
#        No MPI dependency for transport, works reliably on multi-node
_CCL_TRANSPORT="${CCL_TRANSPORT_OVERRIDE:-mpi}"
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT="${_CCL_TRANSPORT}"
if [[ "${_CCL_TRANSPORT}" == "mpi" ]]; then
    export CCL_KVS_MODE=mpi
    export CCL_KVS_USE_MPI_RANKS=1
else
    # OFI transport: use pmix KVS (not MPI KVS)
    unset CCL_KVS_MODE
    unset CCL_KVS_USE_MPI_RANKS
fi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
# CRITICAL: CCL_WORKER_COUNT=4 causes 48x AllGather bandwidth degradation
# (2.4 GiB/s vs 111 GiB/s with default of 1). Keep at 1 unless Intel fixes this.
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
# CCL_REDUCE_SCATTER=ring causes 63x ReduceScatter regression on multi-node
# (1.9 GiB/s vs 138 GiB/s with CCL default). Do NOT set it.
# export CCL_REDUCE_SCATTER=ring  # DISABLED — causes backward pass regression
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZES_ENABLE_SYSMAN=1  # Required for accurate torch.xpu.mem_get_info()
if [[ "${USE_AFFINITY_MASK:-0}" == "1" ]]; then
    export ZE_AFFINITY_MASK="${LOCAL_RANK}"
elif [[ "${USE_AFFINITY_MASK:-0}" == "training" ]]; then
    TRAIN_TILES_MASK=$(seq -s, 0 $((LOCAL_WORLD_SIZE - 1)))
    export ZE_AFFINITY_MASK="${TRAIN_TILES_MASK}"
else
    unset ZE_AFFINITY_MASK
fi
# NOTE: expandable_segments:True is INCOMPATIBLE with oneCCL RDMA (CXI fabric).
# Virtual memory pointers can't be registered for RDMA DMA. Use only CCL-safe options.
# max_split_size_mb=512: Prevents allocator from splitting FSDP AllGather blocks
#   (1-2 GiB each) to serve tiny activation/logprob requests. Without this,
#   small "splinter" allocations fragment large free blocks, making them
#   unreusable for the next AllGather and forcing new L0 allocations.
# garbage_collection_threshold=0.6: Sweeps the small-block pool to coalesce
#   freed activation/logprob blocks when usage exceeds 60% of peak.
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
export TORCH_COMPILE_DISABLE=1

# Paths
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
aurora_export_pythonpath "${TORCHTUNE_DIR}" "${TRL_DIR}" "${VLLM_CUSTOMIZATION}"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV 2>/dev/null; true
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Disable HTTP proxy for training — vLLM requests are local/intra-cluster.
# Aurora compute nodes have http_proxy set to Squid, which blocks inter-node
# HTTP on non-standard ports (vLLM on 8001).
export no_proxy="*"
export NO_PROXY="*"

# ============================================================
# Log rank info
# ============================================================
if [[ "${RANK}" == "0" ]]; then
    echo "[Rank ${RANK}] node=$(hostname) LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "[Rank ${RANK}] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    echo "[Rank ${RANK}] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}"
    echo "[Rank ${RANK}] CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER} CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT} CCL_KVS_MODE=${CCL_KVS_MODE:-<unset>}"
fi

# ============================================================
# Launch training
# ============================================================
cd "${TORCHTUNE_DIR}"

# Launch recipe as script (not module — recipes/__init__.py blocks imports)
python -u "recipes/${RECIPE}.py" \
    --config "${CONFIG}" \
    "$@"

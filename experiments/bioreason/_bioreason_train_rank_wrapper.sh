#!/bin/bash
# Per-rank wrapper for the BioReason N-node HSDP launcher (pmix/mpi).
# Reads PMI rank vars (set by mpiexec --pmi=pmix), pins ZE_AFFINITY_MASK to the
# tile owned by this LOCAL_RANK (full node: tiles 0..11), then execs the recipe.
#
# Required env (exported by the parent launcher):
#   MASTER_ADDR, MASTER_PORT, the full multinode CCL block, the BioReason wsync
#   env (TORCHTUNE_WEIGHT_SYNC_PATH, TORCHTUNE_XCCL_HOST, WSYNC_*), the
#   TORCHTUNE_* fast-path knobs, PYTHONPATH, and BIOREASON_SRC/BIOREASON_DEPS.
set -e

# Resolve rank/size from whichever launcher env is populated. Under Aurora PALS,
# PMI_RANK/PMI_LOCAL_RANK are set but PMI_SIZE is often EMPTY — so for the TOTAL
# rank count we fall back to PALS_NRANKS and then to ${WORLD}, which the parent
# launcher exports explicitly (== mpiexec -n). Do NOT keep a bogus numeric literal
# default for WORLD_SIZE: a wrong value makes the recipe init an oversized process
# group and the absent ranks time out with
# "DistNetworkError: Failed to recv, got 0 bytes".
export RANK="${PMI_RANK:-${PALS_RANKID:-${RANK:-0}}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-${LOCAL_RANK:-0}}}"
export LOCAL_WORLD_SIZE="${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-${LOCAL_WORLD_SIZE:-12}}}"
export WORLD_SIZE="${PMI_SIZE:-${PALS_NRANKS:-${WORLD:-${WORLD_SIZE}}}}"
if [[ -z "${WORLD_SIZE}" ]]; then
    echo "[br-wrapper] FATAL: could not resolve WORLD_SIZE (PMI_SIZE/PALS_NRANKS/WORLD all empty)" >&2
    exit 1
fi

# Full-node training: train rank N owns tile N, LOCAL_RANK in 0..11.
export ZE_AFFINITY_MASK="${LOCAL_RANK}"

# MPI shim hints (oneCCL+pmix path on Aurora)
export MPI_LOCALRANKID="${LOCAL_RANK}"
export MPI_LOCALNRANKS="${LOCAL_WORLD_SIZE}"

if [[ "${RANK}" == "0" ]]; then
    echo "[br-wrapper rank 0] node=$(hostname) WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "[br-wrapper rank 0] MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT} ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
    echo "[br-wrapper rank 0] CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER} CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT} CCL_KVS_MODE=${CCL_KVS_MODE:-<unset>}"
fi

exec python3 "$@"

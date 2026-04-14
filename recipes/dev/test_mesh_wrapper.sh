#!/bin/bash
# Minimal per-rank wrapper for test scripts on Aurora XPU.
# Usage: bash test_mesh_wrapper.sh <script.py> [args...]
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

SCRIPT="${1:?Usage: test_mesh_wrapper.sh <script.py> [args...]}"
shift

# Rank from PALS/PMI
export RANK="${PMI_RANK:-${PALS_RANKID:-${RANK:-0}}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-${LOCAL_RANK:-0}}}"
export LOCAL_WORLD_SIZE="${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-${LOCAL_WORLD_SIZE:-10}}}"
if [[ -n "${PMI_SIZE:-}" ]]; then
    export WORLD_SIZE="${PMI_SIZE}"
elif [[ -n "${NUM_NODES:-}" ]]; then
    export WORLD_SIZE=$((NUM_NODES * LOCAL_WORLD_SIZE))
fi

# GPU affinity — training tiles only
TRAIN_TILES_MASK=$(seq -s, 0 $((LOCAL_WORLD_SIZE - 1)))
export ZE_AFFINITY_MASK="${TRAIN_TILES_MASK}"

# Load frameworks
module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# Re-export CCL (frameworks resets them)
export ZE_AFFINITY_MASK="${TRAIN_TILES_MASK}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1  # was 4; 4 causes 48x AllGather regression
# Allow override from environment; use ring as default for multi-node
if [[ -z "${CCL_ALLREDUCE_OVERRIDE+x}" ]]; then
    export CCL_ALLREDUCE=ring
    export CCL_REDUCE_SCATTER=ring
else
    export CCL_ALLREDUCE="${CCL_ALLREDUCE_OVERRIDE}"
    export CCL_REDUCE_SCATTER="${CCL_REDUCE_SCATTER_OVERRIDE}"
fi
export CCL_CHUNK_SIZE=16777216
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
aurora_export_pythonpath "${TORCHTUNE_DIR}" "${TRL_DIR}"

if [[ "${RANK}" == "0" ]]; then
    echo "[Rank 0] node=$(hostname) LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE}"
    echo "[Rank 0] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
    echo "[Rank 0] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
fi

python3 -u "${SCRIPT}" "$@"

#!/bin/bash
# Per-rank wrapper script for Aurora GRPO training.
# Called by aurora_grpo.sh for each MPI rank.
#
# Usage: bash aurora_grpo_wrapper.sh <RECIPE> <CONFIG> [extra tune args...]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

RECIPE="${1:?Usage: aurora_grpo_wrapper.sh <RECIPE> <CONFIG>}"
CONFIG="${2:?Usage: aurora_grpo_wrapper.sh <RECIPE> <CONFIG>}"
shift 2

# ============================================================
# Rank environment (from PALS/PMI or pre-set by launcher)
# ============================================================
export RANK=${RANK:-${PMI_RANK:-${PMIX_RANK:-${PALS_RANKID:-0}}}}
export LOCAL_RANK=${LOCAL_RANK:-${PMI_LOCAL_RANK:-${PMIX_LOCAL_RANK:-${PALS_LOCAL_RANKID:-0}}}}
export WORLD_SIZE=${WORLD_SIZE:-${PMI_SIZE:-${PMIX_SIZE:-${PALS_NRANKS:-1}}}}
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-${PMI_LOCAL_SIZE:-${PMIX_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-${NGPUS_PER_NODE:-12}}}}}

# GPU affinity: each rank sees only its tile as xpu:0
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-$LOCAL_RANK}

# Master address (should be set by launcher)
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# ============================================================
# Module and paths
# ============================================================
module load frameworks 2>/dev/null || true

# ============================================================
# Log rank info
# ============================================================
if [[ "${RANK}" == "0" ]]; then
    echo "[Rank ${RANK}] LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
    echo "[Rank ${RANK}] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    echo "[Rank ${RANK}] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
    echo "[Rank ${RANK}] ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-not set}"
fi

# ============================================================
# Launch training
# ============================================================
cd "${TORCHTUNE_DIR}"

python -u -m recipes.dev.$(echo "${RECIPE}" | tr '/' '.') \
    --config "${CONFIG}" \
    "$@"

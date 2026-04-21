#!/bin/bash
#PBS -l select=1:system=aurora
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -A ModCon
#PBS -q debug
#PBS -N ppo_xpu
#PBS -j oe
#
# Aurora PPO launcher for torchtune XPU recipe.
#
# Usage:
#   Single node:  qsub aurora_ppo.sh
#   Multi-node:   Edit NUM_NODES below, then: qsub aurora_ppo.sh
#   Interactive:  bash aurora_ppo.sh  (from an interactive session)
#
# Required: module load frameworks

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

# ============================================================
# Configuration
# ============================================================
NUM_NODES=${NUM_NODES:-1}
NGPUS_PER_NODE=12                  # 12 tiles per node (FLAT mode)
RECIPE="dev/ppo_full_finetune_distributed"
CONFIG="dev/1B_ppo_distributed_xpu"
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_ppo_wrapper.sh"

# ============================================================
# Environment setup
# ============================================================
module load frameworks 2>/dev/null || true

# CCL / oneCCL environment (required for multi-node)
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1  # was 4; 4 causes 48x AllGather regression
export CCL_ALLREDUCE=ring
# CCL_REDUCE_SCATTER=ring causes 63x regression on multi-node. Do NOT set.
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072

# GPU hierarchy: FLAT = 12 tiles/node (64GB each)
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Disable CUDA memory allocator warning
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

# ============================================================
# Master address (HSN for multi-node bandwidth)
# ============================================================
if [[ -n "${PBS_NODEFILE:-}" ]]; then
    MASTER_HOST=$(head -1 "$PBS_NODEFILE")
    NUM_NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
else
    MASTER_HOST=$(hostname)
fi
MASTER_HOST_BASE=$(echo "$MASTER_HOST" | cut -d'.' -f1)
export MASTER_ADDR="${MASTER_HOST_BASE}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES
export NGPUS_PER_NODE

echo "=== PPO XPU Launch ==="
echo "Nodes:        ${NUM_NODES}"
echo "GPUs/node:    ${NGPUS_PER_NODE}"
echo "Total ranks:  $((NUM_NODES * NGPUS_PER_NODE))"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Recipe:       ${RECIPE}"
echo "Config:       ${CONFIG}"
echo "========================"

# ============================================================
# Launch
# ============================================================
TOTAL_RANKS=$((NUM_NODES * NGPUS_PER_NODE))

cd "${TORCHTUNE_DIR}"

if [[ -n "${PBS_NODEFILE:-}" ]]; then
    # PBS job — use mpiexec for multi-node
    mpiexec \
        --hostfile "${PBS_NODEFILE}" \
        -n "${TOTAL_RANKS}" \
        -ppn "${NGPUS_PER_NODE}" \
        --no-vni \
        --cpu-bind depth \
        --depth 8 \
        bash -l "${WRAPPER}" "${RECIPE}" "${CONFIG}"
else
    # Interactive / single-node — use torchrun (handles TCP store and rank assignment)
    torchrun \
        --standalone \
        --nproc_per_node="${NGPUS_PER_NODE}" \
        -m "recipes.dev.$(echo "${RECIPE##*/}" | tr '/' '.')" \
        --config "${CONFIG}"
fi

echo "=== PPO XPU training complete ==="

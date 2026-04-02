#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -q debug
#PBS -N grpo_hsdp
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/grpo_hsdp.out
#
# 2-node HSDP GRPO on Aurora XPU (no vLLM — built-in generation).
#
# HSDP layout:
#   Each node: 10 ranks, FSDP shards the model across 10 tiles
#   Across nodes: DDP replicates (gradient AllReduce only)
#   Total: 20 ranks, dp_replicate=2, dp_shard=10
#
# This is much more efficient than pure FSDP across 20 ranks because
# inter-node communication is only gradient AllReduce (not parameter
# AllGather/ReduceScatter).
#
# Usage:
#   qsub recipes/dev/aurora_grpo_hsdp_multinode.sh
#   Or interactive (on 2 allocated nodes):
#     bash recipes/dev/aurora_grpo_hsdp_multinode.sh
#
set -e

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

# ============================================================
# Configuration
# ============================================================
NGPUS_PER_NODE=${NGPUS_PER_NODE:-10}
MODEL_PATH=${MODEL_PATH:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B}
NSTEPS=${NSTEPS:-5}
CONFIG=${CONFIG:-recipes/configs/dev/baseline/qwen3B_grpo_hsdp_multinode_xpu.yaml}
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_grpo_vllm_wrapper.sh"

# ============================================================
# Environment setup
# ============================================================
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment — multi-node requires ring algorithms
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
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# Paths
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
export PYTHONPATH="${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ============================================================
# Node discovery
# ============================================================
if [[ -n "${PBS_NODEFILE:-}" ]]; then
    UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
    NODE0="${UNIQUE_NODES[0]}"
    NODE1="${UNIQUE_NODES[1]:-${UNIQUE_NODES[0]}}"
    NUM_NODES=${#UNIQUE_NODES[@]}
    echo "PBS_NODEFILE unique nodes (order-preserved): ${UNIQUE_NODES[*]}"
else
    NODE0=$(hostname | cut -d'.' -f1)
    NODE1=${NODE1:-$NODE0}
    NUM_NODES=${NUM_NODES:-1}
fi

export MASTER_ADDR="${NODE0}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES
export NGPUS_PER_NODE

TOTAL_RANKS=$((NUM_NODES * NGPUS_PER_NODE))

echo "=== GRPO HSDP Multi-Node (No vLLM) ==="
echo "Nodes:        ${NUM_NODES} (${NODE0}, ${NODE1})"
echo "Training:     ${TOTAL_RANKS} ranks (${NGPUS_PER_NODE}/node)"
echo "HSDP:         dp_replicate=${NUM_NODES} × dp_shard=${NGPUS_PER_NODE}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:       ${CONFIG}"
echo "Model:        ${MODEL_PATH}"
echo "Steps:        ${NSTEPS}"
echo "=================================="

mkdir -p /tmp/torchtune
mkdir -p "${TORCHTUNE_DIR}/logs"

# ============================================================
# Stage model to /tmp on BOTH nodes
# ============================================================
LOCAL_MODEL="/tmp/torchtune/$(basename ${MODEL_PATH})"

stage_model() {
    local node=$1
    if ssh "${node}" "test -f '${LOCAL_MODEL}/config.json'" 2>/dev/null; then
        echo "Model already staged on ${node}"
    else
        echo "Staging model to ${node}:${LOCAL_MODEL}..."
        ssh "${node}" "mkdir -p /tmp/torchtune && cp -r '${MODEL_PATH}' '${LOCAL_MODEL}'" 2>/dev/null
        echo "Staged on ${node}"
    fi
}

stage_model "${NODE0}" &
PID_STAGE0=$!
if [[ "${NODE0}" != "${NODE1}" ]]; then
    stage_model "${NODE1}" &
    PID_STAGE1=$!
fi
wait $PID_STAGE0
[[ "${NODE0}" != "${NODE1}" ]] && wait $PID_STAGE1
MODEL_PATH="${LOCAL_MODEL}"

# ============================================================
# Launch training via mpiexec
# ============================================================
echo "Starting HSDP GRPO training (${TOTAL_RANKS} ranks across ${NUM_NODES} nodes)..."

export MODEL_PATH
export NSTEPS
export CONFIG

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS_PER_NODE}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" \
    "dev/grpo_full_finetune_distributed_xpu" \
    "${CONFIG}" \
    "base_model_path=${MODEL_PATH}" \
    "num_steps=${NSTEPS}" \
    "data_parallel_replicate_dim=${NUM_NODES}"

echo "=== HSDP GRPO training complete ==="

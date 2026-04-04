#!/bin/bash
#
# Run HSDP GRPO test interactively on held 2-node PBS job.
# Usage: SSH to node 0, set PBS env vars, then run this script.
#
set -e

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

# ============================================================
# Configuration
# ============================================================
NGPUS_PER_NODE=${NGPUS_PER_NODE:-10}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/Qwen2.5-3B}
NSTEPS=${NSTEPS:-3}
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
# export CCL_REDUCE_SCATTER=ring  # DISABLED
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# Paths
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"
export PYTHONPATH="${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ============================================================
# Node discovery from PBS_NODEFILE
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Set it first:"
    echo "  export PBS_JOBID=<jobid>"
    echo "  export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
NODE0="${UNIQUE_NODES[0]}"
NODE1="${UNIQUE_NODES[1]:-${UNIQUE_NODES[0]}}"
NUM_NODES=${#UNIQUE_NODES[@]}

export MASTER_ADDR="${NODE0}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES
export NGPUS_PER_NODE

TOTAL_RANKS=$((NUM_NODES * NGPUS_PER_NODE))

echo "=== Interactive HSDP GRPO Test ==="
echo "Nodes:        ${NUM_NODES} (${UNIQUE_NODES[*]})"
echo "Training:     ${TOTAL_RANKS} ranks (${NGPUS_PER_NODE}/node)"
echo "HSDP:         dp_replicate=${NUM_NODES} × dp_shard=${NGPUS_PER_NODE}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:       ${CONFIG}"
echo "Model:        ${MODEL_PATH}"
echo "Steps:        ${NSTEPS}"
echo "=================================="

# Check model is staged
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "Staging model to ${node}:${MODEL_PATH}..."
        ssh "${node}" "mkdir -p /tmp/torchtune && cp -r /lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B ${MODEL_PATH}" 2>/dev/null
    else
        echo "Model already staged on ${node}"
    fi
done

export MODEL_PATH
export NSTEPS
export CONFIG

echo ""
echo "Starting HSDP GRPO training..."
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

echo "=== Done ==="

#!/bin/bash
#
# 2-node GRPO with replicated vLLM + HSDP on Aurora XPU.
#
# Architecture: Each node runs:
#   - vLLM server on tiles 10-11 (TP=2, ZE_AFFINITY_MASK=10,11)
#   - 10 training ranks on tiles 0-9 (FSDP within node, DDP across nodes)
# Total: 2 vLLM servers + 20 training ranks, HSDP dp_replicate=2 dp_shard=10
#
# This doubles generation throughput vs single-node: each shard leader talks
# to its own local vLLM at http://localhost:VLLM_PORT.
#
# Usage (interactive on held 2-node PBS job):
#   export PBS_JOBID=<jobid>
#   export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
#   bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh
#
# Usage (PBS submission):
#   qsub recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh
#
#PBS -l select=2:system=aurora
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/grpo_vllm_hsdp_multinode.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/grpo_vllm_hsdp_multinode.err
#PBS -N grpo_vllm_hsdp
set -e

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

# ============================================================
# Configuration
# ============================================================
NGPUS_PER_NODE=${NGPUS_PER_NODE:-10}
VLLM_TILES=${VLLM_TILES:-2}           # TP degree per node
VLLM_PORT=${VLLM_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
MODEL_SRC=${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/$(basename ${MODEL_SRC})}
NSTEPS=${NSTEPS:-5}
CONFIG=${CONFIG:-recipes/configs/dev/qwen3B_grpo_vllm_hsdp_multinode_xpu.yaml}
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_grpo_vllm_wrapper.sh"

# ============================================================
# Environment setup
# ============================================================
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment — multi-node ring algorithms
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=4
export CCL_ALLREDUCE=ring
export CCL_REDUCE_SCATTER=ring
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
# Node discovery from PBS_NODEFILE
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Set it first:"
    echo "  export PBS_JOBID=<jobid>"
    echo "  export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
NUM_NODES=${#UNIQUE_NODES[@]}
if [[ ${NUM_NODES} -lt 2 ]]; then
    echo "ERROR: Need at least 2 nodes, got ${NUM_NODES}"
    exit 1
fi

export MASTER_ADDR="${UNIQUE_NODES[0]}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES
export NGPUS_PER_NODE

TOTAL_RANKS=$((NUM_NODES * NGPUS_PER_NODE))

echo "=== Replicated vLLM + HSDP GRPO (${NUM_NODES} nodes) ==="
echo "Nodes:          ${NUM_NODES} (${UNIQUE_NODES[*]})"
echo "Training:       ${TOTAL_RANKS} ranks (${NGPUS_PER_NODE}/node, tiles 0-$((NGPUS_PER_NODE-1)))"
echo "vLLM:           TP=${VLLM_TILES} per node (tiles $((12-VLLM_TILES))-11)"
echo "HSDP:           dp_replicate=${NUM_NODES} x dp_shard=${NGPUS_PER_NODE}"
echo "Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:         ${CONFIG}"
echo "Model src:      ${MODEL_SRC}"
echo "Model local:    ${MODEL_PATH}"
echo "vLLM port:      ${VLLM_PORT}"
echo "Steps:          ${NSTEPS}"
echo "================================================="

# ============================================================
# Stage model to /tmp on ALL nodes (parallel)
# ============================================================
VLLM_LOG="/tmp/torchtune/vllm_server.log"
VLLM_TILE_START=$((12 - VLLM_TILES))
VLLM_MASK=$(seq -s, ${VLLM_TILE_START} 11)

for node in "${UNIQUE_NODES[@]}"; do
    (
        if ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
            echo "Model already staged on ${node}"
        else
            echo "Staging model to ${node}:${MODEL_PATH}..."
            ssh "${node}" "mkdir -p /tmp/torchtune && cp -r '${MODEL_SRC}' '${MODEL_PATH}'" 2>/dev/null
            echo "Model staged on ${node}"
        fi
    ) &
done
wait
echo "Model staging complete."

# ============================================================
# Warm vLLM model info cache on all nodes (parallel)
# ============================================================
echo "Warming vLLM model info cache on all nodes..."
WARM_CMD="
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${VLLM_TILE_START}
export TORCH_COMPILE_DISABLE=1
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
export PYTHONPATH='${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python3 -c \"
from vllm.config import ModelConfig
ModelConfig(
    model='${MODEL_PATH}',
    tokenizer='${MODEL_PATH}',
    dtype='bfloat16',
    enforce_eager=True,
)
print('Cache warmed')
\" 2>&1 | tail -1
"
for node in "${UNIQUE_NODES[@]}"; do
    ssh "${node}" "${WARM_CMD}" &
done
wait
echo "vLLM cache warm complete."

# ============================================================
# Launch vLLM on ALL nodes (parallel)
# ============================================================
# Each node gets its own vLLM server on the same port, isolated to tiles 10-11.
# Training ranks connect to localhost — each shard leader talks to its local vLLM.

VLLM_PIDS=()

# Build vLLM launch command: TP=1 uses vllm_serve_xpu.py (has weight sync endpoint),
# TP>1 uses vllm CLI serve (AsyncLLMEngine handles TP>1 on XPU).
#
# CCL isolation for vLLM:
#   - CCL_ATL_TRANSPORT=ofi (same as training — can't use mpi without MPI init)
#   - FI_PROVIDER=shm (shared memory — vLLM TP is intra-node only, no CXI needed)
#   - ZE_AFFINITY_MASK restricts vLLM to tiles 10-11 (no overlap with training 0-9)
#   This avoids CXI endpoint conflicts between vLLM's XCCL and training's XCCL.
VLLM_ENV="
cd ${TORCHTUNE_DIR}
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${VLLM_MASK}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH='${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
mkdir -p /tmp/torchtune
"

if [ ${VLLM_TILES} -gt 1 ]; then
    VLLM_START_CMD="${VLLM_ENV}
python3 -m vllm.entrypoints.openai.api_server \
    --model '${MODEL_PATH}' \
    --tensor-parallel-size ${VLLM_TILES} \
    --port ${VLLM_PORT} \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --distributed-executor-backend mp \
    > '${VLLM_LOG}' 2>&1
"
else
    VLLM_START_CMD="${VLLM_ENV}
python3 recipes/dev/vllm_serve_xpu.py \
    --model '${MODEL_PATH}' \
    --tensor_parallel_size 1 \
    --port ${VLLM_PORT} \
    --enforce_eager \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.80 \
    --max_model_len ${VLLM_MAX_MODEL_LEN} \
    > '${VLLM_LOG}' 2>&1
"
fi

echo "Starting vLLM on all nodes (TP=${VLLM_TILES}, tiles ${VLLM_MASK})..."
for node in "${UNIQUE_NODES[@]}"; do
    ssh "${node}" "${VLLM_START_CMD}" &
    VLLM_PIDS+=($!)
    echo "  vLLM launched on ${node} (PID $!)"
done

# Cleanup: kill vLLM on all nodes on exit
cleanup() {
    echo "Cleaning up vLLM on all nodes..."
    for node in "${UNIQUE_NODES[@]}"; do
        ssh "${node}" "pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null; \
                        pkill -f 'vllm.v1.engine' 2>/dev/null; \
                        pkill -f 'from multiprocessing' 2>/dev/null" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ============================================================
# Wait for vLLM health check on ALL nodes
# ============================================================
echo "Waiting for vLLM servers to become healthy..."
VLLM_TIMEOUT=600
for node in "${UNIQUE_NODES[@]}"; do
    ELAPSED=0
    while ! ssh "${node}" "curl -s http://localhost:${VLLM_PORT}/health/ > /dev/null 2>&1" 2>/dev/null; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [ ${ELAPSED} -ge ${VLLM_TIMEOUT} ]; then
            echo "ERROR: vLLM on ${node} did not start within ${VLLM_TIMEOUT}s"
            echo "=== vLLM log from ${node} ==="
            ssh "${node}" "tail -50 '${VLLM_LOG}'" 2>/dev/null || true
            exit 1
        fi
    done
    echo "  vLLM healthy on ${node} (${ELAPSED}s)"
done
echo "All vLLM servers ready."

# ============================================================
# Launch training via mpiexec
# ============================================================
# GPU affinity: FSDP2 (used for large models >50 GiB) doesn't need multi-tile
# visibility — each rank only materializes one module shard at a time (~3 GiB).
# Setting ZE_AFFINITY_MASK=$LOCAL_RANK prevents CXI conflicts with vLLM on tiles 10-11.
# For FSDP1 (small models), unset it so flatten can spill across tiles.
export USE_AFFINITY_MASK=${USE_AFFINITY_MASK:-1}

echo ""
echo "Starting HSDP GRPO training (${TOTAL_RANKS} ranks, USE_AFFINITY_MASK=${USE_AFFINITY_MASK})..."
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
    "data_parallel_replicate_dim=${NUM_NODES}" \
    "vllm_url=http://localhost:${VLLM_PORT}" \
    "vllm_weight_sync=false"

echo "=== Training complete ==="

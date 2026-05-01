#!/bin/bash
# 32B GRPO with 24-way pure FSDP + XCCL weight sync across 3 nodes.
#
# Architecture (3 PBS nodes):
#   Node 0 (vLLM only):    3 replicas × TP=4 = 12 tiles
#   Nodes 1-2 (training):  12 tiles/node × 2 = 24 tiles, pure FSDP (dp_replicate=1)
#   Weight sync: XCCL 2-hop broadcast (rank 0 → vLLM rank 1 → ranks 2-12)
#
# Memory budget per tile with 24-way FSDP (64 GiB available):
#   FSDP shard (32B / 24 tiles):  ~2.7 GiB
#   Ref model shard:              ~2.7 GiB
#   Optimizer states (2×):        ~5.3 GiB
#   Activations + overhead:       ~10-15 GiB
#   Total estimated:              ~21-26 GiB — massive headroom vs 12-way (~37 GiB)
#
# Uses SSH + torch.distributed.run with c10d rendezvous (no mpiexec needed).
# CCL_PROCESS_LAUNCHER=none (no PALS context from SSH sessions).
#
# Usage (from held 3-node job):
#   export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
#   bash recipes/dev/run_qwen3_32b_grpo_3node.sh
#
# Default config: recipes/configs/dev/production/qwen32B_grpo_3node_24way_stable_xpu.yaml
# (G=16/fbs=16/max_gen=128, ~41s/step, status.md Test A).
#
# Throughput envelope (~53s/step, Test B):
#   CONFIG=recipes/configs/dev/production/qwen32B_grpo_3node_24way_xpu.yaml \
#     bash recipes/dev/run_qwen3_32b_grpo_3node.sh
#
# Hard envelope: G/fbs <= 2; max_gen=192 marginal; max_gen>=256 OOMs;
# G=48+ hangs (CCL external explosion); G=64 hits XPU kernel indexing bug.
# See docs/features/qwen3_32b_dense_grpo.md for the full exemplar.
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TT_DIR="${TT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-${TT_DIR}/experiments/multinode_32b}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_qwen3_32b_grpo_3node_$(date +%Y%m%d_%H%M%S).log"

echo "=== Qwen3-32B 3-Node 24-Way FSDP GRPO ===" | tee "${LOG}"
echo "Date: $(date)  Host: $(hostname)" | tee -a "${LOG}"

# ============================================================
# Configuration
# ============================================================
VLLM_TP=${VLLM_TP:-4}
VLLM_DP=${VLLM_DP:-3}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
TRAIN_TILES_PER_NODE=${TRAIN_TILES_PER_NODE:-12}
MODEL_SRC=${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/$(basename "${MODEL_SRC}")}
NSTEPS=${NSTEPS:-5}
# G/FBS/MAX_GEN: when unset, the YAML's value wins (no CLI override emitted).
# When set, the env var overrides the YAML — useful for sweeps without
# editing the config.
GRPO_SAMPLES=${GRPO_SAMPLES:-}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-}
CONFIG=${CONFIG:-recipes/configs/dev/production/qwen32B_grpo_3node_24way_stable_xpu.yaml}
NUM_TRAIN_NODES=2
TOTAL_TRAIN_TILES=$((NUM_TRAIN_NODES * TRAIN_TILES_PER_NODE))

# Rendezvous port for torch.distributed.run c10d backend
RDZV_PORT=$((29400 + RANDOM % 1000))
MASTER_PORT=$((20000 + RANDOM % 20000))

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Run from a held PBS job." | tee -a "${LOG}"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "${PBS_NODEFILE}" | awk '!seen[$0]++'))
if [ "${#UNIQUE_NODES[@]}" -lt 3 ]; then
    echo "ERROR: Need 3 nodes. Got ${#UNIQUE_NODES[@]}: ${UNIQUE_NODES[*]}" | tee -a "${LOG}"
    exit 1
fi

VLLM_NODE="${UNIQUE_NODES[0]}"
TRAIN_NODES=("${UNIQUE_NODES[1]}" "${UNIQUE_NODES[2]}")
MASTER_NODE="${TRAIN_NODES[0]}"
# Use management network IP for rendezvous — HSN hostname resolves to 8 IPs
# (round-robin across Slingshot NICs) which breaks c10d TCPStore.
# CCL still uses HSN for collectives via CCL_KVS_IFACE=hsn0.
MASTER_ADDR=$(ssh "${MASTER_NODE}" "hostname -i" 2>/dev/null | head -1)
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "ERROR: Could not resolve IP for master training node ${MASTER_NODE}" | tee -a "${LOG}"
    exit 1
fi

VLLM_NODE_IP=$(ssh "${VLLM_NODE}" "hostname -i" 2>/dev/null | head -1)
if [[ -z "${VLLM_NODE_IP}" ]]; then
    echo "ERROR: Could not resolve IP for ${VLLM_NODE}" | tee -a "${LOG}"
    exit 1
fi

# HSN IP for XCCL weight sync: vLLM workers connect to this address via Slingshot
MASTER_HSN_IP=$(ssh "${MASTER_NODE}" "ip -4 addr show hsn0 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1 | head -1")
if [[ -z "${MASTER_HSN_IP}" ]]; then
    echo "WARNING: Could not get hsn0 IP for ${MASTER_NODE}; using MASTER_ADDR for XCCL" | tee -a "${LOG}"
    MASTER_HSN_IP="${MASTER_ADDR}"
fi
echo "MASTER_HSN_IP:  ${MASTER_HSN_IP} (for XCCL weight sync)" | tee -a "${LOG}"

echo "vLLM node:      ${VLLM_NODE} (IP=${VLLM_NODE_IP}, TP=${VLLM_TP}, DP=${VLLM_DP})" | tee -a "${LOG}"
echo "Train nodes:    ${TRAIN_NODES[*]} (${NUM_TRAIN_NODES} × ${TRAIN_TILES_PER_NODE} = ${TOTAL_TRAIN_TILES} tiles)" | tee -a "${LOG}"
echo "FSDP:           24-way pure FSDP (dp_replicate=1, dp_shard=24)" | tee -a "${LOG}"
echo "MASTER_ADDR:    ${MASTER_ADDR}" | tee -a "${LOG}"
echo "RDZV_PORT:      ${RDZV_PORT}" | tee -a "${LOG}"
echo "Config:         ${CONFIG}" | tee -a "${LOG}"
echo "Steps: ${NSTEPS}, G=${GRPO_SAMPLES}, FBS=${FORWARD_BATCH_SIZE}, max_gen=${MAX_GEN_TOKENS}" | tee -a "${LOG}"

export no_proxy="*"
export NO_PROXY="*"

# ============================================================
# Prepare PYTHONPATH
# ============================================================
cd "${TT_DIR}"
source recipes/dev/_aurora_paths.sh
VLLM_CUSTOMIZATION="${TT_DIR}/recipes/dev/_usercustomize_vllm"
VLLM_PYTHONPATH="$(aurora_pythonpath "${TT_DIR}" "${TRL_DIR}" "${VLLM_CUSTOMIZATION}")"
WORKER_EXT="torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension"
TRAIN_PYTHONPATH="$(aurora_pythonpath "${TT_DIR}")"

# ============================================================
# Stage model to all 3 nodes
# ============================================================
echo "Staging model to all nodes..." | tee -a "${LOG}"
STAGE_FAIL=0
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "  Copying to ${node}:${MODEL_PATH}..." | tee -a "${LOG}"
        ssh "${node}" "mkdir -p $(dirname "${MODEL_PATH}") && cp -r ${MODEL_SRC} ${MODEL_PATH}" &
    else
        echo "  Already staged on ${node}" | tee -a "${LOG}"
    fi
done
wait

# Verify staging succeeded on every node
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "  ERROR: staging failed on ${node} — retrying..." | tee -a "${LOG}"
        ssh "${node}" "rm -rf '${MODEL_PATH}' && mkdir -p $(dirname "${MODEL_PATH}") && cp -r ${MODEL_SRC} ${MODEL_PATH}" 2>&1 | tee -a "${LOG}"
        if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
            echo "  FATAL: staging failed on ${node} after retry" | tee -a "${LOG}"
            STAGE_FAIL=1
        fi
    fi
done
if [ "${STAGE_FAIL}" -eq 1 ]; then
    echo "Model staging FAILED — aborting" | tee -a "${LOG}"
    exit 1
fi
echo "Model staging verified on all nodes." | tee -a "${LOG}"

# ============================================================
# Pre-launch cleanup (kill stale vLLM, purge shared memory)
# ============================================================
echo "Cleaning stale processes and shared memory on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
sleep 2
rm -f /dev/shm/vllm* 2>/dev/null || true
" 2>/dev/null || true

# ============================================================
# Launch vLLM replicas on Node 0
# ============================================================
VLLM_PIDS=()
VLLM_URLS=""

echo "Starting ${VLLM_DP} vLLM replicas on ${VLLM_NODE} (TP=${VLLM_TP})..." | tee -a "${LOG}"
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    TILE_START=$((r * VLLM_TP))
    TILE_END=$((TILE_START + VLLM_TP - 1))
    TILE_MASK=$(seq -s, ${TILE_START} ${TILE_END})
    VLLM_LOG="/tmp/torchtune/vllm_replica_${r}.log"

    if [ -n "${VLLM_URLS}" ]; then
        VLLM_URLS="${VLLM_URLS},http://${VLLM_NODE_IP}:${PORT}"
    else
        VLLM_URLS="http://${VLLM_NODE_IP}:${PORT}"
    fi

    ssh "${VLLM_NODE}" "
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
# IMPORTANT: do NOT set PYTHONNOUSERSITE=1 for the vLLM process — it disables
# Python's autoload of usercustomize.py, which installs the registry-subprocess
# SIGSEGV patch. Without the patch, vllm 0.15 crashes during model architecture
# inspection. See feedback_usercustomize_eager_vllm_import.md.
export PYTHONNOUSERSITE=
export VLLM_SERVER_DEV_MODE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${TILE_MASK}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export PYTHONPATH='${VLLM_PYTHONPATH}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD}
mkdir -p /tmp/torchtune
python3 -m vllm.entrypoints.openai.api_server \
    --model '${MODEL_PATH}' \
    --tensor-parallel-size ${VLLM_TP} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --distributed-executor-backend mp \
    --worker-extension-cls ${WORKER_EXT} \
    > '${VLLM_LOG}' 2>&1
" &
    VLLM_PIDS+=($!)
    echo "  Replica ${r}: tiles ${TILE_MASK}, port ${PORT}" | tee -a "${LOG}"
done

echo "vLLM URLs: ${VLLM_URLS}" | tee -a "${LOG}"

# ============================================================
# Cleanup trap
# ============================================================
TRAIN_PIDS=()
cleanup() {
    echo "Cleaning up..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null
pkill -f 'vllm.v1.engine' 2>/dev/null
pkill -f 'from multiprocessing' 2>/dev/null
" 2>/dev/null || true
    for node in "${TRAIN_NODES[@]}"; do
        ssh "${node}" "pkill -f 'grpo_full_finetune_distributed_xpu' 2>/dev/null" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]}" "${TRAIN_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Cleanup done." | tee -a "${LOG}"
}
trap cleanup EXIT

# ============================================================
# Wait for vLLM health
# ============================================================
echo "Waiting for vLLM replicas..." | tee -a "${LOG}"
VLLM_TIMEOUT=600
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    ELAPSED=0
    while ! ssh -o ConnectTimeout=5 "${VLLM_NODE}" "curl --noproxy '*' -s --max-time 5 http://localhost:${PORT}/health/ > /dev/null 2>&1" 2>/dev/null; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [ "${ELAPSED}" -ge "${VLLM_TIMEOUT}" ]; then
            echo "ERROR: vLLM replica ${r} on port ${PORT} not ready within ${VLLM_TIMEOUT}s" | tee -a "${LOG}"
            ssh "${VLLM_NODE}" "tail -50 /tmp/torchtune/vllm_replica_${r}.log" 2>/dev/null | tee -a "${LOG}" || true
            exit 1
        fi
    done
    echo "  Replica ${r} healthy on port ${PORT} (${ELAPSED}s)" | tee -a "${LOG}"
done
echo "All ${VLLM_DP} vLLM replicas ready." | tee -a "${LOG}"

# ============================================================
# Launch training on 2 nodes via SSH + torch.distributed.run
# c10d rendezvous handles cross-node coordination (no mpiexec).
# ============================================================
echo "" | tee -a "${LOG}"
echo "Starting 24-way FSDP training on ${TRAIN_NODES[*]}..." | tee -a "${LOG}"

# Common training env block (shared by both nodes)
TRAIN_ENV="
set -e
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV

# CCL: multi-node ofi transport (no mpiexec = no pmix)
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
# DO NOT set CCL_REDUCE_SCATTER=ring — causes 63× regression on multi-node
export CCL_CHUNK_SIZE=16777216
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZES_ENABLE_SYSMAN=1

# Allocator: with 24-way FSDP, per-tile memory is ~half of 12-way.
# Start conservative with max_split_size_mb + gc threshold.
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
export TORCH_COMPILE_DISABLE=1

# Chunked loss: single backward (RC1 fix from Test CC)
export TORCHTUNE_USE_CHUNKED_LOSS=1

# IPEX varlen passthrough — silent no-op on dense Qwen3 (explicit causal mask
# is built in generate_trajectory). Set TORCHTUNE_USE_IPEX_VARLEN=1 in the
# caller env to confirm the one-shot \"varlen=requested-but-skipped\" log line.
export TORCHTUNE_USE_IPEX_VARLEN="${TORCHTUNE_USE_IPEX_VARLEN:-0}"

# Weight sync: gloo (CPU/TCP) cross-PG avoids XCCL deadlock with FSDP backward
export TORCHTUNE_XCCL_HOST='${MASTER_HSN_IP}'
export GLOO_SOCKET_IFNAME=hsn0
export WSYNC_CROSS_METHOD="${WSYNC_CROSS_METHOD:-gloo}"
export WSYNC_INTRA_METHOD="${WSYNC_INTRA_METHOD:-xccl}"
export TORCHTUNE_PINNED_CPU_BUF="${TORCHTUNE_PINNED_CPU_BUF:-1}"
export TORCHTUNE_D2H_STREAM="${TORCHTUNE_D2H_STREAM:-0}"

export no_proxy='*'
export NO_PROXY='*'

export PYTHONPATH='${TRAIN_PYTHONPATH}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
"

# Launch torch.distributed.run on each training node
for ((n=0; n<NUM_TRAIN_NODES; n++)); do
    NODE="${TRAIN_NODES[$n]}"
    echo "  Launching on ${NODE} (node_rank=${n})..." | tee -a "${LOG}"

    TRAIN_NODE_LOG="/tmp/torchtune/train_node${n}.log"
    # Build optional CLI overrides only when env vars are set, so the YAML wins by default.
    EXTRA_OVERRIDES=""
    [[ -n "${GRPO_SAMPLES}" ]] && EXTRA_OVERRIDES+=" grpo_samples=${GRPO_SAMPLES}"
    [[ -n "${FORWARD_BATCH_SIZE}" ]] && EXTRA_OVERRIDES+=" forward_batch_size=${FORWARD_BATCH_SIZE}"
    [[ -n "${MAX_GEN_TOKENS}" ]] && EXTRA_OVERRIDES+=" max_generated_tokens=${MAX_GEN_TOKENS}"
    ssh "${NODE}" "
${TRAIN_ENV}
python3 -m torch.distributed.run \
    --nnodes=${NUM_TRAIN_NODES} \
    --nproc-per-node=${TRAIN_TILES_PER_NODE} \
    --node-rank=${n} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${RDZV_PORT} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS}${EXTRA_OVERRIDES} \
    vllm_url='${VLLM_URLS}' \
    vllm_tensor_parallel_size=${VLLM_TP} \
    2>&1 | tee ${TRAIN_NODE_LOG}
" 2>&1 | sed "s/^/[node${n}:${NODE}] /" &
    TRAIN_PIDS+=($!)
done

echo "Waiting for training to complete..." | tee -a "${LOG}"

# Wait for all training processes — capture the first non-zero exit code
TRAIN_EXIT=0
for pid in "${TRAIN_PIDS[@]}"; do
    if ! wait "$pid"; then
        TRAIN_EXIT=$?
    fi
done

echo "=== 3-Node 24-Way FSDP Test: exit=${TRAIN_EXIT} at $(date) ===" | tee -a "${LOG}"
exit "${TRAIN_EXIT}"

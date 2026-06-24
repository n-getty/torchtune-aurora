#!/bin/bash
# BioReason 4B GRPO — 2-node Phase 1 split
# Node 0 (TRAIN_NODE):  11 training ranks (rank 0-10, LOCAL_RANK 0-10)
# Node 1 (VLLM_NODE):   1 vLLM rank (rank 11, LOCAL_RANK 0)
# Wsync: gloo PG [0, 11] over Slingshot
#
# Asymmetric launch: bypass torchrun (which requires uniform nproc-per-node).
# Each rank invoked directly via SSH with explicit RANK/WORLD_SIZE/LOCAL_RANK env.
set -o pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
NUM_STEPS=${1:-5}
WORLD_SIZE=12
VLLM_RANK=11

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Run from a held PBS job."
    exit 1
fi
UNIQUE_NODES=($(cut -d'.' -f1 "${PBS_NODEFILE}" | awk '!seen[$0]++'))
if [ "${#UNIQUE_NODES[@]}" -lt 2 ]; then
    echo "ERROR: Need 2 nodes, got ${#UNIQUE_NODES[@]}"
    exit 1
fi
TRAIN_NODE="${UNIQUE_NODES[0]}"
VLLM_NODE="${UNIQUE_NODES[1]}"

MASTER_HSN_IP=$(ssh "${TRAIN_NODE}" "ip -4 addr show hsn0 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1 | head -1")
if [[ -z "${MASTER_HSN_IP}" ]]; then
    echo "ERROR: Could not resolve hsn0 IP for ${TRAIN_NODE}"
    exit 1
fi
MASTER_PORT=$((20000 + RANDOM % 20000))

echo "=== BioReason 2-node split ==="
echo "TRAIN_NODE: ${TRAIN_NODE} (ranks 0-10)"
echo "VLLM_NODE:  ${VLLM_NODE} (rank ${VLLM_RANK})"
echo "MASTER:     ${MASTER_HSN_IP}:${MASTER_PORT}"
echo "num_steps:  ${NUM_STEPS}"

# ============================================================
# Stage model + cleanup
# ============================================================
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
MODEL_DST=/tmp/torchtune/bioreason-pro-sft
echo "=== Staging model ==="
for node in "${UNIQUE_NODES[@]}"; do
    ssh "${node}" "
if ! test -f '${MODEL_DST}/config.json'; then
    mkdir -p $(dirname ${MODEL_DST}) && cp -r ${MODEL_SRC} ${MODEL_DST}
fi
" &
done
wait

echo "=== Cleaning stale processes ==="
for node in "${UNIQUE_NODES[@]}"; do
    ssh "${node}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 -f 'grpo_bioreason_distributed_xpu' 2>/dev/null || true
sleep 2
rm -f /dev/shm/vllm* 2>/dev/null || true
" &
done
wait
sleep 2

# ============================================================
# Common env (sourced by every rank's shell)
# ============================================================
COMMON_ENV='
FW_BIN=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin
export PATH=${FW_BIN}:$(echo "$PATH" | tr ":" "\n" | grep -v myenv | tr "\n" ":" | sed "s/:$//")
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export INFRA_PROVIDER=local
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# CCL: multi-node ofi transport
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled

# Wsync: gloo over Slingshot
export TORCHTUNE_WSYNC_BACKEND=gloo
export GLOO_SOCKET_IFNAME=hsn0

# Recipe knobs
export TORCHTUNE_USE_CHUNKED_LOSS=1
export TORCHTUNE_PINNED_CPU_BUF=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.95

# Distributed env
export MASTER_ADDR='"${MASTER_HSN_IP}"'
export MASTER_PORT='"${MASTER_PORT}"'
export WORLD_SIZE='"${WORLD_SIZE}"'

export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/bioreason_deps:'"${PROJDIR}"':${PYTHONPATH}
export PYTHONUNBUFFERED=1
export no_proxy="*"
export NO_PROXY="*"

cd '"${PROJDIR}"'
source /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/setvars.sh 2>/dev/null || true
'

PYCMD="python3 recipes/dev/grpo_bioreason_distributed_xpu.py \
    --config recipes/configs/dev/production/bioreason_4b_grpo_dedicated_xpu.yaml \
    base_model_path=${MODEL_DST} \
    output_dir=${PROJDIR}/outputs/bioreason_2node_v1 \
    num_steps=${NUM_STEPS} \
    log_peak_memory_stats=true"

# ============================================================
# Launch 11 training ranks on TRAIN_NODE (rank 0-10, LOCAL_RANK 0-10)
# ============================================================
TRAIN_DIR=${PROJDIR}/experiments/bioreason
mkdir -p "${TRAIN_DIR}/run48_logs"
PIDS=()
echo "=== Launching 11 train ranks on ${TRAIN_NODE} ==="
for r in $(seq 0 10); do
    ssh "${TRAIN_NODE}" "
${COMMON_ENV}
export RANK=${r}
export LOCAL_RANK=${r}
export LOCAL_WORLD_SIZE=11
${PYCMD}
" > "${TRAIN_DIR}/run48_logs/rank${r}.log" 2>&1 &
    PIDS+=($!)
done

# ============================================================
# Launch 1 vLLM rank on VLLM_NODE (rank 11, LOCAL_RANK 0)
# ============================================================
echo "=== Launching vLLM rank ${VLLM_RANK} on ${VLLM_NODE} ==="
ssh "${VLLM_NODE}" "
${COMMON_ENV}
export RANK=${VLLM_RANK}
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
${PYCMD}
" > "${TRAIN_DIR}/run48_logs/rank${VLLM_RANK}.log" 2>&1 &
PIDS+=($!)

echo "PIDs: ${PIDS[*]}"
echo "Logs: ${TRAIN_DIR}/run48_logs/"

# Cleanup trap
cleanup() {
    echo "Cleaning up..."
    for node in "${UNIQUE_NODES[@]}"; do
        ssh "${node}" "pkill -9 -f 'grpo_bioreason_distributed_xpu' 2>/dev/null" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do kill ${pid} 2>/dev/null || true; done
}
trap cleanup EXIT

# Wait — capture first non-zero exit
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    wait ${pid} || EXIT_CODE=$?
done

echo "=== Done: exit=${EXIT_CODE} ==="
exit ${EXIT_CODE}

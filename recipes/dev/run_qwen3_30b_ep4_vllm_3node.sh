#!/bin/bash
# EP=4/DP=6 Qwen3-30B-A3B GRPO with dedicated vLLM node — 3-node variant of
# run_qwen3_30b_ep4_vllm_2node.sh.
#
# Architecture:
#   Train node 1 + Train node 2: 24 tiles total, EP=4 / DP_replicate=6 / DP_shard=4
#   vLLM node (VLLM_NODE env var):  12 tiles, vLLM 3xTP=4 serving generation
#
# Doubles the DP_replicate dim from 3→6 vs the 2-node v1-v7 path. Same EP shard
# (4 ranks per EP group, same 32 experts/tile), but 6 parallel replicas instead
# of 3. Lets us drop per-replica batch (or hold per-replica batch and double
# global batch) without rebuilding the EP layout.
#
# Usage:
#   On a train node (rank 0):
#     VLLM_NODE=<vllm_hostname> TRAIN_NODE2=<other_train_hostname> \
#       bash recipes/dev/run_qwen3_30b_ep4_vllm_3node.sh [num_steps]

set -eo pipefail

if [ -z "${VLLM_NODE}" ]; then
    echo "ERROR: VLLM_NODE must be set to the hostname of the dedicated vLLM node"
    echo "Usage: VLLM_NODE=<hostname> TRAIN_NODE2=<hostname> bash $0 [num_steps]"
    exit 1
fi
if [ -z "${TRAIN_NODE2}" ]; then
    echo "ERROR: TRAIN_NODE2 must be set to the second training node hostname"
    exit 1
fi

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL — same recipe as the 2-node run.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring
export CCL_ALLTOALL=naive
unset XPU_USM_ALLOC_SO
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99

# Gloo: loopback NOT viable across 2 train nodes. Use Slingshot HSN explicitly.
# The v153 GLOO_SOCKET_IFNAME=lo trick was 1-node-only — for cross-node EP gloo
# the 4 EP ranks may straddle nodes (depending on rank ordering), so gloo must
# go through the Slingshot NIC. Cross-node EP collectives are inherently slower
# but unavoidable at 2 train nodes.
# NOTE: do NOT set GLOO_DEVICE_TRANSPORT — this PyTorch build (frameworks/2025.3.1)
# does not recognize the env-var-controlled factory and ProcessGroupGloo()
# raises `makeDeviceForInterface(): unsupported gloo device`. GLOO_SOCKET_IFNAME
# alone routes through the named interface via the default tcp transport.
export GLOO_SOCKET_IFNAME=hsn0

FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export no_proxy="*"
export NO_PROXY="*"

NPROC=12          # tiles per node
NNODES=2          # train nodes
WORLD=$((NPROC * NNODES))
NSTEPS=${1:-5}
VLLM_BASE_PORT=8001
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
CONFIG=${PROJDIR}/recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep4_xpu.yaml
VLLM_ADDR=${VLLM_ADDR:-${VLLM_NODE}}
VLLM_URLS="http://${VLLM_ADDR}:${VLLM_BASE_PORT},http://${VLLM_ADDR}:$((VLLM_BASE_PORT+1)),http://${VLLM_ADDR}:$((VLLM_BASE_PORT+2))"

JOB_TAG="${PBS_JOBID:-$$}"
MASTER_PORT=$(( 29500 + ( $(echo "${JOB_TAG}" | tr -dc '0-9' | tail -c 4) % 400 ) ))
MASTER_ADDR=$(hostname -i | awk '{print $1}')

echo "=== EP=4/DP=6 Qwen3-30B-A3B GRPO + Dedicated vLLM Node (3-node) ==="
echo "Train rank-0 node: $(hostname) (${MASTER_ADDR})"
echo "Train rank-1 node: ${TRAIN_NODE2}"
echo "vLLM node: ${VLLM_NODE} (addr=${VLLM_ADDR}, 3xTP=4)"
echo "World size: ${WORLD} (${NNODES} x ${NPROC})"
echo "vLLM URLs: ${VLLM_URLS}"
echo "Config: ${CONFIG}"
echo "Steps: ${NSTEPS}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT} (job tag ${JOB_TAG})"
echo "Date: $(date)"

mkdir -p /tmp/torchtune

# Stage model on BOTH train nodes
LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
stage_model() {
    local node="$1"
    ssh -o StrictHostKeyChecking=no "${node}" "
        if [ ! -f '${LOCAL_MODEL}/config.json' ]; then
            mkdir -p /tmp/torchtune
            t0=\$SECONDS
            cp -r '${MODEL_PATH}' '${LOCAL_MODEL}'
            echo \"Staged on ${node} in \$((SECONDS - t0))s\"
        else
            echo 'Model already staged on ${node} at ${LOCAL_MODEL}'
        fi
    "
}
stage_model "$(hostname)"
stage_model "${TRAIN_NODE2}"

# vLLM health check
for PORT in ${VLLM_BASE_PORT} $((VLLM_BASE_PORT+1)) $((VLLM_BASE_PORT+2)); do
    echo "Checking vLLM health at http://${VLLM_ADDR}:${PORT}/health/ ..."
    ELAPSED=0
    while ! curl -s --max-time 5 http://${VLLM_ADDR}:${PORT}/health/ > /dev/null 2>&1; do
        sleep 10; ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge 2400 ]; then
            echo "ERROR: vLLM on ${VLLM_ADDR}:${PORT} did not respond within 2400s"
            exit 1
        fi
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  Waiting for port ${PORT}... ${ELAPSED}s"
    done
    echo "  Port ${PORT} ready (waited ${ELAPSED}s)"
done
echo "All 3 vLLM replicas ready on ${VLLM_ADDR}"

# DP_replicate override: 24 / 4 = 6 replicas (vs 3 in 2-node)
EXTRA="data_parallel_replicate_dim=6 ${EXTRA_OVERRIDES:-}"

run_torchrun() {
    local node="$1" rank="$2"
    ssh -o StrictHostKeyChecking=no "${node}" "
        cd ${PROJDIR}
        module load frameworks/2025.3.1 2>/dev/null || true
        export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
        unset VIRTUAL_ENV
        $(env | grep -E '^(CCL_|FI_|ZE_|GLOO_|PYTHON|HF_|XPU_|PYTORCH_|no_proxy|NO_PROXY)' | sed 's/^/export /')
        export VLLM_ADDR=${VLLM_ADDR}
        torchrun \
            --nproc_per_node=${NPROC} \
            --nnodes=${NNODES} \
            --node_rank=${rank} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
            --config ${CONFIG} \
            base_model_path=${LOCAL_MODEL} \
            num_steps=${NSTEPS} \
            'vllm_url=${VLLM_URLS}' \
            save_every_n_epochs=100 \
            save_final_checkpoint=false \
            ${EXTRA} \
            2>&1
    "
}

echo "Launching torchrun on rank-1 node ${TRAIN_NODE2}..."
run_torchrun "${TRAIN_NODE2}" 1 &
RANK1_PID=$!

echo "Launching torchrun on rank-0 node $(hostname)..."
run_torchrun "$(hostname)" 0
RANK0_RC=$?

wait ${RANK1_PID}
RANK1_RC=$?

echo "=== Training complete (rank0 rc=${RANK0_RC}, rank1 rc=${RANK1_RC}) ==="
exit ${RANK0_RC}

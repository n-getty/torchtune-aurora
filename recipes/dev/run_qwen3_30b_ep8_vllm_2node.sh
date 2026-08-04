#!/bin/bash
# EP=8 / DP_replicate=1 Qwen3-30B-A3B GRPO with dedicated vLLM node — 2-node
# variant. Topology:
#   1 train node: 8 of 12 tiles, EP=8 dp_replicate=1 dp_shard=8 (4 tiles idle)
#   1 vLLM node:  12 tiles, vLLM 3xTP=4 on ports 8001-8003
#
# Why 2-node single-train: smoke for the v9-helper + EP=8 (16 experts/tile,
# 1/8 non-expert shard, ~625M sharded params per rank). Tests that plain
# torch.optim.AdamW fits the headroom — no CPU bounce — without the cross-node
# straddling EP group of the 3-node EP=8 plan. dp_replicate=1 means the v75
# XCCL grad sync is a no-op, but that path was already validated end-to-end at
# EP=4 in v9b10.
#
# Usage:
#   On training node:
#     VLLM_NODE=<vllm_hostname> bash recipes/dev/run_qwen3_30b_ep8_vllm_2node.sh [num_steps]
#   On vLLM node (run first; wait for "All vLLM replicas ready"):
#     bash recipes/dev/run_qwen3_30b_vllm_server.sh

set -eo pipefail

if [ -z "${VLLM_NODE}" ]; then
    echo "ERROR: VLLM_NODE must be set to the hostname of the dedicated vLLM node"
    echo "Usage: VLLM_NODE=<hostname> bash $0 [num_steps]"
    exit 1
fi

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL — same recipe as the EP=4 2-node run. EP=8 collective stays on-node.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-65536}
echo "CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD}"
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring
export CCL_ALLTOALL=naive
unset XPU_USM_ALLOC_SO
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99

# Cross-node gloo route: the EP=8 collective itself stays on-node (comment
# above), but TORCHTUNE_EP_WSYNC_SHARDED=1 (WS10) opens gloo cross-PGs
# straight to the SEPARATE dedicated vLLM node — genuine cross-node traffic
# that needs Slingshot, unlike the intra-node _shard_pg gloo this launcher
# used before. Without this, gloo silently defaults to the 1 Gbps mgmt NIC
# (or fails to route at all) and the WS10 broadcast hangs at 0.0 GB/s until
# the 1800s HTTP timeout (job 8682162, cellB, 2026-07-21) even though the
# sender-side gather worked perfectly (6.75 GiB/rank in 9.1s, vs 56.87 GiB/
# 25s on the legacy full-reship path). Every OTHER multi-node EP launcher
# (ep16, ep8_3node, ep4_3node, 32b) already sets this; this one was the one
# omission because it never needed cross-node gloo before WS10.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}

FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export no_proxy="*"
export NO_PROXY="*"

NPROC=8           # 8 of 12 tiles for EP=8 dp_shard=8 single-replica
NSTEPS=${1:-5}
VLLM_BASE_PORT=8001
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
CONFIG=${PROJDIR}/recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep8_xpu.yaml
VLLM_ADDR=${VLLM_ADDR:-${VLLM_NODE}}
VLLM_URLS="http://${VLLM_ADDR}:${VLLM_BASE_PORT},http://${VLLM_ADDR}:$((VLLM_BASE_PORT+1)),http://${VLLM_ADDR}:$((VLLM_BASE_PORT+2))"

JOB_TAG="${PBS_JOBID:-$$}"
LAST4=$(echo "${JOB_TAG}" | tr -dc '0-9' | tail -c 4)
MASTER_PORT=$(( 29500 + ( 10#${LAST4:-0} % 400 ) ))

echo "=== EP=8 / dp_replicate=1 Qwen3-30B-A3B GRPO + Dedicated vLLM Node (2-node) ==="
echo "Training node: $(hostname) (8 of 12 tiles, EP=8 dp_shard=8, 4 tiles idle)"
echo "vLLM node: ${VLLM_NODE} (addr=${VLLM_ADDR}, 3xTP=4)"
echo "vLLM URLs: ${VLLM_URLS}"
echo "Config: ${CONFIG}"
echo "Steps: ${NSTEPS}"
echo "Master port: ${MASTER_PORT} (job tag ${JOB_TAG})"
echo "Date: $(date)"

mkdir -p /tmp/torchtune

LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi

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

# Override dp_replicate=1 for single-node 8-tile layout (EP=8 * dp_replicate * dp_shard
# must equal world size; 8 tiles → dp_replicate=1, dp_shard=8). The yaml is
# parameterized for dp_replicate=3 in the 3-node 24-tile case.
EXTRA="data_parallel_replicate_dim=1 ${EXTRA_OVERRIDES:-}"

# GroupedExpertsHF's padded-BMM forward allocates a fresh [E, max_count, dim]
# tensor every call (48 layers/step); at forward_batch_size>=4 this exhausts
# the XPU L0 handle table (UR_RESULT_ERROR_OUT_OF_RESOURCES / banned:1).
# HW-validated fix (job 8681553, 2026-07-21): TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=1
# (no padded allocation) clears G=4/fbs=4 AND G=8/fbs=4, 4/4 steps each. Only
# auto-enable at fbs>=4 — the padded-BMM path is a proven 6.3x speedup and
# stays default at fbs<=2 where it doesn't crash. Caller can still force
# either way by exporting TORCHTUNE_MOE_SEQUENTIAL_EXPERTS before invoking
# this script. See memory/project_qwen3_moe_padded_bmm_l0_exhaustion_fix_20260721.md.
if [ -z "${TORCHTUNE_MOE_SEQUENTIAL_EXPERTS+x}" ]; then
    _FBS=$(echo "${EXTRA}" | grep -oE 'forward_batch_size=[0-9]+' | tail -1 | grep -oE '[0-9]+')
    if [ -n "${_FBS}" ] && [ "${_FBS}" -ge 4 ]; then
        export TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=1
        echo "forward_batch_size=${_FBS} >= 4: auto-enabling TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=1"
    fi
fi

# HSN IP for XCCL weight sync: vLLM (on a DIFFERENT node) connects to this
# address. --master_addr=localhost below is fine for torchrun's own
# single-node rendezvous, but _init_xccl_weight_sync falls back to
# MASTER_ADDR when TORCHTUNE_XCCL_HOST is unset — "localhost" would tell
# the vLLM workers to dial themselves, hanging for 120s then failing
# (see feedback_ab_harness_bugs_not_infra.md point 3/reproduced 2026-07-21).
MASTER_HSN_IP=$(ip -4 addr show hsn0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1 | head -1)
if [ -z "${MASTER_HSN_IP}" ]; then
    echo "WARNING: could not get hsn0 IP; falling back to hostname for XCCL"
    MASTER_HSN_IP=$(hostname)
fi
export TORCHTUNE_XCCL_HOST="${MASTER_HSN_IP}"
echo "TORCHTUNE_XCCL_HOST=${TORCHTUNE_XCCL_HOST} (for cross-node XCCL weight sync)"

# WSYNC_TOPOLOGY/WSYNC_CROSS_METHOD: TESTED node_fanout+gloo here 2026-07-21
# (A/B job 8681387, 4/4 clean steps each leg, both GREEN, same path/transport)
# and it's a ~15% REGRESSION at this topology (3 replicas, 1 vLLM node):
# baseline (unset -> replica_fanout+xccl_sendrecv) steady-state 77.9-78.0s/step
# (wsync ~48.6s, XCCL RDMA send/recv 3 replicas in parallel at 2.4-2.6 GB/s
# each) vs node_fanout+gloo 90.0-90.2s/step (wsync ~62s, single gloo TCP
# cross-node hop degrading 24.0s->30.0s->37.4s->37.6s across steps at
# 2.4->1.5 GB/s, THEN a serialized intra-node XCCL fanout to replicas 1-2).
# BioReason's DP=12 win doesn't transfer here — with only 3 replicas the
# serialization cost of the two-hop path outweighs cutting cross-NIC fanout,
# and gloo TCP itself is slower per-byte than XCCL RDMA on this fabric.
# Leave UNSET (replica_fanout+xccl_sendrecv, the code default) for this
# launcher. See memory/project_wsync_node_fanout_regression_ep8_20260721.md.
export WSYNC_TOPOLOGY=${WSYNC_TOPOLOGY:-replica_fanout}
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-xccl_sendrecv}
echo "WSYNC_TOPOLOGY=${WSYNC_TOPOLOGY} WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD}"

echo "Starting EP=8 Qwen3 training on 8 tiles..."
torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${LOCAL_MODEL} \
    num_steps=${NSTEPS} \
    "vllm_url=${VLLM_URLS}" \
    save_every_n_epochs=100 \
    save_final_checkpoint=false \
    ${EXTRA} \
    2>&1

echo "=== Training complete ==="

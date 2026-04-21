#!/bin/bash
# EP=4/DP=3 GRPO with dedicated vLLM node — the target "best MoE" configuration.
#
# Architecture:
#   Training node (this script): 12 tiles, EP=4/DP=3 torchtune GRPO
#   vLLM node (VLLM_NODE env var): 12 tiles, vLLM TP=4 (or TP=2) serving generation
#
# Usage:
#   On training node:
#     VLLM_NODE=<vllm_hostname> bash recipes/dev/run_ep4_vllm_2node.sh [num_steps]
#   On vLLM node (run first, wait for "vLLM ready"):
#     bash recipes/dev/run_vllm_server_26b.sh
#
# Comparison target:
#   EP=1 (10+2 tile, same node): ~24s/step at grpo_samples=4
#   EP=4 (12 tile training + dedicated vLLM): target < 20s/step at grpo_samples=4,
#     then scale grpo_samples to 8/16 to demonstrate MoE memory advantage.

set -eo pipefail

if [ -z "${VLLM_NODE}" ]; then
    echo "ERROR: VLLM_NODE must be set to the hostname of the dedicated vLLM node"
    echo "Usage: VLLM_NODE=<hostname> bash $0 [num_steps]"
    exit 1
fi

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.2.0 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL for 12-tile EP training (same as nollm EP=4 run that achieved step 0)
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=8192
export CCL_CHUNK_SIZE=16777216
# v54: AllReduce ring (standard) — the monkey-patch in the recipe intercepts ALL
#   reduce_scatter_tensor calls and performs AllReduce on CPU (host) memory instead
#   of XPU. This bypasses CCL's XPU memory registration entirely.
#   CCL_ALLREDUCE=ring applies only to: barriers, loss/metric allreduces (on small
#   CPU-side tensors), and any other non-grad collectives — all safe with ring.
#   CCL_ALLREDUCE=ring is NOT used for FSDP2 grad AllReduce (intercepted by patch).
#   Prior failures: ring→ze_handle_manager (L0 IPC for XPU grad); nreduce→SEND EPERM
#   (OFI GPU-direct for XPU grad). Both fail for freshly sub-allocated XPU tensors.
export CCL_ALLREDUCE=ring
# v46: AllToAll naive (OFI) — gradient tensors fail L0 IPC (ze_handle_manager).
#   New gradient tensor addresses not in CCL IPC cache → zeMemOpenIpcHandle fails.
#   naive uses OFI p2p sends instead of L0 IPC → avoids ze_handle_manager.
export CCL_ALLTOALL=naive
# AllGather: TOPO (default, L0 IPC) — parameter all_gather_output buffers use long-lived
#   base memory blocks whose IPC handles are stable throughout training.
#   L0 topo AllGather is ~7-8× faster than naive OFI → keeps policy_fwd fast.
#   v125 diagnostic: CCL_ALLGATHER=naive did NOT fix backward SIGSEGV (root cause was
#   FSDP2 backward hooks from ref model, not IPC handle caching). Reverted in v126.
#   (CCL_ALLGATHER=naive caused policy_fwd=23.9s vs 21.7s, ref_fwd=8.2s vs 3.8s.)
unset PYTORCH_ALLOC_CONF

FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.2.0/lib/python3.10/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# Bypass Aurora's Squid HTTP proxy for all intra-cluster traffic.
# Without this, requests to internal IPs (10.x.x.x) go through proxy.alcf.anl.gov
# and get 503 errors. This affects VLLMClient's HTTP calls to the vLLM server.
export no_proxy="*"
export NO_PROXY="*"

NPROC=12
NSTEPS=${1:-5}
VLLM_BASE_PORT=8001   # replicas on 8001, 8002, 8003
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
CONFIG=${PROJDIR}/recipes/configs/dev/experimental/gemma4_26b_grpo_ep4_xpu.yaml
# VLLM_ADDR: use IP instead of hostname to bypass Aurora's Squid proxy.
# Set by PBS wrapper via: export VLLM_ADDR=$(ssh ${VLLM_NODE} "hostname -i" | head -1)
# Falls back to VLLM_NODE hostname for interactive use.
VLLM_ADDR=${VLLM_ADDR:-${VLLM_NODE}}
# All 3 vLLM replica URLs (comma-separated; recipe calls them in parallel)
VLLM_URLS="http://${VLLM_ADDR}:${VLLM_BASE_PORT},http://${VLLM_ADDR}:$((VLLM_BASE_PORT+1)),http://${VLLM_ADDR}:$((VLLM_BASE_PORT+2))"

echo "=== EP=4/DP=3 GRPO + Dedicated vLLM Node ==="
echo "Training node: $(hostname) (12 tiles, EP=4/DP=3)"
echo "vLLM node: ${VLLM_NODE} (addr=${VLLM_ADDR}, 3×TP=4)"
echo "vLLM URLs: ${VLLM_URLS}"
echo "Config: ${CONFIG}"
echo "Steps: ${NSTEPS}"
echo "Date: $(date)"
python3 -c "import torchao; print('torchao:', torchao.__version__)"

mkdir -p /tmp/torchtune

# Stage model to /tmp (fast checkpoint loading vs Lustre)
LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi

# Wait for all 3 vLLM replicas to be ready
for PORT in ${VLLM_BASE_PORT} $((VLLM_BASE_PORT+1)) $((VLLM_BASE_PORT+2)); do
    echo "Checking vLLM health at http://${VLLM_ADDR}:${PORT}/health/ ..."
    ELAPSED=0
    while ! curl -s http://${VLLM_ADDR}:${PORT}/health/ > /dev/null 2>&1; do
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

echo "Starting EP=4 training on 12 tiles..."
torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${LOCAL_MODEL} \
    num_steps=${NSTEPS} \
    "vllm_url=${VLLM_URLS}" \
    save_every_n_epochs=100 \
    save_final_checkpoint=false \
    2>&1

echo "=== Training complete ==="

#!/bin/bash
# EP=4/DP=3 Qwen3-30B-A3B GRPO with dedicated vLLM node — Qwen3 mirror of
# run_ep4_vllm_2node.sh. Architecture is identical to the Gemma4 EP run; only the
# config, model path, and vLLM server script differ.
#
# Architecture:
#   Training node (this script): 12 tiles, EP=4/DP=3 torchtune GRPO on Qwen3-30B-A3B
#   vLLM node (VLLM_NODE env var): 12 tiles, vLLM 3×TP=4 serving generation
#
# Usage:
#   On training node:
#     VLLM_NODE=<vllm_hostname> bash recipes/dev/run_qwen3_30b_ep4_vllm_2node.sh [num_steps]
#   On vLLM node (run first; wait for "All vLLM replicas ready"):
#     bash recipes/dev/run_qwen3_30b_vllm_server.sh
#
# Master port: derived from PBS_JOBID (or PID fallback) so re-runs on the same
# node don't collide on a stuck 29500 from a previous torchrun (the v162 failure).

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

# CCL for 12-tile EP training (matches the gemma4 EP launcher; already validated
# end-to-end with the EP=4 v161 attempt — see torchtune/modules/moe/_parallelism.py
# for why these specific values are required).
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
# v3 (2026-04-28): bumped 8192 → 65536 to match production 32B launcher.
# Production reference: experiments/multinode_32b/run_32b_2hop_production.sh
# v2 banned:1 at op ~423 (forward) suggests IPC-handle eviction.
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring
export CCL_ALLTOALL=naive
# v3: production uses default allocator + GC=0.99 (NOT pluggable USM cache).
# v2 inherited XPU_USM_ALLOC_SO from the held shell — known to cause banned:1.
unset XPU_USM_ALLOC_SO
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99

FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export no_proxy="*"
export NO_PROXY="*"

NPROC=12
NSTEPS=${1:-5}
VLLM_BASE_PORT=8001
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
CONFIG=${PROJDIR}/recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep4_xpu.yaml
VLLM_ADDR=${VLLM_ADDR:-${VLLM_NODE}}
VLLM_URLS="http://${VLLM_ADDR}:${VLLM_BASE_PORT},http://${VLLM_ADDR}:$((VLLM_BASE_PORT+1)),http://${VLLM_ADDR}:$((VLLM_BASE_PORT+2))"

# Per-run master port — avoids 29500 EADDRINUSE from a previously-killed torchrun
# still holding the listening socket (the v162 failure mode).
JOB_TAG="${PBS_JOBID:-$$}"
MASTER_PORT=$(( 29500 + ( $(echo "${JOB_TAG}" | tr -dc '0-9' | tail -c 4) % 400 ) ))

echo "=== EP=4/DP=3 Qwen3-30B-A3B GRPO + Dedicated vLLM Node ==="
echo "Training node: $(hostname) (12 tiles, EP=4/DP=3)"
echo "vLLM node: ${VLLM_NODE} (addr=${VLLM_ADDR}, 3×TP=4)"
echo "vLLM URLs: ${VLLM_URLS}"
echo "Config: ${CONFIG}"
echo "Steps: ${NSTEPS}"
echo "Master port: ${MASTER_PORT} (job tag ${JOB_TAG})"
echo "Date: $(date)"
python3 -c "import torchao; print('torchao:', torchao.__version__)" 2>/dev/null || true

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

echo "Starting EP=4 Qwen3 training on 12 tiles..."
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
    ${EXTRA_OVERRIDES:-} \
    2>&1

echo "=== Training complete ==="

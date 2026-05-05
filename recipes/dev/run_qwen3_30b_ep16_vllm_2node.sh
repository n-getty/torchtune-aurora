#!/bin/bash
# EP=16/dp_replicate=1 Qwen3-30B-A3B GRPO with vLLM colocated on each train node.
#
# Architecture (2 nodes, split-tile-per-node):
#   Train: 16 tiles total (8 per node, tiles 0-7), EP=16, dp_shard=16, dp_replicate=1
#   vLLM:  8 tiles total (4 per node, tiles 8-11), 1xTP=4 server per node = 2 servers
#
# Cross-node EP layout: with 16 train ranks and EP=16, ALL 16 belong to ONE
# EP group spanning both nodes. Every EP collective straddles hsn0.
# (EP=8 3-node had 1-of-3 EP groups straddling; here it's 1-of-1.)
# GLOO_SOCKET_IFNAME=hsn0 routes that traffic. Default path is gloo CPU bounce
# (_GLOO_EP_PG); XCCL is opt-in via TORCHTUNE_EP_USE_XCCL=1 (unvalidated for
# cross-node EP — use experiments/ep_parallelism/run_ep16_smoke.sh for the
# 3-phase A/B sweep).
#
# Per-rank tile pinning is required because vLLM owns tiles 8-11 on the SAME
# nodes. The recipe device discovery uses LOCAL_RANK -> xpu:LOCAL_RANK; we
# wrap each torchrun child with a small script that sets
# ZE_AFFINITY_MASK=$LOCAL_RANK so train rank N sees only tile N.
#
# Usage (on rank-0 train node, after holding 2 nodes):
#   TRAIN_NODE2=<other_train_hostname> bash recipes/dev/run_qwen3_30b_ep16_vllm_2node.sh [num_steps]
#
# Optional env passthrough (used by experiments/ep_parallelism/run_ep16_smoke.sh):
#   TORCHTUNE_EP_USE_XCCL=1            # XCCL EP AG/RS instead of gloo CPU bounce
#   TORCHTUNE_EP_GRAD_RELEASE_XCCL=1   # XCCL all_reduce in _ep_release helper
#   EXTRA_OVERRIDES="..."              # appended to the recipe CLI
#   SMOKE_TAG="A|B|C"                  # tag prefix for log files

set -eo pipefail

if [ -z "${TRAIN_NODE2}" ]; then
    echo "ERROR: TRAIN_NODE2 must be set to the second training node hostname"
    echo "Usage: TRAIN_NODE2=<hostname> bash $0 [num_steps]"
    exit 1
fi

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL — same recipe as the EP=8 3-node run. SSH+torchrun (--standalone path),
# so use `none`/`ofi`, NOT `pmix`/`mpi`.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=0
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring
export CCL_ALLTOALL=naive
unset XPU_USM_ALLOC_SO
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99

# Cross-node EP gloo: route via Slingshot HSN.
export GLOO_SOCKET_IFNAME=hsn0

FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export no_proxy="*"
export NO_PROXY="*"

NPROC=8           # train tiles per node (0-7); vLLM owns 8-11
NNODES=2          # train nodes
WORLD=$((NPROC * NNODES))
NSTEPS=${1:-3}
VLLM_PORT=8001
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
CONFIG=${CONFIG:-${PROJDIR}/recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep16_xpu.yaml}

NODE0=$(hostname)
NODE1=${TRAIN_NODE2}
NODE0_ADDR=$(hostname -i | awk '{print $1}')
NODE1_ADDR=$(getent hosts ${TRAIN_NODE2} | awk '{print $1}')
[ -z "${NODE1_ADDR}" ] && NODE1_ADDR=${TRAIN_NODE2}

VLLM_HOST_0=${NODE0_ADDR}
VLLM_HOST_1=${NODE1_ADDR}
VLLM_URLS="http://${VLLM_HOST_0}:${VLLM_PORT},http://${VLLM_HOST_1}:${VLLM_PORT}"

JOB_TAG="${PBS_JOBID:-$$}"
LAST4=$(echo "${JOB_TAG}" | tr -dc '0-9' | tail -c 4)
MASTER_PORT=$(( 29500 + ( 10#${LAST4:-0} % 400 ) ))
MASTER_ADDR=${NODE0_ADDR}

SMOKE_TAG=${SMOKE_TAG:-ep16}
LOG_DIR=${PROJDIR}/experiments/ep_parallelism
mkdir -p ${LOG_DIR}

echo "=== EP=16/dp_replicate=1 Qwen3-30B-A3B GRPO + colocated vLLM (2-node) ==="
echo "Train rank-0 node: ${NODE0} (${NODE0_ADDR})"
echo "Train rank-1 node: ${NODE1} (${NODE1_ADDR})"
echo "Train tiles per node: 0-7 (NPROC=${NPROC})"
echo "vLLM tiles per node: 8-11 (TP=4, port ${VLLM_PORT})"
echo "vLLM URLs: ${VLLM_URLS}"
echo "World size: ${WORLD} (${NNODES} x ${NPROC})"
echo "Config: ${CONFIG}"
echo "Steps: ${NSTEPS}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT} (job tag ${JOB_TAG})"
echo "Smoke tag: ${SMOKE_TAG}"
echo "TORCHTUNE_EP_USE_XCCL=${TORCHTUNE_EP_USE_XCCL:-0}"
echo "TORCHTUNE_EP_GRAD_RELEASE_XCCL=${TORCHTUNE_EP_GRAD_RELEASE_XCCL:-0}"
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
stage_model "${NODE0}"
stage_model "${NODE1}"

# Per-rank wrapper that pins ZE_AFFINITY_MASK to $LOCAL_RANK and execs the recipe.
# torchrun sets LOCAL_RANK before forking each child; the wrapper must be
# present on both nodes (shared FS).
# Use launch-unique wrapper path: re-running on the same hold while a prior
# instance's process table entries linger leaves the kernel briefly considering
# the wrapper image "busy as text" → ETXTBSY on the next `cat >`. PID + epoch
# guarantees a fresh inode every launch.
WRAPPER=${LOG_DIR}/_ep16_train_rank_wrapper_${SMOKE_TAG}_$$_$(date +%s).sh
cat > ${WRAPPER} <<'WRAPPEREOF'
#!/bin/bash
# Per-rank wrapper: pin to tile $LOCAL_RANK then exec the recipe.
# Train ranks 0..7 each see exactly one of tiles 0..7; vLLM owns 8..11.
export ZE_AFFINITY_MASK=${LOCAL_RANK}
exec python3 "$@"
WRAPPEREOF
chmod +x ${WRAPPER}

# Launch one TP=4 vLLM replica on each train node, pinned to tiles 8-11.
launch_vllm_on_node() {
    local node="$1"
    local log_file=${LOG_DIR}/${SMOKE_TAG}_vllm_${node}.log
    echo "Launching vLLM TP=4 on ${node} tiles 8-11, port ${VLLM_PORT}, log ${log_file}"
    # nohup + setsid so the EngineCore survives the SSH parent dropping.
    ssh -o StrictHostKeyChecking=no "${node}" "
        cd ${PROJDIR}
        module load frameworks/2025.3.1 2>/dev/null || true
        export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
        unset VIRTUAL_ENV
        export ZE_FLAT_DEVICE_HIERARCHY=FLAT
        unset PYTORCH_ALLOC_CONF
        # WS5: wsync requires VLLM_SERVER_DEV_MODE (unlocks /collective_rpc),
        # _usercustomize_vllm on PYTHONPATH (registers WeightSyncFromFileExtension
        # routes), and PYTHONNOUSERSITE unset (so usercustomize.py autoloads).
        export VLLM_SERVER_DEV_MODE=1
        export PYTHONNOUSERSITE=
        export PYTHONPATH=${PROJDIR}:${PROJDIR}/recipes/dev/_usercustomize_vllm:${FW_SITE}
        export HF_DATASETS_OFFLINE=1
        export HF_HUB_OFFLINE=1
        export ZE_AFFINITY_MASK=8,9,10,11
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export TORCH_COMPILE_DISABLE=1
        export PYTORCH_ALLOC_CONF=
        export CCL_PROCESS_LAUNCHER=none
        export CCL_ATL_TRANSPORT=ofi
        export FI_PROVIDER=cxi
        export CCL_KVS_IFACE=lo
        nohup setsid python3 -m vllm.entrypoints.openai.api_server \
            --model ${LOCAL_MODEL} \
            --tensor-parallel-size 4 \
            --port ${VLLM_PORT} \
            --host 0.0.0.0 \
            --enforce-eager \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.80 \
            --max-model-len 2048 \
            --distributed-executor-backend mp \
            --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \
            > ${log_file} 2>&1 < /dev/null &
        echo \"vLLM launched on \$(hostname) PID \$!\"
    "
}
launch_vllm_on_node "${NODE0}"
launch_vllm_on_node "${NODE1}"

# Wait for both vLLM replicas to be healthy.
for HOST_ADDR in ${VLLM_HOST_0} ${VLLM_HOST_1}; do
    echo "Checking vLLM health at http://${HOST_ADDR}:${VLLM_PORT}/health/ ..."
    ELAPSED=0
    while ! curl -s --max-time 5 http://${HOST_ADDR}:${VLLM_PORT}/health/ > /dev/null 2>&1; do
        sleep 10; ELAPSED=$((ELAPSED + 10))
        if [ ${ELAPSED} -ge 2400 ]; then
            echo "ERROR: vLLM on ${HOST_ADDR}:${VLLM_PORT} did not respond within 2400s"
            exit 1
        fi
        [ $((ELAPSED % 60)) -eq 0 ] && echo "  Waiting for ${HOST_ADDR}... ${ELAPSED}s"
    done
    echo "  ${HOST_ADDR}:${VLLM_PORT} ready (waited ${ELAPSED}s)"
done
echo "Both vLLM replicas ready."

EXTRA="${EXTRA_OVERRIDES:-}"

run_torchrun() {
    local node="$1" rank="$2"
    local log_file=${LOG_DIR}/${SMOKE_TAG}_train_rank${rank}_${node}.log
    echo "Launching torchrun on ${node} (rank ${rank}), log ${log_file}"
    ssh -o StrictHostKeyChecking=no "${node}" "
        cd ${PROJDIR}
        module load frameworks/2025.3.1 2>/dev/null || true
        export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
        unset VIRTUAL_ENV
        $(env | grep -E '^(CCL_|FI_|ZE_|GLOO_|PYTHON|HF_|XPU_|PYTORCH_|TORCHTUNE_|no_proxy|NO_PROXY)' | sed 's/^/export /')
        export VLLM_HOST_0=${VLLM_HOST_0}
        export VLLM_HOST_1=${VLLM_HOST_1}
        torchrun \
            --nproc_per_node=${NPROC} \
            --nnodes=${NNODES} \
            --node_rank=${rank} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            --no-python \
            ${WRAPPER} \
            ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
            --config ${CONFIG} \
            base_model_path=${LOCAL_MODEL} \
            num_steps=${NSTEPS} \
            'vllm_url=${VLLM_URLS}' \
            save_every_n_epochs=100 \
            save_final_checkpoint=false \
            ${EXTRA} \
            > ${log_file} 2>&1
    "
}

echo "Launching torchrun on rank-1 node ${NODE1}..."
run_torchrun "${NODE1}" 1 &
RANK1_PID=$!

echo "Launching torchrun on rank-0 node ${NODE0}..."
run_torchrun "${NODE0}" 0
RANK0_RC=$?

wait ${RANK1_PID}
RANK1_RC=$?

echo "=== Training complete (rank0 rc=${RANK0_RC}, rank1 rc=${RANK1_RC}) ==="

# Best-effort vLLM cleanup so the next smoke phase starts on a quiet node.
for node in ${NODE0} ${NODE1}; do
    ssh -o StrictHostKeyChecking=no "${node}" "pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null; pkill VLLM 2>/dev/null; true" || true
done
sleep 5

# Propagate either rank's failure. Without this, a rank1-only failure was
# masked by rank0's clean exit and reported as success.
if [ "${RANK0_RC}" -ne 0 ]; then
    exit ${RANK0_RC}
fi
exit ${RANK1_RC}

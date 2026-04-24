#!/bin/bash
# MoE scaling benchmark: G=4 → G=8 → G=16
# Demonstrates that 26B-A4B MoE can run larger grpo_samples than dense 31B
# due to better memory efficiency (20.66 GiB vs 25.6 GiB at grpo_samples=4).
#
# Dense 31B baseline (established): 23.5s/step at grpo_samples=4
# Dense 31B at grpo_samples=16: cannot run (25.6 GiB already at grpo_samples=4)
# MoE 26B G=16 target: < 55s/step → 0.29 samples/s vs dense 0.17 samples/s (+71%)
#
# Usage (on compute node with 12 tiles, job-allocated):
#   bash recipes/dev/run_moe_scaling.sh [g4|g8|g16|all]
#   Default: runs all three configs sequentially

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.2.0 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
unset PYTORCH_ALLOC_CONF

GEMMA4_OVERLAY=${PROJDIR}/recipes/dev/vllm_gemma4_overlay
FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.2.0/lib/python3.10/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${GEMMA4_OVERLAY}:${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

VLLM_TILES=2
TRAIN_TILES=10
VLLM_PORT=8001
VLLM_LOG=/tmp/torchtune/vllm_moe_scaling.log
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B
VLLM_MAX_MODEL_LEN=1024

TARGET=${1:-all}

echo "=== MoE Scaling Benchmark (Gemma4 26B-A4B) ==="
echo "Node: $(hostname), Date: $(date)"
echo "Target: ${TARGET}"
python3 -c "import torchao; print('torchao:', torchao.__version__)"

mkdir -p /tmp/torchtune

# Stage model to /tmp for fast checkpoint loading
LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi

# Fix tokenizer
python3 ${PROJDIR}/recipes/dev/_fix_gemma4_tokenizer.py ${LOCAL_MODEL}/tokenizer_config.json

# vLLM on last 2 tiles
VLLM_TILE_START=$((12 - VLLM_TILES))
VLLM_MASK=$(seq -s, ${VLLM_TILE_START} 11)
VLLM_PID=""

start_vllm() {
    echo "=== Starting vLLM on tiles ${VLLM_MASK} (TP=${VLLM_TILES}) ==="
    # Warm model info cache
    ZE_AFFINITY_MASK=${VLLM_TILE_START} python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${LOCAL_MODEL}', tokenizer='${LOCAL_MODEL}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
" 2>&1 | tail -2

    ZE_AFFINITY_MASK=${VLLM_MASK} \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        TORCH_COMPILE_DISABLE=1 \
        PYTORCH_ALLOC_CONF= \
        CCL_PROCESS_LAUNCHER=none \
        CCL_ATL_TRANSPORT=ofi \
        FI_PROVIDER=cxi \
        CCL_KVS_IFACE=lo \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "${LOCAL_MODEL}" \
        --tensor-parallel-size ${VLLM_TILES} \
        --port ${VLLM_PORT} \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --distributed-executor-backend mp \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: ${VLLM_PID}"

    echo "Waiting for vLLM on port ${VLLM_PORT}..."
    ELAPSED=0
    while ! curl -s http://localhost:${VLLM_PORT}/health/ > /dev/null 2>&1; do
        sleep 5; ELAPSED=$((ELAPSED + 5))
        if [ ${ELAPSED} -ge 600 ]; then
            echo "ERROR: vLLM did not start within 600s"
            tail -50 "${VLLM_LOG}"
            exit 1
        fi
        [ $((ELAPSED % 30)) -eq 0 ] && echo "  Waiting... ${ELAPSED}s" && tail -3 "${VLLM_LOG}" 2>/dev/null
    done
    echo "vLLM ready (${ELAPSED}s)"
}

stop_vllm() {
    if [ -n "${VLLM_PID}" ]; then
        echo "Stopping vLLM (PID ${VLLM_PID})..."
        kill -- -${VLLM_PID} 2>/dev/null || true
        pkill -P ${VLLM_PID} 2>/dev/null || true
        kill ${VLLM_PID} 2>/dev/null || true
        wait ${VLLM_PID} 2>/dev/null || true
        VLLM_PID=""
        sleep 3
        # Kill any stray vLLM processes on the vLLM tiles
        for t in $(seq ${VLLM_TILE_START} 11); do
            fuser /dev/dri/renderD$((128 + t)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        done
        sleep 2
    fi
}

clean_training_tiles() {
    # Kill stray processes on training tiles (0..TRAIN_TILES-1) between runs.
    # vLLM on tiles VLLM_TILE_START..11 is left untouched.
    echo "Cleaning training tiles 0..$((TRAIN_TILES - 1))..."
    for t in $(seq 0 $((TRAIN_TILES - 1))); do
        fuser /dev/dri/renderD$((128 + t)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    sleep 2
}

run_grpo() {
    local CONFIG=$1
    local TAG=$2
    local NSTEPS=${3:-5}
    echo ""
    echo "=== Running GRPO: ${TAG} (${NSTEPS} steps) ==="
    echo "Config: ${CONFIG}"
    # Disable checkpoint saving for clean benchmark timing (no checkpoint I/O in step time)
    python3 -m torch.distributed.run --standalone --nproc_per_node=${TRAIN_TILES} \
        ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
        --config ${CONFIG} \
        base_model_path=${LOCAL_MODEL} \
        num_steps=${NSTEPS} \
        vllm_url=http://localhost:${VLLM_PORT} \
        save_every_n_epochs=100 \
        save_final_checkpoint=false
    local STATUS=$?
    clean_training_tiles
    echo "=== Done: ${TAG} (exit=${STATUS}) ==="
    return ${STATUS}
}

trap "stop_vllm" EXIT

# Start vLLM once — shared across all runs
start_vllm

# Run requested configs
if [ "${TARGET}" = "g4" ] || [ "${TARGET}" = "all" ]; then
    run_grpo \
        ${PROJDIR}/recipes/configs/dev/production/gemma4_26b_a4b_grpo_server_xpu.yaml \
        "G=4 (31B parity baseline)" 5
fi

if [ "${TARGET}" = "g8" ] || [ "${TARGET}" = "all" ]; then
    run_grpo \
        ${PROJDIR}/recipes/configs/dev/production/gemma4_26b_a4b_grpo_server_g8_xpu.yaml \
        "G=8 (2× dense baseline)" 5
fi

if [ "${TARGET}" = "g16" ] || [ "${TARGET}" = "all" ]; then
    run_grpo \
        ${PROJDIR}/recipes/configs/dev/production/gemma4_26b_a4b_grpo_server_g16_xpu.yaml \
        "G=16 (4× dense, Config B target)" 5
fi

echo ""
echo "=== MoE Scaling Benchmark Complete ==="
echo "Summary: check TIMING lines above for gen/grpo/total per config."
echo "Target: grpo_samples × (1/step_time) > dense 31B (4/23.5 = 0.17 samples/s)"

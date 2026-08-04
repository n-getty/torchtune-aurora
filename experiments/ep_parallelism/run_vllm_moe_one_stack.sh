#!/bin/bash
# Track B — run ONE vLLM stack's MoE-vs-dense latency A/B (config A + J).
# One stack per invocation => clean shell + FRESH per-stack TRITON_CACHE_DIR
# (the shared ~/.triton is torch-2.10-built and crashes the nightly triton 3.4
# with 'utf-8 codec can't decode' — a stale-cache mismatch, NOT a MoE bug).
#
# Usage:  run_vllm_moe_one_stack.sh <NEW|OLD> <LOG_DIR>
set -eo pipefail
STACK=$1
LOG_DIR=$2
[ -z "${STACK}" ] || [ -z "${LOG_DIR}" ] && { echo "usage: $0 <NEW|OLD> <LOG_DIR>"; exit 2; }
mkdir -p "${LOG_DIR}"

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
OUTDIR=${PROJDIR}/experiments/ep_parallelism
CLIENT=${OUTDIR}/vllm_moe_latency_client.py
MODEL_MOE=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
MODEL_DENSE=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B
NIGHTLY_VENV=/lus/flare/projects/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu
RESULTS=${LOG_DIR}/results_${STACK}.tsv
echo -e "stack\tlabel\ttp\tep\tbatch\tmax_tokens\tbest_s\tavg_s\ttok_per_s" > "${RESULTS}"

# ---- stack-specific env ----
if [ "${STACK}" = "NEW" ]; then
    module load frameworks 2>/dev/null || true
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
    unset VIRTUAL_ENV
    source "${NIGHTLY_VENV}/bin/activate"
else
    module load frameworks/2025.3.1 2>/dev/null || true
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
    unset VIRTUAL_ENV
fi

# FRESH isolated triton cache per stack (the fix).
export TRITON_CACHE_DIR=/tmp/triton_cache_${STACK}_$$
rm -rf "${TRITON_CACHE_DIR}"; mkdir -p "${TRITON_CACHE_DIR}"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_WORKER_COUNT=1 CCL_OP_SYNC=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp FI_PROVIDER=cxi
# CCL KVS on loopback + MPI local-rank hints. Without these the vllm-xpu-src TP>1
# workers die at XCCL init with "could not get local_idx/count from environment
# variables, trying to get them from ATL" (blocker #3, 2026-07-17). Mirrors the
# known-good _vllm_env_setup.sh (CCL_KVS_IFACE=lo) + MPI_LOCALRANKID hints.
export CCL_KVS_IFACE=lo
export LOCAL_WORLD_SIZE=4 MPI_LOCALNRANKS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=12
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/tmp
export VLLM_RPC_TIMEOUT=1800000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800  # deadline for sample_tokens RPC
# BATCH-HANG FIX (2026-07-17): the first BATCHED request hung 30min then
# sample_tokens-timed-out. Worker-side root cause = repeated "No available shared
# memory broadcast block found in 60s" (shm_broadcast starvation) — a TP worker
# went off doing time-consuming work while peers waited. The nightly config dump
# showed combo_kernels=True + benchmark_combo_kernel=True + flashinfer_autotune +
# INDUCTOR_MAX_AUTOTUNE=1 — inductor autotuning/combo-kernel benchmarking fires on
# the first multi-shape (batched) request and stalls one worker for minutes.
# Disable the autotune/combo paths so batched requests don't wedge.
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_INDUCTOR_COMBO_KERNEL=0   # harmless if unread by this build
export VLLM_DISABLE_COMPILE_CACHE=1
unset PYTORCH_ALLOC_CONF TORCH_XPU_ALLOC_CONF
ulimit -c 0

echo "########## STACK=${STACK} $(date) node=$(hostname -s) ##########"
echo "vllm=$(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null) torch=$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"
python3 -c "import importlib.util as u;print('vllm_xpu_kernels:', 'YES' if u.find_spec('vllm_xpu_kernels') else 'no')" 2>/dev/null
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

start_server() {
    local mask=$1 tp=$2 model=$3 log=$4
    # --compilation-config disables inductor combo-kernel benchmarking (the batch-hang
    # suspect). enforce_eager already avoids graph capture; this kills the remaining
    # combo/autotune passes that stalled a TP worker on the first batched request.
    ZE_AFFINITY_MASK="${mask}" nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "${model}" --tensor-parallel-size "${tp}" \
        --port 8001 --disable-custom-all-reduce --enforce-eager \
        --distributed-executor-backend mp --dtype bfloat16 \
        --gpu-memory-utilization 0.85 --max-model-len 1024 \
        --compilation-config '{"inductor_compile_config": {"combo_kernels": false, "benchmark_combo_kernel": false, "max_autotune": false}}' \
        --kernel-config '{"enable_flashinfer_autotune": false}' \
        > "${log}" 2>&1 &
    echo $!
}
wait_server() {
    local pid=$1 label=$2 max=${3:-1200} log=$4 e=0
    while [ ${e} -lt ${max} ]; do
        curl -sf "http://localhost:8001/health" >/dev/null 2>&1 && { echo "${label} ready ${e}s"; return 0; }
        ps -p ${pid} >/dev/null 2>&1 || { echo "ERROR ${label} died"; tail -30 "${log}"; return 1; }
        sleep 10; e=$((e+10))
    done
    echo "ERROR ${label} timeout"; return 1
}
stop_server() { kill "${1}" 2>/dev/null||true; wait "${1}" 2>/dev/null||true; sleep 8; }
run_latency() {
    local model=$1 tp=$2 stack=$3 config=$4
    for batch in 2 4 8; do for mt in 128 256; do
        python3 "${CLIENT}" --url http://localhost:8001 --model "${model}" \
            --batch "${batch}" --max-tokens "${mt}" --input-len 128 --runs 2 \
            --label "${stack}_${config}" --tp "tp${tp}" --ep no \
            | sed "s/^/${stack}\t/" >> "${RESULTS}" 2>>"${LOG_DIR}/${stack}_${config}_lat.log" || \
            echo "  latency run failed b=${batch} mt=${mt}"
    done; done
}

echo "--- ${STACK} A: MoE 30B-A3B TP=4 no EP ---"
PID=$(start_server "0,1,2,3" 4 "${MODEL_MOE}" "${LOG_DIR}/${STACK}_A_server.log")
wait_server ${PID} "${STACK}_A" 1200 "${LOG_DIR}/${STACK}_A_server.log" && run_latency "${MODEL_MOE}" 4 "${STACK}" A
stop_server ${PID}

echo "--- ${STACK} J: Dense 32B TP=4 no EP ---"
PID=$(start_server "0,1,2,3" 4 "${MODEL_DENSE}" "${LOG_DIR}/${STACK}_J_server.log")
wait_server ${PID} "${STACK}_J" 1200 "${LOG_DIR}/${STACK}_J_server.log" && run_latency "${MODEL_DENSE}" 4 "${STACK}" J
stop_server ${PID}

rm -rf "${TRITON_CACHE_DIR}" 2>/dev/null || true
echo "=== STACK=${STACK} done $(date) ==="; column -t "${RESULTS}"

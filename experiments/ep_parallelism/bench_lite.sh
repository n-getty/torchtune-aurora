#!/bin/bash
# Lite benchmark — runs A, B, C, J with 1 run each.
# Designed to run directly on a held node (not via PBS).
# Skips E/F/H/I (DP+EP / multi-instance) — those need a full 1.5h slot.
set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
OUTDIR=${PROJDIR}/experiments/ep_parallelism
SCRIPT_DIR=${OUTDIR}

LOG_DIR="${OUTDIR}/bench_lite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
RESULTS_TSV="${LOG_DIR}/results.tsv"

exec > >(tee -a "${LOG_DIR}/master.log") 2>&1
echo "=== bench_lite: $(date) ==="
echo "Node: $(hostname)"

# --- ENV ---
module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
unset PYTORCH_ALLOC_CONF
unset TORCH_XPU_ALLOC_CONF
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=12
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp

echo "vLLM: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

MODEL_MOE=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
MODEL_DENSE=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B
LATENCY_CLIENT=${SCRIPT_DIR}/vllm_moe_latency_client.py
RUNS=1

ulimit -c 0

start_server() {
    local port=$1 mask=$2 tp=$3 dp=$4 model=$5 ep_flag=$6 log=$7
    local dp_arg=""
    [ "${dp}" -gt 1 ] && dp_arg="--data-parallel-size ${dp}"
    ZE_AFFINITY_MASK="${mask}" nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --tensor-parallel-size "${tp}" \
        ${dp_arg} \
        ${ep_flag} \
        --port "${port}" \
        --disable-custom-all-reduce \
        --enforce-eager \
        --distributed-executor-backend mp \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 1024 \
        > "${log}" 2>&1 &
    echo $!
}

wait_server() {
    local port=$1 pid=$2 label=$3 max=${4:-600}
    local elapsed=0
    while [ ${elapsed} -lt ${max} ]; do
        curl -sf "http://localhost:${port}/health" > /dev/null 2>&1 && \
            { echo "${label} ready after ${elapsed}s"; return 0; }
        ps -p ${pid} > /dev/null 2>&1 || \
            { echo "ERROR: ${label} (PID ${pid}) died"; tail -20 "${log:-/dev/null}" 2>/dev/null; return 1; }
        sleep 10; elapsed=$((elapsed+10))
    done
    echo "ERROR: ${label} timed out (${max}s)"; return 1
}

stop_server() {
    kill "${1}" 2>/dev/null || true
    wait "${1}" 2>/dev/null || true
    sleep 5
}

run_latency() {
    local url=$1 model=$2 tp=$3 dp=$4 ep=$5 label=$6
    for batch in 2 4 8; do
        for max_tok in 128 256; do
            python3 "${LATENCY_CLIENT}" \
                --url "${url}" --model "${model}" \
                --batch "${batch}" --max-tokens "${max_tok}" \
                --input-len 128 --runs "${RUNS}" \
                --label "${label}" --tp "tp${tp}dp${dp}" --ep "${ep}" \
                >> "${RESULTS_TSV}" 2>>"${LOG_DIR}/${label}_latency.log"
        done
    done
}

echo -e "config\ttp_dp\tep\tbatch\tmax_tokens\tbest_s\tavg_s\ttok_per_s" > "${RESULTS_TSV}"

TILES_4="0,1,2,3"

# A: MoE TP=4, no EP
echo; echo "=== A: Qwen3-30B-A3B TP=4 no EP === $(date)"
PID=$(start_server 8001 "${TILES_4}" 4 1 "${MODEL_MOE}" "" "${LOG_DIR}/A_server.log")
wait_server 8001 ${PID} "A" 1200 && run_latency http://localhost:8001 "${MODEL_MOE}" 4 1 no MoE_tp4_noep
stop_server ${PID}

# B: MoE TP=4, EP dp=1
echo; echo "=== B: Qwen3-30B-A3B TP=4 EP dp=1 === $(date)"
PID=$(start_server 8001 "${TILES_4}" 4 1 "${MODEL_MOE}" "--enable-expert-parallel" "${LOG_DIR}/B_server.log")
wait_server 8001 ${PID} "B" 1200 && run_latency http://localhost:8001 "${MODEL_MOE}" 4 1 yes_dp1 MoE_tp4_ep_dp1
stop_server ${PID}

# C: MoE TP=2, no EP
echo; echo "=== C: Qwen3-30B-A3B TP=2 no EP === $(date)"
PID=$(start_server 8001 "0,1" 2 1 "${MODEL_MOE}" "" "${LOG_DIR}/C_server.log")
wait_server 8001 ${PID} "C" 1200 && run_latency http://localhost:8001 "${MODEL_MOE}" 2 1 no MoE_tp2_noep
stop_server ${PID}

# D: MoE TP=2, EP dp=1
echo; echo "=== D: Qwen3-30B-A3B TP=2 EP dp=1 === $(date)"
PID=$(start_server 8001 "0,1" 2 1 "${MODEL_MOE}" "--enable-expert-parallel" "${LOG_DIR}/D_server.log")
wait_server 8001 ${PID} "D" 1200 && run_latency http://localhost:8001 "${MODEL_MOE}" 2 1 yes_dp1 MoE_tp2_ep_dp1
stop_server ${PID}

# J: Dense 32B TP=4, no EP
echo; echo "=== J: Qwen3-32B TP=4 no EP === $(date)"
PID=$(start_server 8001 "${TILES_4}" 4 1 "${MODEL_DENSE}" "" "${LOG_DIR}/J_server.log")
wait_server 8001 ${PID} "J" 1200 && run_latency http://localhost:8001 "${MODEL_DENSE}" 4 1 no Dense32B_tp4
stop_server ${PID}

echo
echo "=== RESULTS ==="
column -t "${RESULTS_TSV}"
echo
echo "TSV: ${RESULTS_TSV}"
echo "=== bench_lite complete: $(date) ==="

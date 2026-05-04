#!/bin/bash
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A ModCon
#PBS -N vllm_bench_final
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/ep_parallelism/vllm_bench_final.out

# Final benchmark configs — A/B/C/D already done in bench_lite session.
# Remaining:
#   E. MoE TP=4 EP dp=3  — 12 tiles, native DP+EP, AllToAll fires (one server)
#   H. 3x MoE TP=4 no EP — independent instances, round-robin proxy
#   I. 3x MoE TP=4 EP dp=1 — independent + EP per instance (AgRs path)
#   J. Dense 32B TP=4    — hardware-matched dense baseline
#
# Uses frameworks/2025.3.1 (vLLM 0.15.0), 1 node, 12 tiles available.
# Timeout 1500s per server (cold DAOS load takes ~16 min).
# RUNS=2 per data point.

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
OUTDIR=${PROJDIR}/experiments/ep_parallelism
SCRIPT_DIR=${OUTDIR}

LOG_DIR="${OUTDIR}/bench_final_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
RESULTS_TSV="${LOG_DIR}/results.tsv"

exec > >(tee -a "${LOG_DIR}/master.log") 2>&1
echo "=== bench_final: $(date) ==="
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
echo "Tiles: $(python3 -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null)"

MODEL_MOE=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
MODEL_DENSE=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B
LATENCY_CLIENT=${SCRIPT_DIR}/vllm_moe_latency_client.py
PROXY=${SCRIPT_DIR}/proxy_round_robin.py
RUNS=2

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
    local port=$1 pid=$2 label=$3 max=${4:-1500}
    local elapsed=0
    while [ ${elapsed} -lt ${max} ]; do
        curl -sf "http://localhost:${port}/health" > /dev/null 2>&1 && \
            { echo "${label} ready after ${elapsed}s"; return 0; }
        ps -p ${pid} > /dev/null 2>&1 || \
            { echo "ERROR: ${label} (PID ${pid}) died"; return 1; }
        sleep 10; elapsed=$((elapsed+10))
    done
    echo "ERROR: ${label} timed out (${max}s)"; return 1
}

stop_server() {
    kill "${1}" 2>/dev/null || true
    pkill -P "${1}" 2>/dev/null || true
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
TILES_12="0,1,2,3,4,5,6,7,8,9,10,11"

# ============================================================================
# E: MoE TP=4 EP dp=3 — native DP+EP, AllToAll fires; 12 tiles, 1 server
# ep_size = tp * dp = 4 * 3 = 12 (all 128 experts sharded across 12 tiles)
# ============================================================================
echo; echo "=== E: Qwen3-30B-A3B TP=4 EP dp=3 (12 tiles, AllToAll) === $(date)"
PID=$(start_server 8001 "${TILES_12}" 4 3 "${MODEL_MOE}" "--enable-expert-parallel" "${LOG_DIR}/E_server.log")
if wait_server 8001 ${PID} "E" 1500; then
    run_latency http://localhost:8001 "${MODEL_MOE}" 4 3 yes_dp3 MoE_tp4_ep_dp3
fi
stop_server ${PID}

# ============================================================================
# H: 3x MoE TP=4, no EP, independent — zero-coupling baseline on 12 tiles
# Three separate servers on tiles 0-3/4-7/8-11; round-robin proxy on 8000.
# Also measures single-instance latency (port 8001) for direct comparison.
# ============================================================================
echo; echo "=== H: 3x Qwen3-30B-A3B TP=4 no EP independent (round-robin) === $(date)"
PIDS_H=()
for i in 1 2 3; do
    MASK="$(( (i-1)*4 )),$(( (i-1)*4+1 )),$(( (i-1)*4+2 )),$(( (i-1)*4+3 ))"
    pid=$(start_server $((8000+i)) "${MASK}" 4 1 "${MODEL_MOE}" "" \
          "${LOG_DIR}/H_server_${i}.log")
    PIDS_H+=($pid)
    sleep 5
done
READY_H=0
for i in 1 2 3; do
    wait_server $((8000+i)) ${PIDS_H[$((i-1))]} "H${i}" 1500 && READY_H=$((READY_H+1)) || true
done
echo "H: ${READY_H}/3 servers ready"
if [ ${READY_H} -eq 3 ]; then
    python3 "${PROXY}" 8000 \
        http://localhost:8001 http://localhost:8002 http://localhost:8003 \
        > "${LOG_DIR}/H_proxy.log" 2>&1 &
    PROXY_H=$!; sleep 3
    run_latency http://localhost:8000 "${MODEL_MOE}" 4 "3indep" no MoE_3xtp4_noep_proxy
    run_latency http://localhost:8001 "${MODEL_MOE}" 4 "1of3"   no MoE_1xtp4_noep_single
    kill ${PROXY_H} 2>/dev/null || true
fi
for pid in "${PIDS_H[@]}"; do stop_server ${pid}; done

# ============================================================================
# I: 3x MoE TP=4, EP dp=1, independent — EP AgRs per instance; zero coupling
# ============================================================================
echo; echo "=== I: 3x Qwen3-30B-A3B TP=4 EP dp=1 independent (round-robin) === $(date)"
PIDS_I=()
for i in 1 2 3; do
    MASK="$(( (i-1)*4 )),$(( (i-1)*4+1 )),$(( (i-1)*4+2 )),$(( (i-1)*4+3 ))"
    pid=$(start_server $((8000+i)) "${MASK}" 4 1 "${MODEL_MOE}" "--enable-expert-parallel" \
          "${LOG_DIR}/I_server_${i}.log")
    PIDS_I+=($pid)
    sleep 5
done
READY_I=0
for i in 1 2 3; do
    wait_server $((8000+i)) ${PIDS_I[$((i-1))]} "I${i}" 1500 && READY_I=$((READY_I+1)) || true
done
echo "I: ${READY_I}/3 servers ready"
if [ ${READY_I} -eq 3 ]; then
    python3 "${PROXY}" 8000 \
        http://localhost:8001 http://localhost:8002 http://localhost:8003 \
        > "${LOG_DIR}/I_proxy.log" 2>&1 &
    PROXY_I=$!; sleep 3
    run_latency http://localhost:8000 "${MODEL_MOE}" 4 "3indep" yes_dp1 MoE_3xtp4_ep_proxy
    run_latency http://localhost:8001 "${MODEL_MOE}" 4 "1of3"   yes_dp1 MoE_1xtp4_ep_single
    kill ${PROXY_I} 2>/dev/null || true
fi
for pid in "${PIDS_I[@]}"; do stop_server ${pid}; done

# ============================================================================
# J: Dense Qwen3-32B TP=4, no EP — hardware-matched dense baseline
# ============================================================================
echo; echo "=== J: Qwen3-32B TP=4 no EP === $(date)"
PID=$(start_server 8001 "${TILES_4}" 4 1 "${MODEL_DENSE}" "" "${LOG_DIR}/J_server.log")
if wait_server 8001 ${PID} "J" 1500; then
    run_latency http://localhost:8001 "${MODEL_DENSE}" 4 1 no Dense32B_tp4
fi
stop_server ${PID}

# ============================================================================
echo
echo "=== RESULTS ==="
column -t "${RESULTS_TSV}"
echo
echo "TSV: ${RESULTS_TSV}"
echo "=== bench_final complete: $(date) ==="

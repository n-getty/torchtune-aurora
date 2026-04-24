#!/bin/bash
#
# 32B GRPO Performance Test Suite
#
# Runs a series of configs to find optimal 32B GRPO settings on Aurora.
# Designed for interactive use on held PBS nodes (hold_node.sh / hold_2nodes.sh).
#
# Usage (single-node, from held node):
#   bash recipes/dev/test_32b_perf.sh 1node
#
# Usage (2-node, from first node of held 2-node job):
#   bash recipes/dev/test_32b_perf.sh 2node
#
# Usage (single test):
#   bash recipes/dev/test_32b_perf.sh single <config_path> [extra_overrides...]
#
# Requirements:
#   - PBS_JOBID and PBS_NODEFILE must be set (from held PBS job)
#   - Model at /lus/flare/projects/ModCon/ngetty/models/Qwen3-32B
#   - module load frameworks
#
set -eo pipefail

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

MODEL_SRC="/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B"
MODEL_PATH="/tmp/torchtune/Qwen3-32B"
NSTEPS=5
VLLM_TILES=2
VLLM_PORT=8001
VLLM_MAX_MODEL_LEN=2048
RESULTS_FILE="/tmp/torchtune/perf_results_$(date +%Y%m%d_%H%M%S).txt"

PRODUCTION_CONFIG="recipes/configs/dev/production/qwen32B_grpo_server_xpu.yaml"
HSDP_SCRIPT="recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh"
WRAPPER="${TORCHTUNE_DIR}/recipes/dev/aurora_grpo_vllm_wrapper.sh"
VLLM_CUSTOMIZATION="${TORCHTUNE_DIR}/recipes/dev/_usercustomize_vllm"

# ============================================================
# Helpers
# ============================================================
log() { echo "=== [$(date +%H:%M:%S)] $*"; }

check_pbs() {
    if [[ -z "${PBS_JOBID:-}" ]] || [[ -z "${PBS_NODEFILE:-}" ]]; then
        echo "ERROR: PBS_JOBID and PBS_NODEFILE must be set."
        echo "  export PBS_JOBID=<jobid>"
        echo "  export PBS_NODEFILE=/var/spool/pbs/aux/\$PBS_JOBID"
        exit 1
    fi
}

get_nodes() {
    UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
    NUM_NODES=${#UNIQUE_NODES[@]}
    THIS_NODE=$(hostname | cut -d'.' -f1)
    log "Nodes available: ${NUM_NODES} (${UNIQUE_NODES[*]})"
    log "This node: ${THIS_NODE}"
}

stage_model() {
    local node=$1
    if ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        log "Model already staged on ${node}"
    else
        log "Staging model to ${node}:${MODEL_PATH}..."
        ssh "${node}" "mkdir -p /tmp/torchtune && cp -r '${MODEL_SRC}' '${MODEL_PATH}'" 2>/dev/null
        log "Model staged on ${node}"
    fi
}

setup_env() {
    module load frameworks 2>/dev/null || true
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
    unset VIRTUAL_ENV
    export PYTHONPATH="${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:${PYTHONPATH:-}"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    export TORCH_COMPILE_DISABLE=1
}

kill_vllm() {
    local node=${1:-$(hostname | cut -d'.' -f1)}
    ssh "${node}" "pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null; \
                    pkill -f 'vllm.v1.engine' 2>/dev/null; \
                    pkill -f 'from multiprocessing' 2>/dev/null" 2>/dev/null || true
    sleep 2
}

kill_vllm_all() {
    for node in "${UNIQUE_NODES[@]}"; do
        kill_vllm "${node}"
    done
}

start_vllm() {
    local node=$1
    local port=${2:-$VLLM_PORT}
    local tiles=${3:-$VLLM_TILES}
    local tile_start=$((12 - tiles))
    local mask=$(seq -s, ${tile_start} 11)

    log "Starting vLLM on ${node} (TP=${tiles}, tiles ${mask}, port ${port})..."

    local VLLM_ENV="
cd ${TORCHTUNE_DIR}
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${mask}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export PYTHONPATH='${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
mkdir -p /tmp/torchtune
"

    if [ ${tiles} -gt 1 ]; then
        ssh "${node}" "${VLLM_ENV}
python3 -m vllm.entrypoints.openai.api_server \
    --model '${MODEL_PATH}' \
    --tensor-parallel-size ${tiles} \
    --port ${port} \
    --enforce-eager \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --distributed-executor-backend mp \
    > '/tmp/torchtune/vllm_server.log' 2>&1
" &
    else
        ssh "${node}" "${VLLM_ENV}
python3 recipes/dev/vllm_serve_xpu.py \
    --model '${MODEL_PATH}' \
    --tensor_parallel_size 1 \
    --port ${port} \
    --enforce_eager \
    --dtype bfloat16 \
    --gpu_memory_utilization 0.80 \
    --max_model_len ${VLLM_MAX_MODEL_LEN} \
    > '/tmp/torchtune/vllm_server.log' 2>&1
" &
    fi
}

wait_vllm() {
    local node=$1
    local port=${2:-$VLLM_PORT}
    local timeout=600
    local elapsed=0

    log "Waiting for vLLM on ${node}:${port}..."
    while ! ssh "${node}" "curl -s http://localhost:${port}/health/ > /dev/null 2>&1" 2>/dev/null; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ ${elapsed} -ge ${timeout} ]; then
            log "ERROR: vLLM on ${node} did not start within ${timeout}s"
            ssh "${node}" "tail -30 /tmp/torchtune/vllm_server.log" 2>/dev/null || true
            return 1
        fi
    done
    log "vLLM healthy on ${node} (${elapsed}s)"
}

warm_vllm_cache() {
    local node=$1
    local tile_start=$((12 - VLLM_TILES))
    log "Warming vLLM cache on ${node}..."
    ssh "${node}" "
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${tile_start}
export TORCH_COMPILE_DISABLE=1
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
export PYTHONPATH='${TORCHTUNE_DIR}:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
python3 -c \"
from vllm.config import ModelConfig
ModelConfig(model='${MODEL_PATH}', tokenizer='${MODEL_PATH}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
\" 2>&1 | tail -1
" 2>/dev/null || true
}

# ============================================================
# Run a single test
# ============================================================
run_test() {
    local test_name=$1
    local config=$2
    local num_train_nodes=${3:-1}
    shift 3
    local extra_overrides=()
    if [ $# -gt 0 ]; then
        extra_overrides=("$@")
    fi

    local ngpus=10  # training tiles per node
    local total_ranks=$((num_train_nodes * ngpus))

    log "========================================"
    log "TEST: ${test_name}"
    log "Config: ${config}"
    log "Nodes: ${num_train_nodes}, Ranks: ${total_ranks}"
    log "Overrides: ${extra_overrides[*]:-none}"
    log "========================================"

    # Kill any existing vLLM
    kill_vllm_all

    # Determine which nodes run vLLM
    local vllm_nodes=()
    local train_nodes=()
    for node in "${UNIQUE_NODES[@]}"; do
        if [ ${#train_nodes[@]} -lt ${num_train_nodes} ]; then
            vllm_nodes+=("$node")  # colocated vLLM on training nodes
            train_nodes+=("$node")
        fi
    done

    # Stage model + warm cache + start vLLM on all training nodes
    for node in "${vllm_nodes[@]}"; do
        stage_model "${node}"
        warm_vllm_cache "${node}"
    done
    for node in "${vllm_nodes[@]}"; do
        start_vllm "${node}" ${VLLM_PORT} ${VLLM_TILES}
    done
    for node in "${vllm_nodes[@]}"; do
        wait_vllm "${node}" ${VLLM_PORT} || return 1
    done

    # CCL environment
    export CCL_PROCESS_LAUNCHER=pmix
    export CCL_ATL_TRANSPORT=mpi
    export CCL_KVS_MODE=mpi
    export CCL_KVS_USE_MPI_RANKS=1
    export CCL_CONFIGURATION=cpu_gpu_dpcpp
    export CCL_KVS_CONNECTION_TIMEOUT=600
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
    export CCL_OP_SYNC=1
    export FI_PROVIDER=cxi
    export CCL_WORKER_COUNT=1
    export CCL_ALLREDUCE=ring
    export CCL_CHUNK_SIZE=16777216
    export FI_CXI_RX_MATCH_MODE=hybrid
    export FI_CXI_OFLOW_BUF_SIZE=8388608
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_MR_CACHE_MONITOR=disabled
    unset PYTORCH_ALLOC_CONF

    export MASTER_ADDR="${train_nodes[0]}.hsn.cm.aurora.alcf.anl.gov"
    export MASTER_PORT=$((20000 + RANDOM % 20000))
    export NUM_NODES=${num_train_nodes}
    export NGPUS_PER_NODE=${ngpus}
    export USE_AFFINITY_MASK=training

    # Build hostfile for training nodes only
    local hostfile="/tmp/torchtune/train_hostfile_$$"
    > "${hostfile}"
    for node in "${train_nodes[@]}"; do
        echo "${node}" >> "${hostfile}"
    done

    local logfile="/tmp/torchtune/test_${test_name}.log"

    # Build override args
    local dp_replicate=1
    if [ ${num_train_nodes} -gt 1 ]; then
        dp_replicate=${num_train_nodes}
    fi

    local override_args=(
        "base_model_path=${MODEL_PATH}"
        "num_steps=${NSTEPS}"
        "vllm_url=http://localhost:${VLLM_PORT}"
        "vllm_weight_sync=false"
    )
    if [ ${dp_replicate} -gt 1 ]; then
        override_args+=("data_parallel_replicate_dim=${dp_replicate}")
    fi
    if [ ${#extra_overrides[@]} -gt 0 ]; then
        override_args+=("${extra_overrides[@]}")
    fi

    log "Running training: ${total_ranks} ranks across ${num_train_nodes} nodes..."
    mpiexec \
        --pmi=pmix \
        --hostfile "${hostfile}" \
        -n "${total_ranks}" \
        -ppn "${ngpus}" \
        --cpu-bind depth \
        --depth 8 \
        bash "${WRAPPER}" \
        "dev/grpo_full_finetune_distributed_xpu" \
        "${config}" \
        "${override_args[@]}" \
        2>&1 | tee "${logfile}"

    local exit_code=${PIPESTATUS[0]}

    # Extract timing from logs
    log "--- Results for ${test_name} ---"
    if [ ${exit_code} -eq 0 ]; then
        grep -E "Step [0-9]+.*total=" "${logfile}" | tail -6 || true
        # Save summary
        echo "" >> "${RESULTS_FILE}"
        echo "=== ${test_name} ===" >> "${RESULTS_FILE}"
        echo "Config: ${config}" >> "${RESULTS_FILE}"
        echo "Nodes: ${num_train_nodes}, Ranks: ${total_ranks}" >> "${RESULTS_FILE}"
        echo "Overrides: ${extra_overrides[*]:-none}" >> "${RESULTS_FILE}"
        grep -E "Step [0-9]+.*total=" "${logfile}" | tail -6 >> "${RESULTS_FILE}" 2>/dev/null || true
        grep -i "peak.*memory\|OOM\|error" "${logfile}" | tail -5 >> "${RESULTS_FILE}" 2>/dev/null || true
    else
        log "FAILED (exit code ${exit_code})"
        echo "" >> "${RESULTS_FILE}"
        echo "=== ${test_name} === FAILED (exit ${exit_code})" >> "${RESULTS_FILE}"
        grep -i "OOM\|error\|segfault\|crash\|SIGABRT" "${logfile}" | tail -10 >> "${RESULTS_FILE}" 2>/dev/null || true
    fi

    # Cleanup vLLM
    kill_vllm_all
    sleep 5

    return ${exit_code}
}

# ============================================================
# Test suites
# ============================================================
run_1node_tests() {
    log "=========================================="
    log "SINGLE-NODE 32B PERFORMANCE TEST SUITE"
    log "=========================================="

    echo "# 32B GRPO Performance Results — $(date)" > "${RESULTS_FILE}"
    echo "# Node: ${UNIQUE_NODES[0]}" >> "${RESULTS_FILE}"

    # Test 1: Baseline G=4, fbs=4 (re-validate known 18.1s)
    run_test "baseline_G4_fbs4" \
        "${PRODUCTION_CONFIG}" 1 || true

    # Test 2: G=8, fbs=8 (compare against known G=8/fbs=4 = 24.2s)
    run_test "G8_fbs8" \
        "recipes/configs/dev/experimental/qwen32B_grpo_G8_fbs8.yaml" 1 || true

    # Test 3: G=16, fbs=8 (compare against known G=16/fbs=4 = 36.9s)
    run_test "G16_fbs8" \
        "recipes/configs/dev/experimental/qwen32B_grpo_G16_fbs8.yaml" 1 || true

    # Test 4: G=16, fbs=16 (aggressive — may OOM)
    run_test "G16_fbs16" \
        "recipes/configs/dev/experimental/qwen32B_grpo_G16_fbs16.yaml" 1 || true

    # Test 5: G=8, fbs=4 (baseline for comparison, known=24.2s)
    run_test "G8_fbs4" \
        "${PRODUCTION_CONFIG}" 1 \
        "grpo_samples=8" "forward_batch_size=4" || true

    log "=========================================="
    log "SINGLE-NODE TESTS COMPLETE"
    log "Results saved to: ${RESULTS_FILE}"
    log "=========================================="
    cat "${RESULTS_FILE}"
}

run_2node_tests() {
    if [ ${NUM_NODES} -lt 2 ]; then
        log "ERROR: Need 2+ nodes for 2-node tests, got ${NUM_NODES}"
        exit 1
    fi

    log "=========================================="
    log "2-NODE 32B HSDP PERFORMANCE TEST SUITE"
    log "=========================================="

    echo "# 32B GRPO 2-Node Performance Results — $(date)" > "${RESULTS_FILE}"
    echo "# Nodes: ${UNIQUE_NODES[*]}" >> "${RESULTS_FILE}"

    # Test 6: HSDP baseline G=4, fbs=4 (re-validate known 19.4s)
    run_test "hsdp_baseline_G4_fbs4" \
        "${PRODUCTION_CONFIG}" 2 || true

    # Test 7: HSDP G=8, fbs=8 (never tested — key experiment)
    run_test "hsdp_G8_fbs8" \
        "recipes/configs/dev/experimental/qwen32B_grpo_G8_fbs8.yaml" 2 || true

    # Test 8: HSDP G=16, fbs=8 (head-to-head vs dedicated 35.6s)
    run_test "hsdp_G16_fbs8" \
        "recipes/configs/dev/experimental/qwen32B_grpo_G16_fbs8.yaml" 2 || true

    # Test 9: HSDP G=4 with AllGather overlap (reshard=null)
    run_test "hsdp_G4_overlap" \
        "${PRODUCTION_CONFIG}" 2 \
        "disable_prefetch=false" || true

    log "=========================================="
    log "2-NODE TESTS COMPLETE"
    log "Results saved to: ${RESULTS_FILE}"
    log "=========================================="
    cat "${RESULTS_FILE}"
}

# ============================================================
# Main
# ============================================================
check_pbs
setup_env
get_nodes

case "${1:-}" in
    1node)
        run_1node_tests
        ;;
    2node)
        run_2node_tests
        ;;
    single)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 single <config_path> [overrides...]"
            exit 1
        fi
        config=$2
        shift 2
        run_test "single_test" "${config}" 1 "$@"
        ;;
    all)
        run_1node_tests
        if [ ${NUM_NODES} -ge 2 ]; then
            run_2node_tests
        fi
        ;;
    *)
        echo "Usage: $0 {1node|2node|single|all}"
        echo ""
        echo "  1node  — Run single-node performance tests (baseline, G=8, G=16)"
        echo "  2node  — Run 2-node HSDP performance tests (baseline, G=8, G=16)"
        echo "  single — Run a single test: $0 single <config> [overrides...]"
        echo "  all    — Run all tests (1-node then 2-node if available)"
        echo ""
        echo "Prerequisites:"
        echo "  1. Hold nodes: qsub recipes/dev/hold_node.sh (or hold_2nodes.sh)"
        echo "  2. Set PBS vars: export PBS_JOBID=<id> PBS_NODEFILE=/var/spool/pbs/aux/<id>"
        echo "  3. SSH to first node"
        exit 1
        ;;
esac

#!/bin/bash
# Qwen3-4B DENSE GRPO — 2-node server mode launcher (perf comparison vs LoRA)
#
# Apples-to-apples step-time baseline against run_qwen3_4b_lora_2node.sh.
# Same vLLM topology + envelope; full-weight FSDP training with XCCL weight
# sync to vLLM (not LoRA hot-swap). NO LoRA on either side.
#
# Config:  recipes/configs/dev/production/qwen3_4b_grpo_2node_server_xpu.yaml
# Recipe:  recipes/dev/grpo_full_finetune_distributed_xpu.py  (base, NOT lora)
set -o pipefail

TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/run_dense_grpo_2node_$(date +%Y%m%d_%H%M%S).log"

echo "=== Qwen3-4B DENSE GRPO 2-Node Server Mode (perf baseline) ===" | tee "${LOG}"
echo "Date: $(date)  Host: $(hostname)" | tee -a "${LOG}"

# ============================================================
# Configuration
# ============================================================
VLLM_DP=${VLLM_DP:-12}                  # one HTTP server per tile
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-1536}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.70}
TRAIN_TILES=${TRAIN_TILES:-11}
MODEL_PATH=${MODEL_PATH:-/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B}
NSTEPS=${NSTEPS:-5}
GRPO_SAMPLES=${GRPO_SAMPLES:-24}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-8}
REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}  # no-grad ref+rollout chunk; CRITICAL — defaults to FBS otherwise (32× regression when fbs=1)
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-512}
LORA_RANK=${LORA_RANK:-16}
LORA_MAX_LORAS=${LORA_MAX_LORAS:-2}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-2500}
CONFIG=${CONFIG:-recipes/configs/dev/production/qwen3_4b_grpo_2node_server_xpu.yaml}
# No adapter root — dense run does not publish LoRA. (Variable kept unset.)

RDZV_PORT=$((29400 + RANDOM % 1000))
MASTER_PORT=$((20000 + RANDOM % 20000))

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Run from a held PBS job." | tee -a "${LOG}"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "${PBS_NODEFILE}" | awk '!seen[$0]++'))
if [ "${#UNIQUE_NODES[@]}" -lt 2 ]; then
    echo "ERROR: Need 2 nodes. Got ${#UNIQUE_NODES[@]}: ${UNIQUE_NODES[*]}" | tee -a "${LOG}"
    exit 1
fi

# Convention: first node = TRAIN, second node = VLLM.
TRAIN_NODE="${UNIQUE_NODES[0]}"
VLLM_NODE="${UNIQUE_NODES[1]}"

TRAIN_NODE_IP=$(ssh "${TRAIN_NODE}" "hostname -i" 2>/dev/null | head -1)
VLLM_NODE_IP=$(ssh "${VLLM_NODE}" "hostname -i" 2>/dev/null | head -1)
if [[ -z "${TRAIN_NODE_IP}" || -z "${VLLM_NODE_IP}" ]]; then
    echo "ERROR: hostname -i failed. TRAIN=${TRAIN_NODE_IP} VLLM=${VLLM_NODE_IP}" | tee -a "${LOG}"
    exit 1
fi

echo "Train node:  ${TRAIN_NODE} (IP=${TRAIN_NODE_IP}, ${TRAIN_TILES} ranks)" | tee -a "${LOG}"
echo "vLLM node:   ${VLLM_NODE}  (IP=${VLLM_NODE_IP}, DP=${VLLM_DP} HTTP servers)" | tee -a "${LOG}"
echo "Config:      ${CONFIG}" | tee -a "${LOG}"
echo "Model:       ${MODEL_PATH}" | tee -a "${LOG}"
echo "(dense run, no adapter root)" | tee -a "${LOG}"
echo "Steps: ${NSTEPS}, G=${GRPO_SAMPLES}, FBS=${FORWARD_BATCH_SIZE}, max_gen=${MAX_GEN_TOKENS}" | tee -a "${LOG}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*"
export NO_PROXY="*"

# ============================================================
# Verify model weights exist on both nodes before wasting time
# ============================================================
echo "Verifying Qwen3-4B weights exist..." | tee -a "${LOG}"
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "ERROR: Model not found at ${MODEL_PATH} on ${node}." | tee -a "${LOG}"
        echo "  Download Qwen3-4B and place at ${MODEL_PATH}, or set MODEL_PATH=<path>" | tee -a "${LOG}"
        exit 1
    fi
done
echo "Model weights verified on all nodes." | tee -a "${LOG}"

# ============================================================
# Stage model to local /tmp on VLLM_NODE (avoid 12-way concurrent Lustre reads)
# ============================================================
VLLM_MODEL_LOCAL="/tmp/models/$(basename "${MODEL_PATH}")"
echo "Staging model to ${VLLM_NODE}:${VLLM_MODEL_LOCAL} ..." | tee -a "${LOG}"
STAGE_T0=$(date +%s)
ssh "${VLLM_NODE}" "
mkdir -p /tmp/models
if [ -f '${VLLM_MODEL_LOCAL}/config.json' ]; then
    echo '  Already staged on this node, skipping copy.'
else
    echo '  Copying from Lustre to /tmp ...'
    cp -r '${MODEL_PATH}' /tmp/models/
    echo '  Done.'
fi
ls -lh '${VLLM_MODEL_LOCAL}/'*.safetensors 2>/dev/null | awk '{print \"  \",\$5,\$9}' || true
" 2>&1 | tee -a "${LOG}"
STAGE_RC=$?
STAGE_ELAPSED=$(( $(date +%s) - STAGE_T0 ))
if [ "${STAGE_RC}" -ne 0 ]; then
    echo "ERROR: model staging failed (rc=${STAGE_RC}). Aborting." | tee -a "${LOG}"
    exit "${STAGE_RC}"
fi
echo "Model staged in ${STAGE_ELAPSED}s. vLLM will load from ${VLLM_MODEL_LOCAL}." | tee -a "${LOG}"

# ============================================================
# Prepare PYTHONPATH
# ============================================================
cd "${TT_DIR}"
source recipes/dev/_aurora_paths.sh
VLLM_CUSTOMIZATION="${TT_DIR}/recipes/dev/_usercustomize_vllm"
VLLM_PYTHONPATH="$(aurora_pythonpath "${TT_DIR}" "${VLLM_CUSTOMIZATION}")"
# Append user site-packages so math_verify (user-installed) is accessible
# even when PYTHONNOUSERSITE=1. Use the Aurora-frameworks-specific user site (python3.12)
# rather than `python3 -m site --user-site` which resolves to python3.10 on the login
# node and would shadow the frameworks torch with a stale ~/.local/lib/python3.10 install.
_AURORA_USER_SITE="/home/${USER}/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages"
TRAIN_PYTHONPATH="$(aurora_pythonpath "${TT_DIR}"):${_AURORA_USER_SITE}"

# ============================================================
# Pre-launch cleanup on VLLM_NODE
# ============================================================
TT_DIR_REMOTE="${TT_DIR}"
echo "Cleaning stale vLLM on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
sleep 2
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -20 || true
sleep 3
mkdir -p /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true

echo "Cleaning stale training on ${TRAIN_NODE}..." | tee -a "${LOG}"
ssh "${TRAIN_NODE}" "
pkill -9 -f 'grpo_full_finet[u]ne_distributed_xpu' 2>/dev/null || true
pkill -9 -f 'torch[.]distributed[.]run' 2>/dev/null || true
sleep 2
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -20 || true
sleep 3
mkdir -p /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true

# (No adapter root for dense run.)

# ============================================================
# Pre-launch tile-memory verification on VLLM_NODE
# ============================================================
echo "Verifying tile memory before launch on ${VLLM_NODE}..." | tee -a "${LOG}"
TILE_CHECK=$(ssh "${VLLM_NODE}" "bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --check" 2>&1)
echo "${TILE_CHECK}" | tee -a "${LOG}"
if echo "${TILE_CHECK}" | grep -q 'FULL'; then
    echo "WARNING: tiles below 20 GiB free. Re-running clean_tiles --kill..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill" 2>&1 | tail -30 | tee -a "${LOG}"
    sleep 5
    TILE_CHECK2=$(ssh "${VLLM_NODE}" "bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --check" 2>&1)
    echo "${TILE_CHECK2}" | tee -a "${LOG}"
    if echo "${TILE_CHECK2}" | grep -q 'FULL'; then
        echo "ERROR: tiles still FULL after second cleanup pass; aborting." | tee -a "${LOG}"
        exit 1
    fi
    echo "Recovered: tiles now CLEAN." | tee -a "${LOG}"
fi

# Belt-and-suspenders: kill any surviving EngineCore subprocesses on VLLM_NODE
# before launching new servers. pkill -f misses EngineCore (bare cmdline, not
# python-invoked); comm-prefix 'VLLM' matches 'VLLM::EngineCore' (truncated to 15).
ssh "${VLLM_NODE}" "pkill -9 VLLM 2>/dev/null || true; sleep 3" 2>/dev/null || true

# ============================================================
# Launch ${VLLM_DP} vLLM HTTP servers on VLLM_NODE
# Each server loads Qwen3-4B base weights; full-weight sync via XCCL broadcast
# (NO LoRA hot-swap on this dense baseline).
# ============================================================
VLLM_URLS=""
echo "Starting ${VLLM_DP} vLLM HTTP servers on ${VLLM_NODE}..." | tee -a "${LOG}"
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    if [ -n "${VLLM_URLS}" ]; then
        VLLM_URLS="${VLLM_URLS},http://${VLLM_NODE_IP}:${PORT}"
    else
        VLLM_URLS="http://${VLLM_NODE_IP}:${PORT}"
    fi
done

ssh "${VLLM_NODE}" "bash -s" <<EOF | tee -a "${LOG}"
set -o pipefail
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
# PYTHONNOUSERSITE must NOT be set for vLLM workers (disables usercustomize.py
# that patches vllm.model_executor.models.registry for XPU). Train side sets it.
unset PYTHONNOUSERSITE
export PYTHONPATH='${VLLM_PYTHONPATH}'
export no_proxy='*'
export NO_PROXY='*'
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-xccl_sendrecv}
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD:-xccl}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
# Cap Rayon (HF tokenizers chat-template) and Tokio thread pools at 1 worker.
# 12 vLLM tiles each spawning a multi-threaded Rayon pool exhausts the
# per-process thread limit (EAGAIN / ThreadPoolBuildError "WouldBlock").
# Validated 2026-05-02: dropping these to 1 fixes the chat-template panic.
export RAYON_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1
export RUST_BACKTRACE=1
mkdir -p /tmp/torchtune
> /tmp/torchtune/vllm_pids.txt
for i in \$(seq 0 $((VLLM_DP-1))); do
    PORT=\$((${VLLM_BASE_PORT} + i))
    LOG_R=/tmp/torchtune/vllm_http_tile\${i}.log
    echo "[VLLM_NODE] Launching tile \${i} on port \${PORT}"
    setsid nohup env ZE_AFFINITY_MASK=\${i} PYTHONUNBUFFERED=1 python3 -m vllm.entrypoints.openai.api_server \\
        --model '${VLLM_MODEL_LOCAL}' \\
        --tensor-parallel-size 1 \\
        --port \${PORT} \\
        --host 0.0.0.0 \\
        --enforce-eager \\
        --dtype bfloat16 \\
        --gpu-memory-utilization ${VLLM_GPU_MEM} \\
        --max-model-len ${VLLM_MAX_MODEL_LEN} \\
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \\
        --distributed-executor-backend mp \\
        --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \\
        > "\${LOG_R}" 2>&1 < /dev/null &
    echo \$! >> /tmp/torchtune/vllm_pids.txt
done
echo "[VLLM_NODE] Launched ${VLLM_DP} tiles, PIDs:"
cat /tmp/torchtune/vllm_pids.txt
EOF
VLLM_LAUNCH_RC=$?
if [ "${VLLM_LAUNCH_RC}" -ne 0 ]; then
    echo "ERROR: vLLM launch SSH returned non-zero (${VLLM_LAUNCH_RC})." | tee -a "${LOG}"
    exit "${VLLM_LAUNCH_RC}"
fi

echo "vLLM URLs: ${VLLM_URLS}" | tee -a "${LOG}"

# ============================================================
# Cleanup trap
# ============================================================
cleanup() {
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then
        echo "Cleanup: VLLM_LEAVE_RUNNING=1, leaving vLLM tiles alive on ${VLLM_NODE}." | tee -a "${LOG}"
        return 0
    fi
    echo "Cleaning up..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
" 2>/dev/null || true
    ssh "${TRAIN_NODE}" "pkill -9 -f 'grpo_full_finet[u]ne_distributed_xpu' 2>/dev/null" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup done." | tee -a "${LOG}"
}
trap cleanup EXIT

# ============================================================
# Wait for all ${VLLM_DP} vLLM servers to be healthy.
# Single persistent watcher SSH — avoids the SSH storm from polling.
# ============================================================
echo "Waiting for ${VLLM_DP} vLLM servers (${VLLM_STARTUP_TIMEOUT}s timeout)..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "VLLM_DP=${VLLM_DP} VLLM_BASE_PORT=${VLLM_BASE_PORT} TIMEOUT=${VLLM_STARTUP_TIMEOUT} bash -s" <<'WATCH' | tee -a "${LOG}"
mapfile -t PIDS < /tmp/torchtune/vllm_pids.txt
if [ "${#PIDS[@]}" -ne "${VLLM_DP}" ]; then
    echo "FATAL: PID file has ${#PIDS[@]} entries, expected ${VLLM_DP}"
    exit 1
fi
TIMEOUT=${TIMEOUT:-600}
declare -a READY
for i in $(seq 0 $((VLLM_DP-1))); do READY[i]=0; done
DEADLINE=$(( $(date +%s) + TIMEOUT ))
while :; do
    all=1
    for i in $(seq 0 $((VLLM_DP-1))); do
        [ "${READY[i]}" -eq 1 ] && continue
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            echo "FATAL tile $i: PID ${PIDS[i]} died during startup"
            echo "--- tail /tmp/torchtune/vllm_http_tile${i}.log ---"
            tail -60 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
            exit 1
        fi
        port=$((VLLM_BASE_PORT + i))
        if curl --noproxy '*' -s --max-time 2 -o /dev/null "http://localhost:${port}/health"; then
            elapsed=$(( $(date +%s) - (DEADLINE - TIMEOUT) ))
            echo "  Tile $i healthy on port ${port} (${elapsed}s)"
            READY[i]=1
        else
            all=0
        fi
    done
    if [ "$all" -eq 1 ]; then
        echo "All ${VLLM_DP} vLLM servers ready."
        exit 0
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "FATAL: not all tiles ready within ${TIMEOUT}s"
        for i in $(seq 0 $((VLLM_DP-1))); do
            [ "${READY[i]}" -eq 1 ] && continue
            echo "--- tile $i (port $((VLLM_BASE_PORT+i))), PID ${PIDS[i]} ---"
            tail -40 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
        done
        exit 1
    fi
    sleep 3
done
WATCH
WATCH_RC=${PIPESTATUS[0]}
if [ "${WATCH_RC}" -ne 0 ]; then
    echo "WARNING: vLLM startup watcher exited ${WATCH_RC} (transient tile crash?). Retrying startup once..." | tee -a "${LOG}"
    # Kill any surviving tile processes before retry
    ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
sleep 15
bash ${TT_DIR}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -10 || true
sleep 5
" 2>&1 | tee -a "${LOG}" || true
    # Re-launch all tiles (same settings as initial launch)
    echo "Retry: relaunching ${VLLM_DP} vLLM servers on ${VLLM_NODE}..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "bash -s" <<RETRY_LAUNCH 2>&1 | tee -a "${LOG}"
set -o pipefail
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV PYTHONNOUSERSITE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export PYTHONPATH='${VLLM_PYTHONPATH}'
export no_proxy='*' ; export NO_PROXY='*'
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export HF_DATASETS_OFFLINE=1 ; export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none ; export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi ; export CCL_KVS_IFACE=lo
# Same Rayon/Tokio thread-pool caps as initial launch (see comment above).
export RAYON_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1
export RUST_BACKTRACE=1
mkdir -p /tmp/torchtune
> /tmp/torchtune/vllm_pids.txt
for i in \$(seq 0 $((VLLM_DP-1))); do
    PORT=\$((${VLLM_BASE_PORT} + i))
    LOG_R=/tmp/torchtune/vllm_http_tile\${i}.log
    echo "[VLLM_NODE retry] Launching tile \${i} on port \${PORT}"
    setsid nohup env ZE_AFFINITY_MASK=\${i} PYTHONUNBUFFERED=1 python3 -m vllm.entrypoints.openai.api_server \
        --model '${VLLM_MODEL_LOCAL}' \
        --tensor-parallel-size 1 \
        --port \${PORT} \
        --host 0.0.0.0 \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization ${VLLM_GPU_MEM} \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --distributed-executor-backend mp \
        --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \
        > "\${LOG_R}" 2>&1 < /dev/null &
    echo \$! >> /tmp/torchtune/vllm_pids.txt
done
echo "[VLLM_NODE retry] Launched ${VLLM_DP} tiles, PIDs:"
cat /tmp/torchtune/vllm_pids.txt
RETRY_LAUNCH
    RETRY_LAUNCH_RC=${PIPESTATUS[0]}
    if [ "${RETRY_LAUNCH_RC}" -ne 0 ]; then
        echo "ERROR: retry launch failed (${RETRY_LAUNCH_RC})." | tee -a "${LOG}"
        exit "${RETRY_LAUNCH_RC}"
    fi
    # Re-run startup watcher
    echo "Retry: waiting for ${VLLM_DP} vLLM servers (${VLLM_STARTUP_TIMEOUT}s timeout)..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "VLLM_DP=${VLLM_DP} VLLM_BASE_PORT=${VLLM_BASE_PORT} TIMEOUT=${VLLM_STARTUP_TIMEOUT} bash -s" <<'RETRY_WATCH' | tee -a "${LOG}"
mapfile -t PIDS < /tmp/torchtune/vllm_pids.txt
TIMEOUT=${TIMEOUT:-600}
declare -a READY
for i in $(seq 0 $((VLLM_DP-1))); do READY[i]=0; done
DEADLINE=$(( $(date +%s) + TIMEOUT ))
while :; do
    all=1
    for i in $(seq 0 $((VLLM_DP-1))); do
        [ "${READY[i]}" -eq 1 ] && continue
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            echo "FATAL tile $i (retry): PID ${PIDS[i]} died during startup"
            echo "--- tail /tmp/torchtune/vllm_http_tile${i}.log ---"
            tail -60 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
            exit 1
        fi
        port=$((VLLM_BASE_PORT + i))
        if curl --noproxy '*' -s --max-time 2 -o /dev/null "http://localhost:${port}/health"; then
            elapsed=$(( $(date +%s) - (DEADLINE - TIMEOUT) ))
            echo "  Tile $i healthy on port ${port} (${elapsed}s)"
            READY[i]=1
        else
            all=0
        fi
    done
    if [ "$all" -eq 1 ]; then echo "All ${VLLM_DP} vLLM servers ready (retry)."; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "FATAL: not all tiles ready within ${TIMEOUT}s (retry)"
        for i in $(seq 0 $((VLLM_DP-1))); do
            [ "${READY[i]}" -eq 1 ] && continue
            echo "--- tile $i (port $((VLLM_BASE_PORT+i))), PID ${PIDS[i]} ---"
            tail -40 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
        done
        exit 1
    fi
    sleep 3
done
RETRY_WATCH
    RETRY_WATCH_RC=${PIPESTATUS[0]}
    if [ "${RETRY_WATCH_RC}" -ne 0 ]; then
        echo "ERROR: vLLM startup retry also failed (${RETRY_WATCH_RC}). Giving up." | tee -a "${LOG}"
        exit "${RETRY_WATCH_RC}"
    fi
    echo "vLLM startup retry SUCCEEDED." | tee -a "${LOG}"
fi

# ============================================================
# Cross-node connectivity preflight
# ============================================================
echo "Preflight: cross-node /health from ${TRAIN_NODE} to all ${VLLM_DP} tiles..." | tee -a "${LOG}"
PREFLIGHT_FAIL=0
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    URL="http://${VLLM_NODE_IP}:${PORT}/health/"
    if ! ssh "${TRAIN_NODE}" "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; curl --noproxy '*' -L -s --max-time 5 -o /dev/null -w '%{http_code}' '${URL}'" 2>&1 | grep -q '^200$'; then
        echo "  PREFLIGHT FAIL: tile ${r} (${URL}) unreachable from TRAIN_NODE" | tee -a "${LOG}"
        PREFLIGHT_FAIL=1
    fi
done
if [ "${PREFLIGHT_FAIL}" -ne 0 ]; then
    echo "ERROR: cross-node connectivity preflight failed; aborting." | tee -a "${LOG}"
    exit 1
fi
echo "Preflight OK: all ${VLLM_DP} tiles reachable cross-node." | tee -a "${LOG}"

# ============================================================
# EngineCore inference warmup + restart-on-failure
# /health returns 200 even when the EngineCore is broken (Aurora XPU stale
# L0 driver state from prior jobs on the same node).  Test with a real
# generate call; restart all tiles once if it fails.
# ============================================================
_vllm_launch_tiles() {
    # Re-launch all VLLM_DP tiles on VLLM_NODE and write new PIDs.
    ssh "${VLLM_NODE}" "bash -s" <<RELAUNCH 2>&1 | tee -a "${LOG}"
set -o pipefail
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
unset PYTHONNOUSERSITE
export PYTHONPATH='${VLLM_PYTHONPATH}'
export no_proxy='*'
export NO_PROXY='*'
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
# Same Rayon/Tokio thread-pool caps as initial launch (see comment above).
export RAYON_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1
export RUST_BACKTRACE=1
mkdir -p /tmp/torchtune
> /tmp/torchtune/vllm_pids.txt
for i in \$(seq 0 $((VLLM_DP-1))); do
    PORT=\$((${VLLM_BASE_PORT} + i))
    LOG_R=/tmp/torchtune/vllm_http_tile\${i}.log
    echo "[VLLM_NODE restart] Launching tile \${i} on port \${PORT}"
    setsid nohup env ZE_AFFINITY_MASK=\${i} PYTHONUNBUFFERED=1 python3 -m vllm.entrypoints.openai.api_server \
        --model '${VLLM_MODEL_LOCAL}' \
        --tensor-parallel-size 1 \
        --port \${PORT} \
        --host 0.0.0.0 \
        --enforce-eager \
        --dtype bfloat16 \
        --gpu-memory-utilization ${VLLM_GPU_MEM} \
        --max-model-len ${VLLM_MAX_MODEL_LEN} \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --distributed-executor-backend mp \
        --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \
        > "\${LOG_R}" 2>&1 < /dev/null &
    echo \$! >> /tmp/torchtune/vllm_pids.txt
done
echo "[VLLM_NODE restart] Launched ${VLLM_DP} tiles."
RELAUNCH
}

_vllm_wait_healthy() {
    ssh "${VLLM_NODE}" "VLLM_DP=${VLLM_DP} VLLM_BASE_PORT=${VLLM_BASE_PORT} TIMEOUT=${VLLM_STARTUP_TIMEOUT} bash -s" <<'HWAIT' 2>&1 | tee -a "${LOG}"
mapfile -t PIDS < /tmp/torchtune/vllm_pids.txt
TIMEOUT=${TIMEOUT:-600}
declare -a READY
for i in $(seq 0 $((VLLM_DP-1))); do READY[i]=0; done
DEADLINE=$(( $(date +%s) + TIMEOUT ))
while :; do
    all=1
    for i in $(seq 0 $((VLLM_DP-1))); do
        [ "${READY[i]}" -eq 1 ] && continue
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            echo "FATAL tile $i: PID ${PIDS[i]} died during startup"
            tail -30 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
            exit 1
        fi
        port=$((VLLM_BASE_PORT + i))
        if curl --noproxy '*' -s --max-time 2 -o /dev/null "http://localhost:${port}/health"; then
            elapsed=$(( $(date +%s) - (DEADLINE - TIMEOUT) ))
            echo "  Tile $i healthy on port ${port} (${elapsed}s)"
            READY[i]=1
        else
            all=0
        fi
    done
    if [ "$all" -eq 1 ]; then echo "All ${VLLM_DP} vLLM servers ready."; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "FATAL: not all tiles ready within ${TIMEOUT}s"; exit 1
    fi
    sleep 3
done
HWAIT
}

for _warmup_try in 1 2; do
    echo "EngineCore inference warmup (attempt ${_warmup_try}/2)..." | tee -a "${LOG}"
    _warmup_ok=1
    for ((r=0; r<VLLM_DP; r++)); do
        _wport=$((VLLM_BASE_PORT + r))
        _wresp=$(ssh "${VLLM_NODE}" "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; \
            curl --noproxy '*' -sf --max-time 30 -X POST 'http://localhost:${_wport}/v1/completions' \
            -H 'Content-Type: application/json' \
            -d '{\"model\": \"${VLLM_MODEL_LOCAL}\", \"prompt\": \"1+1=\", \"max_tokens\": 1}' 2>&1")
        if echo "${_wresp}" | grep -q '"choices"'; then
            echo "  Tile ${r}: EngineCore OK" | tee -a "${LOG}"
        else
            echo "  Tile ${r}: EngineCore FAIL: ${_wresp:0:300}" | tee -a "${LOG}"
            _warmup_ok=0
        fi
    done
    if [ "${_warmup_ok}" -eq 1 ]; then
        echo "vLLM EngineCore warmup PASSED (attempt ${_warmup_try})." | tee -a "${LOG}"
        break
    fi
    if [ "${_warmup_try}" -eq 2 ]; then
        echo "ERROR: vLLM EngineCore still broken after restart; aborting." | tee -a "${LOG}"
        exit 1
    fi
    echo "vLLM EngineCore broken on attempt 1; restarting all ${VLLM_DP} tiles..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
sleep 5
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -10 || true
sleep 5
" 2>&1 | tee -a "${LOG}" || true
    _vllm_launch_tiles || { echo "ERROR: vLLM relaunch failed." | tee -a "${LOG}"; exit 1; }
    _vllm_wait_healthy || { echo "ERROR: restarted vLLM never became healthy." | tee -a "${LOG}"; exit 1; }
done

# Test hook: VLLM_ONLY=1 stops here.
if [ "${VLLM_ONLY:-0}" = "1" ]; then
    echo "VLLM_ONLY=1: vLLM startup validated; skipping training. Exiting 0." | tee -a "${LOG}"
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then
        echo "VLLM_LEAVE_RUNNING=1: clearing EXIT trap; tiles will remain alive." | tee -a "${LOG}"
        trap - EXIT
    fi
    exit 0
fi

# ============================================================
# Launch training on TRAIN_NODE (single node, 11 ranks)
# ============================================================
echo "" | tee -a "${LOG}"
echo "Starting ${TRAIN_TILES}-rank training on ${TRAIN_NODE}..." | tee -a "${LOG}"

TRAIN_LOG="/tmp/torchtune/train_node0.log"
TRAIN_PID_FILE="/tmp/torchtune/train_pid.txt"
TRAIN_EXIT_FILE="/tmp/torchtune/train_exit.txt"

ssh "${TRAIN_NODE}" "bash -s" <<EOF 2>&1 | tee -a "${LOG}"
set -e
mkdir -p /tmp/torchtune
rm -f ${TRAIN_PID_FILE} ${TRAIN_EXIT_FILE}
> ${TRAIN_LOG}
setsid nohup bash -c '
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ":" "\n" | grep -v myenv | tr "\n" ":" | sed "s/:\\\$//")
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.95
export TORCH_COMPILE_DISABLE=1
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-1}
export TORCHTUNE_PINNED_CPU_BUF=${TORCHTUNE_PINNED_CPU_BUF:-1}
export TORCHTUNE_USE_IPEX_VARLEN=${TORCHTUNE_USE_IPEX_VARLEN:-0}
export TORCHTUNE_MASKFREE_CAUSAL=${TORCHTUNE_MASKFREE_CAUSAL:-0}
export TORCHTUNE_VARLEN_NOGRAD_BYPASS=${TORCHTUNE_VARLEN_NOGRAD_BYPASS:-0}
export TORCHTUNE_SKIP_REF_FWD=${TORCHTUNE_SKIP_REF_FWD:-0}
export TORCHTUNE_SKIP_GRPO_STEP=${TORCHTUNE_SKIP_GRPO_STEP:-0}
export TORCHTUNE_SKIP_GRPO_BACKWARD=${TORCHTUNE_SKIP_GRPO_BACKWARD:-0}
export TORCHTUNE_SKIP_GRPO_UPDATE=${TORCHTUNE_SKIP_GRPO_UPDATE:-0}
export TORCHTUNE_GRPO_BACKWARD_NO_SYNC=${TORCHTUNE_GRPO_BACKWARD_NO_SYNC:-0}
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-xccl_sendrecv}
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD:-xccl}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/lus/flare/projects/ModCon/ngetty/hf_datasets_cache}
export HF_HOME=${HF_HOME:-/lus/flare/projects/ModCon/ngetty/hf_cache}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TRAIN_PYTHONPATH}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*"
export NO_PROXY="*"
mkdir -p /tmp/torchtune
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${TRAIN_TILES} \
    --redirects 3 --tee 3 \
    --log-dir /tmp/torchtune/torchelastic_logs \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} \
    grpo_samples=${GRPO_SAMPLES} \
    forward_batch_size=${FORWARD_BATCH_SIZE} \
    ref_forward_batch_size=${REF_FORWARD_BATCH_SIZE} \
    max_generated_tokens=${MAX_GEN_TOKENS} \
    vllm_url="${VLLM_URLS}" \
    log_peak_memory_stats=true ${EXTRA_OVERRIDES:-} > ${TRAIN_LOG} 2>&1 < /dev/null
echo \$? > ${TRAIN_EXIT_FILE}
' < /dev/null > /dev/null 2>&1 &
echo \$! > ${TRAIN_PID_FILE}
echo "[TRAIN_NODE] dispatched detached training, PID=\$(cat ${TRAIN_PID_FILE})"
EOF
DISPATCH_RC=$?
if [ "${DISPATCH_RC}" -ne 0 ]; then
    echo "ERROR: train dispatch SSH returned ${DISPATCH_RC}." | tee -a "${LOG}"
    exit "${DISPATCH_RC}"
fi

# Persistent watcher SSH. Retries up to 3× on SSH drop; training is decoupled.
echo "Watching detached training on ${TRAIN_NODE}..." | tee -a "${LOG}"
TRAIN_EXIT=99
WATCH_TRIES=0
while [ "${WATCH_TRIES}" -lt 3 ]; do
    WATCH_TRIES=$((WATCH_TRIES + 1))
    ssh "${TRAIN_NODE}" "TRAIN_LOG=${TRAIN_LOG} TRAIN_PID_FILE=${TRAIN_PID_FILE} TRAIN_EXIT_FILE=${TRAIN_EXIT_FILE} bash -s" <<'WATCH' 2>&1 | tee -a "${LOG}"
TPID=$(cat ${TRAIN_PID_FILE} 2>/dev/null)
if [ -z "${TPID}" ]; then
    echo "WATCH: no PID file; training never dispatched"
    exit 98
fi
( tail -n +1 -F ${TRAIN_LOG} 2>/dev/null ) &
TAIL_PID=$!
trap "kill ${TAIL_PID} 2>/dev/null" EXIT
while kill -0 ${TPID} 2>/dev/null; do
    sleep 5
done
sleep 3
kill ${TAIL_PID} 2>/dev/null
EC=$(cat ${TRAIN_EXIT_FILE} 2>/dev/null)
echo "WATCH: training PID ${TPID} exited (exit_code=${EC:-unknown})"
exit ${EC:-97}
WATCH
    WATCH_RC=${PIPESTATUS[0]}
    EXIT_FILE_EXISTS=$(ssh "${TRAIN_NODE}" "test -f ${TRAIN_EXIT_FILE} && echo yes || echo no" 2>/dev/null)
    if [ "${EXIT_FILE_EXISTS}" = "yes" ]; then
        TRAIN_EXIT=$(ssh "${TRAIN_NODE}" "cat ${TRAIN_EXIT_FILE}" 2>/dev/null || echo "${WATCH_RC}")
        break
    fi
    echo "WARN: watcher SSH dropped (rc=${WATCH_RC}); reattaching (try ${WATCH_TRIES}/3)..." | tee -a "${LOG}"
    sleep 5
done

echo "=== Dense GRPO 2-Node: exit=${TRAIN_EXIT} at $(date) ===" | tee -a "${LOG}"
exit "${TRAIN_EXIT}"

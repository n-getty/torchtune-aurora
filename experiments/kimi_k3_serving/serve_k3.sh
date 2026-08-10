#!/usr/bin/env bash
set -euo pipefail

# Start a text-only vLLM OpenAI server on the current PBS allocation.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAY_ENV=${RAY_ENV:-/lus/flare/projects/ModCon/ngetty/torchtune/experiments/ray_smoke/setup_ray_env.sh}
MODEL=${MODEL:-}
TP=${TP:-32}
PP=${PP:-1}
DP=${DP:-1}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
BLOCKS=${BLOCKS:-}
DIAGNOSTIC_BLOCKS=${DIAGNOSTIC_BLOCKS:-0}
LOAD_FORMAT=${LOAD_FORMAT:-auto}
SAFETENSORS_LOAD_STRATEGY=${SAFETENSORS_LOAD_STRATEGY:-}
MULTITHREAD_LOAD=${MULTITHREAD_LOAD:-0}
LOAD_THREADS=${LOAD_THREADS:-8}
MODEL_LOADER_EXTRA_CONFIG=${MODEL_LOADER_EXTRA_CONFIG:-'{"enable_weights_track":true}'}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-}
EP=${EP:-0}
RAY_V2=${RAY_V2:-0}
ASYNC_SCHEDULING=${ASYNC_SCHEDULING:-}
LOG_DIR=${LOG_DIR:-$(pwd)/logs/$(date +%Y%m%d_%H%M%S)_server}
STAGE_MODEL=${STAGE_MODEL:-0}
STAGE_ROOT=${STAGE_ROOT:-/tmp/kimi_k3_models}
STAGE_ONLY=${STAGE_ONLY:-0}
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
VLLM_SRC=${VLLM_SRC:-/flare/ModCon/ngetty/vllm-xpu-src}
SYCL_LIB_DIR=${SYCL_LIB_DIR:-/opt/aurora/26.26.0/oneapi/2025.3/lib}
VERIFY_CHECKPOINT=${VERIFY_CHECKPOINT:-0}
CHECKPOINT_VERIFIER=${CHECKPOINT_VERIFIER:-$SCRIPT_DIR/verify_checkpoint.py}
K3_JOB_ID=${PBS_JOBID:-${K3_JOB_ID:-}}
[[ -n "$K3_JOB_ID" ]] || { echo "ERROR: PBS_JOBID or K3_JOB_ID is required" >&2; exit 1; }
export PBS_JOBID="$K3_JOB_ID"
export DAOS_AGENT_DRPC_DIR=${DAOS_AGENT_DRPC_DIR:-/run/daos_agent_oneScratch}
export D_AGENT_DRPC_DIR=${D_AGENT_DRPC_DIR:-/run/daos_agent_oneScratch}
export VLLM_KIMI_XPU_REQUEST_DIAGNOSTIC_LIMIT=${VLLM_KIMI_XPU_REQUEST_DIAGNOSTIC_LIMIT:-4096}
export VLLM_KIMI_XPU_KDA_VECTORIZED=${VLLM_KIMI_XPU_KDA_VECTORIZED:-0}
export VLLM_KIMI_XPU_CONV1D_VECTORIZED=${VLLM_KIMI_XPU_CONV1D_VECTORIZED:-0}
export VLLM_XPU_ALLOW_TRITON_SAMPLER=${VLLM_XPU_ALLOW_TRITON_SAMPLER:-0}
for ray_env_name in K3_BLOCK_PROFILE_DIR K3_LOADER_ACCOUNTING_DIR VLLM_KIMI_XPU_REQUEST_DIAGNOSTIC_LIMIT; do
    if [[ ",${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-}," != *,${ray_env_name},* ]]; then
        VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:+${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY},}${ray_env_name}"
    fi
done
unset ray_env_name
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
K3_CACHE_ROOT=${K3_CACHE_ROOT:-/tmp/k3_hf_cache_${K3_JOB_ID//[^A-Za-z0-9_.-]/_}}
K3_CACHE_MARKER="$K3_CACHE_ROOT/.created_by_${K3_JOB_ID//[^A-Za-z0-9_.-]/_}"
RAY_TEMP_ID=${K3_JOB_ID%%.*}
RAY_TEMP_ROOT=${RAY_TEMP_ROOT:-/tmp/k3_ray_${RAY_TEMP_ID}}

usage() { echo "Usage: $0 --model PATH [--diagnostic-blocks N] [--tp N] [--safetensors-load-strategy STRATEGY] [--model-loader-extra-config JSON] [--multithread-load [N]] [--stage-model] [--stage-only] [--ep] [--ray-v2]"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --tp) TP=$2; shift 2 ;;
        --pp) PP=$2; shift 2 ;;
        --dp) DP=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS=$2; shift 2 ;;
        --max-num-batched-tokens) MAX_BATCHED_TOKENS=$2; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL=$2; shift 2 ;;
        --blocks) BLOCKS=$2; DIAGNOSTIC_BLOCKS=1; shift 2 ;;
        --diagnostic-blocks) BLOCKS=$2; DIAGNOSTIC_BLOCKS=1; shift 2 ;;
        --load-format) LOAD_FORMAT=$2; shift 2 ;;
        --safetensors-load-strategy) SAFETENSORS_LOAD_STRATEGY=$2; shift 2 ;;
        --multithread-load)
            MULTITHREAD_LOAD=1
            if [[ $# -gt 1 && "$2" != --* ]]; then
                LOAD_THREADS=$2
                shift 2
            else
                shift
            fi
            ;;
        --model-loader-extra-config) MODEL_LOADER_EXTRA_CONFIG=$2; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME=$2; shift 2 ;;
        --python) PYTHON=$2; shift 2 ;;
        --stage-model) STAGE_MODEL=1; shift ;;
        --stage-only) STAGE_MODEL=1; STAGE_ONLY=1; shift ;;
        --ep) EP=1; shift ;;
        --ray-v2) RAY_V2=1; shift ;;
        --no-async-scheduling) ASYNC_SCHEDULING=0; shift ;;
        --async-scheduling) ASYNC_SCHEDULING=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$MULTITHREAD_LOAD" == 0 || "$MULTITHREAD_LOAD" == 1 ]] || {
    echo "ERROR: MULTITHREAD_LOAD must be 0 or 1" >&2
    exit 2
}
if ! [[ "$LOAD_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: load thread count must be a positive integer: $LOAD_THREADS" >&2
    exit 2
fi

[[ -n "$MODEL" ]] || { echo "ERROR: --model is required" >&2; exit 2; }
if [[ "$VERIFY_CHECKPOINT" == 1 ]]; then
    [[ -f "$CHECKPOINT_VERIFIER" ]] || {
        echo "ERROR: checkpoint verifier does not exist: $CHECKPOINT_VERIFIER" >&2
        exit 1
    }
    python3 "$CHECKPOINT_VERIFIER" "$MODEL"
fi
if [[ -n "${K3_NODEFILE:-}" && -f "$K3_NODEFILE" ]]; then
    PBS_NODEFILE=$K3_NODEFILE
    export PBS_NODEFILE
fi
if [[ ! -f "${PBS_NODEFILE:-}" ]]; then
    for _ in $(seq 1 30); do
        mapfile -t discovered_nodefiles < <(find /var/spool/pbs/aux -maxdepth 1 -type f -name "${PBS_JOBID}*" -print)
        [[ ${#discovered_nodefiles[@]} -eq 1 ]] && {
            PBS_NODEFILE=${discovered_nodefiles[0]}
            export PBS_NODEFILE
            break
        }
        sleep 1
    done
fi
[[ -f "${PBS_NODEFILE:-}" ]] || { echo "ERROR: PBS_NODEFILE does not exist: ${PBS_NODEFILE:-unset}" >&2; exit 1; }
if [[ "$STAGE_ONLY" != 1 && -n "$BLOCKS" && "$DIAGNOSTIC_BLOCKS" != 1 ]]; then
    echo "ERROR: block override is diagnostic-only; use --diagnostic-blocks" >&2
    exit 2
fi
if [[ "$EP" == 1 ]]; then
    [[ -f "$MODEL/config.json" ]] || { echo "ERROR: EP requires a model config: $MODEL/config.json" >&2; exit 2; }
    if ! grep -Eq '"(num_experts|num_local_experts)"[[:space:]]*:[[:space:]]*[1-9][0-9]*' "$MODEL/config.json"; then
        echo "ERROR: --ep requires a MoE model with num_experts > 0: $MODEL" >&2
        exit 2
    fi
fi
if command -v qstat >/dev/null; then
    # -w (wide) disables qstat's ~80-column line wrapping. Without it, a long
    # exec_host value (>= ~4-5 nodes' worth of hostnames) wraps onto tab-
    # indented continuation lines that the single-line awk below silently
    # truncates -- it grabbed only line 1 of exec_host, cutting a hostname
    # mid-string (observed on an 8-node job: "...x4704c3s2b0n0/" got cut to
    # "...x43"), which then always fails the host-count/membership check
    # below even though the allocation is completely valid. 3-node jobs
    # never wrapped, so this was invisible until scaling past ~4-5 nodes.
    qstat_output=$(qstat -f -w "$PBS_JOBID" 2>/dev/null) || {
        echo "ERROR: unable to query PBS job $PBS_JOBID" >&2
        exit 1
    }
    job_state=$(awk -F'= ' '/job_state/ {print $2; exit}' <<<"$qstat_output")
    [[ "$job_state" == R ]] || { echo "ERROR: allocation $PBS_JOBID is not running (state=${job_state:-unknown})" >&2; exit 1; }
    exec_host=$(awk -F'= ' '/^[[:space:]]*exec_host[[:space:]]*=/ {print $2; exit}' <<<"$qstat_output")
    [[ -n "$exec_host" ]] || { echo "ERROR: running allocation $PBS_JOBID has no exec_host" >&2; exit 1; }
    mapfile -t allocated_hosts < <(tr '+' '\n' <<<"$exec_host" | sed -E 's#/.*##; s#\..*$##' | sort -u)
    mapfile -t nodefile_hosts < <(sed -E 's/[[:space:]].*$//; s#\..*$##' "$PBS_NODEFILE" | sort -u)
    [[ ${#allocated_hosts[@]} -eq ${#nodefile_hosts[@]} ]] || {
        echo "ERROR: PBS exec_host/nodefile host-count mismatch: exec_host=${allocated_hosts[*]} nodefile=${nodefile_hosts[*]}" >&2
        exit 1
    }
    for host in "${nodefile_hosts[@]}"; do
        if ! printf '%s\n' "${allocated_hosts[@]}" | grep -Fxq "$host"; then
            echo "ERROR: PBS exec_host does not match PBS_NODEFILE host $host: $exec_host" >&2
            exit 1
        fi
    done
fi

mkdir -p "$LOG_DIR"
case "$K3_CACHE_ROOT" in
    ""|/|/tmp|/tmp/) echo "ERROR: unsafe K3_CACHE_ROOT: $K3_CACHE_ROOT" >&2; exit 2 ;;
esac
if [[ -e "$K3_CACHE_ROOT" ]]; then
    echo "ERROR: K3_CACHE_ROOT already exists; refusing to reuse it: $K3_CACHE_ROOT" >&2
    exit 1
fi
mkdir "$K3_CACHE_ROOT"
export HF_HOME="$K3_CACHE_ROOT/hf"
export HF_MODULES_CACHE="$K3_CACHE_ROOT/modules"
export HF_HUB_CACHE="$K3_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$K3_CACHE_ROOT/transformers"
export XDG_CACHE_HOME="$K3_CACHE_ROOT/xdg"
printf '%s\n' "$PBS_JOBID" >"$K3_CACHE_MARKER"
cleanup_cache() {
    if [[ -f "$K3_CACHE_MARKER" ]] && grep -Fxq "$PBS_JOBID" "$K3_CACHE_MARKER" 2>/dev/null; then
        rm -rf -- "$K3_CACHE_ROOT"
    fi
}
trap cleanup_cache EXIT
if [[ -d "$VLLM_SRC/vllm" ]]; then
    export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi
PYTHON_SITE_PACKAGES=$($PYTHON -c 'import site; print(site.getsitepackages()[0])')
if [[ -d "$PYTHON_SITE_PACKAGES" ]]; then
    export PYTHONPATH="$PYTHON_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -d "$SYCL_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$SYCL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
export VLLM_TARGET_DEVICE=xpu
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0}
export VLLM_XPU_DETERMINISTIC_ROUTING=${VLLM_XPU_DETERMINISTIC_ROUTING:-1}
# Default 0: =1 is the slow hand-patched Python row-gather (~4.5x cost) and
# also alters numerics; the Step-5 correctness pass against HF ground truth
# used =0. See the K3 investigation plan (Step 1) -- =1 is opt-in only for
# deliberate A/B, not a safe default.
export VLLM_XPU_DETERMINISTIC_MOE_GATHER=${VLLM_XPU_DETERMINISTIC_MOE_GATHER:-0}
# Timeouts as a safety net, not a fix (see plan step 4): these stop a
# slow-but-progressing run from being killed. ray_executor.py:556 uses
# os.environ.setdefault for RAY_CGRAPH_get_timeout, so an external export
# here wins over vLLM's 300s default. Must be exported before `ray.dag` is
# imported (i.e. before any Ray/vLLM Python process starts), which is
# everywhere below this point in the script.
export RAY_CGRAPH_get_timeout=${RAY_CGRAPH_get_timeout:-300}
export RAY_CGRAPH_submit_timeout=${RAY_CGRAPH_submit_timeout:-300}
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-120000}
# Separate from all three above: this is the single-node `mp` executor's OWN
# RPC timeout (multiproc_executor.py's get_response -> "RPC call to
# sample_tokens timed out"), independent of the Ray-path timeouts. Found via
# the K3 investigation plan's Step 4 48B ladder: the KDA/causal_conv1d Python
# fallback exceeded the 300s default on a 64-token prompt (TP=2, single node)
# and killed the engine with EngineDeadError, orphaning both worker
# processes. Raise it the same way, as a safety net -- it does not make the
# fallback path fast, it just stops a slow-but-progressing single-node run
# from being killed at exactly the boundary this flag's default sits at.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}
mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
[[ ${#NODES[@]} -gt 0 ]] || { echo "ERROR: PBS_NODEFILE has no nodes" >&2; exit 1; }
# DP replicates the whole model, so the allocation must cover DP x TP tiles.
# At 12 tiles/node: DP=1 needs 3 nodes (32 of 36), DP=3 needs 8 (96 of 96).
if [[ "$DP" != 1 ]]; then
    required_tiles=$((TP * DP))
    available_tiles=$(( ${#NODES[@]} * 12 ))
    if [[ $available_tiles -lt $required_tiles ]]; then
        echo "ERROR: DP=$DP x TP=$TP needs $required_tiles tiles but the allocation has $available_tiles (${#NODES[@]} nodes x 12)" >&2
        exit 2
    fi
    echo "data_parallel=$DP tensor_parallel=$TP total_ranks=$required_tiles nodes=${#NODES[@]}"
fi
if [[ "$TP" == 32 && ${#NODES[@]} -lt 3 ]]; then
    echo "ERROR: Kimi-K3 TP=32 requires at least 3 nodes; refusing undersized allocation" >&2
    exit 2
fi
CURRENT_NODE=$(hostname -s)
MODEL_FOR_SERVER=$MODEL
if [[ "$STAGE_MODEL" == 1 ]]; then
    [[ -d "$MODEL" ]] || { echo "ERROR: model directory does not exist: $MODEL" >&2; exit 1; }
    MODEL_REAL=$(realpath -e "$MODEL")
    MODEL_FOR_SERVER="$STAGE_ROOT/$(basename "$MODEL")"
    STAGE_MARKER="$MODEL_FOR_SERVER/.stage_complete"
    STAGE_TMP="${MODEL_FOR_SERVER}.partial.$$"
    STAGE_SOURCE_ID=$(stat -c '%n:%s:%Y' "$MODEL/config.json")
    case "$STAGE_ROOT" in
        ""|/) echo "ERROR: unsafe STAGE_ROOT: $STAGE_ROOT" >&2; exit 2 ;;
    esac
    STAGE_ROOT_REAL=$(realpath -m "$STAGE_ROOT")
    case "$MODEL_REAL" in
        "$STAGE_ROOT_REAL"|"$STAGE_ROOT_REAL"/*)
            echo "ERROR: STAGE_ROOT overlaps source model: $STAGE_ROOT" >&2
            exit 2
            ;;
    esac
    case "$STAGE_ROOT_REAL" in
        "$MODEL_REAL"|"$MODEL_REAL"/*)
            echo "ERROR: STAGE_ROOT is inside source model: $STAGE_ROOT" >&2
            exit 2
            ;;
    esac
    echo "staging_model=$MODEL -> $MODEL_FOR_SERVER nodes=${NODES[*]}" | tee "$LOG_DIR/staging.log"
    cleanup_stage_tmp() {
        rm -rf "$STAGE_TMP"
    }
    trap cleanup_stage_tmp EXIT
    stage_local() {
        if [[ -f "$STAGE_MARKER" ]] && grep -Fxq "$STAGE_SOURCE_ID" "$STAGE_MARKER"; then
            echo "stage_ready node=$(hostname -s) path=$MODEL_FOR_SERVER" >>"$LOG_DIR/staging.log"
            return
        fi
        rm -rf "$STAGE_TMP"
        mkdir -p "$STAGE_ROOT"
        cp -a "$MODEL" "$STAGE_TMP"
        printf '%s\n' "$STAGE_SOURCE_ID" >"$STAGE_TMP/.stage_complete"
        rm -rf "$MODEL_FOR_SERVER"
        mv -T "$STAGE_TMP" "$MODEL_FOR_SERVER"
        echo "stage_complete node=$(hostname -s) path=$MODEL_FOR_SERVER" >>"$LOG_DIR/staging.log"
    }
    stage_remote() {
        local node=$1
        local remote_stage_root=$STAGE_ROOT
        ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" "set -euo pipefail; model='$MODEL'; dest='$MODEL_FOR_SERVER'; marker='$STAGE_MARKER'; tmp='$STAGE_TMP'; source_id='$STAGE_SOURCE_ID'; trap 'rm -rf \"\$tmp\"' EXIT; if [[ -f \"\$marker\" ]] && grep -Fxq \"\$source_id\" \"\$marker\"; then echo stage_ready node=\$(hostname -s) path=\$dest; exit 0; fi; rm -rf \"\$tmp\"; mkdir -p \"$remote_stage_root\"; cp -a \"\$model\" \"\$tmp\"; printf '%s\\n' \"\$source_id\" >\"\$tmp/.stage_complete\"; rm -rf \"\$dest\"; mv -T \"\$tmp\" \"\$dest\"; echo stage_complete node=\$(hostname -s) path=\$dest" \
            >>"$LOG_DIR/staging.log" 2>&1
    }
    stage_local &
    stage_pids=($!)
    for node in "${NODES[@]}"; do
        [[ "${node%%.*}" == "$CURRENT_NODE" ]] && continue
        stage_remote "$node" &
        stage_pids+=("$!")
    done
    stage_rc=0
    for pid in "${stage_pids[@]}"; do
        wait "$pid" || stage_rc=1
    done
    [[ $stage_rc -eq 0 ]] || { echo "ERROR: model staging failed; see $LOG_DIR/staging.log" >&2; exit 1; }
    grep -Fxq "$STAGE_SOURCE_ID" "$STAGE_MARKER" || { echo "ERROR: local staging marker is invalid: $STAGE_MARKER" >&2; exit 1; }
    if [[ "$STAGE_ONLY" == 1 ]]; then
        echo "stage_only_pass model=$MODEL_FOR_SERVER nodes=${NODES[*]}" | tee -a "$LOG_DIR/staging.log"
        trap - EXIT
        exit 0
    fi
    trap - EXIT
fi
HEAD=${NODES[0]}
for node in "${NODES[@]}"; do
    if [[ "${node%%.*}" == "$CURRENT_NODE" ]]; then
        HEAD=$node
        break
    fi
done
resolve_node_ip() {
    local node=$1
    local short_node=${node%%.*}
    local ip
    ip=$(getent hosts "$node" | awk '{print $1}' | head -1)
    ip=${ip:-$(getent hosts "$short_node.hsn.cm.aurora.alcf.anl.gov" | awk '{print $1}' | head -1)}
    ip=${ip:-$(getent hosts "$short_node" | awk '{print $1}' | head -1)}
    printf '%s' "$ip"
}
HEAD_IP=$(resolve_node_ip "$HEAD")
[[ -n "$HEAD_IP" ]] || { echo "ERROR: cannot resolve allocation head node $HEAD" >&2; exit 1; }
RAY_ADDRESS=${RAY_ADDRESS:-$HEAD_IP:6379}
capture_device_snapshot() {
    local output=$1
    "$PYTHON" - <<'PY' 2>"${output%.json}.err" | awk '/^\{/{json=$0} END {if (json != "") print json}' >"$output"
import json
import os
import socket
from datetime import datetime, timezone

import torch

snapshot = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "env": {
        key: os.environ.get(key)
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "ZE_AFFINITY_MASK", "VLLM_HOST_IP")
    },
    "xpu_available": bool(torch.xpu.is_available()),
    "xpu_device_count": 0,
    "devices": [],
}
if snapshot["xpu_available"]:
    snapshot["xpu_device_count"] = torch.xpu.device_count()
    for device_index in range(snapshot["xpu_device_count"]):
        try:
            free_bytes, total_bytes = torch.xpu.mem_get_info(device_index)
            snapshot["devices"].append(
                {"index": device_index, "free_bytes": free_bytes, "total_bytes": total_bytes}
            )
        except Exception as error:
            snapshot["devices"].append({"index": device_index, "error": repr(error)})
print(json.dumps(snapshot, sort_keys=True))
PY
}
capture_device_snapshot "$LOG_DIR/device_snapshot_$(hostname -s).json"
for node in "${NODES[@]}"; do
    [[ "${node%%.*}" == "$CURRENT_NODE" ]] && continue
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" \
        "source '$RAY_ENV' frameworks; $PYTHON -" \
        < <(cat <<'PY'
import json
import os
import socket
from datetime import datetime, timezone

import torch

snapshot = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "env": {
        key: os.environ.get(key)
        for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "ZE_AFFINITY_MASK", "VLLM_HOST_IP")
    },
    "xpu_available": bool(torch.xpu.is_available()),
    "xpu_device_count": 0,
    "devices": [],
}
if snapshot["xpu_available"]:
    snapshot["xpu_device_count"] = torch.xpu.device_count()
    for device_index in range(snapshot["xpu_device_count"]):
        try:
            free_bytes, total_bytes = torch.xpu.mem_get_info(device_index)
            snapshot["devices"].append(
                {"index": device_index, "free_bytes": free_bytes, "total_bytes": total_bytes}
            )
        except Exception as error:
            snapshot["devices"].append({"index": device_index, "error": repr(error)})
print(json.dumps(snapshot, sort_keys=True))
PY
        ) 2>"$LOG_DIR/device_snapshot_${node%%.*}.err" | awk '/^\{/{json=$0} END {if (json != "") print json}' >"$LOG_DIR/device_snapshot_${node%%.*}.json" || {
            echo "WARNING: device snapshot failed on $node; see $LOG_DIR/device_snapshot_${node%%.*}.err" | tee -a "$LOG_DIR/metadata"
        }
done
echo "model=$MODEL model_for_server=$MODEL_FOR_SERVER tp=$TP pp=$PP nodes=${NODES[*]}" | tee "$LOG_DIR/metadata"
echo "node=$(hostname) start=$(date -Is)" | tee -a "$LOG_DIR/metadata"
echo "job_id=$PBS_JOBID ep=$EP diagnostic_blocks=$DIAGNOSTIC_BLOCKS blocks=${BLOCKS:-none} gpu_memory_utilization=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_BATCHED_TOKENS" | tee -a "$LOG_DIR/metadata"
echo "k3_cache_root=$K3_CACHE_ROOT hf_home=$HF_HOME hf_modules_cache=$HF_MODULES_CACHE hf_hub_cache=$HF_HUB_CACHE transformers_cache=$TRANSFORMERS_CACHE xdg_cache_home=$XDG_CACHE_HOME" | tee -a "$LOG_DIR/metadata"
echo "load_format=$LOAD_FORMAT safetensors_load_strategy=${SAFETENSORS_LOAD_STRATEGY:-default} multithread_load=$MULTITHREAD_LOAD load_threads=$LOAD_THREADS model_loader_extra_config=${MODEL_LOADER_EXTRA_CONFIG:-none}" | tee -a "$LOG_DIR/metadata"
echo "ray_v2=$RAY_V2 ray_temp_root=$RAY_TEMP_ROOT" | tee -a "$LOG_DIR/metadata"
echo "vllm_batch_invariant=$VLLM_BATCH_INVARIANT" | tee -a "$LOG_DIR/metadata"
echo "vllm_xpu_deterministic_routing=$VLLM_XPU_DETERMINISTIC_ROUTING" | tee -a "$LOG_DIR/metadata"
echo "vllm_xpu_deterministic_moe_gather=$VLLM_XPU_DETERMINISTIC_MOE_GATHER" | tee -a "$LOG_DIR/metadata"
if [[ -d "$VLLM_SRC/.git" ]]; then
    vllm_commit=$(git -C "$VLLM_SRC" rev-parse HEAD)
    git -C "$VLLM_SRC" status --porcelain=v1 >"$LOG_DIR/vllm_status.txt"
    git -C "$VLLM_SRC" diff --binary HEAD >"$LOG_DIR/vllm_dirty.patch"
    mapfile -t vllm_untracked < <(git -C "$VLLM_SRC" ls-files --others --exclude-standard)
    if [[ ${#vllm_untracked[@]} -gt 0 ]]; then
        tar -C "$VLLM_SRC" -cf "$LOG_DIR/vllm_untracked.tar" "${vllm_untracked[@]}"
    fi
    vllm_diff_id=$(sha256sum "$LOG_DIR/vllm_dirty.patch" | awk '{print $1}')
    echo "vllm_commit=$vllm_commit vllm_diff_sha256=$vllm_diff_id vllm_status=$LOG_DIR/vllm_status.txt vllm_dirty_patch=$LOG_DIR/vllm_dirty.patch" | tee -a "$LOG_DIR/metadata"
fi
"$PYTHON" - <<'PY' >"$LOG_DIR/framework_versions.json"
import importlib.metadata
import json

packages = ("torch", "transformers", "ray", "vllm", "triton")
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps(versions, sort_keys=True))
PY
FRAMEWORK_VERSIONS_FILE="$LOG_DIR/framework_versions.json"
if [[ -d "$VLLM_SRC/.git" ]]; then
    VLLM_COMMIT=$(git -C "$VLLM_SRC" rev-parse HEAD)
    VLLM_DIFF_SHA256=$(sha256sum "$LOG_DIR/vllm_dirty.patch" | awk '{print $1}')
else
    VLLM_COMMIT=NA
    VLLM_DIFF_SHA256=NA
fi
DEVICE_SNAPSHOT_DIR="$LOG_DIR"
BLOCK_PROFILE_DIR="$LOG_DIR/block_profile"
mkdir -p "$BLOCK_PROFILE_DIR"
BLOCK_PROFILE="$LOG_DIR/block_profile.json"
K3_BLOCK_PROFILE_DIR="$BLOCK_PROFILE_DIR"
LOADER_ACCOUNTING_DIR="$LOG_DIR/loader_accounting"
mkdir -p "$LOADER_ACCOUNTING_DIR"
K3_LOADER_ACCOUNTING_DIR="$LOADER_ACCOUNTING_DIR"
export MODEL MODEL_FOR_SERVER SERVED_MODEL_NAME TP PP EP BLOCKS VLLM_COMMIT VLLM_DIFF_SHA256 FRAMEWORK_VERSIONS_FILE DEVICE_SNAPSHOT_DIR BLOCK_PROFILE_DIR K3_BLOCK_PROFILE_DIR BLOCK_PROFILE LOADER_ACCOUNTING_DIR K3_LOADER_ACCOUNTING_DIR
echo "framework_versions=$LOG_DIR/framework_versions.json" | tee -a "$LOG_DIR/metadata"
"$PYTHON" - "$LOG_DIR/metadata.json" <<'PY'
import json
import os
import sys

metadata = {
    "job_id": os.environ["PBS_JOBID"],
    "model": os.environ["MODEL"],
    "model_source": os.environ["MODEL"],
    "served_model": os.environ.get("SERVED_MODEL_NAME") or os.environ["MODEL_FOR_SERVER"],
    "framework_versions": json.dumps(json.load(open(os.environ["FRAMEWORK_VERSIONS_FILE"])), sort_keys=True),
    "vllm_commit": os.environ["VLLM_COMMIT"],
    "vllm_diff_sha256": os.environ["VLLM_DIFF_SHA256"],
    "daos_path": os.environ["MODEL"],
    "device_snapshot_dir": os.environ["DEVICE_SNAPSHOT_DIR"],
    "block_profile": os.environ["BLOCK_PROFILE"],
    "loader_accounting": os.environ["LOADER_ACCOUNTING_DIR"],
    "strict_weight_tracking": True,
    "request_diagnostic_limit": int(
        os.environ["VLLM_KIMI_XPU_REQUEST_DIAGNOSTIC_LIMIT"]
    ),
    "nodes": [line.strip() for line in open(os.environ["PBS_NODEFILE"]) if line.strip()],
    "server": {
        "tp": int(os.environ["TP"]),
        "pp": int(os.environ["PP"]),
        "ep": int(os.environ["EP"]),
        "blocks": os.environ.get("BLOCKS") or "none",
    },
}
with open(sys.argv[1], "w") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
"$PYTHON" - "$LOADER_ACCOUNTING_DIR/expected.json" "$MODEL" <<'PY'
import hashlib
import json
import sys

with open(sys.argv[2] + "/model.safetensors.index.json") as handle:
    names = sorted(json.load(handle)["weight_map"])
skipped = [name for name in names if name.startswith(("vision_tower.", "mm_projector."))]
unexpected = [
    name for name in names
    if not name.startswith(("language_model.", "vision_tower.", "mm_projector."))
]
record = {
    "checkpoint_tensor_count": len(names),
    "skipped_multimodal_count": len(skipped),
    "skipped_multimodal_sha256": hashlib.sha256("\n".join(skipped).encode()).hexdigest(),
    "unexpected_count": len(unexpected),
    "strict_tracking_requested": True,
}
with open(sys.argv[1], "w") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
echo "loader_accounting=$LOADER_ACCOUNTING_DIR expected=$LOADER_ACCOUNTING_DIR/expected.json" | tee -a "$LOG_DIR/metadata"
echo "canonical_metadata=$LOG_DIR/metadata.json" | tee -a "$LOG_DIR/metadata"
"$PYTHON" - "$BLOCK_PROFILE" "$BLOCK_PROFILE_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

profile = {
    "schema": "k3-block-profile-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ["PBS_JOBID"],
    "records_dir": sys.argv[2],
    "expected_world_size": int(os.environ["TP"]) * int(os.environ["PP"]),
    "override_requested": bool(os.environ.get("BLOCKS")),
    "records": [],
}
with open(sys.argv[1], "w") as handle:
    json.dump(profile, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
echo "block_profile=$BLOCK_PROFILE block_profile_dir=$BLOCK_PROFILE_DIR" | tee -a "$LOG_DIR/metadata"

ARGS=(--model "$MODEL_FOR_SERVER" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
    --port "$PORT" --host 0.0.0.0 --enforce-eager --trust-remote-code
    --model-impl vllm --dtype bfloat16 --load-format "$LOAD_FORMAT" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS")
# Data parallelism is how K3 scales past 3 nodes: TP is hard-capped at 32
# because vocab_size 163840 is not divisible by 96, so 8 nodes (96 tiles) must
# run DP=3 x TP=32 rather than TP=96. Each DP replica holds a full copy of the
# weights and serves independently, so aggregate throughput should scale with
# DP while per-user latency stays at the TP=32 value.
if [[ "$DP" != 1 ]]; then
    # --data-parallel-backend ray is required on multi-node: with the default
    # "mp" backend, data_parallel_master_ip stays at its 127.0.0.1 default, so
    # parallel_state builds the torch.distributed TCPStore at tcp://127.0.0.1
    # and every worker off the head node waits on its own loopback forever --
    # no error, all ranks at ~5% CPU, indistinguishable from a hang. The ray
    # backend calls get_ip() instead. Also pass the address explicitly so it
    # does not depend on get_ip() resolving to the routable interface.
    ARGS+=(--data-parallel-size "$DP" --data-parallel-backend ray)
    [[ -n "${HEAD_IP:-}" ]] && ARGS+=(--data-parallel-address "$HEAD_IP")
fi
if [[ "$ASYNC_SCHEDULING" == 0 ]]; then
    ARGS+=(--no-async-scheduling)
elif [[ "$ASYNC_SCHEDULING" == 1 ]]; then
    ARGS+=(--async-scheduling)
fi
if [[ -n "$BLOCKS" ]]; then
    ARGS+=(--num-gpu-blocks-override "$BLOCKS")
fi
# CHUNKED_PREFILL=0 disables chunked prefill. Needed above ~c=128 on K3/XPU:
# once the scheduler splits a prefill, the resumed half has context, which
# takes MLA's chunked-context path into merge_attn_states -- and on XPU the
# prefill backend hands it prefix_lse=None (the FA varlen wrapper asks for the
# softmax LSE but does not get one back), so the merge dies with
# "'NoneType' object has no attribute 'transpose'" and kills the engine.
# Measured on job 8746183: c=384 lost 581 of 768 requests this way.
# Requires max_num_batched_tokens >= max_model_len so a whole prompt fits in
# one scheduling step.
if [[ "${CHUNKED_PREFILL:-1}" == 0 ]]; then
    ARGS+=(--no-enable-chunked-prefill)
fi
if [[ -n "$SAFETENSORS_LOAD_STRATEGY" ]]; then
    ARGS+=(--safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY")
fi
if [[ "$MULTITHREAD_LOAD" == 1 || -n "$MODEL_LOADER_EXTRA_CONFIG" ]]; then
    loader_config=$($PYTHON - "$MODEL_LOADER_EXTRA_CONFIG" "$MULTITHREAD_LOAD" "$LOAD_THREADS" <<'PY'
import json
import sys

raw, multithread, threads = sys.argv[1:]
config = json.loads(raw) if raw else {}
if multithread == "1":
    config.update(enable_multithread_load=True, num_threads=int(threads))
print(json.dumps(config, separators=(",", ":")))
PY
    )
    ARGS+=(--model-loader-extra-config "$loader_config")
fi
if [[ -n "$SERVED_MODEL_NAME" ]]; then
    ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ ${#NODES[@]} -eq 1 ]]; then
    set +u
    module load frameworks
    set -u
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=cxi
    export CCL_KVS_IFACE=${CCL_KVS_IFACE:-lo}
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PYTORCH_ALLOC_CONF=
    export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-$(seq -s, 0 $((TP - 1)))}
    ARGS+=(--distributed-executor-backend mp)
    echo "executor_ready=mp timestamp=$(date -Is)" | tee -a "$LOG_DIR/metadata"
else
    [[ -f "$RAY_ENV" ]] || { echo "ERROR: missing Ray environment helper" >&2; exit 1; }
    NOPROXY_EXTRA="localhost,127.0.0.1"
    for node in "${NODES[@]}"; do
        node_ip=$(resolve_node_ip "$node")
        short_node=${node%%.*}
        NOPROXY_EXTRA="$NOPROXY_EXTRA,$node,$short_node,$short_node.hsn.cm.aurora.alcf.anl.gov,$node_ip"
    done
    export no_proxy="$NOPROXY_EXTRA" NO_PROXY="$NOPROXY_EXTRA" VLLM_HOST_IP="$HEAD_IP" RAY_HEAD_IP="$HEAD_IP"
    set +u
    source "$RAY_ENV" frameworks
    set -u
    # The framework helper may alter Python environment state. Reassert the
    # patched source and its site-packages before starting API/Ray processes.
    export PYTHONPATH="$VLLM_SRC:$PYTHON_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    vllm_source=$($PYTHON -c 'import vllm; print(vllm.__file__)')
    echo "vllm_source=$vllm_source" | tee -a "$LOG_DIR/metadata"
    [[ "$vllm_source" == "$VLLM_SRC/vllm/__init__.py" ]] || {
        echo "ERROR: patched vLLM source is not active: $vllm_source" >&2
        exit 1
    }
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=cxi
    export CCL_KVS_IFACE=${CCL_KVS_IFACE:-hsn0}
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT VLLM_WORKER_MULTIPROC_METHOD=spawn
    export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
    export RAY_DEDUP_LOGS=0
    export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="$RAY_V2"
    export PYTORCH_ALLOC_CONF=
    mkdir -p "$RAY_TEMP_ROOT"
    stop_owned_ray() {
        local ray_root=$1 pid
        mapfile -t owned_pids < <(
            ps -eo pid=,args= | awk -v root="$ray_root" \
                'index($0, root) {print $1}'
        )
        for pid in "${owned_pids[@]}"; do
            [[ "$pid" == "$BASHPID" ]] && continue
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        mapfile -t owned_pids < <(
            ps -eo pid=,args= | awk -v root="$ray_root" \
                'index($0, root) {print $1}'
        )
        for pid in "${owned_pids[@]}"; do
            [[ "$pid" == "$BASHPID" ]] && continue
            kill -KILL "$pid" 2>/dev/null || true
        done
    }
    stop_owned_ray "$RAY_TEMP_ROOT"
    ray_pids=()
    preserve_ray_logs() {
        local node
        mkdir -p "$LOG_DIR/ray_sessions/head"
        cp -a "$RAY_TEMP_ROOT/session_latest/logs" "$LOG_DIR/ray_sessions/head/" 2>/dev/null || true
        for node in "${NODES[@]}"; do
            [[ "$node" == "$HEAD" ]] && continue
            mkdir -p "$LOG_DIR/ray_sessions/${node}"
            ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
                "if [ -d '$RAY_TEMP_ROOT/session_latest/logs' ]; then tar -C '$RAY_TEMP_ROOT/session_latest' -cf - logs; fi" \
                >"$LOG_DIR/ray_sessions/${node}/logs.tar" 2>/dev/null || true
        done
    }
    cleanup_ray_workers() {
        local pid node
        preserve_ray_logs
        for pid in "${ray_pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        for node in "${NODES[@]}"; do
            [[ "$node" == "$HEAD" ]] && continue
            ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
                "self=\$\$; mapfile -t pids < <(ps -eo pid=,args= | awk -v root='$RAY_TEMP_ROOT' -v self=\"\$self\" 'index(\$0, root) && \$1 != self {print \$1}'); for pid in \"\${pids[@]}\"; do kill -TERM \"\$pid\" 2>/dev/null || true; done; sleep 2; mapfile -t pids < <(ps -eo pid=,args= | awk -v root='$RAY_TEMP_ROOT' -v self=\"\$self\" 'index(\$0, root) && \$1 != self {print \$1}'); for pid in \"\${pids[@]}\"; do kill -KILL \"\$pid\" 2>/dev/null || true; done; if [ -f '$K3_CACHE_MARKER' ] && grep -Fxq '$PBS_JOBID' '$K3_CACHE_MARKER'; then rm -rf -- '$K3_CACHE_ROOT'; fi" >/dev/null 2>&1 || true
        done
        stop_owned_ray "$RAY_TEMP_ROOT"
        cleanup_cache
    }
    trap cleanup_ray_workers EXIT
    if ! ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
        --num-gpus="${NUM_GPUS:-12}" --num-cpus=4 --temp-dir="$RAY_TEMP_ROOT" --include-dashboard=false \
        >"$LOG_DIR/ray_head.log" 2>&1; then
        echo "ERROR: Ray head failed to start; see $LOG_DIR/ray_head.log" >&2
        exit 1
    fi
    remote_cache_root_q=$(printf '%q' "$K3_CACHE_ROOT")
    remote_cache_marker_q=$(printf '%q' "$K3_CACHE_MARKER")
    remote_hf_home_q=$(printf '%q' "$HF_HOME")
    remote_hf_modules_cache_q=$(printf '%q' "$HF_MODULES_CACHE")
    remote_hf_hub_cache_q=$(printf '%q' "$HF_HUB_CACHE")
    remote_transformers_cache_q=$(printf '%q' "$TRANSFORMERS_CACHE")
    remote_xdg_cache_home_q=$(printf '%q' "$XDG_CACHE_HOME")
    remote_pythonpath_q=$(printf '%q' "${PYTHONPATH:-}")
    remote_ld_library_path_q=$(printf '%q' "${LD_LIBRARY_PATH:-}")
    remote_no_proxy_q=$(printf '%q' "$NOPROXY_EXTRA")
    remote_torchdynamo_disable_q=$(printf '%q' "$TORCHDYNAMO_DISABLE")
    remote_torch_compile_disable_q=$(printf '%q' "$TORCH_COMPILE_DISABLE")
    remote_ccl_process_launcher_q=$(printf '%q' "$CCL_PROCESS_LAUNCHER")
    remote_ccl_atl_transport_q=$(printf '%q' "$CCL_ATL_TRANSPORT")
    remote_ccl_kvs_iface_q=$(printf '%q' "$CCL_KVS_IFACE")
    remote_fi_provider_q=$(printf '%q' "$FI_PROVIDER")
    remote_ze_flat_device_hierarchy_q=$(printf '%q' "$ZE_FLAT_DEVICE_HIERARCHY")
    remote_vllm_worker_method_q=$(printf '%q' "$VLLM_WORKER_MULTIPROC_METHOD")
    remote_vllm_target_device_q=$(printf '%q' "$VLLM_TARGET_DEVICE")
    remote_ray_no_set_oneapi_q=$(printf '%q' "$RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR")
    remote_ray_dedup_logs_q=$(printf '%q' "$RAY_DEDUP_LOGS")
    remote_ray_v2_q=$(printf '%q' "$RAY_V2")
    remote_kda_xpu_diagnostics_q=$(printf '%q' "${VLLM_KDA_XPU_DIAGNOSTICS:-0}")
    remote_vllm_batch_invariant_q=$(printf '%q' "$VLLM_BATCH_INVARIANT")
    remote_vllm_xpu_deterministic_routing_q=$(printf '%q' "$VLLM_XPU_DETERMINISTIC_ROUTING")
    remote_vllm_xpu_deterministic_moe_gather_q=$(printf '%q' "$VLLM_XPU_DETERMINISTIC_MOE_GATHER")
    remote_kimi_xpu_diagnostics_q=$(printf '%q' "${VLLM_KIMI_XPU_DIAGNOSTICS:-0}")
    remote_kimi_xpu_request_diagnostic_limit_q=$(printf '%q' "${VLLM_KIMI_XPU_REQUEST_DIAGNOSTIC_LIMIT:-4096}")
    remote_kda_vectorized_q=$(printf '%q' "$VLLM_KIMI_XPU_KDA_VECTORIZED")
    remote_conv1d_vectorized_q=$(printf '%q' "$VLLM_KIMI_XPU_CONV1D_VECTORIZED")
    remote_xpu_triton_sampler_q=$(printf '%q' "$VLLM_XPU_ALLOW_TRITON_SAMPLER")
    remote_daos_agent_drpc_q=$(printf '%q' "$DAOS_AGENT_DRPC_DIR")
    remote_d_agent_drpc_q=$(printf '%q' "$D_AGENT_DRPC_DIR")
    remote_ray_temp_root_q=$(printf '%q' "$RAY_TEMP_ROOT")
    remote_k3_block_profile_dir_q=$(printf '%q' "$K3_BLOCK_PROFILE_DIR")
    remote_k3_loader_accounting_dir_q=$(printf '%q' "$K3_LOADER_ACCOUNTING_DIR")
    remote_ray_extra_env_vars_q=$(printf '%q' "$VLLM_RAY_EXTRA_ENV_VARS_TO_COPY")
    for node in "${NODES[@]}"; do
        [[ "$node" == "$HEAD" ]] && continue
        ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" "source '$RAY_ENV' frameworks; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; export K3_CACHE_ROOT=$remote_cache_root_q K3_CACHE_MARKER=$remote_cache_marker_q HF_HOME=$remote_hf_home_q HF_MODULES_CACHE=$remote_hf_modules_cache_q HF_HUB_CACHE=$remote_hf_hub_cache_q TRANSFORMERS_CACHE=$remote_transformers_cache_q XDG_CACHE_HOME=$remote_xdg_cache_home_q PYTHONPATH=$remote_pythonpath_q LD_LIBRARY_PATH=$remote_ld_library_path_q no_proxy=$remote_no_proxy_q NO_PROXY=$remote_no_proxy_q TORCHDYNAMO_DISABLE=$remote_torchdynamo_disable_q TORCH_COMPILE_DISABLE=$remote_torch_compile_disable_q CCL_PROCESS_LAUNCHER=$remote_ccl_process_launcher_q CCL_ATL_TRANSPORT=$remote_ccl_atl_transport_q CCL_KVS_IFACE=$remote_ccl_kvs_iface_q FI_PROVIDER=$remote_fi_provider_q ZE_FLAT_DEVICE_HIERARCHY=$remote_ze_flat_device_hierarchy_q VLLM_WORKER_MULTIPROC_METHOD=$remote_vllm_worker_method_q VLLM_TARGET_DEVICE=$remote_vllm_target_device_q VLLM_BATCH_INVARIANT=$remote_vllm_batch_invariant_q VLLM_XPU_DETERMINISTIC_ROUTING=$remote_vllm_xpu_deterministic_routing_q VLLM_XPU_DETERMINISTIC_MOE_GATHER=$remote_vllm_xpu_deterministic_moe_gather_q VLLM_KIMI_XPU_DIAGNOSTICS=$remote_kimi_xpu_diagnostics_q VLLM_KIMI_XPU_REQUEST_DIAGNOSTIC_LIMIT=$remote_kimi_xpu_request_diagnostic_limit_q VLLM_KIMI_XPU_KDA_VECTORIZED=$remote_kda_vectorized_q VLLM_KIMI_XPU_CONV1D_VECTORIZED=$remote_conv1d_vectorized_q VLLM_XPU_ALLOW_TRITON_SAMPLER=$remote_xpu_triton_sampler_q RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=$remote_ray_no_set_oneapi_q RAY_DEDUP_LOGS=$remote_ray_dedup_logs_q VLLM_USE_RAY_V2_EXECUTOR_BACKEND=$remote_ray_v2_q VLLM_KDA_XPU_DIAGNOSTICS=$remote_kda_xpu_diagnostics_q VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=$remote_ray_extra_env_vars_q DAOS_AGENT_DRPC_DIR=$remote_daos_agent_drpc_q D_AGENT_DRPC_DIR=$remote_d_agent_drpc_q K3_BLOCK_PROFILE_DIR=$remote_k3_block_profile_dir_q K3_LOADER_ACCOUNTING_DIR=$remote_k3_loader_accounting_dir_q; if [ -e $remote_cache_root_q ]; then echo 'ERROR: remote K3 cache already exists' >&2; exit 1; fi; mkdir $remote_cache_root_q; mkdir -p $remote_ray_temp_root_q; printf '%s\\n' '$PBS_JOBID' >$remote_cache_marker_q; self=\$\$; mapfile -t pids < <(ps -eo pid=,args= | awk -v root='$RAY_TEMP_ROOT' -v self=\"\$self\" 'index(\$0, root) && \$1 != self {print \$1}'); for pid in \"\${pids[@]}\"; do kill -TERM \"\$pid\" 2>/dev/null || true; done; sleep 2; mapfile -t pids < <(ps -eo pid=,args= | awk -v root='$RAY_TEMP_ROOT' -v self=\"\$self\" 'index(\$0, root) && \$1 != self {print \$1}'); for pid in \"\${pids[@]}\"; do kill -KILL \"\$pid\" 2>/dev/null || true; done; ray start --address='$RAY_ADDRESS' --num-gpus='${NUM_GPUS:-12}' --num-cpus=4 --temp-dir=$remote_ray_temp_root_q --block" \
        >"$LOG_DIR/ray_${node}.log" 2>&1 &
        ray_pids+=("$!")
    done
    sleep 10
    for pid in "${ray_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || {
            echo "ERROR: remote Ray worker exited during startup; see $LOG_DIR" >&2
            exit 1
        }
    done
    echo "executor_ready=ray timestamp=$(date -Is)" | tee -a "$LOG_DIR/metadata"
    ARGS+=(--distributed-executor-backend ray)
    # In-actor capture: the ssh block above hand-forwards ~25 vars to the
    # remote shell that runs `ray start`, which is NOT the same process as
    # a Ray actor (Ray's own env/runtime_env injection happens per-actor).
    # Capture from inside a real actor so a forwarding gap in the ssh block
    # (see the TORCHDYNAMO_DISABLE miss fixed above) is caught here instead
    # of silently producing an unattributed numerics/hang difference.
    "$PYTHON" - "$RAY_ADDRESS" "$LOG_DIR/effective_environment_actor.txt" <<'PY' || true
import sys

import ray

address, out_path = sys.argv[1], sys.argv[2]
ray.init(address=address, ignore_reinit_error=True)


@ray.remote(num_gpus=1)
def _dump_env():
    import os

    keys = (
        "CCL_", "FI_PROVIDER", "ZE_", "VLLM_", "TORCH", "PYTORCH_",
        "HF_", "TRANSFORMERS_", "XDG_", "RAY_", "PYTHONPATH",
        "LD_LIBRARY_PATH", "PBS_JOBID", "K3_BLOCK_PROFILE",
    )
    skip = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(keys) and k not in skip
    }


env = ray.get(_dump_env.remote())
with open(out_path, "w") as fh:
    for key, value in env.items():
        fh.write(f"{key}={value}\n")
ray.shutdown()
PY
    echo "effective_environment_actor=$LOG_DIR/effective_environment_actor.txt" | tee -a "$LOG_DIR/metadata"
fi
env | sort | grep -E '^(CCL_|FI_PROVIDER|ZE_|VLLM_|TORCH|PYTORCH_|HF_|TRANSFORMERS_|XDG_|RAY_|PYTHONPATH|LD_LIBRARY_PATH|PBS_JOBID|K3_BLOCK_PROFILE)' | grep -vE '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=' >"$LOG_DIR/effective_environment.txt" || true
echo "effective_environment=$LOG_DIR/effective_environment.txt" | tee -a "$LOG_DIR/metadata"
for required_var in ZE_FLAT_DEVICE_HIERARCHY:FLAT TORCHDYNAMO_DISABLE:1 FI_PROVIDER:cxi; do
    required_name=${required_var%%:*}
    required_value=${required_var#*:}
    actual_value=$(awk -F= -v k="$required_name" '$1==k{print $2; f=1} END{if(!f) print "UNSET"}' "$LOG_DIR/effective_environment.txt")
    [[ "$actual_value" == "$required_value" ]] || {
        echo "ERROR: post-export local capture shows $required_name=$actual_value, expected $required_value" >&2
        exit 1
    }
done
if [[ -f "$LOG_DIR/effective_environment_actor.txt" ]]; then
    for required_var in ZE_FLAT_DEVICE_HIERARCHY:FLAT TORCHDYNAMO_DISABLE:1 FI_PROVIDER:cxi; do
        required_name=${required_var%%:*}
        required_value=${required_var#*:}
        actual_value=$(awk -F= -v k="$required_name" '$1==k{print $2; f=1} END{if(!f) print "UNSET"}' "$LOG_DIR/effective_environment_actor.txt")
        [[ "$actual_value" == "$required_value" ]] || {
            echo "ERROR: in-actor capture shows $required_name=$actual_value, expected $required_value" >&2
            exit 1
        }
    done
fi
[[ "$EP" == 1 ]] && ARGS+=(--enable-expert-parallel)
echo "server_args=${ARGS[*]}" | tee -a "$LOG_DIR/metadata"
echo "server_start=$(date -Is)" | tee -a "$LOG_DIR/metadata"
if "$PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" \
    > >(tee "$LOG_DIR/server.log") 2>&1; then
    exit 0
else
    server_rc=$?
    exit "$server_rc"
fi

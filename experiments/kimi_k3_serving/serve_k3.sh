#!/usr/bin/env bash
set -euo pipefail

# Start a text-only vLLM OpenAI server on the current PBS allocation.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAY_ENV=${RAY_ENV:-/lus/flare/projects/ModCon/ngetty/torchtune/experiments/ray_smoke/setup_ray_env.sh}
MODEL=${MODEL:-}
TP=${TP:-32}
PP=${PP:-1}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
BLOCKS=${BLOCKS:-}
LOAD_FORMAT=${LOAD_FORMAT:-auto}
SAFETENSORS_LOAD_STRATEGY=${SAFETENSORS_LOAD_STRATEGY:-}
MULTITHREAD_LOAD=${MULTITHREAD_LOAD:-0}
LOAD_THREADS=${LOAD_THREADS:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-}
EP=${EP:-0}
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
K3_CACHE_ROOT=${K3_CACHE_ROOT:-/tmp/k3_hf_cache_${K3_JOB_ID//[^A-Za-z0-9_.-]/_}}
K3_CACHE_MARKER="$K3_CACHE_ROOT/.created_by_${K3_JOB_ID//[^A-Za-z0-9_.-]/_}"

usage() { echo "Usage: $0 --model PATH --blocks N [--tp N] [--safetensors-load-strategy STRATEGY] [--multithread-load [N]] [--stage-model] [--stage-only] [--ep]"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --tp) TP=$2; shift 2 ;;
        --pp) PP=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS=$2; shift 2 ;;
        --max-num-batched-tokens) MAX_BATCHED_TOKENS=$2; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL=$2; shift 2 ;;
        --blocks) BLOCKS=$2; shift 2 ;;
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
        --served-model-name) SERVED_MODEL_NAME=$2; shift 2 ;;
        --python) PYTHON=$2; shift 2 ;;
        --stage-model) STAGE_MODEL=1; shift ;;
        --stage-only) STAGE_MODEL=1; STAGE_ONLY=1; shift ;;
        --ep) EP=1; shift ;;
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
if [[ "$STAGE_ONLY" != 1 ]]; then
    [[ -n "$BLOCKS" ]] || { echo "ERROR: --blocks is required" >&2; exit 2; }
fi
if [[ "$EP" == 1 ]]; then
    [[ -f "$MODEL/config.json" ]] || { echo "ERROR: EP requires a model config: $MODEL/config.json" >&2; exit 2; }
    if ! grep -Eq '"(num_experts|num_local_experts)"[[:space:]]*:[[:space:]]*[1-9][0-9]*' "$MODEL/config.json"; then
        echo "ERROR: --ep requires a MoE model with num_experts > 0: $MODEL" >&2
        exit 2
    fi
fi
if command -v qstat >/dev/null; then
    qstat_output=$(qstat -f "$PBS_JOBID" 2>/dev/null) || {
        echo "ERROR: unable to query PBS job $PBS_JOBID" >&2
        exit 1
    }
    job_state=$(awk -F'= ' '/job_state/ {print $2; exit}' <<<"$qstat_output")
    [[ "$job_state" == R ]] || { echo "ERROR: allocation $PBS_JOBID is not running (state=${job_state:-unknown})" >&2; exit 1; }
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
mapfile -t NODES < <(sort -u "$PBS_NODEFILE")
[[ ${#NODES[@]} -gt 0 ]] || { echo "ERROR: PBS_NODEFILE has no nodes" >&2; exit 1; }
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
echo "model=$MODEL model_for_server=$MODEL_FOR_SERVER tp=$TP pp=$PP nodes=${NODES[*]}" | tee "$LOG_DIR/metadata"
echo "node=$(hostname) start=$(date -Is)" | tee -a "$LOG_DIR/metadata"
echo "k3_cache_root=$K3_CACHE_ROOT hf_home=$HF_HOME hf_modules_cache=$HF_MODULES_CACHE hf_hub_cache=$HF_HUB_CACHE transformers_cache=$TRANSFORMERS_CACHE xdg_cache_home=$XDG_CACHE_HOME" | tee -a "$LOG_DIR/metadata"
echo "load_format=$LOAD_FORMAT safetensors_load_strategy=${SAFETENSORS_LOAD_STRATEGY:-default} multithread_load=$MULTITHREAD_LOAD load_threads=$LOAD_THREADS" | tee -a "$LOG_DIR/metadata"

ARGS=(--model "$MODEL_FOR_SERVER" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
    --port "$PORT" --host 0.0.0.0 --enforce-eager --trust-remote-code
    --model-impl vllm --dtype bfloat16 --load-format "$LOAD_FORMAT" --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" --num-gpu-blocks-override "$BLOCKS")
if [[ -n "$SAFETENSORS_LOAD_STRATEGY" ]]; then
    ARGS+=(--safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY")
fi
if [[ "$MULTITHREAD_LOAD" == 1 ]]; then
    ARGS+=(--model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":$LOAD_THREADS}")
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
    export PYTORCH_ALLOC_CONF=
    ray stop --force >/dev/null 2>&1 || true
    ray_pids=()
    cleanup_ray_workers() {
        local pid node
        for pid in "${ray_pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        for node in "${NODES[@]}"; do
            [[ "$node" == "$HEAD" ]] && continue
            ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
                "ray stop --force >/dev/null 2>&1 || true; if [ -f '$K3_CACHE_MARKER' ] && grep -Fxq '$PBS_JOBID' '$K3_CACHE_MARKER'; then rm -rf -- '$K3_CACHE_ROOT'; fi" >/dev/null 2>&1 || true
        done
        ray stop --force >/dev/null 2>&1 || true
        cleanup_cache
    }
    trap cleanup_ray_workers EXIT
    if ! ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
        --num-gpus="${NUM_GPUS:-12}" --num-cpus=4 --temp-dir=/tmp --include-dashboard=false \
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
    remote_ccl_process_launcher_q=$(printf '%q' "$CCL_PROCESS_LAUNCHER")
    remote_ccl_atl_transport_q=$(printf '%q' "$CCL_ATL_TRANSPORT")
    remote_ccl_kvs_iface_q=$(printf '%q' "$CCL_KVS_IFACE")
    remote_fi_provider_q=$(printf '%q' "$FI_PROVIDER")
    remote_ze_flat_device_hierarchy_q=$(printf '%q' "$ZE_FLAT_DEVICE_HIERARCHY")
    remote_vllm_worker_method_q=$(printf '%q' "$VLLM_WORKER_MULTIPROC_METHOD")
    remote_vllm_target_device_q=$(printf '%q' "$VLLM_TARGET_DEVICE")
    for node in "${NODES[@]}"; do
        [[ "$node" == "$HEAD" ]] && continue
        ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" "source '$RAY_ENV' frameworks; export K3_CACHE_ROOT=$remote_cache_root_q K3_CACHE_MARKER=$remote_cache_marker_q HF_HOME=$remote_hf_home_q HF_MODULES_CACHE=$remote_hf_modules_cache_q HF_HUB_CACHE=$remote_hf_hub_cache_q TRANSFORMERS_CACHE=$remote_transformers_cache_q XDG_CACHE_HOME=$remote_xdg_cache_home_q PYTHONPATH=$remote_pythonpath_q LD_LIBRARY_PATH=$remote_ld_library_path_q no_proxy=$remote_no_proxy_q NO_PROXY=$remote_no_proxy_q CCL_PROCESS_LAUNCHER=$remote_ccl_process_launcher_q CCL_ATL_TRANSPORT=$remote_ccl_atl_transport_q CCL_KVS_IFACE=$remote_ccl_kvs_iface_q FI_PROVIDER=$remote_fi_provider_q ZE_FLAT_DEVICE_HIERARCHY=$remote_ze_flat_device_hierarchy_q VLLM_WORKER_MULTIPROC_METHOD=$remote_vllm_worker_method_q VLLM_TARGET_DEVICE=$remote_vllm_target_device_q; if [ -e $remote_cache_root_q ]; then echo 'ERROR: remote K3 cache already exists' >&2; exit 1; fi; mkdir $remote_cache_root_q; printf '%s\\n' '$PBS_JOBID' >$remote_cache_marker_q; ray stop --force >/dev/null 2>&1 || true; ray start --address='$RAY_ADDRESS' --num-gpus='${NUM_GPUS:-12}' --num-cpus=4 --temp-dir=/tmp --block" \
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

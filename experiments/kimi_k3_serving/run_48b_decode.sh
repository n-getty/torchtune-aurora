#!/usr/bin/env bash
set -euo pipefail

# Re-runnable Kimi-Linear-48B decode probe, designed to be invoked repeatedly
# inside ONE held allocation (see hold_1node_debug.sh). Each invocation gets
# its own attempt directory, so a failed run never clobbers earlier evidence
# and no requeue is needed between code fixes.
#
# Usage (from the login node, against a running hold):
#   ssh <node> bash /lus/.../run_48b_decode.sh <JOB_ID> [ATTEMPT_TAG]
#
# Absolute paths throughout: Aurora's login .bashrc auto-cd silently no-ops for
# non-interactive SSH, so relative paths are unsafe here.

JOB_ID=${1:?Usage: $0 JOB_ID [ATTEMPT_TAG]}
ATTEMPT_TAG=${2:-$(date +%H%M%S)}

REPO=/lus/flare/projects/ModCon/ngetty/torchtune
EXP=$REPO/experiments/kimi_k3_serving
PYTHON=/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python
MODEL=/flare/ModCon/ngetty/models/Kimi-Linear-48B-A3B-Instruct
SERVED_NAME=kimi-linear
PORT=${PORT:-8000}
TP=${TP:-2}

RUN_DIR=$EXP/logs/kimi_decode_repro_${JOB_ID%%.*}/attempt_${ATTEMPT_TAG}
SERVER_DIR=$RUN_DIR/server
PROBE_DIR=$RUN_DIR/probe
mkdir -p "$SERVER_DIR" "$PROBE_DIR"
exec > >(tee "$RUN_DIR/run.log") 2>&1

echo "phase=start job=$JOB_ID attempt=$ATTEMPT_TAG node=$(hostname -s) time=$(date -Is)"
[[ -d "$MODEL" ]] || { echo "ERROR: model missing: $MODEL" >&2; exit 1; }

# Free the port from any previous attempt in this same hold, but only kill
# processes we own that are bound to it -- never a broad pkill.
stale=$(ss -lptn "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true)
if [[ -n "$stale" ]]; then
    echo "phase=reset stale_pids=$stale"
    for pid in $stale; do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 5
    for pid in $stale; do kill -KILL "$pid" 2>/dev/null || true; done
    sleep 2
fi

SERVER_PID=""
cleanup() {
    local rc=$?
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "phase=cleanup stopping server pid=$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    echo "phase=exit rc=$rc time=$(date -Is)"
    # The allocation intentionally survives this script so the next attempt
    # can start immediately.
}
trap cleanup EXIT

echo "phase=server_start time=$(date -Is)"
LOG_DIR="$SERVER_DIR" \
K3_JOB_ID="$JOB_ID" \
MODEL_LOADER_EXTRA_CONFIG='' \
"$EXP/serve_k3.sh" \
    --model "$MODEL" \
    --tp "$TP" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.90 \
    &
SERVER_PID=$!

echo "phase=wait_health time=$(date -Is)"
HEALTHY=0
for attempt in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server exited before becoming healthy" >&2
        echo "--- root cause (first error in server.log) ---" >&2
        grep -nE "Error|Exception|TypeError|RuntimeError|AssertionError" \
            "$SERVER_DIR/server.log" 2>/dev/null \
            | grep -v "wait_for_ready\|raise e from None" | head -5 >&2 || true
        exit 1
    fi
    if curl --noproxy '*' --fail --silent --max-time 10 \
        "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        HEALTHY=1
        echo "phase=health status=200 attempt=$attempt time=$(date -Is)"
        break
    fi
    sleep 10
done
[[ "$HEALTHY" == 1 ]] || { echo "ERROR: no health within 20 minutes" >&2; exit 1; }

echo "phase=decode_probe time=$(date -Is)"
"$PYTHON" "$EXP/kimi_decode_probe.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$SERVED_NAME" \
    --output-dir "$PROBE_DIR" \
    --repeats 3 \
    --lengths 1,2,4,8,16,32,128

echo "phase=done time=$(date -Is) artifacts=$PROBE_DIR"

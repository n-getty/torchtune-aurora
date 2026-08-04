#!/usr/bin/env bash
set -euo pipefail

# Gate 0: one-node server health plus a small OpenAI benchmark.
# Run only inside a fresh one-node PBS allocation.

ROOT=/lus/flare/projects/ModCon/ngetty/torchtune
SERVER="$ROOT/experiments/kimi_k3_serving/serve_k3.sh"
MODEL=${MODEL:-/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B}
TP=${TP:-4}
BLOCKS=${BLOCKS:-512}
PORT=${PORT:-8000}
LOG_DIR=${LOG_DIR:-$ROOT/experiments/kimi_k3_serving/logs/gate0_$(date +%Y%m%d_%H%M%S)}
STAGE_ROOT=${STAGE_ROOT:-/tmp/kimi_k3_models}

[[ -n "${PBS_JOBID:-}" ]] || { echo "ERROR: PBS_JOBID is required" >&2; exit 1; }
[[ -f "$SERVER" ]] || { echo "ERROR: server script missing: $SERVER" >&2; exit 1; }
mkdir -p "$LOG_DIR"
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
set +u
module load frameworks
set -u
SERVER_LOG="$LOG_DIR/server.log"
BENCH_LOG="$LOG_DIR/bench.log"
SERVED_MODEL="$STAGE_ROOT/$(basename "$MODEL")"

mapfile -t allocation_nodes < <(sort -u "$PBS_NODEFILE")
if [[ ${#allocation_nodes[@]} -eq 0 ]]; then
    allocation_nodes=("$(hostname -s)")
fi
echo "job=$PBS_JOBID nodes=${allocation_nodes[*]}" | tee "$LOG_DIR/metadata"
MODEL="$MODEL" STAGE_MODEL=1 TP="$TP" BLOCKS="$BLOCKS" PORT="$PORT" LOG_DIR="$LOG_DIR/server" \
    SERVED_MODEL_NAME="$(basename "$MODEL")" \
    bash "$SERVER" >"$SERVER_LOG" 2>&1 &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
    if curl --noproxy "*" --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "server_ready=$(date -Is)" | tee -a "$LOG_DIR/metadata"
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: server exited before health" >&2
        tail -80 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 5
done
curl --noproxy "*" --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "ERROR: server did not become healthy" >&2
    tail -80 "$SERVER_LOG" >&2
    exit 1
}

vllm bench serve --backend openai --base-url "http://127.0.0.1:$PORT" \
    --model "$SERVED_MODEL" --tokenizer "$SERVED_MODEL" --num-prompts 8 --dataset-name random \
    --random-input-len 128 --random-output-len 64 --request-rate inf \
    --max-concurrency 2 >"$BENCH_LOG" 2>&1
cat "$BENCH_LOG"
echo "GATE0 PASS logs=$LOG_DIR"

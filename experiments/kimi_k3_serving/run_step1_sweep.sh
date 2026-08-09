#!/usr/bin/env bash
set -euo pipefail

# Step 1 of the throughput plan: re-measure the CURRENT build with the SAME
# harness that produced Qwen3-Coder-480B's 106.19 output tok/s on this machine.
#
# The 1.75 tok/s and 6.489 tok/s figures came from throughput_probe.py, which
# divides completion tokens by wall-clock around the whole ThreadPoolExecutor
# block -- so prefill, queueing and HTTP round-trip are all inside the
# denominator, and the result is bounded by the slowest request. At 16 output
# tokens that fixed overhead dominates completely (c=1 spends 16.59s emitting
# 16 tokens). The 480B number came from `vllm bench serve` at 512 output tokens
# and 128 prompts. sweep_topology.sh (which wraps `vllm bench serve`) has never
# once been run against K3 -- logs/ contains qwen30_sweep_* only.
#
# So this script changes no model code. It only measures the same thing the
# same way, which is a precondition for any statement about a "60x gap".
#
# Run this against an ALREADY-RUNNING server, from the head compute node.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_BASE=${RUN_BASE:?Set RUN_BASE to the run directory (holds server/ and nodefile)}
SERVER_LOG=${SERVER_LOG:-$RUN_BASE/server/server.log}
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
VENV_BIN=$(dirname "$PYTHON")

export PATH="$VENV_BIN:$PATH"
export PYTHONPATH="${VLLM_SRC:-/flare/ModCon/ngetty/vllm-xpu-src}${PYTHONPATH:+:$PYTHONPATH}"
export PBS_NODEFILE=${PBS_NODEFILE:-$RUN_BASE/nodefile}
export PBS_JOBID=${PBS_JOBID:-${K3_JOB_ID:?Set K3_JOB_ID or PBS_JOBID}}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# Match the running server. These are asserted by sweep_topology.sh into the
# results row, so a mismatch here silently mislabels the whole run.
# K3's tokenizer is custom code (tokenization_kimi.py); the bench client loads
# it in its own process, so it needs its own trust flag.
export TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
export MODEL=${MODEL:-/tmp/ngetty/AuroraGPT/prism_models}
export SERVED_MODEL=${SERVED_MODEL:-Kimi-K3}
export TOKENIZER=${TOKENIZER:-$MODEL}
export SERVER_TP=${SERVER_TP:-32}
export SERVER_PP=${SERVER_PP:-1}
export SERVER_EP=${SERVER_EP:-1}
export SERVER_MAX_NUM_SEQS=${SERVER_MAX_NUM_SEQS:-32}
export SERVER_MAX_BATCHED_TOKENS=${SERVER_MAX_BATCHED_TOKENS:-2048}
export SERVER_GPU_MEM_UTIL=${SERVER_GPU_MEM_UTIL:-0.92}
export SERVER_MAX_MODEL_LEN=${SERVER_MAX_MODEL_LEN:-2048}
export BLOCK_PROFILE=${BLOCK_PROFILE:-$RUN_BASE/server/block_profile.json}
export DAOS_PATH=${DAOS_PATH:-$MODEL}
export SERVER_LOG

# INPUT_LEN+OUTPUT_LEN must fit inside the server's max_model_len (2048) or
# every request 400s. The 480B cell was 1024+512 under a larger max_model_len;
# 1024+512=1536 fits here, so the harness shape is preserved exactly.
export INPUT_LEN=${INPUT_LEN:-1024}
export OUTPUT_LEN=${OUTPUT_LEN:-512}
export PROMPTS=${PROMPTS:-128}
export CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 4 16 32"}
# REPEATS=1 for the first pass: a single 128-prompt x 512-token cell is already
# minutes long on this model, and sweep_topology.sh's 5%-spread check across
# repeats is a stability gate we cannot afford before knowing the scale of the
# number. Raise it once a cell's cost is known.
export REPEATS=${REPEATS:-1}
export OUT=${OUT:-$RUN_BASE/sweep_results.tsv}
export RUN_DIR=${RUN_DIR:-$RUN_BASE/sweep}

echo "waiting for server health at ${SERVER_URL:=http://127.0.0.1:8000}"
for attempt in $(seq 1 240); do
    if curl --noproxy '*' --fail --silent "$SERVER_URL/health" >/dev/null 2>&1; then
        echo "server healthy after ${attempt} checks"
        break
    fi
    if [[ "$attempt" == 240 ]]; then
        echo "ERROR: server never became healthy" >&2
        exit 1
    fi
    sleep 10
done
export SERVER_URL

# Gate the pre-existing state before attributing anything to the sweep.
if [[ -x "$SCRIPT_DIR/check_k3_serving_health.sh" ]]; then
    "$SCRIPT_DIR/check_k3_serving_health.sh" "$SERVER_LOG" \
        >"$RUN_BASE/health_before_sweep.log" 2>&1 \
        && echo "pre-sweep health: GREEN" \
        || { echo "pre-sweep health: DEGRADED (see $RUN_BASE/health_before_sweep.log)" >&2; exit 1; }
fi

exec "$SCRIPT_DIR/sweep_topology.sh"

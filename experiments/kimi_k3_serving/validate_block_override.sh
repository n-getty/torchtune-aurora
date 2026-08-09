#!/usr/bin/env bash
set -uo pipefail

# Validate a --num-gpu-blocks-override K3 server: correctness FIRST, then scale.
#
# The override exists because the KV pool is sized from a transient profiling
# reading: 11 of 32 ranks measure ~0.34 GiB free during profile_run (while
# their even sibling tile on the same physical card holds a peak allocation),
# vLLM clamps every rank to that minimum, and the pool ends up 25x too small.
# Once loaded, every tile actually has ~13.8 GiB free -- so the override is
# claiming memory that is genuinely there.
#
# But "genuinely there" is an inference, and a KV pool that is too large fails
# LATE and QUIETLY: it will not refuse to start, it will corrupt long-context
# output or take an L0 fault under load. So this script never reports a
# throughput number before a correctness gate has passed on the same server.
#
#   phase 1  greedy fixed prompts   -- is the model still coherent at all?
#   phase 2  concurrency ladder     -- does the extra pool convert to throughput?
#
# A failure in phase 1 means the override value is wrong (or the memory is not
# actually free); do not interpret phase 2 numbers from such a run.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_BASE=${RUN_BASE:?Set RUN_BASE}
URL=${URL:-http://127.0.0.1:8000}
MODEL=${MODEL:-Kimi-K3}
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
SERVER_LOG=${SERVER_LOG:-$RUN_BASE/server/server.log}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"

echo "=== phase 0: server health ==="
curl --noproxy '*' --fail --silent "$URL/health" >/dev/null || {
    echo "ERROR: server unhealthy" >&2
    exit 1
}
echo "health OK"

echo
echo "=== phase 1: greedy correctness (must pass before any throughput claim) ==="
"$PYTHON" "$SCRIPT_DIR/probe_batched_nan_logprobs.py" \
    --url "$URL" --model "$MODEL" --max-tokens 24 \
    --out "$RUN_BASE/override_correctness.json"
correctness=$?
if [[ $correctness -ne 0 ]]; then
    echo "CORRECTNESS FAILED -- the override value is not safe. Not measuring throughput." >&2
    exit 1
fi

echo
echo "=== phase 2: concurrency ladder ==="
# Short prompts on purpose: the vectorized KDA PREFILL path is gated on
# sequence_count >= 4, so a long prompt at low concurrency silently drops to
# the scalar per-token loop and the cell never finishes. Short prompts keep
# every rung of the ladder comparable and bounded.
"$PYTHON" "$SCRIPT_DIR/throughput_probe.py" \
    --base-url "$URL" --model "$MODEL" --max-tokens 16 --timeout 900 \
    --output "$RUN_BASE/override_throughput.json"
ladder=$?

if [[ -s "$RUN_BASE/override_throughput.json" ]]; then
    "$PYTHON" - "$RUN_BASE/override_throughput.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
print(f"\n{'conc':>5} {'done':>5} {'tokens':>7} {'elapsed s':>10} {'tok/s':>9}")
for row in rows:
    print(f"{row['concurrency']:>5} {row['completed']:>5} "
          f"{row['completion_tokens']:>7} {row['elapsed']:>10.1f} "
          f"{row['aggregate_tok_s']:>9.3f}")
baseline = 6.489  # best pre-override number: c=8, job 8745694, same harness
best = max(row["aggregate_tok_s"] for row in rows)
print(f"\nbest={best:.3f} tok/s vs pre-override best {baseline} tok/s "
      f"({best / baseline:.2f}x)")
PY
fi

echo
echo "=== phase 3: post-run server health (a pool that is too large fails late) ==="
if [[ -x "$SCRIPT_DIR/check_k3_serving_health.sh" && -f "$SERVER_LOG" ]]; then
    "$SCRIPT_DIR/check_k3_serving_health.sh" "$SERVER_LOG" \
        >"$RUN_BASE/health_after_override.log" 2>&1 \
        && echo "post-run health: GREEN" \
        || { echo "post-run health: DEGRADED (see health_after_override.log)"; exit 1; }
fi
exit $ladder

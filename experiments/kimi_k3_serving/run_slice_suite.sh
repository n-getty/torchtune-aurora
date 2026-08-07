#!/usr/bin/env bash
# Unattended prefill-vs-decode suite against the real-weight K3 slice.
#
# Runs the whole battery for one topology with no operator in the loop:
#   serve -> wait for health -> single-step probe -> decode-depth ladder ->
#   long-decode degeneration check -> tear down -> write a verdict file.
#
# Designed for overnight use, so it fails LOUDLY into files rather than
# silently: every stage writes to $OUT, and VERDICT.txt always gets written
# even when a stage dies, so an empty result is distinguishable from a crash.
#
# Usage:
#   run_slice_suite.sh <tag> <tp> [extra vllm args...]
# e.g.
#   run_slice_suite.sh tp8ep 8 --enable-expert-parallel
#
# Notes:
#  * curl MUST use --noproxy '*': Aurora's no_proxy lists localhost but NOT
#    127.0.0.1, so a local request otherwise returns an HTML 503 from the ALCF
#    proxy and looks exactly like a dead server.
#  * TP>1 on one node needs the interactive row of the launcher table
#    (ofi/none, no ZE_AFFINITY_MASK) -- CCL must see every device UUID.

# NOTE: NOT `set -u`. Lmod's `module` shell function dereferences unbound
# variables internally, so under `set -u` the script dies SILENTLY at
# `module load frameworks` -- no error, just no further output. That is
# what produced 7 suite directories containing only a header line.
set -o pipefail

TAG="${1:?usage: run_slice_suite.sh <tag> <tp> [extra args...]}"
TP="${2:?usage: run_slice_suite.sh <tag> <tp> [extra args...]}"
shift 2
EXTRA=("$@")

ROOT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
SLICE=/lus/flare/projects/ModCon/ngetty/k3-slice-4L8E
OUT="$ROOT/logs/suite_${TAG}_$(date +%Y%m%d_%H%M%S)"
PORT=$((8100 + TP))
mkdir -p "$OUT"

exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== suite tag=$TAG tp=$TP extra='${EXTRA[*]:-}' node=$(hostname) $(date -Is) ==="

module load frameworks >/dev/null 2>&1
source /flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/activate 2>/dev/null
export PYTHONPATH=/flare/ModCon/ngetty/vllm-xpu-src:${PYTHONPATH:-}
export ZE_FLAT_DEVICE_HIERARCHY=FLAT VLLM_TARGET_DEVICE=xpu
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_XPU_DETERMINISTIC_ROUTING=1
# VLLM_XPU_DETERMINISTIC_MOE_GATHER defaults OFF here, deliberately.
#
# =1 replaces the fused moe_gather with a Python double loop over
# (num_tokens x topk) where every iteration does 2-3 `.item()` XPU device
# syncs (vllm_xpu_kernels/fused_moe_interface.py:299-308). Cost scales with
# generated length, per MoE layer, per step. At TP=1 the depth ladder's first
# 33-token request wedged EngineCore for 10+ minutes at 100% CPU -- py-spy
# pinned it to that exact line -- while /health kept returning 200 OK.
#
# The probes here compare prefill against decode on the SAME server, so both
# sides see the same gather either way; determinism is not what this
# experiment needs, and the runtime cost destroys the long-generation stages.
# Override to 1 explicitly if reproducing a production-flag configuration.
export VLLM_XPU_DETERMINISTIC_MOE_GATHER=${VLLM_XPU_DETERMINISTIC_MOE_GATHER:-0}
[ "$TP" = "1" ] && export ZE_AFFINITY_MASK=0

verdict() { echo "$1" >> "$OUT/VERDICT.txt"; }

# Kill ONLY the server this script started, by PID.
#
# The first version used `pkill -f "[a]pi_server"`. That pattern matches this
# script's own command line (which contains the string when invoked over ssh),
# so the cleanup killed the caller: every suite died ~20s in, before the server
# even launched, and the whole overnight sweep self-terminated in 3 minutes
# having produced nothing. The [a] bracket trick only protects against matching
# the *pkill* process itself -- it does nothing about other processes whose
# arguments happen to contain the pattern.
#
# PID-scoped teardown cannot make that mistake. SERVER is set after launch;
# guard on it being non-empty so an early failure does not `kill ""`.
SERVER=""
cleanup() {
  if [ -n "$SERVER" ]; then
    kill "$SERVER" 2>/dev/null
    for _ in $(seq 1 10); do kill -0 "$SERVER" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER" 2>/dev/null
  fi
  sleep 3
}
trap cleanup EXIT

python3 -m vllm.entrypoints.openai.api_server \
  --model "$SLICE" --served-model-name k3slice \
  --tensor-parallel-size "$TP" "${EXTRA[@]}" \
  --port "$PORT" --host 127.0.0.1 \
  --enforce-eager --trust-remote-code --model-impl vllm --dtype bfloat16 \
  --max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --no-enable-prefix-caching --num-gpu-blocks-override 8 \
  --gpu-memory-utilization 0.70 > "$OUT/server.log" 2>&1 &
SERVER=$!

echo "waiting for health on :$PORT (server pid $SERVER)"
for _ in $(seq 1 180); do
  curl -sf -m 5 --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER" 2>/dev/null || { echo "SERVER DIED during startup"; break; }
  sleep 10
done

if ! curl -sf -m 5 --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "FATAL: server never became healthy"
  grep -iE "error|assert|Traceback" "$OUT/server.log" | grep -viE "INFO|Route:" | tail -15
  verdict "tp=$TP SERVER_FAILED"
  exit 1
fi
echo "server healthy"
verdict "tp=$TP SERVER_OK"

# NOTE: `cmd | tail` makes $? the exit of TAIL, not of cmd -- which would
# record success for every crashed probe. Use PIPESTATUS[0].
echo; echo "--- single-step prefill-vs-decode ---"
timeout 900 python3 "$ROOT/probe_prefill_vs_decode.py" \
  --base-url "http://127.0.0.1:$PORT" --model k3slice \
  --output-dir "$OUT/single_step" 2>&1 | tail -40
verdict "tp=$TP single_step exit=${PIPESTATUS[0]}"

echo; echo "--- decode-depth ladder ---"
timeout 1800 python3 "$ROOT/probe_depth_ladder.py" \
  "http://127.0.0.1:$PORT" k3slice "$OUT/ladder" 2>&1 | tail -20
verdict "tp=$TP ladder exit=${PIPESTATUS[0]}"

echo; echo "--- long decode (degeneration check, 128 tokens x3) ---"
# The slice has random-ish truncated weights so the TEXT is meaningless; what
# matters is whether it collapses to a repeated token the way real K3 does.
timeout 900 python3 - "$PORT" "$OUT" <<'PY' 2>&1 | tail -20
import json, sys, urllib.request, collections
port, out = sys.argv[1], sys.argv[2]
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(p):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(p).encode(), headers={"Content-Type": "application/json"},
        method="POST")
    return json.loads(OP.open(r, timeout=600).read())
runs = []
for i in range(3):
    d = post({"model": "k3slice", "prompt": "The capital of France is",
              "max_tokens": 128, "temperature": 0, "logprobs": 5,
              "return_tokens_as_token_ids": True})
    t = [x.split(":")[-1] for x in d["choices"][0]["logprobs"]["tokens"]]
    c = collections.Counter(t)
    top_share = max(c.values()) / len(t)
    print(f"  run {i}: {len(t)} tokens, {len(c)} distinct, top_share={top_share:.2f}")
    runs.append(t)
print(f"  repeats identical: {runs[0] == runs[1] == runs[2]}")
json.dump({"runs": runs}, open(f"{out}/long_decode.json", "w"), indent=2)
PY
verdict "tp=$TP long_decode exit=${PIPESTATUS[0]}"

echo; echo "=== suite done $(date -Is) ==="
cat "$OUT/VERDICT.txt"

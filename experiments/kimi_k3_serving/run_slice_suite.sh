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
# SLICE_OVERRIDE lets a control job point at a different slice (e.g. the
# 32-expert one) so cross-node vs single-node can be compared like-for-like.
SLICE=${SLICE_OVERRIDE:-/lus/flare/projects/ModCon/ngetty/k3-slice-4L8E}
OUT="$ROOT/logs/suite_${TAG}_$(date +%Y%m%d_%H%M%S)"
PORT=$((8100 + TP))
mkdir -p "$OUT"

exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== suite tag=$TAG tp=$TP extra='${EXTRA[*]:-}' node=$(hostname) $(date -Is) ==="
# Log the slice. Comparing a 32-expert cross-node run against 8-expert
# single-node runs produced a 5x maxdiff difference that could not be
# attributed without knowing this, so record it in every run.
echo "slice=$SLICE"

module load frameworks >/dev/null 2>&1
source /flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/activate 2>/dev/null
export PYTHONPATH=/flare/ModCon/ngetty/vllm-xpu-src:${PYTHONPATH:-}
export ZE_FLAT_DEVICE_HIERARCHY=FLAT VLLM_TARGET_DEVICE=xpu
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi
# CCL_KVS_IFACE=lo for single-node, matching serve_k3.sh:528. Omitting it let
# TP=4 hang: all four workers spun in all_reduce inside init_device
# (xpu_worker.py:97) for 8+ minutes at 100% CPU while CCL logged "could not get
# local_idx/count from environment variables, trying to get them from ATL".
# TP=1 and TP=2 happened to survive without it, which is exactly what makes this
# kind of omission dangerous -- it looks fine until the rank count grows.
export CCL_KVS_IFACE=${CCL_KVS_IFACE:-lo}
# Raise the executor RPC timeout from its 10 s default (envs.py:98). With async
# scheduling off, r_tp2 got 89 requests deep -- through the whole 60-request
# probe, all of single_step, and several ladder depths -- then died with
# "RPC call to execute_model timed out". A single execute_model on this slice
# is far under 10 s at 22 tok/s, so the default leaves no margin for a slow
# prefill or a scheduler stall, and turns a hiccup into a dead engine.
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-120000}
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
  # Killing the API server is NOT enough at TP>1. Its VLLM::Worker_TPn children
  # survive, get reparented to init (PPID=1), and keep SPINNING at 100% CPU
  # while holding their tiles -- observed: two orphaned TP=2 workers still
  # running 17 minutes after their parent was killed, starving the next
  # topology. Reap any worker whose process group is ours.
  #
  # Matching on the "VLLM::Worker" comm is safe here in a way that
  # `pkill -f api_server` was not: this script's own command line does not
  # contain that string, so it cannot self-match. Restricted to our own
  # process group so a concurrent job on the node is never touched.
  # Resolve the real process group rather than assuming $$ == pgid; that
  # holds for a normally-started script but not when invoked as `bash script`
  # from a driver. Verified on-node that workers inherit the server's pgid and
  # keep it after being reparented to init, so this reaches orphans.
  MY_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
  if [ -n "$MY_PGID" ]; then
    for pid in $(pgrep -g "$MY_PGID" -f "VLLM::Worker" 2>/dev/null); do
      kill -9 "$pid" 2>/dev/null
    done
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
  --no-enable-prefix-caching --num-gpu-blocks-override "${BLOCKS_OVERRIDE:-16}" \
  "$([ -n "${ASYNC_SCHED:-}" ] && echo --async-scheduling || echo --no-async-scheduling)" \
  --gpu-memory-utilization 0.70 > "$OUT/server.log" 2>&1 &
# --no-async-scheduling by default. Async scheduling is auto-enabled
# (config/vllm.py:838) and selects step_with_batch_queue, whose
# `future.result()` at core.py:521 has NO TIMEOUT. When the executor future is
# never fulfilled the engine blocks forever: observed EngineCore idle in `Sl`
# with the busy loop parked on that line and NO worker process alive to
# complete it. /health kept returning 200 the whole time.
#
# That is a permanent, silent hang rather than an error, and it is what has
# killed the depth ladder in every single run so far -- the ladder has never
# once completed. Set ASYNC_SCHED=1 to restore the default for comparison.
# --num-gpu-blocks-override is REQUIRED here, and 16 is measured, not guessed.
# Both extremes fail and they bracket the answer:
#
#   blocks=8     ->     1,024 tokens  -> starved: tp1 AND tp2 both died on
#                                        EXACTLY request #30 after 29 successes.
#                                        An identical count across two
#                                        topologies is a deterministic resource
#                                        limit, not a numerics bug.
#   no override  -> 3,304,832 tokens  -> OOM, "tried to allocate 36.25 GiB" on
#                                        the first request (tp4ep).
#
# Block size here is 128 tokens (8 blocks gave exactly 1024), so 16 blocks =
# 2048 tokens = 4x max_model_len. With max_num_seqs=1 and 128-token
# generations that is ample headroom, and twice what starved.
#
# Both failures LOOK like model defects -- an engine that wedges partway
# through a probe, an engine that dies on request one -- and neither is.
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
# Run the request-count test FIRST, on a fresh engine.
#
# Three topologies (tp1, tp2, tp8ep) each died after exactly 29 successful
# completions, and doubling the KV cache did not move the number -- so the
# cache-starvation explanation is refuted. This fires 60 identical trivial
# requests before anything else touches the server, which distinguishes
# "the engine degrades with request count" from "the depth ladder's request is
# the trigger". It must go first or the other probes consume the budget it is
# trying to measure.
echo; echo "--- request-count limit (fresh engine, 60 trivial requests) ---"
timeout 900 python3 "$ROOT/probe_request_count_limit.py" \
  "http://127.0.0.1:$PORT" k3slice 60 2>&1 | tail -25
verdict "tp=$TP request_count exit=${PIPESTATUS[0]}"

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

echo; echo "--- server health scan (K3 crash signatures) ---"
# scripts/check_run_health.sh keys on GRPO/SFT training-loop log markers and
# would falsely call a healthy vLLM serving log DEGRADED (it never emits
# "TIMING step=" etc). Use the K3-specific scanner instead: it checks the
# SERVER's own log for banned:1/SIGABRT/ActorDiedError/RayChannelTimeoutError,
# which a probe's clean exit code alone would not catch (e.g. the server can
# crash on a later request after earlier ones in the same probe succeeded).
if "$ROOT/check_k3_serving_health.sh" "$OUT/server.log" 2>&1 | tee "$OUT/health_scan.log"; then
  verdict "tp=$TP HEALTH_SCAN_GREEN"
else
  verdict "tp=$TP HEALTH_SCAN_DEGRADED"
fi

echo; echo "=== suite done $(date -Is) ==="
cat "$OUT/VERDICT.txt"

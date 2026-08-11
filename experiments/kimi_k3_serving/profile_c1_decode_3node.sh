#!/usr/bin/env bash
# STEP 1 -- the gating experiment: where does the 1249 ms c=1 decode step go?
#
# Everything downstream is sized by this. No tool anywhere produces a
# prefill/decode/collective breakdown for K3, so the current plan rests on a
# MODEL of the split (~463 collectives x 1-2 ms vs ~20,700 dispatches x 7 us),
# and the two candidate models imply opposite next steps:
#
#   collectives dominate -> cut the collective count (Step 2). Graph capture
#                           replays the same 463 round-trips and cannot help.
#   dispatch dominates   -> graph capture (Step 5), which needs a venv
#                           migration to torch 2.11.
#
# One hour of node time here decides which, instead of finding out after the
# migration. Decision rules are pre-registered in analyze_decode_trace.py and
# applied by it, not chosen by eye afterwards.
#
# Config is byte-identical to the 65.20 tok/s reference run (job 8746183)
# except for --profiler-config. mnbt stays at 2048: 4096 is the confirmed
# banned:1 regime and would fault before producing a trace.
#
# Usage (from a login node, against a running 3-node hold):
#   bash profile_c1_decode_3node.sh <FULL_PBS_JOB_ID>
set -uo pipefail

JOB_ID=${1:?Usage: $0 FULL_PBS_JOB_ID}
case "$JOB_ID" in
    *.aurora-pbs-*) ;;
    *) echo "ERROR: need the FULL PBS job id (with .aurora-pbs-...)" >&2; exit 2 ;;
esac

REPO=/lus/flare/projects/ModCon/ngetty/torchtune
EXP=$REPO/experiments/kimi_k3_serving
PYTHON=/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python
MODEL=${MODEL:-/tmp/ngetty/AuroraGPT/prism_models}
SERVED=${SERVED:-kimi-k3}
PORT=${PORT:-8000}
# 32-in: 512-in gives 0 completions in 544 s on this stack (the KDA prefill
# steps one token at a time), so a long prompt would spend the whole trace in
# prefill and never reach the decode steps this experiment is about.
PROMPT_TOKENS=${PROMPT_TOKENS:-32}
# Tokens to generate inside the profiled window.
DECODE_STEPS=${DECODE_STEPS:-8}
# delay_iterations / max_iterations are BOTH 0 on purpose -- do not "improve"
# this by skipping the first few steps.
#
# Attempt 1 (job 8748640) used delay=3/max=8 to skip prefill and the first
# decodes, and captured NOTHING: 32 workers, zero trace files, HTTP 200 on
# both /start_profile and /stop_profile. With delay>0, WorkerProfiler.start()
# only ARMS the profiler (`_active=True`); the real start is deferred to the
# `_active_iteration_count == delay` check inside step(), which is driven by
# `annotate_profile()` in Worker.execute_model. On this XPU/Ray path that
# step() never ran, so `_running` stayed False, "Starting profiler after
# delay..." never logged, and stop() had nothing to write.
#
# With both at 0, start() calls _call_start() immediately and stop() writes --
# no dependence on the step() hook at all. Cost: the trace also contains the
# prefill step, which is easy to identify (it is the long one) and which
# analyze_decode_trace.py's --steps normalization accounts for.
DELAY_ITERS=${DELAY_ITERS:-0}
MAX_ITERS=${MAX_ITERS:-0}

RUN_DIR=$EXP/logs/prof_c1_${JOB_ID%%.*}
TRACE_DIR=$RUN_DIR/traces
mkdir -p "$TRACE_DIR"
exec > >(tee -a "$RUN_DIR/profile.log") 2>&1

echo "phase=start job=$JOB_ID time=$(date -Is)"

NODEFILE=$RUN_DIR/nodefile
qstat -f "$JOB_ID" | tr -d '\n\t ' \
    | grep -oP 'exec_host=\K.*?(?=exec_vnode)' \
    | tr '+' '\n' | sed 's#/.*##' | grep -oE '^x[0-9a-z]+' | sort -u > "$NODEFILE"
mapfile -t NODES < "$NODEFILE"
echo "nodes=${NODES[*]} count=${#NODES[@]}"
[[ ${#NODES[@]} -eq 3 ]] || { echo "ERROR: need exactly 3 nodes, got ${#NODES[@]}"; exit 2; }
export PBS_NODEFILE="$NODEFILE"
export K3_NODEFILE="$NODEFILE"
HEAD=${NODES[0]}

VLLM_SRC=${VLLM_SRC:-/flare/ModCon/ngetty/vllm-xpu-src}
TREE_SHA=$(git -C "$VLLM_SRC" diff --binary HEAD | sha256sum | cut -d' ' -f1)
echo "vllm_commit=$(git -C "$VLLM_SRC" rev-parse HEAD) tree_sha=$TREE_SHA"

echo "phase=daos_mount"
LOG_DIR="$RUN_DIR" EXPECT_NODES=3 bash "$EXP/mount_daos_models_all_nodes.sh" \
    || { echo "ERROR: DAOS mount failed"; exit 2; }
ssh -o BatchMode=yes "$HEAD" "test -d $MODEL" \
    || { echo "ERROR: model not visible after mount: $MODEL"; exit 2; }
echo "daos_mount_ok"

# The trace dir must exist on every node -- each of the 32 ranks writes its
# own file, and 24 of them are not on the head node.
for node in "${NODES[@]}"; do
    ssh -o BatchMode=yes "$node" "mkdir -p '$TRACE_DIR'" || {
        echo "ERROR: could not create trace dir on $node"; exit 2; }
done

SERVER_DIR=$RUN_DIR/server
mkdir -p "$SERVER_DIR"
cache_root=/tmp/k3_hf_prof_${JOB_ID%%.*}
ssh -o BatchMode=yes "$HEAD" "rm -rf $cache_root" 2>/dev/null

# ignore_frontend: the AsyncLLM front-end profiler does not track iterations,
# so it would capture the entire window and swamp the worker traces we want.
PROFILER_CFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_DIR\",\"delay_iterations\":$DELAY_ITERS,\"max_iterations\":$MAX_ITERS,\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true}"

echo "phase=server_start profiler_cfg=$PROFILER_CFG"
ssh -o BatchMode=yes "$HEAD" \
    "VLLM_KIMI_XPU_KDA_VECTORIZED=1 VLLM_KIMI_XPU_CONV1D_VECTORIZED=1 \
     VLLM_KIMI_XPU_KDA_TRITON=0 VLLM_KIMI_XPU_CAUSAL_CONV1D_TRITON=0 \
     VLLM_XPU_ALLOW_TRITON_SAMPLER=0 \
     EXTRA_SERVER_ARGS='--profiler-config $PROFILER_CFG' \
     K3_JOB_ID='$JOB_ID' K3_NODEFILE='$NODEFILE' PBS_NODEFILE='$NODEFILE' \
     K3_CACHE_ROOT='$cache_root' LOG_DIR='$SERVER_DIR' \
     nohup timeout 7200 bash '$EXP/serve_k3.sh' \
        --model '$MODEL' --served-model-name '$SERVED' \
        --tp 32 --ep --port $PORT \
        --max-model-len 2048 --max-num-seqs 128 \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.92 \
        --diagnostic-blocks 800 --no-async-scheduling" \
    >"$SERVER_DIR/launcher.log" 2>&1 &
SSH_PID=$!

cleanup() {
    echo "phase=drain"
    kill -TERM "$SSH_PID" 2>/dev/null
    local pids=()
    for node in "${NODES[@]}"; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
            "pkill -f '[a]pi_server'; pkill -f '[V]LLM::'; pkill -f '[r]ay::'; \
             sleep 8; pkill -9 -f '[V]LLM::'; pkill -9 -f '[r]ay::'; true" \
            >/dev/null 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
    kill -9 "$SSH_PID" 2>/dev/null
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 270); do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD" \
        "curl -s --noproxy '*' -o /dev/null -w '%{http_code}' \
         http://127.0.0.1:$PORT/health 2>/dev/null" | grep -q 200; then
        ready=1; break
    fi
    sleep 10
done
[[ "$ready" == 1 ]] || {
    echo "RESULT verdict=NO_START"
    tail -30 "$SERVER_DIR/launcher.log" 2>/dev/null
    exit 1
}
echo "server_ready time=$(date -Is)"

# RESULTS_DISCIPLINE: the worker must say what it resolved before any number
# from it is interpretable.
echo "phase=worker_gate_echo"
# Count first, THEN print. `grep ... | head -4 || echo WARNING` looks right and
# is wrong: head exits after 4 lines, grep dies of SIGPIPE, and the pipeline
# returns 141, so the warning fires on every successful run. (Observed on the
# first real run -- four correct gate lines printed, immediately followed by
# "worker echo missing".) A false alarm on the one check whose whole job is to
# tell you the flags are unverified is worse than no check.
gate_lines=$(grep -hc "K3_WORKER_GATES" "$SERVER_DIR"/*.log 2>/dev/null | paste -sd+ | bc)
if [[ "${gate_lines:-0}" -gt 0 ]]; then
    echo "worker_gate_echo_lines=$gate_lines"
    grep -h "K3_WORKER_GATES" "$SERVER_DIR"/*.log 2>/dev/null | sed -n '1,4p'
else
    echo "WARNING: no K3_WORKER_GATES lines -- worker echo missing, flags unverified"
fi

# Warm up OUTSIDE the profiled window: the very first request pays one-time
# costs (Triton/SPIR-V JIT, allocator growth) that are not part of a
# steady-state decode step.
echo "phase=warmup"
ssh -o BatchMode=yes "$HEAD" \
    "curl -s --noproxy '*' -X POST http://127.0.0.1:$PORT/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"$SERVED\",\"prompt\":\"$(head -c 64 /dev/zero | tr '\0' 'a')\",\"max_tokens\":16,\"temperature\":0,\"ignore_eos\":true}'" \
    >"$RUN_DIR/warmup.json" 2>&1
echo "warmup_done rc=$?"

echo "phase=profile c=1 decode_steps=$DECODE_STEPS delay=$DELAY_ITERS max=$MAX_ITERS"
ssh -o BatchMode=yes "$HEAD" \
    "curl -s --noproxy '*' -X POST http://127.0.0.1:$PORT/start_profile" \
    >"$RUN_DIR/start_profile.txt" 2>&1

# /start_profile returns 200 whether or not a worker profiler actually began
# recording -- attempt 1 got 200 on both calls and produced zero bytes. Assert
# the worker-side evidence NOW, before spending the request, so a
# misconfiguration costs seconds instead of a whole profiled run.
sleep 3
if ! grep -qE "Torch profiling enabled|Starting profiler" "$SERVER_DIR/launcher.log" 2>/dev/null; then
    echo "WARNING: no worker-side profiler start evidence in the server log."
    echo "  /start_profile returning 200 only means the API accepted it."
fi

# ONE request, ONE concurrent user: this experiment is about the single-user
# step, and any second in-flight request changes the batch shape.
ssh -o BatchMode=yes "$HEAD" \
    "timeout 900 curl -s --noproxy '*' -X POST http://127.0.0.1:$PORT/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"$SERVED\",\"prompt\":\"$(head -c $((PROMPT_TOKENS * 4)) /dev/zero | tr '\0' 'a')\",\"max_tokens\":$((DECODE_STEPS + DELAY_ITERS + 4)),\"temperature\":0,\"ignore_eos\":true}'" \
    >"$RUN_DIR/profiled_request.json" 2>&1
echo "profiled_request_rc=$?"

ssh -o BatchMode=yes "$HEAD" \
    "curl -s --noproxy '*' -X POST http://127.0.0.1:$PORT/stop_profile" \
    >"$RUN_DIR/stop_profile.txt" 2>&1

# Traces flush asynchronously on stop; give the 32 ranks time to write.
echo "phase=trace_flush"
for _ in $(seq 1 30); do
    n=$(ssh -o BatchMode=yes "$HEAD" "ls '$TRACE_DIR' 2>/dev/null | wc -l")
    echo "  trace_files_on_head=$n"
    [[ "$n" -gt 0 ]] && sleep 20 && break
    sleep 10
done

# Pull every rank's trace back to the shared filesystem. TRACE_DIR is already
# on flare, but ranks on the non-head nodes wrote to their own view of it --
# list from each node so a missing rank is visible rather than assumed.
for node in "${NODES[@]}"; do
    echo "traces_on_${node}: $(ssh -o BatchMode=yes "$node" "ls '$TRACE_DIR' 2>/dev/null | wc -l")"
done

shopt -s nullglob
traces=("$TRACE_DIR"/*.json "$TRACE_DIR"/*.json.gz)
echo "trace_count=${#traces[@]}"
if [[ ${#traces[@]} -eq 0 ]]; then
    echo "RESULT verdict=NO_TRACE -- profiler produced nothing."
    echo "  Check: does the server log mention 'Profiler'? Is the dir writable"
    echo "  from the Ray actors? Is --profiler-config in server_args?"
    grep -h "server_args=\|extra_server_args=" "$SERVER_DIR/metadata" 2>/dev/null
    exit 1
fi

echo "phase=analyze"
# Rank 0 is the driver; analyze it first, then a mid-node rank as a cross-check
# (a 3-node job's collective cost is not necessarily symmetric across nodes).
for trace in "${traces[@]:0:2}"; do
    echo "---- $trace"
    "$PYTHON" "$EXP/analyze_decode_trace.py" "$trace" \
        --steps "$DECODE_STEPS" \
        --json "${trace%.json*}.report.json"
done

echo ""
echo "phase=done time=$(date -Is)"
echo "traces:  $TRACE_DIR"
echo "reports: ${TRACE_DIR}/*.report.json"

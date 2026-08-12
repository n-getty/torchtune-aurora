#!/usr/bin/env bash
# Real single-user (c=1) tok/s A/B across an arbitrary set of env legs.
#
# Why this and not more microbenchmarks: the microbenchmark route has now
# refuted three hypotheses in a row (ZE_ENABLE_API_TRACING, host contention,
# per-launch dispatch cost) while the actual quantity of interest -- c=1
# tok/s -- has never been measured with any lever applied. Model load is ~3
# min once the page cache is warm (attempt 2 of the profiling run), so a leg
# costs ~6 min. Measure the thing we care about.
#
# All legs run in ONE allocation, back to back, with the tree sha re-checked
# between them (the drift trap that voided an earlier A/B).
#
# Usage:
#   bash ab_c1_levers_3node.sh <FULL_PBS_JOB_ID> [--legs "name=ENV=V,ENV2=V2;..."]
# Default legs: baseline, shared-expert AR fusion, host-sync hoist already
# being in the tree (so baseline includes it), CCL_OP_SYNC=0.
set -uo pipefail

JOB_ID=${1:?Usage: $0 FULL_PBS_JOB_ID}
shift || true
case "$JOB_ID" in
    *.aurora-pbs-*) ;;
    *) echo "ERROR: need the FULL PBS job id" >&2; exit 2 ;;
esac

# Refuse to run two drivers against the same allocation. On job 8749119 a
# relaunch was started while the first driver was still alive; both legs used
# the same RAY_TEMP_ROOT (keyed on job id) and killed each other's Ray
# workers, producing a NO_START that looked like a node/memory fault. The
# lock makes that operator error impossible instead of merely unlikely.
LOCK=/tmp/k3_abc1_driver_${JOB_ID%%.*}.lock
exec 9>"$LOCK" || { echo "ERROR: cannot open $LOCK" >&2; exit 2; }
if ! flock -n 9; then
    echo "ERROR: another ab_c1_levers driver is already running for ${JOB_ID%%.*}." >&2
    echo "  Kill it first (pkill -f '[a]b_c1_levers_3node') and drain Ray," >&2
    echo "  or wait for it to finish. Two drivers share RAY_TEMP_ROOT and will" >&2
    echo "  tear down each other's workers." >&2
    exit 2
fi
echo "driver_lock=$LOCK pid=$$"

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
MODEL=${MODEL:-/tmp/ngetty/AuroraGPT/prism_models}
SERVED=${SERVED:-kimi-k3}
PORT=${PORT:-8000}
# 32-in / 64-out at c=1: enough decode steps to average, short enough that a
# leg is ~6 min including load. 512-in is unusable (0 completions in 544 s).
PROMPT_TOKENS=${PROMPT_TOKENS:-32}
MAX_TOKENS=${MAX_TOKENS:-64}
REPEATS=${REPEATS:-3}
# vLLM refuses to start if any tile has less free memory than this fraction
# demands, and Aurora tiles are NOT reliably clean: on job 8749119 one node of
# three had 52.6/64 GiB free per tile (the other two had 62.7 and 60.7) with no
# user process on it, which failed every leg at init with
# "Free memory on device xpu:N ... is less than desired GPU memory utilization".
# Keep this BELOW the worst tile's free fraction. 0.92 needs 58.9 GiB free.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}

# name=VAR=VAL[,VAR=VAL...]  (semicolon-separated legs)
LEGS=${LEGS:-"baseline=;ar_fusion=VLLM_KIMI_FUSE_SHARED_EXPERT_AR=1;op_sync_off=CCL_OP_SYNC=0;both=VLLM_KIMI_FUSE_SHARED_EXPERT_AR=1,CCL_OP_SYNC=0"}
[[ "${1:-}" == "--legs" ]] && LEGS=$2

RUN_DIR=$EXP/logs/abc1_${JOB_ID%%.*}
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/ab.log") 2>&1
echo "phase=start job=$JOB_ID time=$(date -Is)"

NODEFILE=$RUN_DIR/nodefile
qstat -f "$JOB_ID" | tr -d '\n\t ' \
    | grep -oP 'exec_host=\K.*?(?=exec_vnode)' \
    | tr '+' '\n' | sed 's#/.*##' | grep -oE '^x[0-9a-z]+' | sort -u > "$NODEFILE"
mapfile -t NODES < "$NODEFILE"
echo "nodes=${NODES[*]}"
[[ ${#NODES[@]} -eq 3 ]] || { echo "ERROR: need 3 nodes"; exit 2; }
export PBS_NODEFILE="$NODEFILE" K3_NODEFILE="$NODEFILE"
HEAD=${NODES[0]}

VLLM_SRC=${VLLM_SRC:-/flare/ModCon/ngetty/vllm-xpu-src}
tree_sha() { git -C "$VLLM_SRC" diff --binary HEAD | sha256sum | cut -d' ' -f1; }
SHA0=$(tree_sha)
echo "vllm_commit=$(git -C "$VLLM_SRC" rev-parse HEAD) tree_sha=$SHA0"

# Refuse to start a leg that cannot finish. A cold-cache K3 load is ~15-25
# min (7.8 s/shard x 96) and timing adds ~5. Attempt 3 of the capture run
# reached 100% weights and `enforce_eager=False` with ZERO errors, then died
# to walltime seconds before the server accepted a request -- a whole load
# spent for no number. Check the remaining time up front instead.
MIN_MINUTES_PER_LEG=${MIN_MINUTES_PER_LEG:-30}
rem=$(qstat -f "$JOB_ID" 2>/dev/null | tr -d '\n\t ' \
      | grep -oP 'Resource_List.walltime=\K[0-9:]+')
used=$(qstat -f "$JOB_ID" 2>/dev/null | tr -d '\n\t ' \
      | grep -oP 'resources_used.walltime=\K[0-9:]+')
to_min() { awk -F: '{print ($1*60)+$2}' <<<"$1"; }
if [[ -n "$rem" && -n "$used" ]]; then
    left=$(( $(to_min "$rem") - $(to_min "$used") ))
    n_legs=$(tr ';' '\n' <<<"$LEGS" | grep -c .)
    need=$(( MIN_MINUTES_PER_LEG * n_legs ))
    echo "walltime_left=${left}min legs=$n_legs need=${need}min"
    if (( left < need )); then
        echo "ERROR: only ${left} min left but ${n_legs} leg(s) need ~${need} min." >&2
        echo "  A leg that dies to walltime mid-load costs a full model load for" >&2
        echo "  zero data. Submit a fresh hold, or lower MIN_MINUTES_PER_LEG if" >&2
        echo "  you know the page cache is warm." >&2
        exit 2
    fi
fi

# Ray placement groups survive a pkill-based teardown. `ray stop` releases
# them; killing the processes does not, and the GPUs stay reserved in
# bundle_group_* entries. vLLM checks AVAILABLE (not total) GPUs on the
# driver node (ray_utils.py:646-655), so a leaked PG yields
# "Current node has no GPU available" -- which reads like a hardware fault
# and cost a full model load on job 8749119. Force a clean slate first.
echo "phase=ray_stop"
for node in "${NODES[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$node" \
        "source '$EXP/../ray_smoke/setup_ray_env.sh' '${RAY_ENV_MODE:-frameworks}' >/dev/null 2>&1; \
         ray stop --force >/dev/null 2>&1; true" >/dev/null 2>&1 &
done
wait
sleep 5

echo "phase=daos_mount"
LOG_DIR="$RUN_DIR" EXPECT_NODES=3 bash "$EXP/mount_daos_models_all_nodes.sh" \
    || { echo "ERROR: DAOS mount failed"; exit 2; }

drain() {
    local pids=()
    for node in "${NODES[@]}"; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
            "pkill -f '[a]pi_server'; pkill -f '[V]LLM::'; pkill -f '[r]ay::'; \
             sleep 8; pkill -9 -f '[V]LLM::'; pkill -9 -f '[r]ay::'; \
             pkill -9 -f '[a]pi_server'; true" >/dev/null 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
    sleep 30
}
trap drain EXIT

run_leg() {
    local name=$1 envs=$2
    local dir="$RUN_DIR/$name"
    mkdir -p "$dir"
    echo ""
    echo "================ LEG $name  env=[${envs:-none}] ================"

    local now; now=$(tree_sha)
    [[ "$now" == "$SHA0" ]] || {
        echo "RESULT leg=$name verdict=VOID (tree changed: $SHA0 -> $now)"; return; }

    local envstr=""
    [[ -n "$envs" ]] && envstr=$(echo "$envs" | tr ',' ' ')
    local cache=/tmp/k3_abc1_${JOB_ID%%.*}_${name}
    ssh -o BatchMode=yes "$HEAD" "rm -rf $cache" 2>/dev/null

    ssh -o BatchMode=yes "$HEAD" \
        "VLLM_KIMI_XPU_KDA_VECTORIZED=1 VLLM_KIMI_XPU_CONV1D_VECTORIZED=1 \
         VLLM_KIMI_XPU_KDA_TRITON=0 VLLM_KIMI_XPU_CAUSAL_CONV1D_TRITON=0 \
         VLLM_XPU_ALLOW_TRITON_SAMPLER=0 $envstr \
         PYTHON='$PYTHON' RAY_ENV_MODE='${RAY_ENV_MODE:-frameworks}' \
         K3_JOB_ID='$JOB_ID' K3_NODEFILE='$NODEFILE' PBS_NODEFILE='$NODEFILE' \
         K3_CACHE_ROOT='$cache' LOG_DIR='$dir' \
         RAY_TEMP_ROOT='/tmp/k3_ray_${JOB_ID%%.*}_${name}' \
         nohup timeout 3600 bash '$EXP/serve_k3.sh' \
            --model '$MODEL' --served-model-name '$SERVED' \
            --tp 32 --ep --port $PORT \
            --max-model-len 2048 --max-num-seqs 128 \
            --max-num-batched-tokens 2048 \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --diagnostic-blocks 800 --no-async-scheduling" \
        >"$dir/launcher.log" 2>&1 &
    local pid=$!

    local ready=0
    for _ in $(seq 1 180); do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD" \
            "curl -s --noproxy '*' -o /dev/null -w '%{http_code}' \
             http://127.0.0.1:$PORT/health 2>/dev/null" | grep -q 200 && { ready=1; break; }
        sleep 10
    done
    if [[ "$ready" != 1 ]]; then
        local f; f=$(grep -rho "banned: *1" "$dir" 2>/dev/null | wc -l)
        echo "RESULT leg=$name verdict=$([[ $f -gt 0 ]] && echo FAULTED || echo NO_START)"
        kill -TERM "$pid" 2>/dev/null; drain; return
    fi

    # Confirm the leg's env actually reached a worker before believing its number.
    if [[ -n "$envs" ]]; then
        local var=${envs%%=*}
        # Search the launcher log (where Ray worker stdout lands) AND the
        # captured env files. Greping only "$dir"/*.log missed it and printed
        # NOT-FOUND on a leg whose flag WAS active -- a false alarm on the one
        # check that exists to catch inactive flags.
        local seen; seen=$(grep -ho "$var=[^ ]*" "$dir"/launcher.log "$dir"/*.log "$dir"/*.txt 2>/dev/null | sort -u | head -2 | tr '\n' ' ')
        echo "leg_env_in_worker: ${seen:-NOT-FOUND}"
    fi

    # Warm up (first request pays JIT/allocator), then time c=1.
    ssh -o BatchMode=yes "$HEAD" \
        "curl -s --noproxy '*' -X POST http://127.0.0.1:$PORT/v1/completions \
          -H 'Content-Type: application/json' -d '{\"model\":\"$SERVED\",
          \"prompt\":\"$(head -c 64 /dev/zero | tr '\0' 'a')\",\"max_tokens\":8,
          \"temperature\":0,\"ignore_eos\":true}'" >"$dir/warmup.json" 2>&1

    local prompt; prompt=$(head -c $((PROMPT_TOKENS*4)) /dev/zero | tr '\0' 'a')
    for r in $(seq 1 "$REPEATS"); do
        local t0 t1
        t0=$(date +%s.%N)
        ssh -o BatchMode=yes "$HEAD" \
            "timeout 900 curl -s --noproxy '*' -X POST http://127.0.0.1:$PORT/v1/completions \
              -H 'Content-Type: application/json' -d '{\"model\":\"$SERVED\",
              \"prompt\":\"$prompt\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0,
              \"ignore_eos\":true}'" >"$dir/req_$r.json" 2>&1
        t1=$(date +%s.%N)
        local toks; toks=$("$PYTHON" -c "
import json,sys
try: print(json.load(open('$dir/req_$r.json'))['usage']['completion_tokens'])
except Exception: print(0)")
        echo "  rep$r tokens=$toks wall=$(echo "$t1-$t0"|bc)s tok_s=$("$PYTHON" -c "
print(f'{$toks/max(1e-9,$t1-$t0):.3f}')")"
    done

    local best; best=$(grep -oE "tok_s=[0-9.]+" "$RUN_DIR/ab.log" | tail -"$REPEATS" \
                       | cut -d= -f2 | sort -rn | head -1)
    local faults; faults=$(grep -rho "banned: *1" "$dir" 2>/dev/null | wc -l)
    echo "RESULT leg=$name c1_tok_s=$best banned=$faults verdict=$([[ $faults -gt 0 ]] && echo FAULTED || echo OK)"

    kill -TERM "$pid" 2>/dev/null
    drain
    kill -9 "$pid" 2>/dev/null
}

IFS=';' read -ra LEGARR <<< "$LEGS"
for leg in "${LEGARR[@]}"; do
    run_leg "${leg%%=*}" "${leg#*=}"
done

echo ""
echo "================ SUMMARY ================"
grep -h "^RESULT" "$RUN_DIR/ab.log" | sort -u
echo "phase=done time=$(date -Is)"

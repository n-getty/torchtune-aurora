#!/usr/bin/env bash
# Unattended overnight run for hold 8749725 (or $1).
#
# Ordered by (value x feasibility), highest first, so that if the hold dies
# early we still have the most valuable results. Each phase is independent:
# a failure in one does not block the next.
#
# Phase A  fused-KDA CORRECTNESS  -- unblocks a measured +9.8% (flag is OFF
#          today purely because the smoke test was 64 'a's -> 64 'a's).
# Phase B  concurrency sweep      -- the ONLY route to 20-60 tok/s on this
#          hardware tonight (see PATH_TO_20_TOK_S.md). c=1 is capped near
#          7.7 tok/s even with perfect eager-overhead removal.
# Phase C  whole-graph compile    -- attacks the 452 ms dispatch term without
#          the blocked graph_capture() path. Genuinely unknown; may fail.
#
# Everything writes under logs/overnight_<jobid>/ and appends a one-line
# verdict to logs/overnight_<jobid>/SUMMARY.txt.
set -uo pipefail

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
JOB=${1:-}
if [[ -z "$JOB" ]]; then
    JOB=$(qstat -u ngetty 2>/dev/null | awk '/kimi_k3_h/ && $10=="R" {print $1}' | head -1)
    [[ -n "$JOB" ]] && JOB=$(qstat -x -f "$JOB" 2>/dev/null | awk -F': ' '/^Job Id/{print $2}')
fi
[[ -n "$JOB" ]] || { echo "no running kimi hold found"; exit 2; }

OUT=$EXP/logs/overnight_${JOB%%.*}
mkdir -p "$OUT"
SUM=$OUT/SUMMARY.txt
say() { echo "[$(date -Is)] $*" | tee -a "$SUM"; }

say "overnight start job=$JOB"
say "vllm=$(git -C /flare/ModCon/ngetty/vllm-xpu-src rev-parse --short HEAD) torchtune=$(git -C /lus/flare/projects/ModCon/ngetty/torchtune rev-parse --short HEAD)"

mins_left() {
    local r u
    r=$(qstat -f "$JOB" 2>/dev/null | tr -d '\n\t ' | grep -oP 'Resource_List.walltime=\K[0-9:]+')
    u=$(qstat -f "$JOB" 2>/dev/null | tr -d '\n\t ' | grep -oP 'resources_used.walltime=\K[0-9:]+')
    [[ -z "$r" || -z "$u" ]] && { echo 0; return; }
    echo $(( $(awk -F: '{print ($1*60)+$2}' <<<"$r") - $(awk -F: '{print ($1*60)+$2}' <<<"$u") ))
}

run_phase() {  # name need_min env-assignments...
    local name=$1 need=$2; shift 2
    local left; left=$(mins_left)
    if (( left < need )); then
        say "SKIP $name (need ${need}min, have ${left}min)"
        return 1
    fi
    say "START $name (${left}min left)"
    rm -f /tmp/k3_abc1_driver_${JOB%%.*}.lock
    cp "$EXP/ab_c1_levers_3node.sh" "/tmp/ab_ovn_${name}.sh"
    ( env "$@" bash "/tmp/ab_ovn_${name}.sh" "$JOB" ) >"$OUT/$name.log" 2>&1
    grep -E "^RESULT" "$OUT/$name.log" | tail -5 | while read -r l; do say "  $name | $l"; done
    say "END $name"
}

# ---------------------------------------------------------------- Phase A
# Correctness: same prompt with a checkable answer on both legs. If the
# completions differ, the fused kernel is wrong on hardware despite 16 green
# CPU tests -- which is the finding, and more important than any tok/s.
run_phase correctness 40 \
    GPU_MEM_UTIL=0.80 REPEATS=1 MAX_TOKENS=48 MIN_MINUTES_PER_LEG=18 \
    PROMPT='Q: A shop sells pens for 3 dollars each. Ana buys 7 pens and pays with a 50 dollar bill. How much change does she get? Answer with the number only.\nA:' \
    LEGS="corr_base=;corr_fused=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1"

python3 "$EXP/analysis/compare_leg_completions.py" \
    "$EXP/logs/abc1_${JOB%%.*}/corr_base" \
    "$EXP/logs/abc1_${JOB%%.*}/corr_fused" 2>&1 | tee -a "$SUM"

# ---------------------------------------------------------------- Phase B
# Concurrency. PATH_TO_20_TOK_S.md: single-user is capped near 7.7 tok/s even
# with perfect overhead removal, but aggregate scales -- 65.2 tok/s already
# measured at c=128. This re-measures the curve WITH fused KDA on, which has
# never been done, and tells us the best throughput config available today.
# One server load, several concurrency points measured against it -- the
# ladder is legs of the same run so the model is loaded ONCE. Aggregate tok/s
# is what the 20-60 target means on this hardware.
# NOTE: each leg reloads the model (~12 min), so a 4-point ladder as separate
# legs costs ~48 min of pure loading. Accept 2 points -- c=32 and c=64 -- and
# get c=1 for free from tonight's already-measured 1.267. The 65.2 tok/s at
# c=128 is already on record from 2026-08-10; the open question is whether
# fused KDA moves the aggregate curve, and two points answer that.
run_phase conc32 34 \
    GPU_MEM_UTIL=0.80 REPEATS=2 MAX_TOKENS=64 MIN_MINUTES_PER_LEG=16 \
    CONCURRENCY=32 \
    LEGS="c32_fused=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1"

run_phase conc64 34 \
    GPU_MEM_UTIL=0.80 REPEATS=2 MAX_TOKENS=64 MIN_MINUTES_PER_LEG=16 \
    CONCURRENCY=64 \
    LEGS="c64_fused=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1"

# ---------------------------------------------------------------- Phase C
# Whole-graph compile without cudagraphs. graph_capture() is blocked
# (CudaCommunicator assert), but torch.compile / Inductor does not route
# through it. mode=VLLM_COMPILE with cudagraph_mode=NONE is therefore
# UNTESTED and plausibly reachable. Expect it may fail on XPU Inductor; that
# is a legitimate result and is why it is last.
run_phase wholegraph 40 \
    GPU_MEM_UTIL=0.80 REPEATS=2 MIN_MINUTES_PER_LEG=18 \
    LEGS="compile_none=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE,VLLM_KIMI_XPU_KDA_FUSED_DECODE=1"

say "overnight done. walltime left: $(mins_left)min"
say "----- SUMMARY -----"
grep -E "RESULT|COMPLETION|SKIP" "$SUM" | tail -20

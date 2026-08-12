#!/usr/bin/env bash
# Re-run the fused-KDA correctness check, which lost its slot on job 8749725
# to the DAOS mount-visibility race (now fixed in ab_c1_levers_3node.sh).
#
# This is the highest-value cheap task outstanding: the fused KDA kernel is a
# MEASURED +9.8% (1.154 -> 1.267 tok/s) that stays default-OFF only because
# the A/B smoke test prompted with 64 'a's and got 64 'a's back -- degenerate
# output that a numerically wrong kernel would reproduce exactly.
#
# Waits for the overnight run to release the driver lock, then runs both legs
# with a checkable-answer prompt and diffs the completions.
#
# Usage: bash rerun_correctness.sh [FULL_PBS_JOB_ID]
set -uo pipefail

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
JOB=${1:-}
if [[ -z "$JOB" ]]; then
    J=$(qstat -u ngetty 2>/dev/null | awk '/kimi_k3_h/ && $10=="R" {print $1}' | head -1)
    [[ -n "$J" ]] && JOB=$(qstat -x -f "$J" 2>/dev/null | awk -F': ' '/^Job Id/{print $2}')
fi
[[ -n "$JOB" ]] || { echo "no running kimi hold"; exit 2; }

# Wait out any in-flight overnight phase; the driver holds an flock per
# allocation and a second driver is (correctly) refused.
for _ in $(seq 1 240); do
    pgrep -f "[a]b_ovn_" >/dev/null 2>&1 || break
    sleep 30
done

left=$(( $(qstat -f "$JOB" | tr -d '\n\t ' | grep -oP 'Resource_List.walltime=\K[0-9:]+' | awk -F: '{print ($1*60)+$2}') \
      - $(qstat -f "$JOB" | tr -d '\n\t ' | grep -oP 'resources_used.walltime=\K[0-9:]+' | awk -F: '{print ($1*60)+$2}') ))
echo "walltime_left=${left}min"
(( left >= 36 )) || { echo "SKIP: need 36min for two legs, have ${left}"; exit 3; }

rm -f "/tmp/k3_abc1_driver_${JOB%%.*}.lock"
cp "$EXP/ab_c1_levers_3node.sh" /tmp/ab_corr.sh   # frozen copy: never run an editable script

OUT=$EXP/logs/overnight_${JOB%%.*}
mkdir -p "$OUT"

env GPU_MEM_UTIL=0.80 REPEATS=1 MAX_TOKENS=48 MIN_MINUTES_PER_LEG=17 \
    PROMPT='Q: A shop sells pens for 3 dollars each. Ana buys 7 pens and pays with a 50 dollar bill. How much change does she get? Answer with the number only.\nA:' \
    LEGS="corr2_base=;corr2_fused=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1" \
    bash /tmp/ab_corr.sh "$JOB" >"$OUT/correctness2.log" 2>&1

{
  echo "----- correctness rerun -----"
  grep -E "^RESULT" "$OUT/correctness2.log" | tail -4
  python3 "$EXP/analysis/compare_leg_completions.py" \
      "$EXP/logs/abc1_${JOB%%.*}/corr2_base" \
      "$EXP/logs/abc1_${JOB%%.*}/corr2_fused"
} 2>&1 | tee -a "$OUT/SUMMARY.txt"

#!/usr/bin/env bash
# Correctness gate for the compile_seg speedup measured on job 8750347.
#
# WHY THIS EXISTS. compile_seg measured 1.395 tok/s vs eager 1.158 (+20.5%),
# and the diagnostic confirms compile really engaged (mode=VLLM_COMPILE,
# splitting_ops populated, 3 Dynamo transforms, 0 graph breaks) -- so it is
# not a silent-eager-fallback artifact. But BOTH legs ran the default
# PROMPT_TOKENS*4 'a'-repeat prompt and both answered with a run of 'a's:
#   COMPLETION=IDENTICAL_BUT_DEGENERATE
# A numerically wrong path reproduces that exactly. It proves nothing.
#
# This is not hypothetical on this stack. The fused-KDA decode kernel
# measured +9.8% on exactly this harness and was found NUMERICALLY WRONG
# (198ac00a) the moment a real prompt was used: base answered 391/414/437,
# fused answered 391/391/391 -- identical for 18 chars, then permanent
# divergence, the compounding signature of corrupted recurrent state.
# A speedup on an unvalidated path is not a result.
#
# METHOD. Re-run eager and compile_seg with the ONLY prompt known to work
# end-to-end on this stack, then diff the generated text. At temperature=0
# the two legs must produce byte-identical output; compile changes kernel
# fusion, not semantics.
#
# PROMPT CHOICE -- do not "improve" this without reading the history:
#   * 41-token prompt => hangs exactly 300.9s = RAY_CGRAPH_get_timeout, leg
#     returns NO_TOKENS (d9437e2c). The XPU KDA prefill steps one token at a
#     time; 512-in is documented unusable.
#   * This 12-token prompt + VLLM_KIMI_XPU_KDA_CHUNKED=1 is the ONLY
#     configuration that has produced non-degenerate K3 text (corr3, job
#     8749725), and its answers were arithmetically correct.
# CHUNKED=1 is therefore held FIXED ON in both legs -- it is the carrier that
# makes a real prompt viable at all, not the variable under test. The single
# variable between legs is compile.
#
# Usage: bash rerun_compile_correctness.sh [FULL_PBS_JOB_ID]
set -uo pipefail

EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving

JOB=${1:-}
if [[ -z "$JOB" ]]; then
    # qstat -u TRUNCATES the job id ("8750347.aurora-pbs-*"), which then does
    # not resolve with `qstat -x -f`. Take the numeric prefix, re-resolve.
    J=$(qstat -u ngetty 2>/dev/null | awk '/kimi_k3_h/ && $10=="R" {print $1}' | head -1)
    J=${J%%.*}
    [[ -n "$J" ]] && JOB=$(qstat -x -f "$J" 2>/dev/null | awk -F': ' '/^Job Id/{print $2}')
fi
[[ -n "$JOB" ]] || { echo "no running kimi hold"; exit 2; }
echo "job=$JOB"

# Wait out the in-flight compile A/B. The driver holds an flock per
# allocation and correctly refuses a second driver; two drivers would share
# RAY_TEMP_ROOT and tear down each other's Ray workers.
for _ in $(seq 1 240); do
    pgrep -f "[a]b_compile\.sh|[a]b_c1_levers_3node|[a]b_ovn_" >/dev/null 2>&1 || break
    sleep 30
done

left=$(( $(qstat -f "$JOB" | tr -d '\n\t ' | grep -oP 'Resource_List.walltime=\K[0-9:]+' | awk -F: '{print ($1*60)+$2}') \
      - $(qstat -f "$JOB" | tr -d '\n\t ' | grep -oP 'resources_used.walltime=\K[0-9:]+' | awk -F: '{print ($1*60)+$2}') ))
echo "walltime_left=${left}min"
(( left >= 36 )) || { echo "SKIP: need 36min for two legs, have ${left}"; exit 3; }

rm -f "/tmp/k3_abc1_driver_${JOB%%.*}.lock"
cp "$EXP/ab_c1_levers_3node.sh" /tmp/ab_compile_corr.sh  # frozen copy: never run an editable script

OUT=$EXP/logs/compile_corr_${JOB%%.*}
mkdir -p "$OUT"

# Same venv/RAY_ENV_MODE as run_compile_ab.sh -- torch211. Changing the venv
# would add a second variable and void the comparison against 8750347.
env \
  PYTHON=/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python \
  RAY_ENV_MODE=torch211 \
  GPU_MEM_UTIL=0.80 REPEATS=1 MAX_TOKENS=32 MIN_MINUTES_PER_LEG=17 \
  PROMPT='Q: What is 17 times 23? Think step by step.\nA:' \
  LEGS="ccorr_eager=VLLM_KIMI_XPU_KDA_CHUNKED=1;ccorr_compile=VLLM_KIMI_XPU_KDA_CHUNKED=1,ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE" \
  bash /tmp/ab_compile_corr.sh "$JOB" >"$OUT/compile_correctness.log" 2>&1

{
  echo "----- compile correctness -----"
  grep -E "^RESULT" "$OUT/compile_correctness.log" | tail -4
  # Confirm compile actually engaged in the compile leg. Without this a
  # PASS is ambiguous: two eager legs also agree byte-for-byte.
  echo "--- did compile engage in ccorr_compile? ---"
  bash "$EXP/analysis/compile_diagnose.sh" \
      "$EXP/logs/abc1_${JOB%%.*}/ccorr_compile" 2>&1 | sed -n '1,8p'
  echo "--- text diff ---"
  python3 "$EXP/analysis/compare_leg_completions.py" \
      "$EXP/logs/abc1_${JOB%%.*}/ccorr_eager" \
      "$EXP/logs/abc1_${JOB%%.*}/ccorr_compile"
} 2>&1 | tee -a "$OUT/SUMMARY.txt"

#!/usr/bin/env bash
# Whole-graph compile A/B -- the 61% dispatch term. Top of PLAN_C1_ROOT_CAUSES.md.
#
# Baseline is eager with the BROKEN fused-KDA kernel OFF, so all three legs
# are directly comparable and none carries suspect numerics.
set -uo pipefail
EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
JOB=${1:?need FULL PBS job id}
cp "$EXP/ab_c1_levers_3node.sh" /tmp/ab_compile.sh
exec env \
  PYTHON=/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python \
  RAY_ENV_MODE=torch211 \
  GPU_MEM_UTIL=0.80 REPEATS=3 MAX_TOKENS=64 MIN_MINUTES_PER_LEG=30 \
  LEGS="eager_base=;compile_seg=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE;compile_whole=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE,SPLITTING_OPS_EMPTY=1" \
  bash /tmp/ab_compile.sh "$JOB"

#!/usr/bin/env bash
# Fused-KDA-decode A/B leg, ready to fire the moment the capture A/B finishes.
#
# Runs from a FROZEN copy of the driver (see below): bash reads a script
# incrementally, so editing ab_c1_levers_3node.sh while it runs tears the
# running copy -- that is what killed driver 1 on job 8749119 with
# "line 219: unexpected EOF".
#
# Also note: the driver holds an flock per allocation. Do NOT start this while
# the capture A/B is still running; it will (correctly) refuse.
#
# Usage:  bash run_fused_kda_leg.sh <FULL_PBS_JOB_ID>
set -uo pipefail

JOB=${1:?Usage: $0 FULL_PBS_JOB_ID}
EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
FROZEN=/tmp/ab_c1_fused_$$.sh
cp "$EXP/ab_c1_levers_3node.sh" "$FROZEN"

# Same venv as the baseline this is compared against. The fused kernel needs
# NO torch 2.11 -- it is plain Triton -- so use the default frameworks env and
# do not drag the capture leg's venv change into this measurement.
#
# Both legs in ONE invocation so they share nodes and allocation: node
# variance is a known confounder here, and the 1.138 tok/s reference was
# measured on different nodes in a different job.
exec env \
  REPEATS=3 MIN_MINUTES_PER_LEG=28 GPU_MEM_UTIL=0.80 \
  LEGS="base_fw=;fused_kda=VLLM_KIMI_XPU_KDA_FUSED_DECODE=1" \
  bash "$FROZEN" "$JOB"

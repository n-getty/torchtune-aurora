#!/usr/bin/env bash
# Whole-graph compile A/B -- the 61% dispatch term. Top of PLAN_C1_ROOT_CAUSES.md.
#
# Baseline is eager with the BROKEN fused-KDA kernel OFF, so all three legs
# are directly comparable and none carries suspect numerics.
set -uo pipefail
EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
JOB=${1:?need FULL PBS job id}
# The hold log is APPENDED across jobs, so `grep hold_ready` matches a
# PREVIOUS hold's line and returns instantly. On 8750347 that launched the
# A/B 44s before the nodes accepted ssh, and every daos mount failed with
# "Connection closed by ... port 22". Match THIS job id, and verify ssh
# actually works, before doing anything.
for _ in $(seq 1 60); do
    grep -q "hold_ready job=${JOB%%.*}" "$EXP/logs/hold_3n_capacity.out" 2>/dev/null && break
    sleep 10
done
NODES=$(qstat -f "$JOB" | tr -d '\n\t ' | grep -oP 'exec_host=\K.*?(?=exec_vnode)' \
        | tr '+' '\n' | sed 's#/.*##' | grep -oE '^x[0-9a-z]+' | sort -u)
for node in $NODES; do
    ok=0
    for _ in $(seq 1 30); do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" true 2>/dev/null && { ok=1; break; }
        sleep 10
    done
    [[ "$ok" == 1 ]] || { echo "ERROR: $node never accepted ssh"; exit 2; }
    echo "node=$node ssh=ok"
done

cp "$EXP/ab_c1_levers_3node.sh" /tmp/ab_compile.sh
exec env \
  PYTHON=/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu/bin/python \
  RAY_ENV_MODE=torch211 \
  GPU_MEM_UTIL=0.80 REPEATS=3 MAX_TOKENS=64 MIN_MINUTES_PER_LEG=30 \
  LEGS="eager_base=;compile_seg=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE;compile_whole=ENFORCE_EAGER=0,CUDAGRAPH_MODE=NONE,SPLITTING_OPS_EMPTY=1" \
  bash /tmp/ab_compile.sh "$JOB"

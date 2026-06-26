#!/bin/bash
# Autonomous overnight orchestrator for the go_pred RL effort (2026-06-26).
# Chain: (eval already submitted) -> go_pred 2N smoke -> if GREEN, 4N prod go_pred run.
# Survives session cycling; logs every decision. Respects the user's vjepa2 campaign
# (only submits when a global per-user Q-slot is free; never touches others' jobs).
#
# State + log under experiments/bioreason/overnight_state/.
set -uo pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
SD=$TT/experiments/bioreason/overnight_state
mkdir -p "$SD"
LOG=$SD/gopred_orch_$(date -u +%Y%m%d_%H%M%S).log
exec >>"$LOG" 2>&1
ts(){ date -u +%H:%M:%S; }
log(){ echo "[$(ts)] $*"; }

SMOKE_OUT=$TT/experiments/bioreason/batch_go_pred_smoke.out
SMOKE_SH=$TT/experiments/bioreason/batch_go_pred_smoke.sh
PROD_SH=$TT/experiments/bioreason/batch_prod_rl_4n_gopred.sh
PROD_SUBMITTED=$SD/PROD_GOPRED_SUBMITTED
SMOKE_SUBMITTED=$SD/SMOKE_GOPRED_SUBMITTED

qsub_when_free(){  # $1=script ; echo jobid on success, empty on fail
  local out; out=$(cd "$TT" && qsub "$1" 2>&1)
  [[ "$out" == *aurora-pbs* ]] && { echo "$out"; return 0; } || return 1
}

log "orchestrator START. smoke=$SMOKE_SH prod=$PROD_SH"

# ---- Phase 1: ensure the smoke is submitted ----
for i in $(seq 1 360); do
  [ -f "$SMOKE_SUBMITTED" ] && { log "smoke already submitted ($(cat $SMOKE_SUBMITTED))"; break; }
  if jid=$(qsub_when_free "$SMOKE_SH"); then
    echo "$jid" > "$SMOKE_SUBMITTED"; log "SMOKE submitted: $jid"; break
  fi
  log "smoke: no Q slot yet (try $i); sleep 120"; sleep 120
done
[ -f "$SMOKE_SUBMITTED" ] || { log "FATAL: never submitted smoke"; exit 1; }

# ---- Phase 2: wait for smoke to finish ----
log "waiting for smoke completion ($SMOKE_OUT)"
for i in $(seq 1 240); do  # up to 8h
  if [ -f "$SMOKE_OUT" ] && grep -q "go_pred A/B smoke end" "$SMOKE_OUT" 2>/dev/null; then
    log "smoke finished"; break
  fi
  sleep 120
done

# ---- Phase 3: assess smoke health (gopredON leg specifically) ----
# GREEN gate: gopredON leg ran >=4 steps with ratios=1.0 and NO banned:1/Traceback/NaN/
# truncation error, and reward fired (some nonzero). The smoke copies each leg's launcher
# LOG to overnight_state/smoke_<tag>_launcher.log (gopredON = the one we gate on).
GREEN=0
GLOG=$SD/smoke_gopredON_launcher.log
if [ -f "$GLOG" ]; then
  steps=$(grep -cE "METRICS step=[0-9]+|^Step [0-9]+ " "$GLOG" 2>/dev/null | head -1); steps=${steps:-0}
  banned=$(grep -ciE "banned:1|Traceback|UR_RESULT_ERROR|[^a-z]nan[^a-z]|truncat.*error|CUDA error|XPU out of memory" "$GLOG" 2>/dev/null | head -1); banned=${banned:-0}
  nonzero=$(grep -oE "rewards=[0-9.]+" "$GLOG" 2>/dev/null | grep -vE "rewards=0.0+$" | head -1)
  ratios_ok=$(grep -oE "ratios=[0-9.]+" "$GLOG" 2>/dev/null | grep -cE "ratios=1.0" | head -1); ratios_ok=${ratios_ok:-0}
  log "smoke gopredON health: steps=$steps banned/err=$banned nonzero_reward=${nonzero:-none} ratios1.0_count=$ratios_ok log=$GLOG"
  if [ "$steps" -ge 4 ] && [ "$banned" -eq 0 ] && [ -n "$nonzero" ]; then GREEN=1; fi
else
  log "WARN: gopredON launcher log not found at $GLOG; cannot assess smoke"
fi

if [ "$GREEN" -ne 1 ]; then
  log "smoke NOT green (or undetermined) -> NOT auto-launching prod. Leaving for morning review."
  log "orchestrator END (held)"; exit 0
fi
log "smoke GREEN."

# ---- Phase 4: launch prod go_pred run when a Q slot frees ----
[ -f "$PROD_SUBMITTED" ] && { log "prod already submitted ($(cat $PROD_SUBMITTED))"; exit 0; }
for i in $(seq 1 240); do
  if jid=$(qsub_when_free "$PROD_SH"); then
    echo "$jid" > "$PROD_SUBMITTED"; log "PROD go_pred submitted: $jid"; break
  fi
  log "prod: no Q slot yet (try $i); sleep 120"; sleep 120
done
[ -f "$PROD_SUBMITTED" ] && log "orchestrator END (prod launched)" || log "orchestrator END (prod never got a slot)"

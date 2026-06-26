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

# ---- Phase 3: assess smoke health (gopredON leg) ----
# GREEN gate: gopredON leg ran >=4 steps with ratios=1.0 and NO banned:1/Traceback/NaN,
# and reward fired (some nonzero). The launcher's timestamped log holds the Step lines;
# find the most recent train log.
GREEN=0
NEWEST_TRAINLOG=$(ls -t $TT/experiments/bioreason/run_bioreason_2node_*.log 2>/dev/null | head -1)
if [ -n "${NEWEST_TRAINLOG:-}" ]; then
  steps=$(grep -cE "^Step [0-9]+ " "$NEWEST_TRAINLOG" 2>/dev/null | head -1); steps=${steps:-0}
  banned=$(grep -ciE "banned:1|Traceback|UR_RESULT_ERROR|nan" "$NEWEST_TRAINLOG" 2>/dev/null | head -1); banned=${banned:-0}
  nonzero=$(grep -oE "rewards:[0-9.]+" "$NEWEST_TRAINLOG" 2>/dev/null | grep -vE "rewards:0\.0+$" | head -1)
  log "smoke health: steps=$steps banned/err=$banned nonzero_reward_seen=${nonzero:-none} log=$NEWEST_TRAINLOG"
  if [ "$steps" -ge 4 ] && [ "$banned" -eq 0 ] && [ -n "$nonzero" ]; then GREEN=1; fi
else
  log "WARN: no train log found to assess smoke"
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

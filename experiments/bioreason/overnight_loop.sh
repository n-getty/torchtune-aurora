#!/bin/bash
# Overnight autonomous loop for BioReason 2-node testing.
#
# Loop: queue hold (if none queued) → wait for R → run BioReason test → cleanup → repeat.
# Stops after MAX_RUNS or when STOP_FILE is touched.
#
# Each iteration logs to experiments/bioreason/overnight_<TS>.log + run_<TS>.log.
set -euo pipefail

TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
HOLD_SCRIPT=${TT_DIR}/experiments/bioreason/hold_bioreason_2node_debug.sh
LAUNCHER=${TT_DIR}/experiments/bioreason/run_bioreason_2node_server.sh
STATE_DIR=${TT_DIR}/experiments/bioreason/overnight_state
STOP_FILE=${STATE_DIR}/STOP
mkdir -p "${STATE_DIR}"

MAX_RUNS=${MAX_RUNS:-12}
RUN_LOG=${STATE_DIR}/loop_$(date -u +%Y%m%d_%H%M%S).log
exec >>"${RUN_LOG}" 2>&1

ts() { date -u +%Y%m%d_%H%M%S; }
log() { echo "[$(ts)] $*"; }

log "Overnight loop START (max_runs=${MAX_RUNS}); stop_file=${STOP_FILE}"

# Configurable run sweep — each entry: GRPO_SAMPLES,FBS,MAX_GEN,NSTEPS,TAG
# Strategy: confirm baseline (G=4 fbs=4) once, then walk up G with safe FBS,
# then long-horizon stability runs.
SWEEP=(
    "4,4,1024,5,baseline_g4_validate"
    "8,4,1024,5,g8_fbs4"
    "16,2,1024,5,g16_fbs2"
    "24,2,1024,5,g24_fbs2"
    "36,2,1024,5,g36_fbs2"
    "36,1,1024,5,g36_fbs1"
    "16,2,1024,15,g16_fbs2_long"
    "8,4,1024,30,g8_fbs4_stability"
)

ITER=0
for spec in "${SWEEP[@]}"; do
    [[ -f "${STOP_FILE}" ]] && { log "STOP file present, exiting."; exit 0; }
    [[ "${ITER}" -ge "${MAX_RUNS}" ]] && { log "Hit MAX_RUNS=${MAX_RUNS}, exiting."; exit 0; }
    ITER=$((ITER + 1))

    IFS=',' read -r G FBS MAX_GEN NSTEPS TAG <<< "${spec}"
    log "=== ITER ${ITER}/${#SWEEP[@]}: G=${G} FBS=${FBS} MAX_GEN=${MAX_GEN} NSTEPS=${NSTEPS} TAG=${TAG} ==="

    # Find or queue a hold
    JOBID=$(qstat -u "${USER}" 2>/dev/null | awk '/hold_bior/ && ($10 == "Q" || $10 == "R") {print $1}' | head -1 | cut -d'.' -f1)
    if [[ -z "${JOBID}" ]]; then
        log "No bioreason hold found; submitting..."
        SUBMIT=$(qsub "${HOLD_SCRIPT}" 2>&1)
        log "  qsub: ${SUBMIT}"
        JOBID=$(echo "${SUBMIT}" | grep -oE '^[0-9]+' | head -1)
    fi
    log "Using JOBID=${JOBID}"

    # Wait for it to be R
    log "Polling qstat until ${JOBID} is R..."
    while true; do
        STATE=$(qstat -f "${JOBID}" 2>/dev/null | awk -F'= ' '/job_state/ {print $2}' | tr -d ' \n')
        case "${STATE}" in
            R) break ;;
            ""|F|E)
                log "  ${JOBID} is ${STATE:-GONE}; abort iteration"
                continue 2
                ;;
            *) sleep 30 ;;
        esac
    done
    log "  ${JOBID} is R"

    # Look up nodes
    NODES=($(qstat -f "${JOBID}" | awk -F'= ' '/exec_host/ {print $2}' | tr '+' '\n' | cut -d/ -f1 | sort -u))
    if [[ "${#NODES[@]}" -lt 2 ]]; then
        log "  Need >=2 nodes, got: ${NODES[*]}; abort iteration"
        continue
    fi
    TRAIN_NODE=${NODES[0]}
    # PBS aux file lives only on the MoM (head) node. We can't rely on it being
    # on whichever node we SSH into. Synthesize one on the train node from the
    # nodes we already know from qstat.
    NODEFILE="/tmp/torchtune/nodefile_${JOBID}"
    NF_CONTENTS=$(printf '%s\n' "${NODES[@]}")
    ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" "mkdir -p /tmp/torchtune && cat > ${NODEFILE}" <<<"${NF_CONTENTS}"
    log "  TRAIN=${TRAIN_NODE} VLLM=${NODES[1]} NODEFILE=${NODEFILE} (synth: ${NODES[*]})"

    # Pre-launch hygiene
    for N in "${NODES[@]}"; do
        ssh -o StrictHostKeyChecking=no "${N}" "pkill -9 VLLM 2>/dev/null; pkill -9 -f vllm.entrypoints 2>/dev/null; pkill -9 -f grpo_bioreason 2>/dev/null; pkill -9 -f torch.distributed.run 2>/dev/null; true" || true
    done
    sleep 3

    # Launch (synchronous: we want to know when this iteration finishes
    # so the next iteration can run on the next hold).
    LOGFILE=${TT_DIR}/experiments/bioreason/overnight_${TAG}_$(ts).log
    log "  Launching → ${LOGFILE}"
    ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" "PBS_NODEFILE=${NODEFILE} GRPO_SAMPLES=${G} FORWARD_BATCH_SIZE=${FBS} MAX_GEN_TOKENS=${MAX_GEN} NSTEPS=${NSTEPS} bash ${LAUNCHER}" \
        > "${LOGFILE}" 2>&1 &
    SSH_PID=$!
    log "  ssh PID=${SSH_PID}"

    # Wait for launcher to finish, BUT also bail out if the PBS job ends.
    while kill -0 "${SSH_PID}" 2>/dev/null; do
        STATE=$(qstat -f "${JOBID}" 2>/dev/null | awk -F'= ' '/job_state/ {print $2}' | tr -d ' \n')
        if [[ -z "${STATE}" || "${STATE}" == "F" || "${STATE}" == "E" ]]; then
            log "  Job ${JOBID} ended (state=${STATE:-GONE}) while launcher still running; reaping."
            kill -9 "${SSH_PID}" 2>/dev/null || true
            break
        fi
        sleep 30
    done
    wait "${SSH_PID}" 2>/dev/null || true
    log "  Launcher done. Last 20 log lines:"
    tail -20 "${LOGFILE}" | sed 's/^/    /' | tee -a /dev/null

    # Extract step results
    STEPS_OK=$(grep -cE "Step [0-9]+: loss=" "${LOGFILE}" || echo 0)
    log "  steps_completed=${STEPS_OK}"

    # Pre-emptively queue the next hold while we wait (debug-scaling max_run=1, max queued varies)
    qsub "${HOLD_SCRIPT}" 2>&1 | sed 's/^/    /' | tee -a /dev/null || true
done

log "Overnight loop done."

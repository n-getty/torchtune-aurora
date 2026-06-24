#!/bin/bash
# Overnight v2 — refined sweep based on G=24 OOM observation:
# G≥24 OOMs at fbs=2; persistent trajectory accumulators dominate memory.
# Strategy: confirm G=16 (predicted to fit), test G=20 boundary, then
# stability runs at the largest viable G. Also test max_gen=512 to confirm
# trajectory-memory hypothesis.
set -euo pipefail

TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
HOLD_SCRIPT=${TT_DIR}/experiments/bioreason/hold_bioreason_2node_debug.sh
LAUNCHER=${TT_DIR}/experiments/bioreason/run_bioreason_2node_server.sh
STATE_DIR=${TT_DIR}/experiments/bioreason/overnight_state
STOP_FILE=${STATE_DIR}/STOP
mkdir -p "${STATE_DIR}"

MAX_RUNS=${MAX_RUNS:-12}
RUN_LOG=${STATE_DIR}/loopv2_$(date -u +%Y%m%d_%H%M%S).log
exec >>"${RUN_LOG}" 2>&1

ts() { date -u +%Y%m%d_%H%M%S; }
log() { echo "[$(ts)] $*"; }

log "Overnight loop V2 START (max_runs=${MAX_RUNS}); stop=${STOP_FILE}"

# G,FBS,MAX_GEN,NSTEPS,TAG
SWEEP=(
    "16,2,1024,5,v2_g16_fbs2"
    "20,2,1024,5,v2_g20_fbs2_boundary"
    "16,2,1024,15,v2_g16_fbs2_long"
    "12,4,1024,5,v2_g12_fbs4"
    "16,4,1024,5,v2_g16_fbs4"
    "24,2,512,5,v2_g24_mg512"
    "32,2,512,5,v2_g32_mg512"
    "16,2,1024,30,v2_g16_stability30"
)

ITER=0
for spec in "${SWEEP[@]}"; do
    [[ -f "${STOP_FILE}" ]] && { log "STOP file present, exiting."; exit 0; }
    [[ "${ITER}" -ge "${MAX_RUNS}" ]] && { log "Hit MAX_RUNS=${MAX_RUNS}, exiting."; exit 0; }
    ITER=$((ITER + 1))

    IFS=',' read -r G FBS MAX_GEN NSTEPS TAG <<< "${spec}"
    log "=== ITER ${ITER}/${#SWEEP[@]}: G=${G} FBS=${FBS} MAX_GEN=${MAX_GEN} NSTEPS=${NSTEPS} TAG=${TAG} ==="

    JOBID=$(qstat -u "${USER}" 2>/dev/null | awk '/hold_bior/ && ($10 == "Q" || $10 == "R") {print $1}' | head -1 | cut -d'.' -f1)
    if [[ -z "${JOBID}" ]]; then
        log "No bioreason hold; submitting..."
        SUBMIT=$(qsub "${HOLD_SCRIPT}" 2>&1)
        log "  qsub: ${SUBMIT}"
        JOBID=$(echo "${SUBMIT}" | grep -oE '^[0-9]+' | head -1)
    fi
    log "Using JOBID=${JOBID}"

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

    NODES=($(qstat -f "${JOBID}" | awk -F'= ' '/exec_host/ {print $2}' | tr '+' '\n' | cut -d/ -f1 | sort -u))
    if [[ "${#NODES[@]}" -lt 2 ]]; then
        log "  Need >=2 nodes, got: ${NODES[*]}; abort iteration"
        continue
    fi
    TRAIN_NODE=${NODES[0]}
    NODEFILE="/tmp/torchtune/nodefile_${JOBID}"
    NF=$(printf '%s\n' "${NODES[@]}")
    ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" "mkdir -p /tmp/torchtune && cat > ${NODEFILE}" <<<"${NF}"
    log "  TRAIN=${TRAIN_NODE} VLLM=${NODES[1]} synth-NODEFILE=${NODEFILE} contents=${NODES[*]}"

    for N in "${NODES[@]}"; do
        ssh -o StrictHostKeyChecking=no "${N}" "pkill -9 VLLM 2>/dev/null; pkill -9 -f vllm.entrypoints 2>/dev/null; pkill -9 -f grpo_bioreason 2>/dev/null; pkill -9 -f torch.distributed.run 2>/dev/null; true" || true
    done
    sleep 5

    LOGFILE=${TT_DIR}/experiments/bioreason/overnight_${TAG}_$(ts).log
    log "  Launching → ${LOGFILE}"
    ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" "PBS_NODEFILE=${NODEFILE} GRPO_SAMPLES=${G} FORWARD_BATCH_SIZE=${FBS} MAX_GEN_TOKENS=${MAX_GEN} NSTEPS=${NSTEPS} bash ${LAUNCHER}" \
        > "${LOGFILE}" 2>&1 &
    SSH_PID=$!
    log "  ssh PID=${SSH_PID}"

    while kill -0 "${SSH_PID}" 2>/dev/null; do
        STATE=$(qstat -f "${JOBID}" 2>/dev/null | awk -F'= ' '/job_state/ {print $2}' | tr -d ' \n')
        if [[ -z "${STATE}" || "${STATE}" == "F" || "${STATE}" == "E" ]]; then
            log "  Job ${JOBID} ended (state=${STATE:-GONE}); reaping ssh"
            kill -9 "${SSH_PID}" 2>/dev/null || true
            break
        fi
        sleep 30
    done
    wait "${SSH_PID}" 2>/dev/null || true

    STEPS_OK=$(grep -cE "Step [0-9]+: loss=" "${LOGFILE}" || echo 0)
    OOM_LINES=$(grep -cE "out of memory|OutOfMemoryError" "${LOGFILE}" || echo 0)
    TILE_TIMEOUT=$(grep -c "vLLM tile.*not ready within" "${LOGFILE}" || echo 0)
    log "  steps_completed=${STEPS_OK} oom=${OOM_LINES} vllm_tile_timeout=${TILE_TIMEOUT}"

    qsub "${HOLD_SCRIPT}" 2>&1 | sed 's/^/    /' || true
done

log "Overnight loop V2 done."

#!/bin/bash
# Overnight v3 — re-run configs lost to PBS Exit_status=-3 (Aurora infra
# failures during 2026-04-30 06:00-06:04 UTC window killed ITERs 4-7 of v2).
# Adds RETRY_ON_PROLOGUE_FAIL: if ssh launcher exits in <90s with 0 step output,
# treat as infra failure and re-attempt on the next hold instead of advancing.
set -euo pipefail

TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
HOLD_SCRIPT=${TT_DIR}/experiments/bioreason/hold_bioreason_2node_debug.sh
LAUNCHER=${TT_DIR}/experiments/bioreason/run_bioreason_2node_server.sh
STATE_DIR=${TT_DIR}/experiments/bioreason/overnight_state
STOP_FILE=${STATE_DIR}/STOP
mkdir -p "${STATE_DIR}"

MAX_RUNS=${MAX_RUNS:-20}
MIN_RUNTIME_S=${MIN_RUNTIME_S:-90}
MAX_RETRIES_PER_SPEC=${MAX_RETRIES_PER_SPEC:-3}
RUN_LOG=${STATE_DIR}/loopv3_$(date -u +%Y%m%d_%H%M%S).log
exec >>"${RUN_LOG}" 2>&1

ts() { date -u +%Y%m%d_%H%M%S; }
log() { echo "[$(ts)] $*"; }

log "Overnight loop V3 START (max_runs=${MAX_RUNS}, retries=${MAX_RETRIES_PER_SPEC}); stop=${STOP_FILE}"

# Configs lost in v2 + critical reruns
SWEEP=(
    "16,2,1024,15,v3_g16_fbs2_long15"
    "12,4,1024,5,v3_g12_fbs4"
    "16,4,1024,5,v3_g16_fbs4_boundary"
    "24,2,512,5,v3_g24_mg512"
    "32,2,512,5,v3_g32_mg512"
    "16,2,1024,30,v3_g16_stability30"
)

ITER=0
for spec in "${SWEEP[@]}"; do
    [[ -f "${STOP_FILE}" ]] && { log "STOP file present, exiting."; exit 0; }
    [[ "${ITER}" -ge "${MAX_RUNS}" ]] && { log "Hit MAX_RUNS=${MAX_RUNS}, exiting."; exit 0; }

    IFS=',' read -r G FBS MAX_GEN NSTEPS TAG <<< "${spec}"

    RETRY=0
    while [[ ${RETRY} -lt ${MAX_RETRIES_PER_SPEC} ]]; do
        ITER=$((ITER + 1))
        RETRY=$((RETRY + 1))
        log "=== ITER ${ITER} (try ${RETRY}/${MAX_RETRIES_PER_SPEC}): G=${G} FBS=${FBS} MAX_GEN=${MAX_GEN} NSTEPS=${NSTEPS} TAG=${TAG} ==="

        # Prefer an R hold; fall back to Q; submit fresh if neither exists.
        JOBID=$(qstat -u "${USER}" 2>/dev/null | awk '/hold_bior/ && $10 == "R" {print $1}' | head -1 | cut -d'.' -f1)
        if [[ -z "${JOBID}" ]]; then
            JOBID=$(qstat -u "${USER}" 2>/dev/null | awk '/hold_bior/ && $10 == "Q" {print $1}' | head -1 | cut -d'.' -f1)
        fi
        if [[ -z "${JOBID}" ]]; then
            log "No bioreason hold; submitting..."
            SUBMIT=$(qsub "${HOLD_SCRIPT}" 2>&1)
            log "  qsub: ${SUBMIT}"
            JOBID=$(echo "${SUBMIT}" | grep -oE '^[0-9]+' | head -1)
        fi
        log "Using JOBID=${JOBID}"

        log "Polling qstat until ${JOBID} is R..."
        WAIT_SECS=0
        while true; do
            STATE=$(qstat -f "${JOBID}" 2>/dev/null | awk -F'= ' '/job_state/ {print $2}' | tr -d ' \n')
            case "${STATE}" in
                R) break ;;
                ""|F|E)
                    log "  ${JOBID} is ${STATE:-GONE}; will retry"
                    sleep 10
                    continue 2  # next retry of same spec
                    ;;
                *) sleep 30 ; WAIT_SECS=$((WAIT_SECS + 30)) ;;
            esac
            if [[ ${WAIT_SECS} -ge 1800 ]]; then
                log "  Timed out waiting 30min for ${JOBID}; bailing this spec"
                continue 3  # next spec
            fi
        done
        log "  ${JOBID} is R"

        NODES=($(qstat -f "${JOBID}" | awk -F'= ' '/exec_host/ {print $2}' | tr '+' '\n' | cut -d/ -f1 | sort -u))
        if [[ "${#NODES[@]}" -lt 2 ]]; then
            log "  Need >=2 nodes, got: ${NODES[*]}; retry"
            continue
        fi
        TRAIN_NODE=${NODES[0]}
        NODEFILE="/tmp/torchtune/nodefile_${JOBID}"
        NF=$(printf '%s\n' "${NODES[@]}")
        { ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" "mkdir -p /tmp/torchtune && cat > ${NODEFILE}" <<<"${NF}" 2>&1 | head -3 | sed 's/^/    /'; } || true
        log "  TRAIN=${TRAIN_NODE} VLLM=${NODES[1]} synth-NODEFILE=${NODEFILE}"

        for N in "${NODES[@]}"; do
            { ssh -o StrictHostKeyChecking=no "${N}" "pkill -9 VLLM 2>/dev/null; pkill -9 -f vllm.entrypoints 2>/dev/null; pkill -9 -f grpo_bioreason 2>/dev/null; pkill -9 -f torch.distributed.run 2>/dev/null; true" 2>&1 | head -3 | sed 's/^/    /'; } || true
        done
        sleep 5

        LOGFILE=${TT_DIR}/experiments/bioreason/overnight_${TAG}_$(ts).log
        log "  Launching → ${LOGFILE}"
        LAUNCH_START=$(date +%s)
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
        LAUNCH_END=$(date +%s)
        ELAPSED=$((LAUNCH_END - LAUNCH_START))

        STEPS_OK=$(grep -cE "Step [0-9]+: loss=" "${LOGFILE}" 2>/dev/null || echo 0)
        OOM_LINES=$(grep -cE "out of memory|OutOfMemoryError" "${LOGFILE}" 2>/dev/null || echo 0)
        TILE_TIMEOUT=$(grep -c "vLLM tile.*not ready within" "${LOGFILE}" 2>/dev/null || echo 0)
        log "  steps=${STEPS_OK} oom=${OOM_LINES} tile_to=${TILE_TIMEOUT} elapsed=${ELAPSED}s"

        # If launcher died too quickly with 0 steps and no OOM, treat as infra failure → retry
        if [[ ${ELAPSED} -lt ${MIN_RUNTIME_S} && ${STEPS_OK} -eq 0 && ${OOM_LINES} -eq 0 ]]; then
            log "  INFRA FAIL: ssh exited in ${ELAPSED}s (<${MIN_RUNTIME_S}s) with 0 steps and no OOM; retry on fresh hold"
            qsub "${HOLD_SCRIPT}" 2>&1 | sed 's/^/    /' || true
            continue  # retry same spec
        fi

        # Spec done (success or real failure like OOM)
        log "  Spec ${TAG} concluded after ${RETRY} attempt(s)"
        qsub "${HOLD_SCRIPT}" 2>&1 | sed 's/^/    /' || true
        break
    done
done

log "Overnight loop V3 done."

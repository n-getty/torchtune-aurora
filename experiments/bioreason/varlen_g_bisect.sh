#!/bin/bash
# G-bisect for IPEX varlen: run G=12 NSTEPS=10, then G=16 NSTEPS=10 if G=12 clean.
# Detect banned:1 PDE Segfault by grepping for "banned:1" in the log.
# Run from a held PBS job (PBS_NODEFILE must be set).
set -u

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set." >&2
    exit 1
fi

LAUNCHER=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_bioreason_2node_server.sh
LOGDIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason
TS=$(date +%Y%m%d_%H%M%S)
DRIVER_LOG="${LOGDIR}/varlen_gbisect_${TS}.log"

run_one() {
    local G=$1
    local NSTEPS=$2
    local FBS=$3
    local TAG=$4
    echo "" | tee -a "${DRIVER_LOG}"
    echo "─── ${TAG}: G=${G} fbs=${FBS} NSTEPS=${NSTEPS} ───" | tee -a "${DRIVER_LOG}"
    local LOG="${LOGDIR}/varlen_${TAG}_${TS}.log"
    local T0=$(date +%s)
    TORCHTUNE_USE_IPEX_VARLEN=1 \
        GRPO_SAMPLES=${G} \
        NSTEPS=${NSTEPS} \
        FORWARD_BATCH_SIZE=${FBS} \
        MAX_GEN_TOKENS=1024 \
        bash "${LAUNCHER}" > "${LOG}" 2>&1
    local RC=$?
    local DT=$(($(date +%s) - T0))
    local LAST_STEP=$(grep -oE "Step [0-9]+:" "${LOG}" | tail -1 | tr -d 'Step :' )
    # CCL log writes "banned: 1" (with space). The PDE crash always pairs with this token.
    local BANNED1=$(grep -cE "banned:[[:space:]]*1" "${LOG}" 2>/dev/null || echo 0)
    local SEGFAULT=$(grep -c "Segmentation fault from GPU" "${LOG}" 2>/dev/null || echo 0)
    echo "  log:        ${LOG}" | tee -a "${DRIVER_LOG}"
    echo "  rc=${RC}  elapsed=${DT}s  last_step=${LAST_STEP:-none}  banned1=${BANNED1}  segfault=${SEGFAULT}" | tee -a "${DRIVER_LOG}"
    # Surface key error lines
    grep -E "RuntimeError|banned:1|Segfault|out of memory|OOM|ECCL" "${LOG}" 2>/dev/null \
      | tail -5 | sed 's/^/    /' | tee -a "${DRIVER_LOG}"
    # Return clean (0) or fail (1)
    if [[ ${RC} -eq 0 && ${BANNED1} -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

echo "=== Varlen G-bisect driver ===" | tee -a "${DRIVER_LOG}"
echo "Hold: $(echo $PBS_JOBID)" | tee -a "${DRIVER_LOG}"
echo "Date: $(date)" | tee -a "${DRIVER_LOG}"

# Run A: G=12 NSTEPS=10 fbs=4
if run_one 12 10 4 "G12_fbs4"; then
    echo "G=12 CLEAN — proceeding to G=16" | tee -a "${DRIVER_LOG}"
    # Run B: G=16 NSTEPS=10 fbs=4
    if run_one 16 10 4 "G16_fbs4"; then
        echo "G=16 CLEAN — varlen ceiling pushed past G=16" | tee -a "${DRIVER_LOG}"
    else
        echo "G=16 FAILED — ceiling is between G=12 and G=16" | tee -a "${DRIVER_LOG}"
    fi
else
    echo "G=12 FAILED — ceiling is between G=8 and G=12" | tee -a "${DRIVER_LOG}"
fi

echo "=== Done ===" | tee -a "${DRIVER_LOG}"

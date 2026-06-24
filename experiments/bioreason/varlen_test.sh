#!/bin/bash
# IPEX varlen_attention live test on BioReason 2-node (hold 8460388, debug, 1h).
#
# Setup: TORCHTUNE_USE_IPEX_VARLEN=1 enables varlen path in attention_utils.
# Bit-exact vs PyTorch SDPA validated 2026-04-30 (max_diff=0.0 on bf16).
# Bench (single-tile): 27% faster, 64% lower peak/transient memory per call.
#
# Hypothesis under test: lower per-chunk transient allocations → fewer L0 IPC
# handle registrations during FSDP collectives → G>4 banned:1 crash step pushed
# out beyond step 13 (current G=8 ceiling on optimized SDPA).
#
# Test sequence:
#   A: G=4 NSTEPS=5 with varlen — sanity baseline, must match wsync-fix run timing.
#   B: G=8 NSTEPS=15 with varlen — does it survive past step 13?
#   C: only if B clean: G=12 NSTEPS=10 to find new ceiling.
#
# After ANY banned:1, hold is L0-corrupted — STOP. New hold required.

set -o pipefail

TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${SCRIPT_DIR}/varlen_test_${STAMP}.summary"

echo "=== IPEX varlen_attention live test ===" | tee "${SUMMARY}"
echo "Hold: $(qstat -u $USER | grep -E 'debug ' | awk '{print $1}')" | tee -a "${SUMMARY}"
echo "Date: $(date)" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

run_one() {
    local label="$1"
    local g="$2"
    local nsteps="$3"

    local logfile="${SCRIPT_DIR}/varlen_${label}_${STAMP}.log"
    echo "─── Run ${label}: G=${g} NSTEPS=${nsteps} TORCHTUNE_USE_IPEX_VARLEN=1 ───" | tee -a "${SUMMARY}"
    echo "  log: ${logfile}" | tee -a "${SUMMARY}"

    local t0=$(date +%s)
    GRPO_SAMPLES="${g}" \
    NSTEPS="${nsteps}" \
    FORWARD_BATCH_SIZE="${g}" \
    MAX_GEN_TOKENS=1024 \
    TORCHTUNE_USE_IPEX_VARLEN=1 \
    bash "${SCRIPT_DIR}/run_bioreason_2node_server.sh" > "${logfile}" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))

    local last_step=$(grep -oP "PRE-STEP \K\d+" "${logfile}" | tail -1)
    local mem_lines=$(grep "PRE-STEP" "${logfile}" | tail -3)
    local banned=$(grep -c "banned: 1" "${logfile}" 2>/dev/null || echo 0)
    local complete=$(grep -c "training complete" "${logfile}" 2>/dev/null || echo 0)

    echo "  rc=${rc} elapsed=${elapsed}s last_step=${last_step:-none} banned1=${banned} complete=${complete}" | tee -a "${SUMMARY}"
    echo "  last 3 PRE-STEP lines:" | tee -a "${SUMMARY}"
    echo "${mem_lines}" | sed 's/^/    /' | tee -a "${SUMMARY}"
    echo "" | tee -a "${SUMMARY}"

    if [ "${banned}" -gt 0 ]; then
        echo "  → banned:1 detected. Hold is L0-corrupted. STOP." | tee -a "${SUMMARY}"
        return 99
    fi
    return $rc
}

run_one A 4 5
RC_A=$?
[ $RC_A -eq 99 ] && exit 0

run_one B 8 15
RC_B=$?
[ $RC_B -eq 99 ] && exit 0

if [ $RC_A -eq 0 ] && [ $RC_B -eq 0 ]; then
    run_one C 12 10 || true
fi

echo "=== Done ===" | tee -a "${SUMMARY}"

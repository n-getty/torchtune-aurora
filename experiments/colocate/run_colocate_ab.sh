#!/bin/bash
# In-framework A/B matrix for the colocate vLLM generation page fault.
#
# Wraps experiments/lora_grpo/run_lora_colocate.sh, runs each A/B CELL M times back-to-back on
# the SAME node (controls node variance), classifies each run.log (crash vs clean), health-gates
# it, and appends crash-count/M to a TSV. Each cell varies ONE independent variable vs baseline.
#
# Cells (set CELL=...):
#   baseline   : default FSDP colocate, publish_every=1, mg=${MAX_GEN}            (reference rate)
#   nofsdp     : TORCHTUNE_COLOCATE_NO_FSDP=1 RECLAIM_MODE=all  (H1: empty_cache reclaim active)
#   noreclaim  : TORCHTUNE_COLOCATE_NO_FSDP=1 RECLAIM_STRIDE=99999 (no-FSDP but ~never empty_cache)
#   noreset    : TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX=1          (H1: reset_prefix_cache trigger)
#   pub999     : lora.publish_every_steps=999 (one step-0 publish only)          (H1: load_weights)
#   bigkv      : larger num_gpu_blocks_override + gpu_mem 0.55 (vLLM owns non-overlapping KV)
#   server     : control — vllm_mode=server at same mg (expected 0 crashes; confirms in-process axis)
#
# Usage (on a held node, PBS_NODEFILE set to the real single-node nodefile):
#   export PBS_NODEFILE=/path/to/nodefile
#   CELL=baseline M=8 MAX_GEN=768 bash experiments/colocate/run_colocate_ab.sh
#   for c in baseline noreset pub999 bigkv nofsdp noreclaim; do
#       CELL=$c M=8 MAX_GEN=768 bash experiments/colocate/run_colocate_ab.sh; done
# no `set -u`: empty CELL_ENV[@] array + the downstream _env.sh source both trip it.
set -o pipefail

REPO_ROOT="/lus/flare/projects/ModCon/ngetty/torchtune"
EXPDIR="${REPO_ROOT}/experiments/colocate"
COLO="${REPO_ROOT}/experiments/lora_grpo/run_lora_colocate.sh"

CELL="${CELL:-baseline}"
M="${M:-8}"
MAX_GEN="${MAX_GEN:-768}"
NSTEPS="${NSTEPS:-15}"
MODEL_PATH="${MODEL_PATH:-/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B}"
NODE="$(hostname)"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_TSV="${EXPDIR}/ab_results.tsv"

if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "${PBS_NODEFILE}" ]; then
    echo "ERROR: PBS_NODEFILE must point at a valid single-node nodefile." >&2
    exit 1
fi

# --- per-cell env / overrides ---
CELL_ENV=()
VLLM_MODE_OVERRIDE=""           # only 'server' cell changes mode (handled via EXTRA_OVERRIDES)
EXTRA_OVR=""
case "${CELL}" in
    baseline)   ;;
    nofsdp)     CELL_ENV=(TORCHTUNE_COLOCATE_NO_FSDP=1 TORCHTUNE_COLOCATE_RECLAIM_MODE=all) ;;
    noreclaim)  CELL_ENV=(TORCHTUNE_COLOCATE_NO_FSDP=1 TORCHTUNE_COLOCATE_RECLAIM_STRIDE=99999) ;;
    # CANDIDATE FIX: barrier+sync around the colocate load_weights so no XCCL collective overlaps
    # the weight-publish mutation (removes factor (b) of the 2-factor fault). If this → 0/N, it's
    # a real recipe-side mitigation.
    quiesce)    CELL_ENV=(TORCHTUNE_COLOCATE_QUIESCE_WSYNC=1) ;;
    # CANDIDATE FIX 2: KV-only sleep/wake around the colocate load_weights — removes vLLM's KV/L0
    # paging state during the weight mutation (a barrier didn't help; this resets the engine's L0).
    sleep)      CELL_ENV=(TORCHTUNE_COLOCATE_SLEEP_WSYNC=1) ;;
    noreset)    CELL_ENV=(TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX=1) ;;
    pub999)     EXTRA_OVR="lora.publish_every_steps=999" ;;
    bigkv)      EXTRA_OVR="vllm_num_gpu_blocks_override=4000 vllm_gpu_memory_utilization=0.55" ;;
    # IPC-handle cache threshold sweep — the step-3 ADAPTER_AR XCCL explosion (39-41s, 5/5 runs)
    # then banned:1 implicates L0 IPC-handle accumulation under vLLM+trainer co-tenancy. _env.sh
    # sets 65536 (high → accumulates ~10.85 GiB). Test bounding it lower (CLAUDE.md warns the
    # =1000 default EVICTS → banned the other way, so probe middle values, not just 1000).
    ccl_mid)    CELL_ENV=(CCL_ZE_CACHE_THRESHOLD_OVERRIDE=8192) ;;
    ccl_low)    CELL_ENV=(CCL_ZE_CACHE_THRESHOLD_OVERRIDE=2048) ;;
    ccl_hi)     CELL_ENV=(CCL_ZE_CACHE_THRESHOLD_OVERRIDE=262144) ;;
    server)
        echo "NOTE: 'server' cell requires the server-mode config + vLLM tiles; skipping unless"
        echo "      run via run_qwen3_4b_lora_2node.sh. This driver only covers colocate cells."
        exit 2 ;;
    *) echo "unknown CELL=${CELL}" >&2; exit 2 ;;
esac

echo "=== A/B cell=${CELL} M=${M} mg=${MAX_GEN} nsteps=${NSTEPS} node=${NODE} ==="
echo "    env: ${CELL_ENV[*]:-none} | overrides: ${EXTRA_OVR:-none}"

crashed=0
clean=0
declare -a crash_steps=()

for run in $(seq 1 "${M}"); do
    echo "--- ${CELL} run ${run}/${M} ---"
    DMESG_BEFORE="$(dmesg 2>/dev/null | wc -l)"
    # Launch the colocate run with this cell's env + overrides. run_lora_colocate.sh
    # makes its own timestamped RUN_DIR and prints it; capture the log path.
    OUT="$(env "${CELL_ENV[@]}" \
        MODEL_PATH="${MODEL_PATH}" NSTEPS="${NSTEPS}" MAX_GEN="${MAX_GEN}" \
        EXTRA_OVERRIDES="${EXTRA_OVR}" PBS_NODEFILE="${PBS_NODEFILE}" \
        bash "${COLO}" 2>&1)"
    RC=$?
    LOG="$(echo "${OUT}" | grep -oE '/[^ ]+/run\.log' | head -1)"
    [ -z "${LOG}" ] && LOG="(unknown)"

    # Classify: crash if rc!=0 OR fault signature in the recipe log OR new dmesg fault.
    sig=0
    if [ -f "${LOG}" ]; then
        sig="$(grep -ciE 'banned:1|NotPresent|UR_RESULT_ERROR|Segmentation fault from GPU|SIGABRT' "${LOG}" 2>/dev/null | head -1 | tr -dc '0-9')"; sig="${sig:-0}"
    fi
    DMESG_NEW="$(dmesg 2>/dev/null | wc -l)"
    dmesg_fault="$(dmesg 2>/dev/null | tail -n $((DMESG_NEW - DMESG_BEFORE)) | grep -ciE 'banned|GPU HANG' 2>/dev/null | head -1 | tr -dc '0-9')"; dmesg_fault="${dmesg_fault:-0}"

    if [[ "${RC}" != "0" || "${sig}" != "0" || "${dmesg_fault}" != "0" ]]; then
        crashed=$((crashed + 1))
        laststep="$(grep -oE 'METRICS step=[0-9]+' "${LOG}" 2>/dev/null | tail -1 | grep -oE '[0-9]+')"
        crash_steps+=("${laststep:-0}")
        echo "  -> CRASH (rc=${RC} sig=${sig} dmesg=${dmesg_fault} last_step=${laststep:-?}) log=${LOG}"
    else
        clean=$((clean + 1))
        echo "  -> clean log=${LOG}"
    fi
    # Health gate (records degraded-path even on clean runs).
    [ -f "${LOG}" ] && bash "${REPO_ROOT}/scripts/check_run_health.sh" "${LOG}" >/dev/null 2>&1 \
        && echo "  health: GREEN" || echo "  health: DEGRADED/!GREEN (see log)"
done

med="na"
if [[ ${#crash_steps[@]} -gt 0 ]]; then
    sorted=($(printf '%s\n' "${crash_steps[@]}" | sort -n))
    med="${sorted[$((${#sorted[@]} / 2))]}"
fi
echo "=== RESULT cell=${CELL}: crashed=${crashed}/${M} clean=${clean} median_crash_step=${med} ==="
printf '%s\t%s\t%s\t%d\t%d\t%s\tmg%s\tnsteps%s\n' \
    "${TS}" "${NODE}" "${CELL}" "${crashed}" "${M}" "${med}" "${MAX_GEN}" "${NSTEPS}" >> "${RESULTS_TSV}"
echo "appended to ${RESULTS_TSV}"

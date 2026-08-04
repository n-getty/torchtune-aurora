#!/bin/bash
# Per-rank wrapper for the multi-tile standalone reproducer (mirrors the bake-off
# _rank_wrapper.sh): resolve rank/affinity from PALS/PMI, pin ZE_AFFINITY_MASK, exec python.
set -e
export RANK="${PMI_RANK:-${PALS_RANKID:-${RANK:-0}}}"
export LOCAL_RANK="${PMI_LOCAL_RANK:-${PALS_LOCAL_RANKID:-${LOCAL_RANK:-0}}}"
export LOCAL_WORLD_SIZE="${PMI_LOCAL_SIZE:-${PALS_LOCAL_SIZE:-${LOCAL_WORLD_SIZE:-12}}}"
export WORLD_SIZE="${PMI_SIZE:-${PALS_NRANKS:-${WORLD:-${WORLD_SIZE}}}}"
if [[ -z "${WORLD_SIZE}" ]]; then
    echo "[mt-wrapper] FATAL: WORLD_SIZE unresolved" >&2; exit 1
fi
export ZE_AFFINITY_MASK="${LOCAL_RANK}"
[[ "${RANK}" == "0" ]] && echo "[mt-wrapper r0] node=$(hostname) WORLD=${WORLD_SIZE} LOCAL_WORLD=${LOCAL_WORLD_SIZE}"
# Per-rank log path for the zeContext diagnostic shim (claim #1 test, legacy custom-C path),
# if requested. Superseded by the vendor-official ZEL_LOADER_LOG_FILE path below (safer: no
# custom code, no risk of a broken symbol hook crashing the process).
if [[ -n "${ZECTX_LOG_DIR:-}" ]]; then
    export ZECTX_LOG="${ZECTX_LOG_DIR}/r${RANK}.log"
fi
if [[ -n "${ZEL_LOADER_LOG_DIR:-}" ]]; then
    export ZEL_LOADER_LOG_FILE="${ZEL_LOADER_LOG_DIR}/r${RANK}.log"
fi
# SYCL_UR_TRACE prints to stderr, not a configurable log file. Tee (not redirect) so the
# fault abort signature still reaches the driver's combined 2>&1 stream for crash detection.
if [[ -n "${SYCL_UR_TRACE_DIR:-}" ]]; then
    exec python3 "$@" 2> >(tee "${SYCL_UR_TRACE_DIR}/r${RANK}.stderr.log" >&2)
fi
exec python3 "$@"

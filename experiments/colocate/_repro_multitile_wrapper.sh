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
exec python3 "$@"

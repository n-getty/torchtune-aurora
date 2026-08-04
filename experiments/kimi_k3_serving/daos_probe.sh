#!/usr/bin/env bash
set -euo pipefail

POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-serving_models}
MNT=${MNT:-/tmp/${POOL}/${CONT}}
LOGIN_ONLY=${DAOS_LOGIN_ONLY:-0}

if [[ "${1:-}" == "--login" ]]; then
    LOGIN_ONLY=1
fi

usage() {
    echo "Usage: $0 [--login]"
    echo "  --login  inspect DAOS from a login node without PBS checks or mounting"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--login" ) ]]; then
    usage >&2
    exit 2
fi

if [[ "$LOGIN_ONLY" != 1 ]]; then
    [[ -n "${PBS_JOBID:-}" ]] || { echo "ERROR: run inside a PBS allocation (or use --login)" >&2; exit 1; }
    [[ -n "${PBS_NODEFILE:-}" ]] || { echo "ERROR: PBS_NODEFILE is required" >&2; exit 1; }

    qstat_output=$(qstat -f "$PBS_JOBID" 2>/dev/null) || {
        echo "ERROR: unable to query PBS job $PBS_JOBID" >&2
        exit 1
    }
    job_state=$(awk -F'= ' '/job_state/ {print $2; exit}' <<<"$qstat_output")
    [[ "$job_state" == R ]] || {
        echo "ERROR: allocation $PBS_JOBID is not running (state=${job_state:-unknown})" >&2
        exit 1
    }

    job_queue=$(awk -F'= ' '/^[[:space:]]*queue/ {print $2; exit}' <<<"$qstat_output")
    [[ "$job_queue" == debug || "$job_queue" == debug-scaling ]] || {
        echo "ERROR: DAOS probe requires queue=debug or debug-scaling, got ${job_queue:-unknown}" >&2
        exit 1
    }
fi

module use /soft/modulefiles
module load daos
module load mpifileutils
command -v daos >/dev/null || { echo "ERROR: daos command is unavailable" >&2; exit 1; }
command -v launch-dfuse-with-caching.sh >/dev/null || {
    echo "ERROR: launch-dfuse-with-caching.sh is unavailable" >&2
    exit 1
}

echo "Checking existing DAOS container ${POOL}:${CONT} (read-only)"
LIST_LOG=/tmp/daos_container_list.${PBS_JOBID:-login}
daos container list "$POOL" | tee "$LIST_LOG"
if ! daos container list "$POOL" | awk -v cont="$CONT" '$2 == cont {found=1} END {exit !found}'; then
    echo "DAOS container ${POOL}:${CONT} is absent; refusing to create it" >&2
    exit 3
fi

[[ "$LOGIN_ONLY" == 1 ]] && {
    echo "DAOS login-node inventory passed: ${POOL}:${CONT} exists"
    exit 0
}

mkdir -p "$MNT"
if ! mount | grep -Fq " $MNT "; then
    launch-dfuse-with-caching.sh "${POOL}:${CONT}"
fi

timeout 60 ls -la "$MNT"
mapfile -t entries < <(ls -1A "$MNT" | sort | head -20)
printf 'entries=%s\n' "${entries[*]:-<empty>}"
echo "DAOS read-only probe passed: ${POOL}:${CONT} mounted at ${MNT}"

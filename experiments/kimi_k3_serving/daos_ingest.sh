#!/usr/bin/env bash
set -euo pipefail

# Ingest a model into an existing POSIX DAOS/dfuse mount.

SRC=${SRC:?Set SRC to the source model directory}
MNT=${MNT:?Set MNT to the mounted dfuse destination}
N_RANKS=${N_RANKS:-0}
PPN=${PPN:-12}
BUF_SIZE=${BUF_SIZE:-64MB}
VERIFY_CHECKPOINT=${VERIFY_CHECKPOINT:-0}
CHECKPOINT_VERIFIER=${CHECKPOINT_VERIFIER:-$(dirname "${BASH_SOURCE[0]}")/verify_checkpoint.py}

[[ -d "$SRC" ]] || { echo "ERROR: source directory does not exist: $SRC" >&2; exit 1; }
[[ -d "$MNT" ]] || { echo "ERROR: dfuse mount does not exist: $MNT" >&2; exit 1; }
if [[ "$VERIFY_CHECKPOINT" == 1 ]]; then
    [[ -f "$CHECKPOINT_VERIFIER" ]] || {
        echo "ERROR: checkpoint verifier does not exist: $CHECKPOINT_VERIFIER" >&2
        exit 1
    }
    python3 "$CHECKPOINT_VERIFIER" "$SRC"
fi
command -v dsync >/dev/null || { echo "ERROR: dsync is required" >&2; exit 1; }
command -v mpiexec >/dev/null || { echo "ERROR: mpiexec is required" >&2; exit 1; }
timeout 60 ls "$MNT" >/dev/null
[[ -n "${PBS_JOBID:-}" ]] || { echo "ERROR: PBS_JOBID is required" >&2; exit 1; }
if command -v qstat >/dev/null; then
    qstat_output=$(qstat -f "$PBS_JOBID" 2>/dev/null) || {
        echo "ERROR: unable to query PBS job $PBS_JOBID" >&2
        exit 1
    }
    job_state=$(awk -F'= ' '/job_state/ {print $2; exit}' <<<"$qstat_output")
    [[ "$job_state" == R ]] || { echo "ERROR: allocation $PBS_JOBID is not running" >&2; exit 1; }
fi

if [[ "$N_RANKS" == 0 ]]; then
    [[ -n "${PBS_NODEFILE:-}" ]] || { echo "ERROR: PBS_NODEFILE required" >&2; exit 1; }
    N_RANKS=$(( $(sort -u "$PBS_NODEFILE" | wc -l) * PPN ))
fi

echo "Ingesting $SRC -> $MNT with $N_RANKS ranks"
mpiexec -n "$N_RANKS" -ppn "$PPN" --cpu-bind none --no-vni \
    dsync --progress 30 --bufsize "$BUF_SIZE" --dereference "$SRC" "$MNT"
timeout 60 ls "$MNT" >/dev/null
echo "Ingest complete"

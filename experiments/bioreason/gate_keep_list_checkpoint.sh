#!/bin/bash
# Gate a keep-list checkpoint before an expensive F_max evaluation.
# Usage: gate_keep_list_checkpoint.sh <run_log> <ckpt_dir> [proj_dir] [num_proteins]
set -euo pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${1:?usage: gate_keep_list_checkpoint.sh <run_log> <ckpt_dir> [proj_dir] [num_proteins]}
CKPT_DIR=${2:?usage: gate_keep_list_checkpoint.sh <run_log> <ckpt_dir> [proj_dir] [num_proteins]}
PROJ_DIR=${3:-$CKPT_DIR}
NUM_PROTEINS=${4:-5}

if [ ! -f "${LOG}" ]; then
    echo "missing run log: ${LOG}" >&2
    exit 2
fi
if [ ! -d "${CKPT_DIR}" ]; then
    echo "missing checkpoint directory: ${CKPT_DIR}" >&2
    exit 2
fi

"${TT}/scripts/check_run_health.sh" "${LOG}"
export BIOREASON_SRC=${BIOREASON_SRC:-/lus/flare/projects/ModCon/ngetty/BioReason-Pro}
export BIOREASON_DEPS=${BIOREASON_DEPS:-/lus/flare/projects/ModCon/ngetty/bioreason_deps}
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${TT}"

python3 "${TT}/experiments/bioreason/probe_keep_list_accuracy.py" \
    --ckpt_dir "${CKPT_DIR}" --proj_dir "${PROJ_DIR}" \
    --num_proteins "${NUM_PROTEINS}"

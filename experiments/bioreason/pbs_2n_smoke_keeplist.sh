#!/bin/bash
#PBS -N br_s1norm_keeplist_2n_smoke
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_2n_smoke_keeplist.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_2n_smoke_keeplist.err
#
# Safe 32B keep-list smoke. A one-node 32B FSDP2 run is not used here: the
# documented single-node envelope accumulates CCL IPC memory and reaches
# banned:1/OOM, while 2-node HSDP is the validated smoke topology.
set -eo pipefail
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "${PROJDIR}"
export CONFIG=recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_keeplist_xpu.yaml
export OUTDIR="${OUTDIR:-${PROJDIR}/experiments/bioreason/runs/sft_qwen3_32b_stage1norm_keeplist_smoke}"
export EPOCHS=1
export SAVE_EVERY="${SAVE_EVERY:-10}"
export MAX_STEPS="${MAX_STEPS:-15}"
export BIOREASON_RESUME="${BIOREASON_RESUME:-0}"
export COMPILE_MODEL="${COMPILE_MODEL:-true}"
exec bash "${PROJDIR}/experiments/bioreason/pbs_4n_sft_full.sh"

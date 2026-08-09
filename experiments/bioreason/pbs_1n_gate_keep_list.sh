#!/bin/bash
#PBS -N br_keeplist_gate
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_1n_gate_keep_list.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_1n_gate_keep_list.err
set -eo pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
CKPT=${CKPT:-${TT}/experiments/bioreason/runs/sft_qwen3_32b_stage1norm_keeplist_smoke/epoch_0}
RUN_LOG=${RUN_LOG:-${TT}/experiments/bioreason/runs/sft_qwen3_32b_stage1norm_keeplist_smoke/segment_20260809_152410.log}
PARQUET=${PARQUET:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl}
CACHE=${CACHE:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt}
cd "${TT}"
module load frameworks/2025.3.1
set -u
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${TT}"
export ZE_AFFINITY_MASK=0,1
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export FI_PROVIDER=cxi ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCHTUNE_USE_XPU_FLASH=1

echo "python=$(command -v python3) version=$(python3 --version)"

bash "${TT}/scripts/check_run_health.sh" "${RUN_LOG}"
python3 "${TT}/experiments/bioreason/probe_generation_health.py" \
  --ckpt_dir "${CKPT}" --proj_dir "${CKPT}" --local_parquet "${PARQUET}" \
  --esm3_cache_path "${CACHE}" --native_prompt --no_vllm --backbone_device_map auto \
  --num_proteins 2 --max_new_tokens 128 --max_protein_len 2048 --num_go_tokens 200 \
  --out "${CKPT}/health_probe"
python3 "${TT}/experiments/bioreason/probe_keep_list_accuracy.py" \
  --ckpt_dir "${CKPT}" --proj_dir "${CKPT}" --num_proteins 2

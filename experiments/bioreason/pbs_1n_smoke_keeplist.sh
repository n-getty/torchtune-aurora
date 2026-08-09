#!/bin/bash
#PBS -N br_s1norm_keeplist_smoke
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_1n_smoke_keeplist.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_1n_smoke_keeplist.err
#
# Short keep-list format smoke.  This is deliberately separate from the 4-node
# production launcher so target-format failures are caught before queueing a long run.
set -eo pipefail
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "${PROJDIR}"

module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export HF_HOME=/lus/flare/projects/ModCon/ngetty/hf_cache
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export no_proxy="*" NO_PROXY="*"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT PYTHONUNBUFFERED=1
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 FI_PROVIDER=cxi CCL_KVS_IFACE=lo
export TORCHTUNE_USE_XPU_FLASH=1 TORCHTUNE_USE_XPU_FLEX=0
export PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}"

RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
CONFIG="${PROJDIR}/recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_keeplist_xpu.yaml"
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/esm3_cache_2048_l37.safetensors
MAXSTEPS=${MAXSTEPS:-15}
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/keeplist_smoke_${TS}"
mkdir -p "${LOGDIR}"

python3 -m torch.distributed.run --standalone --nproc_per_node=12 \
  --redirects 3 --tee 3 --log-dir "${LOGDIR}/torchelastic" \
  "${RECIPE}" --config "${CONFIG}" \
  output_dir="${LOGDIR}/run_out" esm3_cache_path="${CACHE}" \
  compile.model=true 'pad_buckets=[4096,6144]' 'bucket_batch_sizes=[2,1]' \
  save_every_n_epochs=999 save_every_n_steps=10 max_steps_per_epoch="${MAXSTEPS}" \
  2>&1 | tee "${LOGDIR}/run.log"

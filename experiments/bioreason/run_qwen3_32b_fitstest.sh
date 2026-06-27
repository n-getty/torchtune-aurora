#!/bin/bash
# Qwen3-32B text-only SFT fits-test (12-tile single-node FSDP2). Disambiguates whether the
# Gemma4-31B 12-tile crash is Gemma-specific or general-32B. Stock dense recipe, alpaca, LinearCE.
set -eo pipefail   # no set -u

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
RECIPE="${PROJDIR}/recipes/dev/full_finetune_distributed_xpu.py"
CONFIG="${PROJDIR}/recipes/configs/dev/production/sft_qwen3_32B_fitstest_xpu.yaml"
NPROC="${NPROC:-12}"

module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export HF_HOME=/lus/flare/projects/ModCon/ngetty/hf_cache
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*" NO_PROXY="*"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# Same memory/MR strategy as the BioReason ablation: default alloc + userfaultfd.
export XPU_USM_ALLOC_SO=${XPU_USM_ALLOC_SO:-}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.99}
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-userfaultfd}
export PYTHONUNBUFFERED=1
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 FI_PROVIDER=cxi CCL_KVS_IFACE=lo

TS=$(date +%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/qwen32b_fitstest_${TS}"
mkdir -p "${LOGDIR}"
cd "${PROJDIR}"
echo "=== QWEN3-32B FITS-TEST | host=$(hostname) NPROC=${NPROC} ===" | tee "${LOGDIR}/run.log"

PYTHONPATH="${PROJDIR}" \
python3 -m torch.distributed.run \
    --standalone --nproc_per_node=${NPROC} \
    --redirects 3 --tee 3 --log-dir "${LOGDIR}/torchelastic" \
    "${RECIPE}" --config "${CONFIG}" \
    output_dir="${LOGDIR}/run_out" \
    2>&1 | tee -a "${LOGDIR}/run.log"
rc=${PIPESTATUS[0]}
echo "=== qwen32b fitstest exit rc=${rc} ===" | tee -a "${LOGDIR}/run.log"
echo "LOGDIR=${LOGDIR}"
exit ${rc}

#!/bin/bash
# BioReason native-Gemma4 SFT — 1-node smoke runner (Aurora / XPU).
#
# Step 1 (if missing): precompute the ESM3 cache over the SFT validation shard at
#   max_protein_len=2048 (ESM3 needs XPU; ~7365 seqs).
# Step 2: run the SFT recipe with the smoke config (5 steps, seq=4096, FSDP2, AC).
#
# Usage (after SSHing into a held node):
#   bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_native_gemma4_sft_smoke.sh
#
# Env:
#   NPROC=<N>           nproc_per_node (default 6 — half a node; 31B/6 tiles ~ smoke fit)
#   SKIP_CACHE=1        skip the ESM3 precompute (cache already exists)
#   EXTRA_OVERRIDES     extra key=val recipe overrides

set -eo pipefail
# NB: no `set -u` — frameworks/venv activate touches unbound vars.

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
DATA_DIR=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/data
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/esm3_cache_2048.pt
VAL_PARQUET=${DATA_DIR}/validation-00000-of-00001.parquet

# ── Environment (frameworks + BioReason deps + offline HF) ────────────────────
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export HF_HOME=/lus/flare/projects/ModCon/ngetty/hf_cache
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*" NO_PROXY="*"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset XPU_USM_ALLOC_SO PYTORCH_ALLOC_CONF
export PYTHONUNBUFFERED=1

# ── Single-node CCL row (NOT pmix/mpi — hangs torchrun --standalone) ──────────
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/native_sft_smoke_${TS}"
mkdir -p "${LOGDIR}"
cd "${PROJDIR}"

# ── Step 1: ESM3 cache over the SFT validation shard (one tile) ───────────────
# The precompute script takes a DIRECTORY (processes every .parquet in it). Stage the
# validation shard alone into a smoke dir so we only encode ~7365 seqs, not all 124K.
STAGE_DIR=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/_smoke_val
if [ "${SKIP_CACHE:-0}" != "1" ] && [ ! -f "${CACHE}" ]; then
    mkdir -p "${STAGE_DIR}"
    ln -sf "${VAL_PARQUET}" "${STAGE_DIR}/validation-00000-of-00001.parquet"
    echo "=== [smoke] ESM3 precompute -> ${CACHE} $(date) ===" | tee "${LOGDIR}/precompute.log"
    ZE_AFFINITY_MASK=0 PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}" \
        python experiments/bioreason/precompute_esm3_cache.py \
        --data_dir "${STAGE_DIR}" \
        --out "${CACHE}" \
        --max_protein_len 2048 \
        --log_every 200 --flush_every 1000 \
        2>&1 | tee -a "${LOGDIR}/precompute.log"
else
    echo "=== [smoke] ESM3 cache present (or SKIP_CACHE=1): ${CACHE} ==="
fi

# ── Step 2: SFT smoke ─────────────────────────────────────────────────────────
RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
CONFIG="${PROJDIR}/recipes/configs/dev/smoke/sft_bioreason_gemma4_31B_smoke_xpu.yaml"
NPROC=${NPROC:-6}

echo "=== BioReason native-Gemma4 SFT SMOKE ===" | tee "${LOGDIR}/launcher.log"
echo "  TS=${TS} HOST=$(hostname) NPROC=${NPROC}" | tee -a "${LOGDIR}/launcher.log"
echo "  CONFIG=${CONFIG}" | tee -a "${LOGDIR}/launcher.log"

PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}" \
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${NPROC} \
    --redirects 3 --tee 3 \
    --log-dir "${LOGDIR}/torchelastic" \
    "${RECIPE}" \
    --config "${CONFIG}" \
    output_dir="${LOGDIR}/run_out" \
    esm3_cache_path="${CACHE}" \
    ${EXTRA_OVERRIDES:-} \
    2>&1 | tee -a "${LOGDIR}/launcher.log"
rc=${PIPESTATUS[0]}
echo "=== smoke exit rc=${rc} at $(date) ===" | tee -a "${LOGDIR}/launcher.log"

# Run-health gate (per CLAUDE.md RESULTS_DISCIPLINE).
if [ -x "${PROJDIR}/scripts/check_run_health.sh" ]; then
    "${PROJDIR}/scripts/check_run_health.sh" "${LOGDIR}/launcher.log" || \
        echo "WARN: check_run_health flagged DEGRADED (see above)"
fi
echo "LOGDIR=${LOGDIR}"
exit ${rc}

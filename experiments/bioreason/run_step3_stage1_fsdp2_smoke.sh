#!/bin/bash
# Step 3 of the Qwen3.6-27B integration plan (jiggly-jumping-garden.md): full 12-tile
# FSDP2 Stage-1 (frozen-backbone/projector-only) smoke, 5 real training steps.
#
# Verifies: get_shard_conditions() matches all backbone *.layers.N modules, AC engages on
# Qwen3_5DecoderLayer, loss finite/decreasing, check_run_health.sh green, memory fits.
#
# Run on a held debug-scaling node (see hold_qwen36_stage1_smoke.sh) via SSH + nohup, NOT
# via a fresh one-shot PBS submission (holds are for iteration; see
# feedback_iterate_in_hold_not_resubmit.md).
set -eo pipefail
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "${PROJDIR}"

module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
# HF-wrapper backbone needs torch211_venv's transformers==5.7.0 (has Qwen3_5*); the
# frameworks module's bundled transformers lacks the qwen3_5 model entirely.
source /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/torch211_venv/bin/activate
export PYTHONNOUSERSITE=1
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export HF_HOME=/lus/flare/projects/ModCon/ngetty/hf_cache
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; export no_proxy="*" NO_PROXY="*"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99
export FI_MR_CACHE_MONITOR=userfaultfd
export PYTHONUNBUFFERED=1
# Interactive single-node row of CLAUDE.md's launcher decision table: none/ofi, no device_id.
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 FI_PROVIDER=cxi CCL_KVS_IFACE=lo
export TORCHTUNE_BIOREASON_TIMING=1
export PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}"

RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
CFG="${PROJDIR}/recipes/configs/dev/production/sft_bioreason_qwen36_27B_stage1norm_xpu.yaml"
MAXSTEPS=${MAXSTEPS:-5}
TS=$(date +%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/step3_stage1_${TS}"
mkdir -p "${LOGDIR}"

echo "==== STEP 3: Qwen3.6-27B Stage-1 FSDP2 smoke, ${MAXSTEPS} steps, 12 tiles ====" | tee "${LOGDIR}/run.log"
python3 -m torch.distributed.run --standalone --nproc_per_node=12 \
    --redirects 3 --tee 3 --log-dir "${LOGDIR}/torchelastic" \
    "${RECIPE}" --config "${CFG}" \
    output_dir="${LOGDIR}/run_out" \
    save_every_n_epochs=999 save_every_n_steps=100000 max_steps_per_epoch=${MAXSTEPS} \
    2>&1 | tee -a "${LOGDIR}/run.log" || true
rc=${PIPESTATUS[0]}
echo "==== STEP 3 rc=${rc} LOGDIR=${LOGDIR} ===="
echo "--- AC engage / shard info ---"
grep -hE "set_activation_checkpointing|Qwen3_5DecoderLayer|Sharding model|FSDP" "${LOGDIR}/run.log" 2>/dev/null | head -10
echo "--- step times ---"
grep -hoE "[0-9.]+s/it\]" "${LOGDIR}/run.log" 2>/dev/null | tail -8
echo "--- peak mem ---"
grep -hoE "peak_memory_reserved:[0-9.]+" "${LOGDIR}/run.log" 2>/dev/null | tail -5
echo "--- loss (finite/decreasing?) ---"
grep -hoE "Loss: [0-9.]+" "${LOGDIR}/run.log" 2>/dev/null | head -15
echo "--- banned/OOM/error scan ---"
grep -hE "banned|OutOfMemory|PDE|Traceback|Error|error" "${LOGDIR}/run.log" 2>/dev/null | head -10 || echo "NONE"

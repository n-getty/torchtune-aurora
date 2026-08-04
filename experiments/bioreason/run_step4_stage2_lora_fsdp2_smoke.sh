#!/bin/bash
# Step 4 of the Qwen3.6-27B integration plan (jiggly-jumping-garden.md): full 12-tile
# FSDP2 Stage-2 LoRA smoke, 5 real training steps. This is the plan's overall pass criterion.
#
# Verifies: PEFT LoRA applies to both sub-layer types (attn q/k/v/o + MLP + Gated-DeltaNet
# in_proj/out_proj/conv1d), enable_input_require_grads() is load-bearing, FSDP2 shards through
# the PEFT nesting prefix, adapters stay bf16 (autocast_adapter_dtype=False), loss
# finite/decreasing, check_run_health.sh green.
#
# Run on a held debug-scaling node via SSH + nohup, NOT a fresh one-shot PBS submission.
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
# torch211_venv's own site-packages MUST take precedence over BIOREASON_DEPS: the latter
# pins peft==0.17.1 (for the frameworks-module transformers used by other backbones), whose
# peft_model.py unconditionally imports HybridCache from transformers -- a name that no longer
# exists in torch211_venv's transformers==5.7.0 (needed for Qwen3_5*). torch211_venv has its
# own co-located peft==0.20.0 which imports cleanly against its own transformers; PYTHONPATH
# entries are searched before the interpreter's own site-packages, so without this the older
# BIOREASON_DEPS peft always wins regardless of venv activation order. bioreason_deps has no
# other overlapping packages (checked: no transformers/torch/accelerate/safetensors there).
VENV_SITE_PACKAGES="${PROJDIR}/experiments/lora_grpo/torch211_venv/lib/python3.12/site-packages"
export PYTHONPATH="${VENV_SITE_PACKAGES}:${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}"

RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
CFG="${PROJDIR}/recipes/configs/dev/production/sft_bioreason_qwen36_27B_stage2_xpu.yaml"
MAXSTEPS=${MAXSTEPS:-5}
TS=$(date +%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/step4_stage2_${TS}"
mkdir -p "${LOGDIR}"

echo "==== STEP 4: Qwen3.6-27B Stage-2 LoRA FSDP2 smoke, ${MAXSTEPS} steps, 12 tiles ====" | tee "${LOGDIR}/run.log"
python3 -m torch.distributed.run --standalone --nproc_per_node=12 \
    --redirects 3 --tee 3 --log-dir "${LOGDIR}/torchelastic" \
    "${RECIPE}" --config "${CFG}" \
    output_dir="${LOGDIR}/run_out" \
    save_every_n_epochs=999 save_every_n_steps=100000 max_steps_per_epoch=${MAXSTEPS} \
    2>&1 | tee -a "${LOGDIR}/run.log" || true
rc=${PIPESTATUS[0]}
echo "==== STEP 4 rc=${rc} LOGDIR=${LOGDIR} ====" | tee -a "${LOGDIR}/run.log"
{
echo "--- LoRA / PEFT engage info ---"
grep -aE "get_peft_model|lora_A|lora_B|trainable params|enable_input_require_grads|PEFT" "${LOGDIR}/run.log" | head -10
echo "--- AC engage / shard info ---"
grep -aE "set_activation_checkpointing|Qwen3_5DecoderLayer|Sharding model|FSDP|enable_activation_checkpointing" "${LOGDIR}/run.log" | head -10
echo "--- step times ---"
grep -aoE "[0-9.]+s/it\]" "${LOGDIR}/run.log" | tail -8
echo "--- peak mem ---"
grep -aiE "peak memory|memory_reserved" "${LOGDIR}/run.log" | tail -5
echo "--- loss (finite/decreasing?) ---"
grep -aoE "1\|[0-9]\|Loss: [0-9.]+" "${LOGDIR}/run.log" | sort -u
echo "--- banned/OOM/error scan ---"
grep -aE "banned|OutOfMemory|PDE|Traceback|Error|error" "${LOGDIR}/run.log" | head -10 || echo "NONE"
} | tee -a "${LOGDIR}/run.log"

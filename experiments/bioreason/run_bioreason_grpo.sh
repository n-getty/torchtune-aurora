#!/bin/bash
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -A ModCon
#PBS -N bioreason_grpo
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/bioreason_grpo.out

# BioReason-Pro 4B GRPO on Aurora — 1-node debug run (validation, not full training).
# Layout: 12 tiles / node
#   Tiles 0-1: DDP training (policy + reference model, 4B each)
#   Tile  2:   vLLM colocated (generate rollouts with enable_prompt_embeds)
#
# Weight sync: shared memory (/dev/shm) — same node, fast
# Embedding computation: ESM3 + projectors run on training tiles before vLLM call

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

# Add ESM + BioReason deps
export PYTHONPATH="/lus/flare/projects/ModCon/ngetty/bioreason_deps:${PYTHONPATH}"

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
MODEL_DST=/tmp/torchtune/bioreason-pro-sft
mkdir -p /tmp/torchtune /dev/shm/torchtune

if [ ! -f "${MODEL_DST}/config.json" ]; then
    echo "=== Staging bioreason-pro-sft to NVMe... ==="
    t0=$SECONDS
    cp -r "$MODEL_SRC" "$MODEL_DST"
    echo "Staged in $((SECONDS - t0))s"
fi

echo "=== BioReason 4B GRPO — debug validation run ==="
echo "  Model: ${MODEL_DST}"
echo "  Dataset: /lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl"

VLLM_DP=1 bash recipes/dev/run_grpo_vllm_xpu.sh \
    2 \
    10 \
    ${MODEL_DST} \
    20 \
    recipes/configs/dev/production/bioreason_4b_grpo_xpu.yaml \
    output_dir=${PROJDIR}/outputs/bioreason_grpo_debug \
    base_model_path=${MODEL_DST}

echo "=== Done ==="

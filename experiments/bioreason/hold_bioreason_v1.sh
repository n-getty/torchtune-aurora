#!/bin/bash
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A ModCon
#PBS -N bioreason_grpo_v1
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_v1.out

# BioReason-Pro 4B GRPO integration test — 1-node capacity queue.
# Validates: T8 (full pipeline), T9 (recipe hooks: vllm_param_iter, prompt_embeds)
# Does NOT run the full GRPO recipe yet — just validates the multimodal plumbing.

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
export INFRA_PROVIDER=local

# BioReason deps must be first (before any ESM import)
export PYTHONPATH="/lus/flare/projects/ModCon/ngetty/bioreason_deps:${PROJDIR}:${PYTHONPATH}"

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
MODEL_DST=/tmp/torchtune/bioreason-pro-sft

if [ ! -f "${MODEL_DST}/config.json" ]; then
    echo "=== Staging bioreason-pro-sft to NVMe ==="
    t0=$SECONDS
    mkdir -p /tmp/torchtune
    cp -r "$MODEL_SRC" "$MODEL_DST"
    echo "=== Staged in $((SECONDS - t0))s ==="
fi

echo "=== BioReason T8+T9 integration tests ==="
echo "  Node: $(hostname)"
echo "  Model: ${MODEL_DST}"
echo ""

# Single tile (T8 + T9 don't need distributed)
ZE_AFFINITY_MASK=0 python experiments/bioreason/test_bioreason_xpu.py

STATUS=$?
echo ""
echo "=== Done (exit=$STATUS) ==="
exit $STATUS

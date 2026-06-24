#!/bin/bash
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A ModCon
#PBS -N bioreason_grpo_v1
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_grpo_v1.out

# BioReason-Pro 4B GRPO — recipe startup validation on 1 node.
# colocate_sleep: each of 2 training tiles also runs a TP=1 vLLM engine
# (tiles alternate between training fwd/bwd and vLLM rollout generation).
# prompt_embeds flow: ESM3+GO run on training tile → tensor passed in-process to vLLM.
#
# This run validates: recipe setup, model load, optimizer init, trajectory generation
# (num_steps=5 — just enough to confirm dataflow is correct end-to-end).

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CRITICAL: must be set before any esm.* import (model.py _ensure_paths calls ESM)
export INFRA_PROVIDER=local

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

# bioreason_deps first so it shadows nothing important
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

echo "=== BioReason 4B GRPO — v1 recipe validation ==="
echo "  Node: $(hostname)"
echo "  Model: ${MODEL_DST}"
echo "  Training tiles: 2 (ranks 0+1, colocate_sleep vLLM)"

# 2 training ranks; each creates a TP=1 vLLM engine on its own tile (colocate_sleep).
# Fix: ZE_AFFINITY_MASK is set to local_rank inside _init_vllm_tp1 so vLLM's
# mem_get_info() sees the correct tile's independent 64 GiB pool, not tile 0's.
# ref_cpu_offload=true: ref model stays on CPU (~saves 8GB/tile for vLLM KV cache).
# num_steps=5: just enough to confirm setup + trajectory generation work end-to-end.
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    recipes/dev/grpo_bioreason_distributed_xpu.py \
    --config recipes/configs/dev/production/bioreason_4b_grpo_xpu.yaml \
    base_model_path=${MODEL_DST} \
    output_dir=${PROJDIR}/outputs/bioreason_grpo_v1 \
    num_steps=5 \
    ref_cpu_offload=true \
    log_peak_memory_stats=true

STATUS=$?
echo ""
echo "=== Done (exit=${STATUS}) ==="
exit ${STATUS}

#!/bin/bash
# Explicitly set frameworks Python path (module load doesn't modify PATH in nohup bash)
FW_BIN=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin
export PATH=${FW_BIN}:$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
echo "Python: $(which python3)"
python3 -c "import torch; print('torch OK:', torch.__version__)" || { echo "FAIL"; exit 1; }

export PYTHONNOUSERSITE=1
export INFRA_PROVIDER=local
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/bioreason_deps:${PROJDIR}:${PYTHONPATH}

# Also source the aurora environment for CCL/MKL/etc.
source /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/setvars.sh 2>/dev/null || true

MODEL_DST=/tmp/torchtune/bioreason-pro-sft
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
if [ ! -f "${MODEL_DST}/config.json" ]; then
    echo "=== Staging model ==="
    mkdir -p /tmp/torchtune
    cp -r "$MODEL_SRC" "$MODEL_DST"
fi
cd ${PROJDIR}
echo "=== Starting 2-rank bioreason GRPO test (empty_cache fix) ==="
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    recipes/dev/grpo_bioreason_distributed_xpu.py \
    --config recipes/configs/dev/production/bioreason_4b_grpo_xpu.yaml \
    base_model_path=${MODEL_DST} \
    output_dir=${PROJDIR}/outputs/bioreason_grpo_v1 \
    num_steps=5 \
    log_peak_memory_stats=true 2>&1
STATUS=$?
echo "=== Done: exit=${STATUS} ==="
exit ${STATUS}

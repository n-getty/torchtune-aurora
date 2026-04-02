#!/bin/bash
# Quick test: colocated vLLM with gloo PG trick inside torchrun
set -e
cd /lus/flare/projects/ModCon/ngetty/torchtune

module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU env
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

VLLM_CUSTOMIZATION=/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/_usercustomize_vllm
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/torchtune:/flare/ModCon/ngetty/trl:${VLLM_CUSTOMIZATION}:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B
MODEL=/tmp/torchtune/Qwen2.5-3B
NPROC=${1:-2}

mkdir -p /tmp/torchtune
if [ ! -f "${MODEL}/config.json" ]; then
    echo "Staging model..."
    cp -r "${MODEL_SRC}" "${MODEL}"
    echo "Model staged"
fi

# Warm vLLM model info cache
echo "Warming vLLM cache..."
ZE_AFFINITY_MASK=$((NPROC - 1)) python3 -c "
from vllm.config import ModelConfig
ModelConfig(model='${MODEL}', tokenizer='${MODEL}', dtype='bfloat16', enforce_eager=True)
print('Cache warmed')
" 2>&1 | tail -2

# Clean up any leftover sync files
rm -f /tmp/torchtune/vllm_init_done

echo "=== Testing colocated vLLM (gloo PG trick) with ${NPROC} ranks ==="
python3 -m torch.distributed.run --standalone --nproc_per_node=${NPROC} \
    recipes/dev/_test_vllm_colocate_gloo.py ${MODEL}

echo "=== Test complete ==="

#!/bin/bash
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}
# Use 2025.2.0 — 2025.3.1 has broken XCCL allreduce (USM pointer validation)
module load frameworks/2025.2.0 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
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
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${PROJDIR}:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC=${1:-10}
NSTEPS=${2:-5}
LOCAL_MODEL=/tmp/torchtune/gemma-4-31B
echo "=== Gemma4 31B GRPO Baseline ==="
echo "Node: $(hostname), Date: $(date)"
echo "CWD: $(pwd)"
echo "Model: ${LOCAL_MODEL}"
echo "Tiles: ${NPROC}, Steps: ${NSTEPS}"

python3 -m torch.distributed.run --standalone --nproc_per_node=${NPROC} \
    ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${PROJDIR}/recipes/configs/dev/production/gemma4_31B_grpo_novllm_xpu.yaml \
    base_model_path=${LOCAL_MODEL} \
    num_steps=${NSTEPS}

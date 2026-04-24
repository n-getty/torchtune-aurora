#!/bin/bash
# EP=4/DP=3 GRPO no-vLLM correctness test — v15
# Fix: move registered buffers (layer_scalar, rope cache) to XPU after
#       ref model FSDP2 cpu_offload setup. FSDP2 only manages params;
#       buffers stay on CPU and cause device mismatch during ref forward.

set -euo pipefail

module load frameworks/2025.2.0

REPO=/lus/flare/projects/ModCon/ngetty/torchtune
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.2.0/lib/python3.10/site-packages
FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages

# PYTHONNOUSERSITE=1 prevents ~/.local/lib/python3.10/site-packages (plain)
# from prepending torchao-0.16.0. Explicit PYTHONPATH: FW_SITE first so
# torchao-0.12.0 wins; LOCAL_SITE (frameworks-specific) restores math_verify.
export PYTHONNOUSERSITE=1
export PYTHONPATH=${REPO}:${FW_SITE}:${LOCAL_SITE}:${PYTHONPATH:-}

# CCL / transport config
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_OP_SYNC=1
export CCL_CHUNK_SIZE=16777216
export CCL_ALLREDUCE=ring

# ZE / Level Zero
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

CONFIG=${REPO}/recipes/configs/dev/experimental/gemma4_26b_grpo_ep4_nollm_xpu.yaml
NNODES=${NNODES:-1}
NPROC=12

echo "=== EP4 GRPO v15 — buffer device fix ==="
echo "Config: ${CONFIG}"
echo "Nodes: ${NNODES}, Procs: ${NPROC}"
python3 -c "import torchao; print('torchao:', torchao.__version__)"
python3 -c "import math_verify; print('math_verify:', math_verify.__file__)"

torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=${NNODES} \
    --master_addr=localhost \
    --master_port=29500 \
    ${REPO}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config ${CONFIG} \
    2>&1

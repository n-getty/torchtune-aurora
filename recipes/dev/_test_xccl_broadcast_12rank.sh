#!/bin/bash
# Production-scale XCCL broadcast test: 10 training + 2 vLLM on all 12 tiles
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_aurora_paths.sh"
cd "${TORCHTUNE_DIR}"

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
unset PYTORCH_ALLOC_CONF
unset PYTORCH_CUDA_ALLOC_CONF

echo "=== XCCL Broadcast 12-Rank Test (10 train + 2 vLLM) ==="
echo "Node: $(hostname), Date: $(date)"
echo "Python: $(which python3)"

python3 -m torch.distributed.run --standalone --nproc_per_node=12 \
    recipes/dev/_test_xccl_broadcast_12rank.py

echo "=== Done ==="

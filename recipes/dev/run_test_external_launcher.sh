#!/bin/bash
# Run external_launcher vLLM test on XPU.
# Usage from compute node: bash recipes/dev/run_test_external_launcher.sh [tp] [model]
#   tp: tensor parallel size (default: 1)
#   model: model path (default: /tmp/torchtune/Qwen2.5-3B)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment
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
unset PYTORCH_ALLOC_CONF
aurora_export_pythonpath "${TORCHTUNE_DIR}"

TP=${1:-1}
MODEL=${2:-/tmp/torchtune/Qwen2.5-3B}

echo "=== Test: vLLM external_launcher on XPU ==="
echo "Node: $(hostname), Date: $(date)"
echo "TP: ${TP}, Model: ${MODEL}"
echo "Python: $(which python3)"
echo "============================================="

torchrun --standalone --nproc_per_node=$TP \
    recipes/dev/_test_vllm_external_launcher.py \
    "$MODEL" --tp $TP --gpu-mem 0.5 --max-model-len 512

echo "=== Test complete ==="

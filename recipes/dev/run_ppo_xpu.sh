#!/bin/bash
# Quick launcher for PPO XPU baseline test
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

# Load Aurora frameworks module (provides XPU-enabled PyTorch 2.10 + python 3.12)
module load frameworks 2>/dev/null || true

# Remove user virtualenv from PATH so frameworks python is used
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL / XPU environment
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
aurora_export_pythonpath "${TORCHTUNE_DIR}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

NPROC=${1:-2}
NSTEPS=${2:-5}

echo "=== PPO XPU baseline: ${NPROC} tiles, ${NSTEPS} steps ==="
echo "Node: $(hostname), Date: $(date)"
echo "Python: $(which python3)"

python3 -m torch.distributed.run --standalone --nproc_per_node=$NPROC \
    recipes/dev/ppo_full_finetune_distributed.py \
    --config recipes/configs/dev/baseline/1B_ppo_xpu_baseline.yaml \
    num_steps=$NSTEPS

#!/bin/bash
# Run EP smoke test on compute node via torchrun (Aurora/XPU).
# Usage: bash recipes/dev/run_ep_smoke.sh [ep_degree]
#   ep_degree: 2 (default) or 4

set -e
EP=${1:-2}
REPO=/lus/flare/projects/ModCon/ngetty/torchtune

cd $REPO

module load frameworks/2025.2.0

# Strip myenv from PATH to avoid conflicts
export PATH=$(echo $PATH | tr ':' '\n' | grep -v myenv | tr '\n' ':')

# Put our repo first so it shadows the system torchtune-0.0.0
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# CCL env vars for torchrun (no MPI, ofi transport — matches working aurora_grpo pattern)
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_MODE=pmi
export CCL_KVS_IFACE=hsn0
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

echo "=== EP Smoke Test: EP=$EP ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

export MASTER_PORT=$((29600 + RANDOM % 400))
export MASTER_ADDR=127.0.0.1

torchrun \
  --nproc_per_node=$EP \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $REPO/recipes/dev/test_ep_smoke.py --ep $EP 2>&1

echo "=== Done ==="

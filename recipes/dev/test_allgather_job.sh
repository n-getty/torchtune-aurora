#!/bin/bash
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -l walltime=00:15:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -o logs/test_allgather.out
#PBS -e logs/test_allgather.err
#PBS -N test_allgather

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=recipes/dev/_aurora_paths.sh
source "${SCRIPT_DIR}/_aurora_paths.sh"

cd "${TORCHTUNE_DIR}"

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
export MASTER_ADDR="${UNIQUE_NODES[0]}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES=2
export USE_AFFINITY_MASK=training

echo "=== AllGather multi-node test ==="
echo "Nodes: ${UNIQUE_NODES[*]}"
echo "MASTER_ADDR: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

# Single test: WITH ring algorithms
echo "=== WITH ring algorithms ==="
mpiexec --hostfile "$PBS_NODEFILE" -n 20 -ppn 10 --no-vni --cpu-bind depth --depth 8 \
    bash "${TORCHTUNE_DIR}/recipes/dev/test_mesh_wrapper.sh" \
    "${TORCHTUNE_DIR}/recipes/dev/test_allgather_multinode.py"

echo ""
echo "=== DONE ==="

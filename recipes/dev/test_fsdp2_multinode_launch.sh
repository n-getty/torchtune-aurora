#!/bin/bash
#
# Launch matrix of FSDP2 multi-node benchmarks to isolate CCL sub-communicator issue.
#
# Runs the same minimal FSDP2 model under different configurations:
#   1. Single-node baseline (torchrun, 10 tiles)
#   2. Multi-node with different ZE_AFFINITY_MASK modes
#   3. Multi-node with/without ring algorithm forcing
#   4. Raw AllGather comparison (world vs sub-group)
#
# Prerequisites:
#   - 2-node PBS allocation (select=2)
#   - export PBS_JOBID=<jobid>
#   - export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
#
# Usage (interactive):
#   bash recipes/dev/test_fsdp2_multinode_launch.sh
#
# Usage (PBS):
#   qsub recipes/dev/test_fsdp2_multinode_launch.sh
#
#PBS -l select=2:system=aurora
#PBS -l filesystems=home:flare
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/fsdp2_multinode_bench.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/fsdp2_multinode_bench.err
#PBS -N fsdp2_bench
# No set -e: we want all tests to run even if some fail

TORCHTUNE_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TORCHTUNE_DIR}"

# ============================================================
# Environment
# ============================================================
module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

NGPUS=10
SCRIPT="recipes/dev/test_fsdp2_multinode_minimal.py"
WRAPPER="recipes/dev/test_fsdp2_multinode_wrapper.sh"

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set."
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "$PBS_NODEFILE" | awk '!seen[$0]++'))
NUM_NODES=${#UNIQUE_NODES[@]}
NODE0="${UNIQUE_NODES[0]}"

export MASTER_ADDR="${NODE0}.hsn.cm.aurora.alcf.anl.gov"
export MASTER_PORT=$((20000 + RANDOM % 20000))
export NUM_NODES

TOTAL_RANKS=$((NUM_NODES * NGPUS))

echo "================================================================"
echo "FSDP2 Multi-Node CCL Sub-Communicator Benchmark"
echo "================================================================"
echo "Nodes:    ${NUM_NODES} (${UNIQUE_NODES[*]})"
echo "Tiles:    ${NGPUS}/node, ${TOTAL_RANKS} total"
echo "Master:   ${MASTER_ADDR}:${MASTER_PORT}"
echo "================================================================"

# Common env for all tests
export MODEL_SIZE=small   # Fast iterations; use "large" for 32B-like
export SEQ_LEN=512
export BATCH_SIZE=1
export NUM_ITERS=5
export WARMUP=2
export TEST_BACKWARD=1

# ============================================================
# Test 1: Single-node baseline (torchrun on node0)
# ============================================================
echo ""
echo "================================================================"
echo "TEST 1: Single-node baseline (torchrun, ${NGPUS} tiles, node0)"
echo "================================================================"

ssh "${NODE0}" "
cd ${TORCHTUNE_DIR}
module load frameworks 2>/dev/null
export PATH=\$(echo \"\$PATH\" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV 2>/dev/null
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=\$(seq -s, 0 $((NGPUS - 1)))
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export MODEL_SIZE=${MODEL_SIZE}
export SEQ_LEN=${SEQ_LEN}
export BATCH_SIZE=${BATCH_SIZE}
export NUM_ITERS=${NUM_ITERS}
export WARMUP=${WARMUP}
export TEST_BACKWARD=${TEST_BACKWARD}
export SKIP_1D=0
torchrun --standalone --nproc_per_node=${NGPUS} ${SCRIPT}
"

# ============================================================
# Test 2: Multi-node, AFFINITY_MODE=training, CCL_RING=1
#   (current production config)
# ============================================================
echo ""
echo "================================================================"
echo "TEST 2: Multi-node HSDP, AFFINITY=training, RING=1 (current config)"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=training
export CCL_ALGO=ring
export CCL_TRANSPORT=ofi
export SKIP_1D=0

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 2 FAILED (exit code $?)"

# ============================================================
# Test 3: Multi-node, AFFINITY_MODE=none (unset mask)
#   Tests whether the mask itself causes sub-group regression
# ============================================================
echo ""
echo "================================================================"
echo "TEST 3: Multi-node HSDP, AFFINITY=none (no mask), RING=1"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=none
export CCL_ALGO=ring
export CCL_TRANSPORT=ofi
export SKIP_1D=1

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 3 FAILED (exit code $?)"

# ============================================================
# Test 4: Multi-node, AFFINITY=training, CCL_RING=0 (default algos)
#   Tests whether ring algorithm forcing causes sub-group regression
# ============================================================
echo ""
echo "================================================================"
echo "TEST 4: Multi-node HSDP, AFFINITY=training, RING=0 (default algos)"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=training
export CCL_ALGO=0
export CCL_TRANSPORT=ofi
export SKIP_1D=1

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 4 FAILED (exit code $?)"

# ============================================================
# Test 5: Multi-node, AFFINITY=single (mask=$LOCAL_RANK), RING=1
#   Tests original single-tile mask (known to cause "narrow" warning)
# ============================================================
echo ""
echo "================================================================"
echo "TEST 5: Multi-node HSDP, AFFINITY=single (mask=\$LOCAL_RANK), RING=1"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=single
export CCL_ALGO=ring
export CCL_TRANSPORT=ofi
export SKIP_1D=1

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 5 FAILED (exit code $?)"

# ============================================================
# Test 6: Multi-node, AFFINITY=all (mask=0-11), RING=1
#   Full tile visibility — tests if CCL needs all 12 UUIDs
#   NOTE: Only safe without vLLM running
# ============================================================
echo ""
echo "================================================================"
echo "TEST 6: Multi-node HSDP, AFFINITY=all (mask=0-11), RING=1, OFI"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=all
export CCL_ALGO=ring
export CCL_TRANSPORT=ofi
export SKIP_1D=1

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 6 FAILED (exit code $?)"

# ============================================================
# Test 7: Multi-node, MPI transport + topo algorithm (--pmi=pmix)
#   Official Aurora recommended CCL config (tested to 1024 nodes)
#   This is the HIGHEST PRIORITY test — if sub-communicators work
#   correctly with MPI transport, this is the fix.
#   Uses --pmi=pmix as shown in official Aurora examples.
# ============================================================
echo ""
echo "================================================================"
echo "TEST 7: Multi-node HSDP, MPI transport, TOPO algorithm, AFFINITY=training"
echo "  (Official Aurora recommended CCL config, --pmi=pmix)"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=training
export CCL_TRANSPORT=mpi
export CCL_ALGO=topo
export SKIP_1D=1

mpiexec \
    --pmi=pmix \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 7 FAILED (exit code $?) — trying fallback test 7b"

# ============================================================
# Test 7b: Fallback — MPI transport WITHOUT pmix launcher
#   If test 7 crashes with "Fatal error in internal_Init_thread",
#   this uses CCL_PROCESS_LAUNCHER=none + CCL_ATL_TRANSPORT=mpi.
#   Relies on mpi4py pre-init in the Python script.
# ============================================================
echo ""
echo "================================================================"
echo "TEST 7b: Multi-node HSDP, MPI-nolauncher transport, TOPO algorithm"
echo "  (Fallback: MPI data transport without PMIx launcher)"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=training
export CCL_TRANSPORT=mpi-nolauncher
export CCL_ALGO=topo
export SKIP_1D=1

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 7b FAILED (exit code $?)"

# ============================================================
# Test 8: Multi-node, MPI transport + ring algorithm (--pmi=pmix)
#   Same MPI transport but with ring — isolates transport vs algorithm
# ============================================================
echo ""
echo "================================================================"
echo "TEST 8: Multi-node HSDP, MPI transport, RING algorithm, AFFINITY=training"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=training
export CCL_TRANSPORT=mpi
export CCL_ALGO=ring
export SKIP_1D=1

mpiexec \
    --pmi=pmix \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 8 FAILED (exit code $?)"

# ============================================================
# Test 9: Multi-node, OFI transport + topo algorithm
#   Isolates algorithm effect independent of transport
# ============================================================
echo ""
echo "================================================================"
echo "TEST 9: Multi-node HSDP, OFI transport, TOPO algorithm, AFFINITY=training"
echo "================================================================"

export MASTER_PORT=$((20000 + RANDOM % 20000))
export AFFINITY_MODE=training
export CCL_TRANSPORT=ofi
export CCL_ALGO=topo
export SKIP_1D=1

mpiexec \
    --hostfile "${PBS_NODEFILE}" \
    -n "${TOTAL_RANKS}" \
    -ppn "${NGPUS}" \
    --no-vni \
    --cpu-bind depth \
    --depth 8 \
    bash "${WRAPPER}" "${SCRIPT}" \
|| echo "TEST 9 FAILED (exit code $?)"

echo ""
echo "================================================================"
echo "ALL TESTS COMPLETE"
echo "================================================================"

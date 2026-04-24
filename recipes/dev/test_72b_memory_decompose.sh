#!/bin/bash
#
# Decompose 72B backward memory overhead (30 GiB) into components.
#
# This runs the no-offload config with:
#   - Per-layer memory hooks (every 10th layer) during forward & backward
#   - FSDP structure dump (verify per-layer wrapping correctness)
#   - AC verification (confirm activation checkpointing on all 80 layers)
#   - Phase-level memory tracking (pre_forward, post_forward, post_backward, post_optimizer)
#   - Prefetch disabled (saves 4.6 GiB, no downside)
#
# Expected output pattern:
#   If AC is working + resharding correct:
#     [LAYER_FWD  0/80] alloc=X resv=Y  → flat across layers
#     [LAYER_BWD 79/80] alloc=X resv=Y  → flat across layers (reverse order)
#   If AC is BROKEN:
#     [LAYER_FWD  0/80] alloc=X  → grows linearly with layers
#   If resharding is broken:
#     [LAYER_BWD 79/80] alloc=X  → grows as backward progresses
#
# Requires 4 nodes (1 vLLM + 3 training = 36 tiles).
#
# Usage:
#   ssh <node0> "bash /lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/test_72b_memory_decompose.sh"
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune

# Auto-detect PBS job
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    for f in /var/spool/pbs/aux/*; do
        if [ -f "$f" ]; then
            export PBS_NODEFILE="$f"
            break
        fi
    done
fi

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: Cannot find PBS_NODEFILE"
    exit 1
fi

echo "PBS_NODEFILE: ${PBS_NODEFILE}"
echo "Nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo ""

GENERIC_LAUNCHER="recipes/dev/aurora_grpo_dedicated_vllm_generic.sh"
mkdir -p /tmp/torchtune
LOGFILE="/tmp/torchtune/test_72b_mem_decompose.log"

echo "============================================================"
echo "  72B Memory Decomposition Diagnostic"
echo "  Config: no_offload, prefetch OFF, fsdp_diagnostics ON"
echo "  Per-layer hooks: every 10th layer (+ first/last 3)"
echo "  Steps: 2 (step 0 = no optimizer states, step 1 = with optimizer)"
echo "============================================================"
echo ""

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload_no_prefetch.yaml \
MIN_NODES=4 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=900 \
NSTEPS=2 \
GRPO_SAMPLES=4 \
bash "${GENERIC_LAUNCHER}" 2>&1 | tee "${LOGFILE}" || true

echo ""
echo "============================================================"
echo "  MEMORY DECOMPOSITION RESULTS"
echo "============================================================"
echo ""
echo "--- FSDP Structure ---"
grep -A 200 "FSDP Module Structure" "${LOGFILE}" | head -100
echo ""
echo "--- AC Verification ---"
grep "AC_CHECK" "${LOGFILE}"
echo ""
echo "--- Phase Memory ---"
grep "MEM " "${LOGFILE}"
echo ""
echo "--- Per-Layer Forward Memory ---"
grep "LAYER_FWD" "${LOGFILE}" | head -20
echo ""
echo "--- Per-Layer Backward Memory ---"
grep "LAYER_BWD" "${LOGFILE}" | head -20
echo ""
echo "--- Step Times ---"
grep "TIMING" "${LOGFILE}"
echo ""
echo "--- OOM/Errors ---"
grep -i "OOM\|OutOfMemory\|error\|FAILED" "${LOGFILE}" | tail -10
echo ""
echo "Full log: ${LOGFILE}"

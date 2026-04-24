#!/bin/bash
#
# FSDP Prefetch Diagnostic Test for 72B
#
# Tests two configurations back-to-back on 4 nodes (3 training + 1 vLLM):
#   1. WITH prefetch (default) — baseline, expect step 1 OOM
#   2. WITHOUT prefetch (disable_prefetch=True) — expect lower peak memory
#
# Goal: Determine if FSDP2 AllGather prefetch is causing the 41 GiB overhead.
#
# Usage:
#   ssh <node0> "bash /lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/test_fsdp_prefetch_diag.sh"
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

# ============================================================
# Test 1: WITH prefetch (baseline) — expect OOM at step 1
# ============================================================
echo "============================================================"
echo "  TEST 1: 72B no-offload WITH prefetch (baseline)"
echo "  Expected: step 0 works, step 1 OOMs"
echo "  Goal: capture FSDP structure + per-phase memory stats"
echo "============================================================"
echo ""

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload_diag.yaml \
MIN_NODES=4 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=900 \
NSTEPS=2 \
GRPO_SAMPLES=4 \
bash "${GENERIC_LAUNCHER}" 2>&1 | tee /tmp/torchtune/test_72b_prefetch_ON.log || true

echo ""
echo "=== Test 1 complete (OOM expected) ==="
echo ""

# Clean up between tests — kill any leftover processes
sleep 15

# ============================================================
# Test 2: WITHOUT prefetch — should have lower peak memory
# ============================================================
echo "============================================================"
echo "  TEST 2: 72B no-offload WITHOUT prefetch (disable_prefetch=True)"
echo "  Expected: lower peak memory, possibly survives step 1"
echo "  Goal: confirm prefetch is the source of 41 GiB overhead"
echo "============================================================"
echo ""

MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-72B-Instruct \
CONFIG=recipes/configs/dev/experimental/qwen72B_grpo_no_offload_no_prefetch.yaml \
MIN_NODES=4 \
VLLM_TP=4 \
VLLM_DP=3 \
VLLM_TIMEOUT=900 \
NSTEPS=3 \
GRPO_SAMPLES=4 \
bash "${GENERIC_LAUNCHER}" 2>&1 | tee /tmp/torchtune/test_72b_prefetch_OFF.log || true

echo ""
echo "============================================================"
echo "  FSDP PREFETCH DIAGNOSTIC RESULTS"
echo "============================================================"
echo ""
echo "--- Test 1: WITH prefetch ---"
echo "FSDP Structure:"
grep -A 100 "FSDP Module Structure" /tmp/torchtune/test_72b_prefetch_ON.log | head -100
echo ""
echo "Memory phases:"
grep "MEM " /tmp/torchtune/test_72b_prefetch_ON.log
echo ""
echo "Step times:"
grep "TIMING\|step_time\|OOM\|OutOfMemory\|memory" /tmp/torchtune/test_72b_prefetch_ON.log | tail -20
echo ""
echo "--- Test 2: WITHOUT prefetch ---"
echo "FSDP Structure:"
grep -A 100 "FSDP Module Structure" /tmp/torchtune/test_72b_prefetch_OFF.log | head -100
echo ""
echo "Memory phases:"
grep "MEM " /tmp/torchtune/test_72b_prefetch_OFF.log
echo ""
echo "Step times:"
grep "TIMING\|step_time\|OOM\|OutOfMemory\|memory" /tmp/torchtune/test_72b_prefetch_OFF.log | tail -20
echo ""
echo "Full logs:"
echo "  /tmp/torchtune/test_72b_prefetch_ON.log"
echo "  /tmp/torchtune/test_72b_prefetch_OFF.log"

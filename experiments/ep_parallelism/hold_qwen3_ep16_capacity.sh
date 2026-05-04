#!/bin/bash
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -N ep16_capacity
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/ep_parallelism/ep16_capacity.out

# EP=16 capacity run — GSM8K, 12 steps, legacy wsync path (no WS10).
#
# Purpose:
#   (1) Confirm steady-state memory is stable past the 3-step smoke.
#   (2) Establish a clean before/after timing baseline for WS10 comparison.
#       Current legacy wsync: wsync_gather=~75s, step_total=~140-150s.
#   (3) Validate ZeRO-2 colocate fix (Fix A1+A2) holds over more steps.
#
# NSTEPS=12: at ~150s/step = 30 min training + ~16 min model staging = ~46 min.
# Fits comfortably in 1h debug-scaling slot.
#
# Configuration:
#   - GSM8K reward (max_gen=256 safe per OOM check at job 8468039)
#   - Legacy wsync (_load_fused_moe_experts, not sharded) — deferred=false
#   - TORCHTUNE_EP_GRAD_RELEASE_XCCL=1 (32% wall improvement, validated iter2)
#   - varlen + maskfree (validated on Qwen3-8B; worth carrying on MoE)
#
# After run: compare wsync_gather vs the WS10 validation hold.

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
OUTDIR=${PROJDIR}/experiments/ep_parallelism
NODES=( $(sort -u "$PBS_NODEFILE" | awk -F. '{print $1}') )
N0=${NODES[0]}
N1=${NODES[1]}

echo "=== EP=16 capacity hold start (legacy wsync baseline): $(date) ==="
echo "Job: ${PBS_JOBID}"
echo "Nodes: ${N0} ${N1}"

ssh -o StrictHostKeyChecking=no "${N0}" "
    cd ${PROJDIR}
    export TRAIN_NODE2=${N1}
    export NSTEPS=${NSTEPS:-12}
    export CONFIG=${PROJDIR}/recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep16_gsm8k_xpu.yaml
    export TORCHTUNE_EP_GRAD_RELEASE_XCCL=1
    export TORCHTUNE_USE_IPEX_VARLEN=1
    export TORCHTUNE_MASKFREE_CAUSAL=1
    export SMOKE_TAG=ep16_capacity
    nohup bash ${PROJDIR}/recipes/dev/run_qwen3_30b_ep16_vllm_2node.sh \${NSTEPS} \
        > ${OUTDIR}/ep16_capacity_run.log 2>&1 &
    echo \"run PID \$!\"
" || echo "WARN: SSH launch returned non-zero"

echo ""
echo "Run self-launched. Holding 60 min for walltime coverage."
echo "Monitor:"
echo "  tail -f ${OUTDIR}/ep16_capacity_run.log | grep -E 'TIMING|BATCH_REWARD|ERROR|Traceback'"
sleep 3600

echo "=== EP=16 capacity hold end: $(date) ==="

# --- Post-run timing summary ---
echo ""
echo "=== EP=16 CAPACITY RUN SUMMARY ==="
LOG=${OUTDIR}/ep16_capacity_run.log
if [ ! -f "${LOG}" ]; then
    echo "ERROR: run log not found: ${LOG}"
    exit 1
fi

echo "Steps completed:"
grep "TIMING" "${LOG}" | tail -20 || echo "(none found)"
echo ""
echo "Rewards:"
grep "BATCH_REWARD" "${LOG}" | tail -12 || echo "(none found)"
echo ""
echo "Memory (last 6 lines):"
grep "PRE-STEP\|resv=" "${LOG}" | tail -12 || echo "(none found)"

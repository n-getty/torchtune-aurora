#!/bin/bash
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -N ep16_ws10_validate
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/ep_parallelism/ep16_ws10_validate.out

# WS10 end-to-end validation hold (2026-05-04).
#
# Tests that TORCHTUNE_EP_WSYNC_SHARDED=1 actually delivers trained-policy
# weights to vLLM. Previous holds (8467299, 8467388) had the receiver raise
# "FusedMoE.expert_map; got None" on every layer — vLLM stayed on init
# weights, rewards=0, ratios=1.0000 (vacuous). Receiver TP-only branch fix
# has landed; this hold is the first real end-to-end test.
#
# Acceptance gates (all four required to declare WS10 validated):
#   (a) vLLM logs: no "RuntimeError: expert_map; got None"
#   (b) trainer logs: no "NOT bumping weight version"
#   (c) rewards: non-zero and varying across steps (GSM8K)
#   (d) resp_len: not pinned to max_generated_tokens (31 or 255)
#
# Timing baseline (legacy path, WS5): wsync_gather=75s / step_total=140s.
# WS10 target: wsync_gather << 10s (each EP rank broadcasts ~1/16 of experts).
#
# NSTEPS=5: enough to confirm weights flow and reward signal; step 0 may
# still score 0 (max_gen=256 can truncate hard GSM8K traces) — that's OK.

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
OUTDIR=${PROJDIR}/experiments/ep_parallelism
NODES=( $(sort -u "$PBS_NODEFILE" | awk -F. '{print $1}') )
N0=${NODES[0]}
N1=${NODES[1]}

echo "=== EP=16 WS10 validate hold start: $(date) ==="
echo "Job: ${PBS_JOBID}"
echo "Nodes: ${N0} ${N1}"

# WS10 requires TORCHTUNE_EP_WSYNC_SHARDED=1 and the per-rank gloo sharded PGs.
# Method=gloo (default): cross-node Slingshot via gloo TCP (hsn0).
# DO NOT set TORCHTUNE_EP_WSYNC_SHARDED_METHOD=xccl until gloo is validated —
# XCCL on wsync PGs has had OFI CQ deadlock issues at op #259.

ssh -o StrictHostKeyChecking=no "${N0}" "
    cd ${PROJDIR}
    export TRAIN_NODE2=${N1}
    export NSTEPS=${NSTEPS:-5}
    export CONFIG=${PROJDIR}/recipes/configs/dev/experimental/qwen3_30b_a3b_grpo_ep16_gsm8k_xpu.yaml
    export TORCHTUNE_EP_GRAD_RELEASE_XCCL=1
    export TORCHTUNE_USE_IPEX_VARLEN=1
    export TORCHTUNE_MASKFREE_CAUSAL=1
    export TORCHTUNE_EP_WSYNC_SHARDED=1
    export TORCHTUNE_EP_WSYNC_SHARDED_METHOD=gloo
    export SMOKE_TAG=ep16_ws10_validate
    nohup bash ${PROJDIR}/recipes/dev/run_qwen3_30b_ep16_vllm_2node.sh \${NSTEPS} \
        > ${OUTDIR}/ep16_ws10_validate_run.log 2>&1 &
    echo \"run PID \$!\"
" || echo "WARN: SSH launch returned non-zero"

echo ""
echo "Run self-launched. Holding 60 min for walltime coverage."
echo "Monitor acceptance gates with:"
echo "  grep -E 'expert_map|NOT bumping|BATCH_REWARD|resp_len|WS10|wsync' ${OUTDIR}/ep16_ws10_validate_run.log | tail -30"
sleep 3600

echo "=== EP=16 WS10 validate hold end: $(date) ==="

# --- Post-run acceptance gate summary ---
echo ""
echo "=== WS10 ACCEPTANCE GATE CHECK ==="
LOG=${OUTDIR}/ep16_ws10_validate_run.log
if [ ! -f "${LOG}" ]; then
    echo "ERROR: run log not found: ${LOG}"
    exit 1
fi

GATE_A=$(grep -c "expert_map; got None" "${LOG}" 2>/dev/null || echo 0)
GATE_B=$(grep -c "NOT bumping weight version" "${LOG}" 2>/dev/null || echo 0)
GATE_C=$(grep "BATCH_REWARD" "${LOG}" 2>/dev/null | awk -F'reward_mean=' '{print $2}' | awk '{print $1}' | grep -v '^0\.0000$' | wc -l || echo 0)
GATE_D=$(grep "resp_len" "${LOG}" 2>/dev/null | grep -v "resp_len=31\." | grep -v "resp_len=255\." | wc -l || echo 0)

echo "(a) expert_map errors (want 0):          ${GATE_A}"
echo "(b) NOT bumping weight version (want 0):  ${GATE_B}"
echo "(c) non-zero reward_mean lines (want >0): ${GATE_C}"
echo "(d) resp_len not pinned to cap (want >0): ${GATE_D}"
echo ""

if [ "${GATE_A}" -eq 0 ] && [ "${GATE_B}" -eq 0 ] && \
   [ "${GATE_C}" -gt 0 ] && [ "${GATE_D}" -gt 0 ]; then
    echo "=== WS10 VALIDATED: all 4 acceptance gates PASSED ==="
else
    echo "=== WS10 NOT VALIDATED: one or more acceptance gates FAILED ==="
fi

echo ""
echo "WS10 wsync timing from log:"
grep -E "wsync_gather|wsync_bcast|WS10" "${LOG}" | tail -20 || echo "(none found)"

#!/bin/bash
# PROD faithful RL run: BioReason 4N HSDP, long, from the SFT base, with the
# validated throughput + stability fixes (stop_token_ids, checkpoint FULL_STATE_DICT,
# HSDP, max_gen=1024, reward go_ids+DAG). Saves the LoRA adapter+projectors every
# SAVE_EVERY_N_STEPS so we can eval F_max vs SFT 0.414 at intervals.
#
# THE GOAL (#27/#44): does our GRPO measurably improve F_max over the SFT 0.414 baseline?
#
# Queue: capacity (168h max). Default walltime sized for ~NSTEPS at the validated
# step time. Resume a prior adapter with RESUME_ADAPTER=<dir>.
#
# DO NOT submit until the stop-token A/B (8556890) confirms GREEN + no reward
# degradation. Submit with:
#   qsub -v NSTEPS=400,SAVE_EVERY_N_STEPS=50 experiments/bioreason/batch_prod_rl_4n.sh
#
# Walltime sized for the step budget: at ~130-170s/step (4N, post stop-fix), 200 steps
# = ~7-9.5h. 12h gives headroom + vLLM boot. save_every_n_steps + RESUME_ADAPTER make a
# walltime cutoff non-catastrophic (resume the last saved adapter in a follow-up job).
#PBS -l select=4
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q capacity
#PBS -N br_prod_rl_4n
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_prod_rl_4n.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason PROD RL 4N start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
# 200 steps: AGPT-2B needed ~225 to climb past the step-ceiling plateau; 200 is a
# reasonable first prod target within a 12h walltime. Bump via -v NSTEPS= + resume.
export NSTEPS=${NSTEPS:-200}
# G=24 (paper group size) — HW-validated 2026-06-23 (job 8557139) as the BETTER config
# than G=8: feasible (no OOM), sub-linear cost (1.1x wall for 1.5x rollouts), and a
# richer non-collapsing advantage signal (group_std 0.12-0.19 vs G=8's 0.03-0.14 that
# frequently degenerates). bs=1 keeps num_seqs/replica=24. See
# memory/project_bioreason_G_not_memory_capped_20260623.
export GRPO_SAMPLES=${GRPO_SAMPLES:-24}
export BATCH_SIZE=${BATCH_SIZE:-1}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
# ref_fbs=16 (NOT BATCH_SIZE*GRPO_SAMPLES=24) — caps ref-forward logits at ~15 GiB in
# chunked cycles. The old G=24 OOM came from ref_fbs>=num_seqs (45 GiB). Do NOT auto-
# derive ref_fbs from G here.
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
# stop tokens ON (the validated throughput fix) — explicit so it's unambiguous.
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}
# Distinct wsync path so this prod run can run CONCURRENTLY with any A/B 4N job
# without clobbering weight_update.raw on the shared Lustre path.
export WSYNC_PATH=${WSYNC_PATH:-/lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync_prod/weight_update.raw}

# Checkpoint cadence + output dir for the eval handoff.
export SAVE_EVERY_N_STEPS=${SAVE_EVERY_N_STEPS:-50}
export OUTPUT_DIR=${OUTPUT_DIR:-$TT_DIR/outputs/bioreason_prod_rl_4n_$(date +%Y%m%d)}
export RESUME_ADAPTER=${RESUME_ADAPTER:-}

echo "PROD RL: NSTEPS=$NSTEPS save_every=$SAVE_EVERY_N_STEPS out=$OUTPUT_DIR resume=${RESUME_ADAPTER:-<fresh>}"

# Retry once on a fast (vLLM boot-flake) failure; a slow failure is a real run error.
BOOT_WINDOW=${BOOT_WINDOW:-720}
RC=1
for attempt in 1 2; do
    t0=$(date +%s)
    bash "$TT_DIR/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
    RC=$?
    dt=$(( $(date +%s) - t0 ))
    [ $RC -eq 0 ] && break
    if [ $dt -ge $BOOT_WINDOW ]; then
        echo "=== attempt $attempt failed rc=$RC after ${dt}s (>boot window) — real failure, NOT retrying ==="
        break
    fi
    echo "=== attempt $attempt failed rc=$RC after ${dt}s (<boot window) — likely vLLM boot flake, retrying ==="
    sleep 20
done
echo "=== BioReason PROD RL 4N end rc=$RC $(date) ==="
exit $RC

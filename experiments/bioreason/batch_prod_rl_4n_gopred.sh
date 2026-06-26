#!/bin/bash
# PROD faithful RL run WITH go_pred prompt injection (the train/SFT/eval distribution
# match, 2026-06-26). Same 4N HSDP stack as batch_prod_rl_4n.sh, but:
#   - dataset.inject_go_pred=true (the SFT/eval-matched prompt; the leading fix for the
#     flat-vs-SFT result — our prod run scored 0.656 vs SFT 0.657 on a COLD prompt).
#   - raised seq budget so the longer go_pred prompt (p95 ~4290, max ~6100 tok) is NEVER
#     truncated: max_seq_len=6144, vllm_max_model_len=7168 (=6144 prompt + 1024 gen).
#     Both passed via EXTRA_OVERRIDES because the launcher only sends --max-model-len to
#     the vLLM SERVER, not the recipe config.
#
# THE GOAL (#27/#44): does GRPO on the SFT/eval-matched prompt measurably improve F_max
# over the SFT 0.657 baseline (where the cold-prompt run did NOT)?
#
# GATE: do NOT submit until batch_go_pred_smoke.sh confirms GREEN (no truncation/OOM,
# reward fires, ratios=1.0). Submit with:
#   qsub -v NSTEPS=200,SAVE_EVERY_N_STEPS=50 experiments/bioreason/batch_prod_rl_4n_gopred.sh
#
#PBS -l select=4
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q capacity
#PBS -N br_prod_rl_4n_gopred
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_prod_rl_4n_gopred.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason PROD RL 4N go_pred start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export NSTEPS=${NSTEPS:-200}
export GRPO_SAMPLES=${GRPO_SAMPLES:-24}
export BATCH_SIZE=${BATCH_SIZE:-1}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}

# Raised seq budget for the go_pred prompt (see header). MUST agree across the three.
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-7168}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-6144}
# go_pred ON (it's also the config default now, but make it explicit + unambiguous) and
# pin the seq budget into the recipe config (the launcher only sizes the vLLM server).
export EXTRA_OVERRIDES="dataset.inject_go_pred=true max_seq_len=${MAX_SEQ_LEN} vllm_max_model_len=${VLLM_MAX_MODEL_LEN} ${EXTRA_OVERRIDES:-}"

# Distinct wsync path so this can run concurrently with any other 4N job.
export WSYNC_PATH=${WSYNC_PATH:-/lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync_prod_gopred/weight_update.raw}

export SAVE_EVERY_N_STEPS=${SAVE_EVERY_N_STEPS:-50}
export OUTPUT_DIR=${OUTPUT_DIR:-$TT_DIR/outputs/bioreason_prod_rl_4n_gopred_$(date +%Y%m%d)}
export RESUME_ADAPTER=${RESUME_ADAPTER:-}

echo "PROD RL go_pred: NSTEPS=$NSTEPS save_every=$SAVE_EVERY_N_STEPS out=$OUTPUT_DIR"
echo "  EXTRA_OVERRIDES=$EXTRA_OVERRIDES"

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
echo "=== BioReason PROD RL 4N go_pred end rc=$RC $(date) ==="
exit $RC

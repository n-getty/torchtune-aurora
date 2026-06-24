#!/bin/bash
# One-shot PBS batch: reward-fix validation smoke, runs the launcher INSIDE the job
# (immune to login->compute SSH drops + requeue churn that killed the interactive runs).
# Safe envelope for the UR:40-in-backward wall: FBS=1, per-chunk backward
# (TORCHTUNE_USE_CHUNKED_LOSS=0), max_gen=2048 (so the model emits GO terms), G=8.
# Self-terminates (exits with training rc) so PBS tears down immediately.
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug
#PBS -N br_batched_smoke
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_batched_smoke.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
echo "=== reward-fix batch start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export NSTEPS=${NSTEPS:-8}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-8}    # distinct prompts/step — the Phase-A widening
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-2048}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-6144}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}  # >= B*G
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}

bash "$TT/experiments/bioreason/run_bioreason_2node_server.sh"
RC=$?
echo "=== reward-fix batch end rc=$RC $(date) ==="
exit $RC

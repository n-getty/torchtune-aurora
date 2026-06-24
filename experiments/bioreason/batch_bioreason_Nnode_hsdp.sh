#!/bin/bash
# One-shot PBS batch: BioReason N-node HSDP (centralized vLLM). Runs the launcher
# INSIDE the job (immune to login->compute SSH drops + requeue churn that kills
# interactive runs). Self-terminates (exits with training rc) so PBS tears down
# immediately.
#
# TOPOLOGY: select=4 -> 3 train (dp_replicate=3) + 1 vLLM. Scale by overriding
# select=<N> at qsub (run_bioreason_Nnode_hsdp.sh derives N-1 train + 1 vLLM).
#
# All env knobs use ${VAR:-default} so `qsub -v VAR=val` overrides work.
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_hsdp_4n
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_bioreason_Nnode_hsdp.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason N-node HSDP batch start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=${ENABLE_LORA:-1}
export NSTEPS=${NSTEPS:-8}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}                 # distinct prompts/step per replica
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-2048}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-6144}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}  # >= B*G
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
# DP_REPLICATE defaults to (#nodes - 1) inside the launcher; override only if you
# deliberately shrink the train-node set.
export DP_REPLICATE=${DP_REPLICATE:-}

bash "$TT_DIR/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
RC=$?
echo "=== BioReason N-node HSDP batch end rc=$RC $(date) ==="
exit $RC

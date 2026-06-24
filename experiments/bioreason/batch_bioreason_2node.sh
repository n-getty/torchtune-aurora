#!/bin/bash
# Self-terminating PBS batch wrapper for BioReason 2-node server-mode GRPO.
#
# Submit with -v overrides, e.g.:
#   qsub -q debug-scaling -l walltime=00:50:00 \
#        -v NSTEPS=10,GRPO_SAMPLES=8,FORWARD_BATCH_SIZE=4 \
#        experiments/bioreason/batch_bioreason_2node.sh
#
#   qsub -q capacity -l walltime=04:00:00 \
#        -v NSTEPS=200,GRPO_SAMPLES=8,FORWARD_BATCH_SIZE=4 \
#        experiments/bioreason/batch_bioreason_2node.sh
#
# The wrapped run_bioreason_2node_server.sh exits with the training process's
# exit code, so PBS will tear down the allocation as soon as training finishes
# (no hold / no manual qdel required).
#PBS -l select=2
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -N br2n_batch
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_bioreason_2node.out

set -o pipefail

TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TT_DIR}"

echo "=== BioReason 2-node batch start: $(date) ==="
echo "Job: ${PBS_JOBID:-<no PBS_JOBID>}  Host: $(hostname)"
echo "Nodes: $(cat ${PBS_NODEFILE} | sort -u | tr '\n' ' ')"
echo "Overrides: NSTEPS=${NSTEPS:-default} GRPO_SAMPLES=${GRPO_SAMPLES:-default} FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-default} MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-default}"

# Forward all configurable knobs already understood by the launcher.
# (The launcher uses ${VAR:-default} so unset vars take its defaults.)
export NSTEPS GRPO_SAMPLES FORWARD_BATCH_SIZE MAX_GEN_TOKENS
export VLLM_DP VLLM_GPU_MEM TRAIN_TILES MODEL_SRC MODEL_PATH
export TORCHTUNE_USE_IPEX_VARLEN WSYNC_TOPOLOGY WSYNC_CROSS_METHOD WSYNC_INTRA_METHOD
export BIOREASON_SRC BIOREASON_DEPS EXTRA_OVERRIDES
# LoRA run selectors (ENABLE_LORA=1 picks the LoRA server config; CONFIG/
# VLLM_MAX_MODEL_LEN allow explicit override).
export ENABLE_LORA CONFIG VLLM_MAX_MODEL_LEN REF_FORWARD_BATCH_SIZE

bash experiments/bioreason/run_bioreason_2node_server.sh
RC=$?

echo "=== BioReason 2-node batch end: rc=${RC} at $(date) ==="
exit ${RC}

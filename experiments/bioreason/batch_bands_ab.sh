#!/bin/bash
# One-shot PBS batch: BioReason 4N HSDP straggler-band A/B (same-node, back-to-back).
#
# Runs the SAME vLLM pool through two mpiexec legs on the SAME nodes:
#   leg 1: TORCHTUNE_VLLM_REPLICA_BANDS=0 (OLD buggy path — all 3 leaders pile on
#          engines 0..3, idles 4..11)
#   leg 2: TORCHTUNE_VLLM_REPLICA_BANDS=1 (FIXED — disjoint 4-engine bands per replica)
# Same-node back-to-back makes the gen-time delta immune to Aurora's ~1.8x
# node-to-node variance (the only valid way to claim the fix's % win).
#
# Faithful throughput envelope: max_gen=1024, VLLM_MAX_MODEL_LEN=4096 (matches the
# validated config). 8 steps/leg => ~stable median over 6 steady-state intervals.
#
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_bands_ab
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_bands_ab.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason bands A/B batch start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export NSTEPS=${NSTEPS:-8}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
# Run BOTH legs (OFF then ON) in one job, same vLLM pool.
export AB_BANDS="0 1"

bash "$TT_DIR/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
RC=$?
echo "=== BioReason bands A/B batch end rc=$RC $(date) ==="
exit $RC

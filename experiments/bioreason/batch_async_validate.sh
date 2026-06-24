#!/bin/bash
# One-shot PBS batch: BioReason 2N async generation/training overlap CORRECTNESS gate.
#
# Single-replica 2N (1 train + 1 vLLM) — async is HSDP-DISABLED by design (the
# lookahead assumes single-generator rank-0 + world broadcast), so this is the
# topology where async applies. Validates the gate before any timing claim:
#   - runs >=12 steps clean (no banned:1 / wedge / deadlock at mailbox maxsize=1)
#   - ratios ~1.0 at staleness=1 (rollout 1 wsync stale; GRPOLoss IS recompute)
#   - RolloutProducer telemetry (produced/consumed/qsize, blocked_on_put,
#     get_wait) shows actual gen/train OVERLAP, not serialization
#   - check_run_health GREEN
#
# CONFIG = the async YAML (async_generation.enabled=true, loss=GRPOLoss,
# always_compute_rollout_logprobs=true). max_gen=1024 faithful throughput envelope.
#
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug
#PBS -N br_async_val
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_async_validate.out

set -o pipefail
TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
cd "${TT_DIR}"

echo "=== BioReason 2N async validate start: $(date) job=${PBS_JOBID} ==="
echo "Nodes: $(cat ${PBS_NODEFILE} | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export CONFIG=recipes/configs/dev/production/bioreason_4b_lora_grpo_2node_server_xpu_async.yaml
export NSTEPS=${NSTEPS:-12}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}

# Forward all knobs the 2N launcher understands.
export VLLM_DP VLLM_GPU_MEM TRAIN_TILES MODEL_SRC MODEL_PATH
export TORCHTUNE_USE_IPEX_VARLEN WSYNC_TOPOLOGY WSYNC_CROSS_METHOD WSYNC_INTRA_METHOD
export BIOREASON_SRC BIOREASON_DEPS EXTRA_OVERRIDES

bash experiments/bioreason/run_bioreason_2node_server.sh
RC=$?
echo "=== BioReason 2N async validate end: rc=${RC} at $(date) ==="
exit ${RC}

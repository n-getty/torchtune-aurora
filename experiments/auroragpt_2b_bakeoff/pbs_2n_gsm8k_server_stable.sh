#!/bin/bash
#PBS -N agpt2b_gsm8k_stable
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/auroragpt_2b_bakeoff/logs/pbs_2n_gsm8k_server_stable.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/auroragpt_2b_bakeoff/logs/pbs_2n_gsm8k_server_stable.err

# 2N AGPT-2B GRPO on GSM8K — STABLE hparam variant.
#
# Job 8538544 (2026-06-12) proved both wsync-leak and QK fixes work
# (mem flat at 30.2 GiB across step 60; previously leaked to 48 GiB
# by step 49 → UR40 crash). The remaining failure mode at the baseline
# lr=1e-5 + kl_coeff=0 was NaN-induced collapse around step 40 from
# runaway KL drift. This launcher uses the stable YAML variant:
#   - lr 5e-6 (halved)
#   - kl_coeff 0.02 (KL term in gradient)
#   - warmup 10 (doubled)
# (epsilon_high not used: GRPOSimpleLoss ignores pi_old; lives on GRPOLoss.)
#
# Compare runs at:
#   experiments/auroragpt_2b_bakeoff/logs/gsm8k_2n_server_stable_*

set -eo pipefail

REPO=/lus/flare/projects/ModCon/ngetty/torchtune
LAUNCHER=$REPO/experiments/lora_grpo/run_qwen3_4b_dense_2node.sh
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=$REPO/experiments/auroragpt_2b_bakeoff/logs/gsm8k_2n_server_stable_${TS}
mkdir -p "$LOGDIR"

# --- AGPT-2B specific overrides -------------------------------------------
export CONFIG=recipes/configs/dev/production/auroragpt_2b_grpo_2n_gsm8k_server_xpu_stable.yaml
export MODEL_PATH=/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650

# --- RL envelope (matches the colocate-2N YAML) ---------------------------
export NSTEPS=${NSTEPS:-150}
export GRPO_SAMPLES=${GRPO_SAMPLES:-16}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-8}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-8}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-512}
# True chunked train backward avoids the deterministic step-62 single-backward wall.
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}

# --- vLLM topology -------------------------------------------------------
export VLLM_DP=${VLLM_DP:-12}
export VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-1024}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.40}
export TRAIN_TILES=${TRAIN_TILES:-11}
export VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-1200}

# --- Fast-path env (KEEP varlen bypass OFF for un-converged ref) ----------
export TORCHTUNE_USE_IPEX_VARLEN=${TORCHTUNE_USE_IPEX_VARLEN:-1}
# DO NOT enable on AGPT-2B until SFT'd on GSM8K. See
# memory/feedback_varlen_nograd_bypass_unsafe_on_unconverged_ref.md
export TORCHTUNE_VARLEN_NOGRAD_BYPASS=${TORCHTUNE_VARLEN_NOGRAD_BYPASS:-0}
export TORCHTUNE_PINNED_CPU_BUF=${TORCHTUNE_PINNED_CPU_BUF:-1}

# --- L0-leak mitigation knob (off by default) ---------------------------
# Job 8538544 showed memory FLAT at 30.2 GiB without the varlen bypass —
# the visible HBM stays flat. But three runs (8538544/8538722/8538788) all
# crashed at exactly step 62 with the XCCL "ref_fwd staircase" stall
# regardless of hparams or wsync rate. That's the SAME class of CXI MR
# leak that capped Qwen3-32B XCCL-cross runs at ~30 steps. The 32B 3-node
# launcher fixed it by switching the cross-node wsync transport from
# XCCL/RDMA to Gloo/TCP-over-hsn0 (memory/project_gloo_cross_pg_fix.md;
# docs/features/qwen3_32b_dense_grpo.md).
#
# Default ON here for the AGPT-2B 2N path. Override with
#   WSYNC_CROSS_METHOD=xccl_sendrecv qsub ...
# to reproduce the step-62 crash for comparison.
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-gloo}
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD:-xccl}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}

export VLLM_WSYNC_INTERVAL=${VLLM_WSYNC_INTERVAL:-1}

# --- Output overrides ----------------------------------------------------
export EXTRA_OVERRIDES="output_dir=${LOGDIR}/run_out vllm_weight_sync_interval=${VLLM_WSYNC_INTERVAL} ${EXTRA_OVERRIDES_APPEND:-}"

echo "=== AGPT-2B GSM8K 2N SERVER STABLE ===" | tee "${LOGDIR}/launcher.log"
echo "  TS=$TS  LOGDIR=$LOGDIR" | tee -a "${LOGDIR}/launcher.log"
echo "  CONFIG=$CONFIG" | tee -a "${LOGDIR}/launcher.log"
echo "  hparams: lr=5e-6 kl_coeff=0.02 warmup=10" | tee -a "${LOGDIR}/launcher.log"
echo "  G=$GRPO_SAMPLES  fbs=$FORWARD_BATCH_SIZE  max_gen=$MAX_GEN_TOKENS  steps=$NSTEPS" | tee -a "${LOGDIR}/launcher.log"
echo "  VLLM_DP=$VLLM_DP  TRAIN_TILES=$TRAIN_TILES  WSYNC_INTERVAL=$VLLM_WSYNC_INTERVAL" | tee -a "${LOGDIR}/launcher.log"
echo "  PBS_NODEFILE=$PBS_NODEFILE:" | tee -a "${LOGDIR}/launcher.log"
cat "$PBS_NODEFILE" | tee -a "${LOGDIR}/launcher.log"

bash "$LAUNCHER" 2>&1 | tee -a "${LOGDIR}/launcher.log"
rc=${PIPESTATUS[0]}
echo "=== launcher exit rc=$rc at $(date) ===" | tee -a "${LOGDIR}/launcher.log"
exit $rc

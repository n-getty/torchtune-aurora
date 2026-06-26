#!/bin/bash
# One-shot 2N smoke: validate the go_pred prompt-injection RL path end-to-end on HW,
# as an A/B vs the old cold prompt (same nodes, same N, back-to-back).
#
# WHY (2026-06-26): the SFT ckpt was trained to REFINE GO-GPT's go_pred (injected as
# go_speculations) and the fixed eval injects it (lifted published SFT 0.41 -> ~0.65).
# RL trained on a COLD prompt (no go_pred) -> off-distribution from both SFT and eval ->
# flat eval F_max. dataset.inject_go_pred=true now builds the SAME paper-faithful prompt.
#
# This smoke confirms: (1) the longer go_pred prompt (p95 ~4290 tok) fits the raised
# seq/vllm budget with NO truncation + NO OOM; (2) reward fires and the model can refine
# (reward should be >= the cold leg early, since go_pred seeds ~0.53-recall GO terms);
# (3) ratios=1.0, no banned:1/NaN. NOT a learning run — 8 steps each.
#
# Budget: go_pred adds the ~2200-char GO-GPT text. Protein stays at 2048 (cache is keyed
# at 2048 — CANNOT shrink without re-encoding). So raise max_seq_len 4096->6144 and
# vllm_max_model_len to 6144+1024=7168 so the prompt is never truncated.
#
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug
#PBS -N br_gopred_smoke
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_go_pred_smoke.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
echo "=== go_pred A/B smoke start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export NSTEPS=${NSTEPS:-8}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}
# Raised budget so the go_pred prompt is never truncated (p95 ~4290, max ~6100).
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-7168}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-6144}

CONFIG=recipes/configs/dev/production/bioreason_4b_lora_grpo_2node_server_xpu.yaml

run_leg() {
  local tag="$1"; local inject="$2"
  echo "================= LEG=$tag inject_go_pred=$inject $(date) ================="
  export WSYNC_PATH=$TT/outputs/wsync_gopred_${tag}/weight_update.raw
  # Override BOTH max_seq_len (dataset truncation, cascades via ${max_seq_len}) AND
  # vllm_max_model_len (recipe-side prompt truncation; the launcher only passes
  # --max-model-len to the vLLM SERVER, not to the recipe config) so neither truncates
  # the longer go_pred prompt. They must agree with the launcher's VLLM_MAX_MODEL_LEN.
  export EXTRA_OVERRIDES="dataset.inject_go_pred=${inject} max_seq_len=${MAX_SEQ_LEN} vllm_max_model_len=${VLLM_MAX_MODEL_LEN}"
  # The launcher writes a timestamped LOG per invocation (run_bioreason_2node_*.log),
  # so the two legs' logs are distinct even though TRAIN_LOG (/tmp) is reused.
  bash "$TT/experiments/bioreason/run_bioreason_2node_server.sh"
  local rc=$?
  echo "=== leg $tag rc=$rc ==="
  # Copy this leg's launcher LOG (shared FS, timestamped; the watcher SSH tees the
  # train-node Step lines into it) to a stable leg-tagged path so the orchestrator can
  # assess the gopredON leg specifically (the newest run_*.log after both legs is coldOFF).
  mkdir -p "$TT/experiments/bioreason/overnight_state"
  local _newest; _newest=$(ls -t $TT/experiments/bioreason/run_bioreason_2node_*.log 2>/dev/null | head -1)
  [ -n "$_newest" ] && cp -f "$_newest" \
        "$TT/experiments/bioreason/overnight_state/smoke_${tag}_launcher.log" 2>/dev/null || true
}

# Leg A: go_pred ON (the new path) first — if it OOMs/truncates we learn immediately.
run_leg gopredON true
# Leg B: cold prompt (baseline) for the reward A/B at the SAME envelope.
run_leg coldOFF false

echo "=== go_pred A/B smoke end $(date) ==="

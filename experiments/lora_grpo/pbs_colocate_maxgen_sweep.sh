#!/bin/bash
#PBS -N lora_mgsweep
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=00:50:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_mgsweep.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_mgsweep.err

# max_gen THRESHOLD SWEEP (2026-06-24) — diagnose the colocate CCS-NotPresent page fault.
# FINDING from 12 prior runs: crash correlates with per-rank GENERATED TOKENS, not memory/loss.
#   maxtok <= 2048 (mg256) -> survives (up to 29 steps clean).
#   maxtok >= 4600 (mg1024) -> CCS NotPresent banned:1 page fault at step ~7-8, right after a
#   long-rollout burst. SAME fault in GRPOLoss AND LinearGRPOLoss runs (NOT our code, NOT OOM).
# 1-node (dodges the 2N TCPStore flakiness seen all day). GRPOSimpleLoss (default path, isolates
# the fault from LinearGRPOLoss entirely). WARMUP_AT_MAX=1 forces a full-length ignore_eos gen at
# STEP 0 so the per-rank token ceiling is hit DETERMINISTICALLY at step 0 (not a random step-8 draw).
# Sweep max_gen across the suspected 2048->4600 boundary:
#   leg mg256: expect CLEAN (control; total seq ~ prompt+256, well under).
#   leg mg512: ?  leg mg768: ?  -> brackets the threshold.
# If a leg crashes at STEP 0 warmup -> CONFIRMS per-rank-token ceiling + locates it.
# If all clear 10 steps -> threshold is higher / fault is a random-draw transient (re-think).
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_mgsweep_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== mgsweep job up $(date) | jobid=$PBS_JOBID | 1 node ==="; cat "$NF"

run_leg() {
  local MG="$1"
  local LOG="$TT/experiments/lora_grpo/logs/pbs_mgsweep_mg${MG}.log"
  echo; echo "########## LEG max_gen=${MG} $(date) ##########"
  env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
      MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
      NSTEPS=10 NTILES=12 MAX_GEN="${MG}" VLLM_GPU_MEM=0.30 \
      TORCHTUNE_USE_IPEX_VARLEN=1 \
      TORCHTUNE_VARLEN_CACHE_MAX=8 \
      TORCHTUNE_COLOCATE_MEM_PROBE=1 \
      TORCHTUNE_COLOCATE_CACHED_BASE=1 \
      TORCHTUNE_COLOCATE_NO_FSDP=1 \
      TORCHTUNE_COLOCATE_BASE_CPU=0 \
      TORCHTUNE_COLOCATE_WARMUP_AT_MAX=1 \
      TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
      TORCHTUNE_USE_CHUNKED_LOSS=1 \
      EXTRA_OVERRIDES="forward_batch_size=1 ref_forward_batch_size=2" \
      PBS_NODEFILE="$NF" \
    bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
  local rc=$?
  echo "leg mg${MG} rc=${rc} $(date)"
  local RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
  echo "  steps=$(grep -oE 'REWARDS step=[0-9]+' "$RL" 2>/dev/null | sed 's/.*=//' | sort -n | tail -1)"
  echo "  max gen tokens (8-seq total): $(grep -oE 'generated 8 sequences, [0-9]+ tokens' "$RL" 2>/dev/null | grep -oE '[0-9]+ tok' | grep -oE '[0-9]+' | sort -rn | head -1)"
  echo "  warmup reached train loop: $(grep -c 'warmup-at-max' "$RL" 2>/dev/null)"
  echo "  CRASH: $(grep -cE 'banned: 1|Segmentation fault from GPU' "$RL" 2>/dev/null) | fault: $(grep -oE 'level: [01] \(P[TD]E\)' "$RL" 2>/dev/null | head -1)"
  echo "  L0 free min: $(grep -oE 'L0 free [0-9.]+->[0-9.]+' "$RL" 2>/dev/null | sed 's/.*->//' | awk '$1>1' | sort -n | head -1)"
}

run_leg 256
run_leg 512
run_leg 768

echo; echo "================ mgsweep SUMMARY $(date) ================"
echo "Correlate: which max_gen first crashes, and at what per-rank token count + which step."
echo "=== mgsweep DONE $(date) ==="

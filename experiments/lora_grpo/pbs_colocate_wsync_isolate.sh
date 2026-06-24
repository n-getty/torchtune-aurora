#!/bin/bash
#PBS -N lora_wsynciso
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_wssynciso.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_wsynciso.err

# WEIGHT-SYNC ISOLATION (2026-06-24) — decisive test of the mg>256 colocate page fault.
# Sweep 8559141 established: mg256 clean 9 steps; mg512/768 crash at step 0-1 with CCS PDE,
# AFTER warmup's full-length generate SURVIVED. The only thing between warmup's (safe) generate
# and the step-0 (crashing) generate is the LoRA adapter PUBLISH (load_weights into vLLM,
# publish_every_steps=1). HYPOTHESIS: the fault is a vLLM generate that FOLLOWS a load_weights
# at KV size > the tuned 256.
# TEST: same mg512, two legs:
#   leg A: publish_every_steps=999 (NO weight sync after step 0) -> if it SURVIVES 9 steps,
#          CONFIRMS load_weights+generate is the trigger (not KV size alone).
#   leg B: publish_every_steps=1 (control) -> expect crash step 0-1 (reproduces 8559141 mg512).
# 1-node, GRPOSimpleLoss (default path), warmup ON. Decisive single comparison.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_wsynciso_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== wsynciso job up $(date) | jobid=$PBS_JOBID | 1 node ==="; cat "$NF"

run_leg() {
  local TAG="$1"; local PUB="$2"
  local LOG="$TT/experiments/lora_grpo/logs/pbs_wsynciso_${TAG}.log"
  echo; echo "########## LEG ${TAG}: publish_every_steps=${PUB} (mg512) $(date) ##########"
  env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
      MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
      NSTEPS=10 NTILES=12 MAX_GEN=512 VLLM_GPU_MEM=0.30 \
      TORCHTUNE_USE_IPEX_VARLEN=1 \
      TORCHTUNE_VARLEN_CACHE_MAX=8 \
      TORCHTUNE_COLOCATE_MEM_PROBE=1 \
      TORCHTUNE_COLOCATE_CACHED_BASE=1 \
      TORCHTUNE_COLOCATE_NO_FSDP=1 \
      TORCHTUNE_COLOCATE_BASE_CPU=0 \
      TORCHTUNE_COLOCATE_WARMUP_AT_MAX=1 \
      TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
      TORCHTUNE_USE_CHUNKED_LOSS=1 \
      EXTRA_OVERRIDES="forward_batch_size=1 ref_forward_batch_size=2 lora.publish_every_steps=${PUB}" \
      PBS_NODEFILE="$NF" \
    bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
  echo "leg ${TAG} rc=$? $(date)"
  local RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
  echo "  steps=$(grep -oE 'REWARDS step=[0-9]+' "$RL" 2>/dev/null | sed 's/.*=//' | sort -n | tail -1)"
  echo "  CRASH: $(grep -cE 'banned: 1' "$RL" 2>/dev/null) | fault: $(grep -oE 'level: [01] \(P[TD]E\)' "$RL" 2>/dev/null | head -1)"
  echo "  publishes done: $(grep -cE 'colocate LoRA wsync|ADAPTER published|load_weights' "$RL" 2>/dev/null)"
}

run_leg A_nosync 999
run_leg B_sync 1

echo; echo "================ wsynciso SUMMARY $(date) ================"
echo "A_nosync survives + B_sync crashes => load_weights+generate at mg>256 IS the trigger."
echo "Both crash => KV size alone (not weight sync) -> different fix (vllm_max_model_len / blocks)."
echo "=== wsynciso DONE $(date) ==="

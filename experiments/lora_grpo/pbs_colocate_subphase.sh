#!/bin/bash
#PBS -N lora_subphase
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_subphase.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_subphase.err

# SUB-PHASE ACTIVE localization (2026-06-24).
# Job 8558618 proved the ~0.44 GiB/step ACTIVE creep is ENTIRELY in the gen phase
# (sync=+0.000, grpo=+0.062 const). The gen phase contains: the vLLM generate call,
# a rollout-logprob policy fwd (maybe), and the ref fwd. SUBPROBEs now log ACTIVE for
# vllm_generate and ref_fwd. This run pins WHICH sub-call retains the +0.44 live:
#   vllm_generate ACTIVE +0.44  -> vLLM generate retains live device buffers (KV/out)
#   ref_fwd       ACTIVE +0.44  -> the torch ref forward retains (our code / model)
#   neither (~0)              -> it's in trajectory construction between them
# 14 steps, mg256, NO_FSDP=1. debug-scaling (free).
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_subphase_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== subphase job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"
LOG="$TT/experiments/lora_grpo/logs/pbs_subphase.log"
env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
    MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
    NSTEPS=14 NTILES=12 MAX_GEN=256 VLLM_GPU_MEM=0.30 \
    TORCHTUNE_COLOCATE_MEM_PROBE=1 \
    TORCHTUNE_COLOCATE_CACHED_BASE=1 \
    TORCHTUNE_COLOCATE_NO_FSDP=1 \
    TORCHTUNE_COLOCATE_BASE_CPU=0 \
    TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0 \
    TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
    EXTRA_OVERRIDES="forward_batch_size=2 ref_forward_batch_size=4" \
    PBS_NODEFILE="$NF" \
  bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
echo "subphase done rc=$? $(date)"
RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
echo "  run log: $RL  steps=$(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
echo "=== *** SUBPROBE ACTIVE (which sub-call retains +0.44/step?) *** ==="
grep "COLOCATE_SUBPROBE" "$RL" 2>/dev/null | grep "ACTIVE" | awk '!seen[$0]++'
echo "=== subphase DONE $(date) ==="

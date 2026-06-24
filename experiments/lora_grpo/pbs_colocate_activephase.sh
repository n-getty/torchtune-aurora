#!/bin/bash
#PBS -N lora_actphase
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_actphase.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_actphase.err

# ACTIVE-memory phase localization (2026-06-24).
# A/B 8558600 proved the ~0.44 GiB/step colocate creep is per-train-step, in ACTIVE
# (live) memory, empty_cache-count-independent, C++/vLLM-held. Reserved phase probes
# were blind to it. Phase probes now ALSO log ACTIVE delta per phase (gen/grpo/sync).
# This run reads which phase carries the +0.44 ACTIVE/step. That phase = the leak site.
#   gen   -> vLLM generation retains device buffers per call (KV/output not freed)
#   grpo  -> training fwd/bwd retains (unlikely; backward graph freed, would show in active)
#   sync  -> load_weights orphans prior param storage on XPU (the strong prior)
# 14 steps, mg256, NO_FSDP=1. debug-scaling (free).
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_actphase_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== actphase job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"
LOG="$TT/experiments/lora_grpo/logs/pbs_actphase.log"
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
echo "actphase done rc=$? $(date)"
RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
echo "  run log: $RL  steps=$(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
echo "=== *** ACTIVE delta per phase (which phase carries +0.44/step?) *** ==="
grep "COLOCATE_PHASEPROBE" "$RL" 2>/dev/null | grep -oE "step=[0-9]+ (gen|grpo_step|sync) .*ACTIVE [0-9.]+->[0-9.]+ \([+-][0-9.]+\)" | awk '!seen[$0]++'
echo "=== actphase DONE $(date) ==="

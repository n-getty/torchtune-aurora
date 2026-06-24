#!/bin/bash
#PBS -N lora_varlenfix
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:35:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_varlenfix.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_varlenfix.err

# FIX VALIDATION (2026-06-24): bounded _varlen_out_cache, varlen KEPT ON.
# Root cause confirmed: the IPEX-varlen no-grad output-buffer cache grew one
# ~14.7MiB buffer per distinct (b,s) FOREVER on variable-seqlen RL -> ~0.44 GiB/step
# leak in ref_fwd. Fix = FIFO-bounded cache (TORCHTUNE_VARLEN_CACHE_MAX, default 8)
# in attention_utils.py, keeping within-step reuse (the speedup).
# This run: varlen=1 (ON, speedup kept) + bounded cache.
#   EXPECT: ref_fwd ACTIVE ~+0.00/step, end-of-step ALLOC floor FLAT (creep GONE),
#           clears all 30 steps (vs ~50-step OOM before), step time ~unchanged.
# 30 steps, mg256, NO_FSDP=1. debug-scaling (free). The .py fix is read at runtime.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_varlenfix_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== varlenfix job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"
LOG="$TT/experiments/lora_grpo/logs/pbs_varlenfix.log"
env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
    MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
    NSTEPS=30 NTILES=12 MAX_GEN=256 VLLM_GPU_MEM=0.30 \
    TORCHTUNE_USE_IPEX_VARLEN=1 \
    TORCHTUNE_VARLEN_CACHE_MAX=8 \
    TORCHTUNE_COLOCATE_MEM_PROBE=1 \
    TORCHTUNE_COLOCATE_CACHED_BASE=1 \
    TORCHTUNE_COLOCATE_NO_FSDP=1 \
    TORCHTUNE_COLOCATE_BASE_CPU=0 \
    TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0 \
    TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
    EXTRA_OVERRIDES="forward_batch_size=2 ref_forward_batch_size=4" \
    PBS_NODEFILE="$NF" \
  bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
echo "varlenfix done rc=$? $(date)"
RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
echo "  run log: $RL  steps=$(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
echo "  varlen status: $(grep -oE 'varlen=(engaged|disabled[^ ]*)' "$RL" 2>/dev/null | head -1)"
echo "=== *** ref_fwd ACTIVE delta (EXPECT ~+0.00) *** ==="
grep "COLOCATE_SUBPROBE" "$RL" 2>/dev/null | grep ref_fwd | grep -oE "step=[0-9]+ ref_fwd .*ACTIVE [0-9.]+->[0-9.]+ \([+-][0-9.]+\)" | awk '!s[$0]++' | sed -n '1p;5p;10p;15p;20p;25p;29p'
echo "=== *** end-of-step ALLOC floor (EXPECT FLAT) *** ==="
grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null | grep final=1 | grep -oE "ALLOC=[0-9.]+ .* step=[0-9]+" | awk '!a[$0]++' | sed -n '1p;6p;11p;16p;21p;26p;29p'
echo "=== last reward / health ==="
grep -E "REWARDS step=" "$RL" 2>/dev/null | tail -1
grep -E "banned: 1|GREEN|DEGRADED" "$LOG" 2>/dev/null | tail -2
echo "=== varlenfix DONE $(date) ==="

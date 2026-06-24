#!/bin/bash
#PBS -N lora_varlen_ab
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_varlen_ab.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_varlen_ab.err

# =====================================================================
# DECISIVE varlen-cache CONFIRM A/B (2026-06-24)
# Chain of evidence: the ~0.44 GiB/step no-FSDP colocate creep is
#   - per-train-step, ACTIVE/live memory (job 8558600)
#   - entirely in the GEN phase, sync/grpo flat (job 8558618)
#   - entirely in the ref_fwd sub-call, vllm_generate flat +0.000 (job 8558640)
# ROOT CAUSE (found in code): torchtune/modules/attention_utils.py _varlen_out_cache
# (+ _alibi_cache + _seqlens_cache) — the IPEX-varlen no-grad output-buffer cache
# keyed by (b,h,s,d,dtype,dev). s=seqlen VARIES per step on GSM8K -> a NEW ~14.7MiB
# buffer x36 layers is cached EVERY step and NEVER evicted. Shape [b*s,32,128] ==
# the run-8 census growing tensor exactly. varlen is ON because _env.sh exported
# TORCHTUNE_USE_IPEX_VARLEN=1 unconditionally (now made overridable).
#
# This A/B PROVES it: same nodes, back-to-back.
#   LEG A  varlen=1 (current)  -> EXPECT ref_fwd ACTIVE +0.44/step (reproduce)
#   LEG B  varlen=0            -> EXPECT ref_fwd ACTIVE ~+0.00/step (creep GONE)
# DECISION: B flat -> _varlen_out_cache is THE leak. Fix = bound/evict the cache
#   (keep varlen speed, drop the unbounded retention). Then no respawn needed.
# 22 steps/leg, mg256, NO_FSDP=1. debug-scaling (free).
# =====================================================================
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_varlen_ab_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== varlen-ab job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"

run_leg() {
  local TAG="$1"; local VARLEN="$2"
  local LOG="$TT/experiments/lora_grpo/logs/pbs_varlen_${TAG}.log"
  echo; echo "########## LEG ${TAG}: TORCHTUNE_USE_IPEX_VARLEN=${VARLEN} $(date) ##########"
  env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
      MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
      NSTEPS=22 NTILES=12 MAX_GEN=256 VLLM_GPU_MEM=0.30 \
      TORCHTUNE_USE_IPEX_VARLEN="${VARLEN}" \
      TORCHTUNE_COLOCATE_MEM_PROBE=1 \
      TORCHTUNE_COLOCATE_CACHED_BASE=1 \
      TORCHTUNE_COLOCATE_NO_FSDP=1 \
      TORCHTUNE_COLOCATE_BASE_CPU=0 \
      TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0 \
      TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
      EXTRA_OVERRIDES="forward_batch_size=2 ref_forward_batch_size=4" \
      PBS_NODEFILE="$NF" \
    bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
  echo "leg ${TAG} done rc=$? $(date)"
  local RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
  echo "  run log: $RL  steps=$(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
  echo "  varlen status: $(grep -oE 'varlen=(engaged|disabled[^ ]*)' "$RL" 2>/dev/null | head -1)"
  echo "  --- ref_fwd ACTIVE delta (THE number) ---"
  grep "COLOCATE_SUBPROBE" "$RL" 2>/dev/null | grep "ref_fwd" | grep -oE "step=[0-9]+ ref_fwd .*ACTIVE [0-9.]+->[0-9.]+ \([+-][0-9.]+\)" | awk '!seen[$0]++'
  echo "  --- end-of-step ALLOC floor (creep slope) ---"
  grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null | grep "final=1" | grep -oE "ALLOC=[0-9.]+ .* step=[0-9]+" | awk '!s[$0]++' | sed -n '1p;6p;11p;16p;21p'
  eval "RL_${TAG}=$RL"; unset seen
}

run_leg A 1
run_leg B 0

echo; echo "================ varlen A/B SUMMARY $(date) ================"
echo "LEG A (varlen=1) ref_fwd ACTIVE deltas:"; grep "COLOCATE_SUBPROBE" "$RL_A" 2>/dev/null | grep ref_fwd | grep -oE "step=[0-9]+ .*\([+-][0-9.]+\)" | awk '!a[$0]++' | sed -n '2p;11p;21p'
echo "LEG B (varlen=0) ref_fwd ACTIVE deltas:"; grep "COLOCATE_SUBPROBE" "$RL_B" 2>/dev/null | grep ref_fwd | grep -oE "step=[0-9]+ .*\([+-][0-9.]+\)" | awk '!b[$0]++' | sed -n '2p;11p;21p'
echo "INTERPRET: B ~+0.00 and A ~+0.44 -> _varlen_out_cache CONFIRMED as the leak."
echo "=== varlen-ab DONE $(date) ==="

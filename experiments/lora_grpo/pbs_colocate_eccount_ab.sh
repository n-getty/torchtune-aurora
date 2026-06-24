#!/bin/bash
#PBS -N lora_ecount_ab
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=2
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_ecount_ab.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_ecount_ab.err

# =====================================================================
# DECISIVE empty_cache-CALL-COUNT A/B (2026-06-24)
# Question: is the no-FSDP colocate ~0.44 GiB/step ALLOC creep driven by
# the NUMBER of torch.xpu.empty_cache() calls (L0/driver reclamation
# residue) or is it per-TRAIN-STEP regardless (vLLM-internal)?
#
# Both legs KEEP reclamation alive (so both clear steps), but vary the
# count of empty_cache calls per step:
#   LEG A  RECLAIM_MODE=all  -> ~5 empty_cache/step (current behavior)
#   LEG B  RECLAIM_MODE=once -> 1 empty_cache/step (only end-of-step `final`)
#
# DECISION RULE (compare ALLOC creep slope, GiB/step, over steps 4..N):
#   slope_A ~= 5x slope_B  => creep scales with empty_cache CALLS
#                             => leak is empty_cache/L0 residue, NOT vLLM.
#                             => FIX = call empty_cache less (stride), no respawn.
#   slope_A ~= slope_B      => creep is per-train-step (vLLM-internal / KV).
#                             => respawn (W17/W19) really is the lever.
# Cross-check: ec_calls= in the COLOCATE_RECLAIM line confirms call counts
# (A should accumulate ~5x faster than B). Plot ALLOC vs ec_calls, not step.
#
# Same nodes, back-to-back legs (controls Aurora node variance). mg256,
# NO_FSDP=1, ~22 steps/leg (clean slope, below the ~step51 OOM). capacity q.
# =====================================================================
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_ecount_ab_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== ecount-ab job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"

COMMON_ENV() {
  env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
      MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
      NSTEPS=22 NTILES=12 MAX_GEN=256 VLLM_GPU_MEM=0.30 \
      TORCHTUNE_COLOCATE_MEM_PROBE=1 \
      TORCHTUNE_COLOCATE_CACHED_BASE=1 \
      TORCHTUNE_COLOCATE_NO_FSDP=1 \
      TORCHTUNE_COLOCATE_BASE_CPU=0 \
      TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0 \
      TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
      EXTRA_OVERRIDES="forward_batch_size=2 ref_forward_batch_size=4" \
      PBS_NODEFILE="$NF" "$@"
}

run_leg() {
  local TAG="$1"; local MODE="$2"
  local LOG="$TT/experiments/lora_grpo/logs/pbs_ecount_${TAG}.log"
  echo; echo "########## LEG ${TAG}: RECLAIM_MODE=${MODE} $(date) ##########"
  COMMON_ENV TORCHTUNE_COLOCATE_RECLAIM_MODE="${MODE}" \
    bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
  echo "leg ${TAG} done rc=$? $(date)"
  local RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
  echo "  run log: $RL"
  echo "  steps:   $(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
  echo "  --- ALLOC vs ec_calls (creep slope source) ---"
  grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null \
    | grep -oE "ALLOC=[0-9.]+ active=[0-9.]+ n_blocks=[0-9]+ seg=[0-9.]+ inact=[0-9.]+ retries=[0-9]+ ec_calls=[0-9]+ step=[0-9]+ final=[01]" \
    | grep "final=1" | awk '!seen[$0]++'
  echo "  --- last reward / crash ---"
  grep -E "REWARDS step=" "$RL" 2>/dev/null | tail -1
  grep -E "banned: 1|UR_RESULT|RuntimeError" "$LOG" 2>/dev/null | tail -2
  # stash for cross-leg compare
  eval "RL_${TAG}=$RL"
}

run_leg A all
run_leg B once

echo; echo "================ A/B SUMMARY $(date) ================"
echo "LEG A (all, ~5 ec/step)  final-site ALLOC trajectory:"
grep "COLOCATE_RECLAIM" "$RL_A" 2>/dev/null | grep "final=1" \
  | grep -oE "ALLOC=[0-9.]+ .* ec_calls=[0-9]+ step=[0-9]+" | awk '!seen[$0]++'
echo "LEG B (once, 1 ec/step)  final-site ALLOC trajectory:"
unset seen
grep "COLOCATE_RECLAIM" "$RL_B" 2>/dev/null | grep "final=1" \
  | grep -oE "ALLOC=[0-9.]+ .* ec_calls=[0-9]+ step=[0-9]+" | awk '!seen[$0]++'
echo "INTERPRET: if A slope ~= 5x B slope -> empty_cache/L0 residue (fix=stride);"
echo "           if A slope ~= B slope    -> per-step vLLM-internal (fix=respawn)."
echo "=== ecount-ab DONE $(date) ==="

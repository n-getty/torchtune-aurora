#!/bin/bash
#PBS -N lora_mg1024mf
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=2
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_mg1024mf.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_mg1024mf.err

# mg1024 GOAL RUN (2026-06-24): the BioReason/paper target (max_gen=1024) on a 64 GiB
# tile, no-FSDP colocate + varlen-bounded cache. The per-step ACTIVE creep is now FIXED
# (varlen cache bounded; job 8558690 flat+GREEN @ mg256). Remaining open question for
# 1024: does the per-step TRANSIENT (grpo_step bwd graph + vLLM KV at 1024 len) FIT 64
# GiB once empty_cache reclaims it each step? At mg256 the transient was ~8 GiB reclaimed
# cleanly (sawtooth). At 1024 it is larger; this run answers fit vs banned:1.
#   EXPECT (if fits): sawtooth reserved, FLAT end-of-step ALLOC floor, clears 25+ steps.
#   If banned:1 early: 1024 transient too big for 64 GiB at fbs=2 -> drop fbs=1 or
#     TORCHTUNE_USE_CHUNKED_LOSS=1 to bound the bwd graph (NOT a per-step creep issue now).
# 25 steps, NO_FSDP=1, varlen bounded (cap 8). debug-scaling (free). .py read at runtime.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_mg1024mf_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== mg1024 job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"
LOG="$TT/experiments/lora_grpo/logs/pbs_mg1024mf.log"
env CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_colocate_xpu.yaml" \
    MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B" \
    NSTEPS=25 NTILES=12 MAX_GEN=1024 VLLM_GPU_MEM=0.25 \
    TORCHTUNE_USE_IPEX_VARLEN=1 \
    TORCHTUNE_MASKFREE_CAUSAL=1 \
    TORCHTUNE_VARLEN_CACHE_MAX=8 \
    TORCHTUNE_COLOCATE_MEM_PROBE=1 \
    TORCHTUNE_COLOCATE_CACHED_BASE=1 \
    TORCHTUNE_COLOCATE_NO_FSDP=1 \
    TORCHTUNE_COLOCATE_BASE_CPU=0 \
    TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0 \
    TORCHTUNE_COLOCATE_PREFIX_CACHE=0 \
    TORCHTUNE_USE_CHUNKED_LOSS=0 \
    EXTRA_OVERRIDES="forward_batch_size=1 ref_forward_batch_size=2" \
    PBS_NODEFILE="$NF" \
  bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
echo "mg1024 done rc=$? $(date)"
RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
echo "  run log: $RL  steps=$(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
echo "  varlen: $(grep -oE 'varlen=(engaged|disabled[^ ]*)' "$RL" 2>/dev/null | head -1)"
echo "=== maskfree engage/bail + grpo_step path ==="
  grep -oE "maskfree (engaged|bail[^ ]*|skip[^ ]*)|MASKFREE[^ ]* (ENGAGED|engaged|bail[^ ]*)|grpo_step path: [A-Z_]+" "$RL" 2>/dev/null | head -4
  echo "=== end-of-step ALLOC floor (EXPECT FLAT) ==="
grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null | grep final=1 | grep -oE "ALLOC=[0-9.]+ .* step=[0-9]+" | awk '!a[$0]++' | sed -n '1p;5p;10p;15p;20p;25p'
echo "=== reserved sawtooth (grpo transient reclaimed?) ==="
grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null | grep -oE "reserved [0-9.]+->[0-9.]+ \([+-][0-9.]+\)" | awk '!r[$0]++' | sed -n '1p;2p;10p;20p'
echo "=== last reward / health ==="
grep -E "REWARDS step=" "$RL" 2>/dev/null | tail -1
grep -E "TIMING step=" "$RL" 2>/dev/null | sed -n '2p'
grep -E "banned: 1" "$RL" 2>/dev/null | tail -1
grep -E "GREEN|DEGRADED" "$LOG" 2>/dev/null | tail -1
echo "=== mg1024 DONE $(date) ==="

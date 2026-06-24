#!/bin/bash
#PBS -N lora_mg1024lin1n
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_mg1024lin1n.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/logs/pbs_mg1024lin1n.err

# mg1024 + LinearGRPOLoss (chunked-vocab) — 2026-06-24.
# Structural fix for the vocab-logit materialization (the dominant remaining mg1024
# cost after the varlen-cache leak fix + chunked-loss). LinearGRPOLoss returns model
# HIDDEN states from the training forward (skip_output_layer toggled per-call) and
# applies the vocab projection per SEQUENCE-CHUNK inside the loss, so the full
# [B,S,vocab] FP32 logit tensor (~2.7 GiB/seq) is never held. Bit-equivalent to the
# GRPOSimpleLoss formulation (CPU test_linear_grpo_loss_equivalence 4/4).
#
# NOTE: this CHANGES the loss math vs the default GRPOLoss (no IS-ratio clipping;
# GRPOSimple formulation). Requires temperature==1.0 (enforced in recipe).
# Attention is already on the fused FlashAttentionXPU kernel via MASKFREE_CAUSAL=1
# (engages at bs=1; verified job 8558815). NO_FSDP required (projection runs outside
# model.forward; FSDP FULL_SHARD would reshard the weight).
#
# EXPECT vs mg1024ck (job 8558815, GRPOLoss, peak ~42GiB, died step~9):
#   training backward peak drops well below 42 GiB (vocab fraction removed),
#   25 steps clear, flat ALLOC plateau, GREEN. Sanity: compare loss/kl/reward to a
#   GRPOSimpleLoss run (NOT GRPOLoss — different math by design).
# 25 steps, NO_FSDP=1, varlen bounded (cap 8), fbs=1. debug-scaling. .py read at runtime.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
NF="$TT/experiments/lora_grpo/logs/pbs_mg1024lin1n_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
echo "=== mg1024lin job up $(date) | jobid=$PBS_JOBID ==="; cat "$NF"
LOG="$TT/experiments/lora_grpo/logs/pbs_mg1024lin1n.log"
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
    TORCHTUNE_USE_CHUNKED_LOSS=1 \
    EXTRA_OVERRIDES="forward_batch_size=1 ref_forward_batch_size=2 loss._component_=torchtune.dev.rl.linear_grpo_loss.LinearGRPOLoss loss.num_output_chunks=8" \
    PBS_NODEFILE="$NF" \
  bash "$TT/experiments/lora_grpo/run_lora_colocate.sh" > "$LOG" 2>&1
echo "mg1024lin done rc=$? $(date)"
RL=$(ls -dt "$TT"/experiments/lora_grpo/logs/lora_colocate_*/run.log 2>/dev/null | head -1)
echo "  run log: $RL  steps=$(grep -c 'REWARDS step=' "$RL" 2>/dev/null)"
echo "  LinearGRPOLoss wired: $(grep -c 'LinearGRPOLoss wired' "$RL" 2>/dev/null)"
echo "  varlen: $(grep -oE 'varlen=(engaged|disabled[^ ]*)' "$RL" 2>/dev/null | head -1)"
echo "=== training backward peak (MEMCHECK chunk0) — EXPECT << 42 GiB ==="
grep -oE "MEMCHECK grpo_step chunk0 pre-backward: active=[0-9.]+ GiB reserved=[0-9.]+ GiB" "$RL" 2>/dev/null | awk '!m[$0]++' | sed -n '1p;5p;10p;20p'
echo "=== reserved high-water per reclaim (sawtooth peak) ==="
grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null | grep -oE "reserved [0-9.]+->" | grep -oE "[0-9.]+" | sort -rn | head -3
echo "=== end-of-step ALLOC floor (EXPECT FLAT) ==="
grep "COLOCATE_RECLAIM" "$RL" 2>/dev/null | grep final=1 | grep -oE "ALLOC=[0-9.]+ .* step=[0-9]+" | awk '!a[$0]++' | sed -n '1p;5p;10p;15p;20p;25p'
echo "=== reward trajectory + last + health ==="
grep -oE "REWARDS step=[0-9]+ mean=[0-9.]+ .*successes=[0-9.]+" "$RL" 2>/dev/null | sed -n '1p;5p;10p;15p;20p;25p'
grep -E "TIMING step=" "$RL" 2>/dev/null | sed -n '2p'
grep -E "banned: 1" "$RL" 2>/dev/null | tail -1
grep -E "GREEN|DEGRADED" "$LOG" 2>/dev/null | tail -1
echo "=== mg1024lin DONE $(date) ==="

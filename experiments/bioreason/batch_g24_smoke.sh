#!/bin/bash
# One-shot PBS batch: BioReason 4N HSDP G=24 fbs=1 FEASIBILITY + step-time smoke.
#
# QUESTION (user 2026-06-23): is G=24 (the paper group size) reachable on Aurora XPU,
# and is it a better training config than our prod G=8?
#
# HYPOTHESIS (memory project_bioreason_G_not_memory_capped_20260623): the old "G=24
# OOMs -> cap G<=16" was a fbs/G confound. Peak is set by forward_batch_size (per-chunk
# [fbs,seq,vocab] logits), NOT G. grpo_step chunks ref/policy fwd + backward by fbs.
# The old OOM used ref_fbs>=num_seqs (=48 -> 45 GiB ref logits). FIX: ref_fbs=16 (2
# chunked ref cycles ~15 GiB) + fbs=1 (train chunk ~0.9 GiB) => G=24 should FIT.
#
# This smoke proves/disproves feasibility and measures the real step-time COST of G=24
# (more chunks: 24 vs prod 16; +50% more rollouts to gen: 72 vs 48). Whether G=24 is a
# BETTER TRAINER (lower-variance advantage -> faster F_max climb) needs a longer run +
# F_max trajectory compare — this smoke gates that decision on feasibility + cost first.
#
# Config: G=24, batch_size=1 (num_seqs/replica=24), fbs=1, ref_fbs=16, max_gen=1024.
# dp_replicate=3 (4N). Own wsync path (concurrent-safe w/ prod on capacity).
#
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_g24_smoke
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_g24_smoke.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason 4N G=24 fbs=1 smoke start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export NSTEPS=${NSTEPS:-8}
export GRPO_SAMPLES=${GRPO_SAMPLES:-24}      # the paper group size — the thing under test
export BATCH_SIZE=${BATCH_SIZE:-1}           # num_seqs/replica = G*bs = 24
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}   # train chunk: 24 chunks, ~0.9 GiB each
# ref_fbs=16 (NOT >=num_seqs=24): 2 chunked ref cycles ~15 GiB, the fix vs the old OOM.
# Overriding the launcher's default (BATCH_SIZE*GRPO_SAMPLES=24 -> would be 22.6 GiB).
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}   # per-chunk backward (bounds peak)
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}
export TORCHTUNE_MEM_PROBE=${TORCHTUNE_MEM_PROBE:-1}   # capture peak mem to confirm the headroom
export WSYNC_PATH=${WSYNC_PATH:-/lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync_g24/weight_update.raw}

BOOT_WINDOW=${BOOT_WINDOW:-720}
RC=1
for attempt in 1 2; do
    t0=$(date +%s)
    bash "$TT_DIR/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
    RC=$?
    dt=$(( $(date +%s) - t0 ))
    [ $RC -eq 0 ] && break
    if [ $dt -ge $BOOT_WINDOW ]; then
        echo "=== attempt $attempt failed rc=$RC after ${dt}s (>boot window) — real failure (OOM? banned:1?), NOT retrying ==="
        break
    fi
    echo "=== attempt $attempt failed rc=$RC after ${dt}s (<boot window) — likely vLLM boot flake, retrying ==="
    sleep 20
done
echo "=== BioReason 4N G=24 smoke end rc=$RC $(date) ==="
exit $RC

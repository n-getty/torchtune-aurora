#!/bin/bash
# One-shot PBS batch: BioReason 4N HSDP STOP-TOKEN A/B (same-node, back-to-back).
#
# WHY: the band A/B (job 8556851) showed generation is max_gen-BOUND, not dispatch-
# bound — stop_rate=0.000, trunc_rate~0.5: vLLM never received stop tokens, so every
# rollout decoded to the full max_tokens cap. The fix sends stop_token_ids to vLLM.
# This A/B measures the REAL lever:
#   leg 1: TORCHTUNE_VLLM_STOP_TOKENS=0 (old: no stop tokens -> full max_tokens always)
#   leg 2: TORCHTUNE_VLLM_STOP_TOKENS=1 (fixed: vLLM stops at EOS)
# Same vLLM pool, same nodes => variance-immune gen-time delta.
#
# Expect: stopON should cut gen wall-clock on the ~half of rollouts that emit EOS
# before 1024 (resp_len mean ~590 in the band run => big headroom under the cap).
# WATCH: stop_rate should go 0.000 -> >0; reward/F-signal must NOT degrade (EOS-
# stopped completions are the model's natural length, not truncated mid-reasoning).
#
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_stop_ab
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_stop_ab.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason stop-token A/B batch start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export NSTEPS=${NSTEPS:-8}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export AB_STOP="0 1"

# vLLM EngineCore boot flakes (one tile dies during startup -> launcher aborts) are
# intermittent on Aurora (cold SPIR-V cache, tile contention, transient L0). They fail
# FAST (< ~12 min, before any training). Retry once on a fast failure; a slow failure
# (> BOOT_WINDOW) is a real run error -> don't retry.
BOOT_WINDOW=${BOOT_WINDOW:-720}   # 12 min
RC=1
for attempt in 1 2; do
    t0=$(date +%s)
    bash "$TT_DIR/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
    RC=$?
    dt=$(( $(date +%s) - t0 ))
    [ $RC -eq 0 ] && break
    if [ $dt -ge $BOOT_WINDOW ]; then
        echo "=== attempt $attempt failed rc=$RC after ${dt}s (>boot window) — real failure, NOT retrying ==="
        break
    fi
    echo "=== attempt $attempt failed rc=$RC after ${dt}s (<boot window) — likely vLLM boot flake, retrying ==="
    sleep 20
done
echo "=== BioReason stop-token A/B batch end rc=$RC $(date) ==="
exit $RC

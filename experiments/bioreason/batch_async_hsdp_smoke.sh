#!/bin/bash
# One-shot PBS batch: BioReason 4N HSDP ASYNC overlap smoke (the real prod topology).
#
# Validates the async-HSDP + staleness=1 implementation on 4N (dp_replicate=3):
#   - async ENGAGES under HSDP (was force-disabled before today's fix): expect
#     "rollout producer started" on each shard leader (global ranks 0,12,24) +
#     "async consume (... lag=1)" lines (staleness PINNED to 1, not 2).
#   - OVERLAP: step time should drop to ~max(gen,grpo) ~125s vs sync ~170s (~26%).
#   - ratios ~1.0 (staleness=1 is IS-correctable by GRPOLoss; NOT on-policy but bounded).
#   - no banned:1 with 3 concurrent producers (1 per shard-leader node); no mailbox
#     deadlock at maxsize=1; no XPU thread-safety wedge.
#
# Uses the async YAML (async_generation.enabled=true, loss=GRPOLoss,
# always_compute_rollout_logprobs=true) via CONFIG=. Own wsync path (concurrent-safe).
# Compare its TIMING gen/grpo/total against the sync prod run (8556915, ~170s/step).
#
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_async_hsdp
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_async_hsdp_smoke.out

set -o pipefail
TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT_DIR"
echo "=== BioReason 4N async-HSDP smoke start $(date) job=${PBS_JOBID} ==="
echo "nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ENABLE_LORA=1
export CONFIG=recipes/configs/dev/production/bioreason_4b_lora_grpo_2node_server_xpu_async.yaml
export NSTEPS=${NSTEPS:-12}
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}
# Own wsync path so this can run concurrently with the sync prod run on capacity.
export WSYNC_PATH=${WSYNC_PATH:-/lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync_asyncsmoke/weight_update.raw}

# Retry once on a fast (vLLM boot-flake) failure.
BOOT_WINDOW=${BOOT_WINDOW:-720}
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
echo "=== BioReason 4N async-HSDP smoke end rc=$RC $(date) ==="
exit $RC

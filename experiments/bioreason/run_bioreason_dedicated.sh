#!/bin/bash
# BioReason 4B GRPO — 11+1 dedicated vLLM mode (12-tile single node)
# Run from PROJDIR after SSH-ing into a held node.
FW_BIN=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin
export PATH=${FW_BIN}:$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
echo "Python: $(which python3)"
python3 -c "import torch; print('torch OK:', torch.__version__)" || { echo "FAIL"; exit 1; }

export PYTHONNOUSERSITE=1
export INFRA_PROVIDER=local
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
# Run 45: XCCL transport vars (CCL_ATL_TRANSPORT=ofi etc.) had ZERO effect on
# wsync bandwidth (still 0.34 GB/s). XCCL on [0,11] intra-node is genuinely slow.
# Run 46: switch wsync_pg backend to gloo + CPU-staging buffer (loopback SHM
# should give ≥5 GB/s vs XCCL's 0.34 GB/s). GLOO_SOCKET_IFNAME=lo for loopback.
export TORCHTUNE_WSYNC_BACKEND=gloo
export GLOO_SOCKET_IFNAME=lo
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Enable single fwd+bwd path in recipe (line 5484). Multimodal gate dropped, so
# BioReason now uses the same path 32B uses with this env. Avoids per-chunk
# fwd-graph retention that pushed step-1 BWD peak past auto-GC threshold (run 38).
export TORCHTUNE_USE_CHUNKED_LOSS=1
# Run 40: defragment allocator pool + defer GC. Run 39 step 2 BWD spiked
# to 36s (vs 14s steps 0/1) and wsync to 50s — pool fragmentation forced
# CCL to remap pages. max_split_size_mb caps allocator block fragmentation;
# gc:0.95 keeps default-99 behavior intact while leaving 5% headroom for
# OOM-retry recovery before banned:1.
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.95
# Pinned CPU buffer for wsync: 32B saw 8.5x speedup (31s → 3.7s gather).
# Wsync stays on the AllGather→pinned-CPU→broadcast path which avoids
# allocating fresh L0 pages (the IPC handle source) during wsync.
export TORCHTUNE_PINNED_CPU_BUF=1

# FSDP1 weight sync: summon_full_params AllGathers 7.49 GiB on ALL 11 training ranks.
# Pre-warm (in recipe train()) puts 7.49 GiB in cache before training loop; subsequent
# FSDP AllGathers reuse cached block → no new L0 alloc → pool stays ≤54 GiB → GC never fires.
# fbs=8 (1 chunk, no gradient accumulation): POST-BWD pool ~54 GiB; summon_full_params
# reuses cached block → pool stays ~54 GiB (67% of 63.98 GiB) → GC never fires.
# fbs=4 (runs 34-36) REJECTED: 2× no_sync chunks peak at 59.7 GiB POST-BWD (worse, not better).
# usm_caching_alloc_v2.so (runs 34-35) REJECTED: OOM retry → zeMemFree → same banned:1.

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
export PYTHONPATH=/lus/flare/projects/ModCon/ngetty/bioreason_deps:${PROJDIR}:${PYTHONPATH}

source /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/setvars.sh 2>/dev/null || true

MODEL_DST=/tmp/torchtune/bioreason-pro-sft
MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft
if [ ! -f "${MODEL_DST}/config.json" ]; then
    echo "=== Staging model to /tmp ==="
    mkdir -p /tmp/torchtune
    cp -r "$MODEL_SRC" "$MODEL_DST"
fi

cd ${PROJDIR}

# Pre-launch cleanup: kill stale vLLM workers from prior attempts on this node.
# Without this, orphaned VLLM:: subprocesses from a crashed run hold L0 device
# contexts and contaminate the next run (see 32B launcher line 150-157).
echo "=== Cleaning stale vLLM processes ==="
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
sleep 2
rm -f /dev/shm/vllm* 2>/dev/null || true

NUM_STEPS=${1:-3}
echo "=== Starting 12-rank BioReason dedicated-vLLM GRPO (num_steps=${NUM_STEPS}) ==="
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=12 \
    recipes/dev/grpo_bioreason_distributed_xpu.py \
    --config recipes/configs/dev/production/bioreason_4b_grpo_dedicated_xpu.yaml \
    base_model_path=${MODEL_DST} \
    output_dir=${PROJDIR}/outputs/bioreason_dedicated_v1 \
    num_steps=${NUM_STEPS} \
    log_peak_memory_stats=true 2>&1
STATUS=$?
echo "=== Done: exit=${STATUS} ==="
exit ${STATUS}

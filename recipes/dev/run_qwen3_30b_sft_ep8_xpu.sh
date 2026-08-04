#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Qwen3-30B-A3B SFT with Expert Parallelism EP=8 — single node, Aurora XPU.
#
# Standalone MoE SFT launcher (no vLLM, no ref-model, no weight-sync) — the
# isolated-from-GRPO throughput measurement. Topology: 8 of 12 tiles,
# dp_replicate=1, dp_shard=8=expert_parallel_degree (4 tiles idle). Modeled on
# run_qwen3_30b_ep8_vllm_2node.sh's CCL env block, with everything vLLM/wsync/
# cross-node-specific removed (there is no separate vLLM node in this setup;
# no GLOO_SOCKET_IFNAME routing, no TORCHTUNE_XCCL_HOST, no WSYNC_* vars).
#
# Usage:
#   bash recipes/dev/run_qwen3_30b_sft_ep8_xpu.sh [num_steps]
#
# For the EP=1 (no-EP) baseline, point CONFIG at
# recipes/configs/dev/production/qwen3_30b_a3b_sft_ep1_xpu.yaml and set
# NPROC to the desired tile count (e.g. 12 for full single-node FSDP2) instead
# of invoking this script — that config's own default topology (dp_shard=-1)
# infers from NPROC, so no EXTRA_OVERRIDES are needed for that leg.

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd ${PROJDIR}

module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# CCL — interactive single-node (see CLAUDE.md launcher decision table):
# CCL_PROCESS_LAUNCHER=none + CCL_ATL_TRANSPORT=ofi, no pmix/mpi KVS.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_WORKER_COUNT=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-65536}
export CCL_CHUNK_SIZE=${CCL_CHUNK_SIZE:-16777216}
export CCL_ALLREDUCE=ring
export CCL_ALLTOALL=naive
unset XPU_USM_ALLOC_SO

if [ "${TORCHTUNE_USE_AURORA_MOE:-0}" = "1" ]; then
    export CCL_OP_SYNC=0
    AURORA_MOE_TRANSPORT=${AURORA_MOE_TRANSPORT:-native_ccl}
    case "${AURORA_MOE_TRANSPORT}" in
        native_ccl)
            export AURORA_MOE_NATIVE_CCL=1
            export AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM=1
            export AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP=1
            export AURORA_MOE_NATIVE_CCL_ALLTOALLV=1
            ;;
        l0_ipc)
            unset AURORA_MOE_NATIVE_CCL
            unset AURORA_MOE_NATIVE_CCL_PRIVATE_STREAM
            unset AURORA_MOE_NATIVE_CCL_STREAM_FENCE_REAP
            unset AURORA_MOE_NATIVE_CCL_ALLTOALLV
            export AURORA_MOE_L0_IPC_ALLTOALLV=1
            export AURORA_MOE_L0_IPC_PREBUILT=1
            export AURORA_MOE_L0_IPC_BIAS=${AURORA_MOE_L0_IPC_BIAS:-default}
            export AURORA_MOE_L0_IPC_IMPORT_BACKEND=${AURORA_MOE_L0_IPC_IMPORT_BACKEND:-scm}
            export AURORA_MOE_L0_IPC_BARRIER_POINTER_MODE=${AURORA_MOE_L0_IPC_BARRIER_POINTER_MODE:-device_table}
            export AURORA_MOE_L0_IPC_ALLTOALLV_SOCKET=${AURORA_MOE_L0_IPC_ALLTOALLV_SOCKET:-/tmp/aurora_moe_l0_a2av_${PBS_JOBID:-$$}}
            ;;
        *)
            echo "AURORA_MOE_TRANSPORT must be native_ccl or l0_ipc" >&2
            exit 2
            ;;
    esac
    export AURORA_MOE_ALLTOALLV=1
    export AURORA_MOE_SEGMENTED_SONIC=1
    export AURORA_MOE_SEGMENTED_EXPERT_MAJOR=1
    export AURORA_MOE_SEGMENTED_FUSED_SCORE_PAYLOAD=1
    export AURORA_MOE_EXPERT_MAJOR_GEMM=${AURORA_MOE_EXPERT_MAJOR_GEMM:-onemkl}
    export AURORA_MOE_EXPERT_MAJOR_DW=${AURORA_MOE_EXPERT_MAJOR_DW:-onemkl}
    export AURORA_MOE_EXPERT_MAJOR_REORDER=${AURORA_MOE_EXPERT_MAJOR_REORDER:-row_parallel}
    export AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD=${AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD:-reordered}
    export AURORA_MOE_SEGMENTED_POINTWISE=${AURORA_MOE_SEGMENTED_POINTWISE:-sycl}
    export AURORA_MOE_ONEMKL_FUSE_UP_GATE_DX=${AURORA_MOE_ONEMKL_FUSE_UP_GATE_DX:-1}
    export AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE=${AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE:-0}
    export AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT=${AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT:-0}
    export AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT_MARKER=${AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT_MARKER:-0}
fi

# v153 fix (see torchtune/modules/moe/_parallelism.py _ep_all_gather/_ep_reduce_scatter
# comments): after ~256+ EP gloo ops, the CXI NIC can carry residual OFI CQ entries
# that contaminate gloo TCP (gloo also rides CXI on Aurora by default). Forcing gloo
# onto loopback isolates the EP CPU-bounce path from the OFI/CXI fabric entirely.
# Single-node only (loopback has no meaning across nodes) -- multi-node EP launchers
# use GLOO_SOCKET_IFNAME=hsn0 instead. Missing here caused a step-2 banned:1 GPU
# segfault at EP-OP #329 (a ~11s abnormally-slow RS-FWD immediately preceding the
# fault) that reproduced identically across 3 independent configs (batch_size=2,
# batch_size=1, and with TORCHTUNE_EP_GRAD_RELEASE_XCCL=0) on 2 different clean
# nodes -- ruling out node contamination, batch size, and XCCL-grad-release as the
# cause, and pointing at exactly this documented single-node EP gloo/OFI issue.
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99

FW_SITE=/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages
LOCAL_SITE=/home/ngetty/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1
export PYTHONPATH=${PROJDIR}:${FW_SITE}:${LOCAL_SITE}
if [ "${TORCHTUNE_USE_AURORA_MOE:-0}" = "1" ]; then
    AURORA_MOE_PYTHONPATH=${AURORA_MOE_PYTHONPATH:-/lus/flare/projects/ModCon/ngetty/aurora_moe_dropin_overlay}
    export PYTHONPATH=${AURORA_MOE_PYTHONPATH}:${PYTHONPATH}
fi
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/lus/flare/projects/ModCon/ngetty/hf_datasets_cache}
export HF_HOME=${HF_HOME:-/lus/flare/projects/ModCon/ngetty/hf_cache}
export no_proxy="*"
export NO_PROXY="*"

NPROC=8           # 8 of 12 tiles for EP=8 dp_shard=8 single-replica
NSTEPS=${1:-15}
MODEL_PATH=/lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B
CONFIG=${CONFIG:-${PROJDIR}/recipes/configs/dev/production/qwen3_30b_a3b_sft_ep8_xpu.yaml}

# Measurement provenance is derived here so rank artifacts are sealable and
# matched A/B overrides remain visible. Health, gate, and semantic completion
# are intentionally caller-owned because the launcher cannot certify them.
export TORCHTUNE_MOE_SOURCE_REVISION=${TORCHTUNE_MOE_SOURCE_REVISION:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
if [ -z "${TORCHTUNE_MOE_UNCOMMITTED+x}" ]; then
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        export TORCHTUNE_MOE_UNCOMMITTED=dirty
    else
        export TORCHTUNE_MOE_UNCOMMITTED=clean
    fi
fi
export TORCHTUNE_MOE_ROUTING_INDEX_MODE=${TORCHTUNE_MOE_ROUTING_INDEX_MODE:-compact}
export TORCHTUNE_MOE_ROUTER_SEMANTICS=${TORCHTUNE_MOE_ROUTER_SEMANTICS:-probability_topk_v2}
export TORCHTUNE_MOE_PIPELINE_STAGE=${TORCHTUNE_MOE_PIPELINE_STAGE:-0}
export TORCHTUNE_MOE_WARMUP_STEPS=${TORCHTUNE_MOE_WARMUP_STEPS:-4}
export TORCHTUNE_MOE_MEASUREMENT_STEPS=${TORCHTUNE_MOE_MEASUREMENT_STEPS:-8}
export TORCHTUNE_MOE_STEADY_STATE_STEPS=${TORCHTUNE_MOE_STEADY_STATE_STEPS:-4}

JOB_TAG="${PBS_JOBID:-$$}"
LAST4=$(echo "${JOB_TAG}" | tr -dc '0-9' | tail -c 4)
MASTER_PORT=$(( 29500 + ( 10#${LAST4:-0} % 400 ) ))

echo "=== EP=8 / dp_replicate=1 Qwen3-30B-A3B SFT (single node, no vLLM) ==="
echo "Training node: $(hostname) (8 of 12 tiles, EP=8 dp_shard=8, 4 tiles idle)"
echo "Config: ${CONFIG}"
echo "Steps: ${NSTEPS}"
echo "Master port: ${MASTER_PORT} (job tag ${JOB_TAG})"
echo "Date: $(date)"

mkdir -p /tmp/torchtune

LOCAL_MODEL=/tmp/torchtune/$(basename ${MODEL_PATH})
if [ ! -f "${LOCAL_MODEL}/config.json" ]; then
    echo "Staging model to ${LOCAL_MODEL}..."
    t0=$SECONDS
    cp -r "${MODEL_PATH}" "${LOCAL_MODEL}"
    echo "Staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${LOCAL_MODEL}"
fi

# GRPO-proven env-flag defaults (see CLAUDE.md flag table + Phase 0/2 of the
# MoE-SFT-isolation plan). Caller can override any of these by exporting
# before invoking this script.
export TORCHTUNE_EP_GRAD_RELEASE_XCCL=${TORCHTUNE_EP_GRAD_RELEASE_XCCL:-1}
export TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=${TORCHTUNE_MOE_SEQUENTIAL_EXPERTS:-1}
export TORCHTUNE_MOE_GROUPED_EXPERTS=${TORCHTUNE_MOE_GROUPED_EXPERTS:-0}
export TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT=${TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT:-0}
export TORCHTUNE_EP_ALL2ALL=${TORCHTUNE_EP_ALL2ALL:-0}
export TORCHTUNE_MOE_OPTIMIZATION_PROFILE=${TORCHTUNE_MOE_OPTIMIZATION_PROFILE:-qwen3_ep8_seq4096}
echo "TORCHTUNE_EP_GRAD_RELEASE_XCCL=${TORCHTUNE_EP_GRAD_RELEASE_XCCL}"
echo "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=${TORCHTUNE_MOE_SEQUENTIAL_EXPERTS}"
echo "TORCHTUNE_MOE_GROUPED_EXPERTS=${TORCHTUNE_MOE_GROUPED_EXPERTS}"
echo "TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT=${TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT}"
echo "TORCHTUNE_EP_ALL2ALL=${TORCHTUNE_EP_ALL2ALL}"
echo "TORCHTUNE_USE_AURORA_MOE=${TORCHTUNE_USE_AURORA_MOE:-0}"
echo "AURORA_MOE_TRANSPORT=${AURORA_MOE_TRANSPORT:-disabled}"
echo "AURORA_MOE_EXPERT_MAJOR_GEMM=${AURORA_MOE_EXPERT_MAJOR_GEMM:-disabled}"
echo "AURORA_MOE_EXPERT_MAJOR_DW=${AURORA_MOE_EXPERT_MAJOR_DW:-disabled}"
echo "AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT=${AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT:-disabled}"
echo "AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT_MARKER=${AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT_MARKER:-disabled}"
echo "AURORA_MOE_PYTHONPATH=${AURORA_MOE_PYTHONPATH:-disabled}"
echo "TORCHTUNE_AURORA_MOE_MEM_DEBUG=${TORCHTUNE_AURORA_MOE_MEM_DEBUG:-0}"
echo "CCL_OP_SYNC=${CCL_OP_SYNC}"
echo "EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}"

echo "Starting EP=8 Qwen3-30B-A3B SFT on 8 tiles..."
RANK_WRAPPER=/tmp/torchtune/moe_rank_wrapper_${JOB_TAG}_$$.sh
cat > "${RANK_WRAPPER}" <<'WRAPPEREOF'
#!/bin/bash
export ZE_AFFINITY_MASK=${LOCAL_RANK}
exec python3 "$@"
WRAPPEREOF
chmod +x "${RANK_WRAPPER}"
torchrun \
    --nproc_per_node=${NPROC} \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    --no-python "${RANK_WRAPPER}" \
    ${PROJDIR}/recipes/dev/full_finetune_moe_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${LOCAL_MODEL} \
    max_steps_per_epoch=${NSTEPS} \
    ${EXTRA_OVERRIDES:-} \
    2>&1

echo "=== Training complete ==="

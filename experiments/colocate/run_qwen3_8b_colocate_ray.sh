#!/bin/bash
# Qwen3-8B GRPO — Ray-colocate (8 trainer ranks + vLLM TP=8 Ray actors on same tiles)
#
# Strategy:
#   - Ray head on this node, 12 GPUs visible (full node pool)
#   - torchrun spawns 8 FSDP2 trainer ranks on tiles 0..7
#   - Trainer rank 0 starts vLLM with distributed_executor_backend="ray" → spawns
#     8 Ray actors that share the same 8 tiles via gpu_memory_utilization=0.55
#   - Weight sync: rank 0 streams params via vllm_llm.collective_rpc("load_weights")
#
# Validated coexistence (A3b probe, 2026-05-06):
#   PROBE_GIB=10 + PROBE_GPU_MEM=0.55 → vLLM TP=8 boots and generates cleanly
#
# Usage (on held node after SSH):
#   bash /lus/flare/.../experiments/colocate/run_qwen3_8b_colocate_ray.sh [nsteps] [G]

set -e

REPO_ROOT="/lus/flare/projects/ModCon/ngetty/torchtune"
EXPDIR="${REPO_ROOT}/experiments/colocate"
RAY_SMOKE_DIR="${REPO_ROOT}/experiments/ray_smoke"
CONFIG="recipes/configs/dev/experimental/qwen3_8b_grpo_colocate_ray_xpu.yaml"
MODEL_SRC="/lus/flare/projects/ModCon/ngetty/models/Qwen3-8B"
MODEL_STAGED="/tmp/torchtune/Qwen3-8B"

NSTEPS=${1:-3}
G=${2:-4}
NTILES=8
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${EXPDIR}/ray_colo_logs/${TS}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"

echo "============================================================"
echo "  Qwen3-8B Ray-colocate GRPO smoke"
echo "  host=$(hostname)  date=$(date)"
echo "  NSTEPS=${NSTEPS}  G=${G}  TP=${NTILES}"
echo "  log: ${LOG}"
echo "============================================================"

cd "${REPO_ROOT}"

# ── Ray + frameworks env (sets ONEAPI_DEVICE_SELECTOR=level_zero:0..11, etc.) ──
source "${RAY_SMOKE_DIR}/setup_ray_env.sh" "frameworks"

# HSN IP for Ray + vLLM
HSN_IP=$(getent hosts "$(hostname).hsn.cm.aurora.alcf.anl.gov" | awk '{ print $1 }' | sort | head -n 1)
[[ -z "$HSN_IP" ]] && HSN_IP=$(hostname -i | awk '{print $1}')
export VLLM_HOST_IP="$HSN_IP"
export no_proxy="localhost,127.0.0.1,$HSN_IP,$(hostname),$(hostname).hsn.cm.aurora.alcf.anl.gov"
echo "HSN_IP=$HSN_IP"

# Trainer-side env (in addition to setup_ray_env.sh defaults)
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_KVS_IFACE=hsn0
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536

# Fast paths (matching tp8_xpu.sh)
export TORCHTUNE_USE_IPEX_VARLEN=1
export TORCHTUNE_MASKFREE_CAUSAL=1
export TORCHTUNE_PINNED_CPU_BUF=1
# Per-chunk fwd+bwd to release activations immediately (avoids L0 resource
# exhaustion when 4 chunk forwards stack up before any backward).
export TORCHTUNE_USE_CHUNKED_LOSS=1

# Offline HF (override the ray_smoke setup which points to /flare/datasets)
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="/lus/flare/projects/ModCon/ngetty/hf_datasets_cache"

# PYTHONPATH — recipe + vLLM usercustomize
VLLM_CUSTOMIZATION="${REPO_ROOT}/recipes/dev/_usercustomize_vllm"
export PYTHONPATH="${REPO_ROOT}:${VLLM_CUSTOMIZATION}:${PYTHONPATH:-}"

# ── Stage model ───────────────────────────────────────────────────────────────
mkdir -p /tmp/torchtune
if [ ! -f "${MODEL_STAGED}/config.json" ]; then
    echo "Staging Qwen3-8B to /tmp ($(date))..."
    t0=$SECONDS
    cp -r "${MODEL_SRC}" "${MODEL_STAGED}"
    echo "  staged in $((SECONDS - t0))s"
else
    echo "Model already staged at ${MODEL_STAGED}"
fi

# ── Clean any stale Ray and vLLM state ───────────────────────────────────────
echo ""
echo "--- Cleaning up old Ray / vLLM processes..."
ray stop --force 2>/dev/null || true
pkill -u "$USER" -f "ray::" 2>/dev/null || true
pkill -u "$USER" -f "vllm" 2>/dev/null || true
pkill -u "$USER" -f "EngineCore" 2>/dev/null || true
sleep 3

# ── Start Ray head with full 12-GPU pool ────────────────────────────────────
RAY_PORT=6379
echo "--- Starting Ray head on $HSN_IP:$RAY_PORT (12 GPUs)..."
ray start --head --node-ip-address="$HSN_IP" --port="$RAY_PORT" \
    --num-gpus=12 --num-cpus=4 --temp-dir=/tmp --include-dashboard=false \
    > "${LOG_DIR}/ray_head.log" 2>&1
export RAY_ADDRESS="$HSN_IP:$RAY_PORT"

for i in $(seq 1 30); do
    if ray status --address="$RAY_ADDRESS" &>/dev/null; then
        echo "  ray ready in ${i}s — sleeping 15s for GCS"
        sleep 15
        break
    fi
    sleep 1
done
ray status --address="$RAY_ADDRESS"

# ── Run torchrun trainer (8 ranks); rank 0 starts vLLM via Ray ───────────────
echo ""
echo "=== Starting GRPO ($(date)) ==="
set +e
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${NTILES} \
    recipes/dev/grpo_full_finetune_distributed_xpu.py \
    --config "${CONFIG}" \
    base_model_path="${MODEL_STAGED}" \
    num_steps="${NSTEPS}" \
    grpo_samples="${G}" \
    2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e
echo "=== Done ($(date)), exit=${RC} ==="

# ── Stop Ray ─────────────────────────────────────────────────────────────────
echo ""
echo "--- Stopping Ray..."
ray stop --force 2>/dev/null || true

# ── Acceptance check ─────────────────────────────────────────────────────────
echo ""
echo "── wsync timing ──"
grep -E "wsync|weight_sync|WSYNC|_sync_ray_colocate" "${LOG}" | head -10 || echo "  (none)"

echo ""
echo "── banned:1 / PDE faults (expect none) ──"
grep -E "banned:1|Segmentation|PDE|UR_RESULT_ERROR" "${LOG}" | head -5 || echo "  (none — good)"

echo ""
echo "── METRICS ──"
grep -E "METRICS|reward_mean|ratios" "${LOG}" | tail -10 || echo "  (none)"

echo ""
echo "── step timing ──"
grep -E "step_time|grpo_step|TIMING|gen=|Generated" "${LOG}" | tail -10 || echo "  (none)"

echo ""
echo "── memory ──"
grep -E "PRE-STEP|resv=|peak_memory" "${LOG}" | tail -6 || echo "  (none)"

echo ""
echo "Log dir: ${LOG_DIR}"
exit ${RC}

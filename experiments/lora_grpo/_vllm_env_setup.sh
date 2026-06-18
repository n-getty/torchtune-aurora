#!/bin/bash
# vLLM environment setup for the LoRA-GRPO 2-node launcher.
#
# Sourced by run_qwen3_4b_lora_2node.sh inside the SSH heredocs that launch
# vLLM on VLLM_NODE. Branches on LORA_USE_RUNTIME:
#
#   LORA_USE_RUNTIME=1  → torch 2.11+xpu venv, --enable-lora, runtime hot-swap
#                         (validated 2026-05-05 to bypass the IPEX BGMV PDE; see
#                         memory project_lora_grpo_torch211_unblocks_bgmv.md)
#   LORA_USE_RUNTIME=0  → frameworks/2025.3.1, NO --enable-lora, merged-weight
#                         publish via /collective_rpc load_weights_from_raw.
#
# Exports for use by callers:
#   VLLM_LORA_FLAGS   — string of CLI flags for the vllm api_server command
#   VLLM_WORKER_EXT   — '--worker-extension-cls ...' or empty
#   VLLM_PYTHON       — full python3 path to use for vLLM
#   (PATH / LD_LIBRARY_PATH / PYTHONPATH / venv activation done in-place)
#
# Required inputs in the calling environment:
#   TT_DIR_REMOTE     — torchtune root on the remote node (absolute path)
#   VLLM_PYTHONPATH   — PYTHONPATH for the merged-weight (frameworks) path
#   LORA_USE_RUNTIME  — 1 or 0

: "${LORA_USE_RUNTIME:?LORA_USE_RUNTIME must be set (1=hot-swap venv, 0=merged-weight frameworks)}"
: "${TT_DIR_REMOTE:?TT_DIR_REMOTE must be set}"

# Common environment (both paths)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy='*' NO_PROXY='*'
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
# 12 vLLM tiles × multi-threaded Rayon pools exhausts thread limits without these.
export RAYON_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1
export RUST_BACKTRACE=1
mkdir -p /tmp/torchtune

if [ "${LORA_USE_RUNTIME}" = "1" ]; then
    # === Hot-swap path: torch 2.11+xpu venv, --enable-lora ===
    VENV=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/torch211_venv
    if [ ! -x "${VENV}/bin/python3" ]; then
        echo "[vllm-env] FATAL: torch211_venv not found at ${VENV}" >&2
        return 1 2>/dev/null || exit 1
    fi
    module purge 2>/dev/null || true
    module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
    unset PYTHONPATH 2>/dev/null
    unset PYTHONNOUSERSITE 2>/dev/null
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    # pip oneCCL ships a stale libfabric without cxi (Slingshot). The venv's torch
    # RUNPATH resolves libfabric.so to the venv copy first; the .so files are
    # renamed .disabled in venv prep, but prepend the system Cray libfabric anyway.
    export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}
    export VLLM_PYTHON="${VENV}/bin/python3"
    # Runtime adapter hot-swap requires --enable-lora at server startup AND the
    # env var that allows /v1/load_lora_adapter and /v1/unload_lora_adapter HTTP.
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
    # VLLM_SERVER_DEV_MODE not needed (no /collective_rpc on this path).
    unset VLLM_SERVER_DEV_MODE 2>/dev/null
    VLLM_LORA_FLAGS="--enable-lora --max-lora-rank 16 --max-loras 2"
    VLLM_WORKER_EXT=""
    export VLLM_LORA_FLAGS VLLM_WORKER_EXT
    echo "[vllm-env] LORA_USE_RUNTIME=1 (hot-swap)  python=${VLLM_PYTHON}"
    "${VLLM_PYTHON}" -c "import torch, vllm; print(f'[vllm-env] torch={torch.__version__}  vllm={vllm.__version__}')" 2>&1 | tail -1
else
    # === Merged-weight path: frameworks/2025.3.1, NO --enable-lora ===
    module purge 2>/dev/null || true
    module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
    export PATH=$(echo "${PATH}" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
    unset VIRTUAL_ENV 2>/dev/null
    # PYTHONNOUSERSITE must NOT be set for vLLM workers (disables usercustomize.py
    # that patches vllm.model_executor.models.registry for XPU). Train side sets it.
    unset PYTHONNOUSERSITE 2>/dev/null
    : "${VLLM_PYTHONPATH:?VLLM_PYTHONPATH must be set for merged-weight path}"
    export PYTHONPATH="${VLLM_PYTHONPATH}"
    # Required to expose /collective_rpc for load_weights_from_raw.
    export VLLM_SERVER_DEV_MODE=1
    unset VLLM_ALLOW_RUNTIME_LORA_UPDATING 2>/dev/null
    export VLLM_PYTHON=python3
    VLLM_LORA_FLAGS=""
    VLLM_WORKER_EXT="--worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension"
    export VLLM_LORA_FLAGS VLLM_WORKER_EXT
    echo "[vllm-env] LORA_USE_RUNTIME=0 (merged-weight)  python=$(which python3)"
fi

# --- Contract guard (single source of truth for --worker-extension-cls) -------
# The merged/delta publish paths drive vLLM via /collective_rpc load_weights_from_raw,
# which only exists when the WeightSyncFromFileExtension worker-extension-cls is
# registered at server start. A launcher forked from the hot-swap template that drops
# the flag => HTTP 500 init_xccl_communicator on the step-1 weight sync
# (memory: feedback_dense_4b_launcher_missing_worker_extension). Every vLLM launch site
# expands ${VLLM_WORKER_EXT}, so asserting it non-empty here — at the one place that
# sets it — protects all launch sites and all forks that source this helper.
if [ "${LORA_USE_RUNTIME:-0}" != "1" ]; then
    case "${VLLM_WORKER_EXT}" in
        *worker-extension-cls*) : ;;  # ok
        *)
            echo "[vllm-env] FATAL: merged/delta path but VLLM_WORKER_EXT lacks --worker-extension-cls." >&2
            echo "[vllm-env]   /collective_rpc load_weights_from_raw would be missing -> HTTP 500 on step-1 wsync." >&2
            echo "[vllm-env]   See memory: feedback_dense_4b_launcher_missing_worker_extension." >&2
            exit 1
            ;;
    esac
fi

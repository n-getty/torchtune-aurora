#!/usr/bin/env bash
set -euo pipefail

# Build the source vLLM/XPU stack used by the Kimi serving experiments.

VENV=${VENV:-/flare/ModCon/ngetty/venvs/vllm-serve-xpu}
VLLM_SRC=${VLLM_SRC:-/flare/ModCon/ngetty/vllm-xpu-src}
KERNELS_SRC=${KERNELS_SRC:-/flare/ModCon/ngetty/vllm-xpu-kernels}
KERNELS_REF=${KERNELS_REF:-0.1.12.1}
PYTHON=${PYTHON:-$(command -v python3)}
KERNELS_WHEEL_URL=${KERNELS_WHEEL_URL:-https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.7/vllm_xpu_kernels-0.1.7-cp38-abi3-manylinux_2_28_x86_64.whl}

command -v uv >/dev/null || { echo "ERROR: uv is required" >&2; exit 1; }
[[ -d "$VLLM_SRC" ]] || { echo "ERROR: VLLM_SRC does not exist: $VLLM_SRC" >&2; exit 1; }

if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON" -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$VENV/triton-cache}
export VLLM_CHUNK_PREFILL_CONFIG=${VLLM_CHUNK_PREFILL_CONFIG:-chunk_prefill_full.conf}
export VLLM_PAGED_DECODE_CONFIG=${VLLM_PAGED_DECODE_CONFIG:-paged_decode_full.conf}
export VLLM_XPU_AOT_DEVICES=${VLLM_XPU_AOT_DEVICES:-pvc}
export VLLM_XPU_XE2_AOT_DEVICES=${VLLM_XPU_XE2_AOT_DEVICES:-pvc}
export VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE:-xpu}
mkdir -p "$TRITON_CACHE_DIR"

uv pip install --python "$VENV/bin/python" --upgrade pip setuptools wheel
if [[ -d "$KERNELS_SRC" ]]; then
    pushd "$KERNELS_SRC" >/dev/null
    git fetch --tags --quiet 2>/dev/null || true
    git rev-parse --verify --quiet "$KERNELS_REF^{commit}" >/dev/null || {
        echo "ERROR: kernel ref does not exist: $KERNELS_REF" >&2
        exit 1
    }
    git checkout --quiet "$KERNELS_REF"
    uv pip install --python "$VENV/bin/python" .
    popd >/dev/null
else
    echo "Kernel source absent; installing pinned wheel"
    uv pip install --python "$VENV/bin/python" "$KERNELS_WHEEL_URL"
fi
uv pip install --python "$VENV/bin/python" --no-build-isolation --no-deps -e "$VLLM_SRC"

echo "Stack installed in $VENV"

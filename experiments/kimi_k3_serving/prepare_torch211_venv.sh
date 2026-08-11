#!/usr/bin/env bash
# Prepare torchtune-pt-nightly-xpu (torch 2.11) to serve K3, for graph capture.
#
# WHY torch 2.11: supports_xpu_graph() is a bare
# is_torch_equal_or_newer("2.11.0.dev") check (utils/torch_utils.py:891). The
# K3 venv inherits torch 2.10 from frameworks/2025.3.1, so capture is
# unreachable there no matter what else is set. Verified: nightly venv has
# torch 2.11.0+xpu with torch.xpu.XPUGraph present.
#
# WHY IT MATTERS: measured eager floor is 21,293 launches/token x 6.5 us =
# 138 ms/token = 7.2 tok/s. Target is 20-70; upstream reports 118 at c=1.
# Only capture or fusion changes that ceiling.
#
# This script is IDEMPOTENT and touches only the nightly venv.
set -uo pipefail

VENV=${VENV:-/flare/ModCon/ngetty/venvs/torchtune-pt-nightly-xpu}
EXP=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving
PATCH=$EXP/wheel_patches/vllm_xpu_kernels_0.1.7_fused_moe_interface.py
TARGET=$VENV/lib/python3.12/site-packages/vllm_xpu_kernels/fused_moe_interface.py

echo "venv=$VENV"
[[ -d "$VENV" ]] || { echo "ERROR: venv not found"; exit 1; }

echo "== torch =="
"$VENV/bin/python" - <<'PY'
import torch
print(f"  torch {torch.__version__}")
print(f"  XPUGraph present: {hasattr(torch.xpu, 'XPUGraph')}")
from vllm.utils.torch_utils import supports_xpu_graph
print(f"  supports_xpu_graph(): {supports_xpu_graph()}")
PY

# The `situ` activation is MANDATORY -- K3's config sets hidden_act="situ",
# and the stock 0.1.7 wheel has zero situ support (grep -c situ: 7 in the
# patched file, 0 in the stock one). Without this the model cannot execute
# its own activation; it is not a performance patch.
echo "== situ wheel patch =="
have=$(sha256sum "$TARGET" 2>/dev/null | cut -d' ' -f1)
want=$(sha256sum "$PATCH" | cut -d' ' -f1)
if [[ "$have" == "$want" ]]; then
    echo "  already patched ($want)"
else
    cp -v "$PATCH" "$TARGET"
    now=$(sha256sum "$TARGET" | cut -d' ' -f1)
    [[ "$now" == "$want" ]] || { echo "ERROR: copy did not take"; exit 1; }
    echo "  patched -> $now"
fi
echo "  situ occurrences: $(grep -c situ "$TARGET")"

echo "== K3 remote code under this venv's transformers =="
# The nightly venv has transformers 5.7.0 vs the K3 venv's 4.57.1. Tested OK
# on 2026-08-11 for the two entry points vLLM touches at startup, but re-check
# here so a transformers bump does not surface as a mid-run failure.
"$VENV/bin/python" - <<'PY'
import transformers
print(f"  transformers {transformers.__version__}")
from transformers import AutoConfig, AutoTokenizer
p = "/flare/ModCon/ngetty/models/Kimi-K3"
c = AutoConfig.from_pretrained(p, trust_remote_code=True)
print(f"  AutoConfig OK: {type(c).__name__} model_type={getattr(c,'model_type',None)}")
t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
print(f"  AutoTokenizer OK: {type(t).__name__} vocab={len(t)}")
PY

echo ""
echo "READY. To serve K3 under this venv with decode-only graph capture:"
echo "  PYTHON=$VENV/bin/python RAY_ENV_MODE=torch211 \\"
echo "  ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_DECODE_ONLY \\"
echo "  VLLM_XPU_ENABLE_XPU_GRAPH=1 VLLM_XPU_ALLOW_GRAPH_WITH_COMMS=1 \\"
echo "  bash $EXP/serve_k3.sh --model ... --tp 32 --ep ..."
echo ""
echo "PYTHON and RAY_ENV_MODE must move TOGETHER -- setting PYTHON alone puts"
echo "the head on 2.11 and all 24 remote workers on 2.10. Each worker echoes"
echo "its own torch version in K3_WORKER_GATES; check before believing a result."

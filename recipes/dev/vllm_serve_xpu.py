"""
Bootstrap to launch TRL's vllm_serve on Aurora XPU.

Patches:
1. transformers version check (huggingface-hub 1.7.x vs <1.0 requirement)
2. XPU worker's determine_available_memory to use PyTorch memory stats
   instead of broken torch.xpu.mem_get_info() which overcounts non-torch allocs
3. Injects /load_weights_from_path/ endpoint for file-based weight sync
   (avoids XCCL communicator which SIGABRTs with concurrent PGs on XPU)
"""
import importlib.util
import logging
import os
import pathlib
import sys
import types

logger = logging.getLogger("vllm_serve_xpu")

# --- 1. Patch transformers version check ---
_utils_stub = types.ModuleType("transformers.utils")
_utils_stub.__path__ = []
sys.modules["transformers.utils"] = _utils_stub

for p in sys.path:
    _vp = pathlib.Path(p) / "transformers" / "utils" / "versions.py"
    if _vp.exists():
        _spec = importlib.util.spec_from_file_location(
            "transformers.utils.versions", str(_vp)
        )
        _vm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vm)
        _vm._compare_versions = lambda *a, **kw: None
        sys.modules["transformers.utils.versions"] = _vm
        break

del sys.modules["transformers.utils"]

# Set spawn method for XPU
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# --- 2. XPU memory patch ---
# Applied via usercustomize.py (on PYTHONPATH) so it also works in the
# EngineCore subprocess. No need to import XPUWorker here — that would
# trigger CCL init in the parent process and contaminate the CXI fabric.

# --- 3. Inject /load_weights_from_path/ endpoint ---
import uvicorn
import trl.scripts.vllm_serve as _serve_mod

_real_uvicorn_run = uvicorn.run
_worker_connections = []


def _patched_uvicorn_run(app, **kwargs):
    """Add /load_weights_from_path/ endpoint, then run the real uvicorn."""
    from pydantic import BaseModel

    class LoadWeightsRequest(BaseModel):
        path: str

    @app.post("/load_weights_from_path/")
    async def load_weights_from_path(request: LoadWeightsRequest):
        """Load weights from safetensors on local /tmp. No XCCL needed."""
        from safetensors.torch import load_file

        path = request.path
        if not os.path.exists(path):
            return {"status": "error", "message": f"Not found: {path}"}

        try:
            state_dict = load_file(path, device="cpu")
            weight_list = list(state_dict.items())

            # Send to each worker via collective_rpc
            for conn in _worker_connections:
                conn.send({
                    "type": "fire_and_forget",
                    "method": "collective_rpc",
                    "kwargs": {
                        "method": "load_weights",
                        "kwargs": {"weights": weight_list},
                    },
                })

            del state_dict
            logger.info("Loaded %d params from %s", len(weight_list), path)
            return {"status": "ok", "num_params": len(weight_list)}
        except Exception as e:
            logger.error("load_weights_from_path failed: %s", e)
            return {"status": "error", "message": str(e)}

    logger.info("Added /load_weights_from_path/ endpoint")
    return _real_uvicorn_run(app, **kwargs)


uvicorn.run = _patched_uvicorn_run

# Capture worker connections via Pipe patch
from multiprocessing import Pipe as _OrigPipe


def _capturing_pipe(*args, **kwargs):
    parent_conn, child_conn = _OrigPipe(*args, **kwargs)
    _worker_connections.append(parent_conn)
    return parent_conn, child_conn


_serve_mod.Pipe = _capturing_pipe

# --- 4. Run ---
import runpy
runpy.run_module("trl.scripts.vllm_serve", run_name="__main__", alter_sys=True)

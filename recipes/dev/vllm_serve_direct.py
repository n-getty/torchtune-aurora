"""
Direct vLLM server for XPU with TRL-compatible /generate/ endpoint.

Bypasses TRL's vllm_serve (which uses the sync LLM API, broken with V1 TP>1 on XPU).
Uses vLLM's AsyncLLM (same engine path as `vllm serve` CLI) which works with TP>1.

Usage:
    python3 vllm_serve_direct.py --model /path/to/model --tensor-parallel-size 2 \
        --distributed-executor-backend mp --port 8001

For TP>1, --distributed-executor-backend mp is required.
"""
import importlib.util
import logging
import os
import pathlib
import sys
import types

logger = logging.getLogger("vllm_serve_direct")
logging.basicConfig(level=logging.INFO)

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

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def main():
    parser = argparse.ArgumentParser(description="vLLM XPU server (direct)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--distributed-executor-backend", default=None)
    args = parser.parse_args()

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    tp = args.tensor_parallel_size

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        distributed_executor_backend=args.distributed_executor_backend,
    )

    logger.info(f"Initializing AsyncLLM: model={args.model}, TP={tp}")
    engine = AsyncLLM.from_engine_args(engine_args)
    logger.info("AsyncLLM initialized")

    app = FastAPI()

    class GenerateRequest(BaseModel):
        prompts: list
        n: int = 1
        max_tokens: int = 256
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        repetition_penalty: float = 1.0
        logprobs: int | None = None

    class LoadWeightsRequest(BaseModel):
        path: str

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/generate/")
    async def generate(request: GenerateRequest):
        """TRL-compatible /generate/ endpoint using AsyncLLM."""
        is_token_ids = request.prompts and isinstance(request.prompts[0], list)

        params = SamplingParams(
            n=request.n,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            logprobs=request.logprobs,
        )

        # Submit all prompts and collect final results
        async def run_one(prompt, req_id):
            if is_token_ids:
                inp = {"prompt_token_ids": prompt}
            else:
                inp = {"prompt": prompt}
            final = None
            async for output in engine.generate(inp, params, req_id):
                final = output
            return final

        tasks = []
        for prompt in request.prompts:
            req_id = str(uuid.uuid4())
            tasks.append(run_one(prompt, req_id))

        results = await asyncio.gather(*tasks)

        # Build response matching TRL contract
        completion_ids = []
        prompt_ids = []
        for result in results:
            prompt_ids.append(list(result.prompt_token_ids))
            for output in result.outputs:
                completion_ids.append(list(output.token_ids))

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,
            "logprob_token_ids": None,
        }

    @app.post("/load_weights_from_path/")
    async def load_weights_from_path(request: LoadWeightsRequest):
        """Load weights from safetensors file."""
        from safetensors.torch import load_file

        path = request.path
        if not os.path.exists(path):
            return {"status": "error", "message": f"Not found: {path}"}
        try:
            state_dict = load_file(path, device="cpu")
            weight_list = list(state_dict.items())
            engine.engine_core.model_executor.collective_rpc(
                method="load_weights",
                kwargs={"weights": weight_list},
            )
            del state_dict
            logger.info("Loaded %d params from %s", len(weight_list), path)
            return {"status": "ok", "num_params": len(weight_list)}
        except Exception as e:
            logger.error("load_weights_from_path failed: %s", e)
            return {"status": "error", "message": str(e)}

    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

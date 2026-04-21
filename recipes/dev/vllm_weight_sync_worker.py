"""
vLLM worker extension for file-based weight synchronization.

Exposes load_weights_from_path() callable via /collective_rpc endpoint.
Training recipe saves weights to /tmp as safetensors, then calls:

    POST /collective_rpc
    {"method": "load_weights_from_path", "args": ["/tmp/torchtune/weight_update.safetensors"]}

This avoids XCCL communicator setup which SIGABRTs on XPU when a second
process group is created concurrently with the training process group.

Usage: launch vLLM with --worker-extension-cls pointing to this class:
    python3 -m vllm.entrypoints.openai.api_server \
        --model /tmp/model \
        --worker-extension-cls recipes.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \
        ...

Note: The class must be importable in the worker process, so ensure PYTHONPATH
includes the torchtune project directory.
"""
import logging
import os

logger = logging.getLogger("vllm_weight_sync_worker")


class WeightSyncFromFileExtension:
    """vLLM worker extension that reloads weights from a safetensors file.

    Called via collective_rpc('load_weights_from_path', args=[path]).
    All TP workers call this simultaneously — each reads the same file
    independently (no communication needed since file is on local /tmp).
    """

    def load_weights_from_path(self, path: str) -> dict:
        """Load weights from a safetensors file on local disk.

        Args:
            path: Absolute path to a safetensors file written by the training
                  process (via safetensors.torch.save_file).

        Returns:
            dict with "status" and "num_params" keys.
        """
        import torch
        from safetensors.torch import load_file

        if not os.path.exists(path):
            logger.error("Weight sync file not found: %s", path)
            return {"status": "error", "message": f"Not found: {path}"}

        try:
            logger.info("Loading weights from %s", path)
            state_dict = load_file(path, device="cpu")
            weights = list(state_dict.items())
            n = len(weights)

            # Load into the model. self.model_runner.model is the vLLM model
            # (e.g. Qwen2ForCausalLM). load_weights() updates parameters in place.
            self.model_runner.model.load_weights(weights=weights)

            del state_dict
            del weights

            # Free cache after weight update
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()

            logger.info("Loaded %d params from %s", n, path)
            return {"status": "ok", "num_params": n}
        except Exception as e:
            logger.exception("load_weights_from_path failed")
            return {"status": "error", "message": str(e)}

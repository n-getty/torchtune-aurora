"""
vLLM worker extension for weight synchronization.

Provides three loading methods, all called via /collective_rpc endpoint:

  load_weights_from_path(path)   — legacy safetensors format (slow: 1.3 GB/s CPU serial)
  load_weights_from_raw(path)    — raw bytes file format (fast: ~2 GB/s DDR5 write)
  load_weights_from_shm(meta)    — POSIX shared memory (fastest: zero-copy read, ~0.2s write)

The shared memory format (load_weights_from_shm) is the recommended method for large models:
- Training side allocates a single SharedMemory block, copies weights via ctypes.memmove
  (DDR5 bandwidth, ~0.2s for 6 GB / ~1.8s for 62 GB, no Python object allocation)
- vLLM side maps the same physical RAM pages via torch.frombuffer(shm.buf, ...) — zero-copy
- Only metadata (tensor names/shapes/offsets as JSON) is passed via HTTP
- For 62 GB BF16: total ~2s vs ~30s for raw bytes (write+read), vs ~100s for safetensors

The raw bytes format is a fallback when SHM is unavailable:
  Files are written to /dev/shm (RAM-backed tmpfs, 504 GB on Aurora). Both training and
  vLLM processes are on the same node and see the same /dev/shm.

Async flow:
  1. All FSDP ranks participate in full_tensor() gather (~5.5s for 31B, unavoidable).
  2. Shard leader copies to SHM (or writes /dev/shm file) in a background thread.
  3. Background thread POSTs to vLLM to load.
  4. Next step's generation starts immediately after gather — sync is hidden.

This avoids XCCL communicator setup which SIGABRTs on XPU when a second
process group is created concurrently with the training process group.

Usage:
    python3 -m vllm.entrypoints.openai.api_server \\
        --model /tmp/model \\
        --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension
"""
import logging
import os
import time

logger = logging.getLogger("vllm_weight_sync_worker")


class WeightSyncFromFileExtension:
    """vLLM worker extension for file-based weight synchronization.

    Supports both the legacy safetensors format and the fast raw bytes format.
    Called via collective_rpc — all TP workers call simultaneously, each reads
    from the same /dev/shm path independently (no cross-worker communication).
    """

    def load_weights_from_path(self, path: str) -> dict:
        """Load weights from a safetensors file on /dev/shm.

        Legacy format — kept for backward compatibility. Use load_weights_from_raw
        for new code (40× faster for large models).
        """
        import torch
        from safetensors.torch import load_file

        if not os.path.exists(path):
            logger.error("Weight sync file not found: %s", path)
            return {"status": "error", "message": f"Not found: {path}"}

        try:
            t0 = time.perf_counter()
            state_dict = load_file(path, device="cpu")
            t_read = time.perf_counter() - t0
            weights = list(state_dict.items())
            n = len(weights)

            t_load0 = time.perf_counter()
            self.model_runner.model.load_weights(weights=weights)
            t_load = time.perf_counter() - t_load0

            del state_dict, weights

            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()

            logger.info(
                "load_weights_from_path: %d params in %.1fs (read=%.1fs load=%.1fs) from %s",
                n, time.perf_counter() - t0, t_read, t_load, path,
            )
            return {"status": "ok", "num_params": n, "read_s": round(t_read, 2), "load_s": round(t_load, 2)}
        except Exception as e:
            logger.exception("load_weights_from_path failed")
            return {"status": "error", "message": str(e)}

    def load_weights_from_raw(self, path: str) -> dict:
        """Load weights from a raw bytes file written by _save_raw_bytes().

        Format: 8-byte little-endian header length, then JSON header, then
        contiguous raw tensor bytes. BF16 tensors are stored as int16 bytes
        (same bit pattern) and reinterpreted on load.

        This is ~40× faster than safetensors for large models because:
        - No per-tensor serialization overhead
        - frombuffer() is a zero-copy view into the mmap'd file bytes
        - File is in /dev/shm (RAM) so reads are memory-bandwidth-limited
        """
        import struct
        import json
        import torch

        if not os.path.exists(path):
            logger.error("Raw weight sync file not found: %s", path)
            return {"status": "error", "message": f"Not found: {path}"}

        try:
            t0 = time.perf_counter()

            with open(path, "rb") as f:
                # Read header
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))

                # Read all tensor bytes in one pass
                weights = []
                for entry in header:
                    raw = f.read(entry["nbytes"])
                    dtype_str = entry["dtype"]  # e.g. "torch.bfloat16"
                    dtype = getattr(torch, dtype_str.split(".")[-1])
                    shape = entry["shape"]

                    # BF16 stored as int16 bytes (same bit pattern) — reinterpret.
                    # frombuffer requires a bytes-like object with buffer protocol.
                    if dtype == torch.bfloat16:
                        tensor = (
                            torch.frombuffer(raw, dtype=torch.int16)
                            .view(torch.bfloat16)
                            .reshape(shape)
                            .clone()  # detach from the read buffer
                        )
                    else:
                        tensor = (
                            torch.frombuffer(raw, dtype=dtype)
                            .reshape(shape)
                            .clone()
                        )
                    weights.append((entry["name"], tensor))

            t_read = time.perf_counter() - t0
            n = len(weights)

            t_load0 = time.perf_counter()
            self.model_runner.model.load_weights(weights=weights)
            t_load = time.perf_counter() - t_load0

            del weights

            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()

            logger.info(
                "load_weights_from_raw: %d params in %.1fs (read=%.1fs load=%.1fs) from %s",
                n, time.perf_counter() - t0, t_read, t_load, path,
            )
            return {"status": "ok", "num_params": n, "read_s": round(t_read, 2), "load_s": round(t_load, 2)}
        except Exception as e:
            logger.exception("load_weights_from_raw failed")
            return {"status": "error", "message": str(e)}

    def load_weights_from_shm(self, meta: str) -> dict:
        """Load weights from a POSIX shared memory block written by _sync_weights_to_vllm_shm().

        The training process allocates a single SharedMemory block and copies all gathered
        weights into it via ctypes.memmove (DDR5 bandwidth, no Python object allocation).
        This method maps the same physical RAM pages zero-copy via shm.buf, builds CPU
        weight tensors via torch.frombuffer (no copy), then passes them to load_weights()
        which does in-place param.copy_() — no extra GPU allocation needed.

        Args:
            meta: JSON string with keys:
                shm_name   — POSIX shared memory name (as passed to SharedMemory)
                total_bytes — total size of the SHM block
                tensors    — list of {name, shape, dtype, offset, nbytes}

        Key design: tensors are passed as CPU views directly to load_weights().
        This avoids allocating a second full copy of the model on XPU (which OOMs
        on 31B models that fill >95% of device memory). vLLM's load_weights does
        param.copy_(cpu_tensor) in-place — no extra XPU allocation.
        shm is kept open until after load_weights() so the buffer stays valid.
        """
        import json
        import torch
        from multiprocessing.shared_memory import SharedMemory

        try:
            t0 = time.perf_counter()
            meta_dict = json.loads(meta)
            shm_name = meta_dict["shm_name"]
            tensors_meta = meta_dict["tensors"]

            shm = SharedMemory(name=shm_name, create=False)

            weights = []
            for entry in tensors_meta:
                dtype_str = entry["dtype"]  # e.g. "torch.bfloat16"
                dtype = getattr(torch, dtype_str.split(".")[-1])
                shape = entry["shape"]
                offset = entry["offset"]
                nbytes = entry["nbytes"]

                # Zero-copy view into shared memory pages. frombuffer maps the same
                # physical RAM — no copy. BF16 stored as int16 for numpy compat.
                # Do NOT call .to(device) here — that would double XPU memory usage.
                # load_weights() does param.copy_(cpu_tensor) in-place instead.
                if dtype == torch.bfloat16:
                    n_elems = nbytes // 2
                    tensor = (
                        torch.frombuffer(shm.buf, dtype=torch.int16, offset=offset, count=n_elems)
                        .view(torch.bfloat16)
                        .reshape(shape)
                    )
                else:
                    itemsize = torch.tensor([], dtype=dtype).element_size()
                    n_elems = nbytes // itemsize
                    tensor = (
                        torch.frombuffer(shm.buf, dtype=dtype, offset=offset, count=n_elems)
                        .reshape(shape)
                    )

                weights.append((entry["name"], tensor))

            t_read = time.perf_counter() - t0
            n = len(weights)

            t_load0 = time.perf_counter()
            # shm stays open here — tensors are views into shm.buf, must stay valid
            self.model_runner.model.load_weights(weights=weights)
            t_load = time.perf_counter() - t_load0

            shm.close()  # Release mapping after load_weights — training side owns unlink
            del weights

            logger.info(
                "load_weights_from_shm: %d params in %.1fs (map=%.1fs load=%.1fs) from shm:%s",
                n, time.perf_counter() - t0, t_read, t_load, shm_name,
            )
            return {"status": "ok", "num_params": n, "read_s": round(t_read, 2), "load_s": round(t_load, 2)}
        except Exception as e:
            logger.exception("load_weights_from_shm failed")
            return {"status": "error", "message": str(e)}

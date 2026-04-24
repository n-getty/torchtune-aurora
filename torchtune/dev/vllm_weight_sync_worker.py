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

XCCL broadcast mode (init_xccl_communicator + receive_weights_xccl):
  Training rank 0 creates a cross-process XCCL group with this vLLM worker
  via TCPStore + ProcessGroupXCCL constructor. Weight transfer is GPU→GPU
  broadcast — no CPU staging, no file I/O. ~14 GB/s measured on Aurora XeLink.

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
    """vLLM worker extension for weight synchronization.

    Supports file-based (safetensors, raw bytes, SHM) and XCCL broadcast modes.
    Called via collective_rpc — all TP workers call simultaneously.
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

    # ------------------------------------------------------------------
    # XCCL broadcast weight sync
    # ------------------------------------------------------------------

    def init_xccl_communicator(
        self, host: str, port: int, world_size: int, base_rank: int,
        use_two_hop: bool = False,
    ) -> dict:
        """Create a cross-process XCCL group with the training rank.

        Called once at first weight sync via /collective_rpc (all TP workers
        call simultaneously). Training rank 0 is the TCPStore master and rank 0
        in the XCCL group. Each TP worker joins as base_rank + tp_rank.

        Args:
            base_rank: Starting rank for vLLM workers. TP worker i gets
                       rank = base_rank + i in the XCCL group.
            use_two_hop: If True, also create a separate intra-node XCCL PG
                covering all vLLM ranks (1..world_size-1). Rank 1 receives from
                training cross-node and then broadcasts intra-node via XeLink,
                reducing sync time from ~38s (12 sequential Slingshot sends) to
                ~3s (1 Slingshot send + XeLink broadcast).
        """
        import torch
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d

        try:
            t0 = time.perf_counter()

            # Clean up any stale XCCL pgs from a previous run.
            for attr in ('_xccl_pg', '_xccl_cross_pg', '_xccl_intra_pg'):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).abort()
                    except Exception:
                        pass
                    delattr(self, attr)
            self._is_intra_root = False

            device = next(self.model_runner.model.parameters()).device
            tp_rank = dist.get_rank() if dist.is_initialized() else 0
            my_rank = base_rank + tp_rank
            logger.info(
                "init_xccl_communicator: connecting to %s:%d (world=%d, my_rank=%d, tp_rank=%d, "
                "device=%s, two_hop=%s)",
                host, port, world_size, my_rank, tp_rank, device, use_two_hop,
            )

            import datetime
            store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=False,
                timeout=datetime.timedelta(seconds=120),
            )
            opts = c10d.ProcessGroupXCCL.Options()
            self._xccl_rank = my_rank
            self._xccl_device = device

            if use_two_hop:
                # 2-hop design: XCCL send/recv is not supported (hangs); use broadcast.
                #
                # Cross PG (size=2): training rank 0 + vLLM rank 1 only.
                #   broadcast(root=0) sends one cross-Slingshot copy (~2.4s for 61 GiB
                #   at ~25 GB/s) instead of 12 sequential copies (38s flat broadcast).
                #   Only vLLM rank 1 joins this PG — other vLLM ranks skip it.
                #
                # Intra PG (size=world_size-1): all vLLM ranks.
                #   broadcast(root=0) distributes intra-node via XeLink (~0.6s).
                #   Rank 1 (intra_rank=0) is the intra root; it broadcasts after
                #   receiving from training.
                intra_rank = my_rank - 1
                intra_size = world_size - 1

                if my_rank == 1:
                    # Only vLLM rank 1 joins the 2-rank cross PG.
                    cross_prefixed = c10d.PrefixStore("wsync_cross", store)
                    self._xccl_cross_pg = c10d.ProcessGroupXCCL(
                        store=cross_prefixed, rank=1, size=2, options=opts,
                    )
                    self._is_intra_root = True
                    logger.info("init_xccl_communicator: cross PG ready (rank 1/2, intra root)")

                intra_prefixed = c10d.PrefixStore("wsync_intra", store)
                self._xccl_intra_pg = c10d.ProcessGroupXCCL(
                    store=intra_prefixed, rank=intra_rank, size=intra_size, options=opts,
                )
                logger.info(
                    "init_xccl_communicator: intra PG ready (rank=%d/%d, is_root=%s)",
                    intra_rank, intra_size, self._is_intra_root,
                )
            else:
                # Legacy flat broadcast: training rank 0 broadcasts to all vLLM ranks.
                prefixed = c10d.PrefixStore("wsync", store)
                self._xccl_pg = c10d.ProcessGroupXCCL(
                    store=prefixed, rank=my_rank, size=world_size, options=opts,
                )

            dt = time.perf_counter() - t0
            logger.info("init_xccl_communicator: ready in %.1fs", dt)
            return {"status": "ok", "init_s": round(dt, 2), "two_hop": use_two_hop}
        except Exception as e:
            logger.exception("init_xccl_communicator failed")
            return {"status": "error", "message": str(e)}

    def receive_weights_xccl(self, meta: str) -> dict:
        """Receive weights via XCCL broadcast from the training rank.

        Training rank 0 broadcasts a flat bf16 buffer containing all model params.
        This method allocates a receive buffer, does the broadcast receive, splits
        the buffer into individual params, and applies them to the model.

        The broadcast is blocking — this runs inside the /collective_rpc handler
        which is synchronous. Training rank 0 starts its broadcast after POSTing
        this request (the request triggers the receive, then training broadcasts).

        To avoid deadlock: training POSTs this request, the handler enters the
        broadcast receive (blocking), training then broadcasts. The collective
        synchronizes both sides.
        """
        import json
        import torch

        if not hasattr(self, '_xccl_pg'):
            return {"status": "error", "message": "XCCL communicator not initialized"}

        try:
            t0 = time.perf_counter()
            meta_dict = json.loads(meta)
            total_elements = meta_dict["total_elements"]
            tensors_meta = meta_dict["tensors"]

            recv_buf = torch.empty(
                total_elements, device=self._xccl_device, dtype=torch.bfloat16,
            )

            t_bcast0 = time.perf_counter()
            self._xccl_pg.broadcast(recv_buf, root=0).wait()
            torch.xpu.synchronize(self._xccl_device)
            t_bcast = time.perf_counter() - t_bcast0

            weights = []
            offset = 0
            for entry in tensors_meta:
                n_elems = entry["numel"]
                shape = entry["shape"]
                name = entry["name"]
                param_tensor = recv_buf[offset:offset + n_elems].reshape(shape)
                weights.append((name, param_tensor))
                offset += n_elems

            n = len(weights)
            t_load0 = time.perf_counter()
            self.model_runner.model.load_weights(weights=weights)
            t_load = time.perf_counter() - t_load0

            del recv_buf, weights
            torch.xpu.empty_cache()

            gb = total_elements * 2 / 1024**3
            logger.info(
                "receive_weights_xccl: %d params %.2f GiB in %.1fs "
                "(bcast=%.1fs %.1f GB/s, load=%.1fs)",
                n, gb, time.perf_counter() - t0,
                t_bcast, gb / t_bcast if t_bcast > 0 else 0, t_load,
            )
            return {
                "status": "ok", "num_params": n,
                "bcast_s": round(t_bcast, 2), "load_s": round(t_load, 2),
            }
        except Exception as e:
            logger.exception("receive_weights_xccl failed")
            return {"status": "error", "message": str(e)}

    def receive_weights_xccl_streaming(self, manifest: str) -> dict:
        """Batched XCCL weight receive for large models (32B+).

        Training side sends a manifest listing all params, then broadcasts batches
        of params concatenated into flat tensors (~1 GiB per broadcast call).
        This reduces XCCL overhead from 707 calls × ~49ms to ~130 calls for 32B.

        Both sides use the same greedy batching algorithm (batch_max_numel from manifest)
        to ensure broadcast calls match exactly.
        """
        import json
        import torch

        if not hasattr(self, '_xccl_pg') and not hasattr(self, '_xccl_intra_pg'):
            return {"status": "error", "message": "XCCL communicator not initialized"}

        try:
            t0 = time.perf_counter()
            manifest_dict = json.loads(manifest)
            tensors_meta = manifest_dict["tensors"]
            batch_max_numel = manifest_dict.get("batch_max_numel", 0)
            # Legacy per-param mode if no batch_max_numel
            apply_every = manifest_dict.get("apply_every", 64)

            n_params = len(tensors_meta)
            total_elements = sum(e["numel"] for e in tensors_meta)
            gb = total_elements * 2 / 1024**3

            t_bcast_total = 0.0
            t_load_total = 0.0

            two_hop = hasattr(self, '_xccl_intra_pg')

            if batch_max_numel > 0:
                # Batched mode: receive one flat tensor per batch, split back into params.
                # Same greedy split as training side: flush when adding next param exceeds max.
                i = 0
                while i < n_params:
                    batch_start = i
                    batch_numel = 0
                    while i < n_params:
                        pn = tensors_meta[i]["numel"]
                        if batch_numel > 0 and batch_numel + pn > batch_max_numel:
                            break
                        batch_numel += pn
                        i += 1

                    recv_buf = torch.empty(
                        batch_numel, device=self._xccl_device, dtype=torch.bfloat16,
                    )
                    t_b0 = time.perf_counter()
                    if two_hop:
                        # 2-hop: rank 1 receives cross-node from training via 2-rank
                        # broadcast PG, then distributes to all vLLM ranks via intra PG.
                        if self._is_intra_root:
                            self._xccl_cross_pg.broadcast(recv_buf, root=0).wait()
                            self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                        else:
                            self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                    else:
                        self._xccl_pg.broadcast(recv_buf, root=0).wait()
                    t_bcast_total += time.perf_counter() - t_b0

                    # Split flat buffer back into per-param tensors and apply
                    offset = 0
                    batch_weights = []
                    for entry in tensors_meta[batch_start:i]:
                        n = entry["numel"]
                        batch_weights.append(
                            (entry["name"], recv_buf[offset:offset + n].reshape(entry["shape"]))
                        )
                        offset += n
                    t_l0 = time.perf_counter()
                    self.model_runner.model.load_weights(weights=batch_weights)
                    t_load_total += time.perf_counter() - t_l0
                    del recv_buf, batch_weights

            else:
                # Legacy: one broadcast per param
                weights_batch = []
                for idx, entry in enumerate(tensors_meta):
                    recv_buf = torch.empty(
                        entry["numel"], device=self._xccl_device, dtype=torch.bfloat16,
                    )
                    t_b0 = time.perf_counter()
                    if two_hop:
                        if self._is_intra_root:
                            self._xccl_cross_pg.broadcast(recv_buf, root=0).wait()
                            self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                        else:
                            self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                    else:
                        self._xccl_pg.broadcast(recv_buf, root=0).wait()
                    t_bcast_total += time.perf_counter() - t_b0

                    weights_batch.append((entry["name"], recv_buf.reshape(entry["shape"])))

                    if len(weights_batch) >= apply_every or idx == n_params - 1:
                        t_l0 = time.perf_counter()
                        self.model_runner.model.load_weights(weights=weights_batch)
                        t_load_total += time.perf_counter() - t_l0
                        del weights_batch
                        weights_batch = []

            torch.xpu.synchronize(self._xccl_device)

            logger.info(
                "receive_weights_xccl_streaming: %d params %.2f GiB in %.1fs "
                "(bcast=%.1fs %.1f GB/s, load=%.1fs)",
                n_params, gb, time.perf_counter() - t0,
                t_bcast_total, gb / t_bcast_total if t_bcast_total > 0 else 0,
                t_load_total,
            )
            return {
                "status": "ok", "num_params": n_params,
                "bcast_s": round(t_bcast_total, 2), "load_s": round(t_load_total, 2),
            }
        except Exception as e:
            logger.exception("receive_weights_xccl_streaming failed")
            return {"status": "error", "message": str(e)}

    def close_xccl_communicator(self) -> dict:
        """Tear down the XCCL weight sync process group.

        Called by training side during cleanup so both ends abort the PG
        before the training process exits. Uses abort() (unilateral) rather
        than destroy_process_group() (collective) because the PG is created
        via c10d.ProcessGroupXCCL directly and is not in dist's registry.
        """
        try:
            initialized = any(hasattr(self, a) for a in ('_xccl_pg', '_xccl_cross_pg', '_xccl_intra_pg'))
            if not initialized:
                return {"status": "ok", "message": "not initialized"}
            for attr in ('_xccl_intra_pg', '_xccl_cross_pg', '_xccl_pg'):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).abort()
                    except Exception:
                        pass
                    delattr(self, attr)
            logger.info("close_xccl_communicator: XCCL PGs aborted")
            return {"status": "ok"}
        except Exception as e:
            logger.warning("close_xccl_communicator: %s", e)
            return {"status": "error", "message": str(e)}

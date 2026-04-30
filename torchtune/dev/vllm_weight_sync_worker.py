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
import re
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

        MoE expert weights bypass model.load_weights() and are copied directly to
        the fused w13/w2 params. This is necessary because IPEX's GatedMLPMOE transposes
        the weight data in-place on first forward, making vLLM's weight_loader narrow
        logic incompatible with the post-prepack shapes.

        Args:
            meta: JSON string with keys:
                shm_name   — POSIX shared memory name (as passed to SharedMemory)
                total_bytes — total size of the SHM block
                tensors    — list of {name, shape, dtype, offset, nbytes}
        """
        import json
        import re
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
                dtype_str = entry["dtype"]
                dtype = getattr(torch, dtype_str.split(".")[-1])
                shape = entry["shape"]
                offset = entry["offset"]
                nbytes = entry["nbytes"]

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

            fused_re = re.compile(
                r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
            )
            fused_data = {}
            non_expert = []
            for name, tensor in weights:
                m = fused_re.match(name)
                if m:
                    layer_idx = int(m.group(1))
                    kind = m.group(2)
                    fused_data.setdefault(layer_idx, {})[kind] = tensor
                else:
                    non_expert.append((name, tensor))

            if non_expert:
                self.model_runner.model.load_weights(weights=non_expert)

            if fused_data:
                self._load_fused_moe_experts(fused_data)

            t_load = time.perf_counter() - t_load0

            shm.close()
            del weights

            logger.info(
                "load_weights_from_shm: %d params in %.1fs (map=%.1fs load=%.1fs) from shm:%s",
                n, time.perf_counter() - t0, t_read, t_load, shm_name,
            )
            return {"status": "ok", "num_params": n, "read_s": round(t_read, 2), "load_s": round(t_load, 2)}
        except Exception as e:
            logger.exception("load_weights_from_shm failed")
            return {"status": "error", "message": str(e)}

    def _load_fused_moe_experts(self, fused_data: dict) -> None:
        """Copy pre-fused MoE w13/w2 weights directly to vLLM params.

        Receives pre-fused tensors from the training side (gate+up already
        concatenated into w13), so no per-expert stacking is needed. Just
        TP-shard, detect IPEX transpose, and copy.

        Args:
            fused_data: {layer_idx: {"w13": tensor, "w2": tensor}}
                w13: [E, 2*intermediate, hidden]  (gate || up on dim=1)
                w2:  [E, hidden, intermediate]     (down)
        """
        import torch
        import torch.distributed as dist

        tp_rank = dist.get_rank() if dist.is_initialized() else 0
        tp_size = dist.get_world_size() if dist.is_initialized() else 1

        params = dict(self.model_runner.model.named_parameters())

        for layer_idx in sorted(fused_data.keys()):
            w13 = fused_data[layer_idx]["w13"]
            w2 = fused_data[layer_idx]["w2"]

            inter = w13.shape[1] // 2
            inter_per_tp = inter // tp_size

            gate_shard = w13[:, tp_rank * inter_per_tp:(tp_rank + 1) * inter_per_tp, :]
            up_shard = w13[:, inter + tp_rank * inter_per_tp:inter + (tp_rank + 1) * inter_per_tp, :]
            w13_tp = torch.cat([gate_shard, up_shard], dim=1)

            w2_tp = w2[:, :, tp_rank * inter_per_tp:(tp_rank + 1) * inter_per_tp]

            w13_key = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
            w2_key = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
            w13_param = params[w13_key]
            w2_param = params[w2_key]

            num_experts = w13.shape[0]
            e_local = w13_param.shape[0]
            if e_local < num_experts:
                ep_start = tp_rank * e_local
                w13_tp = w13_tp[ep_start:ep_start + e_local]
                w2_tp = w2_tp[ep_start:ep_start + e_local]

            device = w13_param.device
            is_transposed = w13_param.shape[1] != w13_tp.shape[1]
            if is_transposed:
                # Move to GPU before transpose: 1.6 TB/s GPU vs 20 GB/s CPU
                w13_param.data.copy_(w13_tp.to(device).transpose(1, 2).contiguous())
                w2_param.data.copy_(w2_tp.to(device).transpose(1, 2).contiguous())
            else:
                w13_param.data.copy_(w13_tp)
                w2_param.data.copy_(w2_tp)

    # ------------------------------------------------------------------
    # XCCL broadcast weight sync
    # ------------------------------------------------------------------

    def init_xccl_communicator(
        self, host: str, port: int, world_size: int, base_rank: int,
        use_two_hop: bool = False, wsync_method: str = "xccl_sendrecv",
        pool_size: int = 0,
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
            pool_size: Number of sender ranks in the dynamic sender pool.
                0 = legacy single sender (rank 0). >0 = create pool_size
                cross-PGs for rotating sender ranks.
        """
        import torch
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d

        try:
            t0 = time.perf_counter()

            # Clean up any stale PGs from a previous run.
            for attr in ('_xccl_pg', '_xccl_cross_pg', '_xccl_intra_pg'):
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).abort()
                    except Exception:
                        pass
                    delattr(self, attr)
            if hasattr(self, '_xccl_cross_pgs'):
                for pg in self._xccl_cross_pgs:
                    try:
                        pg.abort()
                    except Exception:
                        pass
                del self._xccl_cross_pgs
            self._is_intra_root = False
            self._gloo_recv_buf = None

            device = next(self.model_runner.model.parameters()).device
            tp_rank = dist.get_rank() if dist.is_initialized() else 0
            tp_size_local = dist.get_world_size() if dist.is_initialized() else 1
            my_rank = base_rank + tp_rank
            replica_idx = (my_rank - 1) // tp_size_local
            logger.info(
                "init_xccl_communicator: connecting to %s:%d (world=%d, my_rank=%d, tp_rank=%d, "
                "tp_size=%d, replica=%d, device=%s, two_hop=%s, pool_size=%d)",
                host, port, world_size, my_rank, tp_rank, tp_size_local,
                replica_idx, device, use_two_hop, pool_size,
            )

            import datetime
            store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=False,
                timeout=datetime.timedelta(seconds=120),
            )
            self._xccl_rank = my_rank
            self._xccl_device = device
            self._xccl_store = store
            self._wsync_pg_gen = 0
            self._wsync_pg_reset_interval = int(os.environ.get("WSYNC_PG_RESET_INTERVAL", "0"))

            if use_two_hop:
                intra_rank = tp_rank
                intra_size = tp_size_local
                self._wsync_cross_method = wsync_method
                intra_method = os.environ.get("WSYNC_INTRA_METHOD", "xccl")
                self._wsync_intra_method = intra_method

                if tp_rank == 0:
                    if pool_size > 0:
                        # Dynamic sender pool: create N cross-PGs
                        self._xccl_cross_pgs = []
                        for i in range(pool_size):
                            prefix = f"wsync_sender_{i}"
                            prefixed = c10d.PrefixStore(prefix, store)
                            if wsync_method == "gloo":
                                pg = c10d.ProcessGroupGloo(
                                    store=prefixed, rank=1, size=2,
                                )
                            else:
                                opts_cross = c10d.ProcessGroupXCCL.Options()
                                pg = c10d.ProcessGroupXCCL(
                                    store=prefixed, rank=1, size=2,
                                    options=opts_cross,
                                )
                            self._xccl_cross_pgs.append(pg)
                            logger.info(
                                "init_xccl_communicator: cross PG %d/%d created "
                                "(method=%s)", i, pool_size, wsync_method)
                        self._xccl_cross_pg = self._xccl_cross_pgs[0]
                    else:
                        cross_prefixed = c10d.PrefixStore(f"wsync_cross_{replica_idx}", store)
                        if wsync_method == "gloo":
                            self._xccl_cross_pg = c10d.ProcessGroupGloo(
                                store=cross_prefixed, rank=1, size=2,
                            )
                        else:
                            opts_cross = c10d.ProcessGroupXCCL.Options()
                            self._xccl_cross_pg = c10d.ProcessGroupXCCL(
                                store=cross_prefixed, rank=1, size=2,
                                options=opts_cross,
                            )
                    self._is_intra_root = True
                    self._gloo_recv_buf = None
                    logger.info(
                        "init_xccl_communicator: cross PG ready (rank 1/2, "
                        "method=%s, pool=%d, intra root)",
                        wsync_method, pool_size)

                intra_prefixed = c10d.PrefixStore(f"wsync_intra_{replica_idx}", store)
                if intra_method == "gloo":
                    self._xccl_intra_pg = c10d.ProcessGroupGloo(
                        store=intra_prefixed, rank=intra_rank, size=intra_size,
                    )
                else:
                    opts = c10d.ProcessGroupXCCL.Options()
                    self._xccl_intra_pg = c10d.ProcessGroupXCCL(
                        store=intra_prefixed, rank=intra_rank, size=intra_size,
                        options=opts,
                    )
                self._gloo_intra_buf = None
                logger.info(
                    "init_xccl_communicator: intra PG ready (replica=%d, rank=%d/%d, method=%s, is_root=%s)",
                    replica_idx, intra_rank, intra_size, intra_method, self._is_intra_root,
                )
            else:
                # Legacy flat broadcast: training rank 0 broadcasts to all vLLM ranks.
                opts = c10d.ProcessGroupXCCL.Options()
                prefixed = c10d.PrefixStore("wsync", store)
                self._xccl_pg = c10d.ProcessGroupXCCL(
                    store=prefixed, rank=my_rank, size=world_size, options=opts,
                )

            dt = time.perf_counter() - t0
            logger.info("init_xccl_communicator: ready in %.1fs", dt)
            return {"status": "ok", "init_s": round(dt, 2), "two_hop": use_two_hop,
                    "pool_size": pool_size}
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

            # Select active cross-PG for sender pool
            sender_index = manifest_dict.get("sender_index", -1)
            if sender_index >= 0 and hasattr(self, '_xccl_cross_pgs'):
                self._xccl_cross_pg = self._xccl_cross_pgs[sender_index]

            n_params = len(tensors_meta)
            total_elements = sum(e["numel"] for e in tensors_meta)
            gb = total_elements * 2 / 1024**3

            t_bcast_total = 0.0
            t_load_total = 0.0

            two_hop = hasattr(self, '_xccl_intra_pg')

            if batch_max_numel > 0:
                # Batched mode: receive one flat tensor per batch, split back into params.
                # Same greedy split as training side: flush when adding next param exceeds max.

                # Static buffer: reuse the same VA every step so oneCCL registers
                # the IPC handle once and gets 100% cache hits thereafter.
                # Size must cover the largest actual batch, which can exceed
                # batch_max_numel when a single param is larger than the limit
                # (the greedy split always includes the first param in a batch).
                max_single = max(e["numel"] for e in tensors_meta)
                buf_numel = max(batch_max_numel, max_single)
                if not hasattr(self, '_xccl_recv_buf') or self._xccl_recv_buf is None or self._xccl_recv_buf.numel() < buf_numel:
                    self._xccl_recv_buf = torch.empty(
                        buf_numel, device=self._xccl_device, dtype=torch.bfloat16)
                    logger.info("XCCL recv buf allocated: %d elements, data_ptr=0x%x",
                                buf_numel, self._xccl_recv_buf.data_ptr())

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

                    recv_buf = self._xccl_recv_buf[:batch_numel]
                    t_b0 = time.perf_counter()
                    if two_hop:
                        intra_method = getattr(self, '_wsync_intra_method', 'xccl')
                        if self._is_intra_root:
                            cross_method = getattr(self, '_wsync_cross_method', 'gloo')
                            if cross_method == "gloo":
                                if self._gloo_recv_buf is None or self._gloo_recv_buf.numel() < batch_numel:
                                    self._gloo_recv_buf = torch.empty(batch_numel, dtype=torch.bfloat16)
                                    logger.info("gloo recv buf allocated: %d elements", batch_numel)
                                cpu_recv = self._gloo_recv_buf[:batch_numel]
                                self._xccl_cross_pg.broadcast(cpu_recv, root=0).wait()
                                if intra_method == "gloo":
                                    self._xccl_intra_pg.broadcast(cpu_recv, root=0).wait()
                                    recv_buf.copy_(cpu_recv)
                                else:
                                    recv_buf.copy_(cpu_recv)
                                    self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                            elif cross_method == "xccl_sendrecv":
                                self._xccl_cross_pg.recv([recv_buf], 0, 0).wait()
                                self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                            else:
                                self._xccl_cross_pg.broadcast(recv_buf, root=0).wait()
                                self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                        else:
                            if intra_method == "gloo":
                                if self._gloo_intra_buf is None or self._gloo_intra_buf.numel() < batch_numel:
                                    self._gloo_intra_buf = torch.empty(batch_numel, dtype=torch.bfloat16)
                                    logger.info("gloo intra buf allocated: %d elements", batch_numel)
                                cpu_buf = self._gloo_intra_buf[:batch_numel]
                                self._xccl_intra_pg.broadcast(cpu_buf, root=0).wait()
                                recv_buf.copy_(cpu_buf)
                            else:
                                self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                    else:
                        self._xccl_pg.broadcast(recv_buf, root=0).wait()
                    t_bcast_total += time.perf_counter() - t_b0

                    # Split flat buffer back into per-param tensors, routing
                    # fused MoE experts to _load_fused_moe_experts (GPU-direct)
                    offset = 0
                    non_expert_weights = []
                    fused_data = {}
                    _fused_re = re.compile(
                        r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
                    )
                    for entry in tensors_meta[batch_start:i]:
                        n = entry["numel"]
                        tensor = recv_buf[offset:offset + n].reshape(entry["shape"])
                        m = _fused_re.match(entry["name"])
                        if m:
                            layer_idx = int(m.group(1))
                            kind = m.group(2)
                            fused_data.setdefault(layer_idx, {})[kind] = tensor
                        else:
                            non_expert_weights.append((entry["name"], tensor))
                        offset += n
                    t_l0 = time.perf_counter()
                    if non_expert_weights:
                        self.model_runner.model.load_weights(weights=non_expert_weights)
                    if fused_data:
                        self._load_fused_moe_experts(fused_data)
                    t_load_total += time.perf_counter() - t_l0
                    del non_expert_weights, fused_data

            else:
                # Legacy: one broadcast per param
                weights_batch = []
                for idx, entry in enumerate(tensors_meta):
                    recv_buf = torch.empty(
                        entry["numel"], device=self._xccl_device, dtype=torch.bfloat16,
                    )
                    t_b0 = time.perf_counter()
                    if two_hop:
                        intra_method = getattr(self, '_wsync_intra_method', 'xccl')
                        if self._is_intra_root:
                            cross_method = getattr(self, '_wsync_cross_method', 'gloo')
                            if cross_method == "gloo":
                                cpu_recv = torch.empty(entry["numel"], dtype=torch.bfloat16)
                                self._xccl_cross_pg.broadcast(cpu_recv, root=0).wait()
                                if intra_method == "gloo":
                                    self._xccl_intra_pg.broadcast(cpu_recv, root=0).wait()
                                    recv_buf.copy_(cpu_recv)
                                else:
                                    recv_buf.copy_(cpu_recv)
                                    self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                                del cpu_recv
                            elif cross_method == "xccl_sendrecv":
                                self._xccl_cross_pg.recv([recv_buf], 0, 0).wait()
                                self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                            else:
                                self._xccl_cross_pg.broadcast(recv_buf, root=0).wait()
                                self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                        else:
                            if intra_method == "gloo":
                                cpu_recv = torch.empty(entry["numel"], dtype=torch.bfloat16)
                                self._xccl_intra_pg.broadcast(cpu_recv, root=0).wait()
                                recv_buf.copy_(cpu_recv)
                                del cpu_recv
                            else:
                                self._xccl_intra_pg.broadcast(recv_buf, root=0).wait()
                    else:
                        self._xccl_pg.broadcast(recv_buf, root=0).wait()
                    t_bcast_total += time.perf_counter() - t_b0

                    weights_batch.append((entry["name"], recv_buf.reshape(entry["shape"])))

                    if len(weights_batch) >= apply_every or idx == n_params - 1:
                        t_l0 = time.perf_counter()
                        # Route fused MoE experts to GPU-direct path
                        _fused_re_leg = re.compile(
                            r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
                        )
                        non_expert = []
                        fused_leg = {}
                        for wname, wtensor in weights_batch:
                            m = _fused_re_leg.match(wname)
                            if m:
                                li = int(m.group(1))
                                fused_leg.setdefault(li, {})[m.group(2)] = wtensor
                            else:
                                non_expert.append((wname, wtensor))
                        if non_expert:
                            self.model_runner.model.load_weights(weights=non_expert)
                        if fused_leg:
                            self._load_fused_moe_experts(fused_leg)
                        t_load_total += time.perf_counter() - t_l0
                        del weights_batch, non_expert, fused_leg
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
            self._xccl_recv_buf = None
            logger.info("close_xccl_communicator: XCCL PGs aborted")
            return {"status": "ok"}
        except Exception as e:
            logger.warning("close_xccl_communicator: %s", e)
            return {"status": "error", "message": str(e)}

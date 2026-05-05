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

        Receives pre-fused GLOBAL tensors from training (gate+up already
        concatenated into w13). vLLM may shard those tensors via:
          - tensor parallelism (TP) along the intermediate dim, and/or
          - expert parallelism (EP) along the expert dim.
        Both are read off the FusedMoE module itself (``expert_map``,
        ``local_num_experts``, ``tp_rank``, ``tp_size``) instead of guessed
        from ``dist.get_rank()`` — those guesses are wrong when vLLM has
        folded TP into EP (e.g. tp_=4 + DP_=2 → ep_size=8, tp_size=1).

        Args:
            fused_data: {layer_idx: {"w13": tensor, "w2": tensor}}
                w13: [E_global, 2*intermediate, hidden]  (gate || up on dim=1)
                w2:  [E_global, hidden, intermediate]     (down)
        """
        import torch

        params = dict(self.model_runner.model.named_parameters())
        model = self.model_runner.model

        for layer_idx in sorted(fused_data.keys()):
            w13 = fused_data[layer_idx]["w13"]
            w2 = fused_data[layer_idx]["w2"]

            w13_key = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
            w2_key = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
            w13_param = params[w13_key]
            w2_param = params[w2_key]

            # Resolve the FusedMoE layer to read its actual EP/TP layout.
            try:
                moe_layer = model.get_submodule(
                    f"model.layers.{layer_idx}.mlp.experts"
                )
            except AttributeError:
                moe_layer = None

            global_e = w13.shape[0]
            local_e = w13_param.shape[0]
            expert_map = getattr(moe_layer, "expert_map", None)
            moe_tp_rank = getattr(moe_layer, "tp_rank", 0) or 0
            moe_tp_size = getattr(moe_layer, "tp_size", 1) or 1
            moe_ep_rank = getattr(moe_layer, "ep_rank", 0) or 0
            moe_ep_size = getattr(moe_layer, "ep_size", 1) or 1
            moe_local_n = getattr(moe_layer, "local_num_experts", local_e)

            if layer_idx == 0:
                em_summary = (
                    f"len={expert_map.numel()} owned={(expert_map >= 0).sum().item()}"
                    if expert_map is not None
                    else "None"
                )
                logger.info(
                    "MOE-DIAG L%d: w13_src=%s w2_src=%s w13_param=%s w2_param=%s "
                    "moe_tp=(%d/%d) moe_ep=(%d/%d) local_n=%s expert_map=%s",
                    layer_idx, list(w13.shape), list(w2.shape),
                    list(w13_param.shape), list(w2_param.shape),
                    moe_tp_rank, moe_tp_size,
                    moe_ep_rank, moe_ep_size,
                    moe_local_n, em_summary,
                )

            # --- EP slice along expert dim 0 ---------------------------------
            if local_e < global_e:
                if expert_map is not None:
                    # Authoritative: expert_map[g] = local_idx if owned else -1.
                    em = expert_map.to("cpu")
                    owned_global = (em >= 0).nonzero(as_tuple=True)[0]
                    if owned_global.numel() != local_e:
                        raise RuntimeError(
                            f"layer {layer_idx}: expert_map says "
                            f"{owned_global.numel()} owned, param has {local_e}"
                        )
                    perm = torch.empty(local_e, dtype=torch.long)
                    perm[em[owned_global].long()] = owned_global
                    w13 = w13.index_select(0, perm)
                    w2 = w2.index_select(0, perm)
                else:
                    # Fallback: contiguous EP partition (vLLM's default in
                    # ``determine_expert_map``). Derive ep_rank from sizes.
                    if global_e % local_e != 0:
                        raise RuntimeError(
                            f"layer {layer_idx}: cannot derive EP rank from "
                            f"global={global_e} local={local_e}"
                        )
                    ep_size_inferred = global_e // local_e
                    # Without expert_map we cannot know which slab — log so
                    # the operator can wire ``expert_map`` if this fires.
                    raise RuntimeError(
                        f"layer {layer_idx}: FusedMoE has no expert_map but "
                        f"global={global_e} > local={local_e} "
                        f"(inferred ep_size={ep_size_inferred}). vLLM "
                        f"build appears to predate expert_map; refuse to "
                        f"silently mis-shard."
                    )

            # --- TP slice along intermediate dim -----------------------------
            inter = w13.shape[1] // 2
            inter_per_tp = inter // moe_tp_size
            if moe_tp_size == 1:
                w13_tp = w13
                w2_tp = w2
            else:
                gate_shard = w13[
                    :, moe_tp_rank * inter_per_tp:(moe_tp_rank + 1) * inter_per_tp, :
                ]
                up_shard = w13[
                    :,
                    inter + moe_tp_rank * inter_per_tp:inter + (moe_tp_rank + 1) * inter_per_tp,
                    :,
                ]
                w13_tp = torch.cat([gate_shard, up_shard], dim=1)
                w2_tp = w2[
                    :, :, moe_tp_rank * inter_per_tp:(moe_tp_rank + 1) * inter_per_tp
                ]

            device = w13_param.device
            is_transposed = w13_param.shape[1] != w13_tp.shape[1]
            if is_transposed:
                # Move to GPU before transpose: 1.6 TB/s GPU vs 20 GB/s CPU
                w13_param.data.copy_(w13_tp.to(device).transpose(1, 2).contiguous())
                w2_param.data.copy_(w2_tp.to(device).transpose(1, 2).contiguous())
            else:
                w13_param.data.copy_(w13_tp.to(device))
                w2_param.data.copy_(w2_tp.to(device))

    def _load_fused_moe_experts_sharded(self, sharded_data: dict) -> None:
        """WS10 receiver — assemble per-trainer-rank expert shards via expert_map.

        See ``docs/reports/MoE_EP_status_ws8_ws10_design.md`` §WS10 and
        ``tests/torchtune/dev/rl/test_sharded_vllm_moe_sync_equivalence.py``.

        Trainer EP rank R owns interleaved global expert ids
        ``[R, R+ep_d, R+2*ep_d, ..., R+(n_local-1)*ep_d]``. For each
        (layer, projection), every trainer rank's shard is broadcast over
        ``_xccl_wsync_pg``; this method walks each shard's local index,
        derives the global id, looks it up in the FusedMoE ``expert_map``,
        and scatter-copies into the local param when owned (``>= 0``).

        Args:
            sharded_data: ``{layer_idx: {"w13": {R: shard, ...},
                                         "w2":  {R: shard, ...},
                                         "ep_degree": int,
                                         "n_local_trainer": int}}``
                Each shard is ``[n_local_trainer, ...]`` from trainer rank R.
        """
        import torch

        params = dict(self.model_runner.model.named_parameters())
        model = self.model_runner.model

        for layer_idx in sorted(sharded_data.keys()):
            entry = sharded_data[layer_idx]
            w13_shards = entry["w13"]
            w2_shards = entry["w2"]
            ep_degree = int(entry["ep_degree"])
            n_local_trainer = int(entry["n_local_trainer"])

            if len(w13_shards) != ep_degree or len(w2_shards) != ep_degree:
                raise RuntimeError(
                    f"layer {layer_idx}: sharded payload incomplete — "
                    f"w13 has {len(w13_shards)}/{ep_degree} ranks, "
                    f"w2 has {len(w2_shards)}/{ep_degree} ranks"
                )

            w13_key = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
            w2_key = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
            w13_param = params[w13_key]
            w2_param = params[w2_key]

            try:
                moe_layer = model.get_submodule(
                    f"model.layers.{layer_idx}.mlp.experts"
                )
            except AttributeError:
                moe_layer = None

            expert_map = getattr(moe_layer, "expert_map", None)
            moe_tp_rank = getattr(moe_layer, "tp_rank", 0) or 0
            moe_tp_size = getattr(moe_layer, "tp_size", 1) or 1
            local_n_vllm = w13_param.shape[0]
            device = w13_param.device

            # Two valid vLLM layouts:
            #   (a) EP>1: FusedMoE populates expert_map; local_n_vllm < global_n.
            #   (b) TP-only (no --enable-expert-parallel): every worker holds
            #       all experts; expert_map is None and local_n_vllm == global_n.
            # Reject anything else — silent mis-shard would corrupt weights.
            if expert_map is not None:
                em_cpu = expert_map.to("cpu")
                global_n = em_cpu.numel()
            else:
                # Derive global_n from one of the incoming shards instead of
                # the param (which is local_n_vllm and equals global_n in this
                # path). Cross-check below.
                _sample = next(iter(w13_shards.values()))
                global_n = int(_sample.shape[0]) * ep_degree
                if local_n_vllm != global_n:
                    raise RuntimeError(
                        f"layer {layer_idx}: WS10 sharded receive without "
                        f"expert_map requires TP-only vLLM (local_n="
                        f"{local_n_vllm} == global_n={global_n}); got "
                        f"local_n != global_n. vLLM appears EP-sharded but "
                        f"FusedMoE.expert_map is None — refuse to mis-shard."
                    )
                em_cpu = None  # signal identity-perm path below

            # Stage as bf16 global tensors then reuse the existing TP/transpose
            # logic in _load_fused_moe_experts. Shape matches today's payload.
            sample_w13 = next(iter(w13_shards.values()))
            sample_w2 = next(iter(w2_shards.values()))
            w13_global = torch.empty(
                global_n, *sample_w13.shape[1:], dtype=sample_w13.dtype,
            )
            w2_global = torch.empty(
                global_n, *sample_w2.shape[1:], dtype=sample_w2.dtype,
            )
            n_filled_w13 = 0
            n_filled_w2 = 0
            for trainer_rank in w13_shards:
                R = int(trainer_rank)
                shard_w13 = w13_shards[trainer_rank]
                shard_w2 = w2_shards[trainer_rank]
                n_loc = shard_w13.shape[0]
                idx = torch.arange(R, R + n_loc * ep_degree, ep_degree)
                if idx[-1] >= global_n:
                    idx = idx[idx < global_n]
                    shard_w13 = shard_w13[: idx.numel()]
                    shard_w2 = shard_w2[: idx.numel()]
                w13_global.index_copy_(0, idx, shard_w13)
                w2_global.index_copy_(0, idx, shard_w2)
                n_filled_w13 += idx.numel()
                n_filled_w2 += idx.numel()

            if n_filled_w13 != global_n or n_filled_w2 != global_n:
                raise RuntimeError(
                    f"layer {layer_idx}: sharded assembly under-filled — "
                    f"w13 {n_filled_w13}/{global_n} w2 {n_filled_w2}/{global_n}"
                )

            # Slice EP via expert_map — same logic as _load_fused_moe_experts.
            # TP-only vLLM: identity perm (worker holds all global experts).
            if em_cpu is None:
                w13_ep = w13_global
                w2_ep = w2_global
            else:
                owned_global = (em_cpu >= 0).nonzero(as_tuple=True)[0]
                if owned_global.numel() != local_n_vllm:
                    raise RuntimeError(
                        f"layer {layer_idx}: expert_map owned={owned_global.numel()} "
                        f"!= param local={local_n_vllm}"
                    )
                perm = torch.empty(local_n_vllm, dtype=torch.long)
                perm[em_cpu[owned_global].long()] = owned_global
                w13_ep = w13_global.index_select(0, perm)
                w2_ep = w2_global.index_select(0, perm)

            # TP slice along intermediate dim.
            inter = w13_ep.shape[1] // 2
            inter_per_tp = inter // moe_tp_size
            if moe_tp_size == 1:
                w13_tp = w13_ep
                w2_tp = w2_ep
            else:
                gate_shard = w13_ep[
                    :, moe_tp_rank * inter_per_tp:(moe_tp_rank + 1) * inter_per_tp, :
                ]
                up_shard = w13_ep[
                    :,
                    inter + moe_tp_rank * inter_per_tp:inter + (moe_tp_rank + 1) * inter_per_tp,
                    :,
                ]
                w13_tp = torch.cat([gate_shard, up_shard], dim=1)
                w2_tp = w2_ep[
                    :, :, moe_tp_rank * inter_per_tp:(moe_tp_rank + 1) * inter_per_tp
                ]

            is_transposed = w13_param.shape[1] != w13_tp.shape[1]
            if is_transposed:
                w13_param.data.copy_(w13_tp.to(device).transpose(1, 2).contiguous())
                w2_param.data.copy_(w2_tp.to(device).transpose(1, 2).contiguous())
            else:
                w13_param.data.copy_(w13_tp.to(device))
                w2_param.data.copy_(w2_tp.to(device))

    def _ws10_parallel_recv(
        self, tensors_meta: list, batch_max_numel: int, sharded_cross_pgs: dict
    ) -> tuple:
        """Receive WS10 expert shards from all EP ranks concurrently.

        Each trainer EP rank broadcasts on its own gloo PG.  The serial
        receive loop processes one rank at a time (36s total for EP=16).
        Here we launch one thread per rank so all 16 receives overlap,
        cutting network time from ~36s to max(per-rank_recv) ≈ ~5s.

        Non-expert entries (trainer_ep_rank=None) are sent by trainer rank 0
        on R=0's PG, after that rank's expert batches.  R=0's thread handles
        both expert and non-expert entries in manifest order.

        Args:
            tensors_meta: full manifest entry list (same as serial path)
            batch_max_numel: greedy batch size cap (elements, same on both sides)
            sharded_cross_pgs: {R: gloo_pg} from self._xccl_sharded_cross_pgs

        Returns:
            (all_shards, nonexpert_weights, recv_times, t_parallel)
            - all_shards: {layer_idx: {"w13": {R: cpu_tensor}, "w2": {R: cpu_tensor},
                                        "ep_degree": int, "n_local_trainer": int}}
            - nonexpert_weights: [(param_name, cpu_tensor), ...]
            - recv_times: {R: elapsed_s}  (per-thread)
            - t_parallel: wall time from first thread start to last thread join
        """
        import threading
        import torch

        _fused_re = re.compile(
            r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
        )

        # Group manifest entries by effective EP rank.
        # Non-expert entries (no trainer_ep_rank) are sent by trainer rank 0
        # on PG[0] after expert batches — fold them into R=0's group.
        rank_groups: dict = {}
        for entry in tensors_meta:
            R = entry.get("trainer_ep_rank")
            eff_R = int(R) if R is not None else 0
            rank_groups.setdefault(eff_R, []).append(entry)

        all_shards: dict = {}       # {layer_idx: {"w13": {R: t}, "w2": {R: t}, ...}}
        all_shards_lock = threading.Lock()
        nonexpert_weights: list = []  # [(name, tensor)] — only R=0 thread writes
        recv_times: dict = {}
        errors: dict = {}

        def recv_rank_thread(ep_rank: int, entries: list, pg) -> None:
            t0 = time.perf_counter()
            local_buf = None
            try:
                i = 0
                while i < len(entries):
                    # Greedy batch: same algorithm as sender (_ws10_build_local_payload)
                    batch_numel = 0
                    batch_start = i
                    while i < len(entries):
                        pn = entries[i]["numel"]
                        if batch_numel > 0 and batch_numel + pn > batch_max_numel:
                            break
                        batch_numel += pn
                        i += 1

                    if local_buf is None or local_buf.numel() < batch_numel:
                        local_buf = torch.empty(batch_numel, dtype=torch.bfloat16)
                    cpu_recv = local_buf[:batch_numel]
                    pg.broadcast(cpu_recv, root=0).wait()

                    offset = 0
                    for entry in entries[batch_start:i]:
                        n = entry["numel"]
                        tensor = cpu_recv[offset:offset + n].reshape(entry["shape"]).clone()
                        offset += n
                        m = _fused_re.match(entry["name"])
                        if m:
                            layer_idx = int(m.group(1))
                            kind = m.group(2)
                            ep_degree = int(entry.get("ep_degree", 1))
                            n_local = int(entry.get("n_local", entry["shape"][0]))
                            with all_shards_lock:
                                lyr = all_shards.setdefault(layer_idx, {
                                    "w13": {}, "w2": {},
                                    "ep_degree": ep_degree,
                                    "n_local_trainer": n_local,
                                })
                                lyr[kind][ep_rank] = tensor
                        else:
                            # Non-expert entry — only R=0 thread reaches here
                            nonexpert_weights.append((entry["name"], tensor))
            except Exception as e:
                errors[ep_rank] = e
            recv_times[ep_rank] = time.perf_counter() - t0

        threads = []
        t_launch = time.perf_counter()
        for ep_rank, entries in sorted(rank_groups.items()):
            pg = sharded_cross_pgs[ep_rank]
            t = threading.Thread(
                target=recv_rank_thread,
                args=(ep_rank, entries, pg),
                daemon=True,
                name=f"ws10-recv-R{ep_rank}",
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t_parallel = time.perf_counter() - t_launch

        if errors:
            raise RuntimeError(f"WS10 parallel recv errors: {errors}")

        return all_shards, nonexpert_weights, recv_times, t_parallel

    # ------------------------------------------------------------------
    # XCCL broadcast weight sync
    # ------------------------------------------------------------------

    def init_xccl_communicator(
        self, host: str, port: int, world_size: int, base_rank: int,
        use_two_hop: bool = False, wsync_method: str = "xccl_sendrecv",
        pool_size: int = 0,
        topology: str = "replica_fanout", intra_world: int = 0,
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
                # Intra PG sizing depends on topology:
                #   replica_fanout (legacy): intra PG is per-replica TP only.
                #     Each replica's intra PG has size=tp_size_local; cross is
                #     per-replica (1 cross PG per replica on training side).
                #   node_fanout (real 2-hop): all vLLM workers across the node
                #     share one intra PG of size num_replicas*tp_size. Replica 0
                #     (rank 0 in intra) bridges from the single cross PG to all
                #     other vLLM workers via XeLink/XCCL — drops cross-NIC
                #     traffic from N× to 1×.
                if topology == "node_fanout":
                    if intra_world <= 0:
                        raise ValueError(
                            "node_fanout requires intra_world > 0 from training side")
                    intra_rank = replica_idx * tp_size_local + tp_rank
                    intra_size = intra_world
                    is_cross_root = (replica_idx == 0 and tp_rank == 0)
                    cross_prefix_name = "wsync_cross_0"
                else:
                    intra_rank = tp_rank
                    intra_size = tp_size_local
                    is_cross_root = (tp_rank == 0)
                    cross_prefix_name = f"wsync_cross_{replica_idx}"
                self._wsync_topology = topology
                self._wsync_cross_method = wsync_method
                intra_method = os.environ.get("WSYNC_INTRA_METHOD", "xccl")
                self._wsync_intra_method = intra_method

                if is_cross_root:
                    if pool_size > 0:
                        # Dynamic sender pool (legacy replica_fanout only)
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
                        cross_prefixed = c10d.PrefixStore(cross_prefix_name, store)
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
                        "init_xccl_communicator: cross PG ready (topology=%s, "
                        "rank 1/2, method=%s, pool=%d, intra root)",
                        topology, wsync_method, pool_size)

                # node_fanout: single intra PG shared across all replicas (one prefix
                # for the whole node). replica_fanout: per-replica intra PG.
                intra_prefix_name = (
                    "wsync_intra_node" if topology == "node_fanout"
                    else f"wsync_intra_{replica_idx}"
                )
                intra_prefixed = c10d.PrefixStore(intra_prefix_name, store)
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
                    "init_xccl_communicator: intra PG ready (topology=%s, replica=%d, "
                    "rank=%d/%d, method=%s, is_root=%s)",
                    topology, replica_idx, intra_rank, intra_size, intra_method,
                    self._is_intra_root,
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

    def init_xccl_sharded_pgs(
        self, host: str, port: int, world_size: int,
        ep_degree: int, method: str = "gloo",
    ) -> dict:
        """WS10 Commit B: construct ep_degree extra cross PGs on EVERY vLLM
        replica root.

        Each new PG is a 2-rank group ``[trainer_ep_rank_R, vllm_root_repK]``
        with prefix ``wsync_sharded_R{R}_rep{K}``. Each replica's intra-PG
        root (tp_rank=0 of that replica) joins as rank 1 of its own per-replica
        PG. Other TP workers in each replica no-op — the existing intra PG
        fanout (driven inside ``receive_weights_xccl_streaming``) carries the
        data from that replica's root to its TP workers.

        This makes WS10 multi-replica safe: every replica root receives the
        per-EP-rank shard from the trainer over its own dedicated 2-rank PG.
        """
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d

        try:
            t0 = time.perf_counter()
            tp_rank = dist.get_rank() if dist.is_initialized() else 0
            tp_size_local = dist.get_world_size() if dist.is_initialized() else 1
            my_rank_in_xccl = getattr(self, "_xccl_rank", -1)
            replica_idx = (
                (my_rank_in_xccl - 1) // tp_size_local
                if my_rank_in_xccl >= 1 else -1
            )
            # Each replica's intra-PG root joins as rank 1 of its own per-replica
            # sharded PG. Other TP workers no-op (intra PG fans out from the root).
            is_root = bool(getattr(self, "_is_intra_root", False)) or (
                tp_rank == 0 and replica_idx >= 0
            )
            if not is_root:
                logger.info(
                    "init_xccl_sharded_pgs: noop on non-root vLLM worker "
                    "(tp_rank=%d, replica_idx=%d)",
                    tp_rank, replica_idx,
                )
                return {"status": "ok", "noop": True, "init_s": 0.0,
                        "ep_degree": int(ep_degree),
                        "replica_idx": int(replica_idx)}

            # Reuse the existing TCPStore connection (built by
            # init_xccl_communicator) when present; otherwise open one.
            store = getattr(self, "_xccl_store", None)
            if store is None:
                import datetime as _dt
                store = dist.TCPStore(
                    host_name=host, port=port, world_size=world_size,
                    is_master=False, timeout=_dt.timedelta(seconds=120),
                )
                self._xccl_store = store

            sharded_pgs: dict = {}
            for R in range(int(ep_degree)):
                prefix = f"wsync_sharded_R{R}_rep{int(replica_idx)}"
                prefixed = c10d.PrefixStore(prefix, store)
                if method == "gloo":
                    pg = c10d.ProcessGroupGloo(
                        store=prefixed, rank=1, size=2,
                    )
                else:
                    opts = c10d.ProcessGroupXCCL.Options()
                    pg = c10d.ProcessGroupXCCL(
                        store=prefixed, rank=1, size=2, options=opts,
                    )
                sharded_pgs[R] = pg
                logger.info(
                    "init_xccl_sharded_pgs: PG %d/%d created (prefix=%s, method=%s, replica=%d)",
                    R, int(ep_degree), prefix, method, replica_idx,
                )
            self._xccl_sharded_cross_pgs = sharded_pgs
            self._xccl_sharded_method = method
            dt = time.perf_counter() - t0
            logger.info(
                "init_xccl_sharded_pgs: %d PGs ready in %.1fs (method=%s, replica=%d)",
                len(sharded_pgs), dt, method, replica_idx,
            )
            return {"status": "ok", "init_s": round(dt, 2),
                    "ep_degree": int(ep_degree), "method": method,
                    "replica_idx": int(replica_idx)}
        except Exception as e:
            logger.exception("init_xccl_sharded_pgs failed")
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

                # Stash incomplete MoE w13/w2 pairs across batches: the trainer's
                # greedy batching can split a layer's w13_weight and w2_weight
                # into different broadcast batches. _load_fused_moe_experts
                # requires both keys, so we hold the lonely tensor (cloned, since
                # _xccl_recv_buf is reused next iter) until its pair arrives.
                fused_pending: dict = {}
                # WS10 sharded MoE: same pattern, but each trainer rank's shard
                # arrives as its own entry tagged with trainer_ep_rank/ep_degree.
                # We accumulate all (R, kind) pairs for each layer and dispatch
                # to _load_fused_moe_experts_sharded once both kinds have all
                # ep_degree shards present. Off-path (no tags) -> empty dict.
                sharded_pending: dict = {}
                _recv_load_error = None  # deferred: recv all batches before raising

                # Compute expected batch count (same greedy algorithm as sender)
                # for diagnostic comparison with trainer's logged n_batches.
                # WS10: must also break the batch at every trainer_ep_rank
                # transition (and at the expert -> non-expert transition where
                # the tag disappears) so the receiver's batches line up with
                # the sender's per-rank greedy. Each trainer rank only emits
                # its OWN local entries; the receiver was previously packing
                # across rank boundaries and got a size mismatch (gloo:
                # "preamble.length <= nbytes" terminate, run #6).
                _diag_batches = 0
                _diag_i = 0
                _diag_bn = 0
                while _diag_i < n_params:
                    _diag_inner_bn = 0
                    _diag_R = tensors_meta[_diag_i].get("trainer_ep_rank")
                    while _diag_i < n_params:
                        _pn = tensors_meta[_diag_i]["numel"]
                        _R_now = tensors_meta[_diag_i].get("trainer_ep_rank")
                        if _diag_inner_bn > 0 and _R_now != _diag_R:
                            break
                        if _diag_inner_bn > 0 and _diag_inner_bn + _pn > batch_max_numel:
                            break
                        _diag_inner_bn += _pn
                        _diag_i += 1
                    _diag_batches += 1
                logger.info(
                    "WSYNC-RECV: n_params=%d expected_batches=%d batch_max_numel=%d",
                    n_params, _diag_batches, batch_max_numel,
                )
                # Log first 6 tensors_meta names+numel for ordering diagnosis
                _diag_head = [
                    f"{e['name']} ({e['numel']})" for e in tensors_meta[:6]
                ]
                logger.info("WSYNC-RECV first 6 entries: %s", " | ".join(_diag_head))

                _batch_idx = 0
                i = 0
                # WS10: when batch entries carry trainer_ep_rank tags, the
                # cross PG to receive on switches per batch. The two-hop
                # intra fanout stays the same (vLLM TP workers still need
                # the data). Built-in receiver caches are unchanged on the
                # off-path (no tag → use legacy self._xccl_cross_pg).
                _ws10_sharded_cross_pgs = getattr(
                    self, "_xccl_sharded_cross_pgs", None) or {}

                # WS10 parallel fast path: receive all EP-rank shards concurrently.
                # The serial loop processes ranks one at a time (~2.3s × 16 = 36s).
                # With parallel threads each rank's recv overlaps → ~5s total.
                # Only activates when WS10 sharded PGs are present.
                if _ws10_sharded_cross_pgs:
                    _p_shards, _p_ne, _p_rtimes, _p_wall = self._ws10_parallel_recv(
                        tensors_meta, batch_max_numel, _ws10_sharded_cross_pgs,
                    )
                    t_bcast_total = _p_wall
                    _t_load0 = time.perf_counter()
                    if _p_ne:
                        self.model_runner.model.load_weights(weights=_p_ne)
                    if _p_shards:
                        _p_ready = {
                            li: lyr for li, lyr in _p_shards.items()
                            if (len(lyr["w13"]) == lyr["ep_degree"]
                                and len(lyr["w2"]) == lyr["ep_degree"])
                        }
                        if len(_p_ready) != len(_p_shards):
                            _incomplete = {
                                li: {
                                    "w13_ranks": sorted(_p_shards[li]["w13"].keys()),
                                    "w2_ranks": sorted(_p_shards[li]["w2"].keys()),
                                }
                                for li in _p_shards if li not in _p_ready
                            }
                            raise RuntimeError(
                                f"WS10 parallel recv: {len(_p_shards) - len(_p_ready)} "
                                f"incomplete layers: {_incomplete}"
                            )
                        self._load_fused_moe_experts_sharded(_p_ready)
                    t_load_total = time.perf_counter() - _t_load0
                    logger.info(
                        "WS10 parallel recv done: %d expert layers + %d non-expert "
                        "recv_wall=%.1fs load=%.1fs per_rank=%s",
                        len(_p_shards), len(_p_ne), _p_wall, t_load_total,
                        {R: f"{t:.2f}s" for R, t in sorted(_p_rtimes.items())},
                    )
                    torch.xpu.synchronize(self._xccl_device)
                    logger.info(
                        "receive_weights_xccl_streaming: %d params %.2f GiB in %.1fs "
                        "(bcast=%.1fs %.1f GB/s, load=%.1fs)",
                        n_params, gb, time.perf_counter() - t0,
                        t_bcast_total,
                        gb / t_bcast_total if t_bcast_total > 0 else 0,
                        t_load_total,
                    )
                    return {
                        "status": "ok", "num_params": n_params,
                        "bcast_s": round(t_bcast_total, 2),
                        "load_s": round(t_load_total, 2),
                    }

                while i < n_params:
                    batch_start = i
                    batch_numel = 0
                    # WS10: each batch must come from a single trainer EP rank
                    # so the receiver's batch boundaries match the sender's
                    # per-rank greedy. The expert -> non-expert transition
                    # (tag disappears, but rank 0 ships them on R=0's PG)
                    # also forces a boundary because the sender concatenated
                    # them into separate cpu_batches (see _ws10_build_local_payload
                    # + the rank-0 non-expert tail packing).
                    _ws10_R_run = tensors_meta[i].get("trainer_ep_rank")
                    while i < n_params:
                        pn = tensors_meta[i]["numel"]
                        _R_now = tensors_meta[i].get("trainer_ep_rank")
                        if batch_numel > 0 and _R_now != _ws10_R_run:
                            break
                        if batch_numel > 0 and batch_numel + pn > batch_max_numel:
                            break
                        batch_numel += pn
                        i += 1

                    # WS10 per-batch PG selection: peek the FIRST entry of
                    # this batch. If it carries trainer_ep_rank, use the
                    # corresponding sharded PG. Otherwise fall back to the
                    # legacy single cross PG. Non-expert tail (no tag) is
                    # sent by trainer rank 0 → use sharded PG R=0 if
                    # WS10 is active, else legacy.
                    _ws10_first = tensors_meta[batch_start]
                    _ws10_R_first = _ws10_first.get("trainer_ep_rank")
                    _orig_cross_pg = getattr(self, "_xccl_cross_pg", None)
                    _ws10_active_batch = False
                    if _ws10_sharded_cross_pgs:
                        if _ws10_R_first is not None:
                            _R = int(_ws10_R_first)
                            self._xccl_cross_pg = _ws10_sharded_cross_pgs[_R]
                            _ws10_active_batch = True
                        elif 0 in _ws10_sharded_cross_pgs:
                            # Non-expert batch from trainer rank 0 in WS10
                            # mode: route via R=0's sharded PG.
                            self._xccl_cross_pg = _ws10_sharded_cross_pgs[0]
                            _ws10_active_batch = True
                    try:
                        recv_buf = self._xccl_recv_buf[:batch_numel]
                        t_b0 = time.perf_counter()
                        if two_hop:
                            intra_method = getattr(self, '_wsync_intra_method', 'xccl')
                            if self._is_intra_root:
                                # WS10 sharded PGs are gloo-only by default; force the
                                # gloo cross-recv path when they're active for this batch.
                                cross_method = (
                                    "gloo" if _ws10_active_batch
                                    else getattr(self, '_wsync_cross_method', 'gloo')
                                )
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
                        t_bcast_i = time.perf_counter() - t_b0
                        t_bcast_total += t_bcast_i
                    finally:
                        # Restore the legacy cross PG selection for the
                        # next batch (off-path code references it).
                        self._xccl_cross_pg = _orig_cross_pg

                    # Split flat buffer back into per-param tensors, routing
                    # fused MoE experts to _load_fused_moe_experts (GPU-direct)
                    offset = 0
                    non_expert_weights = []
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
                            trainer_ep_rank = entry.get("trainer_ep_rank")
                            ep_degree = entry.get("ep_degree")
                            if trainer_ep_rank is not None and ep_degree is not None:
                                # WS10 sharded payload — accumulate per-rank shards.
                                # Stage on CPU: with rank-grouped manifest order
                                # ((R, layer, kind, name)) no layer becomes
                                # complete until R=ep_d-1's batches arrive at
                                # the very end. With ep_d=16, 48 layers, 2 kinds,
                                # the pile-up is 48×2×16=1536 tensors (~57 GiB
                                # for 30B-A3B) — does not fit on a 64 GiB tile
                                # on top of the resident vLLM model (run #8 OOM
                                # at batch 9/68). _load_fused_moe_experts_sharded
                                # builds w13_global/w2_global on CPU anyway and
                                # only the final TP slice goes to device, so
                                # CPU staging is bit-equivalent.
                                lyr = sharded_pending.setdefault(layer_idx, {
                                    "w13": {}, "w2": {},
                                    "ep_degree": int(ep_degree),
                                    "n_local_trainer": int(entry.get("n_local", entry["shape"][0])),
                                })
                                lyr[kind][int(trainer_ep_rank)] = tensor.detach().cpu()
                            else:
                                # Clone: recv_buf is reused next iteration.
                                fused_pending.setdefault(layer_idx, {})[kind] = tensor.clone()
                        else:
                            non_expert_weights.append((entry["name"], tensor))
                        offset += n

                    # Apply layers whose w13+w2 are both present.
                    fused_ready = {
                        li: kv for li, kv in fused_pending.items()
                        if "w13" in kv and "w2" in kv
                    }
                    for li in fused_ready:
                        del fused_pending[li]
                    # WS10: apply layers whose w13+w2 each have all ep_degree shards.
                    sharded_ready = {}
                    for li, lyr in list(sharded_pending.items()):
                        ep_d = lyr["ep_degree"]
                        if (len(lyr["w13"]) == ep_d and len(lyr["w2"]) == ep_d):
                            sharded_ready[li] = lyr
                            del sharded_pending[li]

                    t_l0 = time.perf_counter()
                    # Catch load errors but continue receiving remaining batches
                    # so the trainer's send loop is not left hanging. The error
                    # is raised after all batches are consumed.
                    try:
                        if non_expert_weights:
                            self.model_runner.model.load_weights(weights=non_expert_weights)
                        if fused_ready:
                            self._load_fused_moe_experts(fused_ready)
                        if sharded_ready:
                            self._load_fused_moe_experts_sharded(sharded_ready)
                    except Exception as _le:
                        if _recv_load_error is None:
                            logger.error(
                                "receive_weights_xccl_streaming: load error (continuing recv "
                                "to unblock trainer): %s", _le)
                            _recv_load_error = _le
                    t_load_i = time.perf_counter() - t_l0
                    t_load_total += t_load_i
                    # Log first batch and every 10th to diagnose recv vs load breakdown
                    if _batch_idx == 0 or _batch_idx % 10 == 9:
                        _gb_i = batch_numel * 2 / 1024**3
                        logger.info(
                            "WSYNC-RECV batch %d/%d: numel=%d (%.2f GiB) recv=%.3fs load=%.3fs "
                            "n_expert_ready=%d n_non_expert=%d",
                            _batch_idx, _diag_batches, batch_numel, _gb_i,
                            t_bcast_i, t_load_i, len(fused_ready), len(non_expert_weights),
                        )
                    _batch_idx += 1
                    del non_expert_weights, fused_ready, sharded_ready

                logger.info(
                    "WSYNC-RECV done: %d batches (expected %d) total_recv=%.1fs total_load=%.1fs",
                    _batch_idx, _diag_batches, t_bcast_total, t_load_total,
                )

                # Final flush: any layers still pending after all batches must
                # have a missing pair — fail loudly rather than silently skip.
                if fused_pending:
                    incomplete = {li: list(kv.keys()) for li, kv in fused_pending.items()}
                    raise RuntimeError(
                        f"XCCL streaming MoE: {len(fused_pending)} layers missing w13 or w2 "
                        f"after all batches: {incomplete}"
                    )
                if sharded_pending:
                    incomplete = {
                        li: {"w13_ranks": sorted(lyr["w13"].keys()),
                             "w2_ranks": sorted(lyr["w2"].keys()),
                             "ep_degree": lyr["ep_degree"]}
                        for li, lyr in sharded_pending.items()
                    }
                    raise RuntimeError(
                        f"XCCL streaming MoE (WS10 sharded): {len(sharded_pending)} layers "
                        f"missing per-rank shards after all batches: {incomplete}"
                    )
                # Deferred load error: all batches received (trainer unblocked), now raise.
                if _recv_load_error is not None:
                    raise _recv_load_error

            else:
                # Legacy: one broadcast per param. Same w13/w2 split-across-flush
                # hazard as the batched path, so accumulate across apply_every flushes.
                fused_leg_pending: dict = {}
                sharded_leg_pending: dict = {}
                weights_batch = []
                sharded_batch = []  # (layer_idx, kind, R, ep_d, n_local, tensor)
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

                    weights_batch.append((entry, recv_buf.reshape(entry["shape"])))

                    if len(weights_batch) >= apply_every or idx == n_params - 1:
                        t_l0 = time.perf_counter()
                        # Route fused MoE experts to GPU-direct path
                        _fused_re_leg = re.compile(
                            r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
                        )
                        non_expert = []
                        for wentry, wtensor in weights_batch:
                            wname = wentry["name"]
                            m = _fused_re_leg.match(wname)
                            if m:
                                li = int(m.group(1))
                                kind = m.group(2)
                                R = wentry.get("trainer_ep_rank")
                                ep_d = wentry.get("ep_degree")
                                if R is not None and ep_d is not None:
                                    lyr = sharded_leg_pending.setdefault(li, {
                                        "w13": {}, "w2": {},
                                        "ep_degree": int(ep_d),
                                        "n_local_trainer": int(wentry.get("n_local", wentry["shape"][0])),
                                    })
                                    lyr[kind][int(R)] = wtensor
                                else:
                                    # Legacy path uses fresh recv_buf per param, so
                                    # no clone needed.
                                    fused_leg_pending.setdefault(li, {})[kind] = wtensor
                            else:
                                non_expert.append((wname, wtensor))
                        fused_ready = {
                            li: kv for li, kv in fused_leg_pending.items()
                            if "w13" in kv and "w2" in kv
                        }
                        for li in fused_ready:
                            del fused_leg_pending[li]
                        sharded_ready_leg = {}
                        for li, lyr in list(sharded_leg_pending.items()):
                            ep_d = lyr["ep_degree"]
                            if (len(lyr["w13"]) == ep_d and len(lyr["w2"]) == ep_d):
                                sharded_ready_leg[li] = lyr
                                del sharded_leg_pending[li]
                        if non_expert:
                            self.model_runner.model.load_weights(weights=non_expert)
                        if fused_ready:
                            self._load_fused_moe_experts(fused_ready)
                        if sharded_ready_leg:
                            self._load_fused_moe_experts_sharded(sharded_ready_leg)
                        t_load_total += time.perf_counter() - t_l0
                        del weights_batch, non_expert, fused_ready, sharded_ready_leg
                        weights_batch = []

                if fused_leg_pending:
                    incomplete = {li: list(kv.keys()) for li, kv in fused_leg_pending.items()}
                    raise RuntimeError(
                        f"XCCL streaming MoE (legacy): {len(fused_leg_pending)} layers "
                        f"missing w13 or w2 after all params: {incomplete}"
                    )

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Weight synchronization methods for GRPO training on Aurora/XPU.
#
# Extracted from grpo_full_finetune_distributed_xpu.py to reduce recipe file size.
# All functions take `self` as first argument and are bound to the recipe class via:
#   _sync_colocated_weights = _sync_colocated_weights  (in GRPOFullFinetuneDistributedXPU body)
#
# Three sync paths:
#   1. colocate / colocate_sleep: FSDP2 full_tensor() → vLLM load_weights() directly
#   2. server (raw_bytes): FSDP2 full_tensor() → HTTP POST to vLLM server
#   3. server (xccl): XCCL broadcast GPU→GPU across node boundary, streamed

import io
import os
import re
import struct
import threading
import time

import requests
import torch
from torchtune import training, utils

log = utils.get_logger("DEBUG")


class _NullCtx:
    """No-op context manager for FSDP2 path where summon_full_params doesn't apply."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


def _backbone_param_iter(self, policy=None):
    """Generic fallback for backbone weight iteration.

    BioReason-style multimodal wrappers expose `vllm_param_iter()` directly.
    For plain text-only models (Qwen, Gemma, etc.), fall back to
    `model.named_parameters()` with the tune→HF name map applied.
    """
    if policy is None:
        policy = self._model
    if hasattr(policy, 'vllm_param_iter'):
        yield from policy.vllm_param_iter()
        return
    tune_to_hf = getattr(self, '_tune_to_hf_map', {}) or {}
    for tune_name, param in policy.named_parameters():
        clean = tune_name.replace(
            "_checkpoint_wrapped_module.", ""
        ).replace("_fsdp_wrapped_module.", "")
        hf_name = tune_to_hf.get(clean, tune_to_hf.get(tune_name, clean))
        yield hf_name, param


# Llama-family checkpointers (LLAMA2/LLAMA3 / default fallback) route through
# `torchtune.models.convert_weights.hf_to_tune`, which **permutes** Q/K weights
# (interleaves the rotary frequency halves) so torchtune's attention forward
# can consume them in its native layout. Qwen2/3, Gemma2/4, Phi3, etc. all
# load without this permutation — their attention forwards consume HF layout
# directly. See `torchtune/training/checkpointing/_checkpointer.py:736-743`.
#
# At weight-sync time, vLLM expects **HF-format unpermuted** Q/K. If we send
# torchtune-permuted Q/K (via `param.full_tensor()`) without inverting the
# permutation, vLLM's Q/K projections become scrambled and generation
# degenerates after the first sync (the model emits pretraining-corpus
# continuations instead of conditioning on the prompt). This was the silent
# cause of the AuroraGPT-2B GRPO bake-off accuracy gap (RESULTS.md
# 2026-06-11): torchtune-XPU rewards stayed flat ~0.20 over 50 steps while
# ezpz climbed 0.08→0.65 on identical prompts. Empirically reproduces
# whenever a LLAMA3 model_type is used with vLLM colocate / dedicated_rank /
# server modes — Qwen/Gemma paths were unaffected and that's why this hid.
_LLAMA_PERMUTING_MODEL_TYPES = frozenset()  # populated lazily from ModelType


def _needs_qk_unpermute(self) -> bool:
    """True iff this run's checkpointer applies the Llama-family Q/K permutation.

    Triggered by LLAMA2 / LLAMA3 / LLAMA3_VISION model_types plus the default
    `hf_to_tune` fall-through (used for unknown HF llama-architecture models).
    Falls back to False for any model_type whose hf_to_tune does NOT permute
    (Qwen2/3, Gemma2/4, Phi3, etc.).
    """
    global _LLAMA_PERMUTING_MODEL_TYPES
    if not _LLAMA_PERMUTING_MODEL_TYPES:
        from torchtune.training.checkpointing._utils import ModelType
        # The set of model_types whose hf_to_tune is `convert_weights.hf_to_tune`
        # (or a delegating wrapper) — these all permute Q/K. Determined by
        # reading `_checkpointer.py:609-743`: everything that does NOT have an
        # explicit elif branch falls through to the default `hf_to_tune`. Of
        # the branches that DO exist, llama3_vision_hf_to_tune also permutes
        # (see llama3_2_vision/_convert_weights.py:269).
        _LLAMA_PERMUTING_MODEL_TYPES = frozenset({
            ModelType.LLAMA2,
            ModelType.LLAMA3,
            ModelType.LLAMA3_2,
            ModelType.LLAMA3_VISION,
        })
    ckpt = getattr(self, '_checkpointer', None)
    if ckpt is None:
        return False
    mt = getattr(ckpt, '_model_type', None)
    if mt is None:
        return False
    return mt in _LLAMA_PERMUTING_MODEL_TYPES


def _qk_unpermute_for_vllm(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Invert `convert_weights.hf_to_tune`'s Q/K interleave.

    hf_to_tune does: `view(n_heads, 2, head_dim // 2, dim).transpose(1, 2).reshape(...)`
    The inverse (matches `convert_weights.tune_to_hf`): `view(n_heads,
    head_dim // 2, 2, dim).transpose(1, 2).reshape(...)`. We return a
    contiguous copy so vLLM's `load_weights` can index it without surprises.
    """
    dim = weight.shape[1]
    return (
        weight.view(n_heads, head_dim // 2, 2, dim)
        .transpose(1, 2)
        .reshape(n_heads * head_dim, dim)
        .contiguous()
    )


def _maybe_unpermute_qk(self, hf_name: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply the Q/K un-permute if this run uses a Llama-family checkpointer.

    Detection is by HF name suffix (`q_proj.weight` / `k_proj.weight`) so it
    works regardless of whether the param iterator yields tune-format or
    HF-format names. Returns the input unchanged when no un-permute is needed.
    """
    if not _needs_qk_unpermute(self):
        return weight
    is_q = hf_name.endswith("q_proj.weight")
    is_k = hf_name.endswith("k_proj.weight")
    if not (is_q or is_k):
        return weight
    # Pull head counts from the recipe's cached model dims. These are populated
    # in setup() from cfg.model.{num_heads,num_kv_heads,embed_dim}.
    n_heads = getattr(self, '_model_num_heads', None)
    n_kv_heads = getattr(self, '_model_num_kv_heads', None)
    head_dim = getattr(self, '_model_head_dim', None)
    if n_heads is None or n_kv_heads is None or head_dim is None:
        # Fall back to introspecting the model config the FIRST time. Don't
        # crash the sync — better to send slightly wrong weights once and
        # warn loudly than to kill the run.
        log.warning(
            "QK un-permute requested but recipe missing _model_num_heads / "
            "_model_num_kv_heads / _model_head_dim. Sending raw weights for "
            "%s — generation will degrade.", hf_name,
        )
        return weight
    return _qk_unpermute_for_vllm(
        weight, n_heads if is_q else n_kv_heads, head_dim,
    )


def _sync_colocated_weights(self) -> None:
    """Sync FSDP2 weights to colocated vLLM via load_weights().

    For each param: full_tensor() → load_weights() → del. One param
    at a time to minimize peak GPU memory (critical for 32B where
    optimizer states + vLLM weights leave <1 GiB free).

    vLLM's load_weights() handles TP slicing and QKV/gate_up merging
    internally via per-param weight_loader functions.
    """
    import gc

    t0 = time.perf_counter()
    # Phase A asym-optim: spare ranks still join the FSDP full_tensor() collective
    # (12-rank dp_mesh) but have no vLLM engine to load into.
    _vllm_ranks = getattr(self, "_vllm_ranks", None)
    _is_vllm_rank = (_vllm_ranks is None) or (self.rank in _vllm_ranks)
    llm_model = (
        self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model
        if _is_vllm_rank
        else None
    )

    n_synced = 0

    # FSDP1 vs FSDP2 full-param materialization:
    #   - FSDP2 (default colocate, dp_replicate=1): params are DTensors with a
    #     .full_tensor() collective — handled per-param in the loop below.
    #   - FSDP1 HYBRID_SHARD (colocate + dp_replicate>1): params are flat shards
    #     with NO .full_tensor(); `param.data` is only this rank's 1/dp_shard
    #     slice. Loading that into vLLM asserts on shape (vocab_parallel_embedding
    #     weight_loader, 2026-06-18). Wrap the whole iteration in
    #     summon_full_params(rank0_only=False) so EVERY rank sees full params
    #     (each colocated rank loads its own vLLM). writeback=False: read-only.
    _use_fsdp1 = bool(getattr(self, "_use_fsdp1", False))
    if _use_fsdp1 and not hasattr(self._policy, 'vllm_param_iter'):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        _summon_ctx = FSDP.summon_full_params(
            self._model, writeback=False, rank0_only=False
        )
    else:
        _summon_ctx = _NullCtx()

    # BioReasonModel (and other multimodal wrappers) expose vllm_param_iter()
    # which yields (hf_name, param) for backbone-only params, bypassing the
    # _tune_to_hf_map name translation and skipping frozen encoders/projectors.
    if hasattr(self._policy, 'vllm_param_iter'):
        param_iter_factory = self._policy.vllm_param_iter
    else:
        def param_iter_factory():
            for tune_name, param in self._model.named_parameters():
                clean = tune_name.replace(
                    "_checkpoint_wrapped_module.", ""
                ).replace("_fsdp_wrapped_module.", "")
                hf_name = self._tune_to_hf_map.get(
                    clean, self._tune_to_hf_map.get(tune_name, clean)
                )
                yield hf_name, param

    _unperm_needed = _needs_qk_unpermute(self)
    if _unperm_needed and not getattr(self, "_qk_unperm_engaged_logged", False):
        log.info(
            "Rank %d: colocated wsync QK un-permute ENGAGED "
            "(n_heads=%s n_kv_heads=%s head_dim=%s)",
            self.rank,
            getattr(self, "_model_num_heads", None),
            getattr(self, "_model_num_kv_heads", None),
            getattr(self, "_model_head_dim", None),
        )
        self._qk_unperm_engaged_logged = True
    with _summon_ctx:
        for hf_name, param in param_iter_factory():
            if hasattr(param, 'full_tensor'):
                weight_data = param.full_tensor()      # FSDP2 DTensor
            else:
                # FSDP2 non-DTensor OR FSDP1-under-summon (already full).
                weight_data = param.data

            if _is_vllm_rank:
                if _unperm_needed:
                    # Llama-family Q/K were permuted at checkpoint load; un-permute
                    # back to HF layout so vLLM consumes them correctly.
                    weight_data = _maybe_unpermute_qk(self, hf_name, weight_data)
                llm_model.load_weights([(hf_name, weight_data)])
            n_synced += 1
            del weight_data

            # Sync + gc every 5 params to bound UR handle pressure
            # (707 full_tensor all-gathers create UR handles that must be
            # reclaimed before the subsequent FSDP backward pass).
            if n_synced % 5 == 0 and torch.xpu.is_available():
                gc.collect()
                torch.xpu.synchronize(self._device)

    if _is_vllm_rank:
        self._vllm_llm.llm_engine.reset_prefix_cache()

    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.synchronize(self._device)

    log.info(
        "Rank %d: weight sync: %d params in %.1fs",
        self.rank, n_synced, time.perf_counter() - t0,
    )

_RAY_WSYNC_CHUNK_NUMEL = int(os.environ.get(
    "RAY_WSYNC_CHUNK_NUMEL", str(500 * 1024 * 1024)))  # 1 GiB bf16


def _sync_ray_colocate_weights(self) -> None:
    """Sync FSDP2 weights to Ray-colocate vLLM via chunked collective_rpc.

    Gathers FSDP params via ``full_tensor()`` (collective on all trainer
    ranks), accumulates them into chunks of ~RAY_WSYNC_CHUNK_NUMEL elements,
    and sends each chunk in a single ``collective_rpc`` call. This cuts
    399 serial RPCs → ~16-20 chunked RPCs, dropping wsync from ~87s to ~15-20s.

    On rank 0, gathered tensors are moved to CPU immediately so the
    trainer's HBM footprint stays flat. vLLM's ``load_weights`` processes
    the full chunk list in one call.
    """
    import gc
    import pickle as _pickle

    t0 = time.perf_counter()
    is_driver = (self.rank == 0)

    if hasattr(self._policy, 'vllm_param_iter'):
        param_iter = self._policy.vllm_param_iter()
    else:
        def _default_iter():
            for tune_name, param in self._model.named_parameters():
                clean = tune_name.replace(
            "_checkpoint_wrapped_module.", ""
        ).replace("_fsdp_wrapped_module.", "")
                hf_name = self._tune_to_hf_map.get(
                    clean, self._tune_to_hf_map.get(tune_name, clean)
                )
                yield hf_name, param
        param_iter = _default_iter()

    def _ray_load_weights(worker, blob):
        weights = _pickle.loads(blob)
        worker.model_runner.model.load_weights(weights)

    n_synced = 0
    n_chunks = 0
    chunk_pairs = []  # list of (hf_name, cpu_tensor) for rank 0
    chunk_numel = 0
    _unperm_needed = _needs_qk_unpermute(self)
    if (
        _unperm_needed
        and is_driver
        and not getattr(self, "_qk_unperm_ray_engaged_logged", False)
    ):
        log.info(
            "Rank %d: ray-colocate wsync QK un-permute ENGAGED "
            "(n_heads=%s n_kv_heads=%s head_dim=%s)",
            self.rank,
            getattr(self, "_model_num_heads", None),
            getattr(self, "_model_num_kv_heads", None),
            getattr(self, "_model_head_dim", None),
        )
        self._qk_unperm_ray_engaged_logged = True

    for hf_name, param in param_iter:
        # full_tensor() is collective — every FSDP rank must call.
        if hasattr(param, 'full_tensor'):
            weight_data = param.full_tensor()
        else:
            weight_data = param.data

        if _unperm_needed and is_driver:
            # Llama-family Q/K un-permute happens only on the driver — the
            # workers throw their `weight_data` away unread, so we don't waste
            # cycles permuting on them.
            weight_data = _maybe_unpermute_qk(self, hf_name, weight_data)

        param_numel = weight_data.numel()

        if is_driver:
            cpu_weight = weight_data.detach().to("cpu", non_blocking=False)
            del weight_data
            chunk_pairs.append((hf_name, cpu_weight))
        else:
            del weight_data

        chunk_numel += param_numel
        n_synced += 1

        if chunk_numel >= _RAY_WSYNC_CHUNK_NUMEL:
            if is_driver:
                blob = _pickle.dumps(chunk_pairs, protocol=_pickle.HIGHEST_PROTOCOL)
                self._vllm_llm.llm_engine.collective_rpc(
                    _ray_load_weights, args=(blob,),
                )
                del blob
                chunk_pairs = []
            chunk_numel = 0
            n_chunks += 1
            if torch.xpu.is_available():
                gc.collect()
                torch.xpu.synchronize(self._device)

    # Flush remaining params
    if chunk_numel > 0:
        if is_driver:
            blob = _pickle.dumps(chunk_pairs, protocol=_pickle.HIGHEST_PROTOCOL)
            self._vllm_llm.llm_engine.collective_rpc(
                _ray_load_weights, args=(blob,),
            )
            del blob
            chunk_pairs = []
        n_chunks += 1

    if is_driver:
        self._vllm_llm.llm_engine.reset_prefix_cache()

    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.synchronize(self._device)

    log.info(
        "Rank %d: Ray-colocate weight sync: %d params in %d chunks, %.1fs",
        self.rank, n_synced, n_chunks, time.perf_counter() - t0,
    )


def _compute_wsync_layout(self, policy) -> None:
    """Pre-compute chunked broadcast layout for batched weight sync.

    Groups backbone params into chunks of ≤CHUNK_NUMEL bf16 elements so that:
    - Sender (rank 0): packs one chunk_buf at a time (fits in remaining GPU memory)
    - Receiver (rank 11): allocates one chunk_buf at a time (858 MiB free post-vLLM-init)

    Result: ~38 broadcasts instead of ~400, ~10× speedup with no OOM on vLLM rank.
    CHUNK_NUMEL default = 100M elems = 200 MiB bf16. Override via WSYNC_CHUNK_NUMEL.

    For FSDP2 DTensor params, layout uses the *full* (unsharded) shape/numel
    because the sender will materialize via `full_tensor()` before packing.
    """
    CHUNK_NUMEL = int(os.environ.get("WSYNC_CHUNK_NUMEL", str(100 * 1024 * 1024)))

    # Lazy-build the tune→HF param name map if not already done. Generic recipes
    # may not have invoked _build_tune_to_hf_map yet at first sync.
    if not hasattr(self, '_tune_to_hf_map'):
        if hasattr(self, '_build_tune_to_hf_map'):
            self._build_tune_to_hf_map()
        else:
            self._tune_to_hf_map = {}

    layout = []
    total_numel = 0
    if hasattr(policy, 'vllm_param_iter'):
        _piter = policy.vllm_param_iter()
    else:
        _piter = _backbone_param_iter(self, policy)
    for hf_name, param in _piter:
        # FSDP2 DTensor: shape is local-shard shape; full shape lives in _local_tensor metadata.
        if hasattr(param, 'full_tensor'):
            full_shape = tuple(param.shape)  # DTensor.shape is the GLOBAL shape
            n = 1
            for s in full_shape:
                n *= s
        else:
            full_shape = tuple(param.shape)
            n = param.numel()
        layout.append((hf_name, full_shape, n, param.dtype))
        total_numel += n

    # Group into chunks: each chunk holds params until CHUNK_NUMEL is reached.
    # Store (pidx_start, pidx_end, chunk_numel) for each chunk.
    chunks = []
    pidx_start = 0
    chunk_numel = 0
    for i, (_, _, numel, _) in enumerate(layout):
        if chunk_numel + numel > CHUNK_NUMEL and chunk_numel > 0:
            chunks.append((pidx_start, i, chunk_numel))
            pidx_start = i
            chunk_numel = 0
        chunk_numel += numel
    if chunk_numel > 0:
        chunks.append((pidx_start, len(layout), chunk_numel))

    self._wsync_layout = layout
    self._wsync_total_numel = total_numel
    self._wsync_chunk_ranges = chunks  # [(pidx_start, pidx_end, chunk_numel), ...]
    # Persistent broadcast buffer reused across all steps. Avoids per-step
    # torch.empty() that returns fresh L0 pages CCL opens new IPC handles
    # for. Stale handles accumulate across steps (run 39: wsync 31s → 50s
    # by step 2, banned:1 at step 3). Sized to the largest chunk so all
    # 37 broadcasts per step reuse the same VA.
    # See bugs/project_static_xccl_buffer_fix.md for the FSDP2 analog.
    self._wsync_max_chunk_numel = max(c[2] for c in chunks) if chunks else 0
    self._wsync_chunk_buf = None  # lazily allocated on first use
    self._wsync_layout_published = False
    log.info(
        "Rank %d: wsync layout: %d backbone params → %d chunks of ≤%.0f MiB bf16, "
        "max_chunk_numel=%d (%.0f MiB persistent buf)",
        self.rank, len(layout), len(chunks), CHUNK_NUMEL * 2 / 2**20,
        self._wsync_max_chunk_numel, self._wsync_max_chunk_numel * 2 / 2**20,
    )

def _sync_dedicated_vllm_weights(self) -> None:
    """Broadcast updated backbone + projector weights from rank 0 to vLLM rank.

    Uses chunked flat bf16 broadcasts for backbone params (1 broadcast per ~200MiB
    chunk vs ~400 individual broadcasts), reducing weight sync from ~31s to ~1-2s.

    Both FSDP1 (summon_full_params) and FSDP2 (DTensor.full_tensor()) paths
    are supported. In both cases the per-param full tensor materialization is
    collective across all training ranks — non-rank-0 ranks discard the result.

    A2 (`TORCHTUNE_BG_WSYNC_SEND=1`, rank 0 only): split into two phases.
      Phase A (main thread, lockstep): FSDP2 full_tensor pack into per-chunk
        CPU buffers + projector_state_dict() to CPU.
      Phase B (bg thread, rank 0 only): broadcast each CPU chunk over wsync_pg
        + send projector obj. Overlapped with the next step's gen+grpo.
    Bg send must complete before next sync's Phase A begins (the per-chunk
    CPU buffers double as the staging area for the next pack).
    """
    _bg_send = (
        self.rank == 0
        and os.environ.get("TORCHTUNE_BG_WSYNC_SEND", "0") == "1"
    )
    # Bg send only supported on the gloo wsync_pg (xccl path keeps GPU buffers,
    # broadcast races on shared XPU stream → unsafe). _wsync_is_gloo is computed
    # below inside the param-summon ctx; do a non-member-safe pre-check here.
    if _bg_send:
        try:
            _bg_send = (
                torch.distributed.get_backend(self._wsync_pg) == "gloo"
            )
        except Exception:
            _bg_send = False
        if not _bg_send:
            log.warning("Rank 0: TORCHTUNE_BG_WSYNC_SEND=1 ignored (wsync_pg is not gloo)")
    if _bg_send:
        # Wait for previous sync's bg send to drain so we don't trample its
        # CPU buffers while it's still broadcasting them.
        if hasattr(self, '_bg_send_done_evt') and self._bg_send_done_evt is not None:
            t_wait0 = time.perf_counter()
            self._bg_send_done_evt.wait(timeout=300.0)
            t_wait = time.perf_counter() - t_wait0
            if t_wait > 0.05:
                log.info("Rank 0: waited %.2fs for previous bg wsync send", t_wait)
            if getattr(self, '_bg_send_error', None) is not None:
                raise self._bg_send_error
        else:
            self._bg_send_done_evt = threading.Event()
            self._bg_send_done_evt.set()
            self._bg_send_error = None
            self._bg_send_thread = None
    t0 = time.perf_counter()
    _is_fsdp2 = False
    try:
        from torch.distributed.tensor import DTensor as _DT
        _is_fsdp2 = any(
            isinstance(p, _DT) for p in self._model.parameters()
        )
    except Exception:
        _is_fsdp2 = False

    torch.xpu.synchronize()
    t_sum_enter = time.perf_counter()

    if _is_fsdp2:
        # FSDP2 path: no summon. We iterate params; .full_tensor() is collective
        # over the sharding mesh (which is _training_pg, established at setup()).
        # All training ranks 0..N-2 must enter the iteration in lockstep.
        _ctx = _NullCtx()
    else:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        _ctx = FSDP.summon_full_params(self._model, writeback=False, rank0_only=True)

    with _ctx:
        torch.xpu.synchronize()
        t_sum_done = time.perf_counter()
        # Lazy layout compute (generic recipe).
        if not hasattr(self, '_wsync_layout'):
            _compute_wsync_layout(self, self._model)

        # _wsync_pg is [0, vllm_rank] only; non-members (ranks 1..N-2) cannot
        # query its backend. They participate in the FSDP2 full_tensor()
        # collectives over the sharding mesh but never broadcast on _wsync_pg.
        if self.rank == 0:
            _wsync_is_gloo = (
                torch.distributed.get_backend(self._wsync_pg) == "gloo"
            )
        else:
            _wsync_is_gloo = True  # unused on non-rank-0; safe placeholder

        # Layout publish (rank 0 → vLLM rank). Done outside the param-iter so
        # the broadcast happens once across the wsync_pg only.
        if self.rank == 0 and not getattr(self, '_wsync_layout_published', False):
            _layout_obj = {
                "layout": self._wsync_layout,
                "chunk_ranges": self._wsync_chunk_ranges,
                "max_chunk_numel": self._wsync_max_chunk_numel,
                "total_numel": self._wsync_total_numel,
            }
            _layout_dev = "cpu" if _wsync_is_gloo else self._device
            torch.distributed.broadcast_object_list(
                [_layout_obj], src=0, group=self._wsync_pg, device=_layout_dev,
            )
            self._wsync_layout_published = True

        # Buffer alloc on rank 0 only.
        if self.rank == 0:
            if self._wsync_chunk_buf is None:
                self._wsync_chunk_buf = torch.empty(
                    self._wsync_max_chunk_numel, dtype=torch.bfloat16, device=self._device,
                )
            # Bg-send needs ONE pinned CPU buffer per chunk (the staging area
            # the bg thread broadcasts from). Serial mode reuses a single buf.
            if _wsync_is_gloo and _bg_send and not hasattr(self, '_wsync_chunk_cpu_bufs_send'):
                self._wsync_chunk_cpu_bufs_send = [
                    torch.empty(cn, dtype=torch.bfloat16, device="cpu", pin_memory=True)
                    for (_, _, cn) in self._wsync_chunk_ranges
                ]
            if _wsync_is_gloo and not hasattr(self, '_wsync_chunk_cpu_buf'):
                self._wsync_chunk_cpu_buf = torch.empty(
                    self._wsync_max_chunk_numel, dtype=torch.bfloat16,
                    device="cpu", pin_memory=True,
                )

        if hasattr(self._model, 'vllm_param_iter'):
            params_list = list(self._model.vllm_param_iter())
        else:
            params_list = list(_backbone_param_iter(self, self._model))

        t_pack_total = 0.0
        t_d2h_total = 0.0
        t_bcast_total = 0.0
        bcast_bytes_total = 0
        _unperm_needed = _needs_qk_unpermute(self)
        if (
            _unperm_needed
            and self.rank == 0
            and not getattr(self, "_qk_unperm_dedicated_engaged_logged", False)
        ):
            log.info(
                "Rank 0: dedicated wsync QK un-permute ENGAGED "
                "(n_heads=%s n_kv_heads=%s head_dim=%s)",
                getattr(self, "_model_num_heads", None),
                getattr(self, "_model_num_kv_heads", None),
                getattr(self, "_model_head_dim", None),
            )
            self._qk_unperm_dedicated_engaged_logged = True
        for ci, (pidx_start, pidx_end, chunk_numel) in enumerate(self._wsync_chunk_ranges):
            tp0 = time.perf_counter()
            if self.rank == 0:
                chunk_view = self._wsync_chunk_buf[:chunk_numel]
                offset = 0
            for _hf_name, param in params_list[pidx_start:pidx_end]:
                # full_tensor() is COLLECTIVE — every training rank must call.
                # On FSDP1 with summon_full_params(rank0_only=True) the params
                # already are full tensors on rank 0, so just .data works.
                if _is_fsdp2 and hasattr(param, 'full_tensor'):
                    full = param.full_tensor()
                else:
                    full = param.data
                if self.rank == 0:
                    if _unperm_needed:
                        full = _maybe_unpermute_qk(self, _hf_name, full)
                    n = full.numel()
                    chunk_view[offset:offset + n].copy_(
                        full.to(torch.bfloat16).view(-1)
                    )
                    offset += n
                del full
            torch.xpu.synchronize()
            tp1 = time.perf_counter()
            t_pack_total += (tp1 - tp0)
            if self.rank == 0:
                if _bg_send:
                    # Phase A: D2H into per-chunk CPU buf; defer broadcast to bg thread.
                    cpu_view = self._wsync_chunk_cpu_bufs_send[ci][:chunk_numel]
                    cpu_view.copy_(chunk_view, non_blocking=False)
                    tp2 = time.perf_counter()
                    t_d2h_total += (tp2 - tp1)
                elif _wsync_is_gloo:
                    cpu_view = self._wsync_chunk_cpu_buf[:chunk_numel]
                    cpu_view.copy_(chunk_view, non_blocking=False)
                    td2h = time.perf_counter()
                    torch.distributed.broadcast(cpu_view, src=0, group=self._wsync_pg)
                    tp2 = time.perf_counter()
                    t_d2h_total += (td2h - tp1)
                    t_bcast_total += (tp2 - td2h)
                else:
                    torch.distributed.broadcast(chunk_view, src=0, group=self._wsync_pg)
                    torch.xpu.synchronize()
                    tp2 = time.perf_counter()
                    t_bcast_total += (tp2 - tp1)
                bcast_bytes_total += chunk_numel * 2  # bf16

        if self.rank == 0:
            # Multimodal models (BioReason) carry trainable projector modules
            # outside the backbone — broadcast their state dicts as a Python obj.
            # Plain text models have no projectors; broadcast None as a placeholder
            # so the receiver's matching call doesn't deadlock.
            t_proj_start = time.perf_counter()
            if hasattr(self._model, 'projector_state_dict'):
                raw_proj_sd = self._model.projector_state_dict()
                proj_sd = {k: {pk: pv.cpu() for pk, pv in v.items()} for k, v in raw_proj_sd.items()}
            else:
                proj_sd = None
            if not _bg_send:
                _proj_dev = "cpu" if _wsync_is_gloo else self._device
                torch.distributed.broadcast_object_list([proj_sd], src=0, group=self._wsync_pg, device=_proj_dev)
            t_proj_done = time.perf_counter()

            if _bg_send:
                # Phase B: hand staged CPU bufs + proj_sd to bg thread.
                _ranges = list(self._wsync_chunk_ranges)
                _bufs = self._wsync_chunk_cpu_bufs_send
                _wsync_pg = self._wsync_pg

                def _bg_send_run():
                    try:
                        for ci2, (_, _, cn) in enumerate(_ranges):
                            torch.distributed.broadcast(
                                _bufs[ci2][:cn], src=0, group=_wsync_pg
                            )
                        torch.distributed.broadcast_object_list(
                            [proj_sd], src=0, group=_wsync_pg, device="cpu",
                        )
                    except BaseException as exc:  # noqa: BLE001
                        log.exception("Rank 0 bg wsync send: fatal")
                        self._bg_send_error = exc
                    finally:
                        self._bg_send_done_evt.set()

                self._bg_send_done_evt.clear()
                self._bg_send_thread = threading.Thread(
                    target=_bg_send_run, name="rank0_bg_wsync_send", daemon=True,
                )
                self._bg_send_thread.start()
                _phase_a_total = time.perf_counter() - t0
                log.info(
                    "WSYNC_PHASE rank=0 total=%.2fs summon=%.2fs pack=%.2fs d2h=%.2fs proj_pack=%.2fs backend=%s mode=%s bg_send=1 (broadcast deferred)",
                    _phase_a_total,
                    t_sum_done - t_sum_enter,
                    t_pack_total,
                    t_d2h_total,
                    t_proj_done - t_proj_start,
                    "gloo" if _wsync_is_gloo else "xccl",
                    "fsdp2" if _is_fsdp2 else "fsdp1",
                )
            else:
                bw_gbs = (bcast_bytes_total / 1e9) / max(t_bcast_total, 1e-6)
                log.info(
                    "WSYNC_PHASE rank=0 total=%.2fs summon=%.2fs pack=%.2fs d2h=%.2fs bcast=%.2fs (%.2f GB at %.2f GB/s) proj=%.2fs backend=%s mode=%s",
                    time.perf_counter() - t0,
                    t_sum_done - t_sum_enter,
                    t_pack_total,
                    t_d2h_total,
                    t_bcast_total,
                    bcast_bytes_total / 1e9,
                    bw_gbs,
                    t_proj_done - t_proj_start,
                    "gloo" if _wsync_is_gloo else "xccl",
                    "fsdp2" if _is_fsdp2 else "fsdp1",
                )

    log.info(
        "Rank %d: dedicated vLLM weight sync: %d backbone params + projectors in %.1fs",
        self.rank, len(self._wsync_layout), time.perf_counter() - t0,
    )

def _recv_weight_update(self) -> None:
    """Receive updated backbone + projector weights from rank 0 (vLLM rank side).

    Receives a single flat bf16 tensor containing all backbone params, unpacks it
    using the pre-computed layout, and loads each param into the vLLM engine.

    A2 (`TORCHTUNE_VLLM_BG_WSYNC=1`): the bg wsync thread receives gloo
    chunks WITHOUT the engine lock (gloo PG independent of vLLM state);
    only `load_weights` is fenced by the engine lock. To preserve atomicity
    across the multi-chunk update (gen N+1 must see either all-old or
    all-new weights, never a mid-batch mix), all chunks are first staged
    into a contiguous CPU buffer; then a single lock-protected pass calls
    load_weights for every param. Net effect on the vLLM rank's per-step
    time: max(gen, recv) instead of gen + recv.
    """
    # Multimodal recipes (BioReason) attach a projector-bearing model as
    # `_embed_model`; plain text recipes don't. The projector path is
    # gated on `_policy is not None`.
    _policy = getattr(self, '_embed_model', None)
    _engine_lock = getattr(self, '_vllm_engine_lock', None)
    _bg_wsync = _engine_lock is not None
    t0 = time.perf_counter()

    llm_model = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Receive layout from rank 0 on first sync (generic recipe — vLLM rank
    # has no model and can't compute the layout itself). BioReason precomputes
    # in setup() so this branch is skipped there.
    _wsync_is_gloo_pre = (
        torch.distributed.get_backend(self._wsync_pg) == "gloo"
    )
    if not getattr(self, '_wsync_layout_published', False):
        _layout_dev = "cpu" if _wsync_is_gloo_pre else self._device
        _obj = [None]
        torch.distributed.broadcast_object_list(
            _obj, src=0, group=self._wsync_pg, device=_layout_dev,
        )
        _layout_obj = _obj[0]
        self._wsync_layout = _layout_obj["layout"]
        self._wsync_chunk_ranges = _layout_obj["chunk_ranges"]
        self._wsync_max_chunk_numel = _layout_obj["max_chunk_numel"]
        self._wsync_total_numel = _layout_obj["total_numel"]
        self._wsync_chunk_buf = None
        self._wsync_layout_published = True
        log.info(
            "Rank %d (vLLM): received wsync layout: %d params, %d chunks, max_chunk_numel=%d",
            self.rank, len(self._wsync_layout), len(self._wsync_chunk_ranges),
            self._wsync_max_chunk_numel,
        )

    # Backbone: receive backbone params in chunks matching sender's chunk_ranges.
    # Each chunk_buf is ≤200 MiB — fits in rank 11's ~858 MiB post-init headroom.
    # Persistent buffer (allocated once, reused every step) avoids CCL IPC-handle
    # accumulation across steps. Must match sender's persistent buffer size.
    _wsync_is_gloo = (
        torch.distributed.get_backend(self._wsync_pg) == "gloo"
    )
    if self._wsync_chunk_buf is None:
        self._wsync_chunk_buf = torch.empty(
            self._wsync_max_chunk_numel, dtype=torch.bfloat16, device=self._device,
        )
    if _wsync_is_gloo and not hasattr(self, '_wsync_chunk_cpu_buf'):
        self._wsync_chunk_cpu_buf = torch.empty(
            self._wsync_max_chunk_numel, dtype=torch.bfloat16,
            device="cpu", pin_memory=True,
        )

    t_bcast_total = 0.0
    t_h2d_total = 0.0
    t_load_total = 0.0
    t_lock_wait = 0.0
    bcast_bytes_total = 0

    # In bg mode, stage every chunk into its own per-chunk CPU buffer first
    # (lock-free, overlapped with main-thread vllm.generate). In serial mode,
    # reuse the single _wsync_chunk_cpu_buf and apply each chunk inline.
    if _bg_wsync and _wsync_is_gloo:
        if not hasattr(self, '_wsync_chunk_cpu_bufs'):
            self._wsync_chunk_cpu_bufs = [
                torch.empty(cn, dtype=torch.bfloat16, device="cpu", pin_memory=True)
                for (_, _, cn) in self._wsync_chunk_ranges
            ]
        staged_cpu_views = []
        for ci, (pidx_start, pidx_end, chunk_numel) in enumerate(self._wsync_chunk_ranges):
            cpu_view = self._wsync_chunk_cpu_bufs[ci][:chunk_numel]
            tb0 = time.perf_counter()
            torch.distributed.broadcast(cpu_view, src=0, group=self._wsync_pg)
            t_bcast_total += (time.perf_counter() - tb0)
            staged_cpu_views.append((pidx_start, pidx_end, chunk_numel, cpu_view))
            bcast_bytes_total += chunk_numel * 2

        # Projector recv (also lock-free, gloo CPU obj broadcast)
        t_proj_start = time.perf_counter()
        obj = [None]
        torch.distributed.broadcast_object_list(obj, src=0, group=self._wsync_pg, device="cpu")
        proj_sd = obj[0]
        t_proj_recv = time.perf_counter() - t_proj_start

        # Atomic apply: hold engine lock for the whole load_weights pass + projector
        # swap. gen() in main thread waits here briefly.
        t_lock_t0 = time.perf_counter()
        with _engine_lock:
            t_lock_wait = time.perf_counter() - t_lock_t0
            t_apply_t0 = time.perf_counter()
            for pidx_start, pidx_end, chunk_numel, cpu_view in staged_cpu_views:
                chunk_view = self._wsync_chunk_buf[:chunk_numel]
                chunk_view.copy_(cpu_view, non_blocking=False)
                torch.xpu.synchronize()
                t_h2d_total += (time.perf_counter() - t_apply_t0)
                chunk_pairs = []
                local_off = 0
                for hf_name, shape, numel, dtype in self._wsync_layout[pidx_start:pidx_end]:
                    param_data = chunk_view[local_off:local_off + numel].view(shape).to(dtype)
                    chunk_pairs.append((hf_name, param_data))
                    local_off += numel
                tload0 = time.perf_counter()
                llm_model.load_weights(chunk_pairs)
                torch.xpu.synchronize()
                t_load_total += (time.perf_counter() - tload0)
                t_apply_t0 = time.perf_counter()
            if proj_sd is not None:
                for module_name, sd in proj_sd.items():
                    module = getattr(_policy, module_name, None)
                    if module is not None:
                        module.load_state_dict(sd, strict=False)
        t_proj_done = time.perf_counter()
    else:
        for pidx_start, pidx_end, chunk_numel in self._wsync_chunk_ranges:
            chunk_view = self._wsync_chunk_buf[:chunk_numel]
            tb0 = time.perf_counter()
            if _wsync_is_gloo:
                cpu_view = self._wsync_chunk_cpu_buf[:chunk_numel]
                torch.distributed.broadcast(cpu_view, src=0, group=self._wsync_pg)
                tb05 = time.perf_counter()
                chunk_view.copy_(cpu_view, non_blocking=False)
                torch.xpu.synchronize()
                tb1 = time.perf_counter()
                t_bcast_total += (tb05 - tb0)
                t_h2d_total += (tb1 - tb05)
            else:
                torch.distributed.broadcast(chunk_view, src=0, group=self._wsync_pg)
                torch.xpu.synchronize()
                tb1 = time.perf_counter()
                t_bcast_total += (tb1 - tb0)
            chunk_pairs = []
            local_off = 0
            for hf_name, shape, numel, dtype in self._wsync_layout[pidx_start:pidx_end]:
                param_data = chunk_view[local_off:local_off + numel].view(shape).to(dtype)
                chunk_pairs.append((hf_name, param_data))
                local_off += numel
            llm_model.load_weights(chunk_pairs)
            torch.xpu.synchronize()
            tb2 = time.perf_counter()
            t_load_total += (tb2 - tb1)
            bcast_bytes_total += chunk_numel * 2

        # Projectors (serial path)
        t_proj_start = time.perf_counter()
        obj = [None]
        _proj_dev = "cpu" if _wsync_is_gloo else self._device
        torch.distributed.broadcast_object_list(obj, src=0, group=self._wsync_pg, device=_proj_dev)
        proj_sd = obj[0]
        if proj_sd is not None:
            for module_name, sd in proj_sd.items():
                module = getattr(_policy, module_name, None)
                if module is not None:
                    module.load_state_dict(sd, strict=False)
        t_proj_done = time.perf_counter()

    t_reset_start = time.perf_counter()
    if _bg_wsync:
        # reset_prefix_cache also mutates engine state — serialize against gen.
        with _engine_lock:
            self._vllm_llm.llm_engine.reset_prefix_cache()
    else:
        self._vllm_llm.llm_engine.reset_prefix_cache()
    t_reset_done = time.perf_counter()

    bw_gbs = (bcast_bytes_total / 1e9) / max(t_bcast_total, 1e-6)
    log.info(
        "WSYNC_PHASE rank=11 total=%.2fs bcast=%.2fs (%.2f GB at %.2f GB/s) h2d=%.2fs load=%.2fs proj=%.2fs reset=%.2fs lock_wait=%.2fs bg=%s backend=%s",
        time.perf_counter() - t0,
        t_bcast_total,
        bcast_bytes_total / 1e9,
        bw_gbs,
        t_h2d_total,
        t_load_total,
        t_proj_done - t_proj_start,
        t_reset_done - t_reset_start,
        t_lock_wait,
        _bg_wsync,
        "gloo" if _wsync_is_gloo else "xccl",
    )
    log.info(
        "Rank %d: received weight update: %d backbone + projectors in %.1fs",
        self.rank, len(self._wsync_layout), time.perf_counter() - t0,
    )

def _generate_with_dedicated_vllm(
    self,
    batch_input_ids: torch.Tensor,
    context_length: int,
    protein_sequences,
):
    """Generate using the dedicated vLLM rank via the 2-rank gen_pg ring buffer.

    Phase 2: only rank 0 talks to the vLLM rank — the other training ranks no
    longer participate in the generation handshake. Rank 0 is responsible for
    fanning the resulting `query_responses` out to the rest of the training
    cohort via the world-PG broadcast helper that callers already invoke.

    All collectives go over `self._gen_pg` (gloo, [0, vllm_rank]), which means
    payloads are CPU tensors. Returns the query_responses CPU tensor; the caller
    moves it to device + broadcasts to other training ranks.

    Returns:
        query_responses: ``[B*G, context_length + max_generated_tokens]`` (CPU)
    """
    if self.rank != 0:
        raise RuntimeError(
            "_generate_with_dedicated_vllm must be called only on rank 0 "
            "(other training ranks should consume via _broadcast_query_responses)"
        )

    bsz = batch_input_ids.shape[0]
    total_len = context_length + self._max_generated_tokens

    # Send protein_sequences + batch_input_ids over gen_pg as a single object payload.
    payload = {
        "protein_sequences": protein_sequences,
        "batch_input_ids": batch_input_ids.cpu(),
        "context_length": context_length,
    }
    torch.distributed.broadcast_object_list(
        [payload], src=0, group=self._gen_pg
    )

    # Receive query_responses (CPU tensor) from the vLLM rank over gen_pg.
    query_responses = torch.empty(
        (bsz, total_len),
        dtype=batch_input_ids.dtype,
    )
    torch.distributed.recv(
        query_responses, src=self._vllm_dedicated_rank, group=self._gen_pg
    )

    return query_responses

def _run_vllm_generation_server(self) -> None:
    """Run the dedicated vLLM generation server loop (rank N-1 only).

    Phase 2: communicates with rank 0 ONLY, over the 2-rank gloo `_gen_pg`
    ring buffer. The vLLM rank no longer participates in any world-PG
    collective during generation, which lets training ranks 1..N-2 run
    fwd/bwd in parallel with vLLM's generation step.

    Per iteration:
        1. recv payload {protein_sequences, batch_input_ids, context_length}
           from rank 0 over gen_pg.
        2. (BioReason only) compute prompt_embeds via local ESM3 + projectors.
        3. Generate with vLLM.
        4. send query_responses CPU tensor back to rank 0 over gen_pg.
        5. Receive weight update from rank 0 via wsync_pg (default: serial)
           OR drained by background thread if TORCHTUNE_VLLM_BG_WSYNC=1 (A2).

    Loop exits on the shutdown sentinel `{"shutdown": True}` (Step 4) or when
    the legacy total_steps counter is exhausted.

    A2 mode (`TORCHTUNE_VLLM_BG_WSYNC=1`): a background thread blocks on
    `_recv_weight_update()` (gloo on `_wsync_pg`) while the main thread runs
    the next gen. A lock serializes `vllm.generate()` against the swap-in
    inside `_recv_weight_update()` (vLLM is not thread-safe). Net effect:
    the next gen does not block on the prior wsync recv.
    """
    from vllm import SamplingParams

    _bg_wsync = os.environ.get("TORCHTUNE_VLLM_BG_WSYNC", "0") == "1"
    log.info(
        "Rank %d (vLLM server): entering generation loop (gen_pg=2-rank gloo, "
        "exits on shutdown sentinel, bg_wsync=%s)",
        self.rank, _bg_wsync,
    )

    if _bg_wsync:
        self._vllm_engine_lock = threading.Lock()
        self._wsync_done_evt = threading.Event()
        self._wsync_done_evt.set()  # idle at start
        self._wsync_error: "Optional[BaseException]" = None  # noqa: F821
        self._wsync_stop_evt = threading.Event()

        def _bg_wsync_loop():
            log.info("Rank %d (vLLM bg wsync): thread started", self.rank)
            try:
                while not self._wsync_stop_evt.is_set():
                    # Block on the next chunk-broadcast set from rank 0.
                    # _recv_weight_update grabs the engine lock right before
                    # calling load_weights so the main gen thread can keep
                    # using the engine for the prior generation while wsync
                    # bytes arrive over gloo.
                    self._wsync_done_evt.clear()
                    self._recv_weight_update()
                    self._wsync_done_evt.set()
            except BaseException as exc:  # noqa: BLE001
                log.exception("Rank %d (vLLM bg wsync): fatal", self.rank)
                self._wsync_error = exc
                self._wsync_done_evt.set()
        self._wsync_thread = threading.Thread(
            target=_bg_wsync_loop, name="vllm_bg_wsync", daemon=True,
        )
        self._wsync_thread.start()

    step = -1
    while True:
        step += 1
        # 1. Receive request payload from rank 0 over gen_pg.
        obj = [None]
        torch.distributed.broadcast_object_list(obj, src=0, group=self._gen_pg)
        payload = obj[0]
        if isinstance(payload, dict) and payload.get("shutdown"):
            log.info("Rank %d (vLLM server): shutdown sentinel received at step %d", self.rank, step)
            return
        protein_sequences = payload["protein_sequences"]
        batch_input_ids = payload["batch_input_ids"]  # CPU tensor
        context_length = payload["context_length"]
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        # 2. Compute prompt_embeds (BioReason path) or skip (text-only).
        if protein_sequences is not None and getattr(self, "_embed_model", None) is not None:
            batch_input_ids_dev = batch_input_ids.to(self._device)
            grpo_size = self.grpo_samples
            unique_input_ids = batch_input_ids_dev[::grpo_size]  # [B, L]
            with torch.no_grad():
                pe_base = self._embed_model.build_prompt_embeds(
                    unique_input_ids, protein_sequences
                )  # [B, P, H] CPU
            prompt_embeds = (
                pe_base.unsqueeze(1)
                .expand(-1, grpo_size, -1, -1)
                .reshape(bsz, pe_base.shape[1], pe_base.shape[2])
                .contiguous()
            )  # [B*G, P, H] CPU
            vllm_prompts = [{"prompt_embeds": prompt_embeds[i]} for i in range(bsz)]
            pad_id = self._embed_model.tokenizer.pad_token_id
        else:
            vllm_prompts = [
                {"prompt_token_ids": batch_input_ids[i, :context_length].tolist()}
                for i in range(bsz)
            ]
            pad_id = 0

        # 3. Generate with vLLM.
        sampling_params = SamplingParams(
            max_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k if self._top_k else -1,
            detokenize=False,
        )
        t0 = time.perf_counter()
        if _bg_wsync:
            # A2: serialize against bg wsync's load_weights() swap-in.
            with self._vllm_engine_lock:
                outputs = self._vllm_llm.generate(
                    prompts=vllm_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
        else:
            outputs = self._vllm_llm.generate(
                prompts=vllm_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        gen_time = time.perf_counter() - t0

        # Pack into CPU query_responses tensor (gen_pg is gloo).
        query_responses = torch.full(
            (bsz, total_len), pad_id, dtype=batch_input_ids.dtype
        )
        query_responses[:, :context_length] = batch_input_ids[:, :context_length]
        total_tokens = 0
        for i, output in enumerate(outputs):
            comp = output.outputs[0].token_ids
            total_tokens += len(comp)
            length = min(len(comp), self._max_generated_tokens)
            query_responses[i, context_length: context_length + length] = torch.tensor(
                comp[:length], dtype=batch_input_ids.dtype
            )

        log.info(
            "Rank %d (vLLM): step=%d, gen=%d seqs, %d tokens in %.1fs",
            self.rank, step, bsz, total_tokens, gen_time,
        )

        # 4. Send query_responses back to rank 0 over gen_pg.
        torch.distributed.send(query_responses, dst=0, group=self._gen_pg)

        # 5. Receive weight update from rank 0.
        # A2: bg thread drains wsync; main thread continues straight to next gen recv.
        if not _bg_wsync:
            self._recv_weight_update()
        elif self._wsync_error is not None:
            raise self._wsync_error

    if _bg_wsync:
        # Stop bg wsync thread so it doesn't block on a recv that will never come.
        self._wsync_stop_evt.set()
        # Best-effort join; thread may be parked in gloo recv on a torn-down PG.
        self._wsync_thread.join(timeout=5.0)
    log.info("Rank %d (vLLM server): generation loop complete", self.rank)

def _save_raw_bytes(state_dict: dict, path: str) -> int:
    """Write tensors as raw bytes with a simple JSON header.

    Format: [8-byte header_len][JSON header][raw tensor bytes...]

    BF16 tensors are stored as int16 bytes (same bit pattern, no conversion)
    and tagged with dtype "torch.bfloat16" in the header so the reader can
    reinterpret correctly.

    This is ~40× faster than safetensors for large models:
    - safetensors: ~1.3 GB/s (CPU serialization bottleneck)
    - raw bytes: view(int16).numpy().tobytes() = memcpy at DDR5 bandwidth (~50 GB/s)
    - For 62 GB BF16: ~2s vs ~47s

    Returns number of params written.
    """
    import json

    header_entries = []
    np_arrays = []
    offset = 0

    for name, tensor in state_dict.items():
        dtype_str = str(tensor.dtype)  # e.g. "torch.bfloat16"
        shape = list(tensor.shape)

        # BF16 not supported by numpy — view as int16 (same bits, just reinterpreted).
        if tensor.dtype == torch.bfloat16:
            np_arr = tensor.view(torch.int16).numpy()
        else:
            np_arr = tensor.numpy()

        # Ensure contiguous layout so buffer protocol write is a single memcpy.
        if not np_arr.flags['C_CONTIGUOUS']:
            import numpy as _np
            np_arr = _np.ascontiguousarray(np_arr)

        nbytes = np_arr.nbytes

        header_entries.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "offset": offset,
            "nbytes": nbytes,
        })
        np_arrays.append(np_arr)  # zero-copy view — no tobytes() allocation
        offset += nbytes

    header_json = json.dumps(header_entries).encode("utf-8")
    header_len_bytes = struct.pack("<Q", len(header_json))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header_len_bytes)
        f.write(header_json)
        for arr in np_arrays:
            f.write(arr)  # buffer protocol: kernel copies directly from numpy memory

    return len(header_entries)

def _post_weights_to_vllm(self, save_path: str, n_params: int, t_gather: float, t0: float) -> None:
    """POST to all vLLM replicas to reload from raw bytes file.

    Called from background thread — runs while next step's generation proceeds.
    Uses load_weights_from_raw endpoint (fast frombuffer path).

    Fans out the POSTs across all vLLM replicas in parallel via a
    ThreadPoolExecutor so the wall time is max(per-replica load) rather than
    sum. Each /collective_rpc call blocks until that replica's
    load_weights_from_raw returns (~9s for Qwen3-4B / ~8.5 GB on DAOS), so a
    12-way serial loop was costing ~108s while 12-way parallel POSTs
    complete in ~10-15s (limited by per-replica load + DAOS read contention).
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t_http0 = time.perf_counter()

    def _post_one(url: str):
        try:
            r = requests.post(
                f"{url}/collective_rpc",
                json={"method": "load_weights_from_raw", "args": [save_path]},
                timeout=600,
            )
            if r.status_code != 200:
                log.warning("vLLM weight reload (raw) failed (%s): %s %s", url, r.status_code, r.text[:200])
                return
            result = r.json()
            results = result.get("results", [{}])
            first = results[0] if results else {}
            if isinstance(first, dict) and first.get("status") != "ok":
                log.warning("vLLM weight reload (raw) error (%s): %s", url, first)
        except Exception as e:
            log.error("vLLM weight reload HTTP error (%s): %s", url, e)

    urls = list(self._vllm_urls)
    with ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
        futures = [pool.submit(_post_one, url) for url in urls]
        for f in as_completed(futures):
            f.result()
    t_http = time.perf_counter() - t_http0

    with ThreadPoolExecutor(max_workers=max(1, len(self._vllm_clients))) as pool:
        list(pool.map(lambda c: c.reset_prefix_cache(), self._vllm_clients))

    log.info(
        "Rank %d: weight sync %d vLLM replica(s): %d params %.1fs total "
        "(gather=%.1fs [sync] save=%.1fs http=%.1fs) [async raw-bytes: %s]",
        self.rank, len(self._vllm_urls), n_params, time.perf_counter() - t0,
        t_gather, time.perf_counter() - t0 - t_gather - t_http, t_http, save_path,
    )

def _sync_weights_to_vllm(self) -> None:
    """Dispatch weight sync to the configured method (raw_bytes, shm, or xccl).

    Dispatches to:
      raw_bytes — file-based /dev/shm write + HTTP POST (default, ~9s for 3B)
      shm       — POSIX shared memory, zero-copy on vLLM side (~5s for 3B, ~3s for 31B)
      xccl      — GPU→GPU XCCL broadcast, no CPU staging (~14 GB/s on Aurora)

    All methods have an async Phase 2: FSDP gather is synchronous across all ranks,
    then copy/broadcast runs in a background thread overlapping with generation.
    Call _wait_for_sync_complete() before the next sync dispatch.
    """
    _method = getattr(self, "_vllm_weight_sync_method", "raw_bytes")
    if _method == "xccl":
        return self._sync_weights_to_vllm_xccl()
    if _method == "shm":
        return self._sync_weights_to_vllm_shm()
    # raw_bytes path below:
    t0 = time.perf_counter()
    hf_state_dict = {}

    # HSDP-safe publisher gate: the FULL_STATE_DICT gather is COLLECTIVE (all shard
    # ranks must call state_dict()), but only ONE rank may save to _weight_sync_path +
    # POST to the shared vLLM pool. With dp_replicate>1 there are multiple shard
    # leaders (one per replica) — if each saved/POSTed they would clobber the same
    # file and race the same vLLM servers. All replicas are weight-identical (HYBRID_
    # SHARD all-reduces grads across the replicate dim), so global rank 0 publishes for
    # everyone. Single-replica: _is_shard_leader == rank 0, so this is unchanged.
    _is_publisher = (
        (self.rank == 0) if getattr(self, "_dp_replicate", 1) > 1
        else getattr(self, "_is_shard_leader", self._is_rank_zero)
    )

    # BioReason: vLLM only loads the backbone (Qwen3-4B). Strip the
    # 'backbone.' prefix and skip everything else (ESM3, GO encoder,
    # projectors). Without this filter, vLLM's load_weights() would
    # see hundreds of names it doesn't recognize.
    _is_bior = getattr(self, "_is_bioreason", False)

    def _accept_and_rename(name: str):
        """Return the vLLM-side name, or None to skip."""
        if _is_bior:
            # Strip FSDP / activation-checkpointing wrappers first.
            clean = name.replace("_fsdp_wrapped_module.", "")
            clean = clean.replace("_checkpoint_wrapped_module.", "")
            if not clean.startswith("backbone."):
                return None
            return clean[len("backbone."):]
        return self._tune_to_hf_map.get(name, name)

    # FSDP1 path: state_dict() handles gathering within shard group.
    # BioReason wraps FSDP1 with dp_replicate=1 (pure shard), so the older
    # "_dp_replicate > 1" gate would skip this branch; use _use_fsdp1 alone.
    if getattr(self, '_use_fsdp1', False):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
        with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
            full_sd = self._model.state_dict()  # COLLECTIVE — all shard ranks call it
        if _is_publisher:
            for param_name, param in full_sd.items():
                hf_name = _accept_and_rename(param_name)
                if hf_name is None:
                    continue
                hf_state_dict[hf_name] = param.cpu()
        del full_sd
    else:
        # FSDP2 path: gather DTensor → full tensor.
        sharded_sd = self._model.state_dict()
        for param_name, param in sharded_sd.items():
            if _is_bior and not param_name.startswith("backbone."):
                # Skip non-backbone — but still call full_tensor on DTensors
                # if they're sharded (defensive; non-backbone shouldn't be).
                continue
            if param.is_cpu:
                param = param.to(self._device)

    # Barrier: all ranks finish full_tensor() before shard leader saves
    if not self._production_mode:
        torch.distributed.barrier()

    t_gather = time.perf_counter() - t0

    if _is_publisher:
        save_path = self._weight_sync_path
        n_params = len(hf_state_dict)

        # Mark sync in-progress BEFORE spawning thread (prevent race with
        # _wait_for_sync_complete checking the event).
        self._sync_done_event.clear()
        self._sync_error = None
        self._sync_id_counter += 1
        self._pending_sync_id = self._sync_id_counter

        def _bg_save_and_post(state_dict=hf_state_dict):
            try:
                t_save0 = time.perf_counter()
                self._save_raw_bytes(state_dict, save_path)
                t_save = time.perf_counter() - t_save0
                del state_dict  # free ~62 GB as soon as written
                log.info(
                    "Rank %d: async sync save done: %d params in %.1fs (%.1f GB/s) → %s",
                    self.rank, n_params, t_save,
                    (n_params and os.path.getsize(save_path) / 1024**3 / t_save if t_save > 0 else 0),
                    save_path,
                )
                self._post_weights_to_vllm(save_path, n_params, t_gather, t0)
            except Exception as e:
                log.error("Rank %d: async weight sync failed: %s", self.rank, e, exc_info=True)
                self._sync_error = e
            finally:
                self._sync_done_event.set()

        t = threading.Thread(target=_bg_save_and_post, daemon=True, name="weight_sync")
        t.start()
        log.info(
            "Rank %d: weight sync gather done in %.1fs, async save+POST started (%d params → %s)",
            self.rank, t_gather, n_params, save_path,
        )

    # NOTE: do NOT call device_empty_cache() here — empty_cache() after FSDP AllGather
    # frees L0 buffers that CCL still has open IPC handles for → banned:1 GPU fault.
    # See docs/bugs/intel_xpu_resource_leak_bug_report.md.

def _init_sender_pool(self) -> None:
    """Initialize the sender pool for dynamic sender rotation.

    Called by ALL ranks on first _sync_weights_to_vllm_xccl() invocation.
    When WSYNC_SENDER_POOL_SIZE > 0, creates a pool of sender ranks that
    rotate the cross-node broadcast duty, distributing the CXI MR leak
    across N ranks instead of concentrating it on rank 0.
    When WSYNC_SENDER_POOL_SIZE <= 0, falls back to legacy single-sender
    mode (shard leader only).
    """
    import datetime
    import threading
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    import requests

    raw_pool = os.environ.get("WSYNC_SENDER_POOL_SIZE", "0")
    pool_size = int(raw_pool)
    pool_size = min(pool_size, self.world_size - 2) if pool_size > 0 else 0
    self._wsync_round = 0
    log.info("Rank %d: _init_sender_pool raw=%r pool_size=%d world=%d",
             self.rank, raw_pool, pool_size, self.world_size)

    # Pre-compute the shared inputs needed by BOTH the legacy branches AND
    # the WS10 sharded-PG construction below. WS10 must run regardless of
    # WSYNC_SENDER_POOL_SIZE — without this, EP runs (which default
    # pool_size=0) would never construct sharded PGs.
    _ws10_sharded_gate = (
        os.environ.get("TORCHTUNE_EP_WSYNC_SHARDED", "0") == "1"
    )
    _ws10_ep_d_pre = getattr(self, "_expert_parallel_degree", 1)
    _ws10_should_build = (
        _ws10_sharded_gate and _ws10_ep_d_pre > 1
        and getattr(self, "_shard_rank", None) is not None
    )

    if pool_size <= 0:
        self._wsync_sender_pool = None
        # WS3.5: only ONE rank (global rank 0) drives the XCCL wsync init.
        # With dp_replicate>1, _is_shard_leader is true on multiple ranks
        # which would race to bind TCPStore is_master=True on the same port.
        if getattr(self, "_is_xccl_leader", self._is_shard_leader):
            self._init_xccl_weight_sync()
        # WS10 sharded PG construction needs the shared TCPStore + URLs that
        # are set up inside _init_xccl_weight_sync on the leader. Construct
        # sharded PGs HERE before returning so EP runs (legacy single-sender
        # path, pool_size=0) get the same PGs as the multi-sender branch.
        if _ws10_should_build:
            self._build_ws10_sharded_pgs()
        self._wsync_pool_init_done = True
        return

    self._wsync_sender_pool = list(range(2, 2 + pool_size))
    is_sender = self.rank in self._wsync_sender_pool
    pool_index = self._wsync_sender_pool.index(self.rank) if is_sender else -1

    xccl_port = getattr(self, "_vllm_xccl_port", 51217)
    import socket as _socket
    _xccl_host = (
        os.environ.get("TORCHTUNE_XCCL_HOST")
        or os.environ.get("MASTER_ADDR")
        or _socket.gethostname()
    )
    wsync_method = os.environ.get("WSYNC_CROSS_METHOD", "xccl_broadcast")
    tp_size = getattr(self, "_vllm_tp_size", 1)
    num_replicas = len(self._vllm_urls)
    world_size = 1 + num_replicas * tp_size

    log.info(
        "Rank %d: init sender pool (pool_size=%d, pool=%s, is_sender=%s, "
        "pool_index=%d, method=%s, host=%s, port=%d)",
        self.rank, pool_size, self._wsync_sender_pool, is_sender,
        pool_index, wsync_method, _xccl_host, xccl_port,
    )

    vllm_errors = []
    replica_threads = []

    # WS3.5: gate TCPStore master + vLLM HTTP POSTs on global rank 0 only.
    _is_xccl_leader = getattr(self, "_is_xccl_leader", self._is_shard_leader)
    if _is_xccl_leader:
        store = dist.TCPStore(
            host_name="0.0.0.0",
            port=xccl_port,
            world_size=world_size,
            is_master=True,
            timeout=datetime.timedelta(seconds=120),
            wait_for_workers=False,
        )
        self._xccl_store = store

        use_two_hop = num_replicas > 0
        self._xccl_two_hop = use_two_hop

        def _post_replica(r_idx, url):
            base_rank = 1 + r_idx * tp_size
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={
                        "method": "init_xccl_communicator",
                        "args": [_xccl_host, xccl_port, world_size,
                                 base_rank, use_two_hop, wsync_method,
                                 pool_size],
                    },
                    timeout=120,
                )
                if r.status_code != 200:
                    vllm_errors.append(
                        f"init_xccl_communicator failed ({url}): "
                        f"{r.status_code} {r.text}")
                    return
                result = r.json().get("results", [{}])
                first = result[0] if result else {}
                if isinstance(first, dict) and first.get("status") != "ok":
                    vllm_errors.append(
                        f"init_xccl_communicator error ({url}): {first}")
            except Exception as e:
                vllm_errors.append(str(e))

        replica_threads = [
            threading.Thread(target=_post_replica, args=(r_idx, url),
                             daemon=True)
            for r_idx, url in enumerate(self._vllm_urls)
        ]
        for t in replica_threads:
            t.start()

    torch.distributed.barrier()

    if is_sender:
        if not _is_xccl_leader:
            sender_store = dist.TCPStore(
                host_name=_xccl_host,
                port=xccl_port,
                world_size=world_size,
                is_master=False,
                timeout=datetime.timedelta(seconds=120),
            )
        else:
            sender_store = self._xccl_store

        prefix = f"wsync_sender_{pool_index}"
        prefixed = c10d.PrefixStore(prefix, sender_store)
        self._wsync_cross_method = wsync_method

        if wsync_method == "gloo":
            self._my_cross_pg = c10d.ProcessGroupGloo(
                store=prefixed, rank=0, size=2,
            )
        else:
            opts = c10d.ProcessGroupXCCL.Options()
            self._my_cross_pg = c10d.ProcessGroupXCCL(
                store=prefixed, rank=0, size=2, options=opts,
            )

        self._sync_done_event = threading.Event()
        self._sync_done_event.set()
        self._sync_error = None
        self._xccl_bcast_buf = None
        # Sync dispatch counter — incremented when a new sync is dispatched, set
        # to None when consumed by _wait_for_sync_complete (which then bumps the
        # weight version). Prevents over-counting when _wait_for_sync_complete is
        # called more than once between dispatches.
        self._pending_sync_id = None
        self._sync_id_counter = 0

        if not hasattr(self, '_tune_to_hf_map'):
            self._build_tune_to_hf_map()

        from torchtune.dev.rl.vllm_client import VLLMClient
        self._vllm_clients = [
            VLLMClient(
                base_url=url,
                group_port=getattr(self, "_vllm_group_port", 51216),
                connection_timeout=900.0,
            )
            for url in self._vllm_urls
        ]

        log.info(
            "Rank %d: sender pool PG created (pool_index=%d, method=%s)",
            self.rank, pool_index, wsync_method,
        )

    if _is_xccl_leader:
        for t in replica_threads:
            t.join(timeout=120)
        if vllm_errors:
            raise RuntimeError(
                f"vLLM sender pool init failed: {vllm_errors}")

    torch.distributed.barrier()

    if _ws10_should_build:
        self._build_ws10_sharded_pgs()
    self._wsync_pool_init_done = True
    log.info("Rank %d: sender pool init complete (pool_size=%d)",
             self.rank, pool_size)


def _build_ws10_sharded_pgs(self) -> None:
    """WS10 Commit A: construct per-EP-rank sharded cross PGs.

    Called by ``_init_sender_pool`` on ALL trainer ranks (not just senders).
    Builds one 2-rank cross PG per trainer EP shard rank R in
    ``[0..ep_d-1]``, each peered with the vLLM root with prefix
    ``wsync_sharded_R{R}``. The current rank participates only if its
    ``_shard_rank`` is in that range; otherwise it's a no-op (PGs are
    constructed by the other ranks; this rank holds an empty dict).

    Receiver-side construction lives in
    ``vllm_weight_sync_worker.py:init_xccl_sharded_pgs``.

    No bytes flow over these PGs in Commit A — Commit B wires them into
    the deferred broadcast loop. Default backend is gloo (per WS-Varlen
    XCCL+OFI deadlock history); override via
    ``TORCHTUNE_EP_WSYNC_SHARDED_METHOD=xccl_broadcast``.
    """
    import datetime
    import threading
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    import requests

    self._xccl_wsync_sharded_pgs = {}
    self._xccl_wsync_sharded_method = None

    _ep_d = getattr(self, "_expert_parallel_degree", 1)
    _shard_rank = getattr(self, "_shard_rank", None)
    if _ep_d <= 1 or _shard_rank is None:
        return

    xccl_port = getattr(self, "_vllm_xccl_port", 51217)
    tp_size = getattr(self, "_vllm_tp_size", 1)
    num_replicas = len(self._vllm_urls)
    world_size = 1 + num_replicas * tp_size

    import socket as _socket
    _xccl_host = (
        os.environ.get("TORCHTUNE_XCCL_HOST")
        or os.environ.get("MASTER_ADDR")
        or _socket.gethostname()
    )
    _is_xccl_leader = getattr(self, "_is_xccl_leader", self._is_shard_leader)
    _ws10_method = os.environ.get(
        "TORCHTUNE_EP_WSYNC_SHARDED_METHOD", "gloo")
    self._xccl_wsync_sharded_method = _ws10_method

    # Leader POSTs vLLM workers to construct receive-side sharded PGs.
    ws10_threads = []
    _ws10_post_errors = []
    if _is_xccl_leader:
        def _post_sharded(r_idx, url):
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={
                        "method": "init_xccl_sharded_pgs",
                        "args": [_xccl_host, xccl_port, world_size,
                                 int(_ep_d), _ws10_method],
                    },
                    timeout=120,
                )
                if r.status_code != 200:
                    _ws10_post_errors.append(
                        f"init_xccl_sharded_pgs failed ({url}): "
                        f"{r.status_code} {r.text}")
                    return
                result = r.json().get("results", [{}])
                first = result[0] if result else {}
                if isinstance(first, dict) and first.get("status") != "ok":
                    _ws10_post_errors.append(
                        f"init_xccl_sharded_pgs error ({url}): {first}")
            except Exception as e:
                _ws10_post_errors.append(str(e))

        ws10_threads = [
            threading.Thread(target=_post_sharded, args=(r_idx, url),
                             daemon=True)
            for r_idx, url in enumerate(self._vllm_urls)
        ]
        for t in ws10_threads:
            t.start()

    # Each EP shard rank R in [0..ep_d-1] joins one 2-rank PG PER vLLM replica
    # (prefix wsync_sharded_R{R}_rep{rep_idx}). The trainer is rank 0; each
    # replica's TP root joins as rank 1 of its own PG. The deferred broadcast
    # loop iterates over all replica PGs so every replica receives the data
    # and then fans out to its TP workers via the existing intra PG.
    # Other trainer ranks (e.g. dp_replicate>1 with shard_rank >= ep_d) skip.
    if 0 <= int(_shard_rank) < int(_ep_d):
        _ws10_R = int(_shard_rank)
        if _is_xccl_leader and getattr(self, "_xccl_store", None) is not None:
            _ws10_store = self._xccl_store
        else:
            _ws10_store = dist.TCPStore(
                host_name=_xccl_host,
                port=xccl_port,
                world_size=world_size,
                is_master=False,
                timeout=datetime.timedelta(seconds=120),
            )

        _ws10_pg_list = []
        for _rep_idx in range(int(num_replicas)):
            _ws10_prefix = f"wsync_sharded_R{_ws10_R}_rep{_rep_idx}"
            _ws10_prefixed = c10d.PrefixStore(_ws10_prefix, _ws10_store)
            if _ws10_method == "gloo":
                _ws10_pg = c10d.ProcessGroupGloo(
                    store=_ws10_prefixed, rank=0, size=2,
                )
            else:
                _ws10_opts = c10d.ProcessGroupXCCL.Options()
                _ws10_pg = c10d.ProcessGroupXCCL(
                    store=_ws10_prefixed, rank=0, size=2, options=_ws10_opts,
                )
            _ws10_pg_list.append(_ws10_pg)
            log.info(
                "Rank %d: WS10 sharded PG created (R=%d rep=%d, prefix=%s, method=%s)",
                self.rank, _ws10_R, _rep_idx, _ws10_prefix, _ws10_method,
            )
        self._xccl_wsync_sharded_pgs[_ws10_R] = _ws10_pg_list

        # Commit B: every trainer EP rank in [0..ep_d-1] is its own WS10
        # sender (broadcasts its local fused expert shard on its per-rank
        # cross PG). The sync-done event/counter state is normally only
        # created on rank 0 (vllm_backend._setup_vllm_server_mode) or on
        # ranks in the legacy WSYNC_SENDER_POOL. WS10 needs it on every
        # rank that will enter the sender path, otherwise non-rank-0 EP
        # ranks AttributeError at `_sync_done_event.clear()`.
        if not hasattr(self, "_sync_done_event"):
            self._sync_done_event = threading.Event()
            self._sync_done_event.set()
            self._sync_error = None
            self._sync_id_counter = 0
            self._pending_sync_id = None

    if _is_xccl_leader:
        for t in ws10_threads:
            t.join(timeout=120)
        if _ws10_post_errors:
            raise RuntimeError(
                f"WS10 sharded PG init failed on vLLM side: "
                f"{_ws10_post_errors}")
    torch.distributed.barrier()
    log.info(
        "Rank %d: WS10 sharded PG construction complete "
        "(ep_d=%d, num_replicas=%d, my_R_keys=%s)",
        self.rank, int(_ep_d), int(num_replicas),
        list(self._xccl_wsync_sharded_pgs.keys()),
    )

def _init_xccl_weight_sync(self) -> None:
    """Create a cross-process XCCL group between training rank 0 and vLLM worker(s).

    Called once on first weight sync. Training is rank 0 (TCPStore master),
    vLLM worker(s) join as rank 1..N via /collective_rpc POST.

    Ordering: TCPStore (master) first, then POST to vLLM in a background
    thread (vLLM enters PG constructor, blocks waiting for training), then
    training enters PG constructor — both sides unblock simultaneously.
    """
    import datetime
    import threading
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    import requests

    xccl_port = getattr(self, "_vllm_xccl_port", 51217)
    tp_size = getattr(self, "_vllm_tp_size", 1)
    num_replicas = len(self._vllm_urls)
    # Total ranks: 1 (training) + num_replicas * tp_size (all vLLM TP workers)
    # Replica r gets base_rank = 1 + r * tp_size, occupying ranks [base_rank, base_rank + tp_size)
    world_size = 1 + num_replicas * tp_size

    # Address that vLLM workers use to reach this TCPStore master.
    # TORCHTUNE_XCCL_HOST takes priority — set this to the training node's HSN
    # address when using --standalone (which overrides MASTER_ADDR to 127.0.0.1).
    import socket as _socket
    _xccl_host = (
        os.environ.get("TORCHTUNE_XCCL_HOST")
        or os.environ.get("MASTER_ADDR")
        or _socket.gethostname()
    )

    log.info(
        "Rank %d: initializing XCCL weight sync (host=%s, port=%d, world=%d, tp=%d, replicas=%d)",
        self.rank, _xccl_host, xccl_port, world_size, tp_size, num_replicas,
    )

    store = dist.TCPStore(
        host_name="0.0.0.0",
        port=xccl_port,
        world_size=world_size,
        is_master=True,
        timeout=datetime.timedelta(seconds=120),
        wait_for_workers=False,
    )
    self._xccl_store = store
    self._wsync_pg_gen = 0
    self._wsync_pg_reset_interval = int(os.environ.get("WSYNC_PG_RESET_INTERVAL", "0"))

    # POST to all vLLM replicas CONCURRENTLY — each POST blocks until that
    # replica's TP workers enter the XCCL PG constructor. All replicas must
    # join simultaneously (world_size barrier), so sequential posting would
    # deadlock: replica 0 waits for the full group while we never reach replica 1.
    use_two_hop = num_replicas > 0  # always use when there are cross-node vLLM workers
    self._xccl_two_hop = use_two_hop

    # WSYNC_TOPOLOGY selects the fanout shape:
    #   replica_fanout (default, legacy) — 1 cross PG PER replica (size=2 each).
    #     Training rank 0 broadcasts N times in parallel to N replicas. The
    #     "intra PG" is per-replica TP (size = tp_size_local), so for TP=1
    #     vLLM topologies (BioReason DP=12) it is a no-op and all 12 cross
    #     broadcasts contend for the same hsn0 NIC on rank 0's side.
    #   node_fanout (real 2-hop) — 1 cross PG total (size=2: training rank 0
    #     ↔ replica 0). Replica 0 then broadcasts to all other vLLM workers
    #     via a single intra-NODE PG of size=num_replicas*tp_size that all
    #     vLLM workers join. Cross-NIC traffic drops by num_replicas (12×
    #     for DP=12); intra-node fanout uses XeLink (~50 GB/s).
    topology = os.environ.get("WSYNC_TOPOLOGY", "replica_fanout")
    self._xccl_topology = topology
    # WSYNC_CROSS_METHOD selects the cross-PG backend:
    #   xccl_sendrecv — XCCL PG with send/recv (RDMA, ~8 GB/s, may hit CXI MR leak)
    #   gloo          — Gloo PG with broadcast (TCP, ~1.3 GB/s, no CXI leak)
    wsync_method = os.environ.get("WSYNC_CROSS_METHOD", "xccl_sendrecv")

    # node_fanout sizing: vLLM workers form a single intra PG of size
    # (num_replicas*tp_size); the worker-side init keys off this.
    intra_world = num_replicas * tp_size if topology == "node_fanout" else tp_size

    vllm_errors = []
    def _post_replica(r_idx, url):
        base_rank = 1 + r_idx * tp_size
        try:
            r = requests.post(
                f"{url}/collective_rpc",
                json={
                    "method": "init_xccl_communicator",
                    "args": [_xccl_host, xccl_port, world_size, base_rank,
                             use_two_hop, wsync_method, 0, topology, intra_world],
                },
                timeout=120,
            )
            if r.status_code != 200:
                vllm_errors.append(f"init_xccl_communicator failed ({url}): {r.status_code} {r.text}")
                return
            result = r.json().get("results", [{}])
            first = result[0] if result else {}
            if isinstance(first, dict) and first.get("status") != "ok":
                vllm_errors.append(f"init_xccl_communicator error ({url}): {first}")
        except Exception as e:
            vllm_errors.append(str(e))

    replica_threads = [
        threading.Thread(target=_post_replica, args=(r_idx, url), daemon=True)
        for r_idx, url in enumerate(self._vllm_urls)
    ]

    for t in replica_threads:
        t.start()

    # Training enters PG constructor — unblocks vLLM side when all replicas join.
    if use_two_hop:
        if topology == "node_fanout":
            # Single cross PG: training rank 0 ↔ replica 0 (vLLM rank 1).
            # Replica 0 fans out intra-node to ranks 2..N via the size-N intra PG.
            cross_prefixed = c10d.PrefixStore("wsync_cross_0", store)
            if wsync_method == "gloo":
                pg = c10d.ProcessGroupGloo(
                    store=cross_prefixed, rank=0, size=2,
                )
            else:
                opts = c10d.ProcessGroupXCCL.Options()
                pg = c10d.ProcessGroupXCCL(
                    store=cross_prefixed, rank=0, size=2, options=opts,
                )
            self._xccl_wsync_pgs = [pg]
            self._xccl_wsync_pg = pg
            self._wsync_cross_method = wsync_method
            log.info("Rank %d: node_fanout cross PG created (1 PG, method=%s, "
                     "intra_world=%d on vLLM_NODE)",
                     self.rank, wsync_method, intra_world)
        else:
            # replica_fanout (legacy): one cross-PG per replica.
            self._xccl_wsync_pgs = []
            for r_idx in range(num_replicas):
                cross_prefixed = c10d.PrefixStore(f"wsync_cross_{r_idx}", store)
                if wsync_method == "gloo":
                    pg = c10d.ProcessGroupGloo(
                        store=cross_prefixed, rank=0, size=2,
                    )
                else:
                    opts = c10d.ProcessGroupXCCL.Options()
                    pg = c10d.ProcessGroupXCCL(
                        store=cross_prefixed, rank=0, size=2, options=opts,
                    )
                self._xccl_wsync_pgs.append(pg)
            self._xccl_wsync_pg = self._xccl_wsync_pgs[0]
            self._wsync_cross_method = wsync_method
            log.info("Rank %d: %d cross PG(s) created (method=%s)",
                     self.rank, num_replicas, wsync_method)
    else:
        opts = c10d.ProcessGroupXCCL.Options()
        prefixed = c10d.PrefixStore("wsync", store)
        self._xccl_wsync_pg = c10d.ProcessGroupXCCL(
            store=prefixed, rank=0, size=world_size, options=opts,
        )
        self._wsync_cross_method = "xccl_broadcast"

    for t in replica_threads:
        t.join(timeout=120)
    if vllm_errors:
        raise RuntimeError(f"vLLM XCCL init failed: {vllm_errors}")

    log.info("Rank %d: XCCL weight sync communicator ready", self.rank)


# ---------------------------------------------------------------------------
# WS10 helpers — sharded vLLM MoE sync (Layer 3: sender staging)
# ---------------------------------------------------------------------------
#
# Pure-CPU functions: no torch.distributed, no XPU. They take a per-rank
# local sharded state-dict and return ``(cpu_batches, tensors_meta)`` ready
# to be broadcast over a per-rank cross-PG. Manifest unification across
# trainer EP ranks is also pure-CPU and can be exercised without a PG.
#
# Pin-down tests live in
# ``tests/torchtune/dev/rl/test_sharded_vllm_moe_sync_equivalence.py``.

# Module-level so tests and the deferred-broadcast loop share one definition
# of the canonical manifest order. Keep in sync with
# ``_canonical_manifest_sort_key`` in the pin-down test.
_WS10_FUSED_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
)


def _ws10_sort_key(entry: dict) -> tuple:
    """Canonical sort key for WS10 manifest entries.

    Order: (layer_idx asc, w13 before w2, trainer_ep_rank asc).
    Non-fused entries (norms, attention, embeddings) sort after all fused
    entries with stable name-based tiebreak. The trainer + receiver both
    sort with this key so the broadcast loop is lockstep across ranks.

    Note: this is the legacy "AllGather-then-broadcast" ordering. For
    Commit B's per-rank broadcast loop use ``_ws10_sort_key_by_rank``,
    which groups entries by trainer_ep_rank so the receiver can iterate
    one PG at a time without interleaving.
    """
    name = entry["name"]
    m = _WS10_FUSED_RE.match(name)
    if m:
        layer_idx = int(m.group(1))
        kind_rank = 0 if m.group(2) == "w13" else 1
    else:
        layer_idx, kind_rank = 10**9, 9
    R = int(entry.get("trainer_ep_rank", -1))
    return (layer_idx, kind_rank, R, name)


def _ws10_sort_key_by_rank(entry: dict) -> tuple:
    """Per-rank-grouped sort key for WS10 Commit B broadcast.

    Order: (trainer_ep_rank asc, layer_idx asc, w13 before w2, name).
    All entries owned by trainer rank R come together so the receiver
    can drive ``_xccl_sharded_cross_pgs[R].broadcast(...)`` for the
    contiguous run, then advance to ``R+1``'s PG. Non-expert entries
    (norms, attention, embeddings, router) are sent only by trainer
    rank 0 and sort with R=0 AFTER its fused entries (kind_rank=9).
    """
    name = entry["name"]
    m = _WS10_FUSED_RE.match(name)
    if m:
        layer_idx = int(m.group(1))
        kind_rank = 0 if m.group(2) == "w13" else 1
    else:
        # Non-expert: layer_idx large + kind_rank=9 puts them at the
        # end of their rank's run. In Commit B these only ever carry
        # trainer_ep_rank=0 (or default 0 when missing), so they trail
        # rank 0's fused experts.
        layer_idx, kind_rank = 10**9, 9
    R = int(entry.get("trainer_ep_rank", 0))
    return (R, layer_idx, kind_rank, name)


def _ws10_build_local_payload(
    local_experts: dict,
    ep_rank: int,
    ep_degree: int,
    batch_max_numel: int,
):
    """Build per-rank cpu_batches + tensors_meta for the WS10 sharded path.

    Args:
        local_experts: ``{hf_name: local_shard_cpu_tensor}`` for THIS rank.
            ``hf_name`` is post-rename
            (e.g. ``model.layers.7.mlp.experts.w13_weight``). Each tensor is
            the rank-local fused shard on CPU, bf16, contiguous.
        ep_rank: Trainer EP rank of the producing process (0..ep_degree-1).
        ep_degree: Total number of trainer EP ranks.
        batch_max_numel: Max elements per concatenated CPU broadcast batch.

    Returns:
        (cpu_batches, tensors_meta) — tensor list of bf16 1-D CPU concats and
        the matching list of dicts with ``trainer_ep_rank``/``ep_degree``/
        ``n_local`` tags. Order is the canonical WS10 sort applied locally;
        the batched concat respects that order.
    """
    if ep_degree <= 1:
        raise ValueError(
            f"_ws10_build_local_payload requires ep_degree > 1, got {ep_degree}")
    if not (0 <= ep_rank < ep_degree):
        raise ValueError(
            f"ep_rank {ep_rank} out of range [0, {ep_degree})")

    local_meta = []
    for hf_name, t in local_experts.items():
        m = _WS10_FUSED_RE.match(hf_name)
        if m is None:
            raise ValueError(
                f"_ws10_build_local_payload: unexpected non-fused name "
                f"{hf_name!r}; this helper only stages fused expert shards. "
                "Non-expert params follow the existing path.")
        if t.dim() < 1:
            raise ValueError(
                f"local shard {hf_name!r} must be at least 1-D, got shape {tuple(t.shape)}")
        n_local = int(t.shape[0])
        local_meta.append({
            "name": hf_name,
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "trainer_ep_rank": int(ep_rank),
            "ep_degree": int(ep_degree),
            "n_local": n_local,
            "_tensor": t,
        })
    local_meta.sort(key=_ws10_sort_key)

    cpu_batches: list = []
    cur_parts: list = []
    cur_numel = 0
    for entry in local_meta:
        t = entry["_tensor"]
        n = int(t.numel())
        if cur_numel > 0 and cur_numel + n > batch_max_numel:
            cpu_batches.append(torch.cat(cur_parts))
            cur_parts = []
            cur_numel = 0
        cur_parts.append(t.flatten().contiguous())
        cur_numel += n
    if cur_parts:
        cpu_batches.append(torch.cat(cur_parts))

    tensors_meta = [
        {k: v for k, v in e.items() if k != "_tensor"} for e in local_meta
    ]
    return cpu_batches, tensors_meta


def _ws10_unify_manifests(per_rank_metas: list, sort_key=None) -> list:
    """Collapse per-rank manifests into a single canonically-ordered list.

    Args:
        per_rank_metas: ``[meta_rank_0, meta_rank_1, ...]`` — output of
            ``all_gather_object`` of the ``tensors_meta`` returned by
            ``_ws10_build_local_payload``. Each entry must already carry
            ``trainer_ep_rank``, ``ep_degree``, ``n_local``.
        sort_key: Optional sort callable. Defaults to ``_ws10_sort_key``
            (legacy by-layer order). Pass ``_ws10_sort_key_by_rank`` for
            Commit B's per-rank broadcast loop ordering.

    Returns:
        Single sorted list. Receiver dispatcher drives broadcast order from
        this list; the sender's deferred-broadcast loop drives the per-rank
        ``cpu_batches`` send order from the same list (filtered by
        ``trainer_ep_rank == self._ep_rank``).
    """
    if not per_rank_metas:
        return []
    ep_degrees = {
        int(e["ep_degree"])
        for meta in per_rank_metas
        for e in meta
        if "ep_degree" in e
    }
    if len(ep_degrees) > 1:
        raise ValueError(
            f"_ws10_unify_manifests: heterogeneous ep_degree across ranks: "
            f"{sorted(ep_degrees)}")
    if ep_degrees:
        ep_d = next(iter(ep_degrees))
        if len(per_rank_metas) != ep_d:
            raise ValueError(
                f"_ws10_unify_manifests: got {len(per_rank_metas)} per-rank "
                f"manifests but ep_degree={ep_d}")

    flat = [e for meta in per_rank_metas for e in meta]
    flat.sort(key=sort_key if sort_key is not None else _ws10_sort_key)
    return flat



def _xccl_gather_fsdp1(self, _xccl_accept_and_rename, t0):
    """FSDP1 XCCL weight-sync gather branch, extracted verbatim from
    _sync_weights_to_vllm_xccl (no behavior change). Builds flat_gpu /
    tensors_meta on the XCCL leader. _xccl_accept_and_rename is the
    per-call rename closure; t0 is the wsync-start perf_counter stamp."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
    hf_state_dict = {}
    # WS3.5: only the XCCL leader (global rank 0) keeps and broadcasts.
    _is_xccl_leader = getattr(self, "_is_xccl_leader", self._is_shard_leader)
    # Llama-family Q/K un-permute mirrors the FSDP2 path (line ~2987).
    # Engages once with a log line for runtime confirmation.
    _unperm_needed_x = _needs_qk_unpermute(self)
    if (
        _unperm_needed_x
        and _is_xccl_leader
        and not getattr(self, "_qk_unperm_xccl_fsdp1_engaged_logged", False)
    ):
        log.info(
            "Rank %d: xccl FSDP1 wsync QK un-permute ENGAGED "
            "(n_heads=%s n_kv_heads=%s head_dim=%s)",
            self.rank,
            getattr(self, "_model_num_heads", None),
            getattr(self, "_model_num_kv_heads", None),
            getattr(self, "_model_head_dim", None),
        )
        self._qk_unperm_xccl_fsdp1_engaged_logged = True

    # BioReason PEFT-LoRA: vLLM runs a vanilla (adapter-less) Qwen3, so it must
    # receive the merged W_eff = base + sum_a scale_a*(B_a@A_a). We summon full
    # params on every rank, then on the leader form W_eff WITHOUT mutating the
    # frozen base: read base_layer.weight via the PEFT-aware vllm_param_iter
    # (clean HF names; lora_A/lora_B skipped) and add the non-mutating
    # get_delta_weight (fp32) from lora_delta_map(). This avoids the bf16
    # round-trip drift that in-place merge_adapter()/unmerge_adapter() would
    # accumulate INTO the frozen base across many steps (~5e-3/20 cycles,
    # compounding). The .clone()/contiguous() is load-bearing: summon_full_params
    # frees its gather buffer on context exit, so the leader's kept tensors must
    # own their storage. No vLLM co-tenancy here (server mode, vLLM on a separate
    # node) → no summon/colocate wedge.
    _lora_merge = (
        getattr(self, "_is_bioreason", False)
        and getattr(self._model, "_has_lora", False)
    )
    if _lora_merge:
        with torch.no_grad(), FSDP.summon_full_params(
            self._model, writeback=False, rank0_only=False
        ):
            if _is_xccl_leader:
                delta_map = self._model.lora_delta_map()  # {hf_name: fp32 delta}
                for hf_name, param in self._model.vllm_param_iter():
                    weight = param.detach().to(self._device)
                    delta = delta_map.get(hf_name)
                    if delta is not None:
                        # fp32 accumulate, recast to the base dtype (bf16) once.
                        weight = (weight.float() + delta.to(self._device).float()).to(param.dtype)
                    if _unperm_needed_x:
                        weight = _maybe_unpermute_qk(self, hf_name, weight)
                    hf_state_dict[hf_name] = weight.contiguous().clone()
    else:
        # FSDP1: state_dict() handles gathering; result is on CPU already
        with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
            full_sd = self._model.state_dict()
        if _is_xccl_leader:
            for param_name, param in full_sd.items():
                hf_name = _xccl_accept_and_rename(param_name)
                if hf_name is None:
                    continue
                weight = param.to(self._device)
                if _unperm_needed_x:
                    weight = _maybe_unpermute_qk(self, hf_name, weight)
                hf_state_dict[hf_name] = weight
        del full_sd

    if not self._production_mode:
        torch.distributed.barrier()
    t_gather = time.perf_counter() - t0

    # Sync-event bookkeeping (mirror the FSDP2 path): the leader clears the
    # done-event and bumps the pending sync id so _wait_for_sync_complete /
    # _start_deferred_broadcast track this round. Without this the deferred
    # broadcast machinery never fires. (Regression fix 2026-06-18: the f60efefc
    # refactor extracted this gather but dropped the staging+broadcast tail for
    # the FSDP1 branch, so vLLM was never updated → frozen-generator drift →
    # ∇logp blowup → NaN on FSDP1+xccl server runs, e.g. 2N 11+1 HSDP.)
    import json as _json
    _BATCH_MAX_NUMEL = 512 * 1024 * 1024  # 1 GiB bf16, matches FSDP2 path
    if _is_xccl_leader:
        self._sync_done_event.clear()
        self._sync_error = None
        self._sync_id_counter = getattr(self, "_sync_id_counter", 0) + 1
        self._pending_sync_id = self._sync_id_counter

    if _is_xccl_leader:
        # Build the manifest AND the CPU broadcast batches using the SAME greedy
        # split the receiver mirrors (vllm_weight_sync_worker.receive_weights_xccl
        # _streaming): walk params in order, flush a batch when adding the next
        # param would exceed batch_max_numel; a single param larger than the limit
        # forms its own batch. The sender must emit exactly one broadcast per batch
        # or the receiver's recv count mismatches → collective_rpc cancelled →
        # vLLM EngineDeadError (observed 2026-06-18 when the whole model was sent
        # as one mega-batch).
        tensors_meta = []
        total_elements = 0
        cpu_batches = []          # list[CPU flat bf16 tensor], one per broadcast
        _cur_parts = []           # GPU flat chunks accumulating into the current batch
        _cur_numel = 0
        for hf_name, tensor in hf_state_dict.items():
            numel = tensor.numel()
            tensors_meta.append({"name": hf_name, "shape": list(tensor.shape),
                                 "dtype": str(tensor.dtype), "numel": numel})
            total_elements += numel
            # Flush before adding if this param would overflow the current batch
            # (but never flush an empty batch — a >limit param goes alone).
            if _cur_numel > 0 and _cur_numel + numel > _BATCH_MAX_NUMEL:
                cpu_batches.append(torch.cat(_cur_parts).to("cpu"))
                _cur_parts = []
                _cur_numel = 0
            _cur_parts.append(tensor.to(torch.bfloat16).flatten())
            _cur_numel += numel
        if _cur_parts:
            cpu_batches.append(torch.cat(_cur_parts).to("cpu"))
        del _cur_parts, hf_state_dict

        meta_json = _json.dumps({
            "tensors": tensors_meta,
            "batch_max_numel": _BATCH_MAX_NUMEL,
            "sender_index": 0,
        })
        _cross_pgs = getattr(self, "_xccl_wsync_pgs", None) or [self._xccl_wsync_pg]
        self._deferred_broadcast_args = (
            cpu_batches, meta_json,
            tensors_meta, t0, self._device,
            _cross_pgs, self._vllm_clients,
            self._vllm_urls,
        )
        log.info(
            "Rank %d: FSDP1 xccl gather staged %d params (%.2f GiB) in %d batches "
            "→ deferred broadcast armed (regression fix for f60efefc dropped-broadcast).",
            self.rank, len(tensors_meta), total_elements * 2 / 1024**3, len(cpu_batches),
        )

    self._wsync_round = getattr(self, "_wsync_round", 0) + 1

def _xccl_gather_and_stage_fsdp2(self, is_active, pool, _xccl_accept_and_rename, t0):
    """FSDP2/EP XCCL weight-sync gather+stage branch, extracted verbatim
    from _sync_weights_to_vllm_xccl (no behavior change). Gathers sharded
    params, stages to CPU, and sets self._deferred_broadcast_args for the
    post-gen broadcast. is_active/pool select the sender; t0 is the start
    stamp; _xccl_accept_and_rename is the rename closure."""
    # These were module-imported in the dispatcher prologue before this branch
    # was extracted; the FSDP2 path uses json.dumps for the broadcast manifest.
    import json
    import requests  # noqa: F401  (kept for scope parity with the dispatcher)
    # FSDP2 weight sync: gather sharded params and broadcast to vLLM.
    # Two sub-modes selected by TORCHTUNE_XCCL_BATCHED_AG:
    #
    #   0 (default): per-param full_tensor() + batched XCCL broadcast
    #      AllGather per param: <0.12ms for 3B, ~0.8ms for 32B (negligible).
    #      Real bottleneck: XCCL broadcast bandwidth (1.7 GB/s at 12 receivers
    #      = 35.9s for 61 GiB). Tests CD/CE/CF confirm this floor.
    #
    #   1 (BROKEN): batched all_gather_into_tensor() + reconstruct + broadcast
    #      Saves ~0.2s for 3B (<1% for 32B). Leaves FSDP2 shard state
    #      inconsistent → checkpoint save hangs after sync. Do not use.
    _USE_BATCHED_AG = os.environ.get("TORCHTUNE_XCCL_BATCHED_AG", "0") == "1"
    _USE_PINNED_BUF = os.environ.get("TORCHTUNE_PINNED_CPU_BUF", "0") == "1"
    _USE_D2H_STREAM = os.environ.get("TORCHTUNE_D2H_STREAM", "0") == "1"
    if _USE_BATCHED_AG:
        log.warning(
            "TORCHTUNE_XCCL_BATCHED_AG=1 is BROKEN: it leaves FSDP2 shard state "
            "inconsistent, causing the post-training checkpoint save to hang. "
            "Savings are <1%% for 32B (0.2s vs 38s floor). Do not use."
        )
    import threading as _threading

    _BATCH_MAX_NUMEL = 512 * 1024 * 1024  # 512M bf16 elements = 1 GiB
    # WS10 override (2026-07-22): the legacy full-reship path's ~49s wsync
    # is bandwidth-bound on the CROSS-NODE XCCL send (2.5 GB/s), so 1 GiB
    # batches are a reasonable pacing unit there. WS10's bottleneck is
    # different: the vLLM-side INTRA-node fanout (XeLink) pays a fixed
    # Python/collective-call overhead per batch (60 batches/round measured
    # at ~75-87s total_recv despite each individual gloo recv completing in
    # <1s — see project_ws10_ep8_hang_20260721.md). Raising the batch size
    # for WS10 specifically cuts call count without touching the validated
    # legacy path's batch size. Independent env var so a regression here
    # can't silently change the legacy path's behavior.
    _WS10_BATCH_MAX_NUMEL = int(os.environ.get(
        "TORCHTUNE_EP_WSYNC_SHARDED_BATCH_NUMEL", str(_BATCH_MAX_NUMEL)))
    # HARD CEILING (2026-07-22, found via job 8683165 crash): gloo's TCP
    # transport preamble uses a 32-bit signed length field for the byte
    # count of each broadcast (see gloo/transport/tcp/pair.cc). A batch of
    # exactly 2**31 bf16 ELEMENTS = 2**32 BYTES overflows it, corrupting the
    # wire preamble and crashing the vLLM worker with
    # "gloo::EnforceNotMet ... op.preamble.length <= op.nbytes" instead of a
    # clean error. Enforce elements*2 < 2**31 bytes (INT32_MAX) with margin.
    _WS10_MAX_SAFE_ELEMENTS = (2**31 - 1) // 2  # ~1.07B elements = ~2 GiB bf16
    if _WS10_BATCH_MAX_NUMEL >= _WS10_MAX_SAFE_ELEMENTS:
        raise ValueError(
            f"TORCHTUNE_EP_WSYNC_SHARDED_BATCH_NUMEL={_WS10_BATCH_MAX_NUMEL} "
            f"would produce broadcast batches >= {_WS10_MAX_SAFE_ELEMENTS} "
            "bf16 elements (~2 GiB bytes), overflowing gloo's 32-bit TCP "
            "preamble length field and crashing the vLLM worker with a "
            "garbled 'preamble.length <= nbytes' EnforceNotMet error rather "
            "than a clean failure. Use a smaller value."
        )

    sharded_sd = self._model.state_dict()

    if is_active:
        self._sync_done_event.clear()
        self._sync_error = None
        self._sync_id_counter += 1
        self._pending_sync_id = self._sync_id_counter

    # Detect MoE model for expert fusing (used by Mode 0).
    from torchtune.training.checkpointing._utils import ModelType as _MT
    _is_moe = getattr(self._checkpointer, '_model_type', None) == _MT.QWEN3_MOE if self._checkpointer is not None else False

    t_ag = 0.0
    t_bcast = 0.0
    n_params = 0
    n_batches = 0

    if _USE_BATCHED_AG:
        # Mode 1: batched AllGather — reduces AllGather calls from N_params to
        # N_batches by manually gathering local shards with all_gather_into_tensor().
        # All training ranks contribute local shards; rank 0 reconstructs full
        # params from the interleaved output then broadcasts to vLLM.
        _training_ws = torch.distributed.get_world_size()

        # local batch: local shards from THIS rank, accumulated until threshold
        _lb_parts: list = []      # list of bf16 contiguous local shard tensors
        _lb_numel = 0             # total local numel accumulated so far
        _lb_metas: list = []      # (local_numel, full_numel) per param in batch
        _batch_full_numel = 0     # full numel accumulated (for threshold check)

        def _flush_batched_ag():
            nonlocal t_ag, t_bcast, n_batches
            local_flat = torch.cat([p.flatten() for p in _lb_parts])
            # AllGather: shape [world_size * _lb_numel], layout:
            #   [rank0_local | rank1_local | ... | rank_{W-1}_local]
            full_batch = torch.empty(
                _training_ws * _lb_numel,
                dtype=torch.bfloat16, device=self._device,
            )
            tb0 = time.perf_counter()
            torch.distributed.all_gather_into_tensor(full_batch, local_flat)
            t_ag += time.perf_counter() - tb0
            del local_flat

            # WS3.5: only XCCL leader broadcasts to vLLM.
            if getattr(self, "_is_xccl_leader", self._is_shard_leader):
                # Reconstruct full params from interleaved AllGather output.
                # For param_i with local_numel=lnu at local_offset=cum_lnu:
                #   full_param = cat(full_batch[r*_lb_numel+cum : r*_lb_numel+cum+lnu]
                #                    for r in range(_training_ws))[:full_numel]
                bcast_parts = []
                cum_lnu = 0
                for lnu, fnu in _lb_metas:
                    slices = [
                        full_batch[r * _lb_numel + cum_lnu:
                                   r * _lb_numel + cum_lnu + lnu]
                        for r in range(_training_ws)
                    ]
                    bcast_parts.append(torch.cat(slices)[:fnu])
                    cum_lnu += lnu
                bcast_flat = torch.cat([p.flatten() for p in bcast_parts])
                bcast_parts.clear()
                del full_batch

                tb1 = time.perf_counter()
                _bcast_pgs = getattr(self, '_xccl_wsync_pgs', [self._xccl_wsync_pg])
                _bworks = [_bpg.broadcast(bcast_flat, root=0) for _bpg in _bcast_pgs]
                for _bw in _bworks:
                    _bw.wait()
                t_bcast += time.perf_counter() - tb1
                del bcast_flat
            else:
                del full_batch
            n_batches += 1

        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                param = param.to(self._device)
            fnu = param.numel()
            if hasattr(param, "_local_tensor"):
                local_shard = param._local_tensor.to(torch.bfloat16).contiguous()
            else:
                local_shard = param.to(torch.bfloat16).contiguous()
            lnu = local_shard.numel()

            # Flush if adding this param would exceed batch threshold
            if _batch_full_numel > 0 and _batch_full_numel + fnu > _BATCH_MAX_NUMEL:
                _flush_batched_ag()
                _lb_parts.clear()
                _lb_metas.clear()
                _lb_numel = 0
                _batch_full_numel = 0

            _lb_parts.append(local_shard)
            _lb_numel += lnu
            _lb_metas.append((lnu, fnu))
            _batch_full_numel += fnu
            n_params += 1
            del param

        if _lb_parts:
            _flush_batched_ag()

    else:
        # Mode 0 (default): per-param full_tensor() + ASYNC batched XCCL broadcast.
        #
        # Phase A (synchronous, all ranks): full_tensor() loop gathers each param.
        #   Shard leader stages each batch to CPU instead of broadcasting inline.
        #   AllGather per param is negligible (~0.1s total for 32B).
        #
        # Phase B (async, shard leader only): background thread copies CPU-staged
        #   batches back to GPU and broadcasts via XCCL. Overlaps with next step's
        #   vLLM generation (~15s), hiding the 7.7s broadcast cost entirely.
        #
        # 2-hop mode: training rank 0 → vLLM rank 1 (Slingshot), then
        #   rank 1 → ranks 2-12 (XeLink). ~3s vs 38s flat.

        # Streaming gather: full_tensor() each param, immediately copy to
        # CPU, build batches from CPU tensors. Keeps GPU peak at ~1 GiB
        # (one batch) instead of ~60 GiB (full model). Critical for 32B+
        # models where the full model exceeds single-tile 64 GiB capacity.
        _ep_deg = getattr(self, '_expert_parallel_degree', 1)
        if is_active:
            tensors_meta = []
            cpu_batches: list = []
            batch_numel = 0

            if _is_moe:
                hf_state_dict = {}

            if _ep_deg > 1 and _is_moe:
                # Post EP all-gather, fused expert tensors are ep_degree× larger than
                # the pre-allocated pinned buffer (sized from local 16-expert shards).
                # Disable pinned buf for EP MoE to avoid overflow and flush-path mixing.
                _USE_PINNED_BUF = False
            if not _USE_PINNED_BUF:
                batch_parts_cpu: list = []

        t_ft = 0.0
        t_cast = 0.0
        t_d2h = 0.0

        if _USE_PINNED_BUF and is_active:
            if not hasattr(self, '_pinned_cpu_buf') or self._pinned_cpu_buf is None:
                _total_numel = 0
                for _scan_name, _scan_p in sharded_sd.items():
                    _total_numel += _scan_p.numel()
                buf = torch.empty(_total_numel, dtype=torch.bfloat16)
                _pinned = False
                try:
                    buf = buf.pin_memory()
                    _pinned = True
                except Exception:
                    pass
                self._pinned_cpu_buf = buf
                self._cpu_buf_is_pinned = _pinned
                log.info(
                    "Rank %d: pre-allocated CPU buffer: %d elements (%.2f GiB), pinned=%s",
                    self.rank, _total_numel, _total_numel * 2 / 1024**3, _pinned,
                )
            if _USE_D2H_STREAM and not hasattr(self, '_d2h_stream'):
                try:
                    self._d2h_stream = torch.xpu.Stream(device=self._device)
                    log.info("Rank %d: D2H XPU stream created", self.rank)
                except Exception as _e:
                    self._d2h_stream = None
                    log.info("Rank %d: D2H stream not available: %s", self.rank, _e)
            _buf_offset = 0
            _batch_start = 0

        # EP expert streaming: all-gather EP-sharded slices across shard group,
        # fuse per layer immediately to avoid accumulating 128-expert tensors on GPU.
        # With ep_degree=8, each rank holds 16 experts; full model needs 128.
        # Streaming fuse (gate+up+down → w13/w2 per layer) keeps GPU peak at ~1.4 GiB.
        # _ep_deg already computed above (before pinned-buf allocation).
        _ep_expert_re = re.compile(r"\.mlp\.experts\.(gate_proj|up_proj|down_proj)")
        _ep_layer_re = re.compile(r"layers\.(\d+)\.")
        if is_active and _ep_deg > 1 and _is_moe:
            _ep_stream_buf: dict = {}  # layer_idx -> {proj -> bf16 tensor}

        # WS6 opt-in: batch the 3 expert projections of each layer into a
        # single all_gather_into_tensor on _shard_pg. Reduces wsync_gather
        # collective count from 3*L to L (e.g. 144 → 48 for Qwen3-30B-A3B
        # EP=16). Bit-exact equivalent — see
        # tests/torchtune/dev/rl/test_ep_wsync_layer_batch_equivalence.py.
        _LAYER_BATCH_AG = os.environ.get(
            "TORCHTUNE_EP_WSYNC_LAYER_BATCH", "0") == "1"
        _layer_batch_active = (
            _LAYER_BATCH_AG and _ep_deg > 1 and _is_moe
            and getattr(self, '_shard_pg', None) is not None
        )
        # Pending local shards keyed by layer_id (string). Populated on
        # ALL ranks (both is_active and others) — the AG is a collective.
        _lb_pending: dict = {} if _layer_batch_active else None
        _lb_logged_engaged = False

        # WS7 opt-in: replace AllGather with gather(dst=active_rank). Only
        # the is_active rank consumes the full expert tensor; the other
        # _ep_deg-1 ranks discard. AllGather pays N-way fan-out for 1-way
        # consumption; gather sends N-1 shards into the root NIC for the
        # same final bytes. Bit-exact equivalent — see
        # tests/torchtune/dev/rl/test_ep_wsync_gather_root_equivalence.py.
        _GATHER_ROOT = os.environ.get(
            "TORCHTUNE_EP_WSYNC_GATHER_ROOT", "0") == "1"
        _gather_root_active = (
            _GATHER_ROOT and _ep_deg > 1 and _is_moe
            and getattr(self, '_shard_pg', None) is not None
        )
        _gr_active_shard_rank = 0  # is_active rank == rank 0 of _shard_pg
        _gr_logged_engaged = False

        # WS8 opt-in: cast expert shards to fp8 (E4M3) with per-output-row
        # scales before the per-projection AllGather, AllGather fp8+scale
        # only, and decompress on the active rank back to bf16. Cuts wire
        # bytes ~2× on _shard_pg (the ~70s wsync_gather floor at EP=16).
        # Numerics: per-element err bounded by row_amax/4 (E4M3 has 3
        # mantissa bits + per-row scaling). NOT bit-exact — do not enable
        # this on workloads where rollouts are sensitive to weight noise
        # without first running a KL-drift check.
        #
        # Scope: ONLY expert gate_proj/up_proj/down_proj. Norms, attention,
        # router, embedding stay bf16 (per MoE_status_feedback.md §10).
        # Mutually exclusive with WS6/WS7 below — they exit early on the
        # expert path so this branch only triggers when LAYER_BATCH and
        # GATHER_ROOT are both off.
        #
        # See tests/torchtune/dev/rl/test_ep_wsync_fp8_wire_equivalence.py
        # for the round-trip + unshuffle pin-down.
        _FP8_WIRE = os.environ.get(
            "TORCHTUNE_EP_WSYNC_FP8_WIRE", "0") == "1"
        _fp8_wire_active = (
            _FP8_WIRE and _ep_deg > 1 and _is_moe
            and getattr(self, '_shard_pg', None) is not None
            and not _layer_batch_active and not _gather_root_active
            and hasattr(torch, 'float8_e4m3fn')
        )

        # WS10 Commit B (opt-in): per-rank sharded broadcast.
        # Each trainer EP rank R in [0..ep_d-1] broadcasts its OWN local
        # expert shard direct to the vLLM root over a per-rank 2-rank
        # cross-PG (`_xccl_wsync_sharded_pgs[R]`). The receiver assembles
        # the global tensor via FusedMoE.expert_map. Skips the
        # `_shard_pg` AllGather entirely on the expert path. Non-expert
        # params (norms, attention, embedding, router) are sent only by
        # rank 0 (no `trainer_ep_rank` tag → routed via R=0's PG on
        # the receiver). Mutually exclusive with WS6/WS7/WS8.
        # See docs/reports/MoE_EP_status_ws8_ws10_design.md §"WS10 Phase B+C".
        _SHARDED_WIRE = os.environ.get(
            "TORCHTUNE_EP_WSYNC_SHARDED", "0") == "1"
        _ws10_pgs_dict = getattr(self, "_xccl_wsync_sharded_pgs", {}) or {}
        _shard_rank_int = int(getattr(self, "_shard_rank", -1))
        _sharded_active = (
            _SHARDED_WIRE and _ep_deg > 1 and _is_moe
            and getattr(self, '_shard_pg', None) is not None
            and 0 <= _shard_rank_int < int(_ep_deg)
            and _shard_rank_int in _ws10_pgs_dict
            and not _layer_batch_active
            and not _gather_root_active
            and not _fp8_wire_active
            # HSDP+EP would split trainer ranks into [0..ep_d-1]
            # members and non-members; the non-members would fall
            # through to the legacy AllGather path while the EP
            # members early-return → deadlock. Single-replica EP only.
            and getattr(self, '_dp_replicate', 1) <= 1
        )
        if (_SHARDED_WIRE and not _sharded_active and self.rank == 0
                and not getattr(self, "_ws10_skip_warned", False)):
            if _layer_batch_active or _gather_root_active or _fp8_wire_active:
                log.warning(
                    "Rank %d: TORCHTUNE_EP_WSYNC_SHARDED=1 requested but "
                    "skipping (mutually exclusive with WS6=%s WS7=%s WS8=%s)",
                    self.rank, _layer_batch_active,
                    _gather_root_active, _fp8_wire_active,
                )
            else:
                log.warning(
                    "Rank %d: TORCHTUNE_EP_WSYNC_SHARDED=1 requested but "
                    "skipping (ep_deg=%d, is_moe=%s, _shard_pg=%s, "
                    "shard_rank=%d, ws10_pgs=%s)",
                    self.rank, _ep_deg, _is_moe,
                    getattr(self, '_shard_pg', None) is not None,
                    _shard_rank_int, sorted(_ws10_pgs_dict.keys()),
                )
            self._ws10_skip_warned = True
        if _sharded_active and self.rank == 0 and not getattr(
                self, "_ws10_engaged_warned", False):
            log.info(
                "Rank %d: WS10 sharded broadcast ENGAGED "
                "(ep_d=%d, sharded_method=%s, my_R=%d)",
                self.rank, _ep_deg,
                getattr(self, "_xccl_wsync_sharded_method", None),
                _shard_rank_int,
            )
            self._ws10_engaged_warned = True
        if _FP8_WIRE and not _fp8_wire_active:
            log.info(
                "Rank %d: TORCHTUNE_EP_WSYNC_FP8_WIRE=1 requested but "
                "skipping (ep_deg=%d, is_moe=%s, _shard_pg=%s, "
                "layer_batch=%s, gather_root=%s, e4m3=%s)",
                self.rank, _ep_deg, _is_moe,
                getattr(self, '_shard_pg', None) is not None,
                _layer_batch_active, _gather_root_active,
                hasattr(torch, 'float8_e4m3fn'),
            )
        _fp8_logged_engaged = False

        if _sharded_active:
            # WS10 Commit B sender path. ALL trainer ranks in [0..ep_d-1]
            # participate (the non-expert .full_tensor() calls below are
            # collectives on _shard_pg). Rank 0 uniquely stages the
            # non-expert tail and POSTs the manifest.
            _ws10_R = _shard_rank_int
            _ws10_t0 = time.perf_counter()
            _local_experts: dict = {}      # hf_name -> CPU bf16 tensor
            _ws10_pending_proj: dict = {}  # layer_idx -> {gate/up/down: local_tensor (bf16, CPU)}
            _ws10_non_expert_meta: list = []
            _ws10_non_expert_cpu_parts: list = []
            _ws10_non_expert_local_count = 0
            _ws10_non_expert_full_count = 0

            # First pass: walk sharded_sd. Expert params are kept local
            # (no _shard_pg AllGather). Non-expert params still need
            # global tensors (small) — call full_tensor() on ALL ranks
            # (collective), but only rank 0 keeps the result.
            for param_name, param in sharded_sd.items():
                if param.is_cpu:
                    param = param.to(self._device)

                _cpn = param_name.replace(
                    "_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", "")
                _is_expert = bool(_ep_expert_re.search(_cpn))

                if _is_expert:
                    # Take the local shard ONLY — no AllGather over
                    # _shard_pg. Each EP rank already owns its
                    # interleaved n_local experts for this projection.
                    if hasattr(param, "_local_tensor"):
                        _loc = param._local_tensor
                    else:
                        _loc = param
                    _loc_bf16 = _loc.to(torch.bfloat16).contiguous()
                    # Stage to CPU now to keep GPU peak low.
                    _loc_cpu = _loc_bf16.detach().cpu().contiguous()
                    del _loc_bf16
                    _m_lid = _ep_layer_re.search(_cpn)
                    _m_proj = _ep_expert_re.search(_cpn)
                    if _m_lid and _m_proj:
                        _lid = int(_m_lid.group(1))
                        _proj = _m_proj.group(1)
                        _ws10_pending_proj.setdefault(_lid, {})[_proj] = _loc_cpu
                        if len(_ws10_pending_proj[_lid]) == 3:
                            _bundle = _ws10_pending_proj.pop(_lid)
                            # Fuse gate+up -> w13 ALONG dim=1 (local
                            # n_local rows preserved). Same convention
                            # as fuse_experts_for_vllm. Then the local
                            # shard for the layer's w13 has shape
                            # [n_local, 2*intermediate_per_tp, hidden]
                            # for gate/up (matches global shape with
                            # n_local instead of total_experts).
                            _w13_local = torch.cat(
                                [_bundle["gate_proj"], _bundle["up_proj"]],
                                dim=1).contiguous()
                            _w2_local = _bundle["down_proj"].contiguous()
                            _local_experts[
                                f"model.layers.{_lid}.mlp.experts.w13_weight"
                            ] = _w13_local
                            _local_experts[
                                f"model.layers.{_lid}.mlp.experts.w2_weight"
                            ] = _w2_local
                    del param, _loc, _loc_cpu
                    continue

                # Non-expert: full_tensor() is a collective on _shard_pg —
                # ALL ranks must call. Rank 0 keeps it; others discard.
                if hasattr(param, "_local_tensor"):
                    _ft0 = time.perf_counter()
                    param = param.full_tensor()
                    t_ft += time.perf_counter() - _ft0
                if _ws10_R == 0:
                    _hf_name = _xccl_accept_and_rename(param_name)
                    if _hf_name is None:
                        del param
                        continue
                    _ne_bf = param.to(torch.bfloat16).contiguous()
                    _ne_cpu = _ne_bf.detach().cpu()
                    del _ne_bf
                    _ws10_non_expert_meta.append({
                        "name": _hf_name,
                        "shape": list(_ne_cpu.shape),
                        "numel": int(_ne_cpu.numel()),
                    })
                    _ws10_non_expert_cpu_parts.append(_ne_cpu.flatten())
                    _ws10_non_expert_full_count += int(_ne_cpu.numel())
                    del _ne_cpu
                del param

            if _ws10_pending_proj:
                raise RuntimeError(
                    f"WS10: rank {self.rank} has {len(_ws10_pending_proj)} "
                    f"layer(s) with incomplete projections after staging: "
                    f"{sorted(_ws10_pending_proj.keys())[:8]}"
                )

            # Build per-rank payload via the shared helper. Uses the WS10-
            # specific batch size override (see comment at _BATCH_MAX_NUMEL
            # definition above) — independent of the legacy path's batching.
            _local_cpu_batches, _local_meta = _ws10_build_local_payload(
                _local_experts, _ws10_R, int(_ep_deg),
                batch_max_numel=_WS10_BATCH_MAX_NUMEL,
            )
            del _local_experts

            # Rank 0 also appends its non-expert tail (no trainer_ep_rank
            # tag in meta — receiver routes by absence). Keep them in
            # rank 0's broadcast batches so they ride R=0's PG.
            if _ws10_R == 0 and _ws10_non_expert_meta:
                # Reorder non-experts to match the receiver's unified
                # manifest order. The receiver sorts every entry with
                # _ws10_sort_key_by_rank, which places non-experts at
                # (R=0, layer_idx=10**9, kind_rank=9, name) — i.e.
                # lexicographic by name. The sender previously packed
                # in sharded_sd.items() walk order (module registration
                # order), so the greedy split landed at different byte
                # boundaries even though total bytes matched. That is
                # the run #7 wire size mismatch (1064400384 vs 622329856,
                # constant across runs because both orderings are
                # deterministic but different). Sort meta + the
                # parallel cpu_parts list by the same key, then pack.
                _ne_pairs = sorted(
                    zip(_ws10_non_expert_meta, _ws10_non_expert_cpu_parts),
                    key=lambda p: _ws10_sort_key_by_rank(p[0]),
                )
                _ws10_non_expert_meta = [p[0] for p in _ne_pairs]
                _ws10_non_expert_cpu_parts = [p[1] for p in _ne_pairs]
                _ne_flat = torch.cat(_ws10_non_expert_cpu_parts)
                # Greedy split same as expert batching.
                _cur_parts = []
                _cur_n = 0
                _ne_offset = 0
                for _ne_e in _ws10_non_expert_meta:
                    _n = int(_ne_e["numel"])
                    if _cur_n > 0 and _cur_n + _n > _WS10_BATCH_MAX_NUMEL:
                        _local_cpu_batches.append(torch.cat(_cur_parts))
                        _cur_parts = []
                        _cur_n = 0
                    _cur_parts.append(_ne_flat[_ne_offset:_ne_offset + _n])
                    _cur_n += _n
                    _ne_offset += _n
                if _cur_parts:
                    _local_cpu_batches.append(torch.cat(_cur_parts))
                del _ne_flat
            _ws10_non_expert_cpu_parts.clear()

            # Mark this rank as the 'active' sender for sync-id bookkeeping.
            # In WS10 every trainer EP rank in [0..ep_d-1] is its own sender;
            # all of them set the done event/counters. Pool semantics differ.
            self._sync_done_event.clear()
            self._sync_error = None
            self._sync_id_counter += 1
            self._pending_sync_id = self._sync_id_counter

            # Gather meta across all trainer EP ranks (object collective on
            # the default PG — payload is small dict-of-dicts).
            _per_rank_metas = [None] * int(_ep_deg)
            # all_gather_object on _shard_pg (size = ep_d).
            torch.distributed.all_gather_object(
                _per_rank_metas, _local_meta, group=self._shard_pg,
            )
            _meta_json = None
            if _ws10_R == 0:
                # Merge non-expert tail INTO rank 0's per-rank slot
                # (the validator in _ws10_unify_manifests requires
                # len(per_rank_metas) == ep_degree). Mirrors the CPU
                # pin-down test in test_ws10_commitB_sender_manifest.
                _per_rank_metas_for_unify = list(_per_rank_metas)
                if _ws10_non_expert_meta and len(_per_rank_metas_for_unify) > 0:
                    _r0_meta = list(_per_rank_metas_for_unify[0] or [])
                    _r0_meta.extend(_ws10_non_expert_meta)
                    _per_rank_metas_for_unify[0] = _r0_meta
                _unified = _ws10_unify_manifests(
                    _per_rank_metas_for_unify,
                    sort_key=_ws10_sort_key_by_rank,
                )
                pool_index = pool.index(self.rank) if pool else 0
                _meta_json = json.dumps({
                    "tensors": _unified,
                    "batch_max_numel": _WS10_BATCH_MAX_NUMEL,
                    "sender_index": pool_index,
                })

            if not self._production_mode:
                torch.distributed.barrier()
            t_gather = time.perf_counter() - _ws10_t0

            # Each rank stores its own (cpu_batches, per-rank PG list) for
            # the deferred broadcast loop. The PG list has one entry per
            # vLLM replica — every replica root gets the data and fans
            # out to its TP workers via its existing intra PG.
            # The 'sharded' flag distinguishes this from the legacy
            # single-PG path.
            _my_pgs = _ws10_pgs_dict[_ws10_R]
            if not isinstance(_my_pgs, list):
                _my_pgs = [_my_pgs]
            # _local_meta only holds the expert-shard entries; rank 0 also
            # ships _ws10_non_expert_meta (the tail) over the wire but it
            # was never merged in here, so the deferred-broadcast thread's
            # "XCCL deferred broadcast done: X GiB" log undercounted rank
            # 0's true payload by the non-expert tail (~2.87 GiB in the
            # EP=8 30B-A3B production config), making its logged GB/s
            # figure wrong (found 2026-07-22 while diagnosing why rank 0's
            # sync time is disproportionate to its send bytes — the real
            # answer is rank 0 uniquely waits on vLLM's full receive+load
            # pipeline via the HTTP POST thread, not this byte-accounting
            # bug, but the wrong number was actively misleading while
            # investigating that).
            _local_meta_for_accounting = (
                _local_meta + _ws10_non_expert_meta
                if _ws10_R == 0 and _ws10_non_expert_meta
                else _local_meta
            )
            self._deferred_broadcast_args = (
                _local_cpu_batches,        # cpu_batches
                _meta_json,                # rank 0 only; other ranks None
                _local_meta_for_accounting,  # used for byte accounting
                _ws10_t0,                  # t0
                self._device,              # device
                _my_pgs,                   # pgs (one per replica)
                self._vllm_clients,        # only rank 0 will call them
                self._vllm_urls,           # only rank 0 will POST
                True,                      # _ws10_sharded marker
            )

            _total_gb = (sum(e["numel"] for e in _local_meta)
                         + _ws10_non_expert_full_count) * 2 / 1024**3
            log.info(
                "Rank %d: WS10 sharded gather done: %d expert entries "
                "(R=%d) + %d non-expert (rank0=%s) %.2f GiB local in "
                "%.1fs (no _shard_pg AllGather) — broadcast deferred",
                self.rank, len(_local_meta), _ws10_R,
                len(_ws10_non_expert_meta), _ws10_R == 0,
                _total_gb, t_gather,
            )

            self._wsync_round += 1
            del sharded_sd
            return

        # Llama-family Q/K un-permute predicate, captured once outside the
        # loop. See _maybe_unpermute_qk for invariant; applied per-param
        # below at the point where `param` becomes the full tensor that
        # vLLM consumes.
        _unperm_needed_x = _needs_qk_unpermute(self)
        if (
            _unperm_needed_x
            and is_active
            and not getattr(self, "_qk_unperm_xccl_engaged_logged", False)
        ):
            log.info(
                "Rank %d: xccl wsync QK un-permute ENGAGED "
                "(n_heads=%s n_kv_heads=%s head_dim=%s)",
                self.rank,
                getattr(self, "_model_num_heads", None),
                getattr(self, "_model_num_kv_heads", None),
                getattr(self, "_model_head_dim", None),
            )
            self._qk_unperm_xccl_engaged_logged = True

        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                param = param.to(self._device)

            # WS6 layer-batched EP path: short-circuit expert params before
            # the per-projection AG. Buffer locally; on the 3rd projection
            # of a layer, emit one batched AG, slice, unshuffle, and hand
            # each per-projection full tensor back to the existing fuse
            # path via _ep_stream_buf (is_active only).
            if _layer_batch_active:
                _lb_cpn = param_name.replace(
                    "_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", "")
                _lb_m_proj = _ep_expert_re.search(_lb_cpn)
                _lb_m_lid = _ep_layer_re.search(_lb_cpn)
                if _lb_m_proj and _lb_m_lid:
                    if hasattr(param, "_local_tensor"):
                        _ft0 = time.perf_counter()
                        param = param.full_tensor()
                        t_ft += time.perf_counter() - _ft0
                    _c0 = time.perf_counter()
                    _lb_local = param.to(self._device).to(
                        torch.bfloat16).contiguous()
                    t_cast += time.perf_counter() - _c0
                    _lb_proj = _lb_m_proj.group(1)
                    _lb_lid = _lb_m_lid.group(1)
                    # Resolve the hf-side param name once now so the
                    # is_active fuse-path entry matches the per-projection
                    # path bit-for-bit. Non-active ranks won't use it.
                    _lb_hf = _xccl_accept_and_rename(param_name) if is_active else None
                    _lb_pending.setdefault(_lb_lid, {})[_lb_proj] = (
                        _lb_hf, _lb_local, tuple(_lb_local.shape))
                    del param

                    if len(_lb_pending[_lb_lid]) == 3:
                        _bundle = _lb_pending.pop(_lb_lid)
                        _g_hf, _g_loc, _g_sh = _bundle["gate_proj"]
                        _u_hf, _u_loc, _u_sh = _bundle["up_proj"]
                        _d_hf, _d_loc, _d_sh = _bundle["down_proj"]
                        _n_g = _g_loc.numel()
                        _n_u = _u_loc.numel()
                        _n_d = _d_loc.numel()
                        _local_cat = torch.cat([
                            _g_loc.flatten(), _u_loc.flatten(), _d_loc.flatten()
                        ])
                        _cat_size = _local_cat.numel()
                        if _gather_root_active:
                            # WS7: gather(dst=active). Root materializes
                            # _out_cat; other ranks send only their local.
                            _shard_rank_lb = torch.distributed.get_rank(
                                self._shard_pg)
                            if _shard_rank_lb == _gr_active_shard_rank:
                                _gather_list = [
                                    torch.empty_like(_local_cat)
                                    for _ in range(_ep_deg)
                                ]
                                torch.distributed.gather(
                                    _local_cat, gather_list=_gather_list,
                                    dst=_gr_active_shard_rank,
                                    group=self._shard_pg)
                                _out_cat = torch.cat(_gather_list)
                                del _gather_list
                            else:
                                torch.distributed.gather(
                                    _local_cat, gather_list=None,
                                    dst=_gr_active_shard_rank,
                                    group=self._shard_pg)
                                _out_cat = None
                            if not _gr_logged_engaged:
                                log.info(
                                    "Rank %d: EP wsync layer-batched gather-to-root engaged",
                                    self.rank)
                                _gr_logged_engaged = True
                        else:
                            _out_cat = torch.empty(
                                _ep_deg * _cat_size,
                                dtype=torch.bfloat16, device=self._device,
                            )
                            torch.distributed.all_gather_into_tensor(
                                _out_cat, _local_cat, group=self._shard_pg)
                        del _local_cat, _g_loc, _u_loc, _d_loc

                        if not _lb_logged_engaged:
                            log.info(
                                "Rank %d: EP wsync layer-batched AG engaged "
                                "(3-proj per call)", self.rank)
                            _lb_logged_engaged = True

                        if is_active:
                            # Slice each rank's chunk into per-projection
                            # sub-tensors, then stack + interleave-unshuffle
                            # — same final shape as the per-projection path.
                            _g_parts, _u_parts, _d_parts = [], [], []
                            for _r in range(_ep_deg):
                                _b = _r * _cat_size
                                _g_parts.append(
                                    _out_cat[_b:_b + _n_g].reshape(_g_sh))
                                _u_parts.append(
                                    _out_cat[_b + _n_g:_b + _n_g + _n_u].reshape(_u_sh))
                                _d_parts.append(
                                    _out_cat[_b + _n_g + _n_u:_b + _n_g + _n_u + _n_d].reshape(_d_sh))
                            for _hf, _parts in (
                                (_g_hf, _g_parts),
                                (_u_hf, _u_parts),
                                (_d_hf, _d_parts),
                            ):
                                if _hf is None:
                                    continue
                                _stk = torch.stack(_parts, dim=0)
                                _full = _stk.transpose(0, 1).reshape(
                                    -1, *_stk.shape[2:]).contiguous()
                                del _stk
                                # Hand off to the existing _ep_stream_buf
                                # fuse path (mirrors the per-projection
                                # branch below). The fuse trigger fires on
                                # the 3rd projection identically.
                                _m_proj_full = _ep_expert_re.search(_hf)
                                _m_lid_full = _ep_layer_re.search(_hf)
                                if _m_proj_full and _m_lid_full:
                                    _lid_full = _m_lid_full.group(1)
                                    _ep_stream_buf.setdefault(
                                        _lid_full, {})[_m_proj_full.group(1)] = _full
                                    if len(_ep_stream_buf[_lid_full]) == 3:
                                        _d_inner = _ep_stream_buf.pop(_lid_full)
                                        _w13 = torch.cat(
                                            [_d_inner["gate_proj"], _d_inner["up_proj"]], dim=1)
                                        _w2 = _d_inner["down_proj"]
                                        del _d_inner
                                        for _fhname, _ftensor in [
                                            (f"model.layers.{_lid_full}.mlp.experts.w13_weight", _w13),
                                            (f"model.layers.{_lid_full}.mlp.experts.w2_weight", _w2),
                                        ]:
                                            _fpn = _ftensor.numel()
                                            tensors_meta.append({
                                                "name": _fhname,
                                                "shape": list(_ftensor.shape),
                                                "numel": _fpn,
                                            })
                                            if _USE_PINNED_BUF:
                                                if batch_numel > 0 and batch_numel + _fpn > _BATCH_MAX_NUMEL:
                                                    torch.xpu.synchronize()
                                                    cpu_batches.append(
                                                        self._pinned_cpu_buf[_batch_start:_buf_offset])
                                                    _batch_start = _buf_offset
                                                    batch_numel = 0
                                                    n_batches += 1
                                                self._pinned_cpu_buf[_buf_offset:_buf_offset + _fpn].copy_(
                                                    _ftensor.flatten(), non_blocking=True)
                                                _buf_offset += _fpn
                                            else:
                                                _fcpu = _ftensor.flatten().cpu()
                                                if batch_numel > 0 and batch_numel + _fpn > _BATCH_MAX_NUMEL:
                                                    cpu_batches.append(torch.cat(batch_parts_cpu))
                                                    batch_parts_cpu = []
                                                    batch_numel = 0
                                                    n_batches += 1
                                                batch_parts_cpu.append(_fcpu)
                                            batch_numel += _fpn
                                            n_params += 1
                                        del _w13, _w2
                        del _out_cat
                    continue  # expert param handled by layer-batch path

            if hasattr(param, "_local_tensor"):
                _ft0 = time.perf_counter()
                param = param.full_tensor()
                t_ft += time.perf_counter() - _ft0
            # EP expert all-gather: each rank holds an interleaved slice of experts
            # (rank r owns global experts r, r+ep_d, r+2*ep_d, …).  Gather all slices
            # so the active sender has the full expert tensor.  Collective: all ranks
            # participate regardless of is_active.
            _ep_is_expert = False
            if _ep_deg > 1 and _is_moe and getattr(self, '_shard_pg', None) is not None:
                _cpn = param_name.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", "")
                if _ep_expert_re.search(_cpn):
                    _ep_is_expert = True
                    _ep_local_t = param.to(self._device).contiguous()
                    if _gather_root_active:
                        # WS7: gather(dst=active). Root reassembles from
                        # _ep_deg shards; non-root ranks just send.
                        _shard_rank_pp = torch.distributed.get_rank(
                            self._shard_pg)
                        if _shard_rank_pp == _gr_active_shard_rank:
                            _ep_parts_t = [
                                torch.empty_like(_ep_local_t)
                                for _ in range(_ep_deg)
                            ]
                            torch.distributed.gather(
                                _ep_local_t, gather_list=_ep_parts_t,
                                dst=_gr_active_shard_rank,
                                group=self._shard_pg)
                            _ep_stk = torch.stack(_ep_parts_t, dim=0)
                            param = _ep_stk.transpose(0, 1).reshape(
                                -1, *_ep_stk.shape[2:]).contiguous()
                            del _ep_stk, _ep_parts_t
                        else:
                            torch.distributed.gather(
                                _ep_local_t, gather_list=None,
                                dst=_gr_active_shard_rank,
                                group=self._shard_pg)
                        if not _gr_logged_engaged:
                            log.info(
                                "Rank %d: EP wsync per-proj gather-to-root engaged",
                                self.rank)
                            _gr_logged_engaged = True
                    elif _fp8_wire_active:
                        # WS8: cast local shard to E4M3 with per-row
                        # scale, AllGather fp8 + scale, decompress on
                        # active rank. Reduces wire bytes ~2× on
                        # _shard_pg vs the bf16 baseline below.
                        _ep_local_bf = _ep_local_t.to(torch.bfloat16)
                        _ep_local_f32 = _ep_local_bf.to(torch.float32)
                        # Per-output-row scaling (last dim is the
                        # row dim for both gate/up [E, I, H] and down
                        # [E, H, I]; the row whose error matters is
                        # the GEMM input dim).
                        _row_amax = _ep_local_f32.abs().amax(
                            dim=-1, keepdim=True)
                        _scale_local = (_row_amax / 448.0).clamp(min=1e-12)
                        _ep_local_fp8 = (
                            (_ep_local_f32 / _scale_local).clamp(-448.0, 448.0)
                            .to(torch.float8_e4m3fn).contiguous())
                        del _ep_local_bf, _ep_local_f32
                        _scale_local_f32 = _scale_local.to(torch.float32).contiguous()
                        del _scale_local

                        # all_gather_into_tensor (single persistent flat
                        # output buffer) instead of list-style all_gather.
                        # The list path triggers ProcessGroupXCCL's hidden
                        # newLikeFlat() temp, which leaks under
                        # torch.xpu.empty_cache() (Aurora #143 /
                        # torch-xpu-ops #3744). Bit-exact equivalent.
                        _local_fp8 = _ep_local_fp8.contiguous()
                        _local_sc = _scale_local_f32.contiguous()
                        _out_fp8_cat = torch.empty(
                            _ep_deg * _local_fp8.shape[0],
                            *_local_fp8.shape[1:],
                            dtype=_local_fp8.dtype,
                            device=_local_fp8.device)
                        _out_sc_cat = torch.empty(
                            _ep_deg * _local_sc.shape[0],
                            *_local_sc.shape[1:],
                            dtype=_local_sc.dtype,
                            device=_local_sc.device)
                        torch.distributed.all_gather_into_tensor(
                            _out_fp8_cat, _local_fp8,
                            group=self._shard_pg)
                        torch.distributed.all_gather_into_tensor(
                            _out_sc_cat, _local_sc,
                            group=self._shard_pg)
                        del _ep_local_fp8, _scale_local_f32, _local_fp8, _local_sc
                        if is_active:
                            # Reshape concatenated buffer to
                            # [ep_d, n_local, ...] — same layout the old
                            # torch.stack(list, dim=0) produced.
                            _n_local_fp8 = _out_fp8_cat.shape[0] // _ep_deg
                            _n_local_sc = _out_sc_cat.shape[0] // _ep_deg
                            _fp8_view = _out_fp8_cat.view(
                                _ep_deg, _n_local_fp8,
                                *_out_fp8_cat.shape[1:])
                            _sc_view = _out_sc_cat.view(
                                _ep_deg, _n_local_sc,
                                *_out_sc_cat.shape[1:])
                            # Per-rank dequantize (shape preserved),
                            # then unshuffle identically to bf16 path.
                            _bf16_parts = []
                            for _r in range(_ep_deg):
                                _bf16_parts.append(
                                    (_fp8_view[_r].to(torch.float32)
                                     * _sc_view[_r])
                                    .to(torch.bfloat16))
                            _ep_stk = torch.stack(_bf16_parts, dim=0)
                            param = _ep_stk.transpose(0, 1).reshape(
                                -1, *_ep_stk.shape[2:]).contiguous()
                            del _ep_stk, _bf16_parts, _fp8_view, _sc_view
                        del _out_fp8_cat, _out_sc_cat
                        if not _fp8_logged_engaged:
                            log.info(
                                "Rank %d: EP wsync per-proj fp8 wire engaged "
                                "(local bytes %d → %d + %d scale)",
                                self.rank,
                                _ep_local_t.numel() * _ep_local_t.element_size(),
                                _ep_local_t.numel() * 1,
                                int(_row_amax.numel() * 4),
                            )
                            _fp8_logged_engaged = True
                    else:
                        # all_gather_into_tensor (single persistent flat
                        # output buffer) instead of list-style all_gather.
                        # The list path triggers ProcessGroupXCCL's hidden
                        # newLikeFlat() temp, which leaks under
                        # torch.xpu.empty_cache() (Aurora #143 /
                        # torch-xpu-ops #3744). Bit-exact equivalent of
                        # the previous torch.stack(list, 0) layout.
                        _local_c = _ep_local_t.contiguous()
                        _out_cat = torch.empty(
                            _ep_deg * _local_c.shape[0],
                            *_local_c.shape[1:],
                            dtype=_local_c.dtype,
                            device=_local_c.device)
                        torch.distributed.all_gather_into_tensor(
                            _out_cat, _local_c, group=self._shard_pg)
                        if is_active:
                            # Reshape [ep_d * n_local, ...] to
                            # [ep_d, n_local, ...] (same layout as
                            # torch.stack(list, dim=0)), then unshuffle:
                            # transpose → [n_local, ep_d, ...] → reshape →
                            # [total_experts, ...]
                            _n_local = _out_cat.shape[0] // _ep_deg
                            _ep_stk = _out_cat.view(
                                _ep_deg, _n_local, *_out_cat.shape[1:])
                            param = _ep_stk.transpose(0, 1).reshape(
                                -1, *_ep_stk.shape[2:]).contiguous()
                            del _ep_stk
                        del _out_cat, _local_c
                    del _ep_local_t
            if is_active:
                hf_name = _xccl_accept_and_rename(param_name)
                if hf_name is None:
                    del param
                    continue
                # Llama-family Q/K un-permute: torchtune permutes Q/K at
                # checkpoint load (convert_weights.hf_to_tune); vLLM expects
                # HF-format unpermuted Q/K. Without this, server-mode XCCL
                # wsync silently scrambles Q/K projections on every sync
                # (mirrors the colocate fix landed 2026-06-11). Bug latent
                # since this xccl path was added — Qwen/Gemma never tripped
                # it (they don't permute), AGPT-2B is the first Llama-family
                # server-mode GRPO target.
                if _unperm_needed_x:
                    param = _maybe_unpermute_qk(self, hf_name, param)
                _c0 = time.perf_counter()
                gpu_tensor = param.to(torch.bfloat16).contiguous()
                t_cast += time.perf_counter() - _c0
                if _ep_is_expert:
                    # Streaming EP fuse: accumulate gate/up/down per layer; fuse+stage
                    # to CPU when all 3 projections for a layer have arrived.  Keeps GPU
                    # peak at ~3 expert tensors instead of 48 layers × 128 experts.
                    _m_proj = _ep_expert_re.search(hf_name)
                    _m_lid = _ep_layer_re.search(hf_name)
                    if _m_proj and _m_lid:
                        _lid = _m_lid.group(1)
                        _ep_stream_buf.setdefault(_lid, {})[_m_proj.group(1)] = gpu_tensor
                        if len(_ep_stream_buf[_lid]) == 3:
                            _d = _ep_stream_buf.pop(_lid)
                            _w13 = torch.cat([_d["gate_proj"], _d["up_proj"]], dim=1)
                            _w2 = _d["down_proj"]
                            del _d
                            for _fhname, _ftensor in [
                                (f"model.layers.{_lid}.mlp.experts.w13_weight", _w13),
                                (f"model.layers.{_lid}.mlp.experts.w2_weight", _w2),
                            ]:
                                _fpn = _ftensor.numel()
                                tensors_meta.append({
                                    "name": _fhname,
                                    "shape": list(_ftensor.shape),
                                    "numel": _fpn,
                                })
                                if _USE_PINNED_BUF:
                                    if batch_numel > 0 and batch_numel + _fpn > _BATCH_MAX_NUMEL:
                                        torch.xpu.synchronize()
                                        cpu_batches.append(
                                            self._pinned_cpu_buf[_batch_start:_buf_offset])
                                        _batch_start = _buf_offset
                                        batch_numel = 0
                                        n_batches += 1
                                    self._pinned_cpu_buf[_buf_offset:_buf_offset + _fpn].copy_(
                                        _ftensor.flatten(), non_blocking=True)
                                    _buf_offset += _fpn
                                else:
                                    _fcpu = _ftensor.flatten().cpu()
                                    if batch_numel > 0 and batch_numel + _fpn > _BATCH_MAX_NUMEL:
                                        cpu_batches.append(torch.cat(batch_parts_cpu))
                                        batch_parts_cpu = []
                                        batch_numel = 0
                                        n_batches += 1
                                    batch_parts_cpu.append(_fcpu)
                                batch_numel += _fpn
                                n_params += 1
                            del _w13, _w2
                    del param  # gpu_tensor stays alive in _ep_stream_buf if layer incomplete
                    continue   # skip normal staging path; outer del param already done
                # For MoE, non-expert params go to hf_state_dict and are
                # added to tensors_meta + cpu_batches after fuse_experts_for_vllm.
                # Adding to tensors_meta here would duplicate every non-expert
                # entry for EP mode (which appends rather than resets tensors_meta),
                # causing tensors_meta (966) vs cpu_batches (531) desync → n_expert_ready=0.
                # Non-EP mode resets tensors_meta at line 1882, so the duplicate
                # is discarded anyway — skip for clarity.
                if not _is_moe:
                    tensors_meta.append({
                        "name": hf_name,
                        "shape": list(gpu_tensor.shape),
                        "numel": gpu_tensor.numel(),
                    })
                if _is_moe:
                    hf_state_dict[hf_name] = gpu_tensor
                elif _USE_PINNED_BUF:
                    pn = gpu_tensor.numel()
                    if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                        torch.xpu.synchronize()
                        cpu_batches.append(self._pinned_cpu_buf[_batch_start:_buf_offset])
                        _batch_start = _buf_offset
                        batch_numel = 0
                        n_batches += 1
                    _d2h0 = time.perf_counter()
                    if _USE_D2H_STREAM and hasattr(self, '_d2h_stream') and self._d2h_stream is not None:
                        _ev = torch.xpu.Event()
                        _ev.record()
                        with torch.xpu.stream(self._d2h_stream):
                            _ev.wait()
                            self._pinned_cpu_buf[_buf_offset:_buf_offset + pn].copy_(
                                gpu_tensor.flatten(), non_blocking=True)
                    else:
                        self._pinned_cpu_buf[_buf_offset:_buf_offset + pn].copy_(
                            gpu_tensor.flatten(), non_blocking=True)
                    t_d2h += time.perf_counter() - _d2h0
                    _buf_offset += pn
                    del gpu_tensor
                    batch_numel += pn
                    n_params += 1
                else:
                    pn = gpu_tensor.numel()
                    _d2h0 = time.perf_counter()
                    cpu_tensor = gpu_tensor.flatten().cpu()
                    t_d2h += time.perf_counter() - _d2h0
                    del gpu_tensor
                    if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                        cpu_batches.append(torch.cat(batch_parts_cpu))
                        batch_parts_cpu = []
                        batch_numel = 0
                        n_batches += 1
                    batch_parts_cpu.append(cpu_tensor)
                    batch_numel += pn
                    n_params += 1
            del param

        # WS6: sanity check that no layer-batched expert pairs are
        # dangling. If they are, a layer didn't have all 3 projections
        # in sharded_sd (unexpected for Qwen3-30B-A3B; would silently
        # drop weights from the wsync round).
        if _layer_batch_active and _lb_pending:
            _dangling = sorted(_lb_pending.keys())
            raise RuntimeError(
                f"EP wsync layer-batch path: {len(_dangling)} layer(s) had "
                f"incomplete projection set after param loop: {_dangling[:8]}. "
                "This means the model state_dict is missing one of "
                "{gate_proj, up_proj, down_proj} for those layers."
            )

        if is_active:
            if _is_moe:
                from torchtune.models.qwen3_moe._convert_weights import fuse_experts_for_vllm
                if _ep_deg > 1:
                    # EP mode: expert params already streamed + fused above (tensors_meta
                    # already has w13/w2 entries).  hf_state_dict has only non-expert
                    # params (attention, norms, embeddings).  Pass through fuse (no-op
                    # for non-expert keys) and APPEND — do NOT reset tensors_meta.
                    hf_state_dict = fuse_experts_for_vllm(hf_state_dict)
                    for hf_name, tensor in hf_state_dict.items():
                        pn = tensor.numel()
                        tensors_meta.append({
                            "name": hf_name,
                            "shape": list(tensor.shape),
                            "numel": pn,
                        })
                        if _USE_PINNED_BUF:
                            if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                                torch.xpu.synchronize()
                                cpu_batches.append(
                                    self._pinned_cpu_buf[_batch_start:_buf_offset])
                                _batch_start = _buf_offset
                                batch_numel = 0
                                n_batches += 1
                            self._pinned_cpu_buf[_buf_offset:_buf_offset + pn].copy_(
                                tensor.flatten(), non_blocking=True)
                            _buf_offset += pn
                        else:
                            cpu_tensor = tensor.flatten().cpu()
                            if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                                cpu_batches.append(torch.cat(batch_parts_cpu))
                                batch_parts_cpu = []
                                batch_numel = 0
                                n_batches += 1
                            batch_parts_cpu.append(cpu_tensor)
                        batch_numel += pn
                        n_params += 1
                    del hf_state_dict
                else:
                    # Non-EP MoE: fuse all experts, reset tensors_meta from fused output.
                    hf_state_dict = fuse_experts_for_vllm(hf_state_dict)
                    tensors_meta = []
                    for hf_name, tensor in hf_state_dict.items():
                        tensors_meta.append({
                            "name": hf_name,
                            "shape": list(tensor.shape),
                            "numel": tensor.numel(),
                        })
                        pn = tensor.numel()
                        if _USE_PINNED_BUF:
                            if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                                torch.xpu.synchronize()
                                cpu_batches.append(
                                    self._pinned_cpu_buf[_batch_start:_buf_offset])
                                _batch_start = _buf_offset
                                batch_numel = 0
                                n_batches += 1
                            _d2h0 = time.perf_counter()
                            self._pinned_cpu_buf[_buf_offset:_buf_offset + pn].copy_(
                                tensor.flatten(), non_blocking=True)
                            t_d2h += time.perf_counter() - _d2h0
                            _buf_offset += pn
                        else:
                            cpu_tensor = tensor.flatten().cpu()
                            if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                                cpu_batches.append(torch.cat(batch_parts_cpu))
                                batch_parts_cpu = []
                                batch_numel = 0
                                n_batches += 1
                            batch_parts_cpu.append(cpu_tensor)
                        batch_numel += pn
                        n_params += 1
                    del hf_state_dict

            if _USE_PINNED_BUF:
                if batch_numel > 0:
                    if _USE_D2H_STREAM and hasattr(self, '_d2h_stream') and self._d2h_stream is not None:
                        self._d2h_stream.synchronize()
                    else:
                        torch.xpu.synchronize()
                    cpu_batches.append(self._pinned_cpu_buf[_batch_start:_buf_offset])
                    n_batches += 1
            else:
                if batch_parts_cpu:
                    cpu_batches.append(torch.cat(batch_parts_cpu))
                    batch_parts_cpu = []
                    n_batches += 1

            pool_index = pool.index(self.rank) if pool else 0
            meta_json = json.dumps({
                "tensors": tensors_meta,
                "batch_max_numel": _BATCH_MAX_NUMEL,
                "sender_index": pool_index,
            })

    del sharded_sd

    if not self._production_mode:
        torch.distributed.barrier()
    t_gather = time.perf_counter() - t0

    if is_active:
        if _USE_BATCHED_AG:
            # Batched-AG mode (BROKEN, kept for reference): sync path
            for mt in manifest_threads:
                mt.join(timeout=600)
            if post_errors:
                log.error("Rank %d: XCCL streaming sync errors: %s", self.rank, post_errors)
                self._sync_error = RuntimeError(str(post_errors))
            for client in self._vllm_clients:
                client.reset_prefix_cache()
            log.info(
                "Rank %d: XCCL batched-AG sync: %d params %d batches in %.1fs "
                "(ag=%.1fs bcast=%.1fs)",
                self.rank, n_params, n_batches,
                time.perf_counter() - t0, t_ag, t_bcast,
            )
            self._sync_done_event.set()
        else:
            # Mode 0 async: defer broadcast to after generation completes.
            _cross_pgs = [self._my_cross_pg] if pool else getattr(self, '_xccl_wsync_pgs', [self._xccl_wsync_pg])
            total_gb = sum(e["numel"] for e in tensors_meta) * 2 / 1024**3
            _buf_info = ""
            if _USE_PINNED_BUF:
                _buf_info = (
                    f", pinned={getattr(self, '_cpu_buf_is_pinned', False)}"
                    f", d2h_stream={_USE_D2H_STREAM and getattr(self, '_d2h_stream', None) is not None}"
                )
            log.info(
                "Rank %d: XCCL async gather done: %d params %d batches %.2f GiB "
                "staged to CPU in %.1fs (ft=%.1fs cast=%.1fs d2h=%.1fs%s), "
                "broadcast deferred to post-gen (sender pool round %d)",
                self.rank, n_params, n_batches, total_gb, t_gather,
                t_ft, t_cast, t_d2h, _buf_info,
                self._wsync_round,
            )
            self._deferred_broadcast_args = (
                cpu_batches, meta_json,
                tensors_meta, t0, self._device,
                _cross_pgs, self._vllm_clients,
                self._vllm_urls,
            )

    self._wsync_round += 1

def _sync_weights_to_vllm_xccl(self) -> None:
    """Gather sharded params then broadcast to vLLM via XCCL (GPU→GPU).

    GPU-direct path: full_tensor() returns a GPU tensor, so we concat
    directly into a flat GPU buffer — no CPU staging. The background
    thread only does the HTTP POST + broadcast.

    For 32B+ models where the full flat buffer won't fit on one tile,
    falls back to CPU staging (same as SHM path).
    """
    t0 = time.perf_counter()

    # Init sender pool on first call (ALL ranks participate)
    if not hasattr(self, '_wsync_pool_init_done'):
        self._init_sender_pool()
        # Flush any GPU state from XCCL communicator init (ProcessGroupXCCL
        # constructor may leave pending GPU ops on some hardware/firmware combos
        # that deadlock the subsequent FSDP2 dp_shard AllGather). Observed on
        # x4311 chassis nodes (WS8 job 8465300); WS7 nodes unaffected.
        if self._device.type == "xpu" and torch.xpu.is_available():
            torch.xpu.synchronize()

    # Determine active sender for this round
    pool = self._wsync_sender_pool
    if pool:
        active_sender = pool[self._wsync_round % len(pool)]
        is_active = (self.rank == active_sender)
    else:
        # WS3.5: legacy single-sender path drives one cross-PG; only the
        # global XCCL leader (rank 0) can broadcast. With dp_replicate>1
        # multiple shard leaders would race for the same vLLM-side rank.
        is_active = getattr(self, "_is_xccl_leader", self._is_shard_leader)

    # _use_fsdp1 is set by both _setup_model_fsdp1_hsdp (dp_replicate > 1) and
    # _setup_model_fsdp1_flat_zero2 (flat FSDP1 SHARD_GRAD_OP, dp_replicate == 1).
    # Both wrap with the FSDP1 API; .full_tensor() doesn't exist on the params
    # so we MUST go through FSDP.state_dict() in either case.
    use_fsdp1 = getattr(self, '_use_fsdp1', False)

    # BioReason: vLLM only loads the backbone (Qwen3-4B). Strip the
    # 'backbone.' prefix and skip everything else (ESM3, GO encoder,
    # projectors). Mirrors the raw_bytes _accept_and_rename above.
    _is_bior = getattr(self, "_is_bioreason", False)

    def _xccl_accept_and_rename(name: str):
        """Return the vLLM-side name, or None to skip this param."""
        if _is_bior:
            clean = name.replace("_fsdp_wrapped_module.", "")
            clean = clean.replace("_checkpoint_wrapped_module.", "")
            if not clean.startswith("backbone."):
                return None
            return clean[len("backbone."):]
        return self._tune_to_hf_map.get(name, name)

    if use_fsdp1:
        self._xccl_gather_fsdp1(_xccl_accept_and_rename, t0)
    else:
        self._xccl_gather_and_stage_fsdp2(is_active, pool, _xccl_accept_and_rename, t0)

def _sync_weights_to_vllm_shm(self) -> None:
    """Gather sharded params then async-dispatch to vLLM via POSIX shared memory.

    Faster alternative to _sync_weights_to_vllm (raw bytes file) for large models:

      Phase 1 (synchronous): FSDP full_tensor() gather — same as raw bytes path.
      Phase 2 (async, shard leader only):
        - Allocate a single SharedMemory block (one mmap, one OS call).
        - Copy all tensors in via ctypes.memmove (DDR5 bandwidth, ~0.2s for 6 GB,
          ~1.8s for 62 GB). No Python object allocation — avoids GC pressure.
        - Write metadata JSON (tensor names/shapes/offsets) to /dev/shm.
        - POST metadata path to vLLM load_weights_from_shm.
        - vLLM maps the SAME physical RAM pages zero-copy via frombuffer(shm.buf).
        - After vLLM confirms, unlink the SHM block.

    Total for 62 GB BF16 model:
      Raw bytes: write=24s + HTTP(read=6s+load=?s) = 30s+
      SHM:       copy=1.8s + HTTP(map=0s+to_xpu=1.2s+load=?s) = 3s+
      Savings: ~27s per sync for 31B.

    Enable with config: vllm_weight_sync_method: shm
    """
    import ctypes
    import json
    import re
    from multiprocessing.shared_memory import SharedMemory

    t0 = time.perf_counter()
    hf_state_dict = {}

    # Phase 1: FSDP gather (same as raw bytes path).
    # _use_fsdp1 is set by both the HSDP path (dp_replicate>1) and the new
    # flat ZeRO-2 path (dp_replicate==1); both wrap with FSDP1 API.
    if getattr(self, '_use_fsdp1', False):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
        with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
            full_sd = self._model.state_dict()
        if self._is_shard_leader:
            for param_name, param in full_sd.items():
                hf_name = self._tune_to_hf_map.get(param_name, param_name)
                hf_state_dict[hf_name] = param.cpu()
        del full_sd
    else:
        sharded_sd = self._model.state_dict()
        _expert_re = re.compile(
            r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)"
        )
        _pending_experts: dict[int, dict[str, torch.Tensor]] = {}
        t_ag0 = time.perf_counter()
        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                param = param.to(self._device)
            if hasattr(param, "_local_tensor"):
                param = param.full_tensor()
            if self._is_shard_leader:
                hf_name = self._tune_to_hf_map.get(param_name, param_name)
                m = _expert_re.search(hf_name)
                if m:
                    # Accumulate expert tensors on GPU; fuse when all 3 arrive
                    layer_idx = int(m.group(1))
                    proj = m.group(2)
                    _pending_experts.setdefault(layer_idx, {})[proj] = param
                    if len(_pending_experts[layer_idx]) == 3:
                        d = _pending_experts.pop(layer_idx)
                        w13 = torch.cat([d["gate_proj"], d["up_proj"]], dim=1)
                        hf_state_dict[f"model.layers.{layer_idx}.mlp.experts.w13_weight"] = w13.cpu()
                        hf_state_dict[f"model.layers.{layer_idx}.mlp.experts.w2_weight"] = d["down_proj"].cpu()
                        del d, w13
                else:
                    hf_state_dict[hf_name] = param.cpu()
            del param
        t_ag_done = time.perf_counter()
        del sharded_sd

    if not self._production_mode:
        torch.distributed.barrier()

    t_gather = time.perf_counter() - t0
    if self._is_shard_leader:
        log.info(
            "Rank %d: SHM gather breakdown: full_tensor+fuse+cpu=%.1fs total=%.1fs",
            self.rank, t_ag_done - t_ag0, t_gather,
        )

    if self._is_shard_leader:
        n_params = len(hf_state_dict)
        meta_path = "/dev/shm/torchtune/wsync_meta.json"

        self._sync_done_event.clear()
        self._sync_error = None
        self._sync_id_counter += 1
        self._pending_sync_id = self._sync_id_counter

        def _bg_shm_and_post(state_dict=hf_state_dict):
            try:
                import numpy as _np

                # Compute total bytes and build numpy views
                tensors_meta = []
                np_arrays = []
                total_bytes = 0
                for hf_name, tensor in state_dict.items():
                    if tensor.dtype == torch.bfloat16:
                        np_arr = tensor.view(torch.int16).numpy()
                    else:
                        np_arr = tensor.numpy()
                    if not np_arr.flags['C_CONTIGUOUS']:
                        np_arr = _np.ascontiguousarray(np_arr)
                    nbytes = np_arr.nbytes
                    tensors_meta.append({
                        "name": hf_name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "offset": total_bytes,
                        "nbytes": nbytes,
                    })
                    np_arrays.append(np_arr)
                    total_bytes += nbytes

                del state_dict  # free gathered tensors ASAP

                # Reuse the persistent SHM block if size matches — avoids page-fault
                # overhead (~4s for 6 GB) caused by OS demand-paging fresh SHM pages.
                # On first call or size change, allocate a new block and warm it up.
                shm_name = "torchtune_weights"
                existing = getattr(self, '_shm_block', None)
                if existing is not None and existing.size == total_bytes:
                    shm = existing
                    is_new = False
                else:
                    # Clean up old block (size change or first run)
                    if existing is not None:
                        try:
                            existing.close()
                            existing.unlink()
                        except Exception:
                            pass
                    # Also clean up any stale SHM from a previous process
                    try:
                        stale = SharedMemory(name=shm_name, create=False)
                        stale.close()
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                    shm = SharedMemory(name=shm_name, create=True, size=max(total_bytes, 1))
                    self._shm_block = shm
                    is_new = True

                # Copy tensors into SHM via ctypes.memmove — no Python object allocation.
                # On reused blocks, pages are already faulted → DDR5 bandwidth (~30 GB/s).
                # On new blocks, first-touch faults limit to ~1-2 GB/s (one-time cost).
                t_copy0 = time.perf_counter()
                for arr, entry in zip(np_arrays, tensors_meta):
                    dst = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, entry["offset"]))
                    src = arr.ctypes.data
                    ctypes.memmove(dst, src, entry["nbytes"])
                del np_arrays
                t_copy = time.perf_counter() - t_copy0

                gb = total_bytes / 1024**3
                log.info(
                    "Rank %d: shm copy done: %d params %.2f GiB in %.1fs (%.1f GB/s) → shm:%s%s",
                    self.rank, n_params, gb, t_copy, gb / t_copy if t_copy > 0 else 0,
                    shm_name, " [new block, page-fault warmup]" if is_new else " [reused]",
                )

                # Build and POST metadata to vLLM
                meta_json = json.dumps({
                    "shm_name": shm_name,
                    "total_bytes": total_bytes,
                    "tensors": tensors_meta,
                })

                import requests
                t_http0 = time.perf_counter()
                for url in self._vllm_urls:
                    try:
                        r = requests.post(
                            f"{url}/collective_rpc",
                            json={"method": "load_weights_from_shm", "args": [meta_json]},
                            timeout=300,
                        )
                        if r.status_code != 200:
                            log.warning("vLLM shm reload failed (%s): %s %s", url, r.status_code, r.text[:200])
                        else:
                            result = r.json()
                            results = result.get("results", [{}])
                            first = results[0] if results else {}
                            if isinstance(first, dict) and first.get("status") != "ok":
                                log.warning("vLLM shm reload error (%s): %s", url, first)
                    except Exception as e:
                        log.error("vLLM shm reload HTTP error (%s): %s", url, e)
                t_http = time.perf_counter() - t_http0

                for client in self._vllm_clients:
                    client.reset_prefix_cache()

                log.info(
                    "Rank %d: weight sync shm %d vLLM replica(s): %d params %.1fs total "
                    "(gather=%.1fs [sync] copy=%.1fs http=%.1fs) [shm:%s]",
                    self.rank, len(self._vllm_urls), n_params, time.perf_counter() - t0,
                    t_gather, t_copy, t_http, shm_name,
                )

            except Exception as e:
                log.error("Rank %d: shm weight sync failed: %s", self.rank, e, exc_info=True)
                self._sync_error = e
            finally:
                # SHM block is kept alive (self._shm_block) for reuse next step.
                # Cleanup happens in teardown() or on size change above.
                self._sync_done_event.set()

        t = threading.Thread(target=_bg_shm_and_post, daemon=True, name="weight_sync_shm")
        t.start()
        log.info(
            "Rank %d: weight sync (shm) gather done in %.1fs, async copy+POST started (%d params)",
            self.rank, t_gather, n_params,
        )

    # NOTE: do NOT call device_empty_cache() here — same reason as raw_bytes path.

def _wait_for_sync_complete(self) -> None:
    """Block until the background weight sync thread finishes.

    Called before the NEXT sync dispatch (not before generation) to ensure the
    XCCL PG is free. The deferred broadcast typically completes during GRPO/backward
    (~14s window for ~12s broadcast), so wait time is usually zero.

    Side effect (Phase 2 Step 2): on rank 0, bump
    ``self._weight_versions`` after the previous sync is confirmed done.
    This is the moment we *know* vLLM holds the new weights, so any
    rollout the producer kicks off after this returns is generated
    under the new version.
    """
    if not self._vllm_weight_sync:
        return
    if not hasattr(self, "_sync_done_event"):
        return
    if not self._sync_done_event.is_set():
        t_wait0 = time.perf_counter()
        self._sync_done_event.wait()
        waited = time.perf_counter() - t_wait0
        if waited > 0.05:
            log.info("Rank %d: waited %.1fs for async weight sync to complete", self.rank, waited)
    # Bump the weight version ONLY if (a) a sync was actually dispatched since
    # the last bump AND (b) the background sync did not error. Without the
    # _sync_error gate, a failed sync still inflated the version counter and
    # cleared _pending_sync_id, telling downstream telemetry/producers that
    # vLLM held new weights when in fact the broadcast never landed.
    sync_failed = self._sync_error is not None
    if sync_failed:
        log.error(
            "Rank %d: previous async weight sync had an error: %s — NOT bumping weight version",
            self.rank, self._sync_error,
        )
        self._sync_error = None  # reset so training continues
    # Version-bump rank set. The async RolloutProducer runs on every replica's
    # SHARD LEADER (global ranks 0, dp_shard, 2*dp_shard, ... under HSDP), so each
    # of those leaders must advance ITS OWN _weight_versions tracker in lockstep —
    # otherwise the staleness telemetry on the non-rank-0 leaders would stay pinned
    # at 0 (the per-replica producers tag against their local tracker). All replicas
    # are weight-identical (HYBRID_SHARD all-reduces grads), so each leader bumps
    # for its own replica. Only rank 0 actually PUBLISHES to vLLM (_is_publisher),
    # but every shard leader completes the FULL_STATE_DICT gather collective every
    # step. Single-replica: _is_shard_leader == rank 0, so this is unchanged.
    _is_leader = getattr(self, "_is_shard_leader", self._is_rank_zero)
    if _is_leader and getattr(self, "_weight_versions", None) is not None:
        # Non-publisher shard leaders (HSDP ranks 12, 24, ...) never run the
        # publisher block that sets _pending_sync_id, but they DO sync every step
        # (their replica's producer consumes a fresh rollout each step). Synthesize
        # a pending id for them so the bump fires in lockstep with rank 0.
        _is_publisher = (
            (self.rank == 0) if getattr(self, "_dp_replicate", 1) > 1
            else getattr(self, "_is_shard_leader", self._is_rank_zero)
        )
        if (not _is_publisher) and getattr(self, "_pending_sync_id", None) is None and not sync_failed:
            self._sync_id_counter = getattr(self, "_sync_id_counter", 0) + 1
            self._pending_sync_id = self._sync_id_counter
        if getattr(self, "_pending_sync_id", None) is None:
            log.warning(
                "Rank %d: _wait_for_sync_complete called with no pending sync — "
                "skipping version bump (telemetry guard)", self.rank,
            )
        elif sync_failed:
            # Keep _pending_sync_id set so the next successful retry can bump.
            log.warning(
                "Rank %d: sync_id=%d failed — version stays at %d, pending sync retained",
                self.rank, self._pending_sync_id,
                self._weight_versions.version,
            )
        else:
            new_v = self._weight_versions.bump()
            log.info(
                "Rank %d: weight version bumped → %d (sync_id=%d)",
                self.rank, new_v, self._pending_sync_id,
            )
            self._pending_sync_id = None

def _start_deferred_broadcast(self) -> None:
    """Start the deferred XCCL broadcast after vLLM generation completes.

    Called after generate_trajectory_batched() so the broadcast runs during
    GRPO/backward (when vLLM is idle), avoiding Slingshot/XeLink contention
    that doubles vLLM inference latency.

    Posts the manifest to vLLM HERE (not during gather) so vLLM workers
    don't enter broadcast-receive mode until we're ready to send data.
    """
    if not self._vllm_weight_sync:
        return
    args = getattr(self, "_deferred_broadcast_args", None)
    if args is None:
        return
    self._deferred_broadcast_args = None
    import threading as _threading
    import requests

    # WS10 path packs an extra trailing marker. Detect by length so the
    # off-path 8-tuple stays binary identical.
    _ws10_sharded = False
    if len(args) == 9:
        (cpu_batches, meta_json,
         tensors_meta, t0, device, pgs, clients, urls,
         _ws10_sharded) = args
    else:
        (cpu_batches, meta_json,
         tensors_meta, t0, device, pgs, clients, urls) = args

    log.info("Rank %d: starting deferred XCCL broadcast (post-gen)%s",
             self.rank, " [WS10 sharded]" if _ws10_sharded else "")

    post_errors = []
    # Timeout for receive_weights_xccl_streaming HTTP response. This call blocks
    # until vLLM finishes receiving AND loading all weights into the model GPU.
    # For 30B MoE with 128 experts, weight loading (XCCL recv + permute + GPU copy)
    # can take >10 min. 1800s gives 30 min headroom for production-envelope runs.
    _wsync_http_timeout = int(os.environ.get("TORCHTUNE_WSYNC_HTTP_TIMEOUT", "1800"))
    def _post_manifest(url):
        try:
            r = requests.post(
                f"{url}/collective_rpc",
                json={"method": "receive_weights_xccl_streaming",
                      "args": [meta_json]},
                timeout=_wsync_http_timeout,
            )
            if r.status_code != 200:
                post_errors.append(
                    f"streaming init failed ({url}): {r.status_code} {r.text}")
            else:
                result = r.json().get("results", [{}])
                first = result[0] if result else {}
                if isinstance(first, dict) and first.get("status") != "ok":
                    post_errors.append(f"streaming error ({url}): {first}")
        except Exception as e:
            post_errors.append(str(e))

    # WS10: only rank 0 POSTs (it's the only rank with meta_json). Other
    # ranks skip the HTTP path entirely; their job is purely the broadcast
    # of their own per-rank cpu_batches.
    if _ws10_sharded and meta_json is None:
        manifest_threads = []
    else:
        manifest_threads = [
            _threading.Thread(target=_post_manifest, args=(url,), daemon=True)
            for url in urls
        ]
        for mt in manifest_threads:
            mt.start()

    time.sleep(0.3)

    def _bg_xccl_broadcast(
        bg_cpu_batches, bg_manifest_threads, bg_post_errors,
        bg_tensors_meta, bg_t0, bg_device, bg_pgs, bg_clients,
        bg_ws10_sharded=False,
    ):
        try:
            tb_total = 0.0
            # WS10 sharded path is gloo-only on per-rank cross PGs (see
            # _bg_xccl_broadcast WS10 branch below). Don't deref
            # self._wsync_cross_method here — it's only set on rank 0 /
            # the legacy sender pool, but every EP rank in [0..ep_d-1]
            # enters this bg thread under WS10. Default to "gloo" so the
            # post-WS10 legacy code (unreachable in WS10 path) doesn't
            # AttributeError on non-rank-0 ranks.
            cross_method = getattr(self, "_wsync_cross_method", "gloo")
            if bg_ws10_sharded:
                # WS10: per-rank gloo broadcast on the per-rank sharded PG.
                # XCCL+OFI on these PGs has known deadlock risk (WS-Varlen
                # history); gloo TCP is the safe path. Trainer is rank 0
                # within the 2-rank sharded PG.
                for cpu_flat in bg_cpu_batches:
                    tb0 = time.perf_counter()
                    works = [pg.broadcast(cpu_flat, root=0) for pg in bg_pgs]
                    for w in works:
                        w.wait()
                    tb_total += time.perf_counter() - tb0
                log.info(
                    "Rank %d: WS10 sharded gloo broadcast: %d batches × %d "
                    "PG(s) sent (per-rank cross-PG)",
                    self.rank, len(bg_cpu_batches), len(bg_pgs),
                )
            elif cross_method == "gloo":
                for cpu_flat in bg_cpu_batches:
                    tb0 = time.perf_counter()
                    works = [pg.broadcast(cpu_flat, root=0) for pg in bg_pgs]
                    for w in works:
                        w.wait()
                    tb_total += time.perf_counter() - tb0
                log.info("Rank %d: gloo cross broadcast: %d batches × %d replicas sent (CPU, parallel)",
                         self.rank, len(bg_cpu_batches), len(bg_pgs))
            elif cross_method == "xccl_sendrecv":
                max_numel = max(b.numel() for b in bg_cpu_batches)
                if self._xccl_bcast_buf is None or self._xccl_bcast_buf.numel() < max_numel:
                    self._xccl_bcast_buf = torch.empty(
                        max_numel, dtype=torch.bfloat16, device=bg_device)
                    log.info("Rank %d: XCCL send buf allocated: %d elements, data_ptr=0x%x",
                             self.rank, max_numel, self._xccl_bcast_buf.data_ptr())
                gpu_temp = self._xccl_bcast_buf
                for cpu_flat in bg_cpu_batches:
                    n = cpu_flat.numel()
                    gpu_temp[:n].copy_(cpu_flat)
                    tb0 = time.perf_counter()
                    for pg in bg_pgs:
                        pg.send([gpu_temp[:n]], 1, 0).wait()
                    tb_total += time.perf_counter() - tb0
                log.info("Rank %d: XCCL send/recv: %d batches × %d replicas sent (GPU→Slingshot)",
                         self.rank, len(bg_cpu_batches), len(bg_pgs))
            else:
                max_numel = max(b.numel() for b in bg_cpu_batches)
                if self._xccl_bcast_buf is None or self._xccl_bcast_buf.numel() < max_numel:
                    self._xccl_bcast_buf = torch.empty(
                        max_numel, dtype=torch.bfloat16, device=bg_device)
                    log.info("Rank %d: XCCL bcast buf allocated: %d elements, data_ptr=0x%x",
                             self.rank, max_numel, self._xccl_bcast_buf.data_ptr())
                gpu_temp = self._xccl_bcast_buf
                for cpu_flat in bg_cpu_batches:
                    n = cpu_flat.numel()
                    gpu_temp[:n].copy_(cpu_flat)
                    tb0 = time.perf_counter()
                    for pg in bg_pgs:
                        pg.broadcast(gpu_temp[:n], root=0).wait()
                    tb_total += time.perf_counter() - tb0
            del bg_cpu_batches

            for mt in bg_manifest_threads:
                mt.join(timeout=_wsync_http_timeout + 60)

            if bg_post_errors:
                log.error("Rank %d: XCCL async broadcast errors: %s",
                          self.rank, bg_post_errors)
                self._sync_error = RuntimeError(str(bg_post_errors))

            # WS10: only rank 0 owns the HTTP control path. Non-rank-0
            # ranks do not even hold _vllm_clients-as-control objects in
            # the canonical sense (we passed them through for symmetry),
            # but skipping reset_prefix_cache prevents racey duplicate
            # calls and simplifies the receiver-side bookkeeping.
            if not bg_ws10_sharded or self.rank == 0:
                for client in bg_clients:
                    client.reset_prefix_cache()

            _total_gb = sum(e["numel"] for e in bg_tensors_meta) * 2 / 1024**3
            log.info(
                "Rank %d: XCCL deferred broadcast done: %.2f GiB in %.1fs "
                "(bcast=%.1fs %.1f GB/s, total_sync=%.1fs)",
                self.rank, _total_gb, time.perf_counter() - bg_t0,
                tb_total, _total_gb / tb_total if tb_total > 0 else 0,
                time.perf_counter() - bg_t0,
            )
        except Exception as e:
            log.error("Rank %d: XCCL deferred broadcast failed: %s",
                      self.rank, e, exc_info=True)
            self._sync_error = e
        finally:
            self._sync_done_event.set()

    t = _threading.Thread(
        target=_bg_xccl_broadcast, daemon=True, name="xccl_wsync",
        args=(cpu_batches, manifest_threads, post_errors,
              tensors_meta, t0, device, pgs, clients),
        kwargs={"bg_ws10_sharded": _ws10_sharded})
    t.start()


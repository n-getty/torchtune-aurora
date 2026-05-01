# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# XPU-adapted variant of grpo_full_finetune_distributed.py with:
# - Device-agnostic memory ops (XPU / CUDA / CPU)
# - Gradient accumulation with no_sync() for multi-node efficiency
# - Production mode sync skipping (FSDP_PRODUCTION_MODE env var)

import io
import os
import struct
import sys
import threading
import time
from functools import partial
from typing import Any, Optional, Union
from warnings import warn

# -- XPU / XCCL compatibility shim ---------------------------------------------
# On Intel XPU (Aurora), running torchtune's ``__init__.py`` while an XCCL
# process group is active corrupts the L0 USM pointer table, causing every
# subsequent collective to fail.  The root cause is an interaction between
# ``torchtune.__init__``'s ``import torchao`` / env-var setup and the XCCL
# backend's device-context bookkeeping.
#
# Workaround:  pre-register the ``torchtune`` package in ``sys.modules``
# (as a plain ``types.ModuleType``) *before* importing any submodules.
# This prevents Python from executing ``torchtune/__init__.py`` while still
# allowing all ``from torchtune.xxx import ...`` statements to work normally.
#
# Each rank uses ``device_id=xpu:{LOCAL_RANK + offset}`` instead of
# ``ZE_AFFINITY_MASK`` — CCL needs to see all device UUIDs for allreduce.
# --------------------------------------------------------------------------

# 1. Tile affinity — use device_id=xpu:{LOCAL_RANK} (no offset).
#
#    IMPORTANT: Do NOT set ZE_AFFINITY_MASK for multi-rank training.
#    CCL's allreduce needs to see all device UUIDs on the node to build the
#    topology. When ZE_AFFINITY_MASK restricts each rank to 1 tile, CCL only
#    discovers 1 UUID and falls back to the "scheduler path" which lacks
#    ReduceOp.AVG support, causing errors for 3+ ranks.
#
#    Instead, leave ZE_AFFINITY_MASK unset and use device_id=xpu:{LOCAL_RANK}.
#    CCL requires device_id index == LOCAL_RANK for correct topology routing.
#    vLLM is launched separately with ZE_AFFINITY_MASK on higher-numbered tiles
#    (e.g., tile 11) to avoid collisions.
#
#    Valid rank counts: must evenly divide CCL's topology view (works: 2, 4, 6,
#    10, 12; fails: 3, 5, 7, 8, 9, 11). CCL maps ranks across 6 cards × 2 tiles.
#
#    Multi-node: ZE_AFFINITY_MASK must be set (e.g., to LOCAL_RANK) so each rank
#    sees only its tile as xpu:0. With ring algorithms, device_id is not needed.
_use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ and os.environ["ZE_AFFINITY_MASK"] != ""
# With single-tile affinity (e.g. "3"), rank sees 1 tile as xpu:0.
# With multi-tile affinity (e.g. "0,1,...,9"), rank sees N tiles and must select by LOCAL_RANK.
# Without affinity, all 12 tiles visible — also select by LOCAL_RANK.
_affinity_tiles = os.environ.get("ZE_AFFINITY_MASK", "").split(",") if _use_affinity_mask else []
_xpu_device_index = 0 if (len(_affinity_tiles) == 1) else int(os.environ.get("LOCAL_RANK", "0"))

# In dedicated-rank vLLM mode, the last rank (world_size-1) runs vLLM exclusively.
# vLLM calls torch.xpu.mem_get_info() during KV cache allocation, which invokes
# getMemoryInfo on the pluggable allocator. The custom caching allocators
# (usm_caching_alloc.so, usm_caching_alloc_v2.so) do not implement getMemoryInfo,
# causing a NotImplementedError crash. Unset XPU_USM_ALLOC_SO on the vLLM rank
# BEFORE any XPU context is created so the vLLM rank uses the default allocator.
_this_rank = int(os.environ.get("RANK", "0"))
_world_size = int(os.environ.get("WORLD_SIZE", "1"))
if _this_rank == _world_size - 1 and "XPU_USM_ALLOC_SO" in os.environ:
    # vLLM rank: use default XPU allocator (has getMemoryInfo support)
    os.environ.pop("XPU_USM_ALLOC_SO")

import torch

# 2. Pre-register torchtune package to bypass its __init__.py on XPU
import types as _types
import importlib.util as _imp_util

if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
    else:
        # Fallback: assume editable install layout
        _torchtune_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "torchtune",
        )
    if os.path.isdir(_torchtune_path):
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

# 3. Ensure torchao is available (torchtune.__init__ normally checks this)
import torchao  # noqa

from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.datasets import ConcatDataset
from torchtune.dev.rl.generation import generate
from torchtune.dev.rl.rewards import batched_rewards, gene_recall_batched_rewards
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    device_record_memory_history,
    disable_dropout,
    DummyProfiler,
    get_xpu_distributed_backend,
    init_xpu_process_group,
    PROFILER_KEY,
    supports_memory_stats,
)
from torchtune.training.lr_schedulers import get_lr
from torchtune.dev.rl.distributed import (
    install_xpu_patches,
    set_process_groups,
    _apply_split_ac,
    _ep_post_backward_grad_sync,
    _ep_post_backward_grad_sync_xccl,
    _ep_release_fsdp_unsharded_grads,
    _ep_build_grad_release_pg_map,
    _slice_trajectory,
    device_empty_cache,
    _orig_all_reduce,
    _orig_reduce_scatter_tensor,
)
import torchtune.dev.rl.weight_sync as _weight_sync_module
import torchtune.dev.rl.vllm_backend as _vllm_backend_module
from tqdm import tqdm

log = utils.get_logger("DEBUG")
_colocate_vllm_mode = False
install_xpu_patches()


def _async_lookahead_iter(recipe, dataloader):
    """Iterator that overlaps the next batch's vLLM HTTP call with the
    current step's training. Active only when async generation is enabled
    AND the recipe is on rank 0 in vLLM server mode. Otherwise it's a
    transparent passthrough over the dataloader.

    Rank-0 only: a :class:`RolloutProducer` daemon thread calls vLLM HTTP
    for the next batch while the main thread runs the current step's
    fwd/bwd/opt. The producer's payload is stashed in
    ``recipe._pending_async_query_responses`` so generate_trajectory's
    server-mode branch picks it up instead of issuing the HTTP call inline.
    The matching broadcast collective still runs on every rank —
    non-rank-0 ranks see no behavior change.

    Step 2 of the Phase 2 plan: this used to be an inline thread; now
    delegates to :class:`torchtune.dev.rl.async_rollout.RolloutProducer` so
    server-mode and (future) dedicated-rank async share one abstraction.
    """
    is_async = (
        recipe._async_generation_enabled
        and recipe._vllm_mode == "server"
        and recipe._is_rank_zero
    )
    if not is_async:
        yield from dataloader
        return

    from torchtune.dev.rl.async_rollout import RolloutProducer

    def _produce_one(batch):
        tokens = batch["tokens"].to(recipe._device)
        bsz, ctx = tokens.shape
        G = recipe.grpo_samples
        bii = tokens[:, None, :].expand(-1, G, -1).reshape(bsz * G, -1)
        qr = recipe._call_vllm_http(bii, ctx)
        return qr, {}

    _it = iter(dataloader)

    def _next_batch():
        try:
            return next(_it)
        except StopIteration:
            return None

    producer = RolloutProducer(
        produce_fn=_produce_one,
        batch_iter_fn=_next_batch,
        weight_versions=recipe._weight_versions,
        max_staleness=recipe._async_generation_max_staleness,
        name="rollout_producer",
    )
    # Expose for the consumer (telemetry + watchdog readers in Step 3).
    recipe._rollout_producer = producer
    producer.start()
    log.info("Rank 0: rollout producer thread started (max_staleness=%d)",
             recipe._async_generation_max_staleness)
    try:
        for item in producer:
            recipe._pending_async_query_responses = item.batch_meta["rollout_payload"]
            recipe._last_rollout_item = item
            log.info(
                "Rank 0: consumer pop (producer_latency=%.1fs, qsize=%d, w_ver=%d)",
                item.produce_latency_s, producer.qsize(), item.weight_version,
            )
            yield item.batch_meta["batch"]
    finally:
        producer.stop()
        recipe._rollout_producer = None


class GRPOFullFinetuneDistributedXPU(FTRecipeInterface):
    """
    Distributed GRPO full-finetune recipe adapted for Intel XPU (Aurora HPC).

    Key differences from the CUDA-only version:
    - Device-agnostic memory management (empty_cache, memory stats)
    - Gradient accumulation with ``no_sync()`` to reduce AllReduce overhead
    - XPU-safe distributed init (no ``device_id`` for XCCL)
    - Production mode sync skipping via ``FSDP_PRODUCTION_MODE`` env var
    """

    def __init__(self, cfg: DictConfig) -> None:
        _usm_so = os.environ.get("XPU_USM_ALLOC_SO")
        if _usm_so:
            from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
            _usm_alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
            change_current_allocator(_usm_alloc)
            log.info(f"USM arena allocator registered: {_usm_so}")
            # Pluggable allocator doesn't support getDeviceStats/emptyCache.
            # Monkeypatch memory query functions to return 0 instead of crashing.
            torch.xpu.memory_allocated = lambda device=None: 0
            torch.xpu.memory_reserved = lambda device=None: 0
            torch.xpu.max_memory_allocated = lambda device=None: 0
            torch.xpu.max_memory_reserved = lambda device=None: 0
            torch.xpu.reset_peak_memory_stats = lambda device=None: None
            torch.xpu.empty_cache = lambda: None
            torch.xpu.memory_stats = lambda device=None: {}

        # Use the correct XPU device index (LOCAL_RANK + tile offset)
        if cfg.device == "xpu":
            self._device = torch.device(f"xpu:{_xpu_device_index}")
            torch.xpu.set_device(_xpu_device_index)
        else:
            self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        self._output_dir = cfg.output_dir

        # Logging attributes
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and not supports_memory_stats(self._device):
            log.info(
                "log_peak_memory_stats was set to True, however, the device does "
                "not support memory stats. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Initialize the distributed environment
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        # ref_cpu_offload: keep ref model on CPU, policy on GPU (needed for inline gen).
        # Frees ~12 GiB XPU HBM by offloading ref model params to CPU; FSDP2 moves
        # them to GPU on demand during ref forward passes only.
        self._ref_cpu_offload = cfg.get("ref_cpu_offload", False)
        self._disable_prefetch = cfg.get("disable_prefetch", False)
        self._fsdp_diagnostics = cfg.get("fsdp_diagnostics", False)
        self._empty_cache_before_backward = cfg.get("empty_cache_before_backward", False)
        self.distributed_backend = get_xpu_distributed_backend(
            self._device.type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        # Colocated vLLM must be initialized BEFORE the training PG because
        # vLLM creates gloo sub-groups that crash if the default PG was
        # initialized with device_id=xpu:0.
        vllm_mode = cfg.get("vllm_mode", None)
        if cfg.get("vllm_url", None) is not None and vllm_mode is None:
            vllm_mode = "server"
        if vllm_mode in ("colocate", "colocate_sleep"):
            self._init_vllm_early(cfg)
        elif vllm_mode == "dedicated_rank":
            _ded_rank = cfg.get("vllm_dedicated_rank", None)
            _cur_rank = int(os.environ.get("RANK", "0"))
            if _ded_rank is not None and _cur_rank == _ded_rank:
                self._init_vllm_early_dedicated(cfg)

        # If using MPI transport (CCL_ATL_TRANSPORT=mpi), pre-init MPI so CCL can
        # use it. This follows the official Aurora DDP pattern.
        if os.environ.get("CCL_ATL_TRANSPORT") == "mpi":
            try:
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
            except ImportError:
                pass

        if not torch.distributed.is_initialized():
            init_xpu_process_group(self.distributed_backend, device_index=_xpu_device_index)
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        # SDPA backend selection.
        # NOTE: torch.backends.cuda.enable_flash_sdp() is a NO-OP on XPU — those
        # toggles only affect the CUDA SDPA dispatcher (validated 2026-04-30, see
        # CLAUDE.md). The flag is kept for CUDA portability but does nothing on
        # Aurora. The real XPU SDPA fast path is TORCHTUNE_USE_IPEX_VARLEN=1.
        if self._device.type == "xpu" and cfg.get("force_math_sdpa", True):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            if self._is_rank_zero:
                log.info(
                    "force_math_sdpa=True is a no-op on XPU (CUDA-only toggle). "
                    "For the validated XPU SDPA fast path, set TORCHTUNE_USE_IPEX_VARLEN=1."
                )
        elif self._device.type == "cuda" and not cfg.get("force_math_sdpa", True):
            log.info("Rank %d: using default SDPA backends (flash/mem_efficient enabled)", self.rank)

        # Production mode: skip non-essential barriers/synchronize() calls.
        # Multi-node XPU REQUIRES production mode because world-level barriers
        # conflict with FSDP's sub-PG operations on XCCL (both HSDP sub-PGs and
        # FSDP2's per-module sub-communicators trigger broadcast_scaleout failures).
        _is_multinode = self.world_size > int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self._production_mode = (
            os.environ.get("FSDP_PRODUCTION_MODE", "0") == "1"
            or cfg.get("data_parallel_replicate_dim", 1) > 1
            or (_is_multinode and self._device.type == "xpu")
            # dedicated_rank: vLLM rank exits setup() early and never participates
            # in world-PG barriers. Forcing production_mode skips those barriers
            # so training ranks don't deadlock waiting for the vLLM rank.
            or cfg.get("vllm_mode", None) == "dedicated_rank"
        )

        # Training attributes
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._compile = cfg.get("compile", False)
        # dynamic=True uses symbolic shapes to avoid recompilation on
        # variable-length sequences (essential for RL workloads).
        # Default True on XPU (matches PRISM's proven approach).
        self._compile_dynamic = cfg.get(
            "compile_dynamic", True if self._device.type == "xpu" else False
        )

        # Compile + multi-node on XPU: historically deadlocked with oneCCL,
        # but may work with fixed CCL config (WORKER_COUNT=1, MPI transport).
        # Set allow_compile_multinode=True to test.
        if self._compile and self._device.type == "xpu":
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
            if self.world_size > local_world_size:
                if cfg.get("allow_compile_multinode", False):
                    log.warning(
                        "compile + multi-node XPU: EXPERIMENTAL (allow_compile_multinode=True)"
                    )
                else:
                    log.warning(
                        "torch.compile disabled for multi-node XPU. "
                        "Set allow_compile_multinode=True to test."
                    )
                    self._compile = False

        # HSDP: dp_replicate × dp_shard mesh for multi-node
        # dp_replicate=1 (default): pure FSDP across all ranks
        # dp_replicate=NUM_NODES: FSDP within node, DDP across nodes (HSDP)
        self._dp_replicate = cfg.get("data_parallel_replicate_dim", 1)
        if self._dp_replicate > 1:
            from torch.distributed.device_mesh import init_device_mesh
            self._dp_shard = self.world_size // self._dp_replicate
            log.info(
                "HSDP enabled: dp_replicate=%d × dp_shard=%d (world_size=%d)",
                self._dp_replicate, self._dp_shard, self.world_size,
            )
            # Create a simple 2D mesh directly (avoid ParallelDims.build_mesh
            # which creates many submeshes that can deadlock XCCL).
            self._dp_mesh = init_device_mesh(
                self._device.type,
                (self._dp_replicate, self._dp_shard),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
            # Get shard PG for generation early-stopping all_reduce.
            # The shard group = ranks within same node (FSDP shards of same model).
            self._shard_pg = self._dp_mesh.get_group("dp_shard")
            # v57: Create gloo process groups mirroring BOTH dp_replicate AND dp_shard.
            # FSDP2 HSDP grad sync has two CCL steps that both need to be intercepted:
            #   1. reduce_scatter_tensor(grad, group=dp_shard_pg) — our monkey-patch
            #      converts to gloo AllReduce across dp_shard, then scatters.
            #   2. all_reduce(grad_shard, group=dp_replicate_pg) — separate FSDP2 step
            #      to average across replicas; also patched to use gloo.
            # v56 bug: used _GLOO_DP_REP_PG (3 ranks) for step 1 which uses dp_shard (4 ranks).
            # v57 fix: _GLOO_DP_SHARD_PG for step 1, _GLOO_DP_REP_PG for step 2.
            # bound_device_id must be cleared (v56 fix) before any new_group(backend="gloo")
            # to prevent PyTorch from trying pg._get_backend(xpu_device) on the gloo group.
            _n_dp_rep = self._dp_replicate   # 3
            _n_dp_shd = self._dp_shard       # 4 (= ep_degree)
            _GLOO_DP_REP_PG = None
            _GLOO_DP_SHARD_PG = None
            _GLOO_GLOBAL_PG = None
            import torch.distributed.distributed_c10d as _dc10d
            _default_pg = _dc10d._get_default_group()
            _orig_bound_device_id = _default_pg.bound_device_id
            _default_pg.bound_device_id = None
            try:
                # dp_replicate gloo groups (3 ranks each, one per shard position):
                # e.g. shard_idx=0: [0,4,8], shard_idx=1: [1,5,9], etc.
                for _shard_idx in range(_n_dp_shd):
                    _gloo_ranks = [_shard_idx + _n_dp_shd * j for j in range(_n_dp_rep)]
                    _gloo_pg = torch.distributed.new_group(_gloo_ranks, backend="gloo")
                    if self.rank in _gloo_ranks:
                        _GLOO_DP_REP_PG = _gloo_pg
                # dp_shard gloo groups (4 ranks each, one per replicate group):
                # e.g. rep_idx=0: [0,1,2,3], rep_idx=1: [4,5,6,7], rep_idx=2: [8,9,10,11]
                for _rep_idx in range(_n_dp_rep):
                    _gloo_ranks = [_rep_idx * _n_dp_shd + j for j in range(_n_dp_shd)]
                    _gloo_pg = torch.distributed.new_group(_gloo_ranks, backend="gloo")
                    if self.rank in _gloo_ranks:
                        _GLOO_DP_SHARD_PG = _gloo_pg
                    # v152: EP dispatch needs its OWN gloo communicator — separate from
                    # _GLOO_DP_SHARD_PG. FSDP2 grad sync (monkey-patched) uses _GLOO_DP_SHARD_PG
                    # concurrently with EP RS-BWD. Sharing the same gloo communicator causes
                    # sequence number collision → TCP timeout at op #259 (same hang as XCCL).
                    # new_group() with same ranks but called again → independent communicator,
                    # own sequence counters, own TCP connections. No shared state.
                    # v153: 120s timeout was set for fast diagnostics. Qwen3 EP first-run
                    # (2026-04-28) hit the 120s on op #417 RS-FWD on rank 11 — likely the
                    # XPU kernel queue waiting on a slow expert BMM, not gloo network.
                    # Restore the gloo default (1800s) so timing instrumentation can show
                    # us which phase actually took the time.
                    import datetime as _dt
                    _ep_gloo_pg = torch.distributed.new_group(
                        _gloo_ranks, backend="gloo",
                        timeout=_dt.timedelta(seconds=1800),
                    )
                    if self.rank in _gloo_ranks:
                        try:
                            from torchtune.modules.moe import _parallelism as _ep_par
                            _ep_par._GLOO_EP_PG = _ep_gloo_pg
                        except Exception:
                            pass
                # v106: gloo CPU-bounce ONLY for no-grad (ref forward); XCCL for all else.
                # One gloo FWD group per DP replica (4 ranks = EP ranks for that replica).
                # Ref forward runs under torch.no_grad() → _XPUSyncAllToAll.forward checks
                # torch.is_grad_enabled()==False → uses gloo CPU-bounce → avoids
                # ze_handle_manager crash (cpu_offload freshly-allocated XPU buffers ≠ USM).
                # Training forward + AC recompute: is_grad_enabled()==True → XCCL (lockstep).
                # v116: Backward uses XCCL (reverted from v115 gloo BWD).
                #   v115 hypothesis was wrong: use_reentrant=True serializes WITHIN a rank
                #   but gloo P2P is NOT a collective — provides no cross-rank synchronization.
                #   Different EP ranks can be at different backward layers → gloo size mismatch.
                #   v114 crash (SIGSEGV call#27-30, consistent splits) was OOM, not FSDP2+XCCL
                #   conflict. XCCL backward is safe: it IS a collective (forces lockstep).
                #   v116 fix: keep XCCL backward + halve fbs (4→2) to reduce backward peak.
                #   _XPU_A2A_BWD_GLOO_GROUP remains None → backward uses XCCL path.
                # v127: Backward uses gloo CPU-bounce (same group as FWD).
                #   v116-v126 XCCL backward crashes with SIGSEGV at call#0 (v122-v126) or
                #   call#42 (v116-v119). All attempts to fix XCCL backward have failed:
                #   IPC handle caching (v124), naive AllGather (v125), requires_grad_(False)
                #   for ref (v126). The SIGSEGV is in XPU driver XCCL code, not Python.
                #   v127 fix: force gloo CPU-bounce for backward AllToAll too.
                #   Why v115 failed but v127 should work:
                #     v115 attempted separate BWD gloo group (v101 implementation). This
                #     deadlocked because AC recompute forward AllToAll and backward AllToAll
                #     were concurrent on different TCP socket pairs.
                #   v127 reuses the same FWD group for backward. This is safe because:
                #     (1) AC recompute is CACHED (returns cached_output, no communication).
                #         The FWD gloo group is completely idle during backward.
                #     (2) With use_reentrant=True, backward is sequential per rank — all
                #         12 ranks proceed in lockstep (same barrier discipline as forward).
                #     (3) The same P2P send/recv split logic in backward (ctx.output_splits →
                #         ctx.input_splits) is proven correct (v101 implementation).
                #   Setting _XPU_A2A_BWD_GLOO_GROUP = FWD group activates gloo CPU-bounce
                #   in _XPUSyncAllToAll.backward instead of XCCL all_to_all_single.
                # AllGather+ReduceScatter EP dispatch (v137+): no AllToAll gloo groups needed.
                # EP communication uses dist.all_gather_into_tensor and dist.reduce_scatter_tensor
                # (XCCL) which are natively optimized on Aurora and have correct autograd.
                # Global gloo group (all ranks): used for barrier before dp_replicate
                # AllReduce to synchronize EP groups (v58 fix for gloo size mismatch).
                _GLOO_GLOBAL_PG = torch.distributed.new_group(
                    list(range(self.world_size)), backend="gloo"
                )
                # v150: inject global gloo group into _parallelism for cross-EP serialization.
                # All 12 ranks form one gloo group; barrier before/after each XCCL EP
                # collective prevents concurrent XCCL ops on different ep_pg groups
                # from cross-contaminating each other's OFI CQ events.
                try:
                    from torchtune.modules.moe import _parallelism as _ep_par
                    _ep_par._GLOO_GLOBAL_PG = _GLOO_GLOBAL_PG
                except Exception:
                    pass
            finally:
                _default_pg.bound_device_id = _orig_bound_device_id
            # v75: Get the XCCL dp_replicate process group from the device mesh.
            _XCCL_DP_REP_PG = self._dp_mesh.get_group("dp_replicate")
            # Register PG handles in distributed.py so patch functions can use them.
            set_process_groups(
                _GLOO_DP_REP_PG, _GLOO_DP_SHARD_PG, _GLOO_GLOBAL_PG, _XCCL_DP_REP_PG,
                _n_dp_rep, _n_dp_shd,
            )
            log.info(
                "Created gloo dp_replicate (%d-rank), dp_shard (%d-rank), global gloo. "
                "EP dispatch: AllGather+ReduceScatter (no AllToAll gloo groups). "
                "dp_rep_pg=%s dp_shard_pg=%s global_pg=%s xccl_rep_pg=%s",
                _n_dp_rep, _n_dp_shd,
                _GLOO_DP_REP_PG, _GLOO_DP_SHARD_PG,
                _GLOO_GLOBAL_PG, _XCCL_DP_REP_PG,
            )
            # Shard group leader = local rank 0 within each shard group (= each node).
            # For replicated vLLM, each shard leader talks to its local vLLM.
            self._shard_rank = torch.distributed.get_rank(self._shard_pg)
            self._is_shard_leader = (self._shard_rank == 0)
            # Global rank of this shard group's leader (for broadcast src).
            # torch.distributed.broadcast requires global rank as src, even with group.
            shard_ranks = torch.distributed.get_process_group_ranks(self._shard_pg)
            self._shard_leader_global_rank = shard_ranks[0]
            self._dp_degree = self.world_size
            self._dp_rank = self.rank
        else:
            self._dp_shard = self.world_size
            self._dp_degree = self.world_size
            self._dp_rank = self.rank
            self._dp_mesh = None
            self._shard_pg = None
            self._shard_rank = self.rank
            self._is_shard_leader = self._is_rank_zero

            # v206: non-HSDP single-replicate path also needs a gloo group for the
            # _xpu_reduce_scatter_via_allreduce patch to use as CPU-bounce backend.
            # Without this, the patch falls through to its XCCL fallback (line 306)
            # which calls all_reduce on freshly-allocated XPU grad buffers — exactly
            # the v44-v58 failure pattern. Step 0 sometimes works (warmup buffers),
            # but step 1 reliably hits UR:40 with vLLM colocated.
            # (Globals already declared at line 889 in HSDP branch — same function,
            # so Python's static analysis already treats them as global here.)
            _DP_SHARD_DEGREE = self.world_size
            _DP_REP_DEGREE = 1
            import torch.distributed.distributed_c10d as _dc10d
            _default_pg = _dc10d._get_default_group()
            _orig_bound_device_id = _default_pg.bound_device_id
            _default_pg.bound_device_id = None
            try:
                _all_ranks = list(range(self.world_size))
                _GLOO_DP_SHARD_PG = torch.distributed.new_group(
                    _all_ranks, backend="gloo",
                )
                _GLOO_GLOBAL_PG = _GLOO_DP_SHARD_PG
            finally:
                _default_pg.bound_device_id = _orig_bound_device_id
            # Stash on self so the EP single-replica branch below can call
            # set_process_groups(...) with the right handle without re-creating.
            self._gloo_dp_shard_pg = _GLOO_DP_SHARD_PG
            self._gloo_global_pg = _GLOO_GLOBAL_PG
            log.info(
                "v206: non-HSDP gloo PG initialized (world=%d) for "
                "_xpu_reduce_scatter_via_allreduce CPU-bounce path",
                self.world_size,
            )

        # Expert Parallelism (reuses dp_shard process group — no new communicators)
        self._expert_parallel_degree = cfg.get("expert_parallel_degree", 1)
        self._expert_cpu_offload = cfg.get("expert_cpu_offload", False)
        if self._expert_parallel_degree > 1:
            # Single-replica EP path (dp_replicate=1, world == ep == dp_shard):
            # the non-HSDP branch above leaves _dp_mesh=None and _shard_pg=None.
            # EP reads ep_mesh = self._dp_mesh["dp_shard"] (line ~1173) and the
            # grad-norm path reads self._shard_pg, so we need a 1D dp_shard mesh
            # before the asserts below.
            if self._dp_mesh is None:
                from torch.distributed.device_mesh import init_device_mesh
                self._dp_mesh = init_device_mesh(
                    self._device.type,
                    (self.world_size,),
                    mesh_dim_names=("dp_shard",),
                )
                self._shard_pg = self._dp_mesh.get_group("dp_shard")
                self._shard_rank = torch.distributed.get_rank(self._shard_pg)
                self._is_shard_leader = (self._shard_rank == 0)
                shard_ranks = torch.distributed.get_process_group_ranks(self._shard_pg)
                self._shard_leader_global_rank = shard_ranks[0]
                # Register the existing gloo PG (built in the non-HSDP branch above
                # under the name _GLOO_DP_SHARD_PG, world-sized) into distributed.py
                # so the v9 helper barrier and reduce-scatter patches see it.
                # No dp_replicate gloo/XCCL groups exist at dp_replicate=1 — that's
                # fine because every consumer of those is gated on dp_replicate > 1.
                try:
                    set_process_groups(
                        None,                              # gloo_dp_rep_pg — unused at dp_replicate=1
                        getattr(self, "_gloo_dp_shard_pg", None),  # world-sized gloo
                        getattr(self, "_gloo_global_pg", None),    # same group
                        None,                              # xccl_dp_rep_pg — unused at dp_replicate=1
                        1,                                 # dp_rep_degree
                        self.world_size,                   # dp_shard_degree
                    )
                except Exception as _e:
                    log.warning(
                        "EP single-replica: set_process_groups skipped (%s) — v9 "
                        "helper barrier may be a no-op.", _e,
                    )
                log.info(
                    "EP single-replica: built 1D dp_shard mesh (world=%d), "
                    "_dp_mesh.ndim=%d, _shard_pg established.",
                    self.world_size, self._dp_mesh.ndim,
                )
            assert self._dp_replicate >= 1, (
                f"expert_parallel_degree > 1 requires data_parallel_replicate_dim >= 1 "
                f"(got {self._dp_replicate})"
            )
            assert self._expert_parallel_degree == self._dp_shard, (
                f"expert_parallel_degree={self._expert_parallel_degree} must equal "
                f"dp_shard={self._dp_shard} (EP reuses the dp_shard process group)"
            )
            log.info(
                "Expert Parallelism enabled: ep_degree=%d (reuses dp_shard group)",
                self._expert_parallel_degree,
            )

        # Gradient accumulation
        self._gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

        # vLLM generation (optional — None means use native generation)
        # mode: "server" = external vLLM HTTP server on separate tile(s)
        #        "colocate" = in-process vLLM engine per rank (TRL-style)
        #        "colocate_sleep" = colocate with sleep/wake memory management
        self._vllm_mode = cfg.get("vllm_mode", None)  # None, "server", "colocate", "colocate_sleep", "dedicated_rank"
        self._vllm_dedicated_rank = cfg.get("vllm_dedicated_rank", None)
        self._is_vllm_rank = (
            self._vllm_mode == "dedicated_rank"
            and self._vllm_dedicated_rank is not None
            and self.rank == self._vllm_dedicated_rank
        )
        self._vllm_url = cfg.get("vllm_url", None)
        # Multi-URL support: comma-separated URLs for DP vLLM replicas
        if self._vllm_url and "," in self._vllm_url:
            self._vllm_urls = [u.strip() for u in self._vllm_url.split(",")]
        elif self._vllm_url:
            self._vllm_urls = [self._vllm_url]
        else:
            self._vllm_urls = []
        self._vllm_group_port = cfg.get("vllm_group_port", 51216)
        self._vllm_weight_sync = cfg.get("vllm_weight_sync", True)
        self._vllm_weight_sync_interval = cfg.get("vllm_weight_sync_interval", 1)
        # Weight sync method: "raw_bytes" (file-based, default) or "shm" (POSIX shared memory,
        # faster for large models — eliminates the file read step on the vLLM side).
        self._vllm_weight_sync_method = cfg.get("vllm_weight_sync_method", "raw_bytes")

        # Async generation (Phase 0/1/2) — see docs plan eventual-juggling-prism.md.
        # Phase 0: `always_compute_rollout_logprobs` forces the policy fwd at rollout
        # time even when ppo_epochs == 1, so trajectory.logprobs is the rollout-time
        # snapshot. Required for any off-policy correction; off by default.
        self._always_compute_rollout_logprobs = cfg.get(
            "always_compute_rollout_logprobs", False
        )
        _async_cfg = cfg.get("async_generation", {}) or {}
        self._async_generation_enabled = bool(_async_cfg.get("enabled", False))
        self._async_generation_max_staleness = int(_async_cfg.get("max_staleness", 1))
        # Behavior-policy correctness guard.
        # Async lookahead generates batch N+1 under vLLM weights v_k while the
        # trainer is still on step N. After step N's optimizer.step + weight
        # sync, vLLM holds v_{k+1}. The trainer then consumes batch N+1 and
        # recomputes pi_old_logprobs on the *current* training model (≈ v_{k+1}),
        # but the rollout itself was sampled under v_k. So even at staleness=1
        # pi_old_logprobs are biased and GRPO IS ratios are not exactly 1.
        # The bias is strictly larger at staleness > 1 — hard-cap there.
        # The real fix is to capture rollout-time logprobs from vLLM (or to
        # recompute against a frozen copy of the policy version that generated
        # the rollout); until then, async should be considered EXPERIMENTAL.
        if self._async_generation_enabled and self._async_generation_max_staleness > 1:
            raise ValueError(
                "async_generation.max_staleness>1 is not safe yet: "
                "rollout logprobs are recomputed on current training weights, so "
                "pi_old_logprobs will not match the behavior policy that produced "
                "the rollout (biased GRPO IS ratios). "
                "Either set max_staleness=1 or implement vLLM-time logprob capture."
            )
        if self._async_generation_enabled:
            log.warning(
                "async_generation enabled (max_staleness=%d): EXPERIMENTAL. "
                "pi_old_logprobs are recomputed on the current training model, "
                "but the rollout was sampled under the *previous* vLLM weight "
                "version (v_k vs v_{k+1} after the step-N sync). GRPO IS ratios "
                "carry a small bias even at staleness=1; the bias grows with "
                "staleness. Do NOT treat async as on-policy until vLLM-time "
                "logprob capture (or frozen-policy recompute) is implemented.",
                self._async_generation_max_staleness,
            )
        # Rollout-time logprobs are required when off-policy by k>=1 OR explicitly
        # requested. ppo_epochs>1 already computes them via the existing branch.
        self._compute_rollout_logprobs_required = (
            self._always_compute_rollout_logprobs
            or self._async_generation_enabled
        )
        self._vllm_clients = []  # initialized in setup() if vllm_mode == "server"
        self._vllm_client = None  # backward compat: first client
        # _vllm_llm may already be set by _init_vllm_early() for colocate mode
        if not hasattr(self, "_vllm_llm"):
            self._vllm_llm = None
        self._vllm_gpu_memory_utilization = cfg.get("vllm_gpu_memory_utilization", 0.3)
        self._vllm_max_model_len = cfg.get("vllm_max_model_len", 2048)
        self._vllm_tp_size = cfg.get("vllm_tensor_parallel_size", 1)

        # Backward compat: if vllm_url is set but vllm_mode is not, infer "server"
        if self._vllm_url is not None and self._vllm_mode is None:
            self._vllm_mode = "server"

        # Eval config
        self._eval_every_n_steps = cfg.get("eval_every_n_steps", 0)
        self._eval_max_examples = cfg.get("eval_max_examples", 50)
        self._eval_grpo_samples = cfg.get("eval_grpo_samples", None)
        self._eval_enabled = False  # set in setup() if eval_dataset configured

        # Step-based checkpointing
        self._save_every_n_steps = cfg.get("save_every_n_steps", None)
        self._save_final_checkpoint = cfg.get("save_final_checkpoint", True)

        # Recipe state attributes
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = 0
        self._epochs_run = 0
        # torch.Generator does not support XPU — use CPU generator instead
        _rng_device = self._device if self._device.type == "cuda" else torch.device("cpu")
        self._rng = torch.Generator(_rng_device).manual_seed(self.seed)

    _init_vllm_early = _vllm_backend_module._init_vllm_early
    _init_vllm_early_dedicated = _vllm_backend_module._init_vllm_early_dedicated
    _init_vllm_tp1 = _vllm_backend_module._init_vllm_tp1
    _init_vllm_tp = _vllm_backend_module._init_vllm_tp
    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        if self._is_vllm_rank:
            # Rank runs as dedicated vLLM generation server — skip all training setup.
            # Create _training_pg + _wsync_pg + _gen_pg + seed gen params in same
            # new_group order as training ranks (all ranks must call new_group
            # identically or deadlock).
            self._setup_dedicated_vllm_rank(cfg)
            return
        # Dedicated_rank training ranks must create the same PGs in lockstep with
        # the vLLM rank BEFORE any other default-world-PG collective (barrier,
        # model load, etc.). Otherwise training ranks block on a later barrier
        # while the vLLM rank blocks at new_group().
        if self._vllm_mode == "dedicated_rank":
            self._setup_dedicated_training_pgs(cfg)
            # FSDP2 sharding mesh must exclude the vLLM rank — otherwise the
            # AllGather targets the world PG (12 ranks) and the vLLM rank,
            # which is not in the FSDP forward path, causes a deadlock at the
            # first policy-fwd AllGather.
            from torch.distributed.device_mesh import DeviceMesh
            self._dp_mesh = DeviceMesh.from_group(
                self._training_pg, self._device.type,
            )
            log.info(
                "Rank %d: FSDP2 dp_mesh built from _training_pg (size=%d)",
                self.rank, self.world_size - 1,
            )
        # Setup model to train
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        # Reshard parameters after forward to avoid keeping the full unsharded
        # model in memory. For 31B+ with FSDP-10/12, full unshard = 62+ GiB
        # which exceeds tile capacity. Always reshard except in colocate mode
        # (non-sleep) where model stays on GPU anyway.
        # Can be overridden via config: reshard_after_forward: false
        reshard_policy = cfg.get("reshard_after_forward",
                                 self._vllm_mode != "colocate")
        # MEMPROBE: baseline (pre-policy). Default-off; opt in via TORCHTUNE_MEM_PROBE=1.
        # Imports the experiment-local probe at experiments/multinode_32b/mem_probe.py.
        _dump_mem_init = None
        if os.environ.get("TORCHTUNE_MEM_PROBE"):
            try:
                import sys as _sys
                _mp_path = "/lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b"
                if _mp_path not in _sys.path:
                    _sys.path.insert(0, _mp_path)
                from mem_probe import dump_mem as _dump_mem_init
                _dump_mem_init("INIT pre-policy")
            except Exception as _e:
                log.warning("mem_probe pre-policy failed: %r", _e)
                _dump_mem_init = None
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_sd=checkpoint_dict[training.MODEL_KEY],
            reshard_after_forward=reshard_policy,
        )
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-policy")
            except Exception:
                pass
        # Free the policy state-dict references that load_from_full_model_state_dict
        # nullified internally (release_sd=True). Without this, the checkpoint_dict
        # outer keys + the FullModelHFCheckpointer's own buffers stay live on rank 0.
        # Test T showed rank 0/1 carry +10 GiB external HBM through training; this is
        # the targeted release.
        # Capture opt/dataloader state BEFORE clearing — they are consumed below
        # by _setup_optimizer / _setup_data and must survive the cleanup.
        self._opt_state_dict = (
            checkpoint_dict.get(training.OPT_KEY)
            if self._resume_from_checkpoint
            else None
        )
        self._dataloader_state_dict = (
            checkpoint_dict.get(training.DATALOADER_KEY)
            if self._resume_from_checkpoint
            else None
        )
        import gc as _gc_post_policy
        try:
            # Drop only MODEL_KEY (giant) — keep opt/dl refs we just hoisted.
            if training.MODEL_KEY in checkpoint_dict:
                checkpoint_dict[training.MODEL_KEY] = None
            checkpoint_dict.clear()
        except Exception:
            pass
        del checkpoint_dict
        _gc_post_policy.collect()
        if self._device.type == "xpu":
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-policy-cleanup")
            except Exception:
                pass
        # Setup reference model
        ref_checkpoint_dict = self.load_checkpoint(
            cfg_checkpointer=cfg.ref_checkpointer
        )
        # ref model: use ref_cpu_offload if set (saves ~12 GiB XPU HBM).
        # Policy model stays on GPU for inline generation; ref can live on CPU.
        _ref_offload = self.fsdp_cpu_offload or self._ref_cpu_offload
        self._ref_model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=_ref_offload,
            model_sd=ref_checkpoint_dict[training.MODEL_KEY],
            eval_mode=True,
            reshard_after_forward=True,
        )
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-ref")
            except Exception:
                pass
        # Free ref state-dict references (mirror of policy cleanup above).
        try:
            ref_checkpoint_dict.clear()
        except Exception:
            pass
        del ref_checkpoint_dict
        _gc_post_policy.collect()
        if self._device.type == "xpu":
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-ref-cleanup")
            except Exception:
                pass
        if _ref_offload and not getattr(self, "_ref_no_fsdp2", False):
            # FSDP2 cpu_offload manages parameters only; registered buffers
            # (layer_scalar, rope cache, etc.) are left on CPU after setup.
            # Move them to XPU so they don't cause device mismatch against XPU
            # inputs during ref forward passes. Params still live on CPU and are
            # fetched to XPU on-demand by FSDP2's param-fetch hook.
            # v120: skip for _ref_no_fsdp2 case — the whole model (params+buffers)
            # is on CPU; generate_trajectory() does model.to(device) before fwd.
            for _buf_name, _buf in self._ref_model.named_buffers():
                if _buf is not None and _buf.device.type == "cpu":
                    _buf.data = _buf.data.to(self._device)
            if self._is_rank_zero:
                log.info(
                    "ref_cpu_offload: moved registered buffers to XPU "
                    "(params remain on CPU for FSDP2 on-demand fetch)"
                )
        # Skip barrier on multi-node (XCCL sub-communicator creation deadlocks
        # on 2D mesh). FSDP will implicitly sync during first forward pass.
        if not self._production_mode:
            torch.distributed.barrier()

        # Utilize the same tokenizer for both models (hack)
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=self._opt_state_dict,
        )
        self._opt_state_dict = None  # release reference once consumed

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        # Detect chunked output loss (takes raw logits instead of logprobs)
        self._use_chunked_loss = hasattr(self._loss_fn, "num_output_chunks")
        if self._use_chunked_loss:
            log.info(
                "Using chunked output loss with %d chunks",
                self._loss_fn.num_output_chunks,
            )
        if self._compile:
            _saved_tcd = os.environ.pop("TORCH_COMPILE_DISABLE", None)
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)
            if _saved_tcd is not None:
                os.environ["TORCH_COMPILE_DISABLE"] = _saved_tcd

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.get(
            "collate_fn", "torchtune.dev.rl.data.padded_collate_rl"
        )
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            dataloader_state_dict=self._dataloader_state_dict,
        )
        self._dataloader_state_dict = None  # release reference once consumed

        # Setup eval dataset (if configured)
        cfg_eval_dataset = cfg.get("eval_dataset", None)
        if cfg_eval_dataset is not None and self._eval_every_n_steps > 0:
            eval_ds = config.instantiate(cfg_eval_dataset, self._tokenizer)
            max_ex = min(self._eval_max_examples, len(eval_ds))
            self._eval_examples = [eval_ds[i] for i in range(max_ex)]
            self._eval_enabled = True
            if self._is_rank_zero:
                log.info("Eval dataset loaded: %d examples (from %d total), eval every %d steps",
                         max_ex, len(eval_ds), self._eval_every_n_steps)
        else:
            self._eval_examples = []

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        self._steps_per_epoch = len(self._dataloader)
        self.global_step = self._epochs_run * self._steps_per_epoch

        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # RL params
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size

        # Reward mode: "math" (default) or "gene_recall"
        self._reward_mode = cfg.get("reward_mode", "math")
        self._gene_reward_metric = cfg.get("gene_reward_metric", "f1")

        # Honor cfg.reward_functions for math mode when declared. None falls
        # back to the legacy hardcoded batched_rewards() path.
        self._cfg_reward_functions = None
        if self._reward_mode == "math" and cfg.get("reward_functions"):
            from torchtune import config as _tt_config
            self._cfg_reward_functions = [
                _tt_config.instantiate(fn) for fn in cfg.reward_functions
            ]
            log.info(
                "Reward: using %d cfg.reward_functions (%s)",
                len(self._cfg_reward_functions),
                [type(f).__name__ for f in self._cfg_reward_functions],
            )

        # Sequence packing: bin-pack variable-length sequences to eliminate
        # padding waste in the training forward/backward pass.
        self._enable_packing = cfg.get("enable_packing", False)
        if self._enable_packing:
            log.info("Sequence packing enabled for GRPO training forward/backward")

        self._ppo_epochs = cfg.ppo_epochs
        self._save_every_n_epochs = cfg.save_every_n_epochs
        self._total_steps = cfg.num_steps

        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer, "stop_tokens"):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

        # --- vLLM initialization ---
        if self._vllm_mode == "server":
            self._setup_vllm_server_mode()
        elif self._vllm_mode in ("colocate", "colocate_sleep"):
            self._setup_vllm_colocate_mode(cfg)
            # Use synchronize-only mode for device_empty_cache — the full
            # torch.xpu.empty_cache() can deadlock when vLLM engines are
            # present alongside FSDP collectives on the same XPU tile.
            global _colocate_vllm_mode
            _colocate_vllm_mode = True
        # dedicated_rank PG setup happens at the top of setup() (in lockstep with
        # the vLLM rank), not here.

    _setup_vllm_server_mode = _vllm_backend_module._setup_vllm_server_mode
    _setup_vllm_colocate_mode = _vllm_backend_module._setup_vllm_colocate_mode
    _setup_dedicated_vllm_rank = _vllm_backend_module._setup_dedicated_vllm_rank
    _setup_dedicated_training_pgs = _vllm_backend_module._setup_dedicated_training_pgs
    def _build_tune_to_hf_map(self):
        """Build torchtune -> HuggingFace parameter name mapping for weight sync.

        Uses named_parameters() instead of state_dict() to avoid triggering
        FSDP1's AllGather collective (which would deadlock if only shard leaders
        call this function).
        """
        from torchtune.models.convert_weights import get_mapped_key
        from torchtune.training.checkpointing._utils import ModelType

        # Select the model-specific _FROM_HF map based on checkpointer model_type.
        # This avoids hardcoding Qwen2's map and works for Gemma4, etc.
        _model_type = getattr(self._checkpointer, "_model_type", None) if self._checkpointer is not None else None
        if _model_type == ModelType.QWEN3_MOE:
            from torchtune.models.qwen3_moe._convert_weights import (
                build_tune_to_hf_map_moe,
            )
            self._tune_to_hf_map = build_tune_to_hf_map_moe(
                self._model.named_parameters()
            )
            return
        elif _model_type == ModelType.GEMMA4:
            from torchtune.models.gemma4._convert_weights import _GEMMA4_FROM_HF as _FROM_HF
        elif _model_type == ModelType.GEMMA2:
            from torchtune.models.gemma2._convert_weights import _FROM_HF
        elif _model_type in (None, ModelType.QWEN2):
            from torchtune.models.qwen2._convert_weights import _FROM_HF
        elif _model_type == ModelType.QWEN3:
            from torchtune.models.qwen3._convert_weights import _FROM_HF
        else:
            from torchtune.models.qwen2._convert_weights import _FROM_HF
            log.warning(
                "Unknown model type %s for weight sync mapping, falling back to Qwen2 _FROM_HF",
                _model_type,
            )
        inverted = {v: k for k, v in _FROM_HF.items() if v is not None}
        self._tune_to_hf_map = {}
        for tune_name, _ in self._model.named_parameters():
            clean_name = tune_name.replace("_fsdp_wrapped_module.", "")
            clean_name = clean_name.replace("_checkpoint_wrapped_module.", "")
            self._tune_to_hf_map[clean_name] = get_mapped_key(
                clean_name, inverted
            )

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                log.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        optimizer = self._optimizer

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]
                self.profiler_num_cycles = profiler_cfg["num_cycles"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        fsdp_cpu_offload: bool,
        model_sd: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        eval_mode: bool = False,
        reshard_after_forward: bool = True,
    ) -> tuple[nn.Module, nn.Module]:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``

        HSDP mode (dp_replicate > 1):
           Uses FSDP1 FullyShardedDataParallel with HYBRID_SHARD strategy.
           FSDP2's per-layer fully_shard() creates too many XCCL sub-communicators
           which hang on Aurora XPU. FSDP1's single wrapping call avoids this.
        """
        # HSDP mode: use FSDP1 path for small models (fits on single tile).
        # For large models (>60 GiB), FSDP1 OOMs during flatten (needs full model
        # on device + flat buffer simultaneously). Fall through to FSDP2 FULL_SHARD
        # which uses meta device init and never materializes the full model.
        # FSDP2 FULL_SHARD across all ranks (no HSDP mesh) avoids the sub-PG deadlock.
        model_bytes = sum(v.numel() * v.element_size() for v in model_sd.values())
        model_gib = model_bytes / (1024 ** 3)
        # EP requires FSDP2: FSDP1 flattening is incompatible with post-load EP weight slicing.
        _ep_active = self._expert_parallel_degree > 1 and self._dp_mesh is not None
        if self._dp_replicate > 1 and model_gib < 50.0 and not _ep_active:
            self._use_fsdp1 = True
            return self._setup_model_fsdp1_hsdp(
                cfg_model=cfg_model,
                enable_activation_checkpointing=enable_activation_checkpointing,
                model_sd=model_sd,
                eval_mode=eval_mode,
                reshard_after_forward=reshard_after_forward,
            )
        elif self._dp_replicate > 1:
            reason = "EP active" if _ep_active else f"model too large ({model_gib:.1f} GiB)"
            utils.log_rank_zero(
                log,
                f"Using FSDP2 with HSDP mesh ({reason}): "
                f"dp_replicate={self._dp_replicate} × dp_shard={self._dp_shard}.",
            )

        self._use_fsdp1 = False
        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if eval_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        # v120: early-return path for ref model when EP active + ref_cpu_offload.
        # At this point the model is on meta device with clean (pre-AC, pre-FSDP2) param names
        # that match model_sd keys directly. We skip all EP/AC/FSDP2 wrapping and do a plain
        # to_empty('cpu') + load_state_dict. The expert params in model_sd are pre-sliced later
        # (below), but here we do it before the early return since we won't reach that code.
        # Ref forward uses gloo AllToAll (no_grad path) — no EP FSDP2 needed.
        if self._compile:
            # Temporarily allow torch.compile (TORCH_COMPILE_DISABLE=1 set for vLLM)
            _saved_tcd = os.environ.pop("TORCH_COMPILE_DISABLE", None)
            training.compile_model(
                model, verbose=self._is_rank_zero, dynamic=self._compile_dynamic
            )
            if _saved_tcd is not None:
                os.environ["TORCH_COMPILE_DISABLE"] = _saved_tcd

        # EP v42: Pre-shrink expert meta params BEFORE activation checkpointing wrapping.
        # v41 bug: meta param shrinking was done AFTER set_activation_checkpointing, which
        # inserts "_checkpoint_wrapped_module" into module names. model_sd uses clean names
        # (e.g. "layers.0.moe_block.experts.gate_proj") but named_modules() after AC wrapping
        # yields "layers.0._checkpoint_wrapped_module.moe_block.experts.gate_proj" → 0 keys
        # matched in model_sd pre-slicing → full [128,...] tensors loaded into [32,...] params
        # → "start (0) + length (128) exceeds dimension size (32)".
        # v42 fix: shrink meta params BEFORE AC wrapping so names match model_sd keys.
        _ep_active = self._expert_parallel_degree > 1 and self._dp_mesh is not None
        _expert_param_names: set = set()  # expert param full names for model_sd pre-slicing
        if _ep_active:
            from torchtune.modules.moe.experts import GroupedExperts as _GE_pre
            from torchtune.models.qwen3_moe._experts import GroupedExpertsHF as _GEHF_pre
            _expert_classes = (_GE_pre, _GEHF_pre)
            ep_mesh = self._dp_mesh["dp_shard"]  # 4-rank submesh per DP replica
            _ep_rank = ep_mesh.get_local_rank()
            _ep_degree = ep_mesh.shape[0]
            # Collect expert param names and pre-shrink meta params from [128,...] → [32,...].
            # At this point model has original (clean) module names matching model_sd keys.
            for _ename, _emod in model.named_modules():
                if not (_ename.endswith(".experts") and isinstance(_emod, _expert_classes)):
                    continue
                for _pname, _param in list(_emod.named_parameters(recurse=False)):
                    _full_shape = _param.shape
                    assert _full_shape[0] % _ep_degree == 0, (
                        f"num_experts ({_full_shape[0]}) not divisible by ep_degree ({_ep_degree})"
                    )
                    _n_local = _full_shape[0] // _ep_degree
                    _new_shape = torch.Size([_n_local] + list(_full_shape[1:]))
                    setattr(_emod, _pname, nn.Parameter(
                        torch.empty(_new_shape, dtype=_param.dtype, device="meta"),
                        requires_grad=_param.requires_grad,
                    ))
                    _expert_param_names.add(f"{_ename}.{_pname}")
            utils.log_rank_zero(
                log,
                f"EP v42: pre-shrunk {len(_expert_param_names)} expert meta params "
                f"[{_full_shape[0]},...] → [{_n_local},...] "
                f"(EP rank {_ep_rank}/{_ep_degree}, before AC wrapping)",
            )

        if enable_activation_checkpointing:
            _n_moe_self_ac = _apply_split_ac(model)
            utils.log_rank_zero(
                log,
                f"v158 split AC: {_n_moe_self_ac} MoE layers self-checkpoint "
                f"(MoE outside AC); other layers wrapped by apply_activation_checkpointing.",
            )

        # Expert Parallelism: install AllGather/ReduceScatter dispatch on expert modules.
        # Must happen BEFORE shard_model so EP metadata is attached to original modules.
        # Dispatch on model architecture: Gemma4 uses GroupedExperts under .moe_block.experts;
        # Qwen3 MoE uses GroupedExpertsHF under .mlp.experts. Both plans return the same
        # dict[str, ExpertParallel] shape, so the rest of the path is identical.
        if _ep_active:
            from torch.distributed.tensor.parallel import parallelize_module
            from torchtune.models.gemma4._parallelism import gemma4_ep_plan
            from torchtune.models.qwen3_moe._parallelism import qwen3_moe_ep_plan
            ep_plan = gemma4_ep_plan(model)
            if not ep_plan:
                ep_plan = qwen3_moe_ep_plan(model)
            if ep_plan:
                parallelize_module(model, ep_mesh, ep_plan)
                utils.log_rank_zero(
                    log,
                    f"EP={self._expert_parallel_degree}: registered EP dispatch on "
                    f"{len(ep_plan)} expert module(s)",
                )
            else:
                utils.log_rank_zero(
                    log,
                    f"EP={self._expert_parallel_degree}: NO expert modules matched any "
                    f"known EP plan (gemma4, qwen3_moe). Check model architecture.",
                )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # Initialize here so it's defined even when _ep_active=False (EP=1 path).
        _expert_cpu_offload = False

        if _ep_active:
            # EP model setup (v40: expert FSDP2 on dp_replicate, not ep_mesh):
            # Expert params are EP-partitioned (32/rank). Previously we wrapped them
            # with FSDP2 on ep_mesh (ep_pg, 4-rank). This caused:
            # - v18-v31: OFI EPERM — AllToAll backward contaminates ep_pg's OFI CQ;
            #   FSDP2 all_gather on same ep_pg reads stale error.
            # - v32-v39: separate fsdp2_ep_pg (same 4 EP ranks, different CCL comm)
            #   fixed EPERM but introduced DEADLOCK: FSDP2 all_gather (fsdp2_ep_pg)
            #   and AllToAll backward (ep_pg) run concurrently over the same 4 ranks.
            #   With CCL_WORKER_COUNT=1, different ranks advance at different speeds →
            #   some call all_gather, others call AllToAll → both block → deadlock.
            #
            # Fix (v40): wrap experts with FSDP2 on dp_replicate (3-rank) instead.
            # dp_replicate groups ({0,4,8}, {1,5,9}, etc.) are DISJOINT from ep_mesh
            # groups ({0,1,2,3}, {4,5,6,7}, {8,9,10,11}).
            # Expert FSDP2 all_gather and AllToAll can NEVER deadlock — they operate
            # on different, non-overlapping sets of ranks.
            # Expert FSDP2 semantics: each rank shards its 32-expert EP slice across
            # 3 dp_replicate ranks (32/3 ≈ 10-11 per rank at rest). Before expert
            # forward (dispatch+compute), FSDP2 all_gathers → each rank gets its full
            # 32 experts. Expert compute is correct. Reduce_scatter after backward
            # averages grads across 3 dp_replicate replicas — correct for DP.
            # Wire EP dispatch/combine directly onto each MoE module.
            # parallelize_module stores _ep_instance on GroupedExperts; wire_ep_to_moe_modules
            # reads it and sets moe._ep_dispatch/_ep_combine as plain callables so that
            # MoE.forward() calls them directly (no FSDP2 hooks needed).
            from torchtune.modules.moe import wire_ep_to_moe_modules
            n_ep_wired = wire_ep_to_moe_modules(model)
            utils.log_rank_zero(
                log,
                f"EP: wired dispatch/combine callables on {n_ep_wired} MoE modules",
            )
            # v42: wrap expert modules with trivial 1-rank FSDP2 (no communication).
            # Meta params were pre-shrunk to [32,...] BEFORE AC wrapping (above),
            # so FSDPParam is initialized with the correct EP-local sizes.
            # v123: skip for ref model (eval_mode=True). The ref model uses 1-rank solo FSDP2
            # for the entire model (root shard_model call below), so separate expert FSDP2
            # is not needed. Expert dispatch is wired by parallelize_module above.
            if not eval_mode:
                from torch.distributed._composable.fsdp import fully_shard as _fully_shard
                from torchtune.modules.moe.experts import GroupedExperts as _GE
                from torchtune.models.qwen3_moe._experts import GroupedExpertsHF as _GEHF
                _solo_expert_classes = (_GE, _GEHF)
                # Wrap with 1-rank solo FSDP2. FSDPParam sees [32,...] shapes.
                # ALL ranks must call new_group for each 1-rank group (one per rank) in the
                # same order. 12 new_group calls total (world_size), each of size 1.
                _solo_groups = []
                for _r in range(self.world_size):
                    _sg = torch.distributed.new_group([_r])
                    _solo_groups.append(_sg)
                _my_solo_pg = _solo_groups[self.rank]
                from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh
                _solo_mesh = _DeviceMesh.from_group(_my_solo_pg, "xpu")
                # Use isinstance only (not name suffix) — after AC wrapping the module path
                # contains "_checkpoint_wrapped_module" so endswith(".experts") would miss them.
                _n_solo_wrapped = 0
                _solo_wrapped_mods = []
                for _ename, _emod in model.named_modules():
                    if isinstance(_emod, _solo_expert_classes):
                        # reshard_after_forward=False: no-op (1-rank group, nothing to gather)
                        _fully_shard(_emod, mesh=_solo_mesh, reshard_after_forward=False)
                        _solo_wrapped_mods.append(_emod)
                        _n_solo_wrapped += 1
                utils.log_rank_zero(
                    log,
                    f"EP v43: wrapped {_n_solo_wrapped} expert modules with trivial 1-rank FSDP2 "
                    f"(no communication, excludes them from root dp_replicate FSDP2)",
                )
                # Suppress reduce_grads on expert 1-rank FSDP2 groups.
                # CCL's reduce_scatter_tensor for 1-rank groups tries to register L0 IPC handles
                # for the grad buffer → ze_handle_manager "unknown memory type" crash (v42).
                # With reduce_grads=False, FSDP2 skips reduce_scatter in post_backward entirely;
                # expert grads accumulate in param.grad for manual AllReduce in train().
                # After _fully_shard(), each expert module is an FSDPModule — use ._get_fsdp_state()
                # which is the stable public API on FSDPModule.
                from torch.distributed.fsdp import FSDPModule as _FSDPModule
                _n_grads_suppressed = 0
                for _emod in _solo_wrapped_mods:
                    if isinstance(_emod, _FSDPModule):
                        _fsdp_state = _emod._get_fsdp_state()
                        if _fsdp_state is not None and _fsdp_state._fsdp_param_group is not None:
                            _fsdp_state._fsdp_param_group.reduce_grads = False
                            _n_grads_suppressed += 1
                utils.log_rank_zero(
                    log,
                    f"EP v43: suppressed reduce_grads on {_n_grads_suppressed} expert FSDPParamGroups "
                    f"(prevents CCL ze_handle_manager crash for 1-rank reduce_scatter)",
                )
            else:
                # Ref model (eval_mode=True): create NEW solo process groups for the root
                # 1-rank FSDP2 (below). All 12 ranks must participate in all 12 new_group
                # calls (PyTorch collective constraint). Store as instance attr so
                # shard_model() sees it via fsdp2_mesh.
                # v123: using 1-rank solo mesh for root FSDP2 instead of dp_replicate.
                from torch.distributed.device_mesh import DeviceMesh as _DeviceMeshRef
                _ref_root_solo_groups = []
                for _r in range(self.world_size):
                    _ref_root_sg = torch.distributed.new_group([_r])
                    _ref_root_solo_groups.append(_ref_root_sg)
                self._ref_root_solo_mesh = _DeviceMeshRef.from_group(
                    _ref_root_solo_groups[self.rank], "xpu"
                )
                utils.log_rank_zero(
                    log, "EP v123: created 1-rank solo mesh for ref root FSDP2 (no dp_replicate all-gather)"
                )

        # Standard shard conditions — solo-wrapped experts are already FSDP2-wrapped;
        # shard_model will skip them (inner FSDP units are opaque to outer).
        fsdp_shard_conditions = [partial(training.get_shard_conditions, names_to_match=custom_sharded_layers)]

        # Policy doesn't reshard after forward for faster generation.
        # Reference net reshards after forward because it never calls .backward()
        # EP-mode: use dp_replicate (3-rank) mesh for non-expert FSDP2 instead of the
        # EP+FSDP2 communicator conflict fix:
        # XCCL cannot handle concurrent collectives from different process groups on the
        # same OFI fabric. In EP mode, the EP AllToAll backward (on ep_mesh/dp_shard
        # 4-rank group) fires concurrently with FSDP2's pre_backward all-gather (on
        # dp_replicate 3-rank group) → OFI EPERM (err=265) → allgatherv EXCEPTION.
        #
        # Root cause: FSDP2 reshard_after_forward=True causes each layer to reshard
        # after forward, then re-all-gather during backward (pre_backward hook). This
        # all-gather collides with EP AllToAll backward at the same time.
        #
        # Fix: Use reshard_after_forward=False (ZeRO-2 style) for non-expert params
        # in EP mode. Parameters stay gathered (not resharded) after forward, so there
        # is NO all-gather during backward — only reduce-scatter (post_backward).
        # Reduce-scatter runs AFTER backward completes, when EP AllToAll is already done.
        # Cost: ~2× more memory for non-expert params vs reshard_after_forward=True.
        # At 12.96 GiB alloc with 24 GiB HBM, this is acceptable.
        #
        # Without EP: use full dp_mesh (HSDP) with standard reshard_after_forward.
        # FSDP2 mesh and reshard strategy:
        # - Policy (eval_mode=False, EP active): dp_replicate (3-rank) mesh, ZeRO-2
        #   (reshard_after_forward=False) — params stay gathered after fwd, no all-gather
        #   during bwd (prevents XCCL conflict with EP AllToAll backward).
        # - Ref (eval_mode=True, EP active): 1-rank solo mesh, ZeRO-3 (reshard_after_forward=True)
        #   + cpu_offload=True. Each layer's params are fetched CPU→XPU individually during fwd
        #   via simple memory copy (no XCCL, no IPC handles). Released back to CPU after each layer.
        #   v123 rationale:
        #     v118: ZeRO-3 on dp_replicate + cpu_offload=True → FSDP2 bug: 14.94 GiB stays in
        #       XPU allocated after reshard (XCCL IPC handle cache holds reference).
        #     v120-v122: .to(device)/.cpu() approach → XCCL IPC handle corruption → SIGSEGV.
        #     v123: 1-rank solo mesh → no inter-rank communication → no XCCL IPC handles
        #       involved in ref FSDP2 all-gather. Pure CPU→XPU memory copy per layer.
        #       Pre-bwd: 12.23 + 15.57 ≈ 27.8 GiB. Peak backward: 27.8 + 16.18 ≈ 44 GiB << 64.
        if _ep_active and self._dp_mesh.ndim > 1:
            if eval_mode:
                # Ref model: 1-rank solo FSDP2 (created in EP block above).
                fsdp2_mesh = self._ref_root_solo_mesh
                fsdp2_raf = True  # reshard after each layer → CPU→XPU→CPU per layer
                # v126: set requires_grad=False on all ref model params BEFORE FSDP2 wrapping.
                # With 1-rank solo FSDP2, FSDPParams inherit requires_grad from original params.
                # If requires_grad=True (default), FSDP2 registers AccumulateGrad hooks on
                # FSDPParams → pre_backward unshard hooks fire at policy backward() start →
                # try to all-gather CPU ref params back to XPU → crash (state machine conflict
                # or null pointer in FSDP2 C++ hook since no gradient flows through ref model).
                # With requires_grad=False: NO AccumulateGrad hooks → NO FSDP2 backward hooks
                # registered → ref model FSDP2 is transparent to policy backward.
                # Forward (unshard + compute + reshard) still works — hooks are module-level
                # (forward hooks), not gradient-level.
                model.requires_grad_(False)
                log.info(
                    "EP v126: ref model requires_grad_(False) before FSDP2 wrapping — "
                    "prevents FSDP2 backward hooks from firing during policy backward."
                )
                log.info(
                    "EP v123: ref model using 1-rank solo FSDP2 mesh + cpu_offload=True + "
                    "reshard_after_forward=True. Per-layer CPU→XPU copy, no XCCL."
                )
            else:
                # Policy model: dp_replicate ZeRO-2.
                fsdp2_mesh = self._dp_mesh["dp_replicate"]
                fsdp2_raf = False
                log.info(
                    "EP active: using reshard_after_forward=False (ZeRO-2) for non-expert "
                    "FSDP2 on dp_replicate mesh to prevent XCCL conflict with EP AllToAll."
                )
        else:
            fsdp2_mesh = self._dp_mesh
            fsdp2_raf = reshard_after_forward

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=fsdp2_raf,
            dp_mesh=fsdp2_mesh,
            disable_prefetch=self._disable_prefetch,
        )

        # Disable FSDP2 backward prefetch for ALL wrapped modules when EP is active.
        # v40: expert FSDP2 is on dp_replicate (disjoint from ep_pg) — backward prefetch
        # would fire async all_gathers on dp_replicate during AllToAll backward. While
        # these are on different groups (no deadlock possible), disabling prefetch ensures
        # sequential unshard → AllToAll → next layer, matching the backward graph order.
        # Disabling prefetch also avoids the pre-v40 issue where prefetch triggered
        # expert all_gathers on ep_pg during AllToAll backward → OFI EPERM.
        if _ep_active:
            training.disable_fsdp2_backward_prefetch(model)

        # v59: Suppress reduce_grads on ALL FSDPParamGroups when EP is active.
        # Expert groups already have reduce_grads=False (set above, v43).
        # Non-expert groups (on dp_replicate 1D mesh) still fire reduce_scatter_tensor
        # during backward, which goes through _xpu_reduce_scatter_via_allreduce and
        # calls gloo AllReduce on _GLOO_DP_REP_PG. But different EP replica groups
        # ({0,4,8}, {1,5,9}, ...) may be at different params when the AllReduce fires
        # (EP load imbalance → different backward execution order per replica group) →
        # gloo preamble size mismatch → SIGABRT.
        # Fix: suppress reduce_grads=False on ALL FSDPParamGroups. FSDP2 then skips
        # reduce_scatter entirely during backward. Grads accumulate in param.grad locally.
        # Post-backward: _ep_post_backward_grad_sync() iterates all params sequentially
        # and does gloo AllReduce in a fixed parameter order — no ordering race possible.
        if _ep_active:
            from torch.distributed.fsdp import FSDPModule as _FSDPModuleV59
            _n_all_suppressed = 0
            for _mod in model.modules():
                if isinstance(_mod, _FSDPModuleV59):
                    _fsdp_state = _mod._get_fsdp_state()
                    if _fsdp_state is not None and _fsdp_state._fsdp_param_group is not None:
                        if _fsdp_state._fsdp_param_group.reduce_grads:
                            _fsdp_state._fsdp_param_group.reduce_grads = False
                            _n_all_suppressed += 1
            utils.log_rank_zero(
                log,
                f"EP v59: suppressed reduce_grads on {_n_all_suppressed} additional "
                f"non-expert FSDPParamGroups (post-backward manual gloo AllReduce instead)",
            )
            # v9: build per-FSDPParamGroup gloo PG map for the unsharded-grad release
            # helper. Must happen AFTER fully_shard / shard_experts_for_ep so that all
            # FSDPParamGroups are constructed.
            self._ep_grad_release_pg_map = _ep_build_grad_release_pg_map(model)

        # Load checkpoint.
        # v41: expert meta params were pre-shrunk to [32,...] before fully_shard, so
        # FSDPParam expects [32,...] tensors. Pre-slice model_sd expert params here so
        # load_from_full_model_state_dict copies the correct [32,...] EP shard.
        # Non-expert params: FSDP2 DTensors on dp_replicate (3-rank), auto-sliced.
        # Slice MUST be interleaved [_ep_rank::_ep_degree] to match
        # ExpertParallel._token_dispatch ownership formula
        # (g = ep_rank + local_exp_idx * ep_degree). Contiguous slicing here would
        # silently route tokens to the wrong experts on every EP rank — see
        # tests/torchtune/dev/rl/test_ep_slice_contract.py.
        if _ep_active:
            _n_sd_sliced = 0
            for _sd_name in list(model_sd.keys()):
                if _sd_name in _expert_param_names:
                    _ft = model_sd[_sd_name]
                    assert _ft.shape[0] % _ep_degree == 0, (
                        f"Expert param {_sd_name}: shape[0]={_ft.shape[0]} not divisible by ep_degree={_ep_degree}"
                    )
                    _n_local = _ft.shape[0] // _ep_degree
                    model_sd[_sd_name] = _ft[_ep_rank::_ep_degree].contiguous()
                    _n_sd_sliced += 1
            utils.log_rank_zero(
                log,
                f"EP: pre-sliced {_n_sd_sliced} expert params in model_sd "
                f"(interleaved {_ft.shape[0]}→{_n_local} for EP rank {_ep_rank}/{_ep_degree}; "
                f"rank r owns global experts r, r+{_ep_degree}, r+{2*_ep_degree}, ...)",
            )
        training.load_from_full_model_state_dict(
            model,
            model_sd,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )
        # EP weight slicing was done via model_sd pre-slicing above (interleaved).

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        # Store vocab size for OOB clamping (FSDP2 with use_orig_params keeps shapes)
        if not hasattr(self, '_vocab_size') and hasattr(model, 'tok_embeddings'):
            self._vocab_size = model.tok_embeddings.weight.shape[0]

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            try:
                memory_stats = training.get_memory_stats(device=self._device)
                training.log_memory_stats(memory_stats)
            except RuntimeError:
                pass

        # Verify allocator config is applied (PYTORCH_ALLOC_CONF, not TORCH_XPU_ALLOC_CONF)
        if self._is_rank_zero:
            _alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "<NOT SET>")
            _bad_conf = os.environ.get("TORCH_XPU_ALLOC_CONF")
            log.info("PYTORCH_ALLOC_CONF=%s", _alloc_conf)
            if _bad_conf:
                log.warning(
                    "TORCH_XPU_ALLOC_CONF=%s is set but NOT recognized by PyTorch. "
                    "Use PYTORCH_ALLOC_CONF instead.", _bad_conf
                )

        # FSDP diagnostic: log wrapping structure and per-unit reshard settings
        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_structure(model, log=log)
            training.verify_activation_checkpointing(model, log=log)
            if self._disable_prefetch:
                log.info("FSDP prefetch DISABLED (reshard_after_forward at root = per-layer setting)")
            else:
                log.info("FSDP prefetch ENABLED (reshard_after_forward=None at root)")

        # Register per-layer memory hooks for diagnostics (rank 0 only)
        if self._fsdp_diagnostics and self._is_rank_zero:
            self._layer_mem_hooks = training.register_per_layer_memory_hooks(
                model, self._device, log, sample_every=10,
            )
        else:
            self._layer_mem_hooks = []

        disable_dropout(model)

        # v40: no manual FSDP2 param group tracking needed.
        # Non-expert: FSDP2 reduce_scatter on dp_replicate handles grad sync natively.
        # Expert: 1-rank solo FSDP2 (no-op); manual AllReduce in train() after grpo_step.
        self._expert_fsdp_modules = []
        self._fsdp2_param_groups_meta = []  # empty — no manual sync needed in v40

        return model

    def _setup_model_fsdp1_hsdp(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_sd: dict[str, Any],
        eval_mode: bool = False,
        reshard_after_forward: bool = True,
    ) -> nn.Module:
        """FSDP1-based HSDP model setup.

        Uses FullyShardedDataParallel with HYBRID_SHARD strategy and a 2D
        DeviceMesh. This is proven to work on Aurora XPU (used by PRISM).

        FSDP2's composable fully_shard() creates per-layer XCCL sub-communicators
        which hang on Aurora. FSDP1's single FSDP() call avoids this by managing
        all sub-communicators internally in one initialization.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        utils.log_rank_zero(
            log,
            f"HSDP (FSDP1): Instantiating model on device "
            f"(dp_replicate={self._dp_replicate} × dp_shard={self._dp_shard})...",
        )
        init_start = time.perf_counter()

        # Instantiate on CPU first, load state dict, then move to device.
        # For large models (32B), ZE_AFFINITY_MASK must be unset so the
        # allocator can spill across tiles during FSDP1 flatten.
        with training.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_model)

        if eval_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        # Load checkpoint BEFORE wrapping with FSDP1
        model.load_state_dict(model_sd, strict=True)
        del model_sd  # free CPU memory

        # Move to device
        model = model.to(device=self._device, dtype=self._dtype)

        # Store vocab size BEFORE FSDP wrapping (FSDP shards tok_embeddings,
        # making .weight.shape[0] return shard size instead of vocab size)
        if hasattr(model, 'tok_embeddings'):
            self._vocab_size = model.tok_embeddings.weight.shape[0]

        # Initialize RoPE
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

        if enable_activation_checkpointing:
            _n_moe_self_ac = _apply_split_ac(model)
            utils.log_rank_zero(
                log,
                f"v158 split AC: {_n_moe_self_ac} MoE layers self-checkpoint "
                f"(MoE outside AC); other layers wrapped by apply_activation_checkpointing.",
            )

        # Mixed precision policy
        mp_policy = MixedPrecision(
            param_dtype=self._dtype,
            reduce_dtype=self._dtype,
            buffer_dtype=self._dtype,
        )

        # Sharding strategy: reshard_after_forward=True → HYBRID_SHARD (FULL_SHARD intra-node)
        #                     reshard_after_forward=False → _HYBRID_SHARD_ZERO2 (SHARD_GRAD_OP intra-node)
        if reshard_after_forward:
            sharding = ShardingStrategy.HYBRID_SHARD
            shard_label = "HYBRID_SHARD"
        else:
            try:
                sharding = ShardingStrategy._HYBRID_SHARD_ZERO2
                shard_label = "_HYBRID_SHARD_ZERO2"
            except AttributeError:
                sharding = ShardingStrategy.HYBRID_SHARD
                shard_label = "HYBRID_SHARD (fallback)"

        utils.log_rank_zero(log, f"HSDP: Using {shard_label} strategy")

        # Wrap with FSDP1 — single call, minimal sub-communicator creation
        model = FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp_policy,
            device_mesh=self._dp_mesh,
            use_orig_params=True,
            limit_all_gathers=True,
        )

        utils.log_rank_zero(
            log,
            f"HSDP model setup took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            try:
                memory_stats = training.get_memory_stats(device=self._device)
                training.log_memory_stats(memory_stats)
            except RuntimeError:
                pass

        disable_dropout(model)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if hasattr(self._model, 'trainable_parameters'):
            params = [p for _, p in self._model.trainable_parameters()]
        else:
            params = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = config.instantiate(cfg_optimizer, params)
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )
        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        # Instantiate collate_fn
        collate_fn = _get_component_from_path(collate_fn)

        # When using TP>1 for vLLM, all ranks in a TP group must see identical
        # data so they generate the same prompts (deterministic scheduling).
        # Use dp_size/dp_rank instead of world_size/rank for the sampler.
        #
        # For HSDP: ranks within the same dp_shard group process the same batch
        # (they're FSDP shards of the same model copy). Only dp_replicate copies
        # need different data. So sampler_replicas = dp_replicate, sampler_rank =
        # replicate index.
        if self._vllm_mode == "server":
            # vLLM server mode: rank 0 generates and broadcasts to all ranks.
            # All ranks must see the same batch for matching tensor shapes.
            sampler_replicas = 1
            sampler_rank = 0
        elif self._dp_replicate > 1:
            # HSDP: each replicate group sees different data
            sampler_replicas = self._dp_replicate
            sampler_rank = self.rank // self._dp_shard
        elif self._vllm_tp_size > 1:
            sampler_replicas = self.world_size // self._vllm_tp_size
            sampler_rank = self.rank // self._vllm_tp_size
        else:
            sampler_replicas = self.world_size
            sampler_rank = self.rank

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            shuffle=shuffle,
            seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)
            list(dataloader)
        return dataloader

    def _gather_cpu_state_dict_safe(
        self,
        model,
        is_rank_zero: bool,
        device=None,
    ) -> dict:
        """Like training.gather_cpu_state_dict but uses shard PG for barriers.

        On multi-node XPU, world-level barriers crash after FSDP2 sub-communicators
        are created (broadcast_scaleout_sycl.cpp:65). This replaces the per-parameter
        world barrier with a shard PG barrier when available.
        """
        from torchtune.training._distributed import _gather_nf4_tensor
        try:
            from torchao.dtypes import NF4Tensor
        except ImportError:
            NF4Tensor = None

        cpu_state_dict = {}
        sharded_sd = model.state_dict()
        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                param = param.to(device)
            if hasattr(param, "_local_tensor"):
                if NF4Tensor is not None and isinstance(param._local_tensor, NF4Tensor):
                    param = _gather_nf4_tensor(param)
                else:
                    param = param.full_tensor()
            if NF4Tensor is not None and isinstance(param, NF4Tensor):
                param = param.to(param.dtype)
            if is_rank_zero:
                cpu_state_dict[param_name] = param.cpu()
            # Skip barrier — FSDP full_tensor() is itself a synchronization point
            if not self._production_mode:
                torch.distributed.barrier()
        return cpu_state_dict

    @property
    def _policy(self):
        """Return the underlying policy model, unwrapping DDP if present."""
        return self._model.module if hasattr(self._model, 'module') else self._model

    def _training_barrier(self):
        """Barrier over training ranks only (skips vLLM rank in dedicated_rank mode)."""
        pg = getattr(self, '_training_pg', None) if self._vllm_mode == "dedicated_rank" else None
        torch.distributed.barrier(group=pg)

    def _extract_batch_kwargs(self, batch: dict) -> dict:
        """Hook for subclasses to forward extra batch fields into
        ``generate_trajectory_batched``. Default returns an empty dict, so the
        base recipe calls ``generate_trajectory_batched(tokens, answers)``.

        Why a hook (not subclass-overrides train()): preserving the train loop
        in one place is the only way to avoid silently re-introducing the
        missing-weight-sync class of bug (see project_bioreason_train_missing_wsync).
        """
        return {}

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0.
        # Use shard PG for barriers when available (multi-node XPU: world barriers
        # crash after FSDP2 sub-communicators are created).
        cpu_state_dict = self._gather_cpu_state_dict_safe(
            self._model,
            self._is_rank_zero,
            device=self._device,
        )

        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if intermediate_checkpoint:
            start = time.perf_counter()
            utils.log_rank_zero(log, "Getting optimizer state dict...")
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._model,
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
            utils.log_rank_zero(
                log,
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs",
            )
        else:
            opt_state_dict = None

        if self._is_rank_zero:
            start = time.perf_counter()
            checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self._epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.RNG_KEY: self._rng.get_state(),
                        training.DATALOADER_KEY: self._dataloader.state_dict(),
                    }
                )

            if self._checkpointer is not None:
                self._checkpointer.save_checkpoint(
                    checkpoint_dict,
                    epoch=epoch,
                    intermediate_checkpoint=intermediate_checkpoint,
                )
                log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")
            else:
                log.info("No checkpointer configured — skipping checkpoint save")

        # Skip barrier in production mode — checkpoint save is rank-0 only,
        # other ranks just need to stay out of the way during full_tensor() gather.
        if not self._production_mode:
            torch.distributed.barrier()

    def _broadcast_query_responses(self, query_responses: torch.Tensor) -> torch.Tensor:
        """Broadcast rank-0 query_responses to training ranks.

        In dedicated_rank mode the vLLM rank is outside `_training_pg` and
        must NOT participate (it's busy generating / receiving weights).
        Server / colocate modes still use the world PG.
        """
        _grp = (
            self._training_pg
            if (self._vllm_mode == "dedicated_rank" and getattr(self, "_training_pg", None) is not None)
            else None
        )
        torch.distributed.broadcast(query_responses, src=0, group=_grp)
        return query_responses

    def _generate_with_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Call vLLM server for generation, broadcast results to all ranks.

        Only global rank 0 calls vLLM and broadcasts to all ranks via the world
        process group. Uses a fixed buffer size (tokenizer.max_seq_len) so all
        ranks have matching tensor shapes regardless of per-rank batch variation.

        Returns:
            query_responses: ``[B*G, max_seq_len]`` (padded to fixed length)
        """
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        if self._is_rank_zero:
            query_responses = self._call_vllm_http(batch_input_ids, context_length)
        else:
            query_responses = batch_input_ids.new_empty(bsz, total_len)
        return self._broadcast_query_responses(query_responses)

    def _call_vllm_http(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Rank-0-only vLLM HTTP round-trip. No collectives — safe from a
        producer thread. Caller is responsible for broadcasting the result
        to other ranks (see :meth:`_broadcast_query_responses`).
        """
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens
        # Strip padding and convert to Python lists for HTTP
        prompts = []
        for i in range(bsz):
            ids = batch_input_ids[i].cpu().tolist()
            ids = [t for t in ids if t != self._tokenizer.pad_id]
            prompts.append(ids)

        gen_kwargs = dict(
            n=1,  # prompts already expanded by grpo_samples
            max_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k or 0,
        )

        t0 = time.perf_counter()
        num_clients = len(self._vllm_clients)
        if num_clients > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            chunks = [prompts[i::num_clients] for i in range(num_clients)]

            def _call_vllm(client, chunk):
                return client.generate(prompts=chunk, **gen_kwargs) if chunk else []

            with ThreadPoolExecutor(max_workers=num_clients) as pool:
                futures = {
                    pool.submit(_call_vllm, client, chunk): idx
                    for idx, (client, chunk) in enumerate(zip(self._vllm_clients, chunks))
                }
                chunk_results = [None] * num_clients
                for future in as_completed(futures):
                    idx = futures[future]
                    chunk_results[idx] = future.result()

            completions = [None] * bsz
            for i in range(bsz):
                client_idx = i % num_clients
                within_idx = i // num_clients
                completions[i] = chunk_results[client_idx][within_idx]
        else:
            completions = self._vllm_client.generate(prompts=prompts, **gen_kwargs)
        gen_time = time.perf_counter() - t0

        query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
        query_responses[:, :context_length] = batch_input_ids
        for i, comp in enumerate(completions):
            length = min(len(comp), self._max_generated_tokens)
            query_responses[i, context_length : context_length + length] = torch.tensor(
                comp[:length], dtype=batch_input_ids.dtype, device=self._device
            )

        total_tokens = sum(len(c) for c in completions)
        log.info(
            "Rank %d: vLLM generation: %d sequences (%d clients), %d tokens in %.1fs (%.1f tok/s)",
            self.rank, bsz, num_clients, total_tokens, gen_time, total_tokens / max(gen_time, 0.01),
        )
        return query_responses

    def _generate_with_colocated_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Generate using this rank's colocated vLLM engine.

        With a DistributedSampler, each rank already has its own subset of
        prompts. Each rank generates ALL grpo_samples completions for its own
        prompts locally — no cross-rank communication needed.

        Returns:
            query_responses: ``[B*G, context_length + max_generated_tokens]``
        """
        from vllm import SamplingParams

        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        sampling_params = SamplingParams(
            max_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k if self._top_k else -1,
            detokenize=False,
        )

        # Text-only: strip padding and pass token ID lists.
        raw_prompts = []
        for i in range(bsz):
            ids = batch_input_ids[i].cpu().tolist()
            ids = [t for t in ids if t != self._tokenizer.pad_id]
            raw_prompts.append(ids)
        vllm_prompts = [{"prompt_token_ids": p} for p in raw_prompts]

        t0 = time.perf_counter()
        outputs = self._vllm_llm.generate(
            prompts=vllm_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        gen_time = time.perf_counter() - t0

        # Build query_responses: [prompt | completion | padding]
        query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
        query_responses[:, :context_length] = batch_input_ids
        total_tokens = 0
        for i, output in enumerate(outputs):
            comp = output.outputs[0].token_ids
            total_tokens += len(comp)
            length = min(len(comp), self._max_generated_tokens)
            query_responses[i, context_length : context_length + length] = torch.tensor(
                comp[:length], dtype=batch_input_ids.dtype, device=self._device
            )

        log.info(
            "Rank %d: generated %d sequences, %d tokens in %.1fs (%.1f tok/s)",
            self.rank, bsz, total_tokens, gen_time, total_tokens / max(gen_time, 0.01),
        )

        # Re-set XPU device context after vLLM generation (vLLM may have
        # called torch.xpu.set_device internally, shifting the default device).
        torch.xpu.set_device(_xpu_device_index)
        torch.xpu.synchronize()

        return query_responses

    _file_barrier_gen = 0

    def _serialized_empty_cache(self) -> None:
        """Empty XPU cache one rank at a time using file-based serialization.

        NOTE: On XPU with FSDP, empty_cache() leaks UR handles (see
        docs/bugs/intel_xpu_resource_leak_bug_report.md). This method is now a
        no-op on XPU — kept for non-XPU use or future driver fix.
        """
        if self._device.type == "xpu":
            # Skip — empty_cache + FSDP storage.resize_ leaks UR handles
            if not self._production_mode:
                torch.distributed.barrier()
            return

        gen = self._file_barrier_gen
        self.__class__._file_barrier_gen = gen + 1
        barrier_dir = f"/tmp/torchtune/empty_cache_barriers/gen{gen}"
        os.makedirs(barrier_dir, exist_ok=True)

        for r in range(self.world_size):
            if r == self.rank:
                # My turn to call empty_cache
                torch.xpu.synchronize()
                log.info("Rank %d: serialized empty_cache gen%d start", self.rank, gen)
                torch.xpu.empty_cache()
                log.info("Rank %d: serialized empty_cache gen%d done", self.rank, gen)
                # Signal done
                with open(os.path.join(barrier_dir, f"r{r}_done"), "w") as f:
                    f.write("done")
            else:
                # Wait for rank r to finish
                done_file = os.path.join(barrier_dir, f"r{r}_done")
                while not os.path.exists(done_file):
                    time.sleep(0.001)

        # No cleanup — leave barrier files (they're in /tmp and tiny)

    # ---------------------------------------------------------------------------
    # Weight sync methods (see torchtune/dev/rl/weight_sync.py)
    # ---------------------------------------------------------------------------
    _sync_colocated_weights = _weight_sync_module._sync_colocated_weights
    _compute_wsync_layout = _weight_sync_module._compute_wsync_layout
    _sync_dedicated_vllm_weights = _weight_sync_module._sync_dedicated_vllm_weights
    _recv_weight_update = _weight_sync_module._recv_weight_update
    _generate_with_dedicated_vllm = _weight_sync_module._generate_with_dedicated_vllm
    _run_vllm_generation_server = _weight_sync_module._run_vllm_generation_server
    _save_raw_bytes = staticmethod(_weight_sync_module._save_raw_bytes)
    _post_weights_to_vllm = _weight_sync_module._post_weights_to_vllm
    _sync_weights_to_vllm = _weight_sync_module._sync_weights_to_vllm
    _init_sender_pool = _weight_sync_module._init_sender_pool
    _init_xccl_weight_sync = _weight_sync_module._init_xccl_weight_sync
    _sync_weights_to_vllm_xccl = _weight_sync_module._sync_weights_to_vllm_xccl
    _sync_weights_to_vllm_shm = _weight_sync_module._sync_weights_to_vllm_shm
    _wait_for_sync_complete = _weight_sync_module._wait_for_sync_complete
    _start_deferred_broadcast = _weight_sync_module._start_deferred_broadcast


    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
    ) -> GRPOTrajectory:
        """
        Generates a trajectory given the current policy model, the reference policy model,
        the reward function, and batch of inputs.
        """
        # Synchronize XPU between steps.
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if not _colocate_vllm_mode:
            device_empty_cache(self._device)
        elif self._vllm_mode == "colocate_sleep" and self._vllm_llm is not None and hasattr(self, '_vllm_is_sleeping') and self._vllm_is_sleeping:
            # Sleep mode: wake weights → sync updated FSDP weights → wake KV cache
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            log.info("Rank %d: waking up vLLM for generation", self.rank)
            t_wake = time.perf_counter()
            # 1. Restore vLLM weight storage (from CPU backup — old weights)
            self._vllm_llm.wake_up(tags=["weights"])
            # 2. Overwrite with updated FSDP weights
            self._sync_colocated_weights()
            # 3. Reallocate KV cache
            self._vllm_llm.wake_up(tags=["kv_cache"])
            self._vllm_is_sleeping = False
            log.info("Rank %d: vLLM wake_up + weight sync completed in %.2fs", self.rank, time.perf_counter() - t_wake)
        elif self._vllm_llm is not None and hasattr(self, '_vllm_kv_cache_shapes'):
            # Non-sleep colocate: reclaim cached training memory, reallocate KV cache.
            # NOTE: skip empty_cache() — leaks UR handles with FSDP.
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            kv_caches = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
            for i, (shape, dtype) in enumerate(self._vllm_kv_cache_shapes):
                kv_caches[i] = torch.zeros(shape, dtype=dtype, device=self._device)
            del self._vllm_kv_cache_shapes
            self._vllm_llm.llm_engine.reset_prefix_cache()

        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)


        # step 1: generate responses using the current policy (or vLLM)
        _vllm_t0 = time.perf_counter()
        if self._vllm_mode in ("colocate", "colocate_sleep"):
            query_responses = self._generate_with_colocated_vllm(
                batch_input_ids, context_length
            )
        elif self._vllm_mode == "dedicated_rank":
            # Fan out qr to training ranks [0..N-2] over gloo. Each rank's
            # own batch has its own padded context_length, so broadcast rank
            # 0's actual qr shape first to keep gloo recv buffers aligned.
            # The xccl _training_pg is unexercised (FSDP2 reduce_scatter is
            # patched to gloo) so its first xccl collective returns
            # fi_cq_readerr EPERM — gloo is the safe path here.
            if self._is_rank_zero:
                qr_cpu = self._generate_with_dedicated_vllm(
                    batch_input_ids, context_length, None
                )
                shape_obj = [list(qr_cpu.shape)]
            else:
                shape_obj = [None]
            torch.distributed.broadcast_object_list(
                shape_obj, src=0, group=self._training_fanout_pg, device="cpu",
            )
            qr_shape = tuple(shape_obj[0])
            if not self._is_rank_zero:
                qr_cpu = torch.empty(
                    qr_shape, dtype=batch_input_ids.dtype, device="cpu"
                )
            torch.distributed.broadcast(qr_cpu, src=0, group=self._training_fanout_pg)
            query_responses = qr_cpu.to(self._device)
        elif self._vllm_mode == "server":
            if getattr(self, "_pending_async_query_responses", None) is not None:
                # Phase 1 async: producer thread already issued HTTP on rank 0
                # and stashed the tensor here. Run the matching collective
                # broadcast on all ranks; consumers see the same tensor.
                bsz = batch_input_ids.shape[0]
                total_len = context_length + self._max_generated_tokens
                if self._is_rank_zero:
                    query_responses = self._pending_async_query_responses
                    assert query_responses.shape == (bsz, total_len), (
                        f"async qr shape mismatch: got {tuple(query_responses.shape)}, "
                        f"expected ({bsz}, {total_len})"
                    )
                else:
                    query_responses = batch_input_ids.new_empty(bsz, total_len)
                self._pending_async_query_responses = None
                query_responses = self._broadcast_query_responses(query_responses)
            else:
                query_responses = self._generate_with_vllm(batch_input_ids, context_length)
        else:
            with local_kv_cache(
                model=self._model,
                batch_size=batch_size * grpo_size,
                device=self._device,
                dtype=self._dtype,
                decoder_max_seq_len=context_length + self._max_generated_tokens,
            ):
                # For HSDP: disable early stopping during generation.
                # With HSDP, different replicate groups process different data
                # and finish at different times. Early stopping via all_reduce
                # on the shard PG works per-node, but then the world-level
                # barrier after generation deadlocks because one node finishes
                # before the other. Always generating max tokens ensures all
                # ranks take the same number of steps (stop_token_mask zeros
                # out tokens past EOS).
                _stop_tokens = (
                    None if self._dp_replicate > 1
                    else self._tokenizer.stop_tokens
                )
                query_responses, _ = generate(
                    model=self._model,
                    prompt=batch_input_ids,
                    max_generated_tokens=self._max_generated_tokens,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    pad_id=self._tokenizer.pad_id,
                    rng=self._rng if self._device.type == "cuda" else None,
                    stop_tokens=_stop_tokens,
                    return_logits=False,
                )

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _vllm_time = time.perf_counter() - _vllm_t0

        # Barrier: all ranks must finish generation before FSDP forward passes.
        # For vLLM server mode, the world broadcast already synchronizes all ranks.
        # For dedicated_rank mode, the broadcast in _generate_with_dedicated_vllm
        # synchronizes training ranks; rank 11 is in the server loop (not here).
        if self._vllm_mode not in ("server", "dedicated_rank") and not self._production_mode:
            torch.distributed.barrier()

        # Free vLLM GPU memory to reclaim space for training forward/backward passes.
        if _colocate_vllm_mode and self._vllm_llm is not None:
            if torch.xpu.is_available():
                mem_before = torch.xpu.memory_allocated(self._device) / 1024**3

            if self._vllm_mode == "colocate_sleep":
                # Sleep mode: offload weights to CPU and release all GPU storage
                # (weights + KV cache). This frees maximum memory for training.
                log.info("Rank %d: sleeping vLLM (weights + KV cache) for training", self.rank)
                t_free = time.perf_counter()
                self._vllm_llm.sleep(level=1)
                self._vllm_is_sleeping = True
            else:
                # Non-sleep colocate: only free KV cache, weights stay on GPU
                log.info("Rank %d: freeing vLLM KV cache for training", self.rank)
                t_free = time.perf_counter()
                kv_caches = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
                self._vllm_kv_cache_shapes = []
                for i, cache in enumerate(kv_caches):
                    self._vllm_kv_cache_shapes.append((cache.shape, cache.dtype))
                    kv_caches[i] = torch.empty(0, device="cpu")

            if torch.xpu.is_available():
                mem_after = torch.xpu.memory_allocated(self._device) / 1024**3
                log.info("Rank %d: vLLM memory freed in %.1fs (%.2f -> %.2f GiB, freed %.2f GiB)",
                         self.rank, time.perf_counter() - t_free,
                         mem_before, mem_after, mem_before - mem_after)
            else:
                log.info("Rank %d: vLLM memory freed in %.1fs", self.rank, time.perf_counter() - t_free)

        responses = query_responses[:, context_length:].clone()

        # Clamp token IDs to valid vocab range (XPU scatter kernel crashes on OOB)
        vocab_size = getattr(self, '_vocab_size', None)
        if vocab_size is not None and vocab_size > 0:
            oob_mask = responses >= vocab_size
            if oob_mask.any():
                log.warning("Clamping %d OOB token IDs (max=%d, vocab=%d)",
                            oob_mask.sum().item(), responses.max().item(), vocab_size)
                responses = responses.clamp(max=vocab_size - 1)
                query_responses = torch.cat([query_responses[:, :context_length], responses], dim=1)

        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy
        num_seqs = query_responses.shape[0]
        fwd_bs = self._forward_batch_size

        # Rollout-time logprobs.
        #   ppo_epochs > 1: must compute old_logprobs from the rollout-time policy
        #     because the multi-epoch update mutates the weights between epochs.
        #   async_generation / always_compute_rollout_logprobs: same requirement —
        #     the trajectory's pi_old_logprobs must come from the policy that
        #     produced the rollout, not the (potentially newer) training policy.
        #   Otherwise (default, single-epoch sync): skip the policy fwd and let
        #     grpo_step() detach pi_logprobs (ratios collapse to 1, identical to
        #     pre-async behavior).
        if self._ppo_epochs > 1 or self._compute_rollout_logprobs_required:
            _policy_fwd_t0 = time.perf_counter()
            # Rollout-time logprobs are pi_old in the loss; autograd never
            # backwards through them. Wrap in no_grad so we don't build (and
            # retain) the activation graph — without this, resv climbs from
            # ~24 GiB step 0 to ~61 GiB by step 4 on 3B/10 tiles and the run
            # hits banned:1.
            with torch.no_grad():
                if fwd_bs >= num_seqs:
                    log.info("Rank %d: policy forward start (shape=%s)", self.rank, list(query_responses.shape))
                    logits = self._model(query_responses, input_pos=position_ids, mask=masks)
                    log.info("Rank %d: policy forward done", self.rank)
                    logits = logits[:, context_length - 1 :]
                    logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
                    del logits
                else:
                    log.info("Rank %d: policy forward start CHUNKED (total=%d, chunk=%d)", self.rank, num_seqs, fwd_bs)
                    if self.rank == 0 and self._device.type == "xpu":
                        log.info("Rank 0: PRE-policy-fwd memory: alloc=%.2f GiB, resv=%.2f GiB",
                                 torch.xpu.memory_allocated() / 1024**3, torch.xpu.memory_reserved() / 1024**3)
                    logprobs_chunks = []
                    for cs in range(0, num_seqs, fwd_bs):
                        ce = min(cs + fwd_bs, num_seqs)
                        chunk_logits = self._model(
                            query_responses[cs:ce],
                            input_pos=position_ids[cs:ce],
                            mask=masks[cs:ce],
                        )
                        if self.rank == 0 and self._device.type == "xpu":
                            log.info("Rank 0: POST-chunk[%d:%d] memory: alloc=%.2f GiB, resv=%.2f GiB",
                                     cs, ce, torch.xpu.memory_allocated() / 1024**3, torch.xpu.memory_reserved() / 1024**3)
                        chunk_logits = chunk_logits[:, context_length - 1 :]
                        logprobs_chunks.append(
                            rlhf.batched_logits_to_logprobs(chunk_logits, responses[cs:ce], self._temperature)
                        )
                        del chunk_logits
                    logprobs = torch.cat(logprobs_chunks, dim=0)
                    del logprobs_chunks
                    log.info("Rank %d: policy forward done (chunked)", self.rank)
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _policy_fwd_time = time.perf_counter() - _policy_fwd_t0
        else:
            logprobs = None
            _policy_fwd_time = 0.0
            log.info(
                "Rank %d: policy_fwd SKIPPED (single-epoch sync; "
                "rollout logprobs not required, ratios will collapse to 1)",
                self.rank,
            )

        # NOTE: Removed policy→ref defrag. With fbs >= grpo_samples (no chunking),
        # post-policy reserved ~29 GiB leaves enough headroom for ref forward.
        # Each empty_cache() accelerates UR handle leak, so minimize calls.

        # step 2.1 estimate logprobs of the responses using the reference policy
        # Barrier: sync before ref model forward. FSDP2 uses lazy allgather
        # which is itself a synchronization point — skip world barrier for
        # multi-node to avoid CCL broadcast_scaleout failures.
        _ref_fwd_t0 = time.perf_counter()
        log.info("Rank %d: pre-ref forward", self.rank)
        # v79: XCCL SHARD barrier before ref forward (EP case).
        # Root cause of v78 a2a#241 deadlock: after the post-optimizer XCCL SHARD
        # sync (v78), all SHARD members start step N+1 generate() together. But
        # policy fwd has 60 gloo AllToAlls — when token routing imbalance causes
        # ep_rank 0 to process 32k tokens vs ep_ranks 1-3 processing ~0, the 60
        # policy fwd AllToAlls finish but the TRANSITION to ref fwd has no sync
        # in production mode. Fast ep_ranks (1-3) enter ref fwd and start
        # a2a#241 (first ref fwd AllToAll). ep_rank 0 is still processing its
        # policy fwd tokens → never calls a2a#241 → 1800s timeout.
        # Fix: XCCL SHARD all_reduce before ref forward. Forces all 4 SHARD
        # members to rendezvous between policy fwd and ref fwd. ep_ranks 1-3
        # wait here for ep_rank 0 to finish policy fwd before any rank starts
        # ref fwd AllToAlls. No XCCL SHARD collision: the only other XCCL SHARD
        # op in this step (post-optimizer, v78) fired in the PREVIOUS step and
        # is fully complete. Zero gloo AllToAlls are active when this fires.
        if self._expert_parallel_degree > 1 and self._shard_pg is not None:
            _pre_ref_sync = torch.zeros(1, dtype=torch.float32, device=self._device)
            _orig_all_reduce(_pre_ref_sync, op=torch.distributed.ReduceOp.SUM,
                             group=self._shard_pg)
            log.info("Rank %d: EP v79 pre-ref XCCL SHARD sync done", self.rank)
        elif not self._production_mode:
            self._training_barrier()  # dedicated_rank: training_pg only (rank 11 not here)
        # Dynamic ref offload: move ref model to XPU for fast ref forward.
        # Use actual model parameter device (more robust than stored attr).
        _ref_dev = next(self._ref_model.parameters()).device
        log.info("Rank %d: ref model device=%s, position_ids.device=%s",
                 self.rank, _ref_dev, position_ids.device)
        if fwd_bs >= num_seqs:
            log.info("Rank %d: ref forward start", self.rank)
            ref_logits = self._ref_model(
                query_responses, input_pos=position_ids, mask=masks
            )
            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, responses, self._temperature
            )
            del ref_logits
        else:
            log.info("Rank %d: ref forward start CHUNKED (total=%d, chunk=%d)", self.rank, num_seqs, fwd_bs)
            ref_logprobs_chunks = []
            for cs in range(0, num_seqs, fwd_bs):
                ce = min(cs + fwd_bs, num_seqs)
                chunk_ref_logits = self._ref_model(
                    query_responses[cs:ce],
                    input_pos=position_ids[cs:ce],
                    mask=masks[cs:ce],
                )
                chunk_ref_logits = rlhf.truncate_sequence_for_logprobs(chunk_ref_logits, context_length)
                ref_logprobs_chunks.append(
                    rlhf.batched_logits_to_logprobs(chunk_ref_logits, responses[cs:ce], self._temperature)
                )
                del chunk_ref_logits
                device_empty_cache(self._device)
            ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
            del ref_logprobs_chunks
            log.info("Rank %d: ref forward done (chunked)", self.rank)
        device_empty_cache(self._device)
        # Dynamic ref offload: move ref model back to CPU to free XPU HBM for backward.
        # NOTE: do NOT call torch.xpu.empty_cache() here. With FSDP wrapping the policy,
        # empty_cache() returns L0 pages whose addresses CCL has cached as IPC handles
        # → step 1 banned:1 (see bugs/project_xpu_emptycache_revalidated.md).
        # The ref params being freed will live in the caching allocator until reused.
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if self._is_rank_zero:
            _alloc_after_ref = torch.xpu.memory_allocated(self._device) / 1e9
            _resv_after_ref = torch.xpu.memory_reserved(self._device) / 1e9
            log.info(
                "Rank 0: post-ref-fwd alloc=%.2f GiB resv=%.2f GiB",
                _alloc_after_ref, _resv_after_ref,
            )
        _ref_fwd_time = time.perf_counter() - _ref_fwd_t0

        log.info(
            "Rank %d: GENTIMING vllm=%.1fs policy_fwd=%.1fs ref_fwd=%.1fs",
            self.rank, _vllm_time, _policy_fwd_time, _ref_fwd_time,
        )

        # step 4. replace tokens after first stop token with padding
        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # Compute rewards
        responses = responses.reshape(batch_size, grpo_size, -1)
        if self._reward_mode == "gene_recall":
            rewards, successes, metadata = gene_recall_batched_rewards(
                self._tokenizer, responses, answers, device=self._device,
                reward_metric=self._gene_reward_metric,
            )
        elif self._reward_mode == "sum_digits":
            from torchtune.dev.rl.rewards import sum_digits_batched_rewards
            rewards, successes, metadata = sum_digits_batched_rewards(
                self._tokenizer, responses, answers, device=self._device,
            )
        elif self._cfg_reward_functions:
            # Honor cfg.reward_functions when declared. Each Reward instance
            # returns total_reward / successes shape [N=batch*grpo]; we stack
            # along the function axis and align with the legacy [B, G, F]
            # layout the rest of the recipe assumes.
            decoded = [
                self._tokenizer.decode(responses[b, g].tolist())
                for b in range(batch_size) for g in range(grpo_size)
            ]
            flat_answers = [
                answers[b] for b in range(batch_size) for _ in range(grpo_size)
            ]
            flat_completion_ids = responses.reshape(batch_size * grpo_size, -1)
            r_stack, s_stack = [], []
            for fn in self._cfg_reward_functions:
                out = fn(flat_completion_ids, decoded, flat_answers)
                r_stack.append(out.total_reward.to(self._device))
                s_stack.append(out.successes.to(self._device))
            rewards = torch.stack(r_stack, dim=-1).reshape(batch_size, grpo_size, -1)
            successes = torch.stack(s_stack, dim=-1).reshape(batch_size, grpo_size, -1)
            metadata = {"func_names": [type(fn).__name__ for fn in self._cfg_reward_functions]}
        else:
            rewards, successes, metadata = batched_rewards(
                self._tokenizer, responses, answers, device=self._device
            )
        rewards = rewards.to(self._device)
        successes = successes.to(self._device)

        # Aggregate rewards and successes across reward functions
        rewards = rewards.sum(dim=-1)
        successes = successes.sum(dim=-1)

        # Log sample responses for verification (rank 0 only, first sample)
        if self._is_rank_zero:
            try:
                sample_resp = responses[0, 0]  # first prompt, first sample
                # Decode only non-pad tokens
                non_pad = sample_resp[sample_resp != self._tokenizer.pad_id]
                decoded = self._tokenizer.decode(non_pad.tolist())
                log.info(
                    "SAMPLE_RESPONSE step=%d reward=%.1f success=%.1f answer=%s response=%s",
                    self._steps_run,
                    rewards[0, 0].item(),
                    successes[0, 0].item(),
                    answers[0][:80],
                    decoded[:200],
                )
            except Exception as e:
                log.warning("Could not decode sample response: %s", e)

        self._log_batch_reward(rewards, successes)

        advantages = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.reshape(batch_size * grpo_size)
        del responses
        device_empty_cache(self._device)

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        # Use masked_fill_ to avoid boolean index L0 sub-allocation (UR:40 risk).
        if logprobs is not None:
            logprobs.masked_fill_(response_padding_masks, 1.0)
        ref_logprobs.masked_fill_(response_padding_masks, 1.0)

        return GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards.reshape(batch_size * grpo_size),
            successes=successes.reshape(batch_size * grpo_size),
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
            answers=answers,
        )

    def generate_trajectory_batched(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
    ) -> GRPOTrajectory:
        """
        Generates trajectories using forward_batch_size micro-batches.
        """
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_answers = answers[
                    batch_start : batch_start + self._forward_batch_size
                ]
                device_empty_cache(self._device)
                trajectories.append(
                    self.generate_trajectory(batch_input_ids, batch_answers)
                )
                device_empty_cache(self._device)

        # Concatenate all trajectory fields.
        # - Tensor fields: torch.cat along batch dim (skip if all None)
        # - List fields (answers): extend
        # - None fields: keep as None
        concatenated_fields = {}
        for field_name in trajectories[0]._fields:
            values = [getattr(traj, field_name) for traj in trajectories]
            if field_name == "answers":
                result = []
                for v in values:
                    result.extend(v)
                concatenated_fields[field_name] = result
            elif all(v is None for v in values):
                concatenated_fields[field_name] = None
            else:
                concatenated_fields[field_name] = torch.cat(values)

        return GRPOTrajectory(**concatenated_fields)

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """
        Perform a single GRPO optimization step over a batch of trajectories.
        """
        # Synchronize + flush UR resources before forward pass (prevents
        # UR_RESULT_ERROR_OUT_OF_RESOURCES after ~240 total sequences)
        if self._device.type == "xpu":
            torch.xpu.synchronize()

        # FSDP memory diagnostics: track memory at each phase
        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "pre_forward", log=log)
            # Reset peak stats to measure per-phase peaks
            if self._device.type == "xpu":
                try:
                    torch.xpu.reset_peak_memory_stats()
                except RuntimeError:
                    pass

        # estimate logprobs from the policy at the current optimisation step
        _fwd_t0 = time.perf_counter()


        if self._enable_packing:
            # Pack sequences to eliminate padding waste in forward/backward.
            from torchtune.dev.rl.packing import (
                pack_trajectory_for_training,
                unpack_tensor,
            )
            packed_tokens, packed_positions, packed_masks, bins, actual_lens = (
                pack_trajectory_for_training(
                    trajectory.query_responses,
                    trajectory.position_ids,
                    self._tokenizer.pad_id,
                )
            )
            log.info(
                "Rank %d: grpo_step packed forward start (%d seqs -> %d packs)",
                self.rank, trajectory.query_responses.shape[0], packed_tokens.shape[0],
            )
            packed_logits = self._model(
                packed_tokens, input_pos=packed_positions, mask=packed_masks,
            )
            del packed_tokens, packed_positions, packed_masks
            # Unpack back to per-sequence layout
            pi_logits = unpack_tensor(
                packed_logits, bins, actual_lens,
                num_sequences=trajectory.query_responses.shape[0],
                total_len=trajectory.query_responses.shape[1],
            )
            del packed_logits
        elif (
            os.environ.get("TORCHTUNE_USE_CHUNKED_LOSS") == "1"
            and self._expert_parallel_degree <= 1
        ):
            # Single forward + single backward (non-EP only).
            total_seqs = trajectory.query_responses.shape[0]
            grad_scale = max(1, self._gradient_accumulation_steps)

            log.info("Rank %d: single-backward forward start (total=%d seqs)", self.rank, total_seqs)
            _fwd_t0_sb = time.perf_counter()
            pi_logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
            )
            pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
            pi_logprobs = rlhf.batched_logits_to_logprobs(
                pi_logits,
                trajectory.query_responses[:, context_length:],
                self._temperature,
                chunk_size=1,
            )
            pi_logprobs.masked_fill_(trajectory.response_padding_masks, 1.0)
            del pi_logits
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _fwd_time_sb = time.perf_counter() - _fwd_t0_sb
            log.info("Rank %d: single-backward forward=%.1fs", self.rank, _fwd_time_sb)

            if self._compute_rollout_logprobs_required:
                assert trajectory.logprobs is not None, (
                    "async_generation / always_compute_rollout_logprobs is set but "
                    "trajectory.logprobs is None — rollout-time policy fwd was not "
                    "run; cannot fall back to .detach() without breaking IS ratios"
                )
            old_logprobs = trajectory.logprobs if trajectory.logprobs is not None else pi_logprobs.detach()
            loss, policy_loss, kl_loss, ratios, clipfrac = self._loss_fn(
                old_logprobs,
                pi_logprobs,
                trajectory.ref_logprobs,
                trajectory.advantages,
                padding_masks=~trajectory.response_padding_masks,
            )

            # MEMPROBE: per-rank L0-truth snapshot before backward. Opt-in.
            _dump_mem_sb = None
            if os.environ.get("TORCHTUNE_MEM_PROBE"):
                try:
                    import sys as _sys_sb
                    _mp_path_sb = "/lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b"
                    if _mp_path_sb not in _sys_sb.path:
                        _sys_sb.path.insert(0, _mp_path_sb)
                    from mem_probe import dump_mem as _dump_mem_sb
                    _dump_mem_sb(f"PRE-BWD step={self._steps_run} single-bwd")
                except Exception:
                    _dump_mem_sb = None

            log.info("Rank %d: single-backward backward start", self.rank)
            _bwd_t0_sb = time.perf_counter()
            # Bypass gloo CPU-AllReduce reduce_scatter patch: XCCL reduce_scatter is
            # correct and fast for non-EP single-node FSDP2 (W-old: 7.2s vs gloo: 130s).
            # The module-level patch routes reduce_scatter_tensor through D2H+gloo+H2D
            # per layer, adding ~2s × 64 layers = 130s to backward for non-EP runs.
            import torch.distributed as _tdist_sb_fix
            _rsc_patch_saved = _tdist_sb_fix.reduce_scatter_tensor
            _tdist_sb_fix.reduce_scatter_tensor = _orig_reduce_scatter_tensor
            try:
                try:
                    (loss / grad_scale).backward()  # loss passed to GRPOStats is UNSCALED
                except Exception as _bwd_exc_sb:
                    try:
                        if _dump_mem_sb is not None:
                            _dump_mem_sb(f"BWD-FAIL step={self._steps_run} single-bwd exc={type(_bwd_exc_sb).__name__}")
                    except Exception:
                        pass
                    log.error("Rank %d: SINGLE-BWD FAILED step=%d exc=%r", self.rank, self._steps_run, _bwd_exc_sb)
                    raise
            finally:
                _tdist_sb_fix.reduce_scatter_tensor = _rsc_patch_saved
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _bwd_total = time.perf_counter() - _bwd_t0_sb
            try:
                if _dump_mem_sb is not None:
                    _dump_mem_sb(f"POST-BWD step={self._steps_run} single-bwd")
            except Exception:
                pass
            log.info("Rank %d: single-backward backward=%.1fs", self.rank, _bwd_total)
            _fwd_time = _fwd_time_sb

        else:
            # Chunked training forward+backward: process self._forward_batch_size
            # sequences at a time. Each chunk gets its own backward() call so
            # autograd only holds one chunk's activation graph in memory at once.
            # Gradients accumulate across chunks before the optimizer step.
            # Mathematically equivalent to a single forward+backward over all seqs.
            total_seqs = trajectory.query_responses.shape[0]
            fwd_bs = self._forward_batch_size
            num_fwd_chunks = (total_seqs + fwd_bs - 1) // fwd_bs
            grad_scale = num_fwd_chunks * max(1, self._gradient_accumulation_steps)

            # FSDP2 reduce-scatter suppression for multi-chunk backward.
            # With num_fwd_chunks > 1, each backward() call would normally
            # fire reduce-scatter (AllReduce for FSDP1). That's num_fwd_chunks×
            # the communication cost. Suppress for all but the final chunk:
            # gradients accumulate locally on each rank; final chunk fires once.
            # Use set_requires_gradient_sync() for FSDP2, no_sync() for FSDP1.
            _use_fsdp2_grad_sync = (
                num_fwd_chunks > 1
                and hasattr(self._model, 'set_requires_gradient_sync')
                and not self._use_fsdp1
            )
            _use_fsdp1_no_sync = (
                num_fwd_chunks > 1
                and self._use_fsdp1
                and hasattr(self._model, 'no_sync')
            )
            # DDP (nn.parallel.DistributedDataParallel) also has no_sync().
            _use_ddp_no_sync = (
                num_fwd_chunks > 1
                and not self._use_fsdp1
                and not hasattr(self._model, 'set_requires_gradient_sync')
                and isinstance(self._model, torch.nn.parallel.DistributedDataParallel)
            )

            _chunk_losses, _chunk_policy_losses, _chunk_kl_losses = [], [], []
            _chunk_ratios, _chunk_clipfracs, _chunk_pi_logprobs = [], [], []
            _bwd_total = 0.0

            for _cs in range(0, total_seqs, fwd_bs):
                _is_last_chunk = (_cs + fwd_bs >= total_seqs)
                _ce = min(_cs + fwd_bs, total_seqs)
                if self._device.type == "xpu" and self._is_rank_zero:
                    _pre_fwd_alloc = torch.xpu.memory_allocated() / 1024**3
                    _pre_fwd_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: PRE-train-fwd[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce, _pre_fwd_alloc, _pre_fwd_resv,
                    )
                log.info("Rank %d: grpo_step chunk[%d:%d] fwd", self.rank, _cs, _ce)
                _c_logits = self._model(
                    trajectory.query_responses[_cs:_ce],
                    input_pos=trajectory.position_ids[_cs:_ce],
                    mask=trajectory.masks[_cs:_ce],
                )
                _c_logits = rlhf.truncate_sequence_for_logprobs(_c_logits, context_length)
                _c_pi_lp = rlhf.batched_logits_to_logprobs(
                    _c_logits,
                    trajectory.query_responses[_cs:_ce, context_length:],
                    self._temperature,
                    chunk_size=1,
                )
                # Use masked_fill_ instead of boolean index assignment.
                # Boolean indexing triggers L0 sub-allocation (gather/scatter)
                # while FSDP storage is live, causing UR:40 handle exhaustion.
                _c_pi_lp.masked_fill_(trajectory.response_padding_masks[_cs:_ce], 1.0)
                del _c_logits
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                if self._device.type == "xpu" and self._is_rank_zero:
                    _post_fwd_alloc = torch.xpu.memory_allocated() / 1024**3
                    _post_fwd_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: POST-train-fwd[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce, _post_fwd_alloc, _post_fwd_resv,
                    )

                if self._compute_rollout_logprobs_required:
                    assert trajectory.logprobs is not None, (
                        "async_generation / always_compute_rollout_logprobs is set but "
                        "trajectory.logprobs is None — rollout-time policy fwd was not "
                        "run; cannot fall back to .detach() without breaking IS ratios"
                    )
                _c_old_lp = trajectory.logprobs[_cs:_ce] if trajectory.logprobs is not None else _c_pi_lp.detach()
                _c_loss, _c_pol, _c_kl, _c_rat, _c_clip = self._loss_fn(
                    _c_old_lp,
                    _c_pi_lp,
                    trajectory.ref_logprobs[_cs:_ce],
                    trajectory.advantages[_cs:_ce],
                    padding_masks=~trajectory.response_padding_masks[_cs:_ce],
                )

                # Pre-backward memory logging (no empty_cache — leaks UR handles).
                # With expert_cpu_offload=True, base resv is ~1.2 GiB so no
                # need to return cached blocks to L0 before backward.
                if _cs == 0 and self._device.type == "xpu":
                    torch.xpu.synchronize()
                    if self._is_rank_zero:
                        _pre_bwd_alloc = torch.xpu.memory_allocated() / 1024**3
                        _pre_bwd_resv = torch.xpu.memory_reserved() / 1024**3
                        log.info(
                            "Rank 0: pre-bwd (no empty_cache): alloc=%.2f GiB, resv=%.2f GiB",
                            _pre_bwd_alloc, _pre_bwd_resv,
                        )
                elif self._device.type == "xpu" and self._is_rank_zero:
                    _pre_bwd_alloc = torch.xpu.memory_allocated() / 1024**3
                    _pre_bwd_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: PRE-backward[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce, _pre_bwd_alloc, _pre_bwd_resv,
                    )
                # v84: CPU-bounce AllToAll (via gloo) bypasses XCCL/OFI/ze_handle_manager.
                # v83 pre-backward XCCL sync removed — CCL worker drain no longer needed
                # since AllToAll no longer uses XCCL at all.
                log.info("Rank %d: chunk[%d:%d] backward start", self.rank, _cs, _ce)
                _bwd_chunk_t0 = time.perf_counter()
                # MEMPROBE v1: per-rank L0-truth memory snapshot before backward. Opt-in.
                _dump_mem = None
                if os.environ.get("TORCHTUNE_MEM_PROBE"):
                    try:
                        import sys as _sys, os as _os
                        _mp_path = "/lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b"
                        if _mp_path not in _sys.path:
                            _sys.path.insert(0, _mp_path)
                        from mem_probe import dump_mem as _dump_mem
                        _dump_mem(f"PRE-BWD step={self._steps_run} chunk[{_cs}:{_ce}]")
                    except Exception as _mp_e:
                        log.warning("Rank %d: mem_probe import/PRE-BWD failed: %r", self.rank, _mp_e)
                        _dump_mem = None
                # v111 diag: log token dispatch counts for all ranks (first step only).
                # _ep_dispatch is partial(ep_instance._token_dispatch, ...) — access ep_instance
                # via _ep_dispatch.func.__self__ (bound method's __self__).
                if self._steps_run == 0 and _cs == 0 and self._expert_parallel_degree > 1:
                    from torchtune.modules.moe.moe import MoE as _MoE
                    _total_recv = 0
                    _total_send = 0
                    _n_layers_found = 0
                    for _m in self._model.modules():
                        if not isinstance(_m, _MoE):
                            continue
                        _ep_disp = getattr(_m, '_ep_dispatch', None)
                        if _ep_disp is None:
                            continue
                        # partial.func = bound method; .func.__self__ = ExpertParallel instance
                        _ep_inst = getattr(getattr(_ep_disp, 'func', None), '__self__', None)
                        if _ep_inst is None:
                            continue
                        _splits_out = getattr(_ep_inst, '_output_splits', None)
                        _splits_in = getattr(_ep_inst, '_input_splits', None)
                        if _splits_out is not None:
                            _total_recv += sum(_splits_out)
                            _n_layers_found += 1
                        if _splits_in is not None:
                            _total_send += sum(_splits_in)
                    log.info(
                        "Rank %d: EP TOKEN LOAD step=0 total_recv=%d total_send=%d "
                        "layers=%d (via _ep_dispatch.func.__self__)",
                        self.rank, _total_recv, _total_send, _n_layers_found,
                    )
                # v59: EP mode — all FSDPParamGroups have reduce_grads=False permanently.
                # Do NOT call set_requires_gradient_sync: it would re-enable reduce_grads
                # on the last chunk, undoing our v59 suppression and triggering XCCL.
                # Post-backward gloo AllReduce in _ep_post_backward_grad_sync handles sync.
                # Non-EP path: standard grad accumulation (suppress all but last chunk).
                if not self._fsdp2_param_groups_meta and _use_fsdp2_grad_sync and \
                        self._expert_parallel_degree <= 1:
                    # Non-EP FSDP2: standard grad accumulation (suppress all but last chunk).
                    self._model.set_requires_gradient_sync(_is_last_chunk)
                try:
                    if (_use_fsdp1_no_sync or _use_ddp_no_sync) and not _is_last_chunk:
                        with self._model.no_sync():
                            (_c_loss / grad_scale).backward()
                    else:
                        (_c_loss / grad_scale).backward()
                except Exception as _bwd_exc:
                    try:
                        if _dump_mem is not None:
                            _dump_mem(f"BWD-FAIL step={self._steps_run} chunk[{_cs}:{_ce}] exc={type(_bwd_exc).__name__}")
                    except Exception:
                        pass
                    log.error("Rank %d: BACKWARD FAILED step=%d chunk[%d:%d] exc=%r",
                              self.rank, self._steps_run, _cs, _ce, _bwd_exc)
                    raise
                # MEMPROBE v1: per-rank L0-truth snapshot after backward returns.
                try:
                    if _dump_mem is not None:
                        _dump_mem(f"POST-BWD step={self._steps_run} chunk[{_cs}:{_ce}]")
                except Exception:
                    pass
                # v110 diag: log immediately after backward() returns (before xpu.sync).
                log.info(
                    "Rank %d: chunk[%d:%d] backward() returned (pre-sync), elapsed=%.1fs",
                    self.rank, _cs, _ce, time.perf_counter() - _bwd_chunk_t0,
                )
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                # v9: per-chunk release of FSDP2's unsharded grad pool into sharded
                # `param.grad` as a DTensor. With v59 setting `reduce_grads=False`,
                # FSDP2 never fires reduce_scatter; without this hook, unsharded
                # grads accumulate across chunks (blocks G≥2) and across steps
                # (blocks NSTEPS≥2). The chunk-N>0 call accumulates into the
                # existing sharded grad. v9a probe (2026-04-30) confirmed
                # `param.grad is None` and `_unsharded_param.grad` is ~593 MiB/param.
                if (self._expert_parallel_degree > 1
                        and getattr(self, "_ep_grad_release_pg_map", None)):
                    _rel_t0 = time.perf_counter()
                    _accumulate = (_cs > 0)
                    try:
                        _n_rel = _ep_release_fsdp_unsharded_grads(
                            self._model,
                            self._ep_grad_release_pg_map,
                            accumulate_into_grad=_accumulate,
                        )
                        _rel_dt = time.perf_counter() - _rel_t0
                        log.info(
                            "Rank %d: EP v9 grad release chunk[%d:%d] accumulate=%s "
                            "groups=%d in %.2fs",
                            self.rank, _cs, _ce, _accumulate, _n_rel, _rel_dt,
                        )
                    except Exception as _rel_exc:
                        log.error(
                            "Rank %d: EP v9 grad release FAILED chunk[%d:%d]: %r",
                            self.rank, _cs, _ce, _rel_exc,
                        )
                        raise
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    try:
                        if _dump_mem is not None:
                            _dump_mem(f"POST-REL step={self._steps_run} chunk[{_cs}:{_ce}]")
                    except Exception:
                        pass
                _bwd_total += time.perf_counter() - _bwd_chunk_t0

                _chunk_losses.append(_c_loss.detach())
                _chunk_policy_losses.append(_c_pol.detach() if torch.is_tensor(_c_pol) else _c_pol)
                _chunk_kl_losses.append(_c_kl.detach() if torch.is_tensor(_c_kl) else _c_kl)
                _chunk_ratios.append(_c_rat.detach())
                _chunk_clipfracs.append(_c_clip.detach() if torch.is_tensor(_c_clip) else _c_clip)
                _chunk_pi_logprobs.append(_c_pi_lp.detach())

            if self._device.type == "xpu":
                torch.xpu.synchronize()

            # EP v75 post-backward grad sync via XCCL dp_replicate group.
            # With reduce_grads=False on all FSDPParamGroups, FSDP2 did NOT fire
            # reduce_scatter during backward. All param grads are local. Now sync
            # all param grads across dp_replicate via XCCL all_reduce (AVG).
            #
            # v71-v74 all failed because any sync requiring ALL 12 ranks deadlocks:
            #   fast EP groups finish backward first → their ranks try to sync →
            #   slow EP group ranks are still in gloo SHARD AllToAll → can't join
            #   the 12-rank sync → mutual deadlock → 1800s timeout.
            #
            # v75 fix: use XCCL all_reduce on dp_replicate group (3 ranks) only.
            #   dp_replicate pairs (e.g. (0,4),(0,8),(4,8)) are DISJOINT from
            #   dp_shard/SHARD pairs (within {0,1,2,3},{4,5,6,7},{8,9,10,11}).
            #   XCCL REP all_reduce operates on a different rank-pair set than
            #   gloo SHARD AllToAll → zero interference, no pre-sync needed.
            #   When slow EP group {0,1,2,3} finishes backward, ranks 0,1,2,3
            #   each call XCCL all_reduce on their respective REP groups:
            #     rank 0 → REP {0,4,8}  (pairs (0,4),(0,8) — disjoint from SHARD)
            #     rank 1 → REP {1,5,9}  (pairs (1,5),(1,9) — disjoint from SHARD)
            #   Since fast ranks 4,8 (etc.) are already waiting in their XCCL
            #   all_reduce for this call, the all_reduce completes immediately.
            # v75 XCCL post-bwd grad sync is REDUNDANT under v9: the per-chunk
            # _ep_release_fsdp_unsharded_grads helper already reduces grads on each
            # FSDPParamGroup's correct gloo PG (dp_shard for expert groups, dp_replicate
            # for non-expert/policy groups). Re-running XCCL all_reduce here on
            # _XCCL_DP_REP_PG would (a) double-divide policy grads and (b) trip
            # OFI EPERM (err=265) because the XCCL endpoint was left in a bad state
            # after the long gloo bounce. Skip when v9 helper is active.
            _v9_active = (
                self._expert_parallel_degree > 1
                and getattr(self, "_ep_grad_release_pg_map", None) is not None
            )
            if self._expert_parallel_degree > 1 and self._dp_replicate > 1 and not _v9_active:
                _grad_sync_t0 = time.perf_counter()
                _n_synced = _ep_post_backward_grad_sync_xccl(
                    self._model, self._dp_replicate
                )
                _grad_sync_time = time.perf_counter() - _grad_sync_t0
                log.info(
                    "Rank %d: EP v75 XCCL post-bwd grad sync: %d params in %.2fs",
                    self.rank, _n_synced, _grad_sync_time,
                )

            _fwd_time = time.perf_counter() - _fwd_t0 - _bwd_total
            log.info("Rank %d: grpo_step forward=%.1fs", self.rank, _fwd_time)
            log.info("Rank %d: backward=%.1fs", self.rank, _bwd_total)

            loss = torch.stack(_chunk_losses).mean()
            policy_loss = (
                torch.stack(_chunk_policy_losses).mean()
                if torch.is_tensor(_chunk_policy_losses[0])
                else sum(_chunk_policy_losses) / len(_chunk_policy_losses)
            )
            kl_loss = (
                torch.stack(_chunk_kl_losses).mean()
                if torch.is_tensor(_chunk_kl_losses[0])
                else sum(_chunk_kl_losses) / len(_chunk_kl_losses)
            )
            ratios = torch.stack(_chunk_ratios).mean()
            clipfrac = (
                torch.stack(_chunk_clipfracs).mean()
                if torch.is_tensor(_chunk_clipfracs[0])
                else sum(_chunk_clipfracs) / len(_chunk_clipfracs)
            )
            pi_logprobs = torch.cat(_chunk_pi_logprobs, dim=0)

        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "post_backward", log=log)

        with torch.no_grad():
            _old_lp = trajectory.logprobs if trajectory.logprobs is not None else pi_logprobs
            approx_policy_kls = (
                0.5 * (pi_logprobs - _old_lp).pow(2)
            ).mean()

        return GRPOStats(
            loss,  # already unscaled (chunks averaged, not summed)
            policy_loss,
            kl_loss,
            ratios,
            clipfrac,
            approx_policy_kls,
            None,  # metadata
        )

    def train(self) -> None:
        """
        The core training loop with gradient accumulation and no_sync() support.
        """
        # clean up before training begins
        try:
            training.cleanup_before_training()
        except RuntimeError:
            pass

        # Dedicated vLLM rank: run generation server loop, then exit.
        if self._is_vllm_rank:
            self._run_vllm_generation_server()
            return

        # v86b: Pre-register XPU L0 root blocks with CCL ze_handle_manager before training.
        #
        # v86 finding: 64 MiB × ~576 chunks hit UR_RESULT_ERROR_OUT_OF_RESOURCES (UR:40) —
        # too many concurrent L0 memory objects. Fix: use 2 GiB chunks (~18 total), one
        # XCCL all_reduce per chunk to register each root block individually.
        #
        # Root cause of SIGABRT: At peak HBM during AC recompute backward, FSDP2 AllGather
        # for expert params forces a NEW L0 root block allocation (beyond pool). CCL's
        # ze_handle_manager doesn't know about new root blocks → ZE_MEMORY_TYPE_UNKNOWN → crash.
        #
        # Fix: allocate 2 GiB chunks until pool is at expected peak (~50.5 GiB target),
        # run one XCCL all_reduce on a VIEW of each chunk (same memory → same root block)
        # to register each root block with ze_handle_manager, then free all chunks.
        # During training, no new root blocks needed → no crash.
        if self._device.type == "xpu" and self._expert_parallel_degree > 1:
            # v95: No XPU warmup needed — CPU-bounce AllToAll (gloo P2P) eliminates
            # ze_handle_manager by removing all XPU AllToAll calls entirely.
            #
            # v87-v92 warmup history (all exhausted):
            #   v87: 50.50 GiB pool → +3.38 GiB during gen → crash at backward
            #   v90: 53.50 GiB pool → +3.38 GiB during gen → crash at backward
            #   v91: 58.00 GiB pool → +3.38 GiB during gen → crash at backward
            #   v92: 57.35 GiB (ONE 45 GiB chunk) → +3.38 GiB during gen → crash at backward
            # Root cause: CCL's AllToAll internal workspace is CCL-internal (not pre-registerable).
            # v93: XCCL doesn't support CPU tensors — failed.
            # v94: gloo all_to_all list-API not supported — failed.
            # v95: gloo P2P (batch_isend_irecv) — correct approach.
            log.info(
                "Rank %d: v95 — gloo P2P CPU-bounce AllToAll active, no XPU warmup needed. "
                "cur_resv=%.2f GiB",
                self.rank, torch.xpu.memory_reserved() / 1024**3,
            )

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Pre-warm FSDP1 summon_full_params AllGather buffer before training.
        #
        # Root cause of banned:1 / UR:40 at step 1 (runs 32-36):
        #   - After step 0 backward ReduceScatter, 7.49 GiB param buffer is freed to cache.
        #   - summon_full_params (weight sync) needs 7.49 GiB on ALL training ranks.
        #   - If cache lacks a matching block, L0 allocates a NEW 7.49 GiB block.
        #   - New L0 alloc pushes reserved past GC threshold → GC calls zeMemFree on
        #     backward's XCCL-registered AllGather buffers → stale IPC handles at step 1.
        #
        # Fix: pre-warm puts 7.49 GiB in cache BEFORE the first backward. Subsequent FSDP
        # AllGathers reuse this cached block, so summon_full_params also reuses it (no new
        # L0 alloc at weight sync time). Pool stays ≤54 GiB → GC never fires.
        if hasattr(self._model, '_fsdp_wrapped_module') or (
            hasattr(torch.distributed.fsdp, 'FullyShardedDataParallel')
            and isinstance(self._model, torch.distributed.fsdp.FullyShardedDataParallel)
        ):
            if self._vllm_mode == "dedicated_rank":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                log.info(
                    "Rank %d: pre-warming FSDP summon_full_params cache (7.49 GiB AllGather buffer)...",
                    self.rank,
                )
                with FSDP.summon_full_params(self._model, writeback=False, rank0_only=True):
                    pass  # warm the AllGather buffer into cache; reused by fwd/bwd + weight sync
                if self._device.type == "xpu":
                    log.info(
                        "Rank %d: post-prewarm: alloc=%.2f GiB, resv=%.2f GiB",
                        self.rank,
                        torch.xpu.memory_allocated() / 1024**3,
                        torch.xpu.memory_reserved() / 1024**3,
                    )

        # Initialize tokens count and running loss (for grad accumulation)
        grad_norm = None

        training_completed = False
        self._profiler.start()
        for curr_epoch in range(self._epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(_async_lookahead_iter(self, self._dataloader)):
                # Start tracking memory for active steps
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=True)

                tokens = batch["tokens"]
                answers = batch["answers"]
                tokens = tokens.to(self._device)

                _, context_length = tokens.shape

                # NOTE: No empty_cache() here. With PYTORCH_ALLOC_CONF=expandable_segments:True,
                # the allocator handles fragmentation internally without UR handle leaks.

                # Memory diagnostics before each step
                if self._device.type == "xpu" and self.rank == 0:
                    _alloc = torch.xpu.memory_allocated() / 1024**3
                    _resv = torch.xpu.memory_reserved() / 1024**3
                    log.info("Rank 0: PRE-STEP %d memory: allocated=%.2f GiB, reserved=%.2f GiB",
                             self._steps_run, _alloc, _resv)

                # NOTE: XCCL broadcast is DEFERRED — it starts after gen completes
                # (in _start_deferred_broadcast below), runs during GRPO/backward.
                # vLLM generates with the latest available weights (no contention).

                # Subclass hook: extra kwargs (e.g. multimodal protein_sequences)
                # forwarded into generate_trajectory_batched. Default returns {}.
                _extra_gen_kwargs = self._extract_batch_kwargs(batch)

                _step_t0 = time.perf_counter()
                trajectory = self.generate_trajectory_batched(tokens, answers, **_extra_gen_kwargs)
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _gen_time = time.perf_counter() - _step_t0
                if not self._production_mode:
                    self._training_barrier()  # dedicated_rank: training_pg only

                if self._device.type == "xpu" and self._is_rank_zero:
                    _alloc = torch.xpu.memory_allocated() / 1024**3
                    _resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: post-gen memory: alloc=%.2f GiB, resv=%.2f GiB",
                        _alloc, _resv,
                    )

                # Start deferred XCCL broadcast now that vLLM generation is done.
                # Broadcast runs during GRPO/backward below (vLLM idle, no contention).
                if (self._vllm_mode == "server" and self._vllm_weight_sync
                        and self._vllm_weight_sync_method == "xccl"):
                    self._start_deferred_broadcast()

                grpo_stats: list[GRPOStats] = []
                _grpo_t0 = time.perf_counter()

                for _ in range(self._ppo_epochs):
                    total_samples = trajectory.query_responses.shape[0]

                    if self._gradient_accumulation_steps > 1:
                        # Gradient accumulation: disable gradient sync for
                        # non-final micro-batches.
                        # FSDP2 uses set_requires_gradient_sync()
                        # FSDP1 (HSDP) uses no_sync() context manager
                        micro_batch_size = total_samples // self._gradient_accumulation_steps
                        for ga_step in range(self._gradient_accumulation_steps):
                            start_idx = ga_step * micro_batch_size
                            end_idx = start_idx + micro_batch_size
                            micro_trajectory = _slice_trajectory(
                                trajectory, start_idx, end_idx
                            )

                            is_last = (ga_step == self._gradient_accumulation_steps - 1)
                            if hasattr(self._model, 'set_requires_gradient_sync'):
                                # FSDP2 path
                                self._model.set_requires_gradient_sync(is_last)
                                step_stats = self.grpo_step(
                                    micro_trajectory, context_length
                                )
                            elif not is_last and hasattr(self._model, 'no_sync'):
                                # FSDP1 path
                                with self._model.no_sync():
                                    step_stats = self.grpo_step(
                                        micro_trajectory, context_length
                                    )
                            else:
                                step_stats = self.grpo_step(
                                    micro_trajectory, context_length
                                )
                            grpo_stats.append(step_stats)

                            # Memory cleanup between GA steps — gc only,
                            # skip empty_cache (leaks UR handles with FSDP)
                            if not is_last and self._device.type == "xpu":
                                import gc
                                gc.collect()
                    else:
                        # No gradient accumulation — single step.
                        # NOTE: empty_cache() is now called INSIDE grpo_step before
                        # the first backward chunk (after training forward). This is
                        # more effective because the training forward itself pushes
                        # resv from 11.9 → 18.3 GiB, and we need to reclaim that
                        # before backward can allocate recompute + AllGather buffers.
                        step_stats = self.grpo_step(trajectory, context_length)
                        grpo_stats.append(step_stats)

                    # Sync device before timing grad clip
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _grpo_time = time.perf_counter() - _grpo_t0

                    # v59: ALL grad averaging (expert + non-expert) is now done inside
                    # grpo_step() via _ep_post_backward_grad_sync() using gloo AllReduce.
                    # reduce_grads=False on all FSDPParamGroups prevents FSDP2 from firing
                    # reduce_scatter during backward. The post-backward pass does a single
                    # sequential gloo AllReduce over all params in model.parameters() order,
                    # ensuring consistent ordering across dp_replicate ranks.
                    # No separate expert grad averaging needed here.

                    _clip_t0 = time.perf_counter()
                    if self._clip_grad_norm is not None:
                        if self._expert_parallel_degree > 1:
                            # EP mixes two DTensor meshes: non-expert params on dp_mesh
                            # (2D: dp_replicate×dp_shard) and expert params on ep_mesh
                            # (1D: dp_shard). torch.nn.utils.clip_grad_norm_ calls
                            # torch.stack on per-param norm DTensors, which fails when
                            # the tensors have different meshes.
                            #
                            # v77: compute both non-EP and EP norms purely locally.
                            # v76 root cause: clip_grad_norm_(_non_ep_params, inf) on FSDP2
                            # DTensor params triggers an XCCL all_reduce on self._shard_pg
                            # internally (for DTensor norm aggregation). The subsequent
                            # explicit _orig_all_reduce(_ep_norm_sq_xpu, group=self._shard_pg)
                            # is then a SECOND XCCL collective on the same communicator.
                            # With token routing imbalance (ep_rank 0: 32k tokens, ep_ranks 2,3:
                            # 0 tokens), SHARD groups finish backward at different times → the
                            # two sequential XCCL ops on shard_pg arrive at different clock offsets
                            # across the 4 ranks → communicator sequence mismatch → 1800s timeout.
                            # Fix: compute all norms manually (local tensors only, no distributed
                            # collectives in the norm computation). Each rank computes its local
                            # approximation of the global norm and applies clip_coef locally.
                            # This is already a valid approximation for FSDP2 sharded grads.
                            from torchtune.modules.moe.experts import GroupedExperts
                            from torchtune.models.qwen3_moe._experts import GroupedExpertsHF
                            _ep_param_ids = set()
                            _ep_params = []
                            for _mn, _mm in self._model.named_modules():
                                if _mn.endswith(".experts") and isinstance(_mm, (GroupedExperts, GroupedExpertsHF)):
                                    for _p in _mm.parameters(recurse=False):
                                        _ep_param_ids.add(id(_p))
                                        _ep_params.append(_p)
                            _non_ep_params = [
                                _p for _p in self._model.parameters()
                                if id(_p) not in _ep_param_ids
                            ]
                            # Compute local norm_sq for non-EP params (no XCCL, no DTensor norm).
                            _non_ep_norm_sq_val = 0.0
                            for _p in _non_ep_params:
                                if _p.grad is not None:
                                    _g = _p.grad
                                    if hasattr(_g, '_local_tensor'):
                                        _g = _g._local_tensor
                                    _non_ep_norm_sq_val += float(_g.float().norm().item() ** 2)
                            # Compute local norm_sq for EP params.
                            _ep_norm_sq_val = 0.0
                            for _p in _ep_params:
                                if _p.grad is not None:
                                    _g = _p.grad
                                    if hasattr(_g, '_local_tensor'):
                                        _g = _g._local_tensor
                                    _ep_norm_sq_val += float(_g.float().norm().item() ** 2)
                            # Local total norm (no cross-rank all_reduce — see v77 comment above).
                            _total_norm_f = (_non_ep_norm_sq_val + _ep_norm_sq_val) ** 0.5
                            _max_norm_f = float(self._clip_grad_norm)
                            _clip_coef = _max_norm_f / max(_total_norm_f, 1e-6)
                            if _clip_coef < 1.0:
                                for _p in self._model.parameters():
                                    if _p.grad is not None:
                                        _g = _p.grad
                                        if hasattr(_g, '_local_tensor'):
                                            _g._local_tensor.detach().mul_(_clip_coef)
                                        else:
                                            _g.detach().mul_(_clip_coef)
                            grad_norm = torch.tensor(_total_norm_f, device=self._device)
                        else:
                            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                            if getattr(self, '_use_fsdp1', False) and isinstance(self._model, FSDP):
                                # FSDP1 SHARD_GRAD_OP: grads are sharded — local norm ≠ global.
                                # Use FSDP.clip_grad_norm_ which AllReduces norm² across _training_pg
                                # before clipping. Safe here since training ranks are isolated
                                # from vLLM rank (no shared L0 fabric contention).
                                grad_norm = self._model.clip_grad_norm_(float(self._clip_grad_norm))
                            else:
                                # Non-EP path: avoid torch.nn.utils.clip_grad_norm_ on FSDP2
                                # DTensor params (its internal XCCL all_reduce on shard_pg
                                # deadlocks with vLLM L0/fabric usage when colocated).
                                # ALSO avoid per-param .item() — that's ~700 D2H syncs which
                                # also deadlocks with concurrent vLLM activity (Test D
                                # 2026-04-22). Compute the norm fully on-device with a
                                # single .item() at the end.
                                _local_norm_sq = torch.zeros((), device=self._device, dtype=torch.float32)
                                _grads_to_clip = []
                                for _p in self._model.parameters():
                                    if _p.grad is not None:
                                        _g = _p.grad
                                        if hasattr(_g, '_local_tensor'):
                                            _g = _g._local_tensor
                                        _local_norm_sq = _local_norm_sq + _g.float().pow(2).sum()
                                        _grads_to_clip.append(_p.grad)
                                grad_norm = _local_norm_sq.sqrt()
                                _max_norm = float(self._clip_grad_norm)
                                _clip_coef = (_max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                                for _g in _grads_to_clip:
                                    if hasattr(_g, '_local_tensor'):
                                        _g._local_tensor.detach().mul_(_clip_coef)
                                    else:
                                        _g.detach().mul_(_clip_coef)
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _clip_time = time.perf_counter() - _clip_t0

                    if not self._production_mode:
                        self._training_barrier()  # dedicated_rank: training_pg only

                    # Synchronize before optimizer — no empty_cache in
                    # colocate mode (it hangs with vLLM engines).
                    if _colocate_vllm_mode and self._device.type == "xpu":
                        log.info("Rank %d: pre-optimizer sync", self.rank)
                        torch.xpu.synchronize()

                    _opt_t0 = time.perf_counter()
                    log.info("Rank %d: optimizer.step()", self.rank)
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    # v9: defensive sweep — release any FSDP2 unsharded grad residue
                    # the per-chunk hook missed. On a clean step this is a no-op
                    # (logs a WARN if it finds residue, indicating a chunk-loop leak).
                    if (self._expert_parallel_degree > 1
                            and getattr(self, "_ep_grad_release_pg_map", None)):
                        try:
                            _n_swept = _ep_release_fsdp_unsharded_grads(
                                self._model,
                                self._ep_grad_release_pg_map,
                                accumulate_into_grad=False,
                                warn_on_residue=True,
                            )
                            log.info(
                                "Rank %d: EP v9 post-step defensive sweep groups=%d",
                                self.rank, _n_swept,
                            )
                        except Exception as _swp_exc:
                            log.warning(
                                "Rank %d: EP v9 post-step sweep raised %r (continuing)",
                                self.rank, _swp_exc,
                            )
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _opt_time = time.perf_counter() - _opt_t0
                    log.info("Rank %d: optimizer done", self.rank)

                    if self._fsdp_diagnostics and self._is_rank_zero:
                        training.log_fsdp_memory_per_phase(self._device, "post_optimizer", log=log)

                    # Remove per-layer hooks after first step to avoid noise
                    if self._fsdp_diagnostics and self._steps_run == 0 and self._layer_mem_hooks:
                        for h in self._layer_mem_hooks:
                            h.remove()
                        log.info("Removed per-layer memory hooks after step 0")
                        self._layer_mem_hooks = []

                    if not self._production_mode or self._expert_parallel_degree > 1:
                        # With EP, ranks finish optimizer at different times (ep_mesh
                        # AllGather costs differ by replica group). Without this barrier,
                        # rank 0 exits the training loop and starts Python teardown of
                        # FSDP2/ZE objects while other ranks are still in optimizer.step(),
                        # causing L0 use-after-free (SIGABRT in drm_neo.cpp:426).
                        #
                        # v78: replace dist.barrier() with XCCL SHARD all_reduce.
                        # dist.barrier() on the default (12-rank) gloo PG shares TCP
                        # pairs with SHARD gloo AllToAll → concurrent ops → deadlock
                        # (same root cause as v72 gloo global barrier deadlock).
                        #
                        # Root cause of v77 step-boundary deadlock:
                        #   EP token routing imbalance: ep_rank 0 (ranks 0,4,8) processes
                        #   ~32k tokens → backward takes 20+ seconds longer than ep_ranks
                        #   1-3 (~0-500 tokens). After XCCL REP grad_sync, fast ep_ranks
                        #   (1-3) finish optimizer and start step N+1 forward → call gloo
                        #   SHARD AllToAll at a2a#241. ep_rank 0 is still in optimizer →
                        #   never calls a2a#241 gloo AllToAll → 1800s timeout.
                        #
                        # Fix: XCCL SHARD all_reduce (4-rank) after optimizer.
                        #   - Uses self._shard_pg (XCCL, Slingshot fabric — not TCP).
                        #   - Forces all 4 SHARD members to meet here before step N+1.
                        #   - Safe: v77 removed all other XCCL SHARD calls post-backward,
                        #     so this is the ONLY XCCL SHARD collective in this phase.
                        #     No sequence mismatch possible.
                        #   - ep_ranks 1,2,3 wait for ep_rank 0 here → all proceed
                        #     to step N+1 generate() together → synchronized → no race.
                        if self._expert_parallel_degree > 1 and self._shard_pg is not None:
                            _ep_sync_t = torch.zeros(1, dtype=torch.float32, device=self._device)
                            _orig_all_reduce(_ep_sync_t, op=torch.distributed.ReduceOp.SUM,
                                             group=self._shard_pg)
                            log.info("Rank %d: EP v78 post-optimizer XCCL SHARD sync done", self.rank)
                        else:
                            self._training_barrier()  # dedicated_rank: training_pg only

                    self.global_step += 1

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                # Sync updated weights to vLLM (after all ppo_epochs)
                # For colocate_sleep, sync happens during wake_up in generate_trajectory
                if self._vllm_mode == "colocate":
                    self._sync_colocated_weights()
                elif self._vllm_mode == "dedicated_rank" and not self._is_vllm_rank:
                    # summon_full_params in _sync_dedicated_vllm_weights is a collective
                    # (all training ranks 0-10 must enter it); only rank 0 sends the broadcast.
                    self._sync_dedicated_vllm_weights()
                elif self._vllm_mode == "server" and self._vllm_weight_sync:
                    if self._steps_run % self._vllm_weight_sync_interval == 0:
                        # Wait for previous async broadcast to complete before starting a new one.
                        # This ensures the XCCL PG is free and vLLM has applied the previous weights.
                        self._wait_for_sync_complete()
                        self._sync_weights_to_vllm()

                # Stop tracking memory
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=None)

                _step_time = time.perf_counter() - _step_t0
                if self._is_rank_zero:
                    log.info(
                        "TIMING step=%d  total=%.1fs  gen=%.1fs  grpo=%.1fs  clip=%.1fs  opt=%.1fs  other=%.1fs",
                        self._steps_run, _step_time, _gen_time, _grpo_time,
                        _clip_time, _opt_time,
                        _step_time - _gen_time - _grpo_time - _clip_time - _opt_time,
                    )

                self._steps_run += 1
                if self._steps_run % self._log_every_n_steps == 0:
                    extra_metrics = {}
                    extra_metrics["lr"] = get_lr(self._optimizer)
                    if grad_norm is not None:
                        extra_metrics["grad_norm"] = grad_norm

                    # Concatenate GRPOStats fields properly
                    concatenated_stats = {}
                    for field_name in grpo_stats[0]._fields:
                        if field_name == "metadata":
                            concatenated_stats[field_name] = None
                        else:
                            concatenated_stats[field_name] = torch.stack(
                                [getattr(stat, field_name) for stat in grpo_stats]
                            )

                    self.log_metrics(
                        trajectory,
                        GRPOStats(**concatenated_stats),
                        **extra_metrics,
                    )

                self.cleanup_after_step(trajectory, grpo_stats)

                if self._device.type == "xpu" and not self.fsdp_cpu_offload:
                    torch.xpu.synchronize()
                    _mem_alloc = torch.xpu.memory_allocated() / 1024**3
                    _mem_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank %d: between-step memory: allocated=%.2f GiB, "
                        "reserved=%.2f GiB, gap=%.2f GiB",
                        self.rank, _mem_alloc, _mem_resv, _mem_resv - _mem_alloc,
                    )

                # Periodic evaluation on held-out data
                if (self._eval_enabled and self._steps_run > 0 and
                        self._steps_run % self._eval_every_n_steps == 0):
                    self.run_eval()

                # Step-based checkpointing
                if (self._save_every_n_steps and self._steps_run > 0 and
                        self._steps_run % self._save_every_n_steps == 0):
                    try:
                        self.save_checkpoint(curr_epoch)
                    except Exception as e:
                        utils.log_rank_zero(
                            log,
                            f"WARNING: Step checkpoint save failed: {e}. "
                            "Continuing training.",
                        )

                self._profiler.step()

                pbar.update(1)

                if self._steps_run == self._total_steps:
                    # Final eval (skip if periodic eval already ran this step)
                    if (self._eval_enabled and
                            self._steps_run % self._eval_every_n_steps != 0):
                        self.run_eval()
                    if self._save_final_checkpoint:
                        try:
                            self.save_checkpoint(curr_epoch)
                        except Exception as e:
                            utils.log_rank_zero(
                                log,
                                f"WARNING: Final checkpoint save failed: {e}. "
                                "Training results are still valid.",
                            )
                    training_completed = True
                    break

            self._epochs_run += 1
            if self._epochs_run % self._save_every_n_epochs == 0:
                try:
                    self.save_checkpoint(curr_epoch)
                except Exception as e:
                    utils.log_rank_zero(
                        log,
                        f"WARNING: Epoch checkpoint save failed: {e}. "
                        "Continuing training.",
                    )
            if training_completed:
                # Phase 2 Step 4: signal the dedicated vLLM rank to exit its
                # generation loop cleanly. Only rank 0 is paired with the vLLM
                # rank over gen_pg, so only rank 0 sends the sentinel.
                if (
                    self._vllm_mode == "dedicated_rank"
                    and self._is_rank_zero
                    and getattr(self, "_gen_pg", None) is not None
                ):
                    # A2 bg send: drain in-flight broadcast on wsync_pg so the
                    # vLLM rank's bg wsync thread receives it cleanly before we
                    # shut down its main loop.
                    if getattr(self, '_bg_send_done_evt', None) is not None:
                        self._bg_send_done_evt.wait(timeout=300.0)
                        if getattr(self, '_bg_send_error', None) is not None:
                            log.warning("Rank 0 bg wsync send error at shutdown: %r",
                                        self._bg_send_error)
                    try:
                        torch.distributed.broadcast_object_list(
                            [{"shutdown": True}], src=0, group=self._gen_pg
                        )
                    except Exception as _e:
                        log.warning("Failed to send vLLM shutdown sentinel: %r", _e)
                if self._expert_parallel_degree > 1:
                    # Sync all ranks before teardown. Without this, fast ranks may start
                    # Python GC of FSDP2/ZE objects while slow ranks are still in
                    # train(), causing L0 use-after-free (SIGABRT in drm_neo.cpp:426).
                    # NOTE: destroy_process_group() doesn't help — nested ep_mesh/dp_mesh
                    # groups fail when the world group is destroyed first. The SIGABRT
                    # is a known L0 driver issue with simultaneous ZE context destruction
                    # by 12 processes; it's cosmetic (METRICS are always logged first).
                    torch.distributed.barrier()
                return

        self._profiler.stop()

    def log_metrics(
        self, trajectory: GRPOTrajectory, grpo_stats: GRPOStats, **extras
    ) -> None:
        # In dedicated_rank mode, rank 11 is already in the next generation
        # broadcast loop — it cannot participate in world-group collectives here.
        # Use training_pg (ranks 0-10) to avoid deadlock.
        if self._vllm_mode == "dedicated_rank" and getattr(self, '_training_pg', None) is not None:
            _reduce_pg = self._training_pg
            _n_reduce = self.world_size - 1
        else:
            _reduce_pg = None  # default world group
            _n_reduce = self.world_size

        rewards = trajectory.rewards.mean()
        # HSDP: skip world-level reduce to avoid mixing world PG with FSDP1
        # sub-PGs on XCCL. Each replicate group processes the same model with
        # different data, so rank 0's local metrics are representative.
        if self._shard_pg is None:
            torch.distributed.reduce(rewards, dst=0, op=torch.distributed.ReduceOp.SUM, group=_reduce_pg)
            rewards /= _n_reduce

        successes = trajectory.successes.mean()
        if self._shard_pg is None:
            torch.distributed.reduce(successes, dst=0, op=torch.distributed.ReduceOp.SUM, group=_reduce_pg)
            successes /= _n_reduce

        log_dict = {
            "rewards": rewards,
            "successes": successes,
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "loss": grpo_stats.loss.mean(),
            "policy_loss": grpo_stats.policy_loss.mean(),
            "kl_loss": grpo_stats.kl_loss.mean(),
            "clipfrac": grpo_stats.clipfrac.mean(),
            "ratios": grpo_stats.ratios.mean(),
            "approx_policy_kl": grpo_stats.approx_policy_kls.mean(),
            "response_lengths": trajectory.seq_lens.float().mean(),
            **extras,
        }

        if supports_memory_stats(self._device) and self._log_peak_memory_stats:
            try:
                log_dict.update(training.get_memory_stats(device=self._device))
            except RuntimeError:
                pass
        if self._is_rank_zero:
            self._metric_logger.log_dict(log_dict, step=self.global_step)
            # Phase 2 Step 3: producer telemetry (dedicated/server async modes)
            _prod = getattr(self, "_rollout_producer", None)
            _last = getattr(self, "_last_rollout_item", None)
            if _prod is not None and _last is not None:
                _w_now = self._weight_versions.version if hasattr(self, "_weight_versions") else 0
                _w_lag = max(0, _w_now - _last.weight_version)
                _qsize = _prod.qsize()
                _wait_ms = _prod.read_get_wait_ms()
                _idle_ms = _prod.read_blocked_on_put_ms()
                _async_tail = (
                    "  prod_qsize=%d  weight_lag=%d  prod_wait_ms=%.1f  prod_idle_ms=%.1f"
                    % (_qsize, _w_lag, _wait_ms, _idle_ms)
                )
            else:
                _async_tail = ""
            # Also log key metrics to stdout for verification
            log.info(
                "METRICS step=%d  loss=%.4f  policy_loss=%.4f  kl_loss=%.6f  "
                "rewards=%.3f  successes=%.3f  grad_norm=%.4f  "
                "clipfrac=%.4f  ratios=%.4f  approx_kl=%.6f  resp_len=%.1f"
                + _async_tail,
                self.global_step,
                log_dict["loss"].item() if hasattr(log_dict["loss"], "item") else log_dict["loss"],
                log_dict["policy_loss"].item() if hasattr(log_dict["policy_loss"], "item") else log_dict["policy_loss"],
                log_dict["kl_loss"].item() if hasattr(log_dict["kl_loss"], "item") else log_dict["kl_loss"],
                log_dict["rewards"].item() if hasattr(log_dict["rewards"], "item") else log_dict["rewards"],
                log_dict["successes"].item() if hasattr(log_dict["successes"], "item") else log_dict["successes"],
                log_dict.get("grad_norm", 0.0),
                log_dict["clipfrac"].item() if hasattr(log_dict["clipfrac"], "item") else log_dict["clipfrac"],
                log_dict["ratios"].item() if hasattr(log_dict["ratios"], "item") else log_dict["ratios"],
                log_dict["approx_policy_kl"].item() if hasattr(log_dict["approx_policy_kl"], "item") else log_dict["approx_policy_kl"],
                log_dict["response_lengths"].item() if hasattr(log_dict["response_lengths"], "item") else log_dict["response_lengths"],
            )

    def run_eval(self) -> None:
        """Run evaluation on held-out data. Generates via vLLM, computes rewards.

        All ranks participate (required for vLLM broadcast), but only rank 0 logs.
        """
        if not self._eval_enabled:
            return

        eval_t0 = time.perf_counter()
        if self._is_rank_zero:
            log.info("EVAL starting at step %d (%d examples)",
                     self._steps_run, len(self._eval_examples))

        all_rewards = []
        all_successes = []
        all_response_lengths = []
        eval_grpo_samples = self._eval_grpo_samples or self.grpo_samples

        collate_fn = _get_component_from_path("torchtune.dev.rl.data.padded_collate_rl")

        for eval_idx, example in enumerate(self._eval_examples):
            batch = collate_fn([example], padding_idx=self._tokenizer.pad_id)
            tokens = batch["tokens"].to(self._device)
            answers = batch["answers"]
            _, context_length = tokens.shape

            # Expand for grpo_samples
            batch_input_ids = tokens[:, None, :].expand(-1, eval_grpo_samples, -1)
            batch_input_ids = batch_input_ids.reshape(eval_grpo_samples, -1)

            with torch.no_grad():
                if self._vllm_mode == "server":
                    query_responses = self._generate_with_vllm(batch_input_ids, context_length)
                elif self._vllm_mode in ("colocate", "colocate_sleep"):
                    query_responses = self._generate_with_colocated_vllm(batch_input_ids, context_length)
                else:
                    # Native generation
                    with local_kv_cache(
                        model=self._model,
                        batch_size=eval_grpo_samples,
                        device=self._device,
                        dtype=self._dtype,
                        decoder_max_seq_len=context_length + self._max_generated_tokens,
                    ):
                        query_responses, _ = generate(
                            model=self._model,
                            prompt=batch_input_ids,
                            max_generated_tokens=self._max_generated_tokens,
                            temperature=self._temperature,
                            top_k=self._top_k,
                            pad_id=self._tokenizer.pad_id,
                            stop_tokens=self._tokenizer.stop_tokens if hasattr(self._tokenizer, 'stop_tokens') else None,
                            return_logits=False,
                        )

                responses = query_responses[:, context_length:].clone()

                # Truncate at stop tokens
                response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
                    responses, self._stop_token_ids, self._tokenizer.pad_id
                )

                # Compute rewards
                responses_reshaped = responses.reshape(1, eval_grpo_samples, -1)
                if self._reward_mode == "gene_recall":
                    rewards, successes, _ = gene_recall_batched_rewards(
                        self._tokenizer, responses_reshaped, answers, device=self._device,
                        reward_metric=self._gene_reward_metric,
                    )
                else:
                    rewards, successes, _ = batched_rewards(
                        self._tokenizer, responses_reshaped, answers, device=self._device
                    )
                # Sum across reward functions, mean across samples
                rewards = rewards.sum(dim=-1).mean().item()
                successes_mean = successes[:, :, -1].mean().item()

                all_rewards.append(rewards)
                all_successes.append(successes_mean)

                seq_lens = response_padding_masks.any(-1).sum()
                resp_len = (~response_padding_masks).sum(dim=-1).float().mean().item()
                all_response_lengths.append(resp_len)

            del tokens, batch_input_ids, query_responses, responses, response_padding_masks
            if self._device.type == "xpu":
                import gc
                gc.collect()

        eval_time = time.perf_counter() - eval_t0
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        avg_success = sum(all_successes) / len(all_successes) if all_successes else 0.0
        avg_resp_len = sum(all_response_lengths) / len(all_response_lengths) if all_response_lengths else 0.0

        eval_log_dict = {
            "eval/rewards": avg_reward,
            "eval/successes": avg_success,
            "eval/response_lengths": avg_resp_len,
            "eval/num_examples": len(self._eval_examples),
            "eval/time_seconds": eval_time,
        }

        if self._is_rank_zero:
            self._metric_logger.log_dict(eval_log_dict, step=self.global_step)
            log.info(
                "EVAL step=%d  rewards=%.3f  successes=%.3f  resp_len=%.1f  time=%.1fs  (%d examples)",
                self._steps_run, avg_reward, avg_success, avg_resp_len, eval_time,
                len(self._eval_examples),
            )

    def _log_batch_reward(self, rewards: torch.Tensor, successes: torch.Tensor) -> None:
        """Log batch-aggregated reward/success across all training ranks.

        rewards/successes are shape [B, G] on each rank. Aggregates over the
        full B*G*world batch and emits one line on rank 0:

            BATCH_REWARD step=N reward_mean=… reward_std=… reward_min=… reward_max=… success=…

        Single-line, parseable, gives the actual training signal (GRPO
        loss is mean-zero by construction so doesn't move; KL is bounded;
        reward is the only thing that should trend over a real run).
        """
        try:
            r = rewards.detach().float()
            s = successes.detach().float()
            ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if ws > 1:
                count = torch.tensor([float(r.numel())], device=r.device)
                stats = torch.stack([r.sum(), (r * r).sum(), s.sum(), r.min(), r.max()])
                torch.distributed.all_reduce(stats[:3], op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(stats[3:4], op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(stats[4:5], op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
                n = count.item()
                mean = (stats[0] / n).item()
                var = max((stats[1] / n).item() - mean * mean, 0.0)
                std = var ** 0.5
                succ = (stats[2] / n).item()
                rmin = stats[3].item()
                rmax = stats[4].item()
            else:
                mean = r.mean().item()
                std = r.std(unbiased=False).item() if r.numel() > 1 else 0.0
                succ = s.mean().item()
                rmin = r.min().item()
                rmax = r.max().item()
                n = float(r.numel())
            if self._is_rank_zero:
                log.info(
                    "BATCH_REWARD step=%d n=%d reward_mean=%.4f reward_std=%.4f "
                    "reward_min=%.4f reward_max=%.4f success=%.4f",
                    self._steps_run, int(n), mean, std, rmin, rmax, succ,
                )
        except Exception as e:
            log.warning("BATCH_REWARD log failed: %s", e)

    def cleanup(self) -> None:
        if self._vllm_client is not None and self._vllm_weight_sync and self._vllm_client.communicator is not None:
            self._vllm_client.close_communicator()
        # Abort the XCCL weight-sync PG on both sides before exiting. The PG is created
        # via c10d.ProcessGroupXCCL directly (not dist.new_group), so dist.destroy_process_group
        # doesn't work (not in registry). Use abort() which is unilateral — no collective needed.
        if getattr(self, "_xccl_wsync_pg", None) is not None:
            import threading as _threading

            def _close_xccl_replica(url):
                try:
                    requests.post(
                        f"{url}/collective_rpc",
                        json={"method": "close_xccl_communicator", "args": []},
                        timeout=30,
                    )
                except Exception:
                    pass

            if self._is_shard_leader:
                close_threads = [
                    _threading.Thread(target=_close_xccl_replica, args=(url,), daemon=True)
                    for url in self._vllm_urls
                ]
                for t in close_threads:
                    t.start()
                # Wait for vLLM abort() to complete BEFORE training abort().
                # Intel XCCL abort() appears to require both sides to call it
                # in overlapping time windows — calling training abort() first
                # (before vLLM has entered abort()) deadlocks.
                for t in close_threads:
                    t.join(timeout=30)

            # Abort training-side PG in a daemon thread with a hard timeout so
            # that even if Intel XCCL abort() hangs we don't block indefinitely.
            _pgs_to_abort = getattr(self, '_xccl_wsync_pgs', [self._xccl_wsync_pg])
            self._xccl_wsync_pg = None
            self._xccl_wsync_pgs = []
            self._xccl_bcast_buf = None
            _abort_done = _threading.Event()

            def _do_abort(pgs=_pgs_to_abort):
                for pg in pgs:
                    try:
                        pg.abort()
                    except Exception:
                        pass
                _abort_done.set()

            _threading.Thread(target=_do_abort, daemon=True).start()
            if not _abort_done.wait(timeout=15):
                log.warning("XCCL abort() timed out after 15s — forcing os._exit(0)")
                os._exit(0)
            log.info("XCCL wsync PG(s) aborted cleanly.")
        if self._vllm_llm is not None:
            del self._vllm_llm
            self._vllm_llm = None
        # Release persistent SHM block allocated by _sync_weights_to_vllm_shm()
        shm_block = getattr(self, "_shm_block", None)
        if shm_block is not None:
            try:
                shm_block.close()
                shm_block.unlink()
            except Exception:
                pass
            self._shm_block = None
        if self._is_rank_zero:
            self._metric_logger.close()
        if (getattr(self, "_vllm_weight_sync_method", None) == "xccl" or
                self._vllm_mode == "dedicated_rank" or
                self._vllm_mode == "server"):
            # XCCL teardown race: in dedicated_rank mode, rank 11 exits its server
            # loop and destroys the XCCL wsync_pg communicator while rank 0 is still
            # in save_checkpoint(). Rank 11's teardown corrupts the shared-memory IPC
            # handles that rank 0's subsequent destroy_process_group() tries to use,
            # causing SIGSEGV on rank 0. Fix: skip destroy_process_group() on ALL ranks
            # (including rank 11). The training_pg barrier inside save_checkpoint() is
            # safe even if rank 11 exits first (rank 11 is not a member of training_pg).
            # Checkpoint is already saved before cleanup() is called.
            os._exit(0)
        destroy_process_group()

    def cleanup_after_step(
        self,
        trajectory: GRPOTrajectory,
        l_grpo_stats: list[GRPOStats],
    ) -> None:
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats
        # Memory cleanup — gc only, skip empty_cache (leaks UR handles with FSDP)
        if self._device.type == "xpu":
            import gc
            gc.collect()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """

    recipe = GRPOFullFinetuneDistributedXPU(cfg=cfg)
    config.log_config(recipe_name="GRPOFullFinetuneDistributedXPU", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

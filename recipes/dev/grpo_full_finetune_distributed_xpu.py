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

import os
import sys
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
from torchtune.dev.rl.rewards import batched_rewards
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    device_empty_cache,
    device_record_memory_history,
    disable_dropout,
    DummyProfiler,
    get_xpu_distributed_backend,
    init_xpu_process_group,
    PROFILER_KEY,
    supports_memory_stats,
)
from torchtune.training.lr_schedulers import get_lr
from tqdm import tqdm

log = utils.get_logger("DEBUG")

# -- FSDP2 XPU fix: ReduceOp.AVG not supported by XCCL -----------------------
# FSDP2's reduce_scatter uses ReduceOp.AVG which XCCL doesn't support.
# Monkey-patch _get_gradient_divide_factors to force SUM reduction on XPU
# (same approach as MTIA in upstream PyTorch).
try:
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll
    _orig_get_gradient_divide_factors = _fsdp_coll._get_gradient_divide_factors

    def _xpu_get_gradient_divide_factors(*args, **kwargs):
        # args: (reduce_scatter_group, all_reduce_group, reduce_dtype,
        #        device_type, gradient_divide_factor, force_sum_reduction_for_comms)
        # Force SUM reduction for XPU (XCCL doesn't support AVG)
        if len(args) >= 4 and args[3] == "xpu":
            args = list(args)
            if len(args) >= 6:
                args[5] = True  # force_sum_reduction_for_comms
            else:
                kwargs["force_sum_reduction_for_comms"] = True
            args = tuple(args)
        elif kwargs.get("device_type") == "xpu":
            kwargs["force_sum_reduction_for_comms"] = True
        return _orig_get_gradient_divide_factors(*args, **kwargs)

    _fsdp_coll._get_gradient_divide_factors = _xpu_get_gradient_divide_factors
    log.info("Patched FSDP2 _get_gradient_divide_factors for XPU (force SUM reduction)")
except Exception as e:
    log.warning("Failed to patch FSDP2 for XPU: %s", e)

# Original device_empty_cache imported above. When colocated vLLM engines
# are present, torch.xpu.empty_cache() can deadlock if called while another
# rank is inside an FSDP collective. We replace it with synchronize() which
# forces completion of pending XPU ops without risking the cache-clearing
# deadlock, and still frees some memory via the allocator.
_orig_device_empty_cache = device_empty_cache
_colocate_vllm_mode = False

def device_empty_cache(device: torch.device) -> None:
    if device.type == "xpu":
        # NEVER call empty_cache() on XPU with FSDP. The combination of
        # empty_cache() + FSDP storage.resize_() leaks UR handles in Level
        # Zero, causing UR_RESULT_ERROR_OUT_OF_RESOURCES after ~70 iters.
        # The caching allocator reuses blocks without touching L0 if we
        # skip empty_cache(). See docs/intel_xpu_resource_leak_bug_report.md
        pass
    else:
        _orig_device_empty_cache(device)


def _safe_empty_cache(device: torch.device) -> None:
    """Empty cache at a synchronized point (after a barrier).

    On XPU, this is a no-op — empty_cache() + FSDP leaks UR handles.
    """
    torch.distributed.barrier()
    if device.type == "xpu":
        torch.xpu.synchronize()
        return
    _orig_device_empty_cache(device)


def _slice_trajectory(
    trajectory: GRPOTrajectory, start: int, end: int
) -> GRPOTrajectory:
    """Slice a GRPOTrajectory along the batch dimension.

    Args:
        trajectory: The full trajectory.
        start: Start index (inclusive).
        end: End index (exclusive).

    Returns:
        A new GRPOTrajectory with only the [start:end] samples.
    """
    fields = {}
    for field_name in trajectory._fields:
        val = getattr(trajectory, field_name)
        if isinstance(val, torch.Tensor):
            fields[field_name] = val[start:end]
        elif isinstance(val, list):
            fields[field_name] = val[start:end]
        else:
            fields[field_name] = val
    return GRPOTrajectory(**fields)



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

        # SDPA backend selection for XPU.
        # Default: force math-only SDPA (disable flash/mem_efficient) as a
        # precaution against UR handle leaks observed with broken CCL config.
        # Set force_math_sdpa=False to re-enable optimized SDPA backends.
        if self._device.type == "xpu" and cfg.get("force_math_sdpa", True):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            log.info("Rank %d: forced math-only SDPA (set force_math_sdpa=False to test optimized backends)", self.rank)
        elif self._device.type == "xpu":
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

        # Gradient accumulation
        self._gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

        # vLLM generation (optional — None means use native generation)
        # mode: "server" = external vLLM HTTP server on separate tile(s)
        #        "colocate" = in-process vLLM engine per rank (TRL-style)
        #        "colocate_sleep" = colocate with sleep/wake memory management
        self._vllm_mode = cfg.get("vllm_mode", None)  # None, "server", "colocate", or "colocate_sleep"
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

    def _init_vllm_early(self, cfg):
        """Initialize colocated vLLM engine(s) before the training PG.

        Two modes based on ``vllm_tensor_parallel_size``:

        **TP=1** (default): Each rank creates its own isolated vLLM engine
        using a gloo PG (world_size=1). Sequential init via file barriers.

        **TP>1**: Ranks are grouped into DP groups of size tp_size. Each group
        creates one shared vLLM engine via ``external_launcher`` with an XCCL
        PG of size tp_size. All ranks in a group call generate() together.
        """
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = _xpu_device_index  # The actual XPU tile for this rank

        vllm_mode = cfg.get("vllm_mode", "colocate")
        tp_size = cfg.get("vllm_tensor_parallel_size", 1)

        # For colocate_sleep mode, apply XPU sleep patches BEFORE importing vLLM LLM
        if vllm_mode == "colocate_sleep":
            from torchtune.dev.xpu_sleep import patch_vllm_for_xpu_sleep
            patch_vllm_for_xpu_sleep()

        from vllm import LLM

        model_path = cfg.base_model_path
        gpu_mem = cfg.get("vllm_gpu_memory_utilization", 0.3)
        max_model_len = cfg.get("vllm_max_model_len", 2048)

        log.info(
            "Rank %d: initializing colocated vLLM engine on xpu:%d (tp=%d, gpu_mem=%.2f, max_model_len=%d)",
            rank, local_rank, tp_size, gpu_mem, max_model_len,
        )

        # vLLM V1: disable multiprocessing to avoid EngineCore subprocess hangs.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Disable torch.compile for vLLM init (not viable on XPU for vLLM).
        _prev_compile_disable = os.environ.get("TORCH_COMPILE_DISABLE")
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

        # Each rank generates all grpo_samples for its own prompts (no split)
        max_num_seqs = max(cfg.batch_size * cfg.grpo_samples, 1)

        if tp_size > 1:
            self._init_vllm_tp(
                cfg, rank, world_size, local_rank, tp_size,
                model_path, gpu_mem, max_model_len, max_num_seqs, vllm_mode, LLM,
            )
        else:
            self._init_vllm_tp1(
                cfg, rank, world_size, local_rank,
                model_path, gpu_mem, max_model_len, max_num_seqs, vllm_mode, LLM,
            )

        # Restore TORCH_COMPILE_DISABLE so training model can use torch.compile
        if _prev_compile_disable is not None:
            os.environ["TORCH_COMPILE_DISABLE"] = _prev_compile_disable
        elif "TORCH_COMPILE_DISABLE" in os.environ:
            del os.environ["TORCH_COMPILE_DISABLE"]

        # Re-set the XPU device to our tile (vLLM may have called set_device)
        torch.xpu.set_device(local_rank)

        log.info("Rank %d: colocated vLLM engine initialized on xpu:%d, PG reset for training", rank, local_rank)

    def _init_vllm_tp1(self, cfg, rank, world_size, local_rank,
                        model_path, gpu_mem, max_model_len, max_num_seqs,
                        vllm_mode, LLM):
        """Initialize TP=1 colocated vLLM: one isolated engine per rank.

        Uses gloo PG (world_size=1) with sequential init via file barriers.
        """
        # Sequential init to avoid port/resource conflicts.
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
        barrier_dir = f"/tmp/torchtune/vllm_init_barriers_{run_id}"
        os.makedirs(barrier_dir, exist_ok=True)
        if rank > 0:
            prev_barrier = os.path.join(barrier_dir, f"rank_{rank - 1}_done")
            log.info("Rank %d: waiting for rank %d to finish vLLM init...", rank, rank - 1)
            while not os.path.exists(prev_barrier):
                time.sleep(0.5)

        # Override torchrun env vars so vLLM sees world_size=1
        saved_env = {}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                    "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID"):
            saved_env[key] = os.environ.pop(key, None)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29599 + rank)

        # Pre-init a gloo PG (world_size=1) so vLLM skips its own init.
        import tempfile
        _store_file = tempfile.mktemp(prefix=f"vllm_gloo_store_r{rank}_")
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{_store_file}",
            world_size=1,
            rank=0,
        )

        # Monkey-patch new_group to force gloo backend.
        _orig_new_group = torch.distributed.new_group
        def _gloo_new_group(*args, **kwargs):
            kwargs["backend"] = "gloo"
            return _orig_new_group(*args, **kwargs)
        torch.distributed.new_group = _gloo_new_group

        # Monkey-patch all_reduce to skip XPU tensor ops on gloo.
        _orig_all_reduce = torch.distributed.all_reduce
        def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM,
                             group=None, async_op=False):
            if group is not None and group.size() == 1:
                return None
            if tensor.is_xpu:
                return None
            return _orig_all_reduce(tensor, op=op, group=group,
                                    async_op=async_op)
        torch.distributed.all_reduce = _safe_all_reduce

        # Monkey-patch _distributed_args for correct local_rank.
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        _orig_distributed_args = UniProcExecutor._distributed_args
        _correct_local_rank = local_rank
        def _patched_distributed_args(self_exec):
            method, _rank, _lr = _orig_distributed_args(self_exec)
            return method, _rank, _correct_local_rank
        UniProcExecutor._distributed_args = _patched_distributed_args

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            dtype="bfloat16",
            disable_custom_all_reduce=True,
        )
        if vllm_mode == "colocate_sleep":
            llm_kwargs["enable_sleep_mode"] = True

        self._vllm_llm = LLM(**llm_kwargs)

        # Restore monkey-patches
        UniProcExecutor._distributed_args = _orig_distributed_args
        torch.distributed.new_group = _orig_new_group
        torch.distributed.all_reduce = _orig_all_reduce

        # Destroy vLLM's gloo PG so we can init the training XCCL PG.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._WORLD = None

        try:
            os.unlink(_store_file)
        except OSError:
            pass

        # Restore torchrun env vars for the training PG
        for key, val in saved_env.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]

        # Signal next rank that we're done
        my_barrier = os.path.join(barrier_dir, f"rank_{rank}_done")
        with open(my_barrier, "w") as f:
            f.write("done")

        # Last rank cleans up barrier files
        if rank == world_size - 1:
            time.sleep(0.5)
            for i in range(world_size):
                try:
                    os.unlink(os.path.join(barrier_dir, f"rank_{i}_done"))
                except OSError:
                    pass

    def _init_vllm_tp(self, cfg, rank, world_size, local_rank, tp_size,
                       model_path, gpu_mem, max_model_len, max_num_seqs,
                       vllm_mode, LLM):
        """Initialize TP>1 colocated vLLM: one engine per DP group via external_launcher.

        Groups ranks into DP groups of tp_size. Each group creates an XCCL PG
        of size tp_size and a shared vLLM engine with external_launcher.
        All ranks in a group must call generate() with identical prompts.

        Pattern:
        1. File-based barrier (all ranks in DP group signal ready)
        2. Override env: WORLD_SIZE, RANK, LOCAL_RANK, MASTER_PORT; disable elastic agent store
        3. Create TP-sized XCCL PG per DP group (TCP rendezvous, unique port)
        4. Create vLLM LLM with external_launcher
        5. Destroy TP PG, restore env (training PG created later via elastic agent store)
        """
        assert world_size % tp_size == 0, (
            f"world_size={world_size} must be divisible by vllm_tensor_parallel_size={tp_size}"
        )
        dp_size = world_size // tp_size
        dp_rank = rank // tp_size
        tp_rank = rank % tp_size

        # Store TP/DP info for later use
        self._vllm_dp_rank = dp_rank
        self._vllm_tp_rank = tp_rank
        self._vllm_dp_size = dp_size

        log.info(
            "Rank %d: TP init — dp_rank=%d, tp_rank=%d, dp_size=%d, tp_size=%d",
            rank, dp_rank, tp_rank, dp_size, tp_size,
        )

        # Step 1: File-based barrier — all ranks signal ready before any creates PG.
        # torchrun sets TORCHELASTIC_USE_AGENT_STORE=True; consuming it for a
        # throwaway global PG would prevent reuse for the training PG later.
        _run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
        _barrier_dir = f"/tmp/torchtune/vllm_tp_barrier_{_run_id}"
        os.makedirs(_barrier_dir, exist_ok=True)
        # Signal this rank is ready
        with open(os.path.join(_barrier_dir, f"rank_{rank}"), "w") as f:
            f.write("ready")
        # Wait for all ranks in our DP group to be ready
        dp_group_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        for r in dp_group_ranks:
            while not os.path.exists(os.path.join(_barrier_dir, f"rank_{r}")):
                time.sleep(0.1)
        log.info("Rank %d: all %d ranks in DP group %d ready", rank, tp_size, dp_rank)

        # Step 2: Override env for TP subgroup
        saved_env = {}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_PORT",
                    "TORCHELASTIC_USE_AGENT_STORE"):
            saved_env[key] = os.environ.get(key)

        original_port = os.environ.get("MASTER_PORT", "29500")
        os.environ["WORLD_SIZE"] = str(tp_size)
        os.environ["RANK"] = str(tp_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        # Unique port per DP group to avoid collisions
        os.environ["MASTER_PORT"] = str(int(original_port) + dp_rank + 1)
        # Disable elastic agent store — use TCP rendezvous for TP subgroup
        os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)

        # Step 3: Create XCCL PG for this TP subgroup
        torch.distributed.init_process_group(
            backend="xccl",
            world_size=tp_size,
            rank=tp_rank,
        )
        log.info("Rank %d: TP subgroup PG initialized (dp_group=%d, size=%d)",
                 rank, dp_rank, torch.distributed.get_world_size())

        # Step 5: Create vLLM LLM engine
        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp_size,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            dtype="bfloat16",
        )
        if vllm_mode == "colocate_sleep":
            llm_kwargs["enable_sleep_mode"] = True

        self._vllm_llm = LLM(**llm_kwargs)

        # Step 6: Destroy TP PG — training will create its own global XCCL PG
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._WORLD = None

        # Restore env vars
        for key, val in saved_env.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]

        # Clean up barrier files (best-effort)
        if rank == 0:
            import shutil
            try:
                shutil.rmtree(_barrier_dir, ignore_errors=True)
            except OSError:
                pass

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
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_sd=checkpoint_dict[training.MODEL_KEY],
            reshard_after_forward=reshard_policy,
        )
        # Setup reference model
        ref_checkpoint_dict = self.load_checkpoint(
            cfg_checkpointer=cfg.ref_checkpointer
        )
        self._ref_model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_sd=ref_checkpoint_dict[training.MODEL_KEY],
            eval_mode=True,
            reshard_after_forward=True,
        )
        # Skip barrier on multi-node (XCCL sub-communicator creation deadlocks
        # on 2D mesh). FSDP will implicitly sync during first forward pass.
        if not self._production_mode:
            torch.distributed.barrier()

        # Utilize the same tokenizer for both models (hack)
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

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
            "collate_fn", "torchtune.dev.grpo.data.padded_collate_rl"
        )
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            dataloader_state_dict=(
                checkpoint_dict[training.DATALOADER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

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

    def _setup_vllm_server_mode(self):
        """Initialize vLLM in server mode.

        Only global rank 0 creates HTTP clients and calls vLLM for generation.
        Results are broadcast to all ranks via the world process group.

        NOTE: XCCL on Aurora deadlocks when using shard-level sub-communicators,
        so per-node vLLM generation with shard-level broadcast is not possible.
        Only rank 0's vLLM server is used; other nodes' vLLM servers are idle.
        """
        should_init_client = self._is_rank_zero

        if should_init_client:
            from torchtune.dev.grpo.vllm_client import VLLMClient

            # Create clients for all vLLM URLs (supports DP vLLM replicas)
            self._vllm_clients = []
            for url in self._vllm_urls:
                client = VLLMClient(
                    base_url=url,
                    group_port=self._vllm_group_port,
                    connection_timeout=300.0,
                )
                self._vllm_clients.append(client)
            self._vllm_client = self._vllm_clients[0]  # backward compat

            if self._vllm_weight_sync:
                # On XPU, creating a second ProcessGroupXCCL (for weight sync)
                # alongside the training XCCL PG causes SIGABRT. Use file-based
                # weight sync instead: save to /tmp, POST to vLLM to reload.
                self._build_tune_to_hf_map()
                self._weight_sync_path = "/tmp/torchtune/weight_update.safetensors"
                log.info(
                    "Rank %d: vLLM %d client(s) initialized: %s (%d params mapped, file-based sync via %s, interval=%d)",
                    self.rank,
                    len(self._vllm_clients),
                    ",".join(self._vllm_urls),
                    len(self._tune_to_hf_map),
                    self._weight_sync_path,
                    self._vllm_weight_sync_interval,
                )
            else:
                log.info(
                    "Rank %d: vLLM %d client(s) initialized (generation only, no weight sync): %s",
                    self.rank, len(self._vllm_clients), ",".join(self._vllm_urls),
                )

        # NOTE: Do NOT use shard_pg for any explicit collectives on Aurora.
        # XCCL deadlocks when creating sub-communicators from device_mesh groups.
        # FSDP2 internally manages its own communicators and works fine, but
        # explicit operations on dp_mesh.get_group("dp_shard") deadlock.
        if not self._production_mode:
            torch.distributed.barrier()

    def _setup_vllm_colocate_mode(self, cfg):
        """Initialize vLLM in colocate mode: every rank has its own vLLM engine.

        All ranks have vLLM engines (created in _init_vllm_early). Each rank
        generates all completions for its own prompts locally (no cross-rank
        communication during generation). Weight sync loads gathered FSDP2
        params into each rank's local engine.

        Called from setup() AFTER model init.
        """
        # Build param name mapping for weight sync
        self._build_tune_to_hf_map()

        log.info(
            "Rank %d: colocated vLLM engine ready (%d params mapped, local generation)",
            self.rank, len(self._tune_to_hf_map),
        )
        torch.distributed.barrier()

    def _build_tune_to_hf_map(self):
        """Build torchtune -> HuggingFace parameter name mapping for weight sync.

        Uses named_parameters() instead of state_dict() to avoid triggering
        FSDP1's AllGather collective (which would deadlock if only shard leaders
        call this function).
        """
        from torchtune.models.convert_weights import get_mapped_key
        from torchtune.models.qwen2._convert_weights import _FROM_HF

        inverted = {v: k for k, v in _FROM_HF.items() if v is not None}
        self._tune_to_hf_map = {}
        for tune_name, _ in self._model.named_parameters():
            # Strip FSDP and activation checkpoint wrapper prefixes for mapping
            clean_name = tune_name.replace("_fsdp_wrapped_module.", "")
            clean_name = clean_name.replace("_checkpoint_wrapped_module.", "")
            self._tune_to_hf_map[tune_name] = get_mapped_key(
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
        if self._dp_replicate > 1 and model_gib < 50.0:
            self._use_fsdp1 = True
            return self._setup_model_fsdp1_hsdp(
                cfg_model=cfg_model,
                enable_activation_checkpointing=enable_activation_checkpointing,
                model_sd=model_sd,
                eval_mode=eval_mode,
                reshard_after_forward=reshard_after_forward,
            )
        elif self._dp_replicate > 1:
            utils.log_rank_zero(
                log,
                f"Model too large for FSDP1 HSDP ({model_gib:.1f} GiB). "
                f"Using FSDP2 with HSDP mesh (dp_replicate={self._dp_replicate} × dp_shard={self._dp_shard}).",
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

        if self._compile:
            # Temporarily allow torch.compile (TORCH_COMPILE_DISABLE=1 set for vLLM)
            _saved_tcd = os.environ.pop("TORCH_COMPILE_DISABLE", None)
            training.compile_model(
                model, verbose=self._is_rank_zero, dynamic=self._compile_dynamic
            )
            if _saved_tcd is not None:
                os.environ["TORCH_COMPILE_DISABLE"] = _saved_tcd

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]

        # Policy doesn't reshard after forward for faster generation.
        # Reference net reshards after forward because it never calls .backward()
        # Use HSDP mesh for all models. FSDP2 with 2D DeviceMesh reuses cached
        # ProcessGroups from init_device_mesh (no new XCCL communicators per-module).
        # DeviceMesh uses new_group() (not split()) on XPU since torch.cuda.is_available()=False.
        fsdp2_mesh = self._dp_mesh
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
            dp_mesh=fsdp2_mesh,
            disable_prefetch=self._disable_prefetch,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_sd,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

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
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

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
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
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
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
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

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        # Skip barrier in production mode — checkpoint save is rank-0 only,
        # other ranks just need to stay out of the way during full_tensor() gather.
        if not self._production_mode:
            torch.distributed.barrier()

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
                # DP vLLM: split sequences round-robin across replicas, call in parallel
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

                # Interleave results back to original prompt order
                completions = [None] * bsz
                for i in range(bsz):
                    client_idx = i % num_clients
                    within_idx = i // num_clients
                    completions[i] = chunk_results[client_idx][within_idx]
            else:
                completions = self._vllm_client.generate(prompts=prompts, **gen_kwargs)
            gen_time = time.perf_counter() - t0

            # Build query_responses tensor: [prompt | completion | padding]
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
        else:
            query_responses = batch_input_ids.new_empty(bsz, total_len)

        # Broadcast from rank 0 to all ranks via world PG.
        # Broadcast from rank 0 to all ranks via world PG.
        # XCCL on Aurora deadlocks when creating sub-communicators for shard PG,
        # so we always use the world PG for explicit collectives.
        torch.distributed.broadcast(query_responses, src=0)
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

        # Strip padding and convert to Python lists
        prompts = []
        for i in range(bsz):
            ids = batch_input_ids[i].cpu().tolist()
            ids = [t for t in ids if t != self._tokenizer.pad_id]
            prompts.append(ids)

        t0 = time.perf_counter()
        outputs = self._vllm_llm.generate(
            prompts=[{"prompt_token_ids": p} for p in prompts],
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
        docs/intel_xpu_resource_leak_bug_report.md). This method is now a
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
        llm_model = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model

        n_synced = 0
        for tune_name, param in self._model.named_parameters():
            clean = tune_name.replace("_checkpoint_wrapped_module.", "")
            hf_name = self._tune_to_hf_map.get(
                clean, self._tune_to_hf_map.get(tune_name, clean)
            )

            if hasattr(param, 'full_tensor'):
                weight_data = param.full_tensor()
            else:
                weight_data = param.data

            llm_model.load_weights([(hf_name, weight_data)])
            n_synced += 1
            del weight_data

            # Sync + gc every 5 params to bound UR handle pressure
            # (707 full_tensor all-gathers create UR handles that must be
            # reclaimed before the subsequent FSDP backward pass).
            if n_synced % 5 == 0 and torch.xpu.is_available():
                gc.collect()
                torch.xpu.synchronize(self._device)

        self._vllm_llm.llm_engine.reset_prefix_cache()

        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.synchronize(self._device)

        log.info(
            "Rank %d: weight sync: %d params in %.1fs",
            self.rank, n_synced, time.perf_counter() - t0,
        )

    def _sync_weights_to_vllm(self) -> None:
        """Gather sharded params and push to vLLM server.

        Uses file-based sync: saves weights as safetensors to /tmp, then
        POSTs to vLLM's /load_weights_from_path/ endpoint. This avoids
        creating a second XCCL process group which SIGABRTs on XPU.

        FSDP2: all ranks participate in ``full_tensor()``, shard leader saves.
        FSDP1 (HSDP): uses FSDP.state_dict() which gathers on all ranks in
        the shard group, shard leader saves.

        For HSDP: each shard leader syncs to its local vLLM independently.
        """
        t0 = time.perf_counter()
        hf_state_dict = {}

        if getattr(self, '_use_fsdp1', False) and self._dp_replicate > 1:
            # FSDP1 path: state_dict() handles gathering within shard group.
            # All ranks in the shard group participate.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
            with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
                full_sd = self._model.state_dict()
            if self._is_shard_leader:
                for param_name, param in full_sd.items():
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.cpu()
            del full_sd
        else:
            # FSDP2 path: gather DTensor -> full tensor.
            # With HSDP 2D mesh, full_tensor() all-gathers within shard group only
            # (dp_shard dim). Each shard leader gets a full copy independently.
            sharded_sd = self._model.state_dict()
            for param_name, param in sharded_sd.items():
                if param.is_cpu:
                    param = param.to(self._device)
                if hasattr(param, "_local_tensor"):
                    param = param.full_tensor()
                if self._is_shard_leader:
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.cpu()
            del sharded_sd

        # Barrier: ensure all ranks finished full_tensor() before saving
        if not self._production_mode:
            torch.distributed.barrier()

        if self._is_shard_leader:
            from safetensors.torch import save_file

            n_params = len(hf_state_dict)
            save_path = self._weight_sync_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_file(hf_state_dict, save_path)
            del hf_state_dict

            # Tell vLLM to reload from file
            import requests
            r = requests.post(
                f"{self._vllm_url}/load_weights_from_path/",
                json={"path": save_path},
                timeout=120,
            )
            if r.status_code != 200:
                log.warning("vLLM weight reload failed: %s %s", r.status_code, r.text)
            else:
                result = r.json()
                if result.get("status") != "ok":
                    log.warning("vLLM weight reload error: %s", result)

            self._vllm_client.reset_prefix_cache()
            log.info(
                "Rank %d: weight sync to vLLM: %d params in %.1fs (file-based)",
                self.rank, n_params, time.perf_counter() - t0,
            )

        device_empty_cache(self._device)

    def generate_trajectory(
        self, input_ids: torch.Tensor, answers: list[str]
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
            query_responses = self._generate_with_colocated_vllm(batch_input_ids, context_length)
        elif self._vllm_mode == "server":
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
        # For vLLM server mode, the world broadcast in _generate_with_vllm
        # already synchronizes all ranks. For other modes, use world barrier.
        if self._vllm_mode != "server" and not self._production_mode:
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
        # Chunk the forward pass to avoid OOM with many sequences (Config B).
        # Process forward_batch_size sequences at a time through the model.
        _policy_fwd_t0 = time.perf_counter()
        num_seqs = query_responses.shape[0]
        fwd_bs = self._forward_batch_size
        if fwd_bs >= num_seqs:
            # Single forward pass (no chunking needed)
            log.info("Rank %d: policy forward start (shape=%s)", self.rank, list(query_responses.shape))
            logits = self._model(query_responses, input_pos=position_ids, mask=masks)
            log.info("Rank %d: policy forward done", self.rank)
            logits = logits[:, context_length - 1 :]
            logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
            del logits
        else:
            # Chunked forward pass
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
                # NOTE: empty_cache() between forward chunks causes GPU segfaults
                # on XPU — FSDP2 internal handles reference the freed blocks.
                # Instead, use fbs >= grpo_samples to avoid chunking entirely.
                device_empty_cache(self._device)
            logprobs = torch.cat(logprobs_chunks, dim=0)
            del logprobs_chunks
            log.info("Rank %d: policy forward done (chunked)", self.rank)
        device_empty_cache(self._device)

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _policy_fwd_time = time.perf_counter() - _policy_fwd_t0

        # NOTE: Removed policy→ref defrag. With fbs >= grpo_samples (no chunking),
        # post-policy reserved ~29 GiB leaves enough headroom for ref forward.
        # Each empty_cache() accelerates UR handle leak, so minimize calls.

        # step 2.1 estimate logprobs of the responses using the reference policy
        # Barrier: sync before ref model forward. FSDP2 uses lazy allgather
        # which is itself a synchronization point — skip world barrier for
        # multi-node to avoid CCL broadcast_scaleout failures.
        _ref_fwd_t0 = time.perf_counter()
        log.info("Rank %d: pre-ref forward", self.rank)
        # FSDP AllGather in ref forward provides implicit sync
        if not self._production_mode:
            torch.distributed.barrier()
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
        if self._device.type == "xpu":
            torch.xpu.synchronize()
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

        advantages = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.reshape(batch_size * grpo_size)
        del responses
        device_empty_cache(self._device)

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

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
        self, input_ids: torch.Tensor, answers: list[str]
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

        # Concatenate all trajectory fields except answers (which is a list of strings)
        concatenated_fields = {}
        for field_name in trajectories[0]._fields:
            if field_name == "answers":
                concatenated_fields[field_name] = []
                for traj in trajectories:
                    concatenated_fields[field_name].extend(traj.answers)
            else:
                concatenated_fields[field_name] = torch.cat(
                    [getattr(traj, field_name) for traj in trajectories]
                )

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
                torch.xpu.reset_peak_memory_stats()

        # estimate logprobs from the policy at the current optimisation step
        _fwd_t0 = time.perf_counter()

        if self._enable_packing:
            # Pack sequences to eliminate padding waste in forward/backward
            from torchtune.dev.grpo.packing import (
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
        else:
            log.info("Rank %d: grpo_step policy forward start", self.rank)
            pi_logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
            )

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _fwd_time = time.perf_counter() - _fwd_t0
        log.info("Rank %d: grpo_step forward=%.1fs", self.rank, _fwd_time)

        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)

        if self._use_chunked_loss:
            # Chunked loss: pass raw logits directly to loss function.
            # The loss computes logprobs internally per chunk, avoiding
            # materializing the full [B*G, seq, vocab] fp32 logit tensor.
            targets = trajectory.query_responses[:, context_length:]
            num_chunks = self._loss_fn.num_output_chunks
            pi_logit_chunks = list(pi_logits.chunk(num_chunks, dim=1))
            del pi_logits
            if self._device.type == "xpu":
                torch.xpu.synchronize()

            loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs = (
                self._loss_fn(
                    pi_logit_chunks,
                    targets,
                    trajectory.ref_logprobs,
                    trajectory.advantages,
                    padding_masks=~trajectory.response_padding_masks,
                )
            )
            pi_logprobs[trajectory.response_padding_masks] = 1.0
        else:
            # Standard loss: convert logits to logprobs first
            pi_logprobs = rlhf.batched_logits_to_logprobs(
                pi_logits,
                trajectory.query_responses[:, context_length:],
                self._temperature,
                chunk_size=1,
            )
            pi_logprobs[trajectory.response_padding_masks] = 1.0

            del pi_logits
            if self._device.type == "xpu":
                torch.xpu.synchronize()

            loss, policy_loss, kl_loss, ratios, clipfrac = self._loss_fn(
                trajectory.logprobs,
                pi_logprobs,
                trajectory.ref_logprobs,
                trajectory.advantages,
                padding_masks=~trajectory.response_padding_masks,
            )

        # Scale loss for gradient accumulation
        if self._gradient_accumulation_steps > 1:
            loss = loss / self._gradient_accumulation_steps

        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "post_forward", log=log)
            # Reset peak to measure backward peak separately
            if self._device.type == "xpu":
                torch.xpu.reset_peak_memory_stats()

        # NOTE: Pre-backward empty_cache() removed. Each empty_cache() call on
        # XPU leaks UR handles in Level Zero, causing GPU segfaults after ~4 calls.
        # Instead, rely on between-step empty_cache (1 call/step) + aggressive
        # GC threshold (0.4) to keep fragmentation manageable.
        if self._fsdp_diagnostics and self._is_rank_zero and self._device.type == "xpu":
            log.info(
                "[PRE_BACKWARD] alloc=%.2f GiB, resv=%.2f GiB (no defrag)",
                torch.xpu.memory_allocated(self._device) / (1024**3),
                torch.xpu.memory_reserved(self._device) / (1024**3),
            )

        _bwd_t0 = time.perf_counter()
        log.info("Rank %d: backward start", self.rank)
        loss.backward()
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _bwd_time = time.perf_counter() - _bwd_t0
        log.info("Rank %d: backward=%.1fs", self.rank, _bwd_time)

        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "post_backward", log=log)

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return GRPOStats(
            loss * self._gradient_accumulation_steps,  # Report unscaled loss
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
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        grad_norm = None

        training_completed = False
        self._profiler.start()
        for curr_epoch in range(self._epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
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

                _step_t0 = time.perf_counter()
                trajectory = self.generate_trajectory_batched(tokens, answers)
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _gen_time = time.perf_counter() - _step_t0
                if not self._production_mode:
                    torch.distributed.barrier()

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
                        # No gradient accumulation — single step
                        step_stats = self.grpo_step(trajectory, context_length)
                        grpo_stats.append(step_stats)

                    # Sync device before timing grad clip
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _grpo_time = time.perf_counter() - _grpo_t0

                    _clip_t0 = time.perf_counter()
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _clip_time = time.perf_counter() - _clip_t0

                    if not self._production_mode:
                        torch.distributed.barrier()

                    # Synchronize before optimizer — no empty_cache in
                    # colocate mode (it hangs with vLLM engines).
                    if _colocate_vllm_mode and self._device.type == "xpu":
                        log.info("Rank %d: pre-optimizer sync", self.rank)
                        torch.xpu.synchronize()

                    _opt_t0 = time.perf_counter()
                    log.info("Rank %d: optimizer.step()", self.rank)
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
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

                    if not self._production_mode:
                        torch.distributed.barrier()

                    self.global_step += 1

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                # Sync updated weights to vLLM (after all ppo_epochs)
                # For colocate_sleep, sync happens during wake_up in generate_trajectory
                if self._vllm_mode == "colocate":
                    self._sync_colocated_weights()
                elif self._vllm_mode == "server" and self._vllm_weight_sync:
                    if self._steps_run % self._vllm_weight_sync_interval == 0:
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

                # NOTE: No empty_cache() between steps. With PYTORCH_ALLOC_CONF=
                # expandable_segments:True, the allocator avoids fragmentation by
                # expanding existing segments in-place. This avoids the UR handle
                # leak that empty_cache() causes with FSDP on XPU.
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
                if (self._save_every_n_steps is not None and self._steps_run > 0 and
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
                return

        self._profiler.stop()

    def log_metrics(
        self, trajectory: GRPOTrajectory, grpo_stats: GRPOStats, **extras
    ) -> None:
        rewards = trajectory.rewards.mean()
        # HSDP: skip world-level reduce to avoid mixing world PG with FSDP1
        # sub-PGs on XCCL. Each replicate group processes the same model with
        # different data, so rank 0's local metrics are representative.
        if self._shard_pg is None:
            torch.distributed.reduce(rewards, dst=0, op=torch.distributed.ReduceOp.SUM)
            rewards /= self.world_size

        successes = trajectory.successes.mean()
        if self._shard_pg is None:
            torch.distributed.reduce(successes, dst=0, op=torch.distributed.ReduceOp.SUM)
            successes /= self.world_size

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
            log_dict.update(training.get_memory_stats(device=self._device))
        if self._is_rank_zero:
            self._metric_logger.log_dict(log_dict, step=self.global_step)
            # Also log key metrics to stdout for verification
            log.info(
                "METRICS step=%d  loss=%.4f  policy_loss=%.4f  kl_loss=%.6f  "
                "rewards=%.3f  successes=%.3f  grad_norm=%.4f  "
                "clipfrac=%.4f  ratios=%.4f  approx_kl=%.6f  resp_len=%.1f",
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

        collate_fn = _get_component_from_path("torchtune.dev.grpo.data.padded_collate_rl")

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
                rewards, successes, _ = batched_rewards(
                    self._tokenizer, responses_reshaped, answers, device=self._device
                )
                # Sum across reward functions, mean across samples
                rewards = rewards.sum(dim=-1).mean().item()
                successes_mean = successes[:, :, -1].mean().item()  # math_response_correct is last

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

    def cleanup(self) -> None:
        if self._vllm_client is not None and self._vllm_weight_sync and self._vllm_client.communicator is not None:
            self._vllm_client.close_communicator()
        if self._vllm_llm is not None:
            del self._vllm_llm
            self._vllm_llm = None
        if self._is_rank_zero:
            self._metric_logger.close()
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

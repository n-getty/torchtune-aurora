# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Standalone MoE SFT recipe for Aurora/XPU (Qwen3-MoE family). Isolates the
# MoE training path from the GRPO/RL stack so throughput can be measured
# without generation/ref-forward/weight-sync dilution — comparable to a
# pretraining-style setup.
#
# This is `full_finetune_distributed_xpu.py` (the dense SFT recipe) with the
# Expert Parallelism mechanism ported over from
# `grpo_full_finetune_distributed_xpu.py`, single-replica case only
# (dp_replicate=1, world == ep == dp_shard — the topology this project's
# EP=8/EP=16 experimental configs already use). Multi-replica EP
# (dp_replicate>1) is NOT ported; add it only when a phase actually needs it.
#
# Deliberately excludes: vLLM/generation, ref-model, reward/GRPO-loss,
# weight-sync (raw_bytes/shm/xccl), deferred-broadcast overlap, WS3/WS10
# hang-avoidance code. None of that is reachable in a pure SFT recipe.

import os
import sys
import time

from functools import partial
from typing import Any, Optional, Union
from warnings import warn

# -- XPU / XCCL compatibility shim ---------------------------------------------
# On Intel XPU (Aurora), executing torchtune's __init__.py while an XCCL
# process group is active corrupts the L0 USM pointer table, causing every
# subsequent collective to fail. Pre-register the package in sys.modules so
# `from torchtune.X import ...` works without running __init__.py.
# --------------------------------------------------------------------------

_use_affinity_mask = (
    "ZE_AFFINITY_MASK" in os.environ and os.environ["ZE_AFFINITY_MASK"] != ""
)
_affinity_tiles = (
    os.environ.get("ZE_AFFINITY_MASK", "").split(",") if _use_affinity_mask else []
)
_xpu_device_index = (
    0 if (len(_affinity_tiles) == 1) else int(os.environ.get("LOCAL_RANK", "0"))
)

import importlib.util as _imp_util

import types as _types

import torch

if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
    else:
        _torchtune_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "torchtune",
        )
    if os.path.isdir(_torchtune_path):
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

# Ensure torchao is importable (torchtune.__init__ would normally guard this).
import torchao  # noqa: F401

from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import (
    compute_dataset_lengths,
    LengthGroupedDistributedBatchSampler,
    padded_collate_packed,
)
from torchtune.datasets import ConcatDataset
from torchtune.dev.rl.distributed import (
    _apply_expert_checkpointing,
    _apply_split_ac,
    _ep_build_grad_release_pg_map,
    _ep_release_fsdp_unsharded_grads,
    install_xpu_patches,
    restore_native_xpu_reduce_scatter,
    set_process_groups,
)
from torchtune.modules.embedding_utils import resize_token_embeddings
from torchtune.modules.loss import SFTLoss
from torchtune.modules.moe import utils as moe_utils, wire_ep_to_moe_modules
from torchtune.modules.moe._step_timing import (
    report_step as _moe_timing_report_step,
    reset_step as _moe_timing_reset_step,
    step_record as _moe_timing_step_record,
    STEP_TIMING_ENABLED as _MOE_STEP_TIMING_ENABLED,
    timed as _moe_timed,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    device_record_memory_history,
    DummyProfiler,
    get_xpu_distributed_backend,
    init_xpu_process_group,
    PROFILER_KEY,
    supports_memory_stats,
    VALID_BACKENDS_FOR_MEMORY_STATS,
)
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
)

from tqdm import tqdm

log = utils.get_logger("DEBUG")

install_xpu_patches()

# Opt-in (default off, rank-0 only): bracket torch.xpu.memory_stats() around
# the EP grad-release call and the optimizer step to isolate WHERE the
# mem_reserved ratchet observed in the gradient_accumulation_steps=2/4 HW
# tests actually grows. XPU can never call empty_cache() (standing FSDP+XPU
# UR-handle-leak constraint), so mem_reserved (the allocator's cached pool,
# which gates whether a new allocation succeeds) only ever grows across a
# run — the open question is whether a SPECIFIC ~5.9GB jump seen at the
# microbatch-1->microbatch-2 boundary in that investigation is an avoidable
# allocator artifact (e.g. a transient in grad-release or the optimizer step
# not being freed promptly) or an inherent consequence of the token payload
# itself. See memory/project_moe_sft_profiling_gradrelease_bottleneck_20260724.md.
_MOE_MEM_RATCHET_DEBUG = os.environ.get("TORCHTUNE_MOE_MEM_RATCHET_DEBUG", "0") == "1"


def _log_mem_ratchet_snapshot(tag: str) -> None:
    if not (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
        return
    stats = torch.xpu.memory_stats()
    alloc = torch.xpu.memory_allocated() / 1e9
    reserved = torch.xpu.memory_reserved() / 1e9
    # allocated_bytes.all.current == alloc; reserved_bytes.all.current == reserved
    # (redundant with the two calls above, kept for parity with existing
    # TORCHTUNE_MOE_BMM_DEBUG-style logging). The interesting NEW numbers are
    # the segment/inactive-split counts memory_stats() exposes that
    # memory_allocated()/memory_reserved() don't: if reserved grows but
    # active_bytes doesn't, the gap is INACTIVE cached blocks (fragmentation/
    # allocator-held, potentially avoidable) rather than genuinely live data.
    active = stats.get("active_bytes.all.current", 0) / 1e9
    inactive_split = stats.get("inactive_split_bytes.all.current", 0) / 1e9
    num_segments = stats.get("segment.all.current", 0)
    num_active_allocs = stats.get("allocation.all.current", 0)
    log.info(
        "[mem_ratchet_debug] %s: alloc=%.3fGB reserved=%.3fGB active=%.3fGB "
        "inactive_split=%.3fGB num_segments=%d num_active_allocs=%d",
        tag,
        alloc,
        reserved,
        active,
        inactive_split,
        num_segments,
        num_active_allocs,
    )


def _pipeline_ep_rank_groups(
    world_size: int, pipeline_parallel_degree: int
) -> list[list[int]]:
    if world_size % pipeline_parallel_degree != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by "
            f"pipeline_parallel_degree={pipeline_parallel_degree}"
        )
    stage_size = world_size // pipeline_parallel_degree
    return [
        list(range(stage * stage_size, (stage + 1) * stage_size))
        for stage in range(pipeline_parallel_degree)
    ]


def _order_pipeline_p2p_ops(operations, pipeline_stage: int):
    if pipeline_stage == 0:
        return sorted(operations, key=lambda operation: "recv" in operation.op.__name__)
    return sorted(operations, key=lambda operation: "send" in operation.op.__name__)


class FullFinetuneMoEDistributedXPU(FTRecipeInterface):
    """
    XPU-adapted full finetuning recipe for MoE transformer models (Qwen3-MoE family)
    on Intel Aurora, with Expert Parallelism. This is the dense
    ``FullFinetuneRecipeDistributedXPU`` recipe with EP support ported over from
    ``grpo_full_finetune_distributed_xpu.py`` (single-replica case: dp_replicate=1,
    world_size == expert_parallel_degree == dp_shard).

    Set ``expert_parallel_degree: 1`` (or omit it) to fall back to plain FSDP2 —
    every MoE expert param is then just another FSDP2-sharded param, and this
    recipe is numerically/structurally identical to the dense recipe run on an
    MoE model with no EP metadata at all.

    See ``FullFinetuneRecipeDistributedXPU`` docstring for the base feature set
    (FSDP2, activation checkpointing/offloading, bf16, gradient accumulation,
    checkpointing, logging, gradient clipping) — all unchanged here.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``expert_parallel_degree > 1`` and ``data_parallel_replicate_dim != 1``
            (multi-replica EP is not supported by this recipe).
    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device

        # XPU: route through explicit per-rank tile index (vs. generic
        # utils.get_device which doesn't see ZE_AFFINITY_MASK semantics).
        if device_type == "xpu":
            self._device = torch.device(f"xpu:{_xpu_device_index}")
            torch.xpu.set_device(_xpu_device_index)
        else:
            self._device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (XCCL on XPU, NCCL on CUDA, GLOO on CPU).
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        if device_type == "xpu":
            self.distributed_backend = get_xpu_distributed_backend(
                device_type,
                offload_ops_to_cpu=self.fsdp_cpu_offload
                or self._enable_async_checkpointing,
            )
        else:
            self.distributed_backend = training.get_distributed_backend(
                device_type,
                offload_ops_to_cpu=self.fsdp_cpu_offload
                or self._enable_async_checkpointing,
            )

        # MPI transport pre-init barrier (production multi-node only).
        # init_xpu_process_group internally strips device_id on multi-node.
        if os.environ.get("CCL_ATL_TRANSPORT") == "mpi":
            try:
                from mpi4py import MPI

                MPI.COMM_WORLD.Barrier()
            except ImportError:
                pass

        if device_type == "xpu":
            init_xpu_process_group(
                self.distributed_backend, device_index=_xpu_device_index
            )
        else:
            from torch.distributed import init_process_group as _init_pg

            _init_pg(self.distributed_backend)

        # SDPA note: torch.backends.cuda.enable_flash_sdp/* are no-ops on XPU.
        # The validated XPU SDPA fast path is TORCHTUNE_USE_IPEX_VARLEN=1.
        if device_type == "xpu" and cfg.get("force_math_sdpa", False):
            warn(
                "force_math_sdpa=True is a no-op on XPU (CUDA-only toggle). "
                "Set TORCHTUNE_USE_IPEX_VARLEN=1 for the validated XPU SDPA fast path."
            )

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self.tp_plan = cfg.get("tensor_parallel_plan", None)
        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        if self.tp_degree > 1:
            # DTensor does not support grouped_mm yet
            moe_utils.use_grouped_mm = False
        self.cp_degree = cfg.get("context_parallel_dim", 1)
        self.context_parallel_rotate_method = cfg.get(
            "context_parallel_rotate_method", "allgather"
        )
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)
        self._pipeline_parallel_degree = cfg.get("pipeline_parallel_degree", 1)
        self._pipeline_schedule = cfg.get("pipeline_schedule", "1f1b")
        self._pipeline_microbatch_size = cfg.get("pipeline_microbatch_size", 2)
        self._pipeline_split_layer = cfg.get("pipeline_split_layer", 24)
        self._sequence_length = cfg.tokenizer.get("max_seq_len", None)
        self._pipeline_stage = 0
        if self._pipeline_parallel_degree not in (1, 2):
            raise ValueError("pipeline_parallel_degree must be 1 or 2")
        if self._pipeline_schedule != "1f1b":
            raise ValueError("pipeline_schedule must be 1f1b")
        if self._pipeline_microbatch_size < 1:
            raise ValueError("pipeline_microbatch_size must be positive")
        if self._pipeline_parallel_degree == 2 and self.tp_degree != 1:
            raise ValueError("PP=2 does not currently support tensor parallelism")

        # Expert Parallelism config. EP reuses the dp_shard process group (no new
        # communicators). ParallelDims(ep=...) provides the generic mesh
        # construction + validation (ep == dp_shard, forces the dp_shard PG into
        # existence); its own shard_experts_for_ep()/disable_fsdp2_backward_prefetch()
        # helpers are NOT used here because shard_experts_for_ep only matches
        # torchtune.modules.moe.experts.GroupedExperts, not Qwen3-MoE's
        # GroupedExpertsHF. This recipe instead ports the hand-rolled XPU
        # mechanism proven in grpo_full_finetune_distributed_xpu.py (solo-FSDP2
        # expert wrap, gloo EP dispatch PG, manual grad release), which is
        # model-architecture-generic via qwen3_moe_ep_plan/gemma4_ep_plan.
        # This recipe only supports single-replica EP (dp_replicate=1, world ==
        # expert_parallel_degree == dp_shard) — ParallelDims._validate() enforces
        # ep == dp_shard, but not dp_replicate == 1; add that check explicitly.
        self._expert_parallel_degree = cfg.get("expert_parallel_degree", 1)
        self._expert_cpu_offload = cfg.get("expert_cpu_offload", False)
        self._ep_active = self._expert_parallel_degree > 1
        if self._ep_active and data_replicate != 1:
            raise ValueError(
                "This recipe only supports expert_parallel_degree > 1 with "
                "data_parallel_replicate_dim == 1 (single-replica EP: "
                "world_size == expert_parallel_degree == dp_shard). "
                f"Got data_parallel_replicate_dim={data_replicate}."
            )

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            cp=self.cp_degree,
            world_size=self.world_size,
            ep=self._expert_parallel_degree,
            pp=self._pipeline_parallel_degree,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        if self._ep_active:
            self._ep_mesh = self.parallel_dims.ep_mesh
            self._ep_degree = self._ep_mesh.size()
            self._ep_rank = self._ep_mesh.get_local_rank()
            self._pipeline_stage = (
                self.world_mesh["pp"].get_local_rank()
                if self.parallel_dims.pp_enabled
                else 0
            )
            self._pipeline_group = (
                self.world_mesh["pp"].get_group()
                if self.parallel_dims.pp_enabled
                else None
            )

            # Fix A2 (ported from GRPO recipe): inject a gloo CPU-bounce PG for EP
            # dispatch so it does not fall back to native XCCL (IPC handle
            # accumulation) unless TORCHTUNE_EP_USE_XCCL=1 is explicitly set.
            import torch.distributed.distributed_c10d as _dc10d_ep

            _default_pg_ep = _dc10d_ep._get_default_group()
            _orig_bdev_ep = _default_pg_ep.bound_device_id
            _ep_rank_groups = _pipeline_ep_rank_groups(
                self.world_size, self._pipeline_parallel_degree
            )
            _default_pg_ep.bound_device_id = None
            try:
                import datetime as _dt_ep

                _ep_gloo_groups = [
                    torch.distributed.new_group(
                        ranks,
                        backend="gloo",
                        timeout=_dt_ep.timedelta(seconds=1800),
                    )
                    for ranks in _ep_rank_groups
                ]
            finally:
                _default_pg_ep.bound_device_id = _orig_bdev_ep
            _ep_gloo_pg = _ep_gloo_groups[self._pipeline_stage]
            from torchtune.modules.moe import _parallelism as _ep_par

            _ep_par._GLOO_EP_PG = _ep_gloo_pg
            log.info(
                "EP: injected _GLOO_EP_PG into _parallelism (gloo CPU-bounce for "
                "EP dispatch, avoids XCCL IPC accumulation)."
            )

            # Gloo mirror of the dp_shard (== EP) group, used by
            # _ep_release_fsdp_unsharded_grads / _xpu_reduce_scatter_via_allreduce
            # CPU-bounce. Separate communicator from _GLOO_EP_PG (used for token
            # dispatch) to avoid sequence-number collisions on the same TCP socket.
            _default_pg_ep.bound_device_id = None
            try:
                _gloo_dp_shard_groups = [
                    torch.distributed.new_group(ranks, backend="gloo")
                    for ranks in _ep_rank_groups
                ]
            finally:
                _default_pg_ep.bound_device_id = _orig_bdev_ep
            _gloo_dp_shard_pg = _gloo_dp_shard_groups[self._pipeline_stage]
            self._gloo_dp_shard_pg = _gloo_dp_shard_pg
            xccl_dp_shard_pg = self._ep_mesh.get_group()
            set_process_groups(
                None,  # gloo_dp_rep_pg — unused at dp_replicate=1
                self._gloo_dp_shard_pg,  # world-sized gloo (matches ep/dp_shard size)
                self._gloo_dp_shard_pg,  # gloo_global_pg — same group at dp_replicate=1
                None,  # xccl_dp_rep_pg — unused at dp_replicate=1
                1,  # dp_rep_degree
                self._ep_degree,  # dp_shard_degree (== EP degree)
                xccl_dp_shard_pg=xccl_dp_shard_pg,
            )
            log.info(
                "Expert Parallelism enabled: ep_degree=%d (reuses dp_shard group)",
                self._expert_parallel_degree,
            )

        # Logging attributes
        self._is_metric_rank = (
            self._is_rank_zero
            if self._pipeline_parallel_degree == 1
            else self._pipeline_stage == 1 and self._ep_rank == 0
        )
        self._output_dir = cfg.output_dir
        self._measurement_model = str(cfg.model.get("_component_", "unknown"))
        self._measurement_checkpoint = str(cfg.get("base_model_path", "unknown"))
        self._measurement_batch_size = int(cfg.get("batch_size", 0))
        self._measurement_microbatch_size = int(
            cfg.get("pipeline_microbatch_size", cfg.get("batch_size", 0))
        )
        self._measurement_gradient_accumulation = int(
            cfg.get("gradient_accumulation_steps", 1)
        )
        self._measurement_optimizer = str(
            cfg.optimizer.get("_component_", "unknown")
        )
        self._measurement_step_timings = []
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        # Dataloader async-prefetch knobs (see _setup_data).
        self._dataloader_num_workers = cfg.get("dataloader_num_workers", 2)
        self._dataloader_pin_memory = cfg.get("dataloader_pin_memory", False)
        self._dataloader_prefetch_factor = cfg.get("dataloader_prefetch_factor", 2)
        if (
            self._log_peak_memory_stats
            and self._device.type not in VALID_BACKENDS_FOR_MEMORY_STATS
        ):
            self._logger.info(
                f"log_peak_memory_stats was set to True; however, training device is not in {VALID_BACKENDS_FOR_MEMORY_STATS}."
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._native_ep_sharded_experts = cfg.get("native_ep_sharded_experts", False)
        self._native_fsdp_grad_reduce = cfg.get("native_fsdp_grad_reduce", False)
        self._native_expert_parameter_ids: set[int] = set()
        if self._native_ep_sharded_experts and not self._ep_active:
            raise ValueError(
                "native_ep_sharded_experts requires expert_parallel_degree > 1"
            )
        if self._native_ep_sharded_experts and self._expert_cpu_offload:
            raise ValueError(
                "native_ep_sharded_experts does not support expert_cpu_offload"
            )
        if self._native_fsdp_grad_reduce and not self._native_ep_sharded_experts:
            raise ValueError(
                "native_fsdp_grad_reduce requires native_ep_sharded_experts"
            )
        if self._native_fsdp_grad_reduce:
            restore_native_xpu_reduce_scatter()
        if self._native_ep_sharded_experts:
            os.environ["TORCHTUNE_MOE_CHECKPOINT_EP_DEGREE"] = str(self._ep_degree)
            os.environ["TORCHTUNE_MOE_CHECKPOINT_EP_RANK"] = str(self._ep_rank)
        if self._pipeline_parallel_degree == 2:
            os.environ["TORCHTUNE_MOE_CHECKPOINT_PIPELINE_DEGREE"] = "2"
            os.environ["TORCHTUNE_MOE_CHECKPOINT_PIPELINE_STAGE"] = str(
                self._pipeline_stage
            )
            os.environ["TORCHTUNE_MOE_CHECKPOINT_PIPELINE_SPLIT_LAYER"] = str(
                self._pipeline_split_layer
            )
        # XPU multi-node: each rank sees only its tile as xpu:0 under
        # ZE_AFFINITY_MASK=$LOCAL_RANK (set by the per-rank mpiexec wrapper).
        _wrap_local_rank = device_type == "xpu" and len(_affinity_tiles) == 1
        _saved_local_rank = os.environ.get("LOCAL_RANK")
        if _wrap_local_rank:
            os.environ["LOCAL_RANK"] = "0"
        try:
            self._checkpoint_client = CheckpointClient(cfg)
        finally:
            if _wrap_local_rank:
                if _saved_local_rank is None:
                    os.environ.pop("LOCAL_RANK", None)
                else:
                    os.environ["LOCAL_RANK"] = _saved_local_rank
        self._enable_fp8_training = cfg.get("enable_fp8_training", False)
        self._fp8_recipe_name = cfg.get("fp8_recipe_name", None)
        self.save_every_n_steps = cfg.get("save_every_n_steps")
        self._skip_checkpointing = cfg.get("skip_checkpointing", False)

        self._run_val_every_n_steps = cfg.get("run_val_every_n_steps", None)
        if self._run_val_every_n_steps is not None:
            assert (
                cfg.get("dataset_val") is not None
            ), "run_val_every_n_steps is set but dataset_val is not configured"

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )
            if self._ep_active:
                # EP sets reduce_grads=False on every FSDPParamGroup and defers grad
                # sync to the manual post-backward _ep_release_fsdp_unsharded_grads
                # call in train(). optimizer_in_bwd fires .step() via a
                # post-accumulate-grad hook DURING backward, before that release
                # ever runs — it would silently step on unsynced, per-rank-local
                # unsharded gradients instead of the reduced sharded grad.
                raise RuntimeError(
                    "optimizer_in_bwd is not supported with expert_parallel_degree > 1: "
                    "EP defers gradient sync to a manual post-backward release, but "
                    "optimizer_in_bwd steps during backward on ungathered gradients. "
                    "Please set optimizer_in_bwd=False, or expert_parallel_degree=1."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        self._activation_offloading_use_streams = cfg.get(
            "activation_offloading_use_streams", True
        )
        if (
            self._enable_activation_offloading
            and self._activation_offloading_use_streams
            and self.parallel_dims.tp_enabled
        ):
            warn(
                message=(
                    "Using activation offloading with streams is not advised in tensor parallel, and may "
                    "cause unstable training. It is advised to set activation_offloading_use_streams: False"
                )
            )
        if self._enable_activation_offloading:
            if device_type != "cuda" and device_type != "xpu":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA or XPU"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self.global_step = ckpt_dict[training.STEPS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

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

        if self._is_metric_rank:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        # Load the base model
        state_dict = self._checkpoint_client.load_base_checkpoint()

        compile = cfg.get("compile")
        compile_bool = bool(compile)
        self._compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

        self._compile_model = compile_bool
        self._compile_loss = compile_bool
        self._compile_optimizer_step = compile_bool
        self._compile_scale_grads = compile_bool
        if isinstance(compile, DictConfig):
            self._compile_model = compile.get("model", True)
            self._compile_loss = compile.get("loss", True)
            self._compile_optimizer_step = compile.get("optimizer_step", False)
            _scale_grads_default = self._device.type != "xpu"
            self._compile_scale_grads = compile.get("scale_grads", _scale_grads_default)
        self._compile_dynamic = cfg.get(
            "compile_dynamic", True if self._device.type == "xpu" else False
        )
        if self._compile_model:
            # Capture scalar outputs is required to compile MoE
            torch._dynamo.config.capture_scalar_outputs = True

        # This indirection is needed to apply torch.compile to scale_grads step.
        self._grad_scaler = training.scale_grads_
        if self._compile_scale_grads:
            self._grad_scaler = torch.compile(
                self._grad_scaler, backend=self._compile_backend
            )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        self.use_loss_parallel_ctx_manager = self.parallel_dims.tp_enabled and getattr(
            self._loss_fn,
            "tp_requires_loss_parallel_ctx_manager",
            False,
        )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            activation_offloading_use_streams=self._activation_offloading_use_streams,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=state_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
            checkpoint_experts=cfg.get("checkpoint_experts", False),
            attention_checkpoint_every=cfg.get("attention_checkpoint_every", 1),
        )
        if self._pipeline_parallel_degree == 2:
            if cfg.batch_size % self._pipeline_microbatch_size != 0:
                raise ValueError(
                    "batch_size must be divisible by pipeline_microbatch_size"
                )
            from torch.distributed.pipelining import PipelineStage, Schedule1F1B
            from torch.distributed.pipelining import schedules as pipeline_schedules

            original_batch_p2p = pipeline_schedules._batch_p2p

            def ordered_batch_p2p(operations, desc=None):
                if desc in {"fwd_send_bwd_recv", "bwd_send_fwd_recv"}:
                    operations = _order_pipeline_p2p_ops(
                        operations, self._pipeline_stage
                    )
                return original_batch_p2p(operations, desc)

            pipeline_schedules._batch_p2p = ordered_batch_p2p

            microbatch_shape = (
                self._pipeline_microbatch_size,
                cfg.tokenizer.max_seq_len,
            )
            stage_input = (
                torch.empty(microbatch_shape, dtype=torch.long, device=self._device)
                if self._pipeline_stage == 0
                else torch.empty(
                    (*microbatch_shape, self._model.hidden_dim),
                    dtype=self._dtype,
                    device=self._device,
                )
            )
            stage_output = (
                torch.empty(
                    (*microbatch_shape, self._model.hidden_dim),
                    dtype=self._dtype,
                    device=self._device,
                )
            )
            self._pipeline_stage_runtime = PipelineStage(
                self._model,
                stage_index=self._pipeline_stage,
                num_stages=2,
                device=self._device,
                input_args=stage_input,
                output_args=stage_output,
                group=self._pipeline_group,
            )
            if os.environ.get("TORCHTUNE_MOE_PIPELINE_TRACE", "0") != "0":
                stage_runtime = self._pipeline_stage_runtime
                original_forward_one_chunk = stage_runtime.forward_one_chunk
                original_backward_one_chunk = stage_runtime.backward_one_chunk

                def traced_forward_one_chunk(chunk_id, *args, **kwargs):
                    print(
                        f"PP_CHUNK_FWD_BEGIN rank={self.rank} "
                        f"stage={self._pipeline_stage} chunk={chunk_id}",
                        flush=True,
                    )
                    output = original_forward_one_chunk(chunk_id, *args, **kwargs)
                    print(
                        f"PP_CHUNK_FWD_END rank={self.rank} "
                        f"stage={self._pipeline_stage} chunk={chunk_id}",
                        flush=True,
                    )
                    return output

                def traced_backward_one_chunk(chunk_id, *args, **kwargs):
                    print(
                        f"PP_CHUNK_BWD_BEGIN rank={self.rank} "
                        f"stage={self._pipeline_stage} chunk={chunk_id}",
                        flush=True,
                    )
                    result = original_backward_one_chunk(chunk_id, *args, **kwargs)
                    print(
                        f"PP_CHUNK_BWD_END rank={self.rank} "
                        f"stage={self._pipeline_stage} chunk={chunk_id}",
                        flush=True,
                    )
                    return result

                stage_runtime.forward_one_chunk = traced_forward_one_chunk
                stage_runtime.backward_one_chunk = traced_backward_one_chunk

            def pipeline_loss(outputs, labels):
                valid_tokens = (labels != self._loss_fn.ignore_index).sum()
                return (
                    self._loss_fn(
                        outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1)
                    )
                    * valid_tokens
                )

            self._pipeline_schedule_runtime = Schedule1F1B(
                self._pipeline_stage_runtime,
                n_microbatches=cfg.batch_size // self._pipeline_microbatch_size,
                loss_fn=pipeline_loss,
                scale_grads=False,
            )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        if cfg.get("resize_token_embeddings", False):
            resize_token_embeddings(self._model, self._tokenizer.vocab_size)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                state_dict[training.OPT_KEY] if training.OPT_KEY in state_dict else None
            ),
        )
        if self._compile_optimizer_step:
            if self._optimizer_in_bwd:
                raise ValueError(
                    "optimizer_in_bwd not supported with compiling the optimizer step"
                )
            self._optimizer.step = torch.compile(
                self._optimizer.step,
                backend=self._compile_backend,
            )

        if self._resume_from_checkpoint:
            if self._enable_async_checkpointing:
                try:
                    state_dict = self._checkpoint_client.load_distributed_checkpoint(
                        self._model,
                        (
                            self._optim_ckpt_wrapper
                            if self._optimizer_in_bwd
                            else self._optimizer
                        ),
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )
            self._update_recipe_state(state_dict)

        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile_loss:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        utils.log_rank_zero(self._logger, "Loss is initialized.")

        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            length_grouped_buckets=cfg.get("length_grouped_buckets", None),
            length_grouped_batch_sizes=cfg.get("length_grouped_batch_sizes", None),
            max_seq_len=cfg.tokenizer.get("max_seq_len", None),
            dataloader_state_dict=state_dict.get(training.DATALOADER_KEY, None),
        )

        self._val_dataloader = None
        if cfg.get("dataset_val") is not None:
            batch_size_val = cfg.get("batch_size_val", cfg.batch_size)
            self._val_dataloader = self._setup_data(
                cfg_dataset=cfg.dataset_val,
                batch_size=batch_size_val,
                collate_fn=collate_name,
                shuffle=False,
                dataloader_state_dict=state_dict.get(training.VAL_DATALOADER_KEY, None),
            )

        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch

        if self.save_every_n_steps is None:
            self.save_every_n_steps = self._steps_per_epoch
            self.checkpoint_dir_prefix = "epoch"
        else:
            self.checkpoint_dir_prefix = "step"

        if (
            self._resume_from_checkpoint
            and self.global_step % self._steps_per_epoch == 0
        ):
            list(self._dataloader)

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                self._logger.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            optimizer = self._optimizer

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._optimizer_in_bwd:
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
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
            self._logger, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        activation_offloading_use_streams: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
        checkpoint_experts: bool = False,
        attention_checkpoint_every: int = 1,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
           c. When Expert Parallelism is active, expert meta params are pre-shrunk
              from [num_experts, ...] to [num_experts // ep_degree, ...] BEFORE
              activation-checkpointing wrapping renames modules (AC wrapping inserts
              "_checkpoint_wrapped_module" into module paths — doing the shrink after
              would desync names against model_state_dict's clean keys).
        """
        utils.log_rank_zero(
            self._logger,
            "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)
            if self._pipeline_parallel_degree == 2:
                from torchtune.models.qwen3_moe._pipeline import (
                    build_qwen3_moe_pipeline_stage,
                )

                model = build_qwen3_moe_pipeline_stage(
                    model,
                    stage_index=self._pipeline_stage,
                    split_layer=self._pipeline_split_layer,
                )

        if self._compile_model:
            training.compile_model(
                model, verbose=self._is_rank_zero, dynamic=self._compile_dynamic
            )

        if self._enable_fp8_training:
            if torch.__version__ < "2.8.0.dev20250318":
                raise RuntimeError(
                    "Float8 fine-tuning requires PyTorch 2.8.0.dev20250318 or later."
                )
            if self.cp_degree > 1:
                raise ValueError(
                    "Context Parallel for fp8 training is not currently supported"
                )
            model = convert_to_float8_training(model, self._fp8_recipe_name)

        # Apply tensor parallelism to the model
        if self.parallel_dims.tp_enabled:
            if not self.parallel_dims.dp_enabled and self.fsdp_cpu_offload:
                raise ValueError(
                    "Tensor parallelism is not supported with FSDP CPU offloading when data parallelism is disabled."
                )
            model = training.prepare_mha_for_tp(model, self.world_mesh["tp"])
            if self.tp_plan is not None:
                self.tp_plan = config.instantiate(
                    self.tp_plan,
                    model=model,
                    enable_fp8_training=self._enable_fp8_training,
                )
                if isinstance(self._loss_fn, SFTLoss):
                    self._loss_fn.tp_enabled = True
                    self.tp_plan = self._loss_fn.patch_tp_plan(self.tp_plan)

            parallelize_module(
                model,
                self.world_mesh["tp"],
                parallelize_plan=self.tp_plan,
            )

        # EP: pre-shrink expert meta params [num_experts,...] -> [num_experts/ep,...]
        # BEFORE activation-checkpointing wrapping (which renames modules). Must run
        # before either AC path below. Track full param names for model_state_dict
        # pre-slicing further down.
        _expert_param_names: set = set()
        _native_expert_params: set[nn.Parameter] = set()
        if self._ep_active:
            from torchtune.models.qwen3_moe._experts import (
                GroupedExpertsHF as _GEHF_pre,
            )
            from torchtune.modules.moe.experts import GroupedExperts as _GE_pre

            _expert_classes = (_GE_pre, _GEHF_pre)
            _ep_degree = self._ep_degree
            _ep_rank = self._ep_rank
            _full_shape = None
            _n_local = None
            for _ename, _emod in model.named_modules():
                if not (
                    _ename.endswith(".experts") and isinstance(_emod, _expert_classes)
                ):
                    continue
                for _pname, _param in list(_emod.named_parameters(recurse=False)):
                    _full_shape = _param.shape
                    assert (
                        _full_shape[0] % _ep_degree == 0
                    ), f"num_experts ({_full_shape[0]}) not divisible by ep_degree ({_ep_degree})"
                    _n_local = _full_shape[0] // _ep_degree
                    _new_shape = torch.Size([_n_local] + list(_full_shape[1:]))
                    setattr(
                        _emod,
                        _pname,
                        nn.Parameter(
                            torch.empty(_new_shape, dtype=_param.dtype, device="meta"),
                            requires_grad=_param.requires_grad,
                        ),
                    )
                    _expert_param_names.add(f"{_ename}.{_pname}")
                    if self._native_ep_sharded_experts:
                        _native_expert_params.add(getattr(_emod, _pname))
            utils.log_rank_zero(
                self._logger,
                f"EP: pre-shrunk {len(_expert_param_names)} expert meta params "
                f"[{_full_shape[0] if _full_shape else '?'},...] -> [{_n_local},...] "
                f"(EP rank {_ep_rank}/{_ep_degree}, before AC wrapping)",
            )

        # Activation checkpointing.
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        if enable_activation_checkpointing and ac_mode is None:
            if self._ep_active:
                # Split AC (correctness requirement, not a memory optimization):
                # the router's argsort(stable=True) tie-breaking can flip under
                # AC recompute, shifting num_tokens_per_expert by +/-1 and
                # desyncing ExpertParallel's cached dispatch shapes against the
                # autograd-saved combine shapes. MoE-bearing layers self-checkpoint
                # attention only; MoE runs once, outside AC. See
                # torchtune/dev/rl/distributed.py::_apply_split_ac (v158 fix).
                _n_moe_self_ac = _apply_split_ac(
                    model,
                    attention_checkpoint_every=attention_checkpoint_every,
                )
                utils.log_rank_zero(
                    self._logger,
                    f"EP: split AC applied — {_n_moe_self_ac} MoE attention blocks "
                    f"checkpointed (every {attention_checkpoint_every}; MoE outside AC); "
                    f"other layers wrapped normally.",
                )
                if checkpoint_experts:
                    # Opt-in, separate from split-AC above: checkpoints ONLY
                    # self.experts()'s own compute (safe — does not re-run the
                    # router, cannot reintroduce the v158 argsort-tie-break bug;
                    # see torchtune/modules/moe/moe.py::MoE's checkpoint_experts
                    # docstring and torchtune/dev/rl/distributed.py::
                    # _apply_expert_checkpointing). Motivated by the seq4096
                    # mem_reserved-ratchet finding — trades expert-compute
                    # recompute time for peak activation memory.
                    _n_expert_ckpt = _apply_expert_checkpointing(model)
                    utils.log_rank_zero(
                        self._logger,
                        f"EP: expert-compute checkpointing applied to "
                        f"{_n_expert_ckpt} MoE modules (checkpoint_experts=true).",
                    )
            else:
                if checkpoint_experts:
                    # No-op here, not an error: whole-layer AC below already
                    # wraps (and checkpoints) the MoE block as part of each
                    # TransformerSelfAttentionLayer, so a separate
                    # self.experts()-only checkpoint would double-nest
                    # checkpointing for no benefit. checkpoint_experts exists
                    # specifically to fill the gap split-AC creates (EP-only
                    # path, MoE excluded from AC entirely) — warn loudly so a
                    # config setting it on a non-EP run isn't silently
                    # ignored without explanation.
                    utils.log_rank_zero(
                        self._logger,
                        "checkpoint_experts=true has no effect without EP active "
                        "(whole-layer AC already checkpoints the MoE block here).",
                    )
                training.set_activation_checkpointing(
                    model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
                )

        # Optional: swap eager RMSNorm -> fused Triton RMSNorm on XPU (gated by
        # TORCHTUNE_USE_FUSED_RMSNORM=1). MUST run before shard_model.
        from torchtune.modules._fused_rmsnorm_xpu import maybe_swap_rmsnorm_for_fused

        _n_fused = maybe_swap_rmsnorm_for_fused(model)
        if _n_fused:
            utils.log_rank_zero(
                self._logger, f"fused RMSNorm engaged: {_n_fused} modules swapped"
            )

        # Optional: swap eager RoPE -> fused Triton RoPE on XPU (gated by
        # TORCHTUNE_USE_FUSED_ROPE=1).
        from torchtune.modules._fused_rope_xpu import maybe_swap_rope_for_fused

        _n_rope = maybe_swap_rope_for_fused(model)
        if _n_rope:
            utils.log_rank_zero(
                self._logger, f"fused RoPE engaged: {_n_rope} modules swapped"
            )

        # Expert Parallelism: install AllGather/ReduceScatter (or all_to_all, if
        # TORCHTUNE_EP_ALL2ALL=1) dispatch on expert modules. Must happen BEFORE
        # shard_model so EP metadata is attached to the original (un-FSDP2-wrapped)
        # expert modules.
        if self._ep_active:
            from torchtune.models.gemma4._parallelism import gemma4_ep_plan
            from torchtune.models.qwen3_moe._parallelism import qwen3_moe_ep_plan

            ep_plan = gemma4_ep_plan(model)
            if not ep_plan:
                ep_plan = qwen3_moe_ep_plan(model)
            if ep_plan:
                parallelize_module(model, self._ep_mesh, ep_plan)
                utils.log_rank_zero(
                    self._logger,
                    f"EP={self._expert_parallel_degree}: registered EP dispatch on "
                    f"{len(ep_plan)} expert module(s)",
                )
            else:
                utils.log_rank_zero(
                    self._logger,
                    f"EP={self._expert_parallel_degree}: NO expert modules matched any "
                    f"known EP plan (gemma4, qwen3_moe). Check model architecture.",
                )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        if self._ep_active:
            # Wire EP dispatch/combine callables directly onto each MoE module
            # (FSDP2's fully_shard drops parallelize_module's hooks, so MoE.forward()
            # needs plain-callable access to _ep_dispatch/_ep_combine).
            n_ep_wired = wire_ep_to_moe_modules(model)
            utils.log_rank_zero(
                self._logger,
                f"EP: wired dispatch/combine callables on {n_ep_wired} MoE modules",
            )

            from torchtune.models.qwen3_moe._experts import GroupedExpertsHF as _GEHF

            # Wrap expert modules with trivial 1-rank solo FSDP2 (no communication).
            # Meta params were pre-shrunk to EP-local sizes above, so FSDPParam is
            # initialized with the correct shapes. This excludes expert params from
            # the root dp_shard FSDP2 unit (which would otherwise try to re-shard
            # already-EP-partitioned weights across the wrong group).
            from torchtune.modules.moe.experts import GroupedExperts as _GE

            _solo_expert_classes = (_GE, _GEHF)
            _n_solo_wrapped = 0
            _solo_wrapped_mods = []
            if self._native_ep_sharded_experts:
                utils.log_rank_zero(
                    self._logger,
                    f"EP native state: leaving {len(_native_expert_params)} local "
                    "expert parameters permanently materialized outside FSDP2",
                )
            else:
                from torch.distributed._composable.fsdp import (
                    CPUOffloadPolicy as _SoloCPUOffload,
                    fully_shard as _fully_shard,
                )
                from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh

                _solo_groups = []
                for _r in range(self.world_size):
                    _sg = torch.distributed.new_group([_r])
                    _solo_groups.append(_sg)
                _my_solo_pg = _solo_groups[self.rank]
                _solo_mesh = _DeviceMesh.from_group(_my_solo_pg, self._device.type)
                for _ename, _emod in model.named_modules():
                    if isinstance(_emod, _solo_expert_classes):
                        _solo_kwargs = {
                            "mesh": _solo_mesh,
                            "reshard_after_forward": self._expert_cpu_offload,
                        }
                        if self._expert_cpu_offload:
                            _solo_kwargs["offload_policy"] = _SoloCPUOffload()
                        _fully_shard(_emod, **_solo_kwargs)
                        _solo_wrapped_mods.append(_emod)
                        _n_solo_wrapped += 1
                utils.log_rank_zero(
                    self._logger,
                    f"EP: wrapped {_n_solo_wrapped} expert modules with trivial "
                    "1-rank FSDP2 (no communication, excluded from root dp_shard FSDP2)",
                )
            # Suppress reduce_grads on expert 1-rank FSDP2 groups — CCL's
            # reduce_scatter_tensor for 1-rank groups tries to register L0 IPC
            # handles for the grad buffer -> ze_handle_manager crash. With
            # reduce_grads=False, FSDP2 skips reduce_scatter entirely; expert
            # grads accumulate in param.grad for manual release in train().
            from torch.distributed.fsdp import FSDPModule as _FSDPModule

            _n_grads_suppressed = 0
            for _emod in _solo_wrapped_mods:
                if isinstance(_emod, _FSDPModule):
                    _fsdp_state = _emod._get_fsdp_state()
                    if (
                        _fsdp_state is not None
                        and _fsdp_state._fsdp_param_group is not None
                    ):
                        _fsdp_state._fsdp_param_group.reduce_grads = False
                        _n_grads_suppressed += 1
            utils.log_rank_zero(
                self._logger,
                f"EP: suppressed reduce_grads on {_n_grads_suppressed} expert "
                f"FSDPParamGroups (prevents CCL ze_handle_manager crash for "
                f"1-rank reduce_scatter)",
            )

        # Apply Fully Sharded Data Parallelism to the (non-expert) model.
        # EP mode: ZeRO-2 (reshard_after_forward=False) on the full dp_shard mesh —
        # prevents an XCCL/OFI conflict between the dp_shard all-gather during
        # backward and the EP AllToAll/AllGather backward on the same fabric
        # (ported from grpo_full_finetune_distributed_xpu.py's single-replica EP
        # branch; see its comments for the full OFI EPERM/deadlock history).
        if self.parallel_dims.dp_shard_enabled or self.parallel_dims.cp_enabled:
            if self._pipeline_parallel_degree == 2:
                pipeline_terminal_modules = ["tok_embeddings", "norm", "output"]
                custom_sharded_layers = list(custom_sharded_layers or [])
                custom_sharded_layers.extend(pipeline_terminal_modules)
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]

            if self._ep_active:
                fsdp2_mesh = self.world_mesh["dp_shard"]
                fsdp2_raf = False
                utils.log_rank_zero(
                    self._logger,
                    "EP active: using reshard_after_forward=False (ZeRO-2) for "
                    "non-expert FSDP2 to prevent XCCL conflict with EP AllGather/"
                    "AllToAll backward.",
                )
            else:
                if self.parallel_dims.dp_replicate_enabled:
                    dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
                else:
                    dp_mesh_dim_names = ("dp_shard_cp",)
                fsdp2_mesh = self.world_mesh[dp_mesh_dim_names]
                fsdp2_raf = reshard_after_forward

            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=fsdp2_raf,
                dp_mesh=fsdp2_mesh,
                ignored_params=(
                    _native_expert_params if self._native_ep_sharded_experts else None
                ),
            )

        if self._ep_active:
            # Disable FSDP2 backward prefetch for ALL wrapped modules — prevents
            # async all-gathers from firing during the EP collective's backward
            # (Aurora OFI EPERM; see torchtune/training/_distributed.py
            # ::disable_fsdp2_backward_prefetch for the full mechanism).
            if os.environ.get("TORCHTUNE_EP_DISABLE_BACKWARD_PREFETCH", "1") == "1":
                training.disable_fsdp2_backward_prefetch(model)

            # Suppress reduce_grads on ALL remaining FSDPParamGroups (non-expert
            # groups still had reduce_grads=True from shard_model above). With EP
            # active, all grad reduction goes through the manual gloo/XCCL release
            # helper in train() instead, avoiding an ordering race between EP
            # replica groups hitting reduce_scatter at different backward times.
            if self._native_fsdp_grad_reduce:
                utils.log_rank_zero(
                    self._logger,
                    "EP native state: using FSDP2 native reduce-scatter for "
                    "non-expert gradients; manual post-backward release disabled",
                )
                self._ep_grad_release_pg_map = {}
            else:
                from torch.distributed.fsdp import FSDPModule as _FSDPModuleAll

                _n_all_suppressed = 0
                for _mod in model.modules():
                    if isinstance(_mod, _FSDPModuleAll):
                        _fsdp_state = _mod._get_fsdp_state()
                        if (
                            _fsdp_state is not None
                            and _fsdp_state._fsdp_param_group is not None
                            and _fsdp_state._fsdp_param_group.reduce_grads
                        ):
                            _fsdp_state._fsdp_param_group.reduce_grads = False
                            _n_all_suppressed += 1
                utils.log_rank_zero(
                    self._logger,
                    f"EP: suppressed reduce_grads on {_n_all_suppressed} additional "
                    "non-expert FSDPParamGroups (post-backward manual gloo/XCCL "
                    "release instead)",
                )
                self._ep_grad_release_pg_map = _ep_build_grad_release_pg_map(model)

            if self._native_ep_sharded_experts:
                from torch.distributed.tensor import DTensor as _DTensor

                _native_ids = {id(param) for param in _native_expert_params}
                _remaining_ids = {
                    id(param)
                    for module in model.modules()
                    if isinstance(module, _solo_expert_classes)
                    for param in module.parameters(recurse=False)
                }
                if _native_ids != _remaining_ids:
                    raise RuntimeError(
                        "EP native expert parameter identities changed during FSDP setup"
                    )
                if any(isinstance(param, _DTensor) for param in _native_expert_params):
                    raise RuntimeError(
                        "EP native expert parameters were unexpectedly converted to DTensor"
                    )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # EP: pre-slice expert params in the state dict to match the pre-shrunk
        # meta shapes above. Slice MUST be interleaved [_ep_rank::_ep_degree] to
        # match ExpertParallel._token_dispatch's ownership formula
        # (g = ep_rank + local_exp_idx * ep_degree) — contiguous slicing would
        # silently route tokens to the wrong experts on every EP rank. See
        # tests/torchtune/dev/rl/test_ep_slice_contract.py.
        if self._ep_active:
            _n_sd_sliced = 0
            _last_shape0 = None
            _last_n_local = None
            for _sd_name in list(model_state_dict.keys()):
                if _sd_name in _expert_param_names:
                    _ft = model_state_dict[_sd_name]
                    expected_local_experts = 128 // self._ep_degree
                    if _ft.shape[0] == expected_local_experts:
                        _n_sd_sliced += 1
                        _last_shape0 = 128
                        _last_n_local = expected_local_experts
                        continue
                    assert _ft.shape[0] % self._ep_degree == 0, (
                        f"Expert param {_sd_name}: shape[0]={_ft.shape[0]} not "
                        f"divisible by ep_degree={self._ep_degree}"
                    )
                    _last_shape0 = _ft.shape[0]
                    _last_n_local = _ft.shape[0] // self._ep_degree
                    model_state_dict[_sd_name] = _ft[
                        self._ep_rank :: self._ep_degree
                    ].contiguous()
                    _n_sd_sliced += 1
            utils.log_rank_zero(
                self._logger,
                f"EP: pre-sliced {_n_sd_sliced} expert params in model_state_dict "
                f"(interleaved {_last_shape0}->{_last_n_local} for EP rank "
                f"{self._ep_rank}/{self._ep_degree}; rank r owns global experts "
                f"r, r+{self._ep_degree}, r+2*{self._ep_degree}, ...)",
            )

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading, activation_offloading_use_streams
        )
        # context parallel
        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            rotate_method=self.context_parallel_rotate_method,
            world_mesh=self.world_mesh,
            model=model,
        )
        # remaining context managers for fwd/bwd
        self.train_context = training.get_train_context(
            enable_loss_parallel=self.use_loss_parallel_ctx_manager,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins (XPU XCCL: barrier does not take device_ids)
        if self._device.type == "xpu":
            torch.distributed.barrier()
        else:
            torch.distributed.barrier(device_ids=[self._device.index])

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in self._model.parameters()
            }
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.optim_map[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            utils.log_rank_zero(self._logger, "In-backward optimizers are set up.")
            return None
        else:
            optimizer_params: Any = self._model.parameters()
            if self._native_ep_sharded_experts:
                native_params = []
                distributed_params = []
                for param in self._model.parameters():
                    if isinstance(param, DTensor):
                        distributed_params.append(param)
                    else:
                        native_params.append(param)
                if not native_params or not distributed_params:
                    raise RuntimeError(
                        "native EP state requires both plain expert parameters and "
                        "DTensor non-expert parameters; "
                        f"plain={len(native_params)} distributed={len(distributed_params)}"
                    )
                self._native_expert_parameter_ids = {
                    id(parameter) for parameter in native_params
                }
                optimizer_params = [
                    {"params": distributed_params},
                    {"params": native_params},
                ]
                utils.log_rank_zero(
                    self._logger,
                    f"EP native state: split optimizer into {len(distributed_params)} "
                    f"DTensor and {len(native_params)} plain-tensor parameters",
                )
            optimizer = config.instantiate(cfg_optimizer, optimizer_params)
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(self._logger, "Optimizer is initialized.")
            return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        length_grouped_buckets: Optional[list] = None,
        length_grouped_batch_sizes: Optional[list] = None,
        max_seq_len: Optional[int] = None,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        num_workers = self._dataloader_num_workers
        collate_partial = (
            partial(
                collate_fn,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
                cp_degree=self.cp_degree,
                pad_to_multiple_of=self.parallel_dims.min_seq_len_divisor,
            )
            if not packed
            else padded_collate_packed
        )

        if length_grouped_buckets is not None:
            # Opt-in alternative to packing (mutually exclusive — bucketing IS the
            # packing alternative for engaging native XPU flash, which cannot accept
            # packing's block-diagonal document mask; see attention_utils.py's
            # _xpu_flash_call and memory/project_bioreason_sft_packing_scope_20260715).
            if packed:
                raise ValueError(
                    "length_grouped_buckets is mutually exclusive with dataset.packed=true "
                    "(bucketing replaces packing as the padding-waste mitigation; using "
                    "both is not a supported/meaningful combination)."
                )
            if max_seq_len is None:
                raise ValueError(
                    "length_grouped_buckets requires tokenizer.max_seq_len set."
                )
            if length_grouped_batch_sizes is None or len(
                length_grouped_batch_sizes
            ) != len(length_grouped_buckets):
                raise ValueError(
                    "length_grouped_batch_sizes must be set with the same length as "
                    "length_grouped_buckets."
                )
            buckets = sorted(
                int(b) for b in length_grouped_buckets if int(b) <= max_seq_len
            )
            if not buckets or buckets[-1] < max_seq_len:
                buckets.append(max_seq_len)
            cfg_map = {
                int(b): int(s)
                for b, s in zip(length_grouped_buckets, length_grouped_batch_sizes)
            }
            bbs = [cfg_map.get(b, 1) for b in buckets]
            t0 = time.perf_counter()
            lengths = compute_dataset_lengths(ds, max_seq_len)
            batch_sampler = LengthGroupedDistributedBatchSampler(
                lengths=lengths,
                buckets=buckets,
                bucket_batch_sizes=bbs,
                num_replicas=self.dp_degree,
                rank=self.dp_rank,
                shuffle=shuffle,
                seed=0,
            )
            utils.log_rank_zero(
                self._logger,
                "Length-grouped batch sampling ENABLED: buckets=%s batch_sizes=%s -> "
                "%d batches/rank/epoch (length scan %.1fs over %d examples)."
                % (buckets, bbs, len(batch_sampler), time.perf_counter() - t0, len(ds)),
            )
            dl_kwargs = dict(
                dataset=ds,
                batch_sampler=batch_sampler,
                collate_fn=collate_partial,
                num_workers=num_workers,
                pin_memory=self._dataloader_pin_memory,
            )
            if num_workers > 0:
                dl_kwargs["persistent_workers"] = True
                dl_kwargs["prefetch_factor"] = self._dataloader_prefetch_factor
            dataloader = StatefulDataLoader(**dl_kwargs)
            if dataloader_state_dict is not None:
                dataloader.load_state_dict(dataloader_state_dict)
            return dataloader

        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=shuffle, seed=0
        )
        dl_kwargs = dict(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_partial,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=self._dataloader_pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = self._dataloader_prefetch_factor
        dataloader = StatefulDataLoader(**dl_kwargs)
        if dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)

        return dataloader

    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch.pop("labels")

        with self.activations_handling_ctx:
            outputs = self._model(**batch)

        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
            if isinstance(outputs, DTensor):
                outputs = outputs.full_tensor()

        loss = self._loss_fn(outputs, labels)
        del outputs
        return loss

    def validate(self) -> dict[str, float]:
        self._model.eval()
        total_val_loss = torch.tensor(0.0, device=self._device)
        total_val_tokens = torch.tensor(0.0, device=self._device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._val_dataloader):
                utils.batch_to_device(batch, self._device)
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()
                val_loss = self._loss_step(batch) * current_num_tokens
                total_val_loss += val_loss
                total_val_tokens += current_num_tokens

        torch.distributed.all_reduce(total_val_loss)
        torch.distributed.all_reduce(total_val_tokens)

        avg_val_loss = (
            (total_val_loss / total_val_tokens).item()
            if total_val_tokens > 0
            else float("inf")
        )
        log_dict = {"val_loss": avg_val_loss}

        if self._is_rank_zero:
            self._logger.info(f"Validation loss: {avg_val_loss:.4f}")
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )

        self._model.train()
        return log_dict

    def save_checkpoint(self, *, epoch: int, full_tensors: bool):
        if self._skip_checkpointing:
            utils.log_rank_zero(
                self._logger,
                "Skipping checkpoint write because skip_checkpointing=true",
            )
            return
        if self._native_ep_sharded_experts:
            raise RuntimeError(
                "Checkpointing native_ep_sharded_experts is not implemented: "
                "expert parameters must first be gathered from their interleaved "
                "EP owners into the full checkpoint layout. Refusing to write an "
                "incomplete checkpoint."
            )
        training_progress_epoch = epoch
        if self.global_step % self._steps_per_epoch == 0:
            training_progress_epoch += 1

        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=(
                self._optimizer
                if not self._optimizer_in_bwd
                else self._optim_ckpt_wrapper
            ),
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=training_progress_epoch,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                steps_run=self.global_step,
                total_training_steps=self.total_epochs * self._steps_per_epoch,
                dataloader_state_dict=self._dataloader.state_dict(),
                val_dataloader_state_dict=(
                    self._val_dataloader.state_dict()
                    if self._val_dataloader is not None
                    else {}
                ),
            ),
            epoch=epoch,
            single_device=False,
            full_tensors=full_tensors,
            dir_prefix=self.checkpoint_dir_prefix,
        )

    def _train_pipeline(self) -> None:
        training.cleanup_before_training()
        self._optimizer.zero_grad()
        measurement_snapshot = None
        if os.environ.get("TORCHTUNE_MOE_MEASURE") == "1":
            from torchtune.modules.moe.measurement import snapshot_model_measurements

            measurement_snapshot = snapshot_model_measurements
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            self._dataloader.sampler.set_epoch(curr_epoch)
            for batch in self._dataloader:
                if self.global_step >= self._steps_per_epoch:
                    break
                utils.batch_to_device(batch, self._device)
                labels = batch["labels"]
                local_num_tokens = (labels != self._loss_fn.ignore_index).sum()
                losses = []
                step_start = time.perf_counter()
                if self._pipeline_stage == 0:
                    self._pipeline_schedule_runtime.step(batch["tokens"])
                else:
                    self._pipeline_schedule_runtime.step(
                        target=labels, losses=losses, return_outputs=False
                    )
                if measurement_snapshot is not None:
                    measurement_snapshot(
                        self._model,
                        "pipeline_compute",
                        self._device,
                        step=self.global_step,
                    )
                global_num_tokens = local_num_tokens.clone()
                torch.distributed.all_reduce(
                    global_num_tokens, group=self._ep_mesh.get_group()
                )
                parameters = list(self._model.parameters())
                training.scale_grads_for_native_ep_(
                    parameters,
                    self._native_expert_parameter_ids,
                    self._ep_degree / global_num_tokens,
                    1.0 / global_num_tokens,
                )
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)
                if measurement_snapshot is not None:
                    measurement_snapshot(
                        self._model,
                        "optimizer",
                        self._device,
                        step=self.global_step,
                    )
                    measurement_snapshot(
                        self._model,
                        "steady_state",
                        self._device,
                        step=self.global_step,
                    )
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()
                self.global_step += 1
                if self._pipeline_stage == 1:
                    loss = torch.stack([item.detach() for item in losses]).sum()
                    loss_to_log = loss.item() / local_num_tokens.item()
                    if self._is_metric_rank:
                        elapsed = time.perf_counter() - step_start
                        self._metric_logger.log_dict(
                            {
                                "loss": loss_to_log,
                                "lr": get_lr(self._optimizer),
                                "time_per_step_s": elapsed,
                                "tokens_per_second_per_gpu": (labels.numel() / elapsed),
                                **training.get_memory_stats(device=self._device),
                            },
                            step=self.global_step,
                        )
            self.epochs_run += 1
        self.save_checkpoint(epoch=self.total_epochs - 1, full_tensors=True)

    def train(self) -> None:
        """
        The core training loop.
        """
        if self._pipeline_parallel_degree == 2:
            self._train_pipeline()
            return
        training.cleanup_before_training()

        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        t0 = time.perf_counter()
        running_loss = 0
        measurement_snapshot = None
        if os.environ.get("TORCHTUNE_MOE_MEASURE") == "1":
            from torchtune.modules.moe.measurement import snapshot_model_measurements

            measurement_snapshot = snapshot_model_measurements
        num_tokens = 0
        local_num_tokens = 0

        self._profiler.start()
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            inner_step_count = self.global_step % self._steps_per_epoch
            pbar = tqdm(
                initial=inner_step_count,
                total=self._steps_per_epoch,
                desc=f"{self.epochs_run}|{self.global_step}",
            )

            # DataLoader always exposes a (possibly dummy) .sampler even when
            # batch_sampler= was passed explicitly, so truthiness can't distinguish
            # the two paths — check the actual type of our custom sampler instead.
            epoch_sampler = (
                self._dataloader.batch_sampler
                if isinstance(
                    self._dataloader.batch_sampler, LengthGroupedDistributedBatchSampler
                )
                else self._dataloader.sampler
            )
            epoch_sampler.set_epoch(curr_epoch)
            dataloader_iter = iter(self._dataloader)
            batch_count = 0

            while inner_step_count < self._steps_per_epoch:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break

                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and batch_count
                    == self.profiler_wait_steps + self.profiler_warmup_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=True)

                utils.batch_to_device(batch, self._device)

                if (
                    _MOE_STEP_TIMING_ENABLED
                    and batch_count % self._gradient_accumulation_steps == 0
                ):
                    _moe_timing_reset_step()

                with self.train_context(
                    self.context_parallel_manager(list(batch.values()))
                ):
                    current_num_tokens = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens
                    local_num_tokens += current_num_tokens

                    if _MOE_MEM_RATCHET_DEBUG and self._device.type == "xpu":
                        torch.xpu.synchronize()
                        _log_mem_ratchet_snapshot(f"PRE-FWD batch_count={batch_count}")
                    with _moe_timed("model_fwd_total"):
                        current_loss = self._loss_step(batch) * current_num_tokens
                    if _MOE_MEM_RATCHET_DEBUG and self._device.type == "xpu":
                        torch.xpu.synchronize()
                        _log_mem_ratchet_snapshot(f"POST-FWD batch_count={batch_count}")
                    loss_for_logging = current_loss.detach()
                    if isinstance(loss_for_logging, DTensor):
                        loss_for_logging = loss_for_logging.to_local()
                    running_loss += loss_for_logging.clone()
                    if measurement_snapshot is not None:
                        measurement_snapshot(
                            self._model,
                            "forward",
                            self._device,
                            step=self.global_step,
                            microbatch=batch_count,
                        )
                    with _moe_timed("backward_total"):
                        if self._optimizer_in_bwd:
                            torch.distributed.all_reduce(num_tokens)
                            current_loss = current_loss * (self.dp_degree / num_tokens)
                            current_loss.backward()
                        else:
                            is_last_microbatch = (
                                batch_count + 1
                            ) % self._gradient_accumulation_steps == 0
                            if (
                                self._gradient_accumulation_steps > 1
                                and hasattr(self._model, "set_requires_gradient_sync")
                                and (
                                    not self._ep_active or self._native_fsdp_grad_reduce
                                )
                            ):
                                # FSDP2 path. Skipped when EP is active: grad sync there
                                # goes through the manual per-microbatch release helper
                                # below, not FSDP2's own gradient-sync gating.
                                self._model.set_requires_gradient_sync(
                                    is_last_microbatch
                                )
                                current_loss.backward()
                            elif (
                                self._gradient_accumulation_steps > 1
                                and not is_last_microbatch
                                and hasattr(self._model, "no_sync")
                                and (
                                    not self._ep_active or self._native_fsdp_grad_reduce
                                )
                            ):
                                with self._model.no_sync():
                                    current_loss.backward()
                            else:
                                current_loss.backward()
                    if measurement_snapshot is not None:
                        measurement_snapshot(
                            self._model,
                            "backward",
                            self._device,
                            step=self.global_step,
                            microbatch=batch_count,
                        )

                    # EP: release FSDP2's unsharded grad pool into sharded
                    # `param.grad` as a DTensor. With reduce_grads=False on every
                    # FSDPParamGroup (set in _setup_model when EP is active), FSDP2
                    # never fires reduce_scatter — without this call, unsharded
                    # grads accumulate across microbatches/steps and are never
                    # synced across ranks. accumulate_into_grad=True on every
                    # microbatch after the first within an accumulation window.
                    # EP=1 path: skipped entirely (this block, including the sync
                    # calls, is inert when EP is off — matches the dense recipe).
                    if self._ep_active and not self._native_fsdp_grad_reduce:
                        if self._device.type == "xpu":
                            torch.xpu.synchronize()
                        if _MOE_MEM_RATCHET_DEBUG and self._device.type == "xpu":
                            _log_mem_ratchet_snapshot(
                                f"PRE-REL batch_count={batch_count}"
                            )
                        _accumulate = (
                            batch_count % self._gradient_accumulation_steps
                        ) > 0
                        with _moe_timed("manual_grad_release_total"):
                            _n_rel = _ep_release_fsdp_unsharded_grads(
                                self._model,
                                self._ep_grad_release_pg_map,
                                accumulate_into_grad=_accumulate,
                            )
                        if self._device.type == "xpu":
                            torch.xpu.synchronize()
                        if _MOE_MEM_RATCHET_DEBUG and self._device.type == "xpu":
                            _log_mem_ratchet_snapshot(
                                f"POST-REL batch_count={batch_count}"
                            )

                # Optimizer step (if not fused in backward call)
                if (batch_count + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        if _MOE_MEM_RATCHET_DEBUG and self._device.type == "xpu":
                            torch.xpu.synchronize()
                            _log_mem_ratchet_snapshot(
                                f"PRE-OPTIM batch_count={batch_count}"
                            )
                        with _moe_timed("optimizer_step_total"):
                            torch.distributed.all_reduce(num_tokens)

                            parameters = list(self._model.parameters())
                            foreach = False if self.parallel_dims.tp_enabled else None
                            if self._native_ep_sharded_experts:
                                training.scale_grads_for_native_ep_(
                                    parameters,
                                    self._native_expert_parameter_ids,
                                    self.world_size / num_tokens,
                                    1.0 / num_tokens,
                                    foreach,
                                )
                            else:
                                self._grad_scaler(
                                    parameters,
                                    self.world_size / num_tokens,
                                    foreach,
                                )

                            if self._clip_grad_norm is not None:
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    self._model.parameters(),
                                    max_norm=float(self._clip_grad_norm),
                                )
                                if isinstance(grad_norm, DTensor):
                                    grad_norm = grad_norm.full_tensor()
                            self._optimizer.step()
                            self._optimizer.zero_grad(set_to_none=True)
                        if measurement_snapshot is not None:
                            measurement_snapshot(
                                self._model,
                                "optimizer",
                                self._device,
                                step=self.global_step,
                                microbatch=batch_count,
                            )
                            measurement_snapshot(
                                self._model,
                                "steady_state",
                                self._device,
                                step=self.global_step,
                                microbatch=batch_count,
                            )
                        if _MOE_MEM_RATCHET_DEBUG and self._device.type == "xpu":
                            torch.xpu.synchronize()
                            _log_mem_ratchet_snapshot(
                                f"POST-OPTIM batch_count={batch_count}"
                            )

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    self.global_step += 1
                    inner_step_count += 1

                    if (
                        self._enable_fp8_training
                        and is_fp8_tensorwise_scaling(self._fp8_recipe_name)
                        and self.dp_degree > 1
                    ):
                        precompute_float8_dynamic_scale_for_fsdp(self._model)

                    loss_to_log = running_loss.item() / local_num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    time_per_step = None
                    if _MOE_STEP_TIMING_ENABLED and self._is_rank_zero:
                        time_per_step = time.perf_counter() - t0
                        _moe_timing_report_step(
                            log, prefix=f"moe_step_timing step={self.global_step}"
                        )
                        local_tokens_for_step = int(local_num_tokens.item())
                        global_tokens_for_step = int(num_tokens.item())
                        self._measurement_step_timings.append(
                            {
                                "step": self.global_step,
                                "time_per_step_s": time_per_step,
                                "total_step_s": time_per_step,
                                "local_tokens": local_tokens_for_step,
                                "global_tokens": global_tokens_for_step,
                                "tokens_per_second_per_gpu": (
                                    local_tokens_for_step / time_per_step
                                    if time_per_step > 0
                                    else 0.0
                                ),
                                "aggregate_tokens_per_second": (
                                    global_tokens_for_step / time_per_step
                                    if time_per_step > 0
                                    else 0.0
                                ),
                                **_moe_timing_step_record(),
                            }
                        )

                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        if time_per_step is None:
                            time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            # Raw per-step wall time: the engine-agnostic throughput
                            # metric used by the SFT MoE-vs-dense throughput comparison.
                            "time_per_step_s": time_per_step,
                            "tokens_per_second_per_gpu": (
                                num_tokens / self.parallel_dims.non_data_parallel_size
                            )
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    if self.global_step % self.save_every_n_steps == 0:
                        self.save_checkpoint(epoch=curr_epoch, full_tensors=False)

                    running_loss = 0
                    num_tokens = 0
                    local_num_tokens = 0
                    t0 = time.perf_counter()

                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and batch_count
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=False)

                self._profiler.step()
                batch_count += 1

                if (
                    self._run_val_every_n_steps is not None
                    and self.global_step % self._run_val_every_n_steps == 0
                ):
                    pbar.refresh()
                    self.validate()

            self.epochs_run += 1

        self._profiler.stop()

        self.save_checkpoint(epoch=self.total_epochs - 1, full_tensors=True)

    def cleanup(self) -> None:
        if os.environ.get("TORCHTUNE_MOE_MEASURE_OUTPUT"):
            from torchtune.modules.moe.measurement import export_model_measurements

            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            output = os.environ["TORCHTUNE_MOE_MEASURE_OUTPUT"]
            root, extension = os.path.splitext(output)
            if not extension:
                extension = ".json"
            export_model_measurements(
                self._model,
                f"{root}.rank{rank}{extension}",
                metadata={
                    "rank": rank,
                    "world_size": world_size,
                    "host": os.uname().nodename,
                    "job_id": os.environ.get("PBS_JOBID", "unknown"),
                    "local_rank": os.environ.get("LOCAL_RANK", "unknown"),
                    "device_type": self._device.type,
                    "device_index": self._device.index,
                    "ep_degree": self._expert_parallel_degree,
                    "pipeline_stage": self._pipeline_stage,
                    "global_step": self.global_step,
                    "sequence_length": self._sequence_length,
                    "model": self._measurement_model,
                    "checkpoint": self._measurement_checkpoint,
                    "batch_size": self._measurement_batch_size,
                    "microbatch_size": self._measurement_microbatch_size,
                    "gradient_accumulation_steps": self._measurement_gradient_accumulation,
                    "optimizer": self._measurement_optimizer,
                    "topology": {
                        "ep": self._expert_parallel_degree,
                        "pp": self._pipeline_parallel_degree,
                        "tp": self.tp_degree,
                        "dp_replicate": self.dp_degree,
                        "dp_shard": self.parallel_dims.dp_shard,
                    },
                    "source_revision": os.environ.get(
                        "TORCHTUNE_MOE_SOURCE_REVISION", "unknown"
                    ),
                    "uncommitted_change_state": os.environ.get(
                        "TORCHTUNE_MOE_UNCOMMITTED", "unknown"
                    ),
                    "environment_overrides": {
                        name: os.environ[name]
                        for name in sorted(os.environ)
                        if name.startswith("TORCHTUNE_")
                    },
                    "optimization_profile": os.environ.get(
                        "TORCHTUNE_MOE_OPTIMIZATION_PROFILE", "unknown"
                    ),
                    "routing_index_mode": os.environ.get(
                        "TORCHTUNE_MOE_ROUTING_INDEX_MODE", "unknown"
                    ),
                    "router_semantics": os.environ.get(
                        "TORCHTUNE_MOE_ROUTER_SEMANTICS",
                        (
                            "sigmoid_argsort_v1"
                            if "gemma" in self._measurement_model.lower()
                            else "probability_topk_v2"
                            if "qwen" in self._measurement_model.lower()
                            else "unknown"
                        ),
                    ),
                    "expert_execution_path": (
                        "grouped_mm"
                        if os.environ.get("TORCHTUNE_MOE_GROUPED_EXPERTS", "0")
                        == "1"
                        else (
                            "sequential"
                            if os.environ.get(
                                "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS", "0"
                            )
                            == "1"
                            else "padded_bmm"
                        )
                    ),
                    "device_health": os.environ.get(
                        "TORCHTUNE_MOE_DEVICE_HEALTH", "unknown"
                    ),
                    "gate_status": os.environ.get(
                        "TORCHTUNE_MOE_GATE_STATUS", "unknown"
                    ),
                    "semantic_completion": os.environ.get(
                        "TORCHTUNE_MOE_SEMANTIC_COMPLETION", "unknown"
                    ),
                    "measurement_completion": os.environ.get(
                        "TORCHTUNE_MOE_MEASUREMENT_COMPLETION", "pending"
                    ),
                    "measurement_window": {
                        "warmup_steps": int(
                            os.environ.get("TORCHTUNE_MOE_WARMUP_STEPS", "4")
                        ),
                        "measurement_steps": int(
                            os.environ.get("TORCHTUNE_MOE_MEASUREMENT_STEPS", "8")
                        ),
                        "steady_state_steps": int(
                            os.environ.get("TORCHTUNE_MOE_STEADY_STATE_STEPS", "4")
                        ),
                    },
                },
                step_timings=(
                    self._measurement_step_timings
                    if _MOE_STEP_TIMING_ENABLED
                    else None
                ),
            )
        if self._is_metric_rank:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneMoEDistributedXPU", cfg=cfg)
    recipe = FullFinetuneMoEDistributedXPU(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

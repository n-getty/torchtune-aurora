# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# XPU-adapted variant of recipes/lora_finetune_distributed.py for Intel
# Aurora (Max Series GPU / XCCL).  Same shim/diff pattern as
# recipes/dev/full_finetune_distributed_xpu.py.

import os
import sys
import time

from functools import partial
from typing import Any, Optional, Union
from warnings import warn

# -- XPU / XCCL compatibility shim ---------------------------------------------
_use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ and os.environ["ZE_AFFINITY_MASK"] != ""
_affinity_tiles = os.environ.get("ZE_AFFINITY_MASK", "").split(",") if _use_affinity_mask else []
_xpu_device_index = 0 if (len(_affinity_tiles) == 1) else int(os.environ.get("LOCAL_RANK", "0"))

import torch

import types as _types
import importlib.util as _imp_util

if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
    else:
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

import torchao  # noqa: F401

from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group
from torch.distributed.tensor import DTensor

from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.dev.rl.distributed import install_xpu_patches
from torchtune.modules.loss import SFTLoss
from torchtune.modules.peft import (
    AdapterModule,
    get_adapter_params,
    get_lora_module_names,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    DummyProfiler,
    PROFILER_KEY,
    VALID_BACKENDS_FOR_MEMORY_STATS,
    device_record_memory_history,
    get_xpu_distributed_backend,
    init_xpu_process_group,
    supports_memory_stats,
)
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from tqdm import tqdm

install_xpu_patches()


class LoRAFinetuneRecipeDistributedXPU(FTRecipeInterface):
    """
    XPU-adapted distributed LoRA finetuning recipe for dense transformer-based LLMs (Llama,
    Qwen, AuroraGPT) on Intel Aurora. Mirrors the upstream ``LoRAFinetuneRecipeDistributed``
    body but routes all device-specific ops (PG init, memory ops, barrier) through
    ``torchtune.training`` wrappers so the same recipe runs on CUDA or XPU.

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5.0 or later and will be
            enabled by default if an acceptable torch version is found. Activation offloading can be used in
            conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If world_size is 1
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA or XPU.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
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

        if os.environ.get("CCL_ATL_TRANSPORT") == "mpi":
            try:
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
            except ImportError:
                pass

        if device_type == "xpu":
            init_xpu_process_group(self.distributed_backend, device_index=_xpu_device_index)
        else:
            from torch.distributed import init_process_group as _init_pg
            _init_pg(self.distributed_backend)

        if device_type == "xpu" and cfg.get("force_math_sdpa", False):
            warn(
                "force_math_sdpa=True is a no-op on XPU (CUDA-only toggle). "
                "Set TORCHTUNE_USE_IPEX_VARLEN=1 for the validated XPU SDPA fast path."
            )

        self.world_size, self.rank = utils.get_world_size_and_rank()

        self._is_rank_zero = self.rank == 0

        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        assert (
            self.tp_degree == 1
        ), "Tensor parallelism is not supported in this recipe. Please set tensor_parallel_dim to 1."

        self.cp_degree = cfg.get("context_parallel_dim", 1)
        self.context_parallel_rotate_method = cfg.get(
            "context_parallel_rotate_method", "allgather"
        )
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            cp=self.cp_degree,
            world_size=self.world_size,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type=cfg.device)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        # Dataloader async-prefetch knobs. Default num_workers=2 so the collate (packed
        # block-causal mask build) + H2D overlap with compute instead of stalling the step
        # (~42% of step wasted at num_workers=0). Override via config. On Aurora keep
        # TMPDIR=/tmp (AF_UNIX worker-socket path length) when workers>0.
        self._dataloader_num_workers = cfg.get("dataloader_num_workers", 2)
        # pin_memory defaults to FALSE: the triple of {torch.compile model +
        # pinned-memory forked-worker batches + non-reentrant activation
        # checkpointing} raises `CheckpointError: A different number of tensors
        # was saved during the original forward and recomputation` at the step-0
        # backward on XPU (A/B-proven on the full-FT recipe: needs compile=True
        # AND pin_memory=True AND num_workers>0 together; clearing pin_memory
        # alone fixes it while keeping the async-collate throughput win;
        # compile_dynamic does NOT help). See
        # memory/project_sft_pinmem_compile_ac_checkpoint_error_20260621. Set
        # dataloader_pin_memory=true explicitly on a non-compile or non-AC run.
        self._dataloader_pin_memory = cfg.get("dataloader_pin_memory", False)
        self._dataloader_prefetch_factor = cfg.get("dataloader_prefetch_factor", 2)

        self.save_every_n_steps = cfg.get("save_every_n_steps")

        if (
            self._log_peak_memory_stats
            and self._device.type not in VALID_BACKENDS_FOR_MEMORY_STATS
        ):
            self._logger.info(
                f"log_peak_memory_stats was set to True; however, training device is not in {VALID_BACKENDS_FOR_MEMORY_STATS}."
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        # XPU multi-node: see full_finetune_distributed_xpu.py for explanation.
        # Only wrap on the single-tile-affinity multi-node path.
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

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)

        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self._run_val_every_n_steps = cfg.get("run_val_every_n_steps", None)
        if self._run_val_every_n_steps is not None:
            assert (
                cfg.get("dataset_val") is not None
            ), "run_val_every_n_steps is set but dataset_val is not provided"

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda" and self._device.type != "xpu":
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

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

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
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        if (
            cfg.checkpointer.model_type == "LLAMA4"
            and self._save_adapter_weights_only is False
        ):
            raise ValueError(
                "For Llama4 training, you should set save_adapter_weights_only to True."
            )

        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()

        self._compile = cfg.get("compile", False)
        # Capture scalar outputs is required to compile MoE
        torch._dynamo.config.capture_scalar_outputs = True

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if training.ADAPTER_KEY in checkpoint_dict
                else None
            ),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    checkpoint_dict = (
                        self._checkpoint_client.load_distributed_checkpoint(
                            self._model,
                            self._optimizer,
                            self._adapter_config,
                        )
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        utils.log_rank_zero(self._logger, "Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )

        # Setup validation dataloader if validation dataset is provided
        self._val_dataloader = None
        if cfg.get("dataset_val") is not None:
            batch_size_val = cfg.get("batch_size_val", cfg.batch_size)
            self._val_dataloader = self._setup_data(
                cfg_dataset=cfg.dataset_val,
                batch_size=batch_size_val,
                collate_fn=collate_name,
                shuffle=False,
            )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        self.checkpoint_dir_prefix = ""
        if self.save_every_n_steps is None:
            self.save_every_n_steps = self._steps_per_epoch
            self.checkpoint_dir_prefix = "epoch"
        else:
            self.checkpoint_dir_prefix = "step"

        # Learning rate scheduler can only be set up after number of steps
        # has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
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
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        base_model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        lora_weights_state_dict: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
           c. We register (pre-)forward hooks with ``fully_shard`` instead of wrapping `nn.Module`
        """

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)
        self._adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }

        utils.log_rank_zero(
            self._logger,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )

        if self.cp_degree > 1:
            utils.log_rank_zero(
                self._logger,
                f"CP is enabled with degree {self.cp_degree} and rotate method {self.context_parallel_rotate_method}.",
            )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        set_trainable_params(model, get_adapter_params(model))

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply Fully Sharded Data Parallelism to the model
        if self.parallel_dims.dp_shard_enabled or self.parallel_dims.cp_enabled:
            # For FSDP sharding
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]

            if self.parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_dim_names = ("dp_shard_cp",)

            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=self.world_mesh[dp_mesh_dim_names],
            )

        if lora_weights_state_dict:
            lora_missing, lora_unexpected = training.load_from_full_model_state_dict(
                model,
                lora_weights_state_dict,
                self._device,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            lora_missing, lora_unexpected = None, None

        # Initialize LoRA params and RoPE buffers
        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if (isinstance(m, AdapterModule)) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.to_empty(device=lora_device)
                    m.initialize_parameters()

                if hasattr(m, "rope_init"):
                    m.rope_init()

        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            base_model_state_dict,
            self._device,
            cpu_offload=fsdp_cpu_offload,
        )
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                m.initialize_dora_magnitude()

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            state_dict_keys=model.state_dict().keys(),
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # context parallel
        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            rotate_method=self.context_parallel_rotate_method,
            world_mesh=self.world_mesh,
            model=model,
        )

        # log
        utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(self._logger, "Optimizer is initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        utils.log_rank_zero(self._logger, "Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """
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

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.dp_degree,
            rank=self.dp_rank,
            shuffle=shuffle,
        )
        # Async data loading: with num_workers=0 (the historical default) the collate
        # (block-causal mask build for packed data) + H2D run synchronously on the main
        # process and DO NOT overlap with compute -> measured ~3.4s/step (~42%) of pure
        # dataloader wait on Aurora 4B/seq2048/packed. Make workers/pin/prefetch
        # configurable and default to async prefetch so the next batch is prepared during
        # the current step's backward. (Aurora: long node TMPDIR overflows the AF_UNIX
        # 108-char worker socket path -> launcher sets TMPDIR=/tmp.)
        num_workers = self._dataloader_num_workers
        dl_kwargs = dict(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                    cp_degree=self.cp_degree,
                    pad_to_multiple_of=self.parallel_dims.min_seq_len_divisor,
                )
                if not packed
                else padded_collate_packed
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            num_workers=num_workers,
            pin_memory=self._dataloader_pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = self._dataloader_prefetch_factor
        dataloader = StatefulDataLoader(**dl_kwargs)

        return dataloader

    def save_checkpoint(
        self,
        epoch: int,
        full_tensors: bool,
    ) -> None:
        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=self._optimizer,
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=self.epochs_run,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                dataloader_state_dict=self._dataloader.state_dict(),
            ),
            epoch=epoch,
            adapter_config=self._adapter_config.copy(),
            adapter_only=self._save_adapter_weights_only,
            full_tensors=full_tensors,
            dir_prefix=self.checkpoint_dir_prefix,
        )

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        # Optional per-phase step profiler (env-gated, off by default). Splits each
        # step into fwd+loss / backward / optimizer with device syncs so a slow step
        # can be attributed. Adds sync overhead -> diagnostic use only.
        self._phase_timer = os.environ.get("TORCHTUNE_PHASE_TIMER", "0") == "1"
        self._phase_acc = {
            "fwd_loss": 0.0, "fwd_only": 0.0, "loss_only": 0.0, "bwd": 0.0, "opt": 0.0,
            "data": 0.0,
        }

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not (self.rank == 0))
            self._dataloader.sampler.set_epoch(curr_epoch)
            if self._phase_timer:
                self._device_sync()
                _data_t = time.perf_counter()
            for idx, batch in enumerate(self._dataloader):
                if self._phase_timer:
                    # time spent waiting for this batch (dataloader/getitem/collate)
                    self._phase_acc["data"] = (
                        self._phase_acc.get("data", 0.0) + time.perf_counter() - _data_t
                    )
                # Start tracking device memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=True)

                utils.batch_to_device(batch, self._device)

                with self.context_parallel_manager(list(batch.values())):
                    # Calculate the number of unmasked tokens in the current batch
                    # and increment the total number of tokens seen in the step
                    current_num_tokens = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    num_tokens += current_num_tokens

                    # Loss is normalized by default so we multiply by the number of tokens
                    # This way we can normalize by the total number of tokens if we're accumulating gradients
                    if self._phase_timer:
                        self._device_sync()
                        _tp = time.perf_counter()
                    current_loss = self._loss_step(batch) * current_num_tokens
                    running_loss += current_loss
                    if self._phase_timer:
                        self._device_sync()
                        self._phase_acc["fwd_loss"] += time.perf_counter() - _tp
                        _tp = time.perf_counter()
                    # Gradient accumulation: suppress the FSDP/DDP gradient
                    # all-reduce/reduce-scatter on every micro-batch except the
                    # last of each accumulation window. With ga>1 this collapses
                    # `ga` cross-rank grad reductions per optimizer step down to
                    # one, the dominant comm cost at multi-node scale. The result
                    # is mathematically equivalent to syncing every micro-batch
                    # (grad reduction is linear), but not bit-identical (different
                    # summation order). HW-validated on the full-FT recipe
                    # (2.15x faster at 2N; see memory project_sft_no_sync_zero2_2n
                    # _validated_20260621). This loop has no inner micro-batch
                    # sub-loop, so the last-micro-batch predicate reuses the same
                    # (idx+1) % ga == 0 expression that gates the optimizer step.
                    #   - FSDP2 (fully_shard): set_requires_gradient_sync(is_last)
                    #   - FSDP1 / DDP: no_sync() context manager on non-last.
                    # ga==1 takes the plain backward (no-op gating). Opt-out via
                    # TORCHTUNE_SFT_DISABLE_NO_SYNC=1 (diagnostic, A/B only).
                    is_last_microbatch = (
                        (idx + 1) % self._gradient_accumulation_steps == 0
                    )
                    _no_sync_disabled = (
                        os.environ.get("TORCHTUNE_SFT_DISABLE_NO_SYNC", "0") == "1"
                    )
                    if (
                        self._gradient_accumulation_steps > 1
                        and not _no_sync_disabled
                        and hasattr(self._model, "set_requires_gradient_sync")
                    ):
                        # FSDP2 path
                        self._model.set_requires_gradient_sync(is_last_microbatch)
                        current_loss.backward()
                    elif (
                        self._gradient_accumulation_steps > 1
                        and not _no_sync_disabled
                        and not is_last_microbatch
                        and hasattr(self._model, "no_sync")
                    ):
                        # FSDP1 / DDP path
                        with self._model.no_sync():
                            current_loss.backward()
                    else:
                        # ga==1, diagnostic disable, or no FSDP hooks: plain backward.
                        if (
                            _no_sync_disabled
                            and hasattr(self._model, "set_requires_gradient_sync")
                        ):
                            self._model.set_requires_gradient_sync(True)
                        current_loss.backward()
                    if self._phase_timer:
                        self._device_sync()
                        self._phase_acc["bwd"] += time.perf_counter() - _tp

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if self._phase_timer:
                        self._device_sync()
                        _tp = time.perf_counter()
                    # Get total number of tokens across all ranks to normalize gradients
                    torch.distributed.all_reduce(num_tokens)
                    # This will ensure that the logged loss matches what we're optimizing
                    torch.distributed.all_reduce(running_loss)
                    # Manually scale the gradients from unnormalized loss by total # of tokens
                    # We multiply by world_size to undo FSDP2 gradient normalization.
                    training.scale_grads(self._model, self.world_size / num_tokens)
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        ).full_tensor()
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()
                    if self._phase_timer:
                        self._device_sync()
                        self._phase_acc["opt"] += time.perf_counter() - _tp

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    loss_to_log = running_loss.detach().item() / num_tokens
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            # Raw per-step wall time: the one engine-agnostic throughput
                            # metric (no token-masking / normalization assumptions), used
                            # by the SFT throughput benchmark for cross-engine A/B.
                            "time_per_step_s": time_per_step,
                            "tokens_per_second_per_gpu": num_tokens
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )

                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        if self._phase_timer:
                            log_dict.update(
                                {
                                    "phase_fwd_loss_s": self._phase_acc["fwd_loss"],
                                    "phase_fwd_only_s": self._phase_acc.get("fwd_only", 0.0),
                                    "phase_loss_only_s": self._phase_acc.get("loss_only", 0.0),
                                    "phase_bwd_s": self._phase_acc["bwd"],
                                    "phase_opt_s": self._phase_acc["opt"],
                                    "phase_data_s": self._phase_acc.get("data", 0.0),
                                }
                            )
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    if self._phase_timer:
                        self._phase_acc = {
                            "fwd_loss": 0.0, "fwd_only": 0.0, "loss_only": 0.0,
                            "bwd": 0.0, "opt": 0.0, "data": 0.0,
                        }
                    t0 = time.perf_counter()

                    # Stop tracking device memory now that active steps are complete
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
                        device_record_memory_history(self._device, enabled=False)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                    # If not last checkpoint
                    if (
                        self.global_step % self.save_every_n_steps == 0
                        and curr_epoch != self.total_epochs - 1
                    ):
                        self.save_checkpoint(epoch=curr_epoch, full_tensors=False)

                    # Run validation after gradient update
                    if (
                        self._run_val_every_n_steps is not None
                        and self.global_step % self._run_val_every_n_steps == 0
                    ):
                        pbar.refresh()
                        self.validate()

                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

                if self._phase_timer:
                    self._device_sync()
                    _data_t = time.perf_counter()

            self.epochs_run += 1

        self._profiler.stop()

        # Save final non-distributed ckpt
        self.save_checkpoint(epoch=curr_epoch, full_tensors=True)

    def _device_sync(self) -> None:
        """Block until queued device work completes (for accurate phase timing)."""
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        elif self._device.type == "cuda":
            torch.cuda.synchronize()

    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        _pt = getattr(self, "_phase_timer", False)
        if _pt:
            self._device_sync()
            _t = time.perf_counter()
        with self.activations_handling_ctx:
            outputs = self._model(**batch)

        # post process for third party loss functions
        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
            if isinstance(outputs, DTensor):
                outputs = outputs.full_tensor()

        if _pt:
            self._device_sync()
            self._phase_acc["fwd_only"] = (
                self._phase_acc.get("fwd_only", 0.0) + time.perf_counter() - _t
            )
            _t = time.perf_counter()
        # Compute loss
        loss = self._loss_fn(outputs, labels)
        if _pt:
            self._device_sync()
            self._phase_acc["loss_only"] = (
                self._phase_acc.get("loss_only", 0.0) + time.perf_counter() - _t
            )

        # free logits otherwise it peaks backward memory
        del outputs

        return loss

    def validate(self) -> dict[str, float]:
        """
        Run validation loop and return average validation loss.
        """

        self._model.eval()
        total_val_loss = torch.tensor(0.0, device=self._device)
        total_val_tokens = torch.tensor(0.0, device=self._device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._val_dataloader):
                utils.batch_to_device(batch, self._device)

                # Count tokens excluding padding
                current_num_tokens = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()

                # Compute loss
                val_loss = self._loss_step(batch) * current_num_tokens

                total_val_loss += val_loss
                total_val_tokens += current_num_tokens

        # Aggregate validation metrics across all ranks
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

    def cleanup(self) -> None:
        if self._is_rank_zero:
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
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="LoRAFinetuneRecipeDistributedXPU", cfg=cfg)

    recipe = LoRAFinetuneRecipeDistributedXPU(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

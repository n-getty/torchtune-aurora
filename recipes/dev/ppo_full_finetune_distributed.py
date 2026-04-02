# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Distributed PPO full-finetune recipe adapted for Intel XPU (Aurora HPC).
#
# Key features:
# - FSDP sharding of 4 models (policy, reference, value, reward)
# - Device-agnostic memory ops (XPU / CUDA / CPU)
# - Gradient accumulation with no_sync() on policy and value models
# - XPU-safe distributed init (no device_id for XCCL)
# - Production mode sync skipping (FSDP_PRODUCTION_MODE env var)

import contextlib
import math
import os
import sys
import time
from functools import partial
from itertools import chain
from typing import Any, Optional, Union
from warnings import warn

# -- XPU / XCCL compatibility shim ---------------------------------------------
# On Intel XPU (Aurora), running torchtune's ``__init__.py`` corrupts the L0 USM
# pointer table, causing subsequent XCCL collectives to fail.
# Workaround: pre-register ``torchtune`` in ``sys.modules`` before importing
# submodules, and set ``ZE_AFFINITY_MASK`` for per-tile affinity.
# --------------------------------------------------------------------------

# 1. Tile affinity — must happen before any GPU runtime init
if os.environ.get("ZE_AFFINITY_MASK") is None and os.environ.get("LOCAL_RANK") is not None:
    os.environ["ZE_AFFINITY_MASK"] = os.environ["LOCAL_RANK"]

import torch

# 2. Pre-register torchtune package to bypass its __init__.py on XPU
import importlib.util as _imp_util
import types as _types

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

# 3. Ensure torchao is available
import torchao  # noqa

from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf import PPOStats, Trajectory
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



class PPOFullFinetuneDistributedRecipe(FTRecipeInterface):
    """
    Distributed PPO full-finetune recipe with FSDP sharding of all 4 models.

    Scales the single-device PPO recipe to multi-rank/node training. Supports
    both CUDA (NCCL) and XPU (XCCL) backends.

    4-model FSDP sharding:
        - Policy: trainable, reshard_after_forward=False (needs unsharded params for generation)
        - Reference: frozen, reshard_after_forward=True
        - Value: trainable, reshard_after_forward=True
        - Reward: frozen, reshard_after_forward=True

    Gradient accumulation uses ``no_sync()`` on both policy and value models
    during non-final micro-batch backward passes, deferring AllReduce to the
    last micro-batch only.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # With ZE_AFFINITY_MASK each rank sees only its tile as xpu:0
        if cfg.device == "xpu" and os.environ.get("ZE_AFFINITY_MASK") is not None:
            self._device = torch.device("xpu:0")
            torch.xpu.set_device(0)
        else:
            self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise RuntimeError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and not supports_memory_stats(self._device):
            self._log_peak_memory_stats = False

        # Initialize distributed
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = get_xpu_distributed_backend(
            self._device.type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )
        if not torch.distributed.is_initialized():
            init_xpu_process_group(self.distributed_backend)
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        # Production mode: skip non-essential synchronize() calls
        self._production_mode = (
            os.environ.get("FSDP_PRODUCTION_MODE", "0") == "1"
        )

        # Training config
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._compile = cfg.get("compile", False)

        # Warn about compile + multi-node on XPU
        if self._compile and self._device.type == "xpu":
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
            if self.world_size > local_world_size:
                log.warning(
                    "torch.compile is not supported multi-node on XPU. Disabling."
                )
                self._compile = False

        # Gradient accumulation
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # Recipe state
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        # torch.Generator does not support XPU — use CPU generator instead
        _rng_device = self._device if self._device.type == "cuda" else torch.device("cpu")
        self._rng = torch.Generator(_rng_device).manual_seed(self.seed)
        self._total_steps = 0
        self._steps_run = 0
        self._total_epochs = 0
        self._epochs_run = 0
        self.global_step = 0

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up recipe state: models, optimizer, dataloader, lr scheduler, profiler.
        """
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        # Setup checkpointers and load checkpoints
        (
            self._policy_checkpointer,
            ref_policy_checkpointer,
            self._value_checkpointer,
            reward_checkpointer,
        ) = self._setup_checkpointers(
            cfg.checkpointer,
            cfg.ref_policy_checkpointer,
            cfg.value_checkpointer,
            cfg.reward_checkpointer,
        )

        policy_model_checkpoint_dict = self._policy_checkpointer.load_checkpoint()
        ref_policy_state_dict = ref_policy_checkpointer.load_checkpoint()
        value_model_checkpoint_dict = self._value_checkpointer.load_checkpoint()
        reward_model_state_dict = reward_checkpointer.load_checkpoint()

        self._compile = cfg.compile

        (
            self._policy_model,
            self._value_model,
            self._reward_model,
            self._ref_policy_model,
        ) = self._setup_models(
            cfg_model=cfg.policy_model,
            cfg_reward_value_model=cfg.reward_and_value_model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            compile_model=self._compile,
            policy_state_dict=policy_model_checkpoint_dict[training.MODEL_KEY],
            ref_policy_state_dict=ref_policy_state_dict[training.MODEL_KEY],
            value_model_state_dict=value_model_checkpoint_dict[training.MODEL_KEY],
            reward_model_state_dict=reward_model_state_dict[training.MODEL_KEY],
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
        )

        torch.distributed.barrier()

        # Tokenizer
        self._tokenizer = config.instantiate(cfg.tokenizer)
        utils.log_rank_zero(log, "Tokenizer is initialized from file.")

        # Optimizer — over both policy and value parameters
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                policy_model_checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # Loss
        self._loss_fn = config.instantiate(cfg.loss)
        utils.log_rank_zero(log, "Loss is initialized.")

        # Dataloader
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        self._setup_training_parameters(cfg)
        self._setup_training_hyperparameters(cfg)

        # KV cache context for trajectory generation
        self.cache_ctx_manager = lambda enable_kv_cache, decoder_max_seq_len: (
            local_kv_cache(
                self._policy_model,
                batch_size=self._forward_batch_size,
                dtype=self._dtype,
                device=self._device,
                decoder_max_seq_len=decoder_max_seq_len,
            )
            if enable_kv_cache
            else contextlib.nullcontext()
        )

        if self._resume_from_checkpoint:
            self._update_recipe_state(policy_model_checkpoint_dict)

        # One "step" is a single gradient update over a minibatch of trajectories
        self.global_step = (
            self._steps_run
            * self._ppo_epochs
            * (self.batch_size // self._ppo_batch_size)
        )

        lr_steps = (
            self._total_steps
            * self._ppo_epochs
            * (self.batch_size // self._ppo_batch_size)
        )

        # LR scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=lr_steps,
            last_epoch=self.global_step - 1,
        )

        # Profiler
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

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
            )

        profiler, profiler_cfg = config.instantiate(cfg_profiler)
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

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

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        utils.log_rank_zero(log, "Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_training_hyperparameters(self, cfg) -> None:
        self._kl_coeff = cfg.kl_coeff
        self._gamma = cfg.gamma
        self._lmbda = cfg.lmbda
        self._whiten_rewards = cfg.whiten_rewards

        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens

        self._min_response_length = cfg.min_response_length
        self._penalise_no_eos = cfg.penalise_no_eos
        self._reward_penalty = cfg.reward_penalty

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
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

    def _setup_training_parameters(self, cfg: DictConfig) -> None:
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )
        self.enable_kv_cache = cfg.enable_kv_cache

        if self.batch_size % self._forward_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"forward_batch_size ({self._forward_batch_size})."
            )
        if self.batch_size % self._ppo_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"ppo_batch_size ({self._ppo_batch_size})."
            )
        if self._ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self._ppo_batch_size}) must be exactly divisible "
                f"by gradient_accumulation_steps ({self._gradient_accumulation_steps})."
            )

        self._total_steps = cfg.num_steps // self.batch_size
        batches_per_epoch = max(1, len(self._dataloader))

        self._total_epochs = math.ceil(self._total_steps / batches_per_epoch)
        if self._total_steps == 0:
            raise ValueError(
                f"num_steps {cfg.num_steps} must be greater than the batch size {self.batch_size}."
            )

        utils.log_rank_zero(
            log,
            f"Total steps to run: {self._total_steps}, Total epochs to run: {self._total_epochs}",
        )

    def _setup_checkpointers(
        self,
        policy_cfg: DictConfig,
        ref_policy_cfg: DictConfig,
        value_cfg: DictConfig,
        reward_cfg: DictConfig,
    ) -> tuple:
        if not self._resume_from_checkpoint:
            assert policy_cfg.checkpoint_dir == ref_policy_cfg.checkpoint_dir, (
                "Policy and reference policy should be loaded from the same checkpoint directories"
            )
            assert policy_cfg.checkpoint_files == ref_policy_cfg.checkpoint_files, (
                "Policy and reference policy should be loaded from the same checkpoint files"
            )

        policy_checkpointer = config.instantiate(
            policy_cfg,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        ref_policy_checkpointer = config.instantiate(
            ref_policy_cfg,
            should_load_recipe_state=False,
        )
        value_checkpointer = config.instantiate(
            value_cfg,
            should_load_recipe_state=False,
        )
        reward_checkpointer = config.instantiate(
            reward_cfg,
            should_load_recipe_state=False,
        )

        return (
            policy_checkpointer,
            ref_policy_checkpointer,
            value_checkpointer,
            reward_checkpointer,
        )

    def _setup_model_fsdp(
        self,
        model: nn.Module,
        model_state_dict: dict[str, Any],
        eval_mode: bool = False,
        reshard_after_forward: bool = True,
        custom_sharded_layers: Optional[list[str]] = None,
    ) -> nn.Module:
        """
        Initialize a single model with FSDP sharding.

        Model is instantiated on meta device, then FSDP-sharded, then state dict loaded.
        """
        if eval_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # FSDP shard conditions
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        # Initialize RoPE on device
        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # Load sharded state dict
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=self.fsdp_cpu_offload,
        )

        training.validate_no_params_on_meta_device(model)
        return model

    def _setup_models(
        self,
        cfg_model: DictConfig,
        cfg_reward_value_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        policy_state_dict: dict[str, Any],
        ref_policy_state_dict: dict[str, Any],
        value_model_state_dict: dict[str, Any],
        reward_model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        """
        Sets up and shards all 4 models with FSDP.
        """
        utils.log_rank_zero(
            log,
            "Setting up 4 models (policy, value, reward, reference) with FSDP...",
        )
        init_start = time.perf_counter()

        # Instantiate all 4 models on meta device
        with training.set_default_dtype(self._dtype), torch.device("meta"):
            policy_model = config.instantiate(cfg_model)
            ref_policy_model = config.instantiate(cfg_model)
            reward_model = config.instantiate(cfg_reward_value_model)
            value_model = config.instantiate(cfg_reward_value_model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                policy_model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )
            training.set_activation_checkpointing(
                value_model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Update classifier state dicts
        training.update_state_dict_for_classifier(
            reward_model_state_dict, reward_model.named_parameters()
        )
        training.update_state_dict_for_classifier(
            value_model_state_dict, value_model.named_parameters()
        )

        # FSDP shard each model with appropriate settings
        # Policy: reshard_after_forward=False (generation needs unsharded params)
        policy_model = self._setup_model_fsdp(
            policy_model,
            policy_state_dict,
            eval_mode=False,
            reshard_after_forward=False,
            custom_sharded_layers=custom_sharded_layers,
        )

        # Reference: frozen, reshard after forward
        ref_policy_model = self._setup_model_fsdp(
            ref_policy_model,
            ref_policy_state_dict,
            eval_mode=True,
            reshard_after_forward=True,
            custom_sharded_layers=custom_sharded_layers,
        )

        # Value: trainable, reshard after forward
        value_model = self._setup_model_fsdp(
            value_model,
            value_model_state_dict,
            eval_mode=False,
            reshard_after_forward=True,
            custom_sharded_layers=custom_sharded_layers,
        )

        # Reward: frozen, reshard after forward
        reward_model = self._setup_model_fsdp(
            reward_model,
            reward_model_state_dict,
            eval_mode=True,
            reshard_after_forward=True,
            custom_sharded_layers=custom_sharded_layers,
        )

        # Disable dropout in trainable models
        disable_dropout(policy_model)
        disable_dropout(value_model)

        utils.log_rank_zero(
            log,
            f"All 4 models initialized with FSDP in {time.perf_counter() - init_start:.2f}s, "
            f"precision {self._dtype}",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return policy_model, value_model, reward_model, ref_policy_model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optimizer:
        # Optimize over both policy and value model parameters
        optimizer = config.instantiate(
            cfg_optimizer,
            chain(self._policy_model.parameters(), self._value_model.parameters()),
        )
        if opt_state_dict:
            # For FSDP, load sharded optimizer state
            training.load_from_full_optimizer_state_dict(
                self._policy_model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self, cfg_dataset: DictConfig, shuffle: bool, batch_size: int
    ) -> StatefulDataLoader:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=self._tokenizer)

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=partial(
                padded_collate,
                pad_direction="left",
                keys_to_pad=["tokens", "labels"],
                padding_idx=self._tokenizer.pad_id,
            ),
        )

        return dataloader

    def save_checkpoint(
        self, epoch: int, is_intermediate_checkpoint: bool = False
    ) -> None:
        """Save policy and value model checkpoints using FSDP gather."""
        utils.log_rank_zero(
            log, "Saving checkpoint. Gathering full model state dicts..."
        )
        start = time.perf_counter()

        # Gather policy model state dict
        policy_cpu_state_dict = training.gather_cpu_state_dict(
            self._policy_model,
            self._is_rank_zero,
            device=self._device,
        )

        # Gather value model state dict
        value_cpu_state_dict = training.gather_cpu_state_dict(
            self._value_model,
            self._is_rank_zero,
            device=self._device,
        )

        if is_intermediate_checkpoint:
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._policy_model,
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
        else:
            opt_state_dict = None

        if self._is_rank_zero:
            policy_ckpt_dict = {training.MODEL_KEY: policy_cpu_state_dict}
            value_ckpt_dict = {training.MODEL_KEY: value_cpu_state_dict}

            if is_intermediate_checkpoint:
                policy_ckpt_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self._epochs_run,
                        training.TOTAL_EPOCHS_KEY: self._total_epochs,
                        training.MAX_STEPS_KEY: self._total_steps,
                        training.STEPS_KEY: self._steps_run,
                        training.RNG_KEY: self._rng.get_state(),
                        training.DATALOADER_KEY: self._dataloader.state_dict(),
                    }
                )

            self._policy_checkpointer.save_checkpoint(
                policy_ckpt_dict,
                epoch=epoch,
                intermediate_checkpoint=is_intermediate_checkpoint,
            )
            self._value_checkpointer.save_checkpoint(
                value_ckpt_dict,
                epoch=epoch,
                intermediate_checkpoint=False,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        torch.distributed.barrier()

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        try:
            if (
                self.seed != ckpt_dict[training.SEED_KEY]
                or self._total_steps != ckpt_dict[training.MAX_STEPS_KEY]
                or self._total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]
            ):
                warn(
                    message="Configured value for seed, total_steps, or total_epochs "
                    "does not match the value stored in checkpoint."
                )
            self.seed = training.set_seed(seed=ckpt_dict[training.SEED_KEY])
            self._rng.set_state(ckpt_dict[training.RNG_KEY])
            self._steps_run = ckpt_dict[training.STEPS_KEY]
            self._total_steps = ckpt_dict[training.MAX_STEPS_KEY]
            self._total_epochs = ckpt_dict[training.TOTAL_EPOCHS_KEY]
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state."
            ) from e

    def generate_trajectory(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generate a trajectory: responses, logprobs, ref logprobs, values, rewards.
        """
        _, context_length = input_ids.shape

        # step 1: generate responses using policy model
        with self.cache_ctx_manager(
            self.enable_kv_cache,
            decoder_max_seq_len=context_length + self._max_generated_tokens,
        ):
            query_responses, logits = generation.generate(
                model=self._policy_model,
                prompt=input_ids,
                max_generated_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k,
                pad_id=self._tokenizer.pad_id,
                rng=self._rng if self._device.type == "cuda" else None,
            )
        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # step 2: estimate logprobs from current policy
        logprobs = rlhf.logits_to_logprobs(logits, responses, self._temperature)
        del logits
        device_empty_cache(self._device)

        # step 2.1: estimate logprobs from reference policy
        ref_logits = self._ref_policy_model(
            query_responses, input_pos=position_ids, mask=masks
        )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
        ref_logprobs = rlhf.logits_to_logprobs(ref_logits, responses, self._temperature)
        del ref_logits
        device_empty_cache(self._device)

        # step 3: estimate values from value model
        values = self._value_model(query_responses, input_pos=position_ids, mask=masks)
        values = rlhf.truncate_sequence_for_logprobs(values, context_length).squeeze(-1)
        device_empty_cache(self._device)

        # step 4: truncate responses at first stop token
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # step 5: run reward model on (query, truncated-response) pairs
        scores = self._reward_model(
            torch.cat([input_ids, responses], dim=1),
            input_pos=position_ids,
            mask=masks,
        )
        del responses
        device_empty_cache(self._device)

        # step 5.1: extract reward scores
        seq_lens = training.get_unmasked_sequence_lengths(response_padding_masks)
        scores = scores.gather(1, (seq_lens + context_length)[:, None, None]).squeeze(
            (-1, -2)
        )

        # step 5.2: apply penalties
        if self._penalise_no_eos or self._min_response_length:
            reward_penalty_mask = rlhf.get_reward_penalty_mask(
                response_padding_masks,
                seq_lens,
                self._penalise_no_eos,
                self._min_response_length,
            )
            scores[reward_penalty_mask] = self._reward_penalty

        # step 6: mask invalid values
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        value_seq_idxs = torch.where(
            (seq_lens > 0) & (seq_lens < self._max_generated_tokens - 1),
            seq_lens + 1,
            seq_lens,
        )
        value_padding_masks = response_padding_masks.clone()
        value_padding_masks = value_padding_masks.scatter_(
            1, value_seq_idxs.unsqueeze(-1), False
        )
        values[value_padding_masks] = 0.0

        return Trajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            value_padding_masks=value_padding_masks,
            value_seq_idxs=value_seq_idxs,
            scores=scores,
            seq_lens=seq_lens,
        )

    def generate_trajectory_batched(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generate trajectories using forward_batch_size micro-batches.
        """
        trajectories: list[Trajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                device_empty_cache(self._device)
                trajectories.append(self.generate_trajectory(batch_input_ids))
                device_empty_cache(self._device)
        return Trajectory(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        """
        The core training loop with distributed gradient accumulation.
        """
        training.cleanup_before_training()
        self._optimizer.zero_grad()

        grad_norm = None
        training_completed = False
        self._profiler.start()
        pbar = tqdm(total=self._total_steps, initial=self._steps_run, disable=not self._is_rank_zero)

        for curr_epoch in range(self._epochs_run, self._total_epochs):
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                # Start tracking memory
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=True)

                batch = batch["tokens"].to(self._device)
                _, context_length = batch.shape
                num_tokens = batch.numel()

                # step 1: generate trajectory
                t0_traj = time.perf_counter()
                trajectory = self.generate_trajectory_batched(batch)
                traj_time = time.perf_counter() - t0_traj

                if not self._production_mode:
                    torch.distributed.barrier()

                # step 2: get rewards
                rewards, kl, kl_rewards = rlhf.get_rewards_ppo(
                    trajectory.scores,
                    trajectory.logprobs,
                    trajectory.ref_logprobs,
                    self._kl_coeff,
                    trajectory.value_seq_idxs,
                )

                # step 3: estimate advantages using GAE
                advantages, returns = rlhf.estimate_advantages(
                    trajectory.values,
                    rewards,
                    self._gamma,
                    self._lmbda,
                    masks=~trajectory.response_padding_masks,
                )

                # step 4: PPO optimization over multiple epochs
                t0_ppo = time.perf_counter()
                ppo_stats: list[PPOStats] = []
                for _ in range(self._ppo_epochs):
                    batch_idxs = torch.randperm(self.batch_size, device=self._device)
                    for i in range(0, self.batch_size, self._ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self._ppo_batch_size]

                        batch_ppo_stats: list[PPOStats] = []
                        for j in range(
                            0, self._ppo_batch_size, self._ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self._ppo_backward_batch_size
                            ]

                            batch_trajectory = Trajectory(
                                *map(
                                    partial(
                                        torch.index_select,
                                        dim=0,
                                        index=backward_batch_idxs,
                                    ),
                                    trajectory,
                                )
                            )

                            # Determine if this is the last micro-batch
                            is_last_micro = (
                                j + self._ppo_backward_batch_size >= self._ppo_batch_size
                            )

                            # Use no_sync() on non-final micro-batches for both models
                            if self._gradient_accumulation_steps > 1 and not is_last_micro:
                                policy_sync_ctx = self._policy_model.no_sync()
                                value_sync_ctx = self._value_model.no_sync()
                            else:
                                policy_sync_ctx = contextlib.nullcontext()
                                value_sync_ctx = contextlib.nullcontext()

                            with policy_sync_ctx, value_sync_ctx:
                                batch_ppo_stats.append(
                                    self.ppo_step(
                                        batch_trajectory,
                                        advantages[backward_batch_idxs],
                                        returns[backward_batch_idxs],
                                        context_length,
                                    )
                                )
                            del batch_trajectory

                        ppo_stats.append(PPOStats(*map(sum, zip(*batch_ppo_stats))))

                        if not self._production_mode:
                            torch.distributed.barrier()

                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                chain(
                                    self._policy_model.parameters(),
                                    self._value_model.parameters(),
                                ),
                                max_norm=float(self._clip_grad_norm),
                            )

                        if self._lr_scheduler is not None:
                            self._lr_scheduler.step()
                        self.global_step += 1

                ppo_time = time.perf_counter() - t0_ppo

                current_lr = get_lr(self._optimizer)

                # step 5: log and cleanup
                self._steps_run += 1
                if self._steps_run % self._log_every_n_steps == 0:
                    self.log_metrics(
                        trajectory,
                        PPOStats(*map(torch.stack, zip(*ppo_stats))),
                        kl,
                        kl_rewards,
                        num_tokens / traj_time,
                        num_tokens / ppo_time,
                        current_lr,
                        grad_norm,
                    )
                self.cleanup_after_step(
                    trajectory, ppo_stats, advantages, returns, kl, kl_rewards
                )
                pbar.update(1)

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

                self._profiler.step()

                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            self._epochs_run += 1
            self.save_checkpoint(
                curr_epoch, is_intermediate_checkpoint=not training_completed
            )
            if training_completed:
                self._profiler.stop()
                return

        self._profiler.stop()

    def ppo_step(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        context_length: int,
    ) -> PPOStats:
        """
        Single PPO optimization step over a micro-batch.
        """
        device_empty_cache(self._device)

        # estimate logprobs from policy at current optimization step
        pi_logits = self._policy_model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )
        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        pi_logprobs = rlhf.logits_to_logprobs(
            pi_logits, trajectory.query_responses[:, context_length:], self._temperature
        )
        pi_logprobs[trajectory.response_padding_masks] = 1.0
        del pi_logits
        device_empty_cache(self._device)

        # estimate values from value model at current optimization step
        phi_values = self._value_model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )
        phi_values = rlhf.truncate_sequence_for_logprobs(
            phi_values, context_length
        ).squeeze(-1)
        phi_values[trajectory.value_padding_masks] = 0.0

        # calculate PPO loss
        loss, policy_loss, value_loss, ratios, clipfrac = self._loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            advantages,
            trajectory.values,
            phi_values,
            returns,
            padding_masks=~trajectory.response_padding_masks,
            value_padding_masks=~trajectory.value_padding_masks,
        )

        loss /= self._gradient_accumulation_steps
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return PPOStats(
            loss,
            policy_loss / self._gradient_accumulation_steps,
            value_loss / self._gradient_accumulation_steps,
            ratios / self._gradient_accumulation_steps,
            clipfrac / self._gradient_accumulation_steps,
            approx_policy_kls / self._gradient_accumulation_steps,
        )

    def log_metrics(
        self,
        trajectory: Trajectory,
        ppo_stats: PPOStats,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
        tokens_per_second_trajectory: torch.Tensor,
        tokens_per_second_loss: torch.Tensor,
        lr: float,
        grad_norm: Optional[torch.Tensor] = None,
    ) -> None:
        # Reduce key metrics across ranks
        scores_mean = trajectory.scores.mean()
        torch.distributed.reduce(scores_mean, dst=0, op=torch.distributed.ReduceOp.SUM)
        scores_mean /= self.world_size

        log_dict = {
            "scores": scores_mean,
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "rlhf_reward": trajectory.scores.mean() + kl_rewards.sum(1).mean(),
            "kl": kl.sum(1).mean(),
            "kl_reward": kl_rewards.sum(1).mean(),
            "lr": lr,
            "loss": ppo_stats.loss.mean(),
            "policy_loss": ppo_stats.policy_loss.mean(),
            "value_loss": ppo_stats.value_loss.mean(),
            "clipfrac": ppo_stats.clipfrac.mean(),
            "ratios": ppo_stats.ratios.mean(),
            "approx_policy_kl": ppo_stats.approx_policy_kls.mean(),
            "response_lengths": trajectory.seq_lens.float().mean(),
            "tokens_per_second_per_gpu_trajectory": tokens_per_second_trajectory,
            "tokens_per_second_per_gpu_ppo": tokens_per_second_loss,
        }

        if grad_norm is not None:
            log_dict["grad_norm"] = grad_norm

        if supports_memory_stats(self._device) and self._log_peak_memory_stats:
            log_dict.update(training.get_memory_stats(device=self._device))

        if self._is_rank_zero:
            self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup_after_step(
        self,
        trajectory: Trajectory,
        ppo_stats: PPOStats,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        for v in trajectory:
            del v
        del trajectory
        for v in ppo_stats:
            del v
        del ppo_stats
        del advantages
        del returns
        del kl
        del kl_rewards

    def cleanup(self, **kwargs) -> None:
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
    recipe = PPOFullFinetuneDistributedRecipe(cfg=cfg)
    config.log_config(recipe_name="PPOFullFinetuneDistributedRecipe", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

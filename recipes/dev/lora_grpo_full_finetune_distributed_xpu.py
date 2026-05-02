# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LoRA-GRPO recipe for Aurora/XPU — sibling to grpo_full_finetune_distributed_xpu.py.
#
# Key differences from the full-FT base recipe:
#   - Trainable surface: LoRA adapter only (base weights frozen).
#   - No separate ref model: uses disable_adapter() context instead (~8 GiB saved).
#   - Weight sync: PEFT adapter dir written to shared FS, POSTed to vLLM HTTP servers.
#   - FSDP: FSDP1 SHARD_GRAD_OP (top-level only, BioReason-validated topology).
#   - Mode: server mode only (vLLM on a separate node via HTTP API).
#
# See docs/features/lora_grpo_primer.md and the plan in:
#   /home/ngetty/.claude/plans/mossy-enchanting-wolf.md

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from warnings import warn

# -- XPU / XCCL compatibility shim (mirrors base recipe) ----------------------
# Pre-register torchtune package to prevent torchtune.__init__.py from running
# while an XCCL PG is active (corrupts L0 USM pointer table).

_use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ and os.environ["ZE_AFFINITY_MASK"] != ""
_affinity_tiles = os.environ.get("ZE_AFFINITY_MASK", "").split(",") if _use_affinity_mask else []
_xpu_device_index = 0 if (len(_affinity_tiles) == 1) else int(os.environ.get("LOCAL_RANK", "0"))

import torch  # noqa: E402

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

import torchao  # noqa: E402

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
from torchtune.modules.attention_utils import _compute_maskfree_causal
from torchtune.modules.peft import (
    disable_adapter,
    get_adapter_params,
    get_adapter_state_dict,
    set_trainable_params,
)
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
    device_empty_cache,
)
from torchtune.dev.rl.lora_helpers import (
    build_qwen3_lora_model,
    adapter_optimizer_params,
    torchtune_to_peft_state_dict,
    write_peft_adapter_dir,
    load_lora_adapter_http,
    unload_lora_adapter_http,
    load_peft_adapter_into_model,
    _ATTN_TARGET_MODULES,
    _MLP_TARGET_MODULES,
)
import torchtune.dev.rl.vllm_backend as _vllm_backend_module
from tqdm import tqdm

log = utils.get_logger("DEBUG")
install_xpu_patches()


class LoRAGRPODistributedXPU(FTRecipeInterface):
    """
    LoRA-GRPO recipe for Intel XPU (Aurora HPC) — server mode, FSDP1.

    Implements the BioReason-Pro paper's LoRA training hyperparameters
    (r=16, lr=3e-5, KL β=1e-4, G=24, T=1.0) on the torchtune/XPU stack.

    Architecture decisions:
      - Adapter surface only (7 modules × n_layers × 2 LoRA mats).
      - disable_adapter() replaces a separate ref model copy.
      - FSDP1 SHARD_GRAD_OP: BioReason-validated, no per-module wrap overhead.
      - Weight sync: rank 0 gathers adapter SD, writes PEFT dir to Lustre,
        POSTs /v1/load_lora_adapter to all vLLM HTTP tiles in parallel.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # Device + dtype
        if cfg.device == "xpu":
            self._device = torch.device(f"xpu:{_xpu_device_index}")
            torch.xpu.set_device(_xpu_device_index)
        else:
            self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        self._output_dir = cfg.output_dir

        # Logging
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and not supports_memory_stats(self._device):
            self._log_peak_memory_stats = False

        # Distributed
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = get_xpu_distributed_backend(
            self._device.type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        # MPI pre-init (required for CCL_ATL_TRANSPORT=mpi multi-node)
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

        # SDPA — force_math_sdpa is a no-op on XPU (CUDA-only toggle).
        if self._device.type == "xpu" and cfg.get("force_math_sdpa", False):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        # Production mode: skip non-essential barriers (required for multi-node XPU).
        _is_multinode = self.world_size > int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self._production_mode = (
            os.environ.get("FSDP_PRODUCTION_MODE", "0") == "1"
            or (_is_multinode and self._device.type == "xpu")
        )

        # Training attrs
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get("enable_activation_checkpointing", False)
        self._compile = cfg.get("compile", False)
        if self._compile:
            log.warning("compile=True: LoRA recipe uses FSDP1 — torch.compile is experimental here.")

        # Gradient accumulation
        self._gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

        # LoRA config (from cfg.lora sub-dict)
        _lora_cfg = cfg.get("lora", {}) or {}
        _get = _lora_cfg.get if hasattr(_lora_cfg, "get") else lambda k, d=None: d
        self._lora_publish_every = int(_get("publish_every_steps", 1))
        self._lora_max_loras = int(_get("max_loras", 2))
        # Shared FS root for PEFT adapter dirs (must be cross-node visible).
        # /dev/shm is node-local — use Lustre for 2-node setups.
        self._lora_shm_root = _get("shm_root", str(Path(cfg.output_dir) / "adapters"))
        self._lora_adapter_id = 0       # monotonically increasing per-step adapter slot
        self._prev_lora_name: Optional[str] = None
        self._prev_lora_path: Optional[str] = None
        self._current_lora_name: Optional[str] = None  # set after first _publish_lora_to_vllm

        # Async publish: background thread for IO + HTTP after FSDP gather
        self._publish_thread: Optional[threading.Thread] = None
        self._publish_error: Optional[Exception] = None

        # tmpfs transfer: write to /dev/shm locally, rsync to VLLM_NODE /dev/shm
        self._lora_tmpfs_transfer = bool(_get("tmpfs_transfer", False))
        self._lora_train_shm = str(_get("train_shm", "/dev/shm/lora_adapters"))
        self._lora_vllm_shm = str(_get("vllm_shm", "/dev/shm/lora_adapters"))
        # SSH ControlMaster socket for rsync multiplexing (avoids per-rsync conn overhead)
        self._ssh_control_socket: Optional[str] = None
        self._ssh_control_proc: Optional[subprocess.Popen] = None

        # Derive LoRA target_modules from model config
        _model_cfg = cfg.get("model", {}) or {}
        _mc_get = _model_cfg.get if hasattr(_model_cfg, "get") else lambda k, d=None: d
        _attn_mods = list(_mc_get("lora_attn_modules", ["q_proj", "k_proj", "v_proj", "output_proj"]))
        _apply_mlp = bool(_mc_get("apply_lora_to_mlp", True))
        # Translate torchtune attn module names to HF names
        _attn_hf = {
            "q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "output_proj": "o_proj"
        }
        self._lora_target_modules = [_attn_hf.get(m, m) for m in _attn_mods]
        if _apply_mlp:
            self._lora_target_modules += list(_MLP_TARGET_MODULES)
        self._lora_rank = int(_mc_get("lora_rank", 16))
        self._lora_alpha = float(_mc_get("lora_alpha", 32.0))

        # vLLM server mode (only supported mode for LoRA recipe)
        self._vllm_mode = cfg.get("vllm_mode", "server")
        if self._vllm_mode != "server":
            raise ValueError(
                f"LoRAGRPODistributedXPU only supports vllm_mode='server', got {self._vllm_mode!r}. "
                "colocate/dedicated_rank LoRA support requires vLLM in-process add_lora() — "
                "add those code paths when the HTTP path is validated."
            )
        self._vllm_url = cfg.get("vllm_url", None)
        if self._vllm_url and "," in self._vllm_url:
            self._vllm_urls = [u.strip() for u in self._vllm_url.split(",")]
        elif self._vllm_url:
            self._vllm_urls = [self._vllm_url]
        else:
            self._vllm_urls = []
        self._vllm_group_port = cfg.get("vllm_group_port", 51216)
        # vllm_weight_sync must be False for LoRA recipe (different sync path)
        self._vllm_weight_sync = False
        self._vllm_max_model_len = cfg.get("vllm_max_model_len", 2048)
        self._vllm_clients = []
        self._vllm_client = None

        # Recipe state
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = 0
        self._epochs_run = 0
        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        self._save_every_n_steps = cfg.get("save_every_n_steps", None)
        self._save_final_checkpoint = cfg.get("save_final_checkpoint", True)
        self._save_adapter_only = cfg.get("save_adapter_weights_only", True)

    # Inject vLLM server mode setup from shared backend module
    _setup_vllm_server_mode = _vllm_backend_module._setup_vllm_server_mode

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        self._checkpointer = config.instantiate(cfg_checkpointer)
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def _update_recipe_state(self, checkpoint_dict: dict[str, Any]) -> None:
        self._epochs_run = checkpoint_dict[training.EPOCHS_KEY]
        self._steps_run = checkpoint_dict.get(training.STEPS_KEY, 0)

    def _setup_model_lora(
        self,
        cfg: DictConfig,
        model_sd: dict[str, Any],
    ) -> nn.Module:
        """Build LoRA-wrapped Qwen3 model with FSDP1.

        Sharding strategy is configurable via cfg.fsdp_sharding_strategy or the
        LORA_FSDP_FULL_SHARD=1 env var (both select FULL_SHARD).  Default is
        SHARD_GRAD_OP (ZeRO-2).  For 8B+, FULL_SHARD reduces unsharded-param
        residency at the cost of extra all-gathers — try it when hitting UR:40.

        Loads base weights (strict=False — LoRA params are randomly init'd),
        then wraps with FSDP1 at the top level only.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        utils.log_rank_zero(log, "LoRA-GRPO: Instantiating LoRA model on CPU ...")
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype):
            model = build_qwen3_lora_model(cfg)

        # Load base weights — strict=False because model_sd has no lora_a/lora_b keys
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        lora_missing = [k for k in missing if "lora" in k or "magnitude" in k]
        non_lora_missing = [k for k in missing if "lora" not in k and "magnitude" not in k]
        if non_lora_missing:
            log.warning("Rank 0: %d non-LoRA keys missing from checkpoint: %s",
                        len(non_lora_missing), non_lora_missing[:5])
        if self._is_rank_zero:
            log.info("LoRA model: %d LoRA keys randomly initialized (expected), "
                     "%d unexpected keys in checkpoint",
                     len(lora_missing), len(unexpected))
        del model_sd

        # If resuming from a PEFT adapter checkpoint, load adapter weights before FSDP wrap.
        # Adapter params are on CPU at this point (not yet moved to device).
        _adapter_ckpt_dir = cfg.get("lora_adapter_checkpoint_dir", None)
        if _adapter_ckpt_dir:
            utils.log_rank_zero(log, f"LoRA-GRPO: loading adapter weights from {_adapter_ckpt_dir}")
            n_loaded = load_peft_adapter_into_model(model, _adapter_ckpt_dir)
            utils.log_rank_zero(log, f"LoRA-GRPO: loaded {n_loaded} adapter tensors from checkpoint")

        # Move to device
        model = model.to(device=self._device, dtype=self._dtype)

        # Store vocab size before FSDP wrapping
        if hasattr(model, "tok_embeddings"):
            self._vocab_size = model.tok_embeddings.weight.shape[0]
        elif hasattr(model, "embed_tokens"):
            self._vocab_size = model.embed_tokens.weight.shape[0]
        else:
            self._vocab_size = 0

        # RoPE init
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

        # Mixed precision policy
        mp_policy = MixedPrecision(
            param_dtype=self._dtype,
            reduce_dtype=self._dtype,
            buffer_dtype=self._dtype,
        )

        # Select sharding strategy: FULL_SHARD (ZeRO-3) reduces unsharded-param
        # residency at the cost of more all-gathers; SHARD_GRAD_OP (ZeRO-2) keeps
        # full params after forward.  For 8B+, FULL_SHARD is safer for UR:40.
        # Accept both `fsdp_sharding_strategy` (canonical) and the legacy
        # `fsdp_shard_strategy` (used by every shipped LoRA YAML before 2026-05-02).
        # Without this, only LORA_FSDP_FULL_SHARD=1 actually moved the lever and
        # YAML overrides were silent no-ops.
        _cfg_strat = cfg.get("fsdp_sharding_strategy", None)
        if _cfg_strat is None and "fsdp_shard_strategy" in cfg:
            _cfg_strat = cfg.get("fsdp_shard_strategy")
            utils.log_rank_zero(
                log,
                "LoRA-GRPO: 'fsdp_shard_strategy' is deprecated; please rename to "
                "'fsdp_sharding_strategy' in your YAML.",
            )
        if _cfg_strat is None:
            _cfg_strat = "SHARD_GRAD_OP"
        if os.environ.get("LORA_FSDP_FULL_SHARD") == "1":
            _cfg_strat = "FULL_SHARD"
        _sharding_strategy = ShardingStrategy.FULL_SHARD if _cfg_strat == "FULL_SHARD" else ShardingStrategy.SHARD_GRAD_OP
        utils.log_rank_zero(log, f"LoRA-GRPO: FSDP1 sharding_strategy={_sharding_strategy.name}")

        # Exclude LoRA adapter params (lora_a / lora_b) from FSDP sharding.
        # Base linear weights inside LoRALinear ARE still FSDP-sharded for memory efficiency.
        # Adapter params are tiny (~5 MB total) — replication across 11 tiles is cheap.
        # Without this, each FSDP forward all-gathers adapter params creating L0 IPC handles
        # that accumulate across steps → UR:40 / banned:1 PDE on Aurora XPU.
        _lora_adapter_params = [
            p for n, p in model.named_parameters()
            if "lora_a" in n or "lora_b" in n
        ]
        utils.log_rank_zero(
            log,
            f"LoRA-GRPO: FSDP ignored_states={len(_lora_adapter_params)} adapter params "
            f"(replicated, not sharded; reduces IPC handle accumulation)",
        )
        model = FSDP(
            model,
            sharding_strategy=_sharding_strategy,
            mixed_precision=mp_policy,
            use_orig_params=True,
            limit_all_gathers=True,
            ignored_states=_lora_adapter_params,
        )

        if self._enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )
            utils.log_rank_zero(log, "LoRA-GRPO: activation checkpointing enabled")

        utils.log_rank_zero(
            log,
            f"LoRA-GRPO: FSDP1 {_sharding_strategy.name} model setup took {time.perf_counter() - init_start:.2f}s",
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
        """Create optimizer for adapter parameters only."""
        params = adapter_optimizer_params(self._model)
        if not params:
            raise ValueError(
                "No adapter parameters found — check that lora_attn_modules is non-empty "
                "and that build_qwen3_lora_model / set_trainable_params ran correctly."
            )
        utils.log_rank_zero(log, f"LoRA optimizer: {len(params)} adapter parameter tensors")
        optimizer = config.instantiate(cfg_optimizer, params)
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model, optimizer, opt_state_dict, self._device,
            )
        utils.log_rank_zero(log, "Optimizer is initialized (adapter-only).")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ):
        if cfg_lr_scheduler is None:
            return None
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        utils.log_rank_zero(log, "Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_profiler(self, cfg_profiler: Optional[DictConfig]):
        if cfg_profiler is None or not cfg_profiler.get("enabled", False):
            return DummyProfiler()
        profiler = config.instantiate(cfg_profiler)
        self.profiler_profile_memory = cfg_profiler.get("profile_memory", False)
        self.profiler_wait_steps = cfg_profiler.get("wait_steps", 5)
        self.profiler_warmup_steps = cfg_profiler.get("warmup_steps", 3)
        self.profiler_active_steps = cfg_profiler.get("active_steps", 2)
        return profiler

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
                config.instantiate(single_cfg, self._tokenizer)
                for single_cfg in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        collate_fn_obj = _get_component_from_path(collate_fn)

        # Server mode: all ranks must see the same batch (rank 0 generates + broadcasts)
        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
        )

        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn_obj,
        )
        if self._resume_from_checkpoint and dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)

        utils.log_rank_zero(log, "Dataset and Sampler are initialized.")
        return dataloader

    def setup(self, cfg: DictConfig) -> None:
        """Initialize all recipe components."""
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        # Load checkpoint (base model weights)
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        if self._resume_from_checkpoint:
            _adapter_ckpt_dir = cfg.get("lora_adapter_checkpoint_dir", None)
            _recipe_state_path = (
                os.path.join(_adapter_ckpt_dir, "recipe_state.pt")
                if _adapter_ckpt_dir else None
            )
            if _recipe_state_path and os.path.exists(_recipe_state_path):
                _rstate = torch.load(_recipe_state_path, map_location="cpu")
                self._epochs_run = _rstate.get(training.EPOCHS_KEY, 0)
                self._steps_run = _rstate.get(training.STEPS_KEY, 0)
                utils.log_rank_zero(
                    log,
                    f"LoRA-GRPO: resumed from adapter checkpoint "
                    f"(epochs={self._epochs_run}, steps={self._steps_run})",
                )
            else:
                try:
                    self._update_recipe_state(checkpoint_dict)
                except KeyError:
                    utils.log_rank_zero(
                        log,
                        "LoRA-GRPO: resume_from_checkpoint=True but no recipe state in checkpoint; "
                        "starting from step 0 (set lora_adapter_checkpoint_dir to load recipe_state.pt)",
                    )

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

        # Build LoRA model (no separate ref model)
        self._model = self._setup_model_lora(cfg, checkpoint_dict[training.MODEL_KEY])

        # Release checkpoint memory
        import gc as _gc
        try:
            checkpoint_dict[training.MODEL_KEY] = None
            checkpoint_dict.clear()
        except Exception:
            pass
        del checkpoint_dict
        _gc.collect()
        if self._device.type == "xpu":
            try:
                torch.xpu.synchronize()
            except Exception:
                pass

        if not self._production_mode:
            torch.distributed.barrier()

        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=self._opt_state_dict,
        )
        self._opt_state_dict = None

        self._loss_fn = config.instantiate(cfg.loss)
        self._use_chunked_loss = hasattr(self._loss_fn, "num_output_chunks")

        collate_name = cfg.get("collate_fn", "torchtune.dev.rl.data.padded_collate_rl")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            dataloader_state_dict=self._dataloader_state_dict,
        )
        self._dataloader_state_dict = None

        self._steps_per_epoch = len(self._dataloader)
        self.global_step = self._epochs_run * self._steps_per_epoch

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # RL params
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._total_steps = cfg.num_steps

        # Reward
        self._reward_mode = cfg.get("reward_mode", "math")
        self._cfg_reward_functions = None
        if cfg.get("reward_functions"):
            from torchtune import config as _tt_config
            self._cfg_reward_functions = [
                _tt_config.instantiate(fn) for fn in cfg.reward_functions
            ]

        # Stop tokens
        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
        else:
            stop_token_ids = getattr(self._tokenizer, "stop_tokens", [])
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

        # Async generation: LoRA recipe is sync-only; set attrs that _setup_vllm_server_mode reads.
        self._async_generation_enabled = False
        self._async_generation_max_staleness = 1

        # vLLM server clients
        self._setup_vllm_server_mode()

        # Ensure adapter dirs exist (rank 0 only)
        if self._is_rank_zero:
            os.makedirs(self._lora_shm_root, exist_ok=True)
            log.info("LoRA checkpoint adapter dir: %s", self._lora_shm_root)
            if self._lora_tmpfs_transfer:
                os.makedirs(self._lora_train_shm, exist_ok=True)
                log.info("LoRA publish: tmpfs_transfer=True — train_shm=%s  vllm_shm=%s",
                         self._lora_train_shm, self._lora_vllm_shm)
                # Pre-create vllm_shm parent on VLLM_NODE so first rsync succeeds
                if self._vllm_clients:
                    vllm_ip = self._vllm_clients[0].host
                    user = os.environ.get("USER", "")
                    dest = f"{user}@{vllm_ip}" if user else vllm_ip
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                         dest, f"mkdir -p {self._lora_vllm_shm}"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode != 0:
                        log.warning("Failed to pre-create vllm_shm on %s: %s", dest, result.stderr)
                    else:
                        log.info("Pre-created vllm_shm on %s:%s", dest, self._lora_vllm_shm)
                    # Open a persistent SSH ControlMaster so subsequent rsync calls reuse
                    # the existing connection instead of paying ~1s per-handshake.
                    ctrl_socket = f"/tmp/torchtune_ssh_ctrl_{vllm_ip.replace('.', '_')}"
                    ctrl_proc = subprocess.Popen(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                         "-o", "ControlMaster=yes", "-o", f"ControlPath={ctrl_socket}",
                         "-o", "ControlPersist=yes", "-N", dest],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    # Wait briefly for the socket to be created (SSH handshake <1s on HSN).
                    for _t in range(30):
                        if os.path.exists(ctrl_socket):
                            break
                        time.sleep(0.1)
                    if os.path.exists(ctrl_socket):
                        self._ssh_control_socket = ctrl_socket
                        self._ssh_control_proc = ctrl_proc
                        log.info("SSH ControlMaster ready for %s (socket=%s)", dest, ctrl_socket)
                    else:
                        log.warning("SSH ControlMaster socket not created for %s — rsync will use direct SSH", dest)
                        ctrl_proc.terminate()

    # -------------------------------------------------------------------------
    # LoRA weight sync
    # -------------------------------------------------------------------------

    def _gather_lora_state_dict(self):
        """FSDP collective gather — ALL ranks must call this.

        Returns a tuple (peft_sd, adapter_config, adapter_name, slot, local_path, vllm_path)
        on rank 0, or None on non-rank-0 ranks.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        with FSDP.state_dict_type(
            self._model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_sd = self._model.state_dict()

        if not self._is_rank_zero:
            return None

        adapter_sd = get_adapter_state_dict(full_sd, device="cpu")
        del full_sd
        peft_sd, adapter_config = torchtune_to_peft_state_dict(
            adapter_sd,
            model_name=str(getattr(self._checkpointer, "_checkpoint_dir", "base_model")),
            rank=self._lora_rank,
            alpha=self._lora_alpha,
            target_modules=self._lora_target_modules,
        )
        del adapter_sd

        step_id = self._steps_run
        slot = step_id % self._lora_max_loras
        adapter_name = f"rl_step_{step_id}"

        if self._lora_tmpfs_transfer:
            local_path = os.path.join(self._lora_train_shm, f"slot_{slot}")
            vllm_path = os.path.join(self._lora_vllm_shm, f"slot_{slot}")
        else:
            local_path = os.path.join(self._lora_shm_root, f"slot_{slot}")
            vllm_path = local_path  # Lustre is cross-node visible

        return peft_sd, adapter_config, adapter_name, slot, local_path, vllm_path

    def _publish_lora_background(self, state) -> None:
        """Write adapter + transfer to VLLM_NODE + POST. Runs in background thread (rank 0 only).

        All state mutations (_prev_lora_name, client._model_name) happen here.
        The train loop must join() this thread before calling generate_trajectory_batched().
        """
        peft_sd, adapter_config, adapter_name, slot, local_path, vllm_path = state

        # Write PEFT adapter dir to local path (tmpfs or Lustre)
        t0 = time.perf_counter()
        write_peft_adapter_dir(peft_sd, adapter_config, local_path)
        del peft_sd
        log.info("Rank 0: wrote PEFT adapter to %s in %.2fs", local_path, time.perf_counter() - t0)

        # rsync to VLLM_NODE /dev/shm if tmpfs_transfer enabled
        if self._lora_tmpfs_transfer and self._vllm_clients:
            vllm_ip = self._vllm_clients[0].host
            user = os.environ.get("USER", "")
            dest = f"{user}@{vllm_ip}:{vllm_path}/" if user else f"{vllm_ip}:{vllm_path}/"
            # Use ControlMaster socket if available to reuse the existing SSH connection
            # and avoid ~1s per-handshake overhead; fall back to direct SSH otherwise.
            if self._ssh_control_socket:
                ssh_cmd = (
                    f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes"
                    f" -o ControlMaster=no -o ControlPath={self._ssh_control_socket}"
                )
            else:
                ssh_cmd = "ssh -o StrictHostKeyChecking=no -o BatchMode=yes"
            cmd = [
                "rsync", "-a", "--inplace", "--delete",
                "-e", ssh_cmd,
                local_path + "/", dest,
            ]
            t_rsync = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"rsync to {dest} failed (rc={result.returncode}): {result.stderr}")
            log.info("Rank 0: rsync adapter to %s in %.2fs", dest, time.perf_counter() - t_rsync)

        if not self._vllm_clients:
            log.warning("Rank 0: no vLLM clients to publish adapter to")
            return

        # POST /v1/load_lora_adapter to all vLLM tiles in parallel
        t_post = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(self._vllm_clients)) as pool:
            futures = {
                pool.submit(
                    load_lora_adapter_http,
                    c.session, c.base_url, adapter_name, vllm_path, 120,
                ): i
                for i, c in enumerate(self._vllm_clients)
            }
            failed = []
            for f in as_completed(futures):
                if not f.result():
                    failed.append(futures[f])
        if failed:
            raise RuntimeError(
                f"load_lora_adapter '{adapter_name}' failed on {len(failed)}/{len(self._vllm_clients)} tiles: {failed}"
            )
        log.info(
            "Rank 0: load_lora_adapter '%s' to %d tiles in %.2fs",
            adapter_name, len(self._vllm_clients), time.perf_counter() - t_post,
        )

        # Unload previous adapter to free vLLM KV cache slot
        prev_name = self._prev_lora_name
        if prev_name is not None:
            with ThreadPoolExecutor(max_workers=len(self._vllm_clients)) as pool:
                for f in [
                    pool.submit(unload_lora_adapter_http, c.session, c.base_url, prev_name)
                    for c in self._vllm_clients
                ]:
                    f.result()

        # Update client model names (join() in train loop ensures this runs before next generate)
        for c in self._vllm_clients:
            c._model_name = adapter_name
        self._prev_lora_name = adapter_name
        self._prev_lora_path = vllm_path
        self._current_lora_name = adapter_name

    def _publish_lora_to_vllm(self) -> None:
        """Synchronous wrapper: gather + publish in one blocking call (used by tests/manual calls).

        The train loop uses _gather_lora_state_dict + _publish_lora_background (async) instead.
        """
        state = self._gather_lora_state_dict()
        if state is not None:
            self._publish_lora_background(state)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _generate_with_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Call vLLM server for generation, broadcast results to all ranks.

        Broadcasts a success/failure flag before the data broadcast so that when
        rank 0 fails (e.g. vLLM EngineCore crash), all other ranks are unblocked
        and exit cleanly instead of hanging at the data broadcast barrier.
        """
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        _ok = torch.zeros(1, device=self._device, dtype=torch.int32)
        _exc: Exception | None = None

        if self._is_rank_zero:
            try:
                query_responses = self._call_vllm_http(batch_input_ids, context_length)
                _ok.fill_(1)
            except Exception as _e:
                _exc = _e
        else:
            query_responses = batch_input_ids.new_empty(bsz, total_len)

        # Broadcast success flag so failing rank 0 unblocks all other ranks.
        torch.distributed.broadcast(_ok, src=0)

        if _ok.item() == 0:
            if _exc is not None:
                raise _exc
            raise RuntimeError("vLLM generation failed on rank 0; exiting")

        return self._broadcast_query_responses(query_responses)

    def _call_vllm_http(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Rank-0-only vLLM HTTP round-trip (mirrors base recipe implementation)."""
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        prompts = []
        for i in range(bsz):
            ids = batch_input_ids[i].cpu().tolist()
            ids = [t for t in ids if t != self._tokenizer.pad_id]
            prompts.append(ids)

        gen_kwargs = dict(
            n=1,
            max_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k or 0,
        )

        t0 = time.perf_counter()
        num_clients = len(self._vllm_clients)
        if num_clients > 1:
            chunks = [prompts[i::num_clients] for i in range(num_clients)]

            def _call(client, chunk):
                return client.generate(prompts=chunk, **gen_kwargs) if chunk else []

            with ThreadPoolExecutor(max_workers=num_clients) as pool:
                futures = {
                    pool.submit(_call, client, chunk): idx
                    for idx, (client, chunk) in enumerate(zip(self._vllm_clients, chunks))
                }
                chunk_results = [None] * num_clients
                for future in as_completed(futures):
                    idx = futures[future]
                    chunk_results[idx] = future.result()

            completions = [None] * bsz
            for i in range(bsz):
                completions[i] = chunk_results[i % num_clients][i // num_clients]
        else:
            completions = self._vllm_client.generate(prompts=prompts, **gen_kwargs)

        log.info("Rank 0: vLLM generate %.1fs (%d prompts)", time.perf_counter() - t0, bsz)

        query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
        query_responses[:, :context_length] = batch_input_ids
        for i, comp in enumerate(completions):
            length = min(len(comp), self._max_generated_tokens)
            query_responses[i, context_length : context_length + length] = torch.tensor(
                comp[:length], dtype=batch_input_ids.dtype
            )
        return query_responses

    def _broadcast_query_responses(self, query_responses: torch.Tensor) -> torch.Tensor:
        """Broadcast rank-0's query_responses to all ranks."""
        query_responses = query_responses.to(self._device)
        torch.distributed.broadcast(query_responses, src=0)
        return query_responses

    # -------------------------------------------------------------------------
    # Trajectory generation
    # -------------------------------------------------------------------------

    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
    ) -> GRPOTrajectory:
        """Generate one GRPO trajectory (server mode, LoRA adapter)."""
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        device_empty_cache(self._device)

        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)
        num_seqs = batch_size * grpo_size

        # Step 1: vLLM HTTP generation (adapter-aware: client._model_name = adapter name)
        _vllm_t0 = time.perf_counter()
        query_responses = self._generate_with_vllm(batch_input_ids, context_length)
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _vllm_time = time.perf_counter() - _vllm_t0

        responses = query_responses[:, context_length:].clone()

        # Clamp OOB token IDs (XPU scatter kernel crashes on out-of-range)
        vocab_size = getattr(self, "_vocab_size", 0)
        if vocab_size > 0:
            oob_mask = responses >= vocab_size
            if oob_mask.any():
                log.warning("Clamping %d OOB token IDs", oob_mask.sum().item())
                responses = responses.clamp(max=vocab_size - 1)
                query_responses = torch.cat([query_responses[:, :context_length], responses], dim=1)

        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # Attention masks and position IDs.
        # TORCHTUNE_MASKFREE_CAUSAL=1: skip explicit mask construction so that
        # TORCHTUNE_USE_IPEX_VARLEN=1 can engage (varlen requires mask=None).
        # Guard: XPU only, no packing, no prompt-side padding.
        _maskfree, _mf_reason = _compute_maskfree_causal(
            env_set=os.environ.get("TORCHTUNE_MASKFREE_CAUSAL") == "1",
            device_type=self._device.type,
            packing_enabled=False,
            query_responses=query_responses,
            context_length=context_length,
            pad_id=self._tokenizer.pad_id,
        )
        if _mf_reason == "prompt padding detected":
            log.warning(
                "TORCHTUNE_MASKFREE_CAUSAL=1 but batch has prompt padding; "
                "falling back to explicit mask."
            )
        if _maskfree:
            masks = None
        else:
            masks = generation.get_causal_mask_from_padding_mask(
                query_response_padding_masks, target_seq_len=context_length + self._max_generated_tokens
            )
        position_ids = generation.get_position_ids_from_padding_mask(query_response_padding_masks)

        # Step 2: policy logprobs (adapter-enabled forward)
        fwd_bs = self._forward_batch_size
        _policy_fwd_t0 = time.perf_counter()
        with torch.no_grad():
            if fwd_bs >= num_seqs:
                policy_logits = self._model(query_responses, input_pos=position_ids, mask=masks)
                policy_logits = rlhf.truncate_sequence_for_logprobs(policy_logits, context_length)
                logprobs = rlhf.batched_logits_to_logprobs(policy_logits, responses, self._temperature)
                del policy_logits
            else:
                chunks = []
                for cs in range(0, num_seqs, fwd_bs):
                    ce = min(cs + fwd_bs, num_seqs)
                    chunk_logits = self._model(
                        query_responses[cs:ce],
                        input_pos=position_ids[cs:ce],
                        mask=None if masks is None else masks[cs:ce],
                    )
                    chunk_logits = chunk_logits[:, context_length - 1:]
                    chunks.append(
                        rlhf.batched_logits_to_logprobs(chunk_logits, responses[cs:ce], self._temperature)
                    )
                    del chunk_logits
                logprobs = torch.cat(chunks, dim=0)
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _policy_fwd_time = time.perf_counter() - _policy_fwd_t0

        # Step 3: ref logprobs via disable_adapter (no separate ref model)
        _ref_fwd_t0 = time.perf_counter()
        with disable_adapter(self._model):
            with torch.no_grad():
                if fwd_bs >= num_seqs:
                    ref_logits = self._model(query_responses, input_pos=position_ids, mask=masks)
                    ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
                    ref_logprobs = rlhf.batched_logits_to_logprobs(ref_logits, responses, self._temperature)
                    del ref_logits
                else:
                    ref_chunks = []
                    for cs in range(0, num_seqs, fwd_bs):
                        ce = min(cs + fwd_bs, num_seqs)
                        chunk_ref = self._model(
                            query_responses[cs:ce],
                            input_pos=position_ids[cs:ce],
                            mask=None if masks is None else masks[cs:ce],
                        )
                        chunk_ref = rlhf.truncate_sequence_for_logprobs(chunk_ref, context_length)
                        ref_chunks.append(
                            rlhf.batched_logits_to_logprobs(chunk_ref, responses[cs:ce], self._temperature)
                        )
                        del chunk_ref
                    ref_logprobs = torch.cat(ref_chunks, dim=0)
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _ref_fwd_time = time.perf_counter() - _ref_fwd_t0

        log.info(
            "Rank %d: GENTIMING vllm=%.1fs policy_fwd=%.1fs ref_fwd=%.1fs",
            self.rank, _vllm_time, _policy_fwd_time, _ref_fwd_time,
        )

        # Step 4: truncate at first stop token
        (response_padding_masks, responses) = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # Step 5: rewards
        responses = responses.reshape(batch_size, grpo_size, -1)
        if self._cfg_reward_functions:
            decoded = [
                self._tokenizer.decode(responses[b, g].tolist())
                for b in range(batch_size) for g in range(grpo_size)
            ]
            flat_answers = [answers[b] for b in range(batch_size) for _ in range(grpo_size)]
            flat_ids = responses.reshape(batch_size * grpo_size, -1)
            r_stack, s_stack = [], []
            for fn in self._cfg_reward_functions:
                out = fn(flat_ids, decoded, flat_answers)
                r_stack.append(out.total_reward.to(self._device))
                s_stack.append(out.successes.to(self._device))
            rewards = torch.stack(r_stack, dim=-1).reshape(batch_size, grpo_size, -1)
            successes = torch.stack(s_stack, dim=-1).reshape(batch_size, grpo_size, -1)
        else:
            rewards, successes, _ = batched_rewards(
                self._tokenizer, responses, answers, device=self._device
            )
        rewards = rewards.to(self._device).sum(dim=-1)
        successes = successes.to(self._device).sum(dim=-1)

        # Log first sample
        if self._is_rank_zero:
            try:
                sample = responses[0, 0]
                non_pad = sample[sample != self._tokenizer.pad_id]
                decoded_sample = self._tokenizer.decode(non_pad.tolist())
                log.info(
                    "SAMPLE_RESPONSE step=%d reward=%.1f answer=%s response=%s",
                    self._steps_run, rewards[0, 0].item(), answers[0][:80], decoded_sample[:200],
                )
            except Exception as _e:
                log.warning("Could not decode sample response: %s", _e)

        # Reward stats
        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        if self._is_rank_zero:
            log.info(
                "REWARDS step=%d mean=%.3f std=%.3f successes=%.2f",
                self._steps_run, rewards_mean, rewards_std, successes.float().mean().item(),
            )

        # Advantages
        advantages = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-4)
        advantages = advantages.reshape(batch_size * grpo_size)
        del responses
        device_empty_cache(self._device)

        # Mask padding
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
        """Generate trajectories in forward_batch_size micro-batches."""
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[batch_start: batch_start + self._forward_batch_size]
                batch_answers = answers[batch_start: batch_start + self._forward_batch_size]
                device_empty_cache(self._device)
                trajectories.append(self.generate_trajectory(batch_input_ids, batch_answers))
                device_empty_cache(self._device)

        concatenated_fields = {}
        for field_name in trajectories[0]._fields:
            values = [getattr(traj, field_name) for traj in trajectories]
            if field_name == "answers":
                result = []
                for v in values:
                    result.extend(v)
            elif values[0] is None:
                result = None
            else:
                result = torch.cat(values, dim=0)
            concatenated_fields[field_name] = result
        return GRPOTrajectory(**concatenated_fields)

    # -------------------------------------------------------------------------
    # Training step
    # -------------------------------------------------------------------------

    def grpo_step(self, trajectory: GRPOTrajectory) -> dict[str, float]:
        """Run one GRPO gradient update (adapter parameters only)."""
        num_seqs = trajectory.query_responses.shape[0]
        fwd_bs = self._forward_batch_size
        # context_length = prompt tokens = full seq len minus response len
        context_length = (
            trajectory.query_responses.shape[1] - trajectory.response_padding_masks.shape[1]
        )
        responses = trajectory.query_responses[:, context_length:]

        self._model.train()
        self._optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self._device)
        total_kl = torch.tensor(0.0, device=self._device)

        _chunked = os.environ.get("TORCHTUNE_USE_CHUNKED_LOSS") == "1"

        if _chunked:
            # Per-chunk forward + backward (avoids OOM on long sequences).
            # Matches base GRPO recipe: correct grad scaling + FSDP1 no_sync suppression.
            num_fwd_chunks = (num_seqs + fwd_bs - 1) // fwd_bs
            grad_scale = num_fwd_chunks * max(1, self._gradient_accumulation_steps)

            # Suppress FSDP1 grad sync (reduce-scatter) for all but the final chunk.
            # Each backward() fires a reduce-scatter without this; with no_sync we
            # accumulate gradients locally and fire exactly once on the last chunk.
            _use_fsdp1_no_sync = (
                num_fwd_chunks > 1
                and hasattr(self._model, 'no_sync')
            )

            for cs in range(0, num_seqs, fwd_bs):
                ce = min(cs + fwd_bs, num_seqs)
                _is_last_chunk = (ce >= num_seqs)
                chunk_logits = self._model(
                    trajectory.query_responses[cs:ce],
                    input_pos=trajectory.position_ids[cs:ce],
                    mask=trajectory.masks[cs:ce] if trajectory.masks is not None else None,
                )
                chunk_logits = rlhf.truncate_sequence_for_logprobs(chunk_logits, context_length)
                chunk_pi_lp = rlhf.batched_logits_to_logprobs(
                    chunk_logits, responses[cs:ce], self._temperature
                )
                del chunk_logits
                chunk_old_lp = (
                    trajectory.logprobs[cs:ce]
                    if trajectory.logprobs is not None
                    else chunk_pi_lp.detach()
                )
                # NOTE: padding_masks=True means "include this token in loss" (base recipe
                # convention). response_padding_masks is True for PAD/truncated tokens, so
                # invert it here. Bug in the original: was passing without ~.
                chunk_loss, _, chunk_kl, *_ = self._loss_fn(
                    chunk_old_lp,
                    chunk_pi_lp,
                    trajectory.ref_logprobs[cs:ce],
                    trajectory.advantages[cs:ce],
                    padding_masks=~trajectory.response_padding_masks[cs:ce],
                )
                if self._is_rank_zero and cs == 0 and self._device.type == "xpu":
                    try:
                        ms = torch.xpu.memory_stats(self._device)
                        log.info(
                            "MEMCHECK grpo_step chunk0 pre-backward: active=%.2f GiB reserved=%.2f GiB",
                            ms.get("active_bytes.all.current", 0) / 2**30,
                            ms.get("reserved_bytes.all.current", 0) / 2**30,
                        )
                    except Exception:
                        pass
                if _use_fsdp1_no_sync and not _is_last_chunk:
                    with self._model.no_sync():
                        (chunk_loss / grad_scale).backward()
                else:
                    (chunk_loss / grad_scale).backward()
                total_loss += chunk_loss.detach()
                total_kl += chunk_kl.detach()
        else:
            # Single forward + backward
            grad_scale = max(1, self._gradient_accumulation_steps)
            logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
            )
            logits = rlhf.truncate_sequence_for_logprobs(logits, context_length)
            pi_logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
            del logits
            old_logprobs = (
                trajectory.logprobs if trajectory.logprobs is not None else pi_logprobs.detach()
            )
            # NOTE: padding_masks=True means "include this token in loss" (base recipe
            # convention). response_padding_masks is True for PAD/truncated tokens, so
            # invert it here. Bug in the original: was passing without ~.
            loss, _, kl_loss, *_ = self._loss_fn(
                old_logprobs,
                pi_logprobs,
                trajectory.ref_logprobs,
                trajectory.advantages,
                padding_masks=~trajectory.response_padding_masks,
            )
            (loss / grad_scale).backward()
            total_loss = loss.detach()
            total_kl = kl_loss.detach()

        return {
            "loss": total_loss.item(),
            "kl": total_kl.item() if isinstance(total_kl, torch.Tensor) else float(total_kl),
        }

    def save_checkpoint(self, epoch: int) -> None:
        """Save adapter weights (base weights excluded by default)."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        with FSDP.state_dict_type(
            self._model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_sd = self._model.state_dict()

        if self._is_rank_zero:
            if self._save_adapter_only:
                # Save as PEFT adapter dir — avoids routing lora_a/lora_b keys
                # through the HF checkpointer's qwen3_tune_to_hf which only
                # knows base model keys and raises KeyError on adapter keys.
                adapter_sd = get_adapter_state_dict(full_sd, device="cpu")
                peft_sd, adapter_config = torchtune_to_peft_state_dict(
                    adapter_sd,
                    model_name=str(getattr(self._checkpointer, "_checkpoint_dir", "base_model")),
                    rank=self._lora_rank,
                    alpha=self._lora_alpha,
                    target_modules=self._lora_target_modules,
                )
                save_dir = os.path.join(
                    self._checkpointer._output_dir, f"epoch_{epoch}"
                )
                os.makedirs(save_dir, exist_ok=True)
                write_peft_adapter_dir(peft_sd, adapter_config, save_dir)
                log.info("Rank 0: adapter checkpoint saved to %s (epoch=%d)", save_dir, epoch)
                recipe_state_path = os.path.join(save_dir, "recipe_state.pt")
                torch.save(
                    {training.EPOCHS_KEY: epoch, training.STEPS_KEY: self._steps_run},
                    recipe_state_path,
                )
                log.info("Rank 0: recipe state saved to %s (steps_run=%d)", recipe_state_path, self._steps_run)
            else:
                self._checkpointer.save_checkpoint(
                    {training.MODEL_KEY: full_sd},
                    epoch=epoch,
                )
                log.info("Rank 0: full checkpoint saved (epoch=%d)", epoch)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def _warmup_vllm(self) -> None:
        """Rank-0-only: verify vLLM EngineCore is functional before training starts.

        Sends a trivial 1-token generate request.  Raises immediately on failure —
        the EngineCore process does not self-restart after a crash (Aurora XPU stale
        L0 driver state), so sleeping and retrying is useless.  The launcher is
        responsible for restarting vLLM if the EngineCore is broken.
        """
        if not self._is_rank_zero or not self._vllm_clients:
            return
        _dummy = [self._tokenizer.bos_id or 1]
        try:
            self._vllm_client.generate(
                prompts=[_dummy], n=1, max_tokens=1,
                temperature=self._temperature, top_k=self._top_k or 0,
            )
            log.info("LoRA-GRPO: vLLM warm-up OK")
        except RuntimeError as _e:
            raise RuntimeError(
                f"vLLM EngineCore failed warm-up (node may have stale L0 state): {_e}"
            ) from _e

    def train(self) -> None:
        """Main training loop."""
        # Verify vLLM EngineCore is functional before entering the training loop.
        # On Aurora XPU, the EngineCore can appear healthy but crash on the first
        # inference call due to stale L0 driver state from prior jobs on the same node.
        # Rank 0 retries up to 3× with 60s waits; all ranks barrier after.
        if self._is_rank_zero:
            self._warmup_vllm()
        torch.distributed.barrier()

        # Initialize LoRA adapter publish before training (step 0)
        # so vLLM generates with base model weights on first step.
        if self._is_rank_zero:
            log.info("LoRA-GRPO: starting training (adapter not yet published; "
                     "step 0 generates with base model weights)")

        training_completed = False
        _saved_this_epoch = False
        with self._profiler:
            for curr_epoch in range(self._epochs_run, self.total_epochs):
                self._dataloader.sampler.set_epoch(curr_epoch)

                pbar = tqdm(
                    total=self._steps_per_epoch,
                    disable=not self._is_rank_zero,
                    desc=f"LoRA-GRPO epoch {curr_epoch + 1}/{self.total_epochs}",
                )

                for idx, batch in enumerate(self._dataloader):
                    if self._steps_run >= self._total_steps:
                        training_completed = True
                        break

                    _step_t0 = time.perf_counter()
                    self._profiler.step()

                    tokens = batch["tokens"].to(self._device)
                    answers = batch.get("answers", [""] * tokens.shape[0])

                    # Join pending adapter publish thread before generating.
                    # A stale adapter in vLLM corrupts GRPO semantics (generation
                    # policy diverges from training policy), so we fail fast here.
                    if self._publish_thread is not None:
                        _join_t0 = time.perf_counter()
                        self._publish_thread.join(timeout=120)
                        _timed_out = self._publish_thread.is_alive()
                        _pub_err = self._publish_error
                        self._publish_thread = None
                        self._publish_error = None
                        if self._is_rank_zero:
                            log.info("Rank 0: publish join %.2fs", time.perf_counter() - _join_t0)
                        if _timed_out:
                            raise RuntimeError(
                                "Adapter publish thread timed out (120s) — vLLM adapter "
                                "may not have been updated. Aborting to avoid stale-policy training."
                            )
                        if _pub_err is not None:
                            raise RuntimeError(
                                f"Adapter publish failed before generation: {_pub_err}"
                            )

                    # Generate trajectory (server mode, may use base or LoRA adapter)
                    _gen_t0 = time.perf_counter()
                    with torch.no_grad():
                        trajectory = self.generate_trajectory_batched(tokens, answers)
                    _gen_time = time.perf_counter() - _gen_t0

                    # PPO epochs
                    _grpo_t0 = time.perf_counter()
                    grpo_metrics = {}
                    for _ppo_epoch in range(self._ppo_epochs):
                        grpo_metrics = self.grpo_step(trajectory)
                    _grpo_time = time.perf_counter() - _grpo_t0

                    # All-reduce adapter param gradients across data-parallel ranks.
                    # Adapter params are in FSDP ignored_states (replicated, not sharded),
                    # so FSDP does NOT sync their grads automatically — we must do it here.
                    # Single flat all-reduce instead of one-per-param (504 CCL ops → 1).
                    _ar_t0 = time.perf_counter()
                    _world_size = torch.distributed.get_world_size()
                    if _world_size > 1:
                        _adapter_grads = [
                            _ap.grad for _ap in adapter_optimizer_params(self._model)
                            if _ap.grad is not None
                        ]
                        if _adapter_grads:
                            _flat = torch.cat([g.flatten() for g in _adapter_grads])
                            torch.distributed.all_reduce(_flat)
                            _flat.div_(_world_size)
                            _offset = 0
                            for _g in _adapter_grads:
                                _n = _g.numel()
                                _g.copy_(_flat[_offset: _offset + _n].view_as(_g))
                                _offset += _n
                            del _flat
                            if self._is_rank_zero:
                                _ar_mb = sum(g.numel() * g.element_size() for g in _adapter_grads) / 1e6
                                log.info(
                                    "ADAPTER_AR step=%d %.1fMB in %.2fs (%d params synced)",
                                    self._steps_run, _ar_mb,
                                    time.perf_counter() - _ar_t0, len(_adapter_grads),
                                )
                            del _adapter_grads

                    # Clip gradients
                    _clip_t0 = time.perf_counter()
                    if self._clip_grad_norm is not None:
                        # Clip adapter params only (they're tiny — no FSDP clip_grad_norm needed)
                        adapter_params = adapter_optimizer_params(self._model)
                        torch.nn.utils.clip_grad_norm_(adapter_params, self._clip_grad_norm)
                    _clip_time = time.perf_counter() - _clip_t0

                    # Optimizer step
                    _opt_t0 = time.perf_counter()
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _opt_time = time.perf_counter() - _opt_t0

                    # LoRA publish: Phase A (sync FSDP gather, all ranks),
                    # Phase B (async IO + rsync + HTTP, rank 0 background thread)
                    if self._steps_run % self._lora_publish_every == 0:
                        _pub_t0 = time.perf_counter()
                        publish_state = self._gather_lora_state_dict()  # all ranks participate
                        if self._is_rank_zero and publish_state is not None:
                            _gather_time = time.perf_counter() - _pub_t0
                            log.info(
                                "Rank 0: FSDP gather done in %.2fs — starting async publish",
                                _gather_time,
                            )
                            self._publish_error = None
                            _state = publish_state

                            def _bg(_s=_state):
                                try:
                                    self._publish_lora_background(_s)
                                except Exception as _e:
                                    self._publish_error = _e
                                    log.error("Rank 0: async publish failed: %s", _e)

                            self._publish_thread = threading.Thread(target=_bg, daemon=True)
                            self._publish_thread.start()

                    self.global_step += 1
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    _step_time = time.perf_counter() - _step_t0
                    if self._is_rank_zero:
                        log.info(
                            "TIMING step=%d  total=%.1fs  gen=%.1fs  grpo=%.1fs  clip=%.1fs  opt=%.1fs",
                            self._steps_run, _step_time, _gen_time, _grpo_time, _clip_time, _opt_time,
                        )

                    self._steps_run += 1

                    # Logging
                    if self._steps_run % self._log_every_n_steps == 0 and self._is_rank_zero:
                        lr = get_lr(self._optimizer)
                        metrics = {
                            "loss": grpo_metrics.get("loss", 0.0),
                            "kl": grpo_metrics.get("kl", 0.0),
                            "lr": lr,
                        }
                        if self._log_peak_memory_stats:
                            try:
                                mem = training.get_memory_stats(device=self._device)
                                metrics.update({f"memory/{k}": v for k, v in mem.items()})
                            except RuntimeError:
                                pass
                        self._metric_logger.log_dict(metrics, step=self.global_step)

                    pbar.update(1)

                    # Mid-epoch step checkpoint
                    if (
                        self._save_every_n_steps is not None
                        and self._steps_run % self._save_every_n_steps == 0
                    ):
                        self.save_checkpoint(epoch=curr_epoch)
                        _saved_this_epoch = True

                    if training_completed:
                        break

                pbar.close()

                # Epoch checkpoint
                _saved_this_epoch = (curr_epoch + 1) % self._save_every_n_epochs == 0
                if _saved_this_epoch:
                    self.save_checkpoint(epoch=curr_epoch)

                if training_completed:
                    break

                self._epochs_run += 1

        # Final checkpoint (skip if save_every_n_epochs already saved the last epoch)
        if self._is_rank_zero:
            log.info("LoRA-GRPO training complete (%d steps)", self._steps_run)
        if self._save_final_checkpoint and not _saved_this_epoch:
            self.save_checkpoint(epoch=curr_epoch)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
            # Join the last publish thread before closing ControlMaster / destroying PG.
            if self._publish_thread is not None:
                self._publish_thread.join(timeout=120)
                self._publish_thread = None
            if self._ssh_control_proc is not None:
                try:
                    self._ssh_control_proc.terminate()
                except Exception:
                    pass
                self._ssh_control_proc = None
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for torchtune CLI."""
    config.log_config(recipe_name="LoRAGRPODistributedXPU", cfg=cfg)
    recipe = LoRAGRPODistributedXPU(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

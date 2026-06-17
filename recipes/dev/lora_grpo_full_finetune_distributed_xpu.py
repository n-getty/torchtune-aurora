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
    iter_merged_lora_layers,
    _ATTN_TARGET_MODULES,
    _MLP_TARGET_MODULES,
    _TUNE_MODULE_TO_HF,
)
from torchtune.dev.rl.weight_sync import _save_raw_bytes
import torchtune.dev.rl.weight_sync as _weight_sync_module
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

        # Adapter-delivery path:
        #   use_runtime_lora=False (default) — merged-weight broadcast: gather adapter,
        #     compute W_eff = W_base + (alpha/rank)*(B@A), POST raw_bytes to vLLM
        #     /collective_rpc load_weights_from_raw. Sidesteps the vLLM --enable-lora
        #     PDE crash on Aurora XPU (docs/reports/enable_lora_issue.md).
        #   use_runtime_lora=True — legacy PEFT dir + /v1/load_lora_adapter path.
        self._lora_use_runtime = bool(_get("use_runtime_lora", False))

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
        # Batch-level advantage normalization (matches base GRPO recipe). At B=1
        # this is mathematically identical to per-prompt; for B>1 it keeps the
        # learning signal alive when a prompt-group's rewards are degenerate.
        # Opt out with batch_level_advantages: false to reproduce legacy runs.
        self._batch_level_advantages = cfg.get("batch_level_advantages", True)

    # Inject vLLM server mode setup from shared backend module
    _setup_vllm_server_mode = _vllm_backend_module._setup_vllm_server_mode

    # Inject the shared Llama-family Q/K un-permute helpers from weight_sync.
    # The merged-weight publish path (_gather_merged_lora_weights) sends weights
    # straight to vLLM's load_weights, which expects HF-format unpermuted Q/K.
    # Llama-family checkpointers (LLAMA2/3/3_2/3_VISION) permute Q/K at load
    # time, so without inverting it vLLM's attention is scrambled after the
    # first sync. No-op for Qwen3/Gemma (their hf_to_tune doesn't permute).
    # See torchtune/dev/rl/weight_sync.py and the base GRPO recipe.
    _needs_qk_unpermute = _weight_sync_module._needs_qk_unpermute
    _maybe_unpermute_qk = _weight_sync_module._maybe_unpermute_qk

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

        # NOTE: do NOT cast adapter params to fp32 in-place — LoRALinear
        # forward computes `lora_a(bf16_input)` and the matmul rejects mixed
        # dtypes (RuntimeError: expected mat1 and mat2 to have the same
        # dtype). Master fp32 copy + post-step write-back is wired in
        # _setup_optimizer instead, addressing the bf16 AdamW underflow that
        # froze the adapter (delta=0). See
        # docs/reports/lora_4b_mns8_frozen_adapter_20260503.md.

        # Fix: broadcast adapter params from rank 0. LoRA inits use
        # nn.init.kaiming_uniform_ on lora_a (consumes RNG), and per-rank RNG
        # state diverges from the model build path (dataloader seeding etc.),
        # so without an explicit broadcast each rank's lora_a starts with
        # different values. FSDP1 with ignored_states= does NOT broadcast the
        # ignored params even with sync_module_states=True. After this loop,
        # all ranks hold rank-0's adapter values; subsequent grad all-reduces
        # keep them in sync.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            with torch.no_grad():
                for _p in _lora_adapter_params:
                    torch.distributed.broadcast(_p.data, src=0)
            utils.log_rank_zero(
                log,
                f"LoRA-GRPO: broadcast {len(_lora_adapter_params)} adapter params "
                f"from rank 0 (cross-rank init divergence fix)",
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
        """Create optimizer over fp32 MASTER copies of the bf16 adapter params.

        The model holds bf16 adapter params (LoRALinear forward requires
        matching bf16 input/weight dtypes). AdamW's per-element update during
        warmup (`lr ≈ 3e-6`) underflows bf16's relative epsilon (`~7.8e-3` for
        params of magnitude `~1e-1`) — observed as `delta=0.000000e+00` for
        every step in the 2026-05-03 frozen-adapter run.

        Fix: keep model params bf16, but build the optimizer over fp32 master
        copies. Each step:
          1. The recipe all-reduces and clips bf16 grads on the model's
             ``adapter_optimizer_params`` (unchanged, line 1469-1500).
          2. ``_sync_grads_to_master_fp32()`` copies bf16 grads → fp32 master
             grads.
          3. ``optimizer.step()`` updates the fp32 masters.
          4. ``_sync_master_fp32_to_params()`` copies fp32 masters → bf16
             params (rounding back, but the accumulated update is preserved
             across steps in the fp32 masters).

        VALMET still measures the bf16 model params (what vLLM consumes), so
        a real, non-noise update will show as nonzero ``delta``.
        """
        params = adapter_optimizer_params(self._model)
        if not params:
            raise ValueError(
                "No adapter parameters found — check that lora_attn_modules is non-empty "
                "and that build_qwen3_lora_model / set_trainable_params ran correctly."
            )
        utils.log_rank_zero(log, f"LoRA optimizer: {len(params)} adapter parameter tensors")

        master_params = []
        self._adapter_master_pairs = []
        for _p in params:
            _master = _p.detach().to(torch.float32).clone()
            _master.requires_grad_(True)
            master_params.append(_master)
            self._adapter_master_pairs.append((_p, _master))
        _master_mb = sum(m.numel() * m.element_size() for m in master_params) / 1e6
        utils.log_rank_zero(
            log,
            f"LoRA optimizer: built {len(master_params)} fp32 master copies ({_master_mb:.1f} MB)",
        )

        optimizer = config.instantiate(cfg_optimizer, master_params)
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model, optimizer, opt_state_dict, self._device,
            )
        utils.log_rank_zero(log, "Optimizer is initialized (adapter-only, fp32 masters).")
        return optimizer

    def _sync_grads_to_master_fp32(self) -> None:
        """Copy bf16 adapter grads → fp32 master grads (after all-reduce/clip)."""
        with torch.no_grad():
            for _bf16_p, _fp32_master in self._adapter_master_pairs:
                if _bf16_p.grad is None:
                    if _fp32_master.grad is not None:
                        _fp32_master.grad.zero_()
                    continue
                if _fp32_master.grad is None:
                    _fp32_master.grad = _bf16_p.grad.detach().to(torch.float32).clone()
                else:
                    _fp32_master.grad.copy_(_bf16_p.grad.to(torch.float32))

    def _sync_master_fp32_to_params(self) -> None:
        """Copy fp32 master params → bf16 model params (after optimizer.step)."""
        with torch.no_grad():
            for _bf16_p, _fp32_master in self._adapter_master_pairs:
                _bf16_p.data.copy_(_fp32_master.data.to(_bf16_p.dtype))

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

        # Cache model attention dims for the Q/K un-permute path in
        # _gather_merged_lora_weights → _maybe_unpermute_qk (weight_sync.py).
        # Mirrors the base GRPO recipe; read from cfg.model so it works for any
        # architecture parameterization. No-op for non-permuting checkpointers.
        try:
            _cfg_model = cfg.get("model", {})
            _nh = _cfg_model.get("num_heads")
            _nkv = _cfg_model.get("num_kv_heads")
            _ed = _cfg_model.get("embed_dim")
            _hd = _cfg_model.get("head_dim")
            if _nh is not None:
                self._model_num_heads = int(_nh)
            if _nkv is not None:
                self._model_num_kv_heads = int(_nkv)
            elif _nh is not None:
                self._model_num_kv_heads = int(_nh)
            if _hd is not None:
                self._model_head_dim = int(_hd)
            elif _ed is not None and _nh is not None:
                self._model_head_dim = int(_ed) // int(_nh)
        except Exception as _dim_exc:
            log.warning(
                "Failed to cache model attention dims for wsync Q/K un-permute: %r",
                _dim_exc,
            )

        # Build LoRA model (no separate ref model)
        self._model = self._setup_model_lora(cfg, checkpoint_dict[training.MODEL_KEY])

        # Cache frozen base weights ONCE (collective). Eliminates the per-step
        # FSDP1 FULL_STATE_DICT gather inside _gather_merged_lora_weights, which
        # is the suspected trigger for trainer-side banned:1 PDE on Aurora XPU.
        # Base weights are frozen so the cache is bit-exact for all future steps.
        # Uses the merged-weight publish path; no-op for the legacy PEFT path.
        if not self._lora_use_runtime:
            self._cache_lora_base_weights()

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
        # Separate chunk size for no-grad forwards (ref_fwd inside disable_adapter,
        # optional rollout policy logprobs). Defaults to forward_batch_size for
        # backwards compatibility. Larger values (e.g. 16-24) reduce the number
        # of FSDP all-gather rounds in ref_fwd; safe to raise because no-grad
        # paths do not retain activations for backward.
        self._ref_forward_batch_size = cfg.get(
            "ref_forward_batch_size", cfg.forward_batch_size
        )
        # Outer-loop chunk for trajectory generation. Defaults to batch_size so
        # all G*batch_size streams hit vLLM in one HTTP call (max throughput).
        # Lower it (e.g. to forward_batch_size) only if the resulting tensors
        # don't fit in HBM. policy_fwd / ref_fwd remain chunked at fwd_bs inside
        # generate_trajectory regardless of this value.
        self._gen_batch_size = cfg.get("gen_batch_size", cfg.batch_size)
        self._ppo_epochs = cfg.ppo_epochs
        # Rollout-time policy logprobs gate. Mirrors dense recipe behavior:
        # required when ppo_epochs > 1 (multi-epoch updates mutate weights between
        # epochs so old_logprobs must come from the rollout-time policy) or when
        # explicitly forced via always_compute_rollout_logprobs (off-policy /
        # async setups). Otherwise we skip the no-grad policy fwd in
        # generate_trajectory and grpo_step falls back to chunk_pi_lp.detach()
        # (ratios collapse to 1, identical to pre-async behavior). Saves ~13%
        # of step time at 4B (policy_fwd ~8.3s out of 64.4s).
        self._always_compute_rollout_logprobs = cfg.get(
            "always_compute_rollout_logprobs", False
        )
        self._compute_rollout_logprobs_required = (
            self._ppo_epochs > 1 or self._always_compute_rollout_logprobs
        )
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

        # Validation instrumentation (opt-in; off by default to keep prod logs clean).
        # Set `lora.log_validation_metrics: true` in the YAML to log per-step
        # adapter L2 norm before/after optimizer.step() and the vLLM client _model_name
        # used for each generation. Used for the LoRA learning-validation ladder.
        _lora_cfg = cfg.get("lora", {})
        self._log_validation_metrics = bool(_lora_cfg.get("log_validation_metrics", False))

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
        """Rank-0-only adapter snapshot — no FSDP collective.

        Adapter params live in FSDP ``ignored_states`` (recipe init line ~412),
        are broadcast from rank 0 at init (line ~396), and grads are manually
        all-reduced post-step (line ~1546). VALMET_RANK_EQUAL pins bit-identical
        replication every step. Rank 0 can therefore read its own
        ``named_parameters()`` directly — no FULL_STATE_DICT gather of the
        frozen 4 B base. Saves ~7 s/step at the 4B/2-node envelope.

        Returns a tuple (peft_sd, adapter_config, adapter_name, slot, local_path, vllm_path)
        on rank 0, or None on non-rank-0 ranks.
        """
        if not self._is_rank_zero:
            return None

        # Adapter-only snapshot from local replicated params. Keys still carry
        # FSDP wrapping prefixes (use_orig_params=True), but
        # _translate_lora_key() strips them via _strip_fsdp_prefixes().
        adapter_sd = {
            n: p.detach().cpu().to(torch.float32)
            for n, p in self._model.named_parameters()
            if "lora_a" in n or "lora_b" in n
        }

        # Fail closed: count must match adapter_optimizer_params (the same
        # surface the optimizer trains and the manual all-reduce sweeps).
        expected = len(list(adapter_optimizer_params(self._model)))
        if len(adapter_sd) != expected or expected == 0:
            raise RuntimeError(
                f"_gather_lora_state_dict: adapter tensor count mismatch — "
                f"snapshot={len(adapter_sd)} vs adapter_optimizer_params={expected}. "
                f"Refusing to publish a partial adapter."
            )

        peft_sd, adapter_config = torchtune_to_peft_state_dict(
            adapter_sd,
            model_name=str(getattr(self._checkpointer, "_checkpoint_dir", "base_model")),
            rank=self._lora_rank,
            alpha=self._lora_alpha,
            target_modules=self._lora_target_modules,
        )
        del adapter_sd
        if not peft_sd:
            raise RuntimeError(
                "_gather_lora_state_dict: torchtune_to_peft_state_dict produced "
                "an empty PEFT state dict. Refusing to publish."
            )

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
    # Merged-weight publish path (default — sidesteps vLLM --enable-lora PDE)
    # -------------------------------------------------------------------------

    def _cache_lora_base_weights(self) -> None:
        """Cache frozen LoRA-target base weights on rank 0 (CPU) once at setup.

        Avoids the per-step FSDP1 ``state_dict_type(FULL_STATE_DICT, rank0_only=True)``
        gather inside ``_gather_merged_lora_weights``. The repeated per-step gather
        is a suspected trigger for trainer-side ``banned:1 PDE`` faults on Aurora
        XPU (observed on step 1 after step 0 publish; see
        ``project_lora_grpo_merged_weight_path.md``). Base weights are frozen
        (``requires_grad=False``) so a one-shot cache is bit-exact across all
        future steps.

        Sets ``self._cached_base_weights``: ``{tune_param_name: bf16_cpu_tensor}``
        on rank 0, ``None`` on other ranks.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )
        from torchtune.modules.peft.lora import LoRALinear

        # Collect names of LoRALinear base weights (these are what we need).
        target_names: set[str] = set()
        for mod_name, module in self._model.named_modules():
            if isinstance(module, LoRALinear):
                wn = f"{mod_name}.weight"
                target_names.add(wn)
                # Also store the wrapper-stripped form so iter_merged_lora_layers
                # lookups succeed on either name.
                target_names.add(
                    wn.replace("_fsdp_wrapped_module.", "").replace(
                        "_checkpoint_wrapped_module.", ""
                    )
                )

        _t0 = time.perf_counter()
        with FSDP.state_dict_type(
            self._model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_sd = self._model.state_dict()  # collective on every rank
            if not self._is_rank_zero:
                self._cached_base_weights = None
                return
            cached: dict[str, torch.Tensor] = {}
            for k, v in full_sd.items():
                # Filter to LoRA-target base weights; skip lora_a/lora_b (those
                # are not in the FSDP state_dict anyway since they're ignored,
                # but be defensive) and skip every other base linear that we
                # do not need to broadcast.
                if "lora_a" in k or "lora_b" in k:
                    continue
                if k in target_names:
                    cached[k] = v.detach().to(torch.bfloat16).cpu().contiguous()
            self._cached_base_weights = cached
        if self._is_rank_zero:
            _gb = sum(t.numel() * t.element_size() for t in self._cached_base_weights.values()) / 1024**3
            log.info(
                "LoRA-GRPO: cached %d frozen base weights on rank 0 (%.2f GiB CPU) in %.2fs — "
                "per-step FSDP gather eliminated",
                len(self._cached_base_weights), _gb, time.perf_counter() - _t0,
            )

    def _gather_merged_lora_weights(self) -> Optional[dict]:
        """Build merged LoRA-target weight dict from cached base + live adapter.

        Rank-0 only. Reads ``self._cached_base_weights`` (set by
        ``_cache_lora_base_weights`` at setup) and the live ``lora_a`` /
        ``lora_b`` adapter tensors (replicated across ranks via FSDP
        ``ignored_states`` + manual all-reduce in the train loop, so rank 0's
        copies are identical to all other ranks').

        Returns ``{hf_name: bf16_cpu_tensor}`` on rank 0; ``None`` elsewhere.

        No FSDP ``state_dict_type`` context is entered — there is no
        cross-rank collective in this function. This sidesteps the per-step
        ``FULL_STATE_DICT`` AllGather that was the suspected trigger for
        trainer-side ``banned:1 PDE`` on Aurora XPU.
        """
        if not self._is_rank_zero:
            return None
        if getattr(self, "_cached_base_weights", None) is None:
            raise RuntimeError(
                "_gather_merged_lora_weights: _cached_base_weights not initialized — "
                "_cache_lora_base_weights must be called once after FSDP wrap."
            )

        merged: dict[str, torch.Tensor] = {}
        # iter_merged_lora_layers reads adapter tensors live from the model
        # (lora_a/lora_b are replicated across ranks via ignored_states +
        # manual all-reduce, so rank 0's copy is authoritative). Base weights
        # come from the rank-0 cache; this is rank-0-only and never collective.
        for tune_name, w in iter_merged_lora_layers(
            self._model, base_weights=self._cached_base_weights
        ):
            clean = tune_name.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            import re as _re
            m = _re.match(r"^(?:.*\.)?layers\.(\d+)\.(.+)\.weight$", clean)
            if m is None:
                log.warning(
                    "Skipping unexpected merged-LoRA name (no match): %s", tune_name
                )
                continue
            layer_idx, module_path = m.group(1), m.group(2)
            hf_module = _TUNE_MODULE_TO_HF.get(module_path)
            if hf_module is None:
                log.warning(
                    "Skipping unknown LoRA module path %r in %s", module_path, tune_name
                )
                continue
            hf_name = f"model.layers.{layer_idx}.{hf_module}.weight"
            w = w.cpu().contiguous()
            # Invert the Llama-family Q/K permutation before handing to vLLM.
            # No-op unless this run uses a permuting checkpointer (LLAMA*).
            w = self._maybe_unpermute_qk(hf_name, w)
            merged[hf_name] = w
        return merged

    def _publish_merged_weights_background(self, hf_state_dict: dict) -> None:
        """Rank-0 background: write raw_bytes file, fan out POST to all vLLM tiles.

        Mirrors ``weight_sync._post_weights_to_vllm`` but is self-contained so
        the LoRA recipe doesn't depend on the dense recipe's bound-method
        plumbing (sender pool, _is_xccl_leader, _tune_to_hf_map, etc.).
        """
        import json
        import requests

        if not self._vllm_urls:
            log.warning("Rank 0: no vLLM URLs configured — skipping merged publish")
            return

        # Slot rotation: keep the last few writes around so a still-running
        # POST doesn't read a half-written file. _lora_max_loras already
        # bounds slot count for the legacy path; reuse it here.
        slot = self._steps_run % max(self._lora_max_loras, 1)
        save_path = os.path.join(
            self._lora_shm_root, f"merged_slot_{slot}", "weights.bin"
        )

        t_save0 = time.perf_counter()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        n_params = _save_raw_bytes(hf_state_dict, save_path)
        t_save = time.perf_counter() - t_save0
        size_gb = os.path.getsize(save_path) / 1024**3
        del hf_state_dict
        log.info(
            "Rank 0: merged adapter raw_bytes %d params %.2f GiB in %.2fs (%.2f GB/s) → %s",
            n_params, size_gb, t_save,
            (size_gb / t_save) if t_save > 0 else 0.0, save_path,
        )

        t_http0 = time.perf_counter()

        def _post_one(url: str):
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={"method": "load_weights_from_raw", "args": [save_path]},
                    timeout=600,
                )
                if r.status_code != 200:
                    log.warning(
                        "Merged-weight reload failed (%s): %s %s",
                        url, r.status_code, r.text[:200],
                    )
                    return
                results = r.json().get("results", [{}])
                first = results[0] if results else {}
                if isinstance(first, dict) and first.get("status") not in (None, "ok"):
                    log.warning("Merged-weight reload error (%s): %s", url, first)
            except Exception as _e:
                log.error("Merged-weight HTTP error (%s): %s", url, _e)

        with ThreadPoolExecutor(max_workers=max(1, len(self._vllm_urls))) as pool:
            for f in as_completed(
                [pool.submit(_post_one, u) for u in self._vllm_urls]
            ):
                f.result()
        t_http = time.perf_counter() - t_http0

        # Reset prefix cache so the new weights are not aliased by stale KV.
        if self._vllm_clients:
            with ThreadPoolExecutor(max_workers=len(self._vllm_clients)) as pool:
                list(pool.map(lambda c: c.reset_prefix_cache(), self._vllm_clients))

        log.info(
            "Rank 0: merged-weight publish: %d params, save=%.2fs http=%.2fs",
            n_params, t_save, t_http,
        )

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

        # Step 2: policy logprobs (adapter-enabled forward).
        # Only required when ppo_epochs > 1 or always_compute_rollout_logprobs
        # is set. In single-epoch sync GRPO (default), we skip this and let
        # grpo_step's per-chunk policy fwd produce both pi_logprobs and
        # old_logprobs (via .detach()) — ratios collapse to 1 either way.
        # Saves ~13% of step time at 4B.
        # Both no-grad paths (rollout policy logprobs, ref logprobs) chunk on
        # _ref_forward_batch_size — independent of training _forward_batch_size
        # because no activations are retained for backward here.
        fwd_bs = self._ref_forward_batch_size
        _policy_fwd_t0 = time.perf_counter()
        if self._compute_rollout_logprobs_required:
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
        else:
            logprobs = None
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

        # Step 4: truncate at first stop token.
        # truncate_sequence_at_first_stop_token only marks tokens *after* the first stop
        # as padding. vLLM may return a completion shorter than max_generated_tokens with
        # no stop token (length cutoff or stop fired but the kernel returned without
        # signaling); in that case the synthetic pad_id bytes we wrote at line 912 stay
        # padding_mask=False and would be counted as real tokens in loss/KL. OR the
        # pad-id positions in so they are excluded from the loss.
        (response_padding_masks, responses) = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )
        response_padding_masks = response_padding_masks | (responses == self._tokenizer.pad_id)

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

        # Advantages. batch_level_advantages (default) pools mean/std across the
        # full B*G batch so a single non-degenerate prompt keeps the signal alive
        # (matches base GRPO recipe). Legacy per-prompt path uses unbiased=False
        # so std at G=1 is 0 (not NaN). At B=1 the two are identical.
        if self._batch_level_advantages:
            from torchtune.dev.rl.rewards import batch_level_advantages
            advantages = batch_level_advantages(
                rewards.reshape(batch_size * grpo_size), group_size=grpo_size,
            )
        else:
            advantages = (rewards - rewards.mean(1, keepdim=True)) / (
                rewards.std(1, keepdim=True, unbiased=False) + 1e-4
            )
            advantages = advantages.reshape(batch_size * grpo_size)
        del responses
        device_empty_cache(self._device)

        # Mask padding (logprobs is None when rollout-time policy fwd was skipped).
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
        """Generate trajectories in gen_batch_size micro-batches.

        gen_batch_size defaults to batch_size (one vLLM HTTP call per step).
        policy/ref forwards inside generate_trajectory chunk independently at
        forward_batch_size for memory.
        """
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._gen_batch_size):
                batch_input_ids = input_ids[batch_start: batch_start + self._gen_batch_size]
                batch_answers = answers[batch_start: batch_start + self._gen_batch_size]
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

                    # Validation: log which adapter (or base model) vLLM will use this step.
                    if self._log_validation_metrics and self._is_rank_zero and self._vllm_clients:
                        _names = sorted({getattr(c, "_model_name", "<unset>") for c in self._vllm_clients})
                        log.info(
                            "VALMET step=%d vllm_model_names=%s (n_distinct=%d)",
                            self._steps_run, _names, len(_names),
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

                    # Validation: adapter L2 norm BEFORE optimizer step (rank 0 only).
                    # All ranks have identical adapter params (replicated + manual all-reduce
                    # above), so rank-0-only is sufficient.
                    _adapter_norm_before = None
                    if self._log_validation_metrics and self._is_rank_zero:
                        with torch.no_grad():
                            _ap = list(adapter_optimizer_params(self._model))
                            if _ap:
                                _sq = torch.zeros((), device=self._device, dtype=torch.float32)
                                for _p in _ap:
                                    _sq += _p.detach().float().pow(2).sum()
                                _adapter_norm_before = float(_sq.sqrt().item())

                    # Optimizer step (fp32 master copy path):
                    #   bf16 grads (already all-reduced + clipped) → fp32 master grads,
                    #   AdamW updates fp32 masters,
                    #   fp32 masters → bf16 model params.
                    # See _setup_optimizer for rationale.
                    _opt_t0 = time.perf_counter()
                    self._sync_grads_to_master_fp32()
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    self._sync_master_fp32_to_params()
                    # Also clear bf16 grads (recipe's _adapter_grads will rebuild next step).
                    for _bf16_p, _ in self._adapter_master_pairs:
                        if _bf16_p.grad is not None:
                            _bf16_p.grad = None
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _opt_time = time.perf_counter() - _opt_t0

                    # Validation: adapter L2 norm AFTER optimizer step + delta.
                    # Non-zero delta proves the optimizer actually updated adapter params
                    # (catches: dead grads, frozen params, ignored adapter list mismatch).
                    if self._log_validation_metrics and self._is_rank_zero and _adapter_norm_before is not None:
                        with torch.no_grad():
                            _ap = list(adapter_optimizer_params(self._model))
                            _sq = torch.zeros((), device=self._device, dtype=torch.float32)
                            for _p in _ap:
                                _sq += _p.detach().float().pow(2).sum()
                            _adapter_norm_after = float(_sq.sqrt().item())
                            # fp32 master L2 — proves AdamW actually moved the master copies
                            # even when bf16 round-trip truncates sub-ULP updates.
                            _msq = torch.zeros((), device=self._device, dtype=torch.float64)
                            for _, _m in self._adapter_master_pairs:
                                _msq += _m.detach().to(torch.float64).pow(2).sum()
                            _master_norm = float(_msq.sqrt().item())
                            # Also: fp32 grad L2 just before the step would be ideal but grads
                            # are already zero'd; instead snapshot the optimizer state-step.
                            _opt_step = 0
                            for _g in self._optimizer.param_groups:
                                for _p in _g["params"]:
                                    _st = self._optimizer.state.get(_p, {})
                                    _opt_step = int(_st.get("step", 0)) if "step" in _st else 0
                                    break
                                if _opt_step:
                                    break
                        log.info(
                            "VALMET step=%d adapter_l2_before=%.6e adapter_l2_after=%.6e delta=%.6e master_l2=%.6e opt_step=%d",
                            self._steps_run,
                            _adapter_norm_before,
                            _adapter_norm_after,
                            _adapter_norm_after - _adapter_norm_before,
                            _master_norm,
                            _opt_step,
                        )

                    # Validation: rank-equality checksum on adapter params.
                    # Adapter params are FSDP-ignored (replicated) and grads are
                    # manually all-reduced (~line 1444). After optimizer.step(), all
                    # ranks should hold bit-identical adapter weights. We verify by
                    # building a deterministic float64 hash on each rank, then
                    # all-reducing min/max and asserting they match.
                    #
                    # The opt-in bit is reused (lora.log_validation_metrics) so this
                    # only runs in the validation ladder; production runs stay quiet.
                    if self._log_validation_metrics and torch.distributed.get_world_size() > 1:
                        with torch.no_grad():
                            _ap_all = list(adapter_optimizer_params(self._model))
                            if _ap_all:
                                # Hash = sum-of-product-with-positional-weights in float64.
                                # Cheap, deterministic, sensitive to any single-element drift.
                                _hash = torch.zeros((), device=self._device, dtype=torch.float64)
                                for _i, _p in enumerate(_ap_all):
                                    _flat = _p.detach().to(torch.float64).flatten()
                                    _idx = torch.arange(_flat.numel(), device=self._device, dtype=torch.float64)
                                    _hash += (_flat * (_idx + float(_i + 1))).sum()
                                _h_min = _hash.clone()
                                _h_max = _hash.clone()
                                torch.distributed.all_reduce(_h_min, op=torch.distributed.ReduceOp.MIN)
                                torch.distributed.all_reduce(_h_max, op=torch.distributed.ReduceOp.MAX)
                                _spread = float((_h_max - _h_min).abs().item())
                                if self._is_rank_zero:
                                    if _spread > 0.0:
                                        log.error(
                                            "VALMET_RANK_DIVERGENCE step=%d hash_spread=%.6e "
                                            "(adapter params NOT identical across ranks — "
                                            "manual all-reduce or optimizer state diverged)",
                                            self._steps_run, _spread,
                                        )
                                    else:
                                        log.info(
                                            "VALMET_RANK_EQUAL step=%d hash=%.6e (all ranks match)",
                                            self._steps_run, float(_h_min.item()),
                                        )

                    # Adapter publish to vLLM. Two paths:
                    #   merged (default) — collective FSDP gather, rank 0 builds
                    #     merged W_eff state dict, raw_bytes file + POST. Sidesteps
                    #     vLLM --enable-lora PDE crash on Aurora XPU.
                    #   runtime (legacy) — rank-0 adapter snapshot, PEFT dir +
                    #     /v1/load_lora_adapter HTTP. Requires --enable-lora.
                    if self._steps_run % self._lora_publish_every == 0:
                        _pub_t0 = time.perf_counter()
                        if self._lora_use_runtime:
                            publish_state = self._gather_lora_state_dict()  # rank-0 only; non-rank-0 returns None
                            if self._is_rank_zero and publish_state is not None:
                                _gather_time = time.perf_counter() - _pub_t0
                                log.info(
                                    "Rank 0: adapter snapshot done in %.2fs — starting async publish",
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
                        else:
                            # Merged path: COLLECTIVE — every rank must enter the
                            # FSDP FULL_STATE_DICT context together. Non-rank-0 ranks
                            # get None back and skip the background spawn.
                            merged_sd = self._gather_merged_lora_weights()
                            if self._is_rank_zero and merged_sd is not None:
                                _gather_time = time.perf_counter() - _pub_t0
                                log.info(
                                    "Rank 0: merged-weight gather done in %.2fs (%d tensors) — starting async publish",
                                    _gather_time, len(merged_sd),
                                )
                                self._publish_error = None
                                _sd = merged_sd

                                def _bg(_s=_sd):
                                    try:
                                        self._publish_merged_weights_background(_s)
                                    except Exception as _e:
                                        self._publish_error = _e
                                        log.error("Rank 0: merged async publish failed: %s", _e)

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

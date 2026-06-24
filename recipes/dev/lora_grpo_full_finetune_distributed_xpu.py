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
from torchtune.modules.loss import RLLoss
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
    validate_vllm_mode,
    tune_lora_name_to_hf,
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

        # vLLM mode must be resolved BEFORE the process group is created:
        # colocate inits an in-process vLLM engine per rank, which builds its own
        # gloo sub-group and must run before init_xpu_process_group() makes CCL
        # the default backend (mirrors the dense recipe's _init_vllm_early call
        # site). The full LoRA config parse (publish mode, target modules, etc.)
        # still happens below — only the colocate-gating bits are hoisted here.
        self._vllm_mode = cfg.get("vllm_mode", "server")
        validate_vllm_mode(self._vllm_mode)
        self._colocate = (self._vllm_mode == "colocate")
        # LoRA recipe has no asym-optim spare-rank path; vLLM lives on every rank.
        self._vllm_ranks = None
        self._vllm_llm = None
        # Plain colocate only (no colocate_sleep in the LoRA recipe); pin the
        # sleep flag the shared init/generation helpers read.
        self._vllm_is_sleeping = False
        if self._colocate:
            self._init_vllm_early(cfg)

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
        # Merged-weight (Path A) transport. When BOTH are set, the ~6.77 GiB
        # merged weights.bin is written to train-node tmpfs, rsync'd ONCE to the
        # vLLM node over the HSN (reusing the persistent SSH ControlMaster), and
        # vLLM reads it from local tmpfs instead of contending on Lustre. When
        # unset (default), the file goes to _lora_shm_root (Lustre) and is read
        # cross-node from Lustre — the validated 2026-05-05 behavior.
        self._lora_merged_train_shm = _get("merged_train_shm", None)
        self._lora_merged_vllm_shm = _get("merged_vllm_shm", None)
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

        # Publish mode (supersedes use_runtime_lora; the latter is kept for
        # back-compat). Three values:
        #   "merged"  (default) — Path A: ship the full ~6.77 GiB W_eff every
        #             step via load_weights_from_raw. Bit-exact, but bandwidth-
        #             bound and on the critical path (~28s/step at 4B/2N).
        #   "delta"   — Path C (merge-at-receiver): ship the base ONCE (~6.77 GiB)
        #             then only the ~66 MB lora_a/lora_b each step; the vLLM
        #             worker re-merges W_eff = base + scale*(B@A) from a cached
        #             CPU base. Bit-exact to "merged", ~100x less per-step wire,
        #             no --enable-lora, any TP. See load_lora_delta_from_raw in
        #             torchtune/dev/vllm_weight_sync_worker.py.
        #   "runtime" — Path B: vLLM-native hot-swap (/v1/load_lora_adapter).
        #             Requires --enable-lora + the torch211 venv (TP=1 only).
        # Back-compat: use_runtime_lora=True forces "runtime" unless an explicit
        # publish_mode is given.
        _publish_mode = str(_get("publish_mode", "") or "").strip().lower()
        if not _publish_mode:
            _publish_mode = "runtime" if self._lora_use_runtime else "merged"
        if _publish_mode not in ("merged", "delta", "runtime"):
            raise ValueError(
                f"lora.publish_mode must be one of merged|delta|runtime, "
                f"got {_publish_mode!r}"
            )
        self._lora_publish_mode = _publish_mode
        # Keep use_runtime in sync so the rest of the recipe (PG setup, vLLM
        # client wiring) that branches on it stays correct.
        self._lora_use_runtime = (_publish_mode == "runtime")
        # Colocate overrides the publish mode: there is no HTTP/raw_bytes wire —
        # each rank merges W_eff and loads it into its OWN in-process engine.
        if self._colocate:
            self._lora_publish_mode = "colocate"
            self._lora_use_runtime = False
        # One-time base-ship guard for the delta path.
        self._lora_base_shipped = False

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

        # vLLM mode ('server' | 'colocate') was resolved earlier in __init__
        # (before the process group) so colocate could init its in-process
        # engine. self._vllm_mode / self._colocate are already set here.
        self._vllm_url = cfg.get("vllm_url", None)
        if self._vllm_url and "," in self._vllm_url:
            self._vllm_urls = [u.strip() for u in self._vllm_url.split(",")]
        elif self._vllm_url:
            self._vllm_urls = [self._vllm_url]
        else:
            self._vllm_urls = []
        self._vllm_group_port = cfg.get("vllm_group_port", 51216)
        # Single-replicate defaults for the shared _setup_vllm_server_mode helper
        # (vllm_backend.py). That helper grew HSDP attribute reads in 8b5f0f3f
        # (_dp_replicate / _is_shard_leader); this fork is server-mode,
        # single-replicate only and never set them, so setup() crashed with
        # AttributeError. Pin the non-HSDP values the base recipe uses on its
        # single-replicate path (data_parallel_replicate_dim=1).
        self._dp_replicate = 1
        self._is_shard_leader = self._is_rank_zero
        # vllm_weight_sync must be False for LoRA recipe (different sync path)
        self._vllm_weight_sync = False
        self._vllm_max_model_len = cfg.get("vllm_max_model_len", 2048)
        # Stop strings forwarded to vLLM for raw checkpoints that never emit EOS
        # (e.g. </answer>). Consumed by the shared vllm_http_generate helper.
        self._stop_strings = cfg.get("stop_strings", None)
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
        # Fail-fast on a failed adapter publish to vLLM. The merged-weight POST
        # path historically logged a warning and continued — training then ran
        # off-policy against STALE vLLM weights with no signal. Default True:
        # surface the failure so the next-step publish-thread join aborts.
        # Set lora.fail_on_publish_error: false for best-effort (legacy) behavior.
        self._fail_on_publish_error = bool(_get("fail_on_publish_error", True))
        # Publish/transfer timeouts (config-driven; defaults preserve prior behavior).
        self._publish_join_timeout = float(_get("publish_join_timeout", 120))
        self._collective_rpc_timeout = float(_get("collective_rpc_timeout", 600))
        self._load_lora_http_timeout = float(_get("load_lora_http_timeout", 120))
        self._rsync_timeout = float(_get("rsync_timeout", 60))
        self._ssh_mkdir_timeout = float(_get("ssh_mkdir_timeout", 30))

    # Inject vLLM server mode setup from shared backend module
    _setup_vllm_server_mode = _vllm_backend_module._setup_vllm_server_mode

    # Inject colocate vLLM early-init (TP=1 in-process engine per rank). Reused
    # verbatim from the dense recipe's backend; the new colocate YAML sets no
    # vllm.enable_lora, so _lora_engine_kwargs returns {} and the engine boots
    # exactly like the dense colocate path (frameworks stack, no --enable-lora).
    _init_vllm_early = _vllm_backend_module._init_vllm_early
    _init_vllm_tp1 = _vllm_backend_module._init_vllm_tp1
    _init_vllm_tp = _vllm_backend_module._init_vllm_tp

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

        # A100-equivalent plain-resident colocate path (TORCHTUNE_COLOCATE_NO_FSDP=1):
        # SKIP FSDP entirely. Rationale: for LoRA the only thing FSDP usefully shards
        # is the ~8 GiB frozen base, which fits resident on a 64 GiB tile alongside
        # vLLM (~24 GiB). FSDP is what BANS empty_cache on XPU (empty_cache + FSDP
        # leaks UR handles "proportional to FSDP units per call") — the single reason
        # reserved memory only grows on XPU where CUDA reclaims cleanly. Removing FSDP
        # in colocate lets us re-enable empty_cache between gen/train (see
        # generate_trajectory + grpo_step), matching the TRL/A100 plain colocate loop.
        # Model stays full-precision-replicated per tile; tiny adapter grads are
        # all-reduced manually in train() (already the case — adapters are FSDP-ignored
        # anyway). Only valid in colocate (server/dedicated still shard for cross-node).
        self._no_fsdp = (
            self._colocate
            and os.environ.get("TORCHTUNE_COLOCATE_NO_FSDP", "0") == "1"
        )
        if self._no_fsdp:
            utils.log_rank_zero(
                log,
                "LoRA-GRPO: TORCHTUNE_COLOCATE_NO_FSDP=1 — model NOT FSDP-wrapped "
                "(full base replicated per tile); empty_cache re-enabled in colocate. "
                "A100-equivalent plain-resident path.",
            )
            # Model already lives on self._device from _setup_model_lora; ensure it.
            model = model.to(self._device)
        else:
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

        # Server mode: rank 0 generates + broadcasts, so ALL ranks must see the
        # same batch (num_replicas=1, rank=0).
        # Colocate: each rank generates locally against its own in-process vLLM,
        # so each rank must see a DISTINCT prompt shard — partition the data
        # across the world (num_replicas=world_size, rank=self.rank). Without
        # this every rank would roll out identical prompts and the group-relative
        # advantages would be computed on duplicated data.
        if self._colocate:
            sampler = StatefulDistributedSampler(
                ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
        else:
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
        # Colocate skips the rank-0 CPU cache. Instead, if the cached-base colocate
        # path is enabled (default ON for colocate — it removes the per-step summon
        # that drives UR:40; set TORCHTUNE_COLOCATE_CACHED_BASE=0 to force the
        # legacy per-step-summon path for A/B), snapshot the FULL base per-rank once.
        self._colocate_base_cache = None
        if not self._lora_use_runtime and not self._colocate:
            self._cache_lora_base_weights()
        elif self._colocate:
            _cached_base_on = os.environ.get("TORCHTUNE_COLOCATE_CACHED_BASE", "1") == "1"
            if _cached_base_on:
                self._cache_colocate_base()

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
        # Memory-efficient chunked-vocab loss opt-in. LinearGRPOLoss (and any RLLoss
        # exposing set_model_output) takes the model's HIDDEN states and applies the
        # vocab projection per sequence-chunk inside the loss, so the full
        # [B, S, vocab] FP32 logit tensor (~2.7 GiB/seq at S~1900 for Qwen3-4B) is
        # never materialized. Detected here; wired after the model + RL params exist.
        # Default loss (GRPOLoss) lacks set_model_output -> _linear_loss=False ->
        # the existing full-logit path runs byte-for-byte unchanged.
        self._linear_loss = isinstance(self._loss_fn, RLLoss) and hasattr(
            self._loss_fn, "set_model_output"
        )

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

        # Wire the chunked-vocab loss to the model (after model + temperature exist).
        # set_model_output captures self._loss_fn.linear_projection = model.output and
        # sets model.skip_output_layer=True; we immediately reset it to False so ONLY
        # the training forward (which toggles it per-call in grpo_step) returns hidden
        # states — the ref/rollout forwards keep returning logits unchanged.
        if self._linear_loss:
            # The loss applies model.output OUTSIDE model.forward. Under FSDP
            # FULL_SHARD the projection weight is resharded after forward, so
            # projecting it in the loss would multiply by a shard (wrong numerics).
            # Only the no-FSDP colocate path keeps params resident -> safe. Fail
            # fast otherwise (override at your own risk for SHARD_GRAD_OP).
            if not self._no_fsdp and os.environ.get(
                "TORCHTUNE_LINEAR_LOSS_ALLOW_FSDP", "0"
            ) != "1":
                raise RuntimeError(
                    "LinearGRPOLoss (chunked-vocab) requires TORCHTUNE_COLOCATE_NO_FSDP=1 "
                    "(projection runs outside model.forward; FSDP FULL_SHARD reshards the "
                    "weight -> wrong numerics). Set TORCHTUNE_LINEAR_LOSS_ALLOW_FSDP=1 to "
                    "override (only safe under SHARD_GRAD_OP, untested)."
                )
            # LinearGRPOLoss computes logprobs via CE. Thread the recipe temperature
            # into it so its CE divides logits by T exactly like the standard
            # rlhf.logits_to_logprobs path (log_softmax(logits / T)). Without this the
            # policy logprobs would come from the wrong (T=1) distribution on any
            # sampling-temperature run. Inject onto the instance (the config loss block
            # does not carry temperature; it lives at the recipe top level).
            self._loss_fn.temperature = self._temperature
            utils.log_rank_zero(
                log,
                f"LoRA-GRPO: LinearGRPOLoss temperature set to {self._temperature} "
                "(matched to recipe temperature for correct policy logprobs).",
            )
            # LinearGRPOLoss uses the GRPOSimpleLoss formulation (ratios==1, NO
            # importance-sampling clip). At ppo_epochs==1 and on-policy rollouts
            # (always_compute_rollout_logprobs==False) the policy is the behavior
            # policy, so pi_old==pi.detach() and even GRPOLoss's clip is inert ->
            # bit-equivalent (CPU test). But with ppo_epochs>1 or off-policy/async
            # rollouts, pi diverges from pi_old and the IS clip MATTERS for stable
            # learning; LinearGRPOLoss would silently drop it. Fail fast so the
            # capability difference is never silent.
            _ppo_epochs_cfg = cfg.get("ppo_epochs", 1)
            _async_lp = cfg.get("always_compute_rollout_logprobs", False)
            if _ppo_epochs_cfg > 1 or _async_lp:
                raise RuntimeError(
                    "LinearGRPOLoss uses the simple (no IS-clip) GRPO formulation; it is "
                    f"only equivalent to GRPOLoss when ppo_epochs==1 AND "
                    f"always_compute_rollout_logprobs==False (got ppo_epochs={_ppo_epochs_cfg}, "
                    f"always_compute_rollout_logprobs={_async_lp}). In off-policy/multi-epoch "
                    "regimes the IS clip affects learning stability — use the full-logit "
                    "GRPOLoss path there."
                )
            # The per-call skip_output_layer toggle flips the model forward's return
            # type (logits<->hidden); under torch.compile that changes a guard every
            # step and forces recompiles (or guard failures). Not supported in Phase 1.
            if getattr(self, "_compile", False):
                raise RuntimeError(
                    "LinearGRPOLoss is incompatible with compile=True: the per-call "
                    "skip_output_layer toggle changes the forward return type and breaks "
                    "torch.compile guards. Disable compile or use the GRPOLoss path."
                )
            self._loss_fn.set_model_output(self._model)
            self._model.skip_output_layer = False  # re-enabled per-call in grpo_step
            utils.log_rank_zero(
                log,
                "LoRA-GRPO: LinearGRPOLoss wired (chunked-vocab, "
                f"num_output_chunks={getattr(self._loss_fn, 'num_output_chunks', '?')}); "
                "training forward returns hidden states, projection runs per seq-chunk "
                "in the loss. Ref/rollout forwards unchanged.",
            )
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

        # Run provenance stamp (rank 0, one-shot). Records the code version and
        # the RESOLVED training envelope so a logged number can always be traced
        # back to the config/code that produced it. Purely additive + defensive:
        # wrapped so a missing key or a missing git never affects training.
        if self._is_rank_zero:
            try:
                import subprocess as _subprocess

                _git_hash = (
                    _subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        stderr=_subprocess.DEVNULL,
                        timeout=5,
                    )
                    .decode()
                    .strip()
                    or "unknown"
                )
            except Exception:
                _git_hash = "unknown"
            try:
                _model_name = str(
                    cfg.get("model", {}).get("_component_", None)
                    or getattr(self._checkpointer, "_checkpoint_dir", "unknown")
                )
            except Exception:
                _model_name = "unknown"
            log.info(
                "RUN PROVENANCE | recipe=lora_grpo git=%s | model=%s "
                "vllm_mode=%s lora.publish_mode=%s | G(grpo_samples)=%s "
                "forward_batch_size=%s ref_forward_batch_size=%s "
                "gen_batch_size=%s batch_size=%s | fsdp_sharding_strategy=%s",
                _git_hash,
                _model_name,
                getattr(self, "_vllm_mode", "unknown"),
                getattr(self, "_lora_publish_mode", "unknown"),
                getattr(self, "grpo_samples", "unknown"),
                getattr(self, "_forward_batch_size", "unknown"),
                getattr(self, "_ref_forward_batch_size", "unknown"),
                getattr(self, "_gen_batch_size", "unknown"),
                getattr(self, "batch_size", "unknown"),
                cfg.get("fsdp_sharding_strategy", "SHARD_GRAD_OP(default)"),
            )

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

        # vLLM setup. Colocate already built an in-process TP=1 engine per rank
        # in __init__ (self._vllm_llm); it needs no HTTP clients, no adapter
        # dirs, no tmpfs/SSH transport — weights are loaded in-process at sync
        # time. Server mode wires the HTTP clients + transport scaffolding.
        if self._colocate:
            log.info(
                "Rank %d: colocate vLLM ready (in-process TP=1 engine); "
                "skipping server-mode client + transport setup",
                self.rank,
            )
            torch.distributed.barrier()
            return

        # vLLM server clients
        self._setup_vllm_server_mode()

        # Ensure adapter dirs exist (rank 0 only)
        if self._is_rank_zero:
            os.makedirs(self._lora_shm_root, exist_ok=True)
            log.info("LoRA checkpoint adapter dir: %s", self._lora_shm_root)
            # The merged-weight path (Path A) also uses tmpfs + rsync when
            # merged_train_shm/merged_vllm_shm are set — it needs the same
            # ControlMaster + dest-dir pre-creation set up here.
            _merged_tmpfs = bool(
                self._lora_merged_train_shm and self._lora_merged_vllm_shm
            )
            if self._lora_tmpfs_transfer or _merged_tmpfs:
                os.makedirs(self._lora_train_shm, exist_ok=True)
                if _merged_tmpfs:
                    os.makedirs(self._lora_merged_train_shm, exist_ok=True)
                log.info("LoRA publish: tmpfs_transfer=%s merged_tmpfs=%s — "
                         "train_shm=%s vllm_shm=%s merged_train_shm=%s merged_vllm_shm=%s",
                         self._lora_tmpfs_transfer, _merged_tmpfs,
                         self._lora_train_shm, self._lora_vllm_shm,
                         self._lora_merged_train_shm, self._lora_merged_vllm_shm)
                # tmpfs_transfer rsyncs node-local /dev/shm to a SINGLE vLLM host
                # (clients[0]). With multiple distinct vLLM hosts, the others would
                # silently serve a stale adapter dir. Refuse rather than train on
                # inconsistent adapters — use Lustre shm_root (tmpfs_transfer=False)
                # for multi-host vLLM.
                _distinct_hosts = {c.host for c in self._vllm_clients}
                if len(_distinct_hosts) > 1:
                    raise ValueError(
                        "lora.tmpfs_transfer=True is single-vLLM-host only, but the "
                        f"configured vLLM clients span {len(_distinct_hosts)} hosts "
                        f"({sorted(_distinct_hosts)}). Set tmpfs_transfer=False (use "
                        "the cross-node-visible Lustre lora.shm_root) for multi-host vLLM."
                    )
                # Pre-create vllm_shm parent(s) on VLLM_NODE so first rsync succeeds.
                # Include the merged dir when the merged-tmpfs path is active.
                if self._vllm_clients:
                    vllm_ip = self._vllm_clients[0].host
                    user = os.environ.get("USER", "")
                    dest = f"{user}@{vllm_ip}" if user else vllm_ip
                    _dirs = [self._lora_vllm_shm] if self._lora_tmpfs_transfer else []
                    if _merged_tmpfs:
                        _dirs.append(self._lora_merged_vllm_shm)
                    result = subprocess.run(
                        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                         dest, "mkdir -p " + " ".join(_dirs)],
                        capture_output=True, text=True, timeout=self._ssh_mkdir_timeout,
                    )
                    if result.returncode != 0:
                        log.warning("Failed to pre-create vllm_shm dirs on %s: %s", dest, result.stderr)
                    else:
                        log.info("Pre-created vllm_shm dirs on %s: %s", dest, _dirs)
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
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._rsync_timeout
            )
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
                    c.session, c.base_url, adapter_name, vllm_path,
                    self._load_lora_http_timeout,
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

    def _sync_colocated_lora_weights(self) -> None:
        """Merge W_eff per rank and load it into the rank's own in-process vLLM.

        Colocate publish path. Each training rank:
          1. summons its OWN full frozen base via ``FSDP.summon_full_params``
             (rank0_only=False) — required because under FSDP1 ``module.weight``
             is this rank's flat shard, which would shape-assert in vLLM;
          2. merges ``W_eff = base + (alpha/rank)*(B@A)`` per ``LoRALinear`` via
             ``iter_merged_lora_layers(model, base_weights=None)`` (the
             None path reads the now-full ``module.weight``); the adapter
             tensors are FSDP ``ignored_states`` (replicated) so every rank
             merges the identical W_eff;
          3. loads each merged LoRA-target weight into its OWN engine via
             ``load_weights`` — only LoRA-target weights are pushed; the frozen
             base for non-LoRA modules was loaded from disk at engine init.

        Bit-identical to the validated server merged path (same
        ``iter_merged_lora_layers`` + ``tune_lora_name_to_hf`` + Q/K unpermute),
        but with no HTTP, no rank-0 broadcast, no ``--enable-lora``. COLLECTIVE
        (summon) — the caller must invoke this on ALL ranks.
        """
        import gc
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        t0 = time.perf_counter()
        # Opt-in leak diagnostic: log free HBM before the sync each step so a
        # monotonic decline (UR-handle accumulation under summon+vLLM co-tenancy)
        # is visible in the log. Off by default (set TORCHTUNE_COLOCATE_MEM_PROBE=1).
        _mem_probe = os.environ.get("TORCHTUNE_COLOCATE_MEM_PROBE", "0") == "1"
        if _mem_probe and self._is_rank_zero and self._device.type == "xpu":
            try:
                _free, _total = torch.xpu.mem_get_info(self._device)
                log.info(
                    "COLOCATE_MEMPROBE step=%d pre-sync free=%.2f GiB reserved=%.2f GiB",
                    getattr(self, "_steps_run", -1),
                    _free / 1024**3,
                    torch.xpu.memory_reserved(self._device) / 1024**3,
                )
            except Exception as _mp_exc:
                log.warning("COLOCATE_MEMPROBE failed: %r", _mp_exc)
        llm_model = (
            self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
        n_synced = 0
        skipped = 0

        # Cached-base path (opt-in via TORCHTUNE_COLOCATE_CACHED_BASE=1, or whenever
        # _colocate_base_cache was populated at setup). The frozen base never
        # changes and the adapter is replicated (FSDP ignored_states), so there is
        # NO need to summon_full_params every step — re-all-gathering the full base
        # each step accumulates L0 IPC/UR handles (the exact mechanism the recipe
        # avoids for adapter params at setup) and is the suspected UR:40 driver at
        # ~10 steps. With the base cached full per-rank, merge reads the replicated
        # adapter directly: no per-step collective, no per-step handle churn.
        _use_cached_base = getattr(self, "_colocate_base_cache", None) is not None
        if _use_cached_base:
            for tune_name, merged in iter_merged_lora_layers(
                self._model, base_weights=self._colocate_base_cache
            ):
                hf_name = tune_lora_name_to_hf(tune_name)
                if hf_name is None:
                    skipped += 1
                    log.warning("colocate LoRA sync: no HF mapping for %s — skipping", tune_name)
                    continue
                w = self._maybe_unpermute_qk(hf_name, merged.contiguous())
                llm_model.load_weights([(hf_name, w)])
                n_synced += 1
                del w, merged
                if n_synced % 5 == 0 and torch.xpu.is_available():
                    gc.collect()
                    torch.xpu.synchronize(self._device)
        else:
            import contextlib as _ctxlib
            # No-FSDP: weights are full+resident, read module.weight directly (no
            # summon). FSDP: summon the full sharded param for the merge.
            _summon_ctx = (
                _ctxlib.nullcontext()
                if getattr(self, "_no_fsdp", False)
                else FSDP.summon_full_params(self._model, writeback=False, rank0_only=False)
            )
            with torch.no_grad(), _summon_ctx:
                for tune_name, merged in iter_merged_lora_layers(
                    self._model, base_weights=None
                ):
                    hf_name = tune_lora_name_to_hf(tune_name)
                    if hf_name is None:
                        skipped += 1
                        log.warning(
                            "colocate LoRA sync: no HF mapping for %s — skipping", tune_name
                        )
                        continue
                    # Invert Llama-family Q/K permutation before vLLM (no-op Qwen3).
                    w = self._maybe_unpermute_qk(hf_name, merged.contiguous())
                    llm_model.load_weights([(hf_name, w)])
                    n_synced += 1
                    del w, merged
                    # gc + sync every 5 params to bound UR-handle pressure from the
                    # summon all-gathers before the next FSDP backward.
                    if n_synced % 5 == 0 and torch.xpu.is_available():
                        gc.collect()
                        torch.xpu.synchronize(self._device)

        self._vllm_llm.llm_engine.reset_prefix_cache()
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.synchronize(self._device)

        if self._is_rank_zero:
            log.info(
                "Rank 0: colocate LoRA sync %d merged weights in %.2fs (skipped=%d, path=%s)",
                n_synced, time.perf_counter() - t0, skipped,
                "cached_base" if _use_cached_base else "summon",
            )

    def _cache_colocate_base(self) -> None:
        """Summon the frozen base ONCE and cache the full per-rank tensors.

        Populates ``self._colocate_base_cache`` = ``{module_name}.weight ->
        full bf16 base tensor on self._device`` for every LoRALinear. After this,
        ``_sync_colocated_lora_weights`` merges from the cache with NO per-step
        ``summon_full_params`` — eliminating the per-step all-gather that
        accumulates L0 UR/IPC handles (the suspected UR:40 driver at ~10 steps).

        The base is frozen, so a one-time snapshot is bit-exact for all steps. Keys
        match ``iter_merged_lora_layers``' ``base_weights`` contract
        (``{module_name}.weight``, FSDP/ckpt prefixes stripped). Cached on-device by
        default (4B base ~8 GiB bf16; tile has ~56 GiB free alongside vLLM); set
        ``TORCHTUNE_COLOCATE_BASE_CPU=1`` to cache on CPU if HBM is tight (adds a
        per-step H2D copy of each base inside the merge).
        """
        import contextlib
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torchtune.modules.peft.lora import LoRALinear

        _cpu = os.environ.get("TORCHTUNE_COLOCATE_BASE_CPU", "0") == "1"
        dst = torch.device("cpu") if _cpu else self._device
        cache: dict[str, torch.Tensor] = {}
        t0 = time.perf_counter()
        # No-FSDP path: weights are full+resident already, no summon needed (and the
        # _fsdp_wrapped_module. prefix is absent). FSDP path: summon the full param.
        _summon = (
            contextlib.nullcontext()
            if getattr(self, "_no_fsdp", False)
            else FSDP.summon_full_params(self._model, writeback=False, rank0_only=False)
        )
        with torch.no_grad(), _summon:
            for module_name, module in self._model.named_modules():
                if not isinstance(module, LoRALinear):
                    continue
                clean = module_name.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                key = f"{clean}.weight"
                # .clone() is LOAD-BEARING: under FSDP1 summon_full_params the full
                # weight lives in a TEMPORARY buffer that is freed when the summon
                # context exits. For an on-XPU cache (dst==xpu), `.to(xpu)` on an
                # already-XPU tensor is a no-op returning the SAME storage, so the
                # cache would alias the summoned buffer → use-after-free → PML4
                # NotPresent-Read banned:1 at the first post-cache step (validated
                # 2026-06-18: on-XPU base crashed step 2 at BOTH 4B/0.82-50GiB-free
                # AND 0.6B — size-independent, so a lifetime bug not an OOM). The CPU
                # path happened to be safe because cross-device .to() always copies.
                # .clone() forces fresh storage on either device.
                cache[key] = (
                    module.weight.detach().to(torch.bfloat16).to(dst).contiguous().clone()
                )
        self._colocate_base_cache = cache
        if self._is_rank_zero:
            _gb = sum(t.numel() * t.element_size() for t in cache.values()) / 1024**3
            log.info(
                "LoRA-GRPO colocate: cached %d full base weights per-rank (%.2f GiB on %s) "
                "in %.2fs — per-step summon eliminated",
                len(cache), _gb, "cpu" if _cpu else "xpu", time.perf_counter() - t0,
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
            hf_name = tune_lora_name_to_hf(tune_name)
            if hf_name is None:
                log.warning(
                    "Skipping unexpected merged-LoRA name (no HF mapping): %s", tune_name
                )
                continue
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
        _use_tmpfs = bool(
            self._lora_merged_train_shm and self._lora_merged_vllm_shm
        )
        if _use_tmpfs:
            # Write to train-node tmpfs; vLLM reads from its own tmpfs after the
            # rsync below. save_path = what we write + rsync FROM; vllm_path =
            # what the POST tells vLLM to read.
            save_path = os.path.join(
                self._lora_merged_train_shm, f"merged_slot_{slot}", "weights.bin"
            )
            vllm_path = os.path.join(
                self._lora_merged_vllm_shm, f"merged_slot_{slot}", "weights.bin"
            )
        else:
            save_path = os.path.join(
                self._lora_shm_root, f"merged_slot_{slot}", "weights.bin"
            )
            vllm_path = save_path  # Lustre is cross-node visible

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

        # Cross-node transport: rsync the file to the vLLM node's tmpfs ONCE,
        # reusing the persistent SSH ControlMaster. No-op in the Lustre default.
        if _use_tmpfs and self._vllm_clients:
            vllm_ip = self._vllm_clients[0].host
            user = os.environ.get("USER", "")
            dest = (
                f"{user}@{vllm_ip}:{os.path.dirname(vllm_path)}/"
                if user else f"{vllm_ip}:{os.path.dirname(vllm_path)}/"
            )
            if self._ssh_control_socket:
                ssh_cmd = (
                    "ssh -o StrictHostKeyChecking=no -o BatchMode=yes "
                    f"-o ControlMaster=no -o ControlPath={self._ssh_control_socket}"
                )
            else:
                ssh_cmd = "ssh -o StrictHostKeyChecking=no -o BatchMode=yes"
            subprocess.run(
                ["ssh"] + ssh_cmd.split()[1:] + [
                    f"{user}@{vllm_ip}" if user else vllm_ip,
                    f"mkdir -p {os.path.dirname(vllm_path)}",
                ],
                capture_output=True, text=True, timeout=self._ssh_mkdir_timeout,
            )
            t_rsync = time.perf_counter()
            _r = subprocess.run(
                ["rsync", "-a", "--inplace", "-e", ssh_cmd, save_path, dest],
                capture_output=True, text=True, timeout=self._rsync_timeout,
            )
            if _r.returncode != 0:
                raise RuntimeError(
                    f"merged-weight rsync to {dest} failed "
                    f"(rc={_r.returncode}): {_r.stderr}"
                )
            log.info(
                "Rank 0: rsync merged weights.bin (%.2f GiB) to %s in %.2fs",
                size_gb, dest, time.perf_counter() - t_rsync,
            )

        t_http0 = time.perf_counter()

        def _post_one(url: str) -> Optional[str]:
            """Return None on success, or an error string on failure."""
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={"method": "load_weights_from_raw", "args": [vllm_path]},
                    timeout=self._collective_rpc_timeout,
                )
                if r.status_code != 200:
                    msg = f"{url}: HTTP {r.status_code} {r.text[:200]}"
                    log.warning("Merged-weight reload failed (%s)", msg)
                    return msg
                results = r.json().get("results", [{}])
                first = results[0] if results else {}
                if isinstance(first, dict) and first.get("status") not in (None, "ok"):
                    msg = f"{url}: {first}"
                    log.warning("Merged-weight reload error (%s)", msg)
                    return msg
                return None
            except Exception as _e:
                msg = f"{url}: {_e!r}"
                log.error("Merged-weight HTTP error (%s)", msg)
                return msg

        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, len(self._vllm_urls))) as pool:
            for f in as_completed(
                [pool.submit(_post_one, u) for u in self._vllm_urls]
            ):
                err = f.result()
                if err is not None:
                    failed.append(err)
        t_http = time.perf_counter() - t_http0
        if failed:
            _summary = (
                f"Merged-weight publish failed on {len(failed)}/"
                f"{len(self._vllm_urls)} vLLM tiles: {failed}"
            )
            if self._fail_on_publish_error:
                # Raise so the _bg wrapper records _publish_error and the
                # next-step join aborts before generating off-policy rollouts.
                raise RuntimeError(_summary)
            log.error("%s — continuing (fail_on_publish_error=False)", _summary)

        # Reset prefix cache so the new weights are not aliased by stale KV.
        if self._vllm_clients:
            with ThreadPoolExecutor(max_workers=len(self._vllm_clients)) as pool:
                list(pool.map(lambda c: c.reset_prefix_cache(), self._vllm_clients))

        log.info(
            "Rank 0: merged-weight publish: %d params, save=%.2fs http=%.2fs",
            n_params, t_save, t_http,
        )

    # -------------------------------------------------------------------------
    # Delta publish path (Path C — merge-at-receiver)
    # -------------------------------------------------------------------------
    #
    # Instead of shipping the full ~6.77 GiB merged W_eff every step (Path A),
    # ship the frozen base ONCE (~6.77 GiB, reusing _cached_base_weights) and
    # then only the ~66 MB lora_a/lora_b adapter each step. The vLLM worker
    # re-merges W_eff = base + scale*(B@A) from a cached CPU base, then routes
    # the result through the SAME model.load_weights() call Path A uses — so the
    # placement (and thus the resident weights) is bit-identical to Path A.
    #
    # Q/K un-permute (LLAMA-family) is applied SENDER-side, to BOTH the one-time
    # base and the per-step delta. This works because un-permute is a row
    # permutation on the output dim, so
    #     unpermute(base + scale*(B@A)) == unpermute(base) + unpermute(scale*(B@A))
    # The receiver just adds the two already-unpermuted tensors and needs no
    # checkpointer / head-dim knowledge. No-op for Qwen3 (non-permuting).

    def _gather_lora_base_payload(self) -> Optional[dict]:
        """Build the one-time base-weight payload for the delta path.

        Reuses ``self._cached_base_weights`` (the frozen LoRA-target base
        weights gathered once at setup by ``_cache_lora_base_weights``), renamed
        tune->HF and Q/K-unpermuted exactly as ``_gather_merged_lora_weights``
        does for the merged W_eff. Returns ``{hf_name: bf16_cpu_tensor}`` on
        rank 0, ``None`` elsewhere. Shipped once; the receiver caches it.
        """
        if not self._is_rank_zero:
            return None
        if getattr(self, "_cached_base_weights", None) is None:
            raise RuntimeError(
                "_gather_lora_base_payload: _cached_base_weights not initialized — "
                "_cache_lora_base_weights must be called once after FSDP wrap."
            )
        import re as _re
        from torchtune.modules.peft.lora import LoRALinear

        # Map each LoRALinear's base weight (tune name) -> HF name, reusing the
        # cached base tensor. Iterating LoRALinear modules (not the cache dict)
        # keeps the HF-name derivation identical to the merged path.
        base: dict[str, torch.Tensor] = {}
        for mod_name, module in self._model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            tune_name = f"{mod_name}.weight"
            clean = tune_name.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            base_w = self._cached_base_weights.get(tune_name)
            if base_w is None:
                base_w = self._cached_base_weights.get(clean)
            if base_w is None:
                raise KeyError(
                    f"_gather_lora_base_payload: base weight for {tune_name!r} "
                    f"missing from _cached_base_weights"
                )
            m = _re.match(r"^(?:.*\.)?layers\.(\d+)\.(.+)\.weight$", clean)
            if m is None:
                log.warning("Skipping base weight (no layer match): %s", tune_name)
                continue
            layer_idx, module_path = m.group(1), m.group(2)
            hf_module = _TUNE_MODULE_TO_HF.get(module_path)
            if hf_module is None:
                log.warning("Skipping unknown base module path %r in %s", module_path, tune_name)
                continue
            hf_name = f"model.layers.{layer_idx}.{hf_module}.weight"
            w = base_w.to(torch.bfloat16).cpu().contiguous()
            w = self._maybe_unpermute_qk(hf_name, w)
            base[hf_name] = w
        return base

    def _gather_lora_delta_payload(self) -> Optional[tuple]:
        """Rank-0-only adapter snapshot for the delta path. No FSDP collective.

        Reads the live (replicated, bf16) ``lora_a``/``lora_b`` weights of every
        ``LoRALinear`` — the same tensors the merged path consumes via
        ``iter_merged_lora_layers`` — and emits a flat tensor dict plus a JSON
        ``meta`` mapping each base HF weight name to its A/B keys + scale, so the
        receiver can compute ``delta = scale * (B @ A)`` and add it to the cached
        base. Q/K un-permute is NOT applied here (the per-layer delta is small);
        it is applied by the receiver-side merge in ``load_lora_delta_from_raw``
        for q/k — but because we ship the base ALREADY unpermuted, we must also
        unpermute the delta. To keep the receiver checkpointer-agnostic we
        unpermute the delta SENDER-side. Since A/B are shipped raw, the meta
        carries an ``unpermute`` spec per q/k entry and head dims.

        Returns ``(tensors, meta)`` on rank 0, where ``tensors`` is
        ``{key: bf16_cpu_tensor}`` (keys ``<hf>::lora_A`` / ``<hf>::lora_B``)
        and ``meta`` is a JSON-serializable dict; ``None`` on other ranks.
        """
        if not self._is_rank_zero:
            return None
        import re as _re
        from torchtune.modules.peft.lora import LoRALinear

        tensors: dict[str, torch.Tensor] = {}
        entries: list[dict] = []
        n_modules = 0
        for mod_name, module in self._model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            n_modules += 1
            tune_name = f"{mod_name}.weight"
            clean = tune_name.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            m = _re.match(r"^(?:.*\.)?layers\.(\d+)\.(.+)\.weight$", clean)
            if m is None:
                log.warning("Skipping adapter (no layer match): %s", tune_name)
                continue
            layer_idx, module_path = m.group(1), m.group(2)
            hf_module = _TUNE_MODULE_TO_HF.get(module_path)
            if hf_module is None:
                log.warning("Skipping unknown adapter module path %r in %s", module_path, tune_name)
                continue
            hf_name = f"model.layers.{layer_idx}.{hf_module}.weight"
            a_w = module.lora_a.weight.detach().to(torch.bfloat16).cpu().contiguous()
            b_w = module.lora_b.weight.detach().to(torch.bfloat16).cpu().contiguous()
            a_key = f"{hf_name}::lora_A"
            b_key = f"{hf_name}::lora_B"
            tensors[a_key] = a_w
            tensors[b_key] = b_w
            entries.append({
                "hf_name": hf_name,
                "a_key": a_key,
                "b_key": b_key,
                "scale": float(module.alpha) / float(module.rank),
            })

        # Fail closed: one A and one B per LoRALinear.
        if n_modules == 0 or len(tensors) != 2 * len(entries) or len(entries) != n_modules:
            raise RuntimeError(
                f"_gather_lora_delta_payload: adapter count mismatch — "
                f"modules={n_modules} entries={len(entries)} tensors={len(tensors)}. "
                f"Refusing to publish a partial adapter."
            )

        # Q/K un-permute spec for the receiver to apply to the assembled delta.
        # Mirrors _maybe_unpermute_qk's gate (LLAMA-family checkpointers only).
        needs_unpermute = bool(self._needs_qk_unpermute())
        meta = {
            "entries": entries,
            "needs_qk_unpermute": needs_unpermute,
            "num_heads": int(getattr(self, "_model_num_heads", 0) or 0),
            "num_kv_heads": int(getattr(self, "_model_num_kv_heads", 0) or 0),
            "head_dim": int(getattr(self, "_model_head_dim", 0) or 0),
        }
        return tensors, meta

    def _publish_lora_delta_background(self, payload: tuple) -> None:
        """Rank-0 background: ship base once (if needed) + per-step adapter delta.

        Mirrors ``_publish_merged_weights_background``'s transport (slot dir,
        optional tmpfs rsync, ThreadPool POST fan-out, fail-fast on error,
        prefix-cache reset) but ships ~66 MB/step instead of ~6.77 GiB.
        """
        import json
        import requests

        if not self._vllm_urls:
            log.warning("Rank 0: no vLLM URLs configured — skipping delta publish")
            return

        tensors, meta = payload

        # --- One-time base ship -------------------------------------------------
        if not self._lora_base_shipped:
            base_sd = self._gather_lora_base_payload()
            if base_sd is None:
                raise RuntimeError("delta publish: base payload empty on rank 0")
            base_path = os.path.join(self._lora_shm_root, "delta_base", "base.bin")
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            _t0 = time.perf_counter()
            n_base = _save_raw_bytes(base_sd, base_path)
            _base_gb = os.path.getsize(base_path) / 1024**3
            del base_sd
            log.info(
                "Rank 0: delta base raw_bytes %d params %.2f GiB in %.2fs → %s",
                n_base, _base_gb, time.perf_counter() - _t0, base_path,
            )
            self._post_collective_rpc(
                "load_lora_base_from_raw", [base_path],
                what="delta base", n=n_base,
            )
            self._lora_base_shipped = True

        # --- Per-step adapter delta --------------------------------------------
        slot = self._steps_run % max(self._lora_max_loras, 1)
        save_path = os.path.join(self._lora_shm_root, f"delta_slot_{slot}", "adapter.bin")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t_save0 = time.perf_counter()
        n_params = _save_raw_bytes(tensors, save_path)
        t_save = time.perf_counter() - t_save0
        size_mb = os.path.getsize(save_path) / 1024**2
        del tensors
        log.info(
            "Rank 0: delta adapter raw_bytes %d tensors %.1f MB in %.2fs (%.2f GB/s) → %s",
            n_params, size_mb, t_save,
            (size_mb / 1024 / t_save) if t_save > 0 else 0.0, save_path,
        )

        meta_json = json.dumps(meta)
        t_http0 = time.perf_counter()
        self._post_collective_rpc(
            "load_lora_delta_from_raw", [save_path, meta_json],
            what="delta adapter", n=n_params,
        )
        t_http = time.perf_counter() - t_http0

        # Reset prefix cache so the new weights are not aliased by stale KV.
        if self._vllm_clients:
            with ThreadPoolExecutor(max_workers=len(self._vllm_clients)) as pool:
                list(pool.map(lambda c: c.reset_prefix_cache(), self._vllm_clients))

        log.info(
            "Rank 0: delta publish: %d tensors, save=%.2fs http=%.2fs",
            n_params, t_save, t_http,
        )

    def _post_collective_rpc(self, method: str, args: list, what: str, n: int) -> None:
        """POST a /collective_rpc {method,args} to all vLLM URLs, fail-fast.

        Shared by the delta base-ship and per-step delta publish. Honors
        ``self._collective_rpc_timeout`` and ``self._fail_on_publish_error``
        (same contract as ``_publish_merged_weights_background``).
        """
        import requests

        def _post_one(url: str) -> Optional[str]:
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={"method": method, "args": args},
                    timeout=self._collective_rpc_timeout,
                )
                if r.status_code != 200:
                    msg = f"{url}: HTTP {r.status_code} {r.text[:200]}"
                    log.warning("%s reload failed (%s)", what, msg)
                    return msg
                results = r.json().get("results", [{}])
                first = results[0] if results else {}
                if isinstance(first, dict) and first.get("status") not in (None, "ok"):
                    msg = f"{url}: {first}"
                    log.warning("%s reload error (%s)", what, msg)
                    return msg
                return None
            except Exception as _e:
                msg = f"{url}: {_e!r}"
                log.error("%s HTTP error (%s)", what, msg)
                return msg

        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, len(self._vllm_urls))) as pool:
            for f in as_completed([pool.submit(_post_one, u) for u in self._vllm_urls]):
                err = f.result()
                if err is not None:
                    failed.append(err)
        if failed:
            _summary = (
                f"{what} publish failed on {len(failed)}/{len(self._vllm_urls)} "
                f"vLLM tiles: {failed}"
            )
            if self._fail_on_publish_error:
                raise RuntimeError(_summary)
            log.error("%s — continuing (fail_on_publish_error=False)", _summary)

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _generate_with_colocated_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Generate using this rank's in-process colocated vLLM engine.

        With a DistributedSampler (num_replicas=world_size), each rank holds its
        own subset of prompts and generates ALL grpo_samples completions locally
        — no cross-rank communication. Copied from the dense recipe's colocate
        path so it closes over THIS module's ``_xpu_device_index``.

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

        # Strip padding; truncate prompt so prompt_len + gen_len never overflows
        # the vLLM block table (max_model_len).
        max_prompt_len = self._vllm_max_model_len - self._max_generated_tokens
        raw_prompts = []
        for i in range(bsz):
            ids = batch_input_ids[i].cpu().tolist()
            ids = [t for t in ids if t != self._tokenizer.pad_id]
            raw_prompts.append(ids[-max_prompt_len:] if len(ids) > max_prompt_len else ids)
        vllm_prompts = [{"prompt_token_ids": p} for p in raw_prompts]

        t0 = time.perf_counter()
        outputs = self._vllm_llm.generate(
            prompts=vllm_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        gen_time = time.perf_counter() - t0

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

        # vLLM may have shifted the default XPU device; restore ours.
        if self._device.type == "xpu":
            torch.xpu.set_device(_xpu_device_index)
            torch.xpu.synchronize()

        return query_responses

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
        """Rank-0-only vLLM HTTP round-trip.

        Delegates to the shared ``vllm_http_generate`` helper so this recipe
        gets the same prompt-truncation, stop-token/stop-string forwarding, and
        EOS-injection behavior as the dense GRPO recipe (previously this fork's
        inline copy was missing all three — a silent learning-signal gap for
        raw checkpoints, mirroring the AGPT-2B Stage-1 fix).
        """
        from torchtune.dev.rl.vllm_client import vllm_http_generate

        return vllm_http_generate(
            batch_input_ids,
            context_length,
            vllm_clients=self._vllm_clients,
            pad_id=self._tokenizer.pad_id,
            eos_id=self._tokenizer.eos_id,
            max_generated_tokens=self._max_generated_tokens,
            vllm_max_model_len=self._vllm_max_model_len,
            temperature=self._temperature,
            top_k=self._top_k,
            stop_token_ids=getattr(self, "_stop_token_ids", None),
            stop_strings=getattr(self, "_stop_strings", None),
            device=self._device,
        )

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
        """Generate one GRPO trajectory (server or colocate, LoRA adapter)."""
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        # device_empty_cache leaks UR handles under FSDP + in-process vLLM, so
        # it must NOT run in FSDP colocate (the dense colocate path skips it too).
        # The no-FSDP colocate path DOES reclaim (the whole hypothesis) via
        # _colocate_reclaim; off-colocate uses the standard helper.
        if not self._colocate:
            device_empty_cache(self._device)
        else:
            self._colocate_reclaim()

        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)
        num_seqs = batch_size * grpo_size

        # Step 1: vLLM generation. Colocate generates in-process per rank;
        # server mode calls the HTTP tiles (adapter-aware via client._model_name).
        # Sub-phase creep probe (MEM_PROBE=1): reserved before/after JUST the vLLM
        # generate call, to localize the ~0.5 GiB/step gen-phase floor creep to
        # vLLM-internal growth vs the torch ref/policy forwards that follow.
        _subprobe = (
            os.environ.get("TORCHTUNE_COLOCATE_MEM_PROBE", "0") == "1"
            and self._is_rank_zero and self._device.type == "xpu"
        )
        if _subprobe and self._device.type == "xpu":
            torch.xpu.synchronize()
        _r_pre_vllm = torch.xpu.memory_reserved(self._device) / 1024**3 if _subprobe else 0.0
        _a_pre_vllm = torch.xpu.memory_allocated(self._device) / 1024**3 if _subprobe else 0.0
        _vllm_t0 = time.perf_counter()
        if self._colocate:
            query_responses = self._generate_with_colocated_vllm(
                batch_input_ids, context_length
            )
        else:
            query_responses = self._generate_with_vllm(batch_input_ids, context_length)
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _vllm_time = time.perf_counter() - _vllm_t0
        if _subprobe:
            # ACTIVE (live) is the discriminator — the gen-phase +0.44/step creep
            # (job 8558618) is in ACTIVE; reserved was blind. Pins whether the leak
            # is the vLLM generate call itself or the torch forwards after it.
            _r_post_vllm = torch.xpu.memory_reserved(self._device) / 1024**3
            _a_post_vllm = torch.xpu.memory_allocated(self._device) / 1024**3
            log.info(
                "COLOCATE_SUBPROBE step=%d vllm_generate reserved %.2f->%.2f (%+.2f) "
                "| ACTIVE %.2f->%.2f (%+.3f) GiB",
                getattr(self, "_steps_run", -1), _r_pre_vllm, _r_post_vllm,
                _r_post_vllm - _r_pre_vllm,
                _a_pre_vllm, _a_post_vllm, _a_post_vllm - _a_pre_vllm,
            )

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
        if _subprobe and self._device.type == "xpu":
            torch.xpu.synchronize()
        _r_pre_ref = torch.xpu.memory_reserved(self._device) / 1024**3 if _subprobe else 0.0
        _a_pre_ref = torch.xpu.memory_allocated(self._device) / 1024**3 if _subprobe else 0.0
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
        if _subprobe:
            _r_post_ref = torch.xpu.memory_reserved(self._device) / 1024**3
            _a_post_ref = torch.xpu.memory_allocated(self._device) / 1024**3
            log.info(
                "COLOCATE_SUBPROBE step=%d ref_fwd reserved %.2f->%.2f (%+.2f) "
                "| ACTIVE %.2f->%.2f (%+.3f) GiB",
                getattr(self, "_steps_run", -1), _r_pre_ref, _r_post_ref,
                _r_post_ref - _r_pre_ref,
                _a_pre_ref, _a_post_ref, _a_post_ref - _a_pre_ref,
            )
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
        self._colocate_reclaim()

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
                self._colocate_reclaim()
                trajectories.append(self.generate_trajectory(batch_input_ids, batch_answers))
                device_empty_cache(self._device)
                self._colocate_reclaim()

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
                if getattr(self, "_linear_loss", False):
                    # Chunked-vocab path: model returns HIDDEN states (skip_output_layer
                    # toggled True ONLY around this training forward); LinearGRPOLoss
                    # applies the vocab projection per seq-chunk, so no full [B,S,vocab]
                    # FP32 logit tensor is ever held. The loss reads pi_old from the
                    # hidden it re-projects (ratios collapse to 1, simple formulation).
                    try:
                        self._model.skip_output_layer = True
                        chunk_hidden = self._model(
                            trajectory.query_responses[cs:ce],
                            input_pos=trajectory.position_ids[cs:ce],
                            mask=trajectory.masks[cs:ce] if trajectory.masks is not None else None,
                        )
                    finally:
                        self._model.skip_output_layer = False
                    chunk_hidden = rlhf.truncate_sequence_for_logprobs(
                        chunk_hidden, context_length
                    )
                    chunk_loss, _, chunk_kl, *_ = self._loss_fn(
                        chunk_hidden,                       # pi_old_outputs = HIDDEN
                        responses[cs:ce],                   # pi_outputs = target token ids
                        trajectory.ref_logprobs[cs:ce],     # ref_outputs = ref logprobs
                        trajectory.advantages[cs:ce],
                        padding_masks=~trajectory.response_padding_masks[cs:ce],
                    )
                    del chunk_hidden
                else:
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
            if getattr(self, "_linear_loss", False):
                # Chunked-vocab path (see chunked branch above): hidden states +
                # per-seq-chunk projection inside LinearGRPOLoss.
                try:
                    self._model.skip_output_layer = True
                    hidden = self._model(
                        trajectory.query_responses,
                        input_pos=trajectory.position_ids,
                        mask=trajectory.masks,
                    )
                finally:
                    self._model.skip_output_layer = False
                hidden = rlhf.truncate_sequence_for_logprobs(hidden, context_length)
                loss, _, kl_loss, *_ = self._loss_fn(
                    hidden,
                    responses,
                    trajectory.ref_logprobs,
                    trajectory.advantages,
                    padding_masks=~trajectory.response_padding_masks,
                )
                del hidden
            else:
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
        import contextlib
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        # No-FSDP colocate: the model is a plain replicated module, so state_dict()
        # is already the full dict (no gather needed). FSDP: use FULL_STATE_DICT.
        _sd_ctx = (
            contextlib.nullcontext()
            if getattr(self, "_no_fsdp", False)
            else FSDP.state_dict_type(
                self._model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
        )
        with _sd_ctx:
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

    def _colocate_reclaim(self, final: bool = False) -> None:
        """Real empty_cache for the no-FSDP colocate path (A100-equivalent).

        The standard ``device_empty_cache`` is a hard no-op on XPU because
        ``empty_cache`` + FSDP leaks UR handles.  The no-FSDP colocate hypothesis
        is that, WITHOUT FSDP units, ``torch.xpu.empty_cache()`` reclaims cleanly
        like CUDA — letting reserved memory drop between gen and train instead of
        accumulating to ``banned:1``.  This is the load-bearing call that tests it.
        Only fires on the ``_no_fsdp`` colocate path; everywhere else it is a no-op
        (the FSDP-ban still holds).

        ``final`` marks the end-of-train-step reclamation site (the one
        A100-equivalent point that, on its own, reclaims the big per-step grpo_step
        transient).  ``TORCHTUNE_COLOCATE_RECLAIM_MODE`` controls how many of the
        ~5 per-step sites actually call ``empty_cache``:
          - ``all`` (default): fire at every site (~5 empty_cache/step).
          - ``once``: fire ONLY at ``final`` (1 empty_cache/step) — still reclaims
            the big transient each step, but 5x fewer calls.  DECISIVE TEST: if the
            ~0.44 GiB/step creep is driven by ``empty_cache``/L0 reclamation residue
            it should fall ~5x in ``once`` mode; if it is vLLM-internal/per-step it
            stays the same.  See memory project_nofsdp_colocate_creep_diagnosis.
        ``TORCHTUNE_COLOCATE_RECLAIM_STRIDE`` (int, default 1): fire only every Nth
        eligible call (mitigation knob once the cause is known).
        """
        if not getattr(self, "_no_fsdp", False):
            return
        _mode = os.environ.get("TORCHTUNE_COLOCATE_RECLAIM_MODE", "all")
        if _mode == "once" and not final:
            return
        # stride gate: count eligible calls, fire only every Nth
        _stride = int(os.environ.get("TORCHTUNE_COLOCATE_RECLAIM_STRIDE", "1"))
        if _stride > 1:
            self._reclaim_eligible = getattr(self, "_reclaim_eligible", 0) + 1
            if (self._reclaim_eligible % _stride) != 0:
                return
        _probe = (
            os.environ.get("TORCHTUNE_COLOCATE_MEM_PROBE", "0") == "1"
            and self._is_rank_zero
        )
        if self._device.type == "xpu":
            torch.xpu.synchronize(self._device)
            if _probe:
                _r0 = torch.xpu.memory_reserved(self._device) / 1024**3
                _f0, _ = torch.xpu.mem_get_info(self._device)
            torch.xpu.empty_cache()
            # cumulative count of REAL empty_cache calls (causation test: plot
            # ALLOC creep against this, not against step number).
            self._ec_calls = getattr(self, "_ec_calls", 0) + 1
            if _probe:
                _r1 = torch.xpu.memory_reserved(self._device) / 1024**3
                _f1, _ = torch.xpu.mem_get_info(self._device)
                # CRITICAL SPLIT: allocated = live tensors PyTorch TRACKS; reserved =
                # torch pool; L0 free = whole device. If allocated CREEPS, the leak is
                # a torch-tracked tensor (our code / a torch module retaining refs). If
                # allocated is FLAT but L0 free DROPS, the growth is OUTSIDE torch's
                # allocator entirely → vLLM-internal (its own device buffers) or driver.
                # This is the measurement that ends the bisect.
                _alloc = torch.xpu.memory_allocated(self._device) / 1024**3
                try:
                    _ms = torch.xpu.memory_stats(self._device)
                    _active = _ms.get("active_bytes.all.current", 0) / 1024**3
                    _nalloc = _ms.get("allocation.all.current", 0)  # count of live blocks
                    # allocation-class attribution (the only stack-free signal XPU
                    # gives us — no memory_snapshot on XPU). If the creep is in
                    # `reserved-but-not-active` it is allocator frag; if in `active`
                    # it is genuinely live; `num_alloc_retries` rising == pool churn.
                    _seg = _ms.get("reserved_bytes.all.current", 0) / 1024**3
                    _inact = _ms.get("inactive_split_bytes.all.current", 0) / 1024**3
                    _retries = _ms.get("num_alloc_retries", 0)
                except Exception:
                    _active, _nalloc, _seg, _inact, _retries = -1.0, -1, -1.0, -1.0, -1
                log.info(
                    "COLOCATE_RECLAIM reserved %.2f->%.2f (%+.2f) | L0 free %.2f->%.2f (%+.2f) "
                    "| ALLOC=%.2f active=%.2f n_blocks=%d seg=%.2f inact=%.2f retries=%d "
                    "ec_calls=%d step=%d final=%d GiB",
                    _r0, _r1, _r1 - _r0,
                    _f0 / 1024**3, _f1 / 1024**3, (_f1 - _f0) / 1024**3,
                    _alloc, _active, _nalloc, _seg, _inact, _retries,
                    getattr(self, "_ec_calls", -1), getattr(self, "_steps_run", -1),
                    int(final),
                )
                # Live-tensor census (TORCHTUNE_COLOCATE_TENSOR_CENSUS=1): walk every
                # live torch.Tensor via gc, bucket by (shape,dtype) on-device. The
                # bucket whose COUNT grows ~N/step IS the leak — its shape names it.
                # Ends the bisect: inspection found no growing Python container, so
                # enumerate the actual live tensors instead of guessing.
                if os.environ.get("TORCHTUNE_COLOCATE_TENSOR_CENSUS", "0") == "1":
                    try:
                        import gc as _gc
                        from collections import Counter as _Counter
                        _c = _Counter()
                        _bytes = _Counter()
                        for _o in _gc.get_objects():
                            try:
                                if isinstance(_o, torch.Tensor) and _o.is_xpu:
                                    _k = (tuple(_o.shape), str(_o.dtype))
                                    _c[_k] += 1
                                    _bytes[_k] += _o.numel() * _o.element_size()
                            except Exception:
                                continue
                        _top = sorted(_bytes.items(), key=lambda kv: -kv[1])[:8]
                        for _k, _b in _top:
                            log.info(
                                "TENSOR_CENSUS step=%d shape=%s dtype=%s count=%d tot=%.3fGiB",
                                getattr(self, "_steps_run", -1), _k[0], _k[1],
                                _c[_k], _b / 1024**3,
                            )
                    except Exception as _ce:
                        log.warning("TENSOR_CENSUS failed: %r", _ce)
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

    def _warmup_at_max(self) -> None:
        """Front-load peak vLLM + FSDP buffers at step 0 (colocate only).

        Root cause this addresses (see ``docs/reports/lora_colocate_4b_20260618.md``
        and ``memory/project_overnight_colocate_ur40_plan_20260618``): in colocate
        the vLLM KV working set and the FSDP activation/all-gather buffers GROW the
        first time a rollout exceeds all prior sequence lengths.  XPU cannot
        ``empty_cache`` (the FSDP UR-handle-leak guard makes it a no-op), so the
        caching allocator's ``reserved`` only climbs — a discrete ~5 GiB staircase
        every few steps until ``banned:1``.  This caps colocate at
        ``max_generated_tokens=128`` (the validated-safe ceiling).

        The fix: BEFORE the train loop, run one throwaway generation at the maximum
        prompt+completion length AND one no-grad ref forward + one training fwd/bwd
        at ``vllm_max_model_len``, so BOTH engines allocate their peak buffers up
        front, inside the step-0 budget.  The mid-run jump then never happens — every
        subsequent (shorter or equal) step reuses already-reserved memory and the
        curve is flat from step 0.

        Runs on ALL ranks: the FSDP forward/backward are collective, so every rank
        must participate or the warmup deadlocks.  Gated to colocate (server /
        dedicated modes do not co-resident vLLM with the trainer, so they never hit
        the staircase).  Disable for A/B with ``TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0``.
        """
        if not self._colocate:
            return
        if os.environ.get("TORCHTUNE_COLOCATE_WARMUP_AT_MAX", "1") != "1":
            log.info("colocate warmup-at-max DISABLED (TORCHTUNE_COLOCATE_WARMUP_AT_MAX=0)")
            return

        dev = self._device
        S = int(self._vllm_max_model_len)
        gen_len = int(self._max_generated_tokens)
        context_length = max(1, S - gen_len)
        # Full per-step training batch grpo_step sees (all gen micro-batches concat).
        num_seqs = max(1, int(self.batch_size) * int(self.grpo_samples))
        vocab = int(getattr(self, "_vocab_size", 0)) or 32000
        pad_id = self._tokenizer.pad_id

        def _resv() -> float:
            if dev.type == "xpu":
                return torch.xpu.memory_reserved(dev) / 1024**3
            if dev.type == "cuda":
                return torch.cuda.memory_reserved(dev) / 1024**3
            return 0.0

        _r0 = _resv()
        log.info(
            "Rank %d: colocate warmup-at-max START num_seqs=%d seq_len=%d "
            "(ctx=%d gen=%d) reserved=%.2f GiB",
            self.rank, num_seqs, S, context_length, gen_len, _r0,
        )

        # --- 1. vLLM KV peak: submit num_seqs max-length prompts concurrently with
        #     ignore_eos so each runs the full max_generated_tokens.  The real step
        #     generates batch_size*grpo_samples concurrent sequences, so the KV block
        #     pool's working max is for num_seqs full-length seqs — warming with one
        #     prompt would under-allocate it.  detokenize=False keeps it cheap. ---
        if self._vllm_llm is not None:
            from vllm import SamplingParams

            max_prompt_len = max(1, S - gen_len)
            warm_prompt = [int(self._tokenizer.bos_id or 1)] * max_prompt_len
            try:
                self._vllm_llm.generate(
                    prompts=[{"prompt_token_ids": warm_prompt}] * num_seqs,
                    sampling_params=SamplingParams(
                        max_tokens=gen_len,
                        temperature=self._temperature,
                        top_k=self._top_k if self._top_k else -1,
                        ignore_eos=True,        # run the full length — peak KV
                        detokenize=False,
                    ),
                    use_tqdm=False,
                )
            except Exception as _e:  # pragma: no cover - HW path
                log.warning("Rank %d: warmup vLLM generate failed: %s", self.rank, _e)
            if dev.type == "xpu":
                torch.xpu.set_device(_xpu_device_index)
                torch.xpu.synchronize()

        # --- Synthetic worst-case batch at [num_seqs, S] (random valid tokens). ---
        query_responses = torch.randint(
            0, vocab, (num_seqs, S), dtype=torch.long, device=dev
        )
        responses = query_responses[:, context_length:]
        qr_pad_mask = query_responses != pad_id
        # Mirror generate_trajectory's mask decision so the warmup allocates the
        # SAME mask buffer the real step will (None under maskfree/varlen, else the
        # full [num_seqs, S, S] causal mask — itself a large transient to front-load).
        _maskfree, _ = _compute_maskfree_causal(
            env_set=os.environ.get("TORCHTUNE_MASKFREE_CAUSAL") == "1",
            device_type=dev.type,
            packing_enabled=False,
            query_responses=query_responses,
            context_length=context_length,
            pad_id=pad_id,
        )
        if _maskfree:
            masks = None
        else:
            masks = generation.get_causal_mask_from_padding_mask(
                qr_pad_mask, target_seq_len=S
            )
        position_ids = generation.get_position_ids_from_padding_mask(qr_pad_mask)
        response_padding_masks = torch.zeros_like(responses, dtype=torch.bool)
        ref_logprobs = torch.zeros_like(responses, dtype=torch.float32)
        advantages = torch.zeros(num_seqs, dtype=torch.float32, device=dev)

        # --- 2. No-grad ref forward peak (disable_adapter, chunked at ref_fbs). ---
        ref_fbs = self._ref_forward_batch_size
        with disable_adapter(self._model):
            with torch.no_grad():
                for cs in range(0, num_seqs, ref_fbs):
                    ce = min(cs + ref_fbs, num_seqs)
                    _ = self._model(
                        query_responses[cs:ce],
                        input_pos=position_ids[cs:ce],
                        mask=None if masks is None else masks[cs:ce],
                    )
                    del _
        if dev.type == "xpu":
            torch.xpu.synchronize()

        # --- 3. Training fwd+bwd peak: run the REAL grpo_step on the synthetic
        #     trajectory so activation + all-gather + reduce-scatter buffers all
        #     allocate at max.  Discard the resulting (garbage) grads. ---
        warm_traj = GRPOTrajectory(
            query_responses=query_responses,
            logprobs=None,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            rewards=torch.zeros(num_seqs, device=dev),
            successes=torch.zeros(num_seqs, device=dev),
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
            answers=[""] * num_seqs,
        )
        try:
            self.grpo_step(warm_traj)
        except Exception as _e:  # pragma: no cover - HW path
            log.warning("Rank %d: warmup grpo_step failed: %s", self.rank, _e)
        finally:
            self._optimizer.zero_grad(set_to_none=True)

        del query_responses, responses, masks, position_ids
        del response_padding_masks, ref_logprobs, advantages, warm_traj
        if dev.type == "xpu":
            torch.xpu.synchronize()

        _r1 = _resv()
        log.info(
            "Rank %d: colocate warmup-at-max DONE reserved %.2f->%.2f (+%.2f) GiB "
            "(this peak is now front-loaded; mid-run staircase should not occur)",
            self.rank, _r0, _r1, _r1 - _r0,
        )

    def train(self) -> None:
        """Main training loop."""
        # Verify vLLM EngineCore is functional before entering the training loop.
        # On Aurora XPU, the EngineCore can appear healthy but crash on the first
        # inference call due to stale L0 driver state from prior jobs on the same node.
        # Rank 0 retries up to 3× with 60s waits; all ranks barrier after.
        if self._is_rank_zero:
            self._warmup_vllm()
        torch.distributed.barrier()

        # Colocate only: front-load peak vLLM + FSDP buffers at max sequence length
        # so the per-step memory curve is flat from step 0 (no seq-length staircase
        # to banned:1).  Runs on ALL ranks (FSDP collectives).  No-op off colocate.
        self._warmup_at_max()
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
                        self._publish_thread.join(timeout=self._publish_join_timeout)
                        _timed_out = self._publish_thread.is_alive()
                        _pub_err = self._publish_error
                        self._publish_thread = None
                        self._publish_error = None
                        if self._is_rank_zero:
                            log.info("Rank 0: publish join %.2fs", time.perf_counter() - _join_t0)
                        if _timed_out:
                            raise RuntimeError(
                                f"Adapter publish thread timed out "
                                f"({self._publish_join_timeout}s) — vLLM adapter may not "
                                "have been updated. Aborting to avoid stale-policy training."
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

                    # Sub-phase reserved-memory attribution (opt-in via
                    # TORCHTUNE_COLOCATE_MEM_PROBE=1). On XPU we never empty_cache
                    # (UR-handle leak), so any per-step transient inflates `reserved`
                    # permanently. This logs the reserved DELTA across gen / grpo_step
                    # / sync to localize the ~5 GiB-per-few-steps staircase that drives
                    # colocate banned:1. One run pinpoints the source; off by default.
                    _phase_probe = (
                        os.environ.get("TORCHTUNE_COLOCATE_MEM_PROBE", "0") == "1"
                        and self._is_rank_zero and self._device.type == "xpu"
                    )
                    def _resv():
                        return torch.xpu.memory_reserved(self._device) / 1024**3
                    # ACTIVE (live) tracker — the A/B (job 8558600) proved the
                    # ~0.44 GiB/step creep is in ACTIVE memory (active==alloc,
                    # inact=0, empty_cache-count-independent), so reserved-based
                    # phase probes are blind to it. Track active too, with a sync
                    # so the counter is settled before reading.
                    def _act():
                        torch.xpu.synchronize(self._device)
                        return torch.xpu.memory_allocated(self._device) / 1024**3
                    _r_pre_gen = _resv() if _phase_probe else 0.0
                    _a_pre_gen = _act() if _phase_probe else 0.0

                    # Generate trajectory (server mode, may use base or LoRA adapter)
                    _gen_t0 = time.perf_counter()
                    with torch.no_grad():
                        trajectory = self.generate_trajectory_batched(tokens, answers)
                    _gen_time = time.perf_counter() - _gen_t0
                    if _phase_probe:
                        _r_post_gen = _resv()
                        _a_post_gen = _act()
                        log.info("COLOCATE_PHASEPROBE step=%d gen reserved %.2f->%.2f (+%.2f) "
                                 "| ACTIVE %.2f->%.2f (%+.3f) GiB",
                                 self._steps_run, _r_pre_gen, _r_post_gen, _r_post_gen - _r_pre_gen,
                                 _a_pre_gen, _a_post_gen, _a_post_gen - _a_pre_gen)

                    # PPO epochs
                    _grpo_t0 = time.perf_counter()
                    grpo_metrics = {}
                    for _ppo_epoch in range(self._ppo_epochs):
                        grpo_metrics = self.grpo_step(trajectory)
                    _grpo_time = time.perf_counter() - _grpo_t0
                    if _phase_probe:
                        _r_post_grpo = _resv()
                        _a_post_grpo = _act()
                        log.info("COLOCATE_PHASEPROBE step=%d grpo_step reserved %.2f->%.2f (+%.2f) "
                                 "| ACTIVE %.2f->%.2f (%+.3f) GiB",
                                 self._steps_run, _r_post_gen, _r_post_grpo, _r_post_grpo - _r_post_gen,
                                 _a_post_gen, _a_post_grpo, _a_post_grpo - _a_post_gen)

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

                    # Adapter publish to vLLM. Three modes (self._lora_publish_mode):
                    #   merged (default) — rank-0 builds the full merged W_eff
                    #     state dict from cached base + live adapter, raw_bytes
                    #     file + POST load_weights_from_raw. ~6.77 GiB/step.
                    #   delta — rank-0 ships the base ONCE then ~66 MB lora_a/b
                    #     each step; the vLLM worker re-merges. Bit-exact to
                    #     merged, ~100x less per-step wire.
                    #   runtime (legacy) — rank-0 adapter snapshot, PEFT dir +
                    #     /v1/load_lora_adapter HTTP. Requires --enable-lora.
                    # All three gather rank-0-ONLY (no collective) and publish in a
                    # daemon thread joined before next-step generation (~line 1904).
                    # Non-rank-0 ranks legitimately get None and skip; rank 0 never
                    # blocks on a collective, so do NOT add a barrier here.
                    if self._steps_run % self._lora_publish_every == 0:
                        _pub_t0 = time.perf_counter()
                        _mode = self._lora_publish_mode
                        if _mode == "colocate":
                            # In-process publish: every rank summons its OWN full
                            # base, merges W_eff, and loads it into its OWN engine.
                            # This is COLLECTIVE (summon_full_params) — must run on
                            # ALL ranks, synchronously (no thread, no HTTP).
                            _r_pre_sync = _resv() if _phase_probe else 0.0
                            _a_pre_sync = _act() if _phase_probe else 0.0
                            self._sync_colocated_lora_weights()
                            if self._is_rank_zero:
                                log.info(
                                    "Rank 0: colocate LoRA wsync %.2fs",
                                    time.perf_counter() - _pub_t0,
                                )
                            if _phase_probe:
                                _r_post_sync = _resv()
                                _a_post_sync = _act()
                                log.info("COLOCATE_PHASEPROBE step=%d sync reserved %.2f->%.2f (+%.2f) "
                                         "| ACTIVE %.2f->%.2f (%+.3f) GiB",
                                         self._steps_run, _r_pre_sync, _r_post_sync, _r_post_sync - _r_pre_sync,
                                         _a_pre_sync, _a_post_sync, _a_post_sync - _a_pre_sync)
                        elif _mode == "runtime":
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
                        elif _mode == "delta":
                            # Path C: rank-0-ONLY adapter snapshot (~66 MB); the
                            # vLLM worker re-merges against a cached base shipped
                            # once on the first publish.
                            delta_payload = self._gather_lora_delta_payload()
                            if self._is_rank_zero and delta_payload is not None:
                                _gather_time = time.perf_counter() - _pub_t0
                                log.info(
                                    "Rank 0: delta gather done in %.2fs (%d tensors) — starting async publish",
                                    _gather_time, len(delta_payload[0]),
                                )
                                self._publish_error = None
                                _dp = delta_payload

                                def _bg(_s=_dp):
                                    try:
                                        self._publish_lora_delta_background(_s)
                                    except Exception as _e:
                                        self._publish_error = _e
                                        log.error("Rank 0: delta async publish failed: %s", _e)

                                self._publish_thread = threading.Thread(target=_bg, daemon=True)
                                self._publish_thread.start()
                        else:
                            # Merged path (default): rank-0-ONLY, NO collective.
                            # _gather_merged_lora_weights reads the frozen base
                            # weights from self._cached_base_weights (gathered ONCE
                            # in _cache_lora_base_weights at setup) and the live,
                            # FSDP-ignored (replicated) lora_a/lora_b adapter tensors.
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

                    # No-FSDP colocate: reclaim transients at the end of the step
                    # (the A100-equivalent reclamation point — between this step's
                    # training and the next step's generation). No-op otherwise.
                    # final=True: the one site that fires even in RECLAIM_MODE=once.
                    self._colocate_reclaim(final=True)

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
                self._publish_thread.join(timeout=self._publish_join_timeout)
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

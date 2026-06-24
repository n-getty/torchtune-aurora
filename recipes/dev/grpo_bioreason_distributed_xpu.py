# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# BioReason-specific GRPO recipe for Aurora XPU.
#
# Extends GRPOFullFinetuneDistributedXPU with:
# - BioReasonModel (ESM3 + GO graph encoder + Qwen3-4B backbone)
# - prompt_embeds-based vLLM generation (ESM3+GO embeddings pre-computed on CPU)
# - Dynamic ref-model CPU offload (~8 GiB HBM savings)
# - FSDP1 SHARD_GRAD_OP (ZeRO-2) over training ranks in dedicated_rank mode
# - BioReason GO-term F1 reward function
#
# Usage:
#   python3 -m torch.distributed.run --standalone --nproc_per_node=N \
#       recipes/dev/grpo_bioreason_distributed_xpu.py \
#       --config recipes/configs/dev/production/bioreason_4b_grpo_xpu.yaml

import os
import sys
import time
import json
import logging
from typing import Any, Optional

# Ensure all module-level loggers (including torchtune.dev.bioreason.model) emit
# to stderr. Without this, logger.info() inside BioReasonModel silently drops
# and we cannot tell which step of __init__ crashed.
_RANK_FOR_LOG = os.environ.get("RANK", "?")
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [r{_RANK_FOR_LOG}] %(name)s %(levelname)s: %(message)s",
    stream=sys.stderr,
    force=True,
)

import torch
from omegaconf import DictConfig

from torchtune import config, rlhf, training, utils
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory
from torchtune.dev.rl.distributed import device_empty_cache, _slice_trajectory
from torchtune.dev.rl.rewards import gene_recall_batched_rewards, batched_rewards

# Import the base recipe — it handles all the XPU/XCCL shim setup at import time.
# `recipes/__init__.py` deliberately raises on import (to keep tests from picking
# up the recipes package), so we load the sibling base recipe by file path.
import importlib.util as _importlib_util

_BASE_RECIPE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "grpo_full_finetune_distributed_xpu.py",
)
_spec = _importlib_util.spec_from_file_location(
    "grpo_full_finetune_distributed_xpu", _BASE_RECIPE_PATH
)
_base_module = _importlib_util.module_from_spec(_spec)
sys.modules["grpo_full_finetune_distributed_xpu"] = _base_module
_spec.loader.exec_module(_base_module)
GRPOFullFinetuneDistributedXPU = _base_module.GRPOFullFinetuneDistributedXPU
log = _base_module.log
_colocate_vllm_mode = _base_module._colocate_vllm_mode


class GRPOBioReasonDistributedXPU(GRPOFullFinetuneDistributedXPU):
    """
    BioReason-specific GRPO recipe for Aurora XPU.

    Subclasses GRPOFullFinetuneDistributedXPU and adds:
    - BioReasonModel loading (bypasses TorchTune checkpointer)
    - ESM3+GO prompt embedding computation for vLLM generation
    - inputs_embeds forward path in generate_trajectory / grpo_step
    - BioReason GO-term F1 reward (reward_mode: bioreason)
    - Dynamic ref-model CPU offload for HBM budget management
    """

    # ── Setup overrides ────────────────────────────────────────────────────────

    def setup(self, cfg: DictConfig) -> None:
        """
        Override setup to intercept BioReason-specific initialization paths.

        Two special cases:
        1. Dedicated vLLM rank: load frozen BioReasonModel for embed computation,
           create process groups, then return (skip all training setup).
        2. BioReason training ranks: load BioReasonModel for policy + ref, set
           tokenizer, optionally wrap in FSDP1, then return (skip checkpointer).
        """
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        _is_bioreason = cfg.get("model_type") == "bioreason"

        if self._is_vllm_rank:
            # Rank runs as dedicated vLLM generation server — skip all training setup.
            self._setup_bioreason_vllm_rank(cfg)
            return

        if _is_bioreason:
            self._setup_bioreason_models(cfg)
        else:
            # Fall through to base class setup for non-BioReason configs.
            # Call parent setup but skip the metric_logger / cpu_offload parts
            # already done above by delegating from the point after those checks.
            super().setup(cfg)
            return

        # Complete setup for BioReason training ranks (after _setup_bioreason_models).
        # Mirrors the post-model-loading section of the base class setup().

        # RL hyperparameters (parallel to base class lines 896-933)
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._total_steps = cfg.num_steps
        self._reward_mode = cfg.get("reward_mode", "bioreason")
        self._gene_reward_metric = cfg.get("gene_reward_metric", "f1")
        # GO-hierarchy-aware reward: propagate predicted + GT terms to their is_a
        # ancestor closure before F1 (matches the cafaeval F_max metric). Default ON
        # for bioreason — exact-match F1 vs the correct target is still ~50% zeros and
        # too flat to learn; propagation makes it dense (mean 0.04 -> 0.23 on real
        # rollouts) AND aligns reward with eval. obo ships in the ckpt/source dir.
        self._reward_propagate_hierarchy = cfg.get(
            "reward_propagate_hierarchy", self._reward_mode == "bioreason",
        )
        # obo for reward propagation: explicit config, else the checkpoint dir (each
        # bioreason ckpt ships go-basic.obo), else reward.py's env/source fallback.
        self._reward_obo_path = cfg.get(
            "reward_obo_path", cfg.get("base_model_path", None),
        )
        # Pool advantage normalization across the full batch (BioReason-Pro fix).
        # Default true for bioreason mode (matches the upstream paper's GRPO setup);
        # explicit override possible via config field.
        self._batch_level_advantages = cfg.get(
            "batch_level_advantages", self._reward_mode == "bioreason",
        )
        self._enable_packing = cfg.get("enable_packing", False)
        self._expert_parallel_degree = cfg.get("expert_parallel_degree", 1)
        self._shard_pg = None
        self._always_compute_rollout_logprobs = cfg.get(
            "always_compute_rollout_logprobs", False
        )

        # ── Async generation wiring (BioReason) ───────────────────────────────
        # BioReason overrides setup() and skips the base setup() block that reads
        # the `async_generation` config and sets `_async_generation_enabled` /
        # `_async_generation_max_staleness`. `_setup_vllm_server_mode()` (called
        # below) READS `self._async_generation_enabled`, so it MUST exist before
        # that call. We re-implement the same guards as the base recipe
        # (grpo_full_finetune_distributed_xpu.py setup) so the BioReason async
        # path honours the identical staleness=1-only + server-mode-only +
        # GRPOLoss-IS contract.
        #
        # Producer/consumer boundary (BioReason-specific — differs from the base
        # token-only recipe): the prompt_embeds build (ESM3 cache + GO + trainable
        # projectors under FSDP `summon_full_params`) is an XPU forward AND a world
        # collective, so it CANNOT run in the rank-0 producer thread. Only the pure
        # vLLM HTTP `generate_from_embeds` round-trip is overlapped on the producer;
        # the embeds for the lookahead batch are pre-built on the MAIN thread (all
        # ranks, reusing the existing collective path) and the rank-0 CPU embeds
        # list is handed to the producer. The query_responses broadcast stays on the
        # consumer/main thread on every rank. See _async_lookahead_iter override.
        _async_cfg = cfg.get("async_generation", {}) or {}
        self._async_generation_enabled = bool(_async_cfg.get("enabled", False))
        self._async_generation_max_staleness = int(_async_cfg.get("max_staleness", 1))
        # HSDP (dp_replicate>1): the async lookahead is PER-REPLICA. Each replica's
        # shard-leader (global ranks 0, dp_shard, 2*dp_shard, ...) runs its own
        # RolloutProducer thread that POSTs its replica's DISTINCT prompt slice to
        # the shared vLLM pool, holds its replica's HTTP result, and the consume-time
        # broadcast is NODE-LOCAL over _gloo_dp_shard_pg (NOT the world group). This
        # mirrors the validated SYNC HSDP path (_generate_with_vllm_server_embeds /
        # _broadcast_query_responses) — async just overlaps the pure-HTTP half. The
        # single-replica path (dp_replicate==1) is the special case: _is_shard_leader
        # == rank 0 and the broadcast group collapses to the world group, so it is
        # byte-identical to the previously-validated 2N single-replica async path.
        # No force-disable here any more.
        # Server-mode guard: dedicated_rank uses broadcast_object_list over the
        # training PG (every rank must call together) and colocate runs gen inline
        # on every rank — neither is async-overlappable. Refuse to engage.
        if self._async_generation_enabled and self._vllm_mode != "server":
            log.warning(
                "BioReason: async_generation requested but disabled — only "
                "vllm_mode=server is supported (got %s). dedicated_rank/colocate "
                "generation is a world collective and cannot overlap. Running "
                "synchronously.",
                self._vllm_mode,
            )
            self._async_generation_enabled = False
        # Staleness>1 hard-cap: rollout pi_old_logprobs are recomputed on the
        # current training weights, so they don't match the behavior policy that
        # produced the rollout. The bias grows with staleness; only k=1 is allowed.
        if self._async_generation_enabled and self._async_generation_max_staleness > 1:
            raise ValueError(
                "async_generation.max_staleness>1 is not safe yet: rollout "
                "logprobs are recomputed on the current training weights, so "
                "pi_old_logprobs will not match the behavior policy that produced "
                "the rollout (biased GRPO IS ratios). Set max_staleness=1 or "
                "implement vLLM-time logprob capture."
            )
        if self._async_generation_enabled:
            log.warning(
                "BioReason: async_generation ENABLED (max_staleness=%d): "
                "EXPERIMENTAL. pi_old_logprobs are recomputed on the current "
                "training model, but the rollout was sampled under the previous "
                "weight version — GRPO IS ratios carry a small bias even at "
                "staleness=1. Requires GRPOLoss + always_compute_rollout_logprobs.",
                self._async_generation_max_staleness,
            )
        # Rollout-time logprobs are required when async (off-policy by k>=1) OR when
        # explicitly requested. Mirrors the base recipe's coupling.
        self._compute_rollout_logprobs_required = (
            self._always_compute_rollout_logprobs
            or self._async_generation_enabled
        )

        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        self._eval_every_n_steps = cfg.get("eval_every_n_steps", 0)
        self._eval_max_examples = cfg.get("eval_max_examples", 50)

        stop_token_ids = (
            list(self._tokenizer.stop_tokens)
            if hasattr(self._tokenizer, 'stop_tokens') and self._tokenizer.stop_tokens
            else [self._tokenizer.eos_id]
        )
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)
        # Plain int list for the vLLM /v1/completions payload (JSON), so vLLM stops
        # decoding at EOS server-side instead of always running to max_tokens. The
        # tensor form above is for the train-side post-hoc truncation/masking.
        # TORCHTUNE_VLLM_STOP_TOKENS (default 1=on): set 0 to NOT send stop tokens to
        # vLLM (old behavior: every rollout runs to max_tokens) for same-node A/B.
        if os.environ.get("TORCHTUNE_VLLM_STOP_TOKENS", "1") != "0":
            self._stop_token_ids_list = [int(t) for t in stop_token_ids]
        else:
            self._stop_token_ids_list = None
            log.warning("TORCHTUNE_VLLM_STOP_TOKENS=0: vLLM will NOT stop at EOS "
                        "(every rollout decodes to max_tokens). A/B-only setting.")

        # Optimizer, loss, dataloader
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=None,
        )
        self._loss_fn = config.instantiate(cfg.loss)
        # The chunked-vocab LinearGRPOLoss (set_model_output / skip_output_layer) is
        # NOT supported here: BioReason's backbone is an HF AutoModelForCausalLM
        # (torchtune/dev/bioreason/model.py) whose forward returns out.logits — it has
        # no torchtune `skip_output_layer` hidden-state path to project per-chunk, and
        # BioReason runs FSDP FULL_SHARD (no no-FSDP path). Both conditions break the
        # projection-outside-forward assumption. Fail fast rather than mis-wire. An
        # HF-specific port (expose hidden states + apply lm_head in the loss, under a
        # summon) is future work. Use GRPOSimpleLoss/GRPOLoss here.
        if hasattr(self._loss_fn, "set_model_output"):
            raise RuntimeError(
                "LinearGRPOLoss (chunked-vocab) is not supported in the BioReason "
                "recipe: the HF AutoModelForCausalLM backbone has no skip_output_layer "
                "hidden-state path, and the recipe runs FSDP FULL_SHARD. Use "
                "GRPOSimpleLoss or GRPOLoss."
            )
        self._use_chunked_loss = hasattr(self._loss_fn, "num_output_chunks")
        utils.log_rank_zero(log, "Loss is initialized.")

        collate_name = cfg.get(
            "collate_fn", "torchtune.dev.bioreason.dataset.bioreason_collate_fn"
        )
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
        )
        self._eval_examples = []
        self._eval_enabled = False

        self._steps_per_epoch = len(self._dataloader)
        self.total_epochs = cfg.get("epochs", 1)
        self._epochs_run = 0
        self._steps_run = 0
        self.global_step = 0

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        self._profiler = self._setup_profiler(cfg.get("profiler", None))
        self.profiler_profile_memory = False
        self.profiler_wait_steps = 0
        self.profiler_warmup_steps = 0
        self._layer_mem_hooks = []

        # Set the module-level colocate flag from the RUNTIME mode. This file
        # snapshots `_colocate_vllm_mode` from the base module at import time (when
        # it is still False), and because BioReason overrides setup() the base
        # recipe's line that flips the global never runs. Without this, every
        # `if not _colocate_vllm_mode:` guard in this file stays True in colocate →
        # device_empty_cache fires every step (UR-handle leak → banned:1) AND the
        # colocate weight-merge branch is never reached. (Root cause of the 2026-06-18
        # colocate step-1 banned:1.)
        global _colocate_vllm_mode
        _colocate_vllm_mode = self._vllm_mode in ("colocate", "colocate_sleep")

        # Async lookahead consumer-side stashes (see _async_lookahead_iter_impl).
        # _setup_vllm_server_mode sets _pending_async_query_responses = None; the
        # BioReason embeds path also needs _pending_async_prompt_embeds initialized
        # so generate_trajectory's getattr() default is correct on sync steps.
        self._pending_async_prompt_embeds = None
        # All-ranks flag: True only while consuming an async-overlapped rollout for
        # the current batch (set by _async_lookahead_iter_impl._consume on every
        # rank). Gates the symmetric broadcast branch in generate_trajectory.
        self._async_consume_active = False

        # vLLM setup — reuse base class helpers (they don't depend on model type)
        if self._vllm_mode == "server":
            self._setup_vllm_server_mode()
        elif self._vllm_mode in ("colocate", "colocate_sleep"):
            self._setup_vllm_colocate_mode(cfg)

        # Weight map (empty for BioReason — params already in HF format)
        self._build_tune_to_hf_map()

        utils.log_rank_zero(log, "BioReason setup complete.")

    def _build_tune_to_hf_map(self) -> None:
        """BioReason params are already in HF format — no remapping needed."""
        if getattr(self, '_is_bioreason', False):
            # _tune_to_hf_map is set to {} in _setup_bioreason_models.
            # weight-sync .get(k, k) calls fall back to identity.
            return
        super()._build_tune_to_hf_map()

    def save_checkpoint(self, epoch: int) -> None:
        """Override to add BioReason fast-path checkpoint (projectors + backbone)."""
        # BioReasonModel checkpointing: save backbone + projectors directly.
        # With LoRA, backbone is a PeftModel and save_pretrained writes only the
        # adapter (adapter_model.safetensors + adapter_config.json) — matching the
        # published BioReason-Pro flow; the merged HF backbone is recoverable by
        # PEFT merge_and_unload at load time. Full params must be gathered first:
        # the policy is FSDP-wrapped (server/dedicated modes), so save under
        # summon_full_params on all ranks, write on rank 0.
        if hasattr(self._policy, 'vllm_param_iter'):
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP, StateDictType,
            )
            save_dir = os.path.join(self._output_dir, f"epoch_{epoch}")
            _fsdp = getattr(self, "_use_fsdp1", False) and torch.distributed.is_initialized()

            # GATHER VIA FULL_STATE_DICT, NOT summon_full_params (2026-06-22 fix).
            # summon_full_params materializes the ENTIRE 4B model on-device, allocating
            # ~32 GiB and freeing it on exit — and those freed L0 pages are still
            # referenced by the live XCCL wsync IPC handles, so the NEXT collective
            # faults with banned:1 (NotPresent PDE). Crashed at step 11 right after the
            # step-10 save, TWICE (rank0_only=True did NOT help — FSDP1 still all-gathers
            # on every rank). The PROVEN-SAFE pattern is the one _sync_weights_to_vllm
            # uses EVERY step without crashing: state_dict_type(FULL_STATE_DICT) +
            # state_dict(), which gathers tensor-by-tensor into a CPU dict and releases
            # each gather immediately (no persistent on-device full materialization).
            # We then slice the adapter (lora_*) + projection tensors out of that dict.
            if _fsdp:
                with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
                    _full_sd = self._model.state_dict()  # COLLECTIVE on all ranks, CPU
            else:
                _full_sd = self._model.state_dict()

            if self._is_rank_zero:
                os.makedirs(save_dir, exist_ok=True)
                _has_lora = getattr(self._model, "_has_lora", False)

                def _strip(name):
                    return (name.replace("_fsdp_wrapped_module.", "")
                                .replace("_checkpoint_wrapped_module.", ""))

                # Projections: pull protein_projection.* / go_projection.* (already full
                # tensors in the gathered CPU dict — clone to detach from any shared store).
                for _pname in ("protein_projection", "go_projection"):
                    _sub = {}
                    for k, v in _full_sd.items():
                        ck = _strip(k)
                        if ck.startswith(_pname + "."):
                            _sub[ck[len(_pname) + 1:]] = v.detach().clone()
                    torch.save(_sub, os.path.join(save_dir, f"{_pname}.pt"))

                if _has_lora:
                    # Extract the LoRA adapter (lora_A/lora_B) into PEFT adapter format.
                    from safetensors.torch import save_file
                    _adir = os.path.join(save_dir, "adapter")
                    os.makedirs(_adir, exist_ok=True)
                    _adapter = {}
                    for k, v in _full_sd.items():
                        ck = _strip(k)
                        if "lora_A" in ck or "lora_B" in ck or ".lora_" in ck:
                            # PEFT-canonical keys start with "base_model." — the model
                            # state_dict prefixes them with "backbone." (BioReasonModel
                            # wraps the PeftModel as self.backbone). Strip it so the keys
                            # match what set_peft_model_state_dict expects on resume
                            # (matches _sync_weights_to_vllm's backbone-prefix handling).
                            if ck.startswith("backbone."):
                                ck = ck[len("backbone."):]
                            _adapter[ck] = v.detach().clone().contiguous()
                    save_file(_adapter, os.path.join(_adir, "adapter_model.safetensors"))
                    # adapter_config.json (PEFT loader needs it; mirror the ctor config).
                    try:
                        self._policy.backbone.peft_config["default"].save_pretrained(_adir)
                    except Exception:
                        import json as _json
                        with open(os.path.join(_adir, "adapter_config.json"), "w") as _f:
                            _json.dump({"peft_type": "LORA"}, _f)
                    log.info("BioReason checkpoint saved to %s (adapter, %d lora tensors)",
                             save_dir, len(_adapter))
                else:
                    # Full backbone: save the (stripped) backbone.* tensors.
                    _bk = {_strip(k)[len("backbone."):]: v.detach().clone()
                           for k, v in _full_sd.items() if _strip(k).startswith("backbone.")}
                    torch.save(_bk, os.path.join(save_dir, "backbone.pt"))
                    log.info("BioReason checkpoint saved to %s (backbone)", save_dir)

            del _full_sd
            # Settle any frees before the next collective (no empty_cache — leaks UR
            # handles under FSDP). FULL_STATE_DICT shouldn't churn on-device like summon
            # did, but keep the barrier so ranks resync after rank-0's Lustre writes.
            import gc as _gc
            _gc.collect()
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            if torch.distributed.is_initialized():
                pg = self._training_pg if self._vllm_mode == "dedicated_rank" else None
                torch.distributed.barrier(group=pg)
            return

        super().save_checkpoint(epoch)

    def _sync_colocated_lora_weights(self) -> None:
        """Merge W_eff = base + (alpha/r)*BA per-rank and load into THIS rank's
        in-process vLLM engine (colocate / colocate_sleep + PEFT-LoRA).

        vLLM runs a vanilla (adapter-less) Qwen3, so it must receive merged
        weights. Only the LoRA-TARGET weights are pushed each step — the frozen
        non-target params (norms/embeddings) never change, so re-loading them is
        wasted per-step transient (and churns the allocator). W_eff is formed from
        the frozen base param + the streamed fp32 delta (PEFT get_delta_weight,
        non-mutating → no merge/unmerge bf16 drift).

        STREAMING + MINIMAL TRANSIENT (banned:1 fix, 2026-06-18): an earlier version
        materialized all ~398 deltas (lora_delta_map dict) AND re-pushed all 398
        backbone params with a double fp32 upcast every step — under colocate there
        is no empty_cache (UR-handle guard), so that per-step transient fragmented
        the allocator → reserved staircase +11 GiB/step → banned:1 at step 1. Now:
        one delta at a time via lora_delta_iter(), in-place add into a single reused
        bf16 buffer, freed each iter. In colocate the model is NOT FSDP-wrapped (full
        per-rank), so no summon is needed — the base param is read directly.
        """
        import gc
        import contextlib
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        t0 = time.perf_counter()
        llm_model = (
            self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
        # Map clean-HF-name -> frozen base param (read-only). Built once per call.
        base_by_name = {hf: p for hf, p in self._model.vllm_param_iter()}
        # Only summon if actually FSDP-wrapped (server/dedicated); colocate is not.
        _summon = (
            FSDP.summon_full_params(self._model, writeback=False, rank0_only=False)
            if isinstance(self._model, FSDP) else contextlib.nullcontext()
        )
        n_synced = 0
        with torch.no_grad(), _summon:
            for hf_name, delta in self._model.lora_delta_iter():
                base = base_by_name.get(hf_name)
                if base is None:
                    continue
                # fp32 accumulate then a single bf16 cast; delta is already fp32.
                weff = (base.detach().float() + delta).to(base.dtype).contiguous()
                llm_model.load_weights([(hf_name, weff)])
                n_synced += 1
                del weff, delta
                if n_synced % 5 == 0 and self._device.type == "xpu":
                    gc.collect()
                    torch.xpu.synchronize(self._device)
        del base_by_name
        self._vllm_llm.llm_engine.reset_prefix_cache()
        if self._device.type == "xpu":
            gc.collect()
            torch.xpu.synchronize(self._device)
        log.info(
            "Rank %d: colocate LoRA W_eff sync %d params in %.2fs",
            self.rank, n_synced, time.perf_counter() - t0,
        )

    # Bind the base (inherited) colocate sync under a private name so the LoRA
    # override below can fall back to it for the non-LoRA (full-FT) path. The base
    # method is a class attribute (bound from the weight_sync module at the base
    # class body), so reference it via the base CLASS, not the module.
    _sync_colocated_weights_base = GRPOFullFinetuneDistributedXPU._sync_colocated_weights

    def _sync_colocated_weights(self) -> None:
        """Override: route plain-colocate weight sync (called by the base train()
        loop's _run_wsync_block) to the per-rank LoRA merge when LoRA is active.

        The inherited base impl ships ALL backbone params (incl. the sharded
        embed_tokens), which under FSDP FULL_SHARD trips vLLM's vocab-embedding
        weight_loader assert (loaded shape != org_vocab_size). With LoRA we instead
        push only the merged LoRA-target W_eff (_sync_colocated_lora_weights);
        non-target frozen params were already loaded at engine init. Non-LoRA
        BioReason colocate falls back to the inherited backbone sync.
        """
        if getattr(self._model, "_has_lora", False):
            self._sync_colocated_lora_weights()
        else:
            self._sync_colocated_weights_base()

    # ── BioReason-specific init methods ───────────────────────────────────────

    def _setup_bioreason_vllm_rank(self, cfg: DictConfig) -> None:
        """Initialize the dedicated vLLM generation server rank (rank N-1).

        vLLM engine is already initialized in _init_vllm_early_dedicated() (called
        before the CCL process group in __init__). This method:
        - Loads BioReasonModel (frozen) for ESM3+GO embed computation.
        - Creates training_pg and wsync_pg for coordination with training ranks.
        - Stores generation params for _run_vllm_generation_server().
        """
        from torchtune.dev.bioreason.model import BioReasonModel

        ckpt_dir = cfg.base_model_path
        log.info("Rank %d (vLLM server): loading embed model from %s", self.rank, ckpt_dir)
        self._embed_model = BioReasonModel(
            ckpt_dir=ckpt_dir,
            device=self._device,
            dtype=self._dtype,
            esm3_cache_path=cfg.get("esm3_cache_path", None),
        )
        self._embed_model.eval()
        for p in self._embed_model.parameters():
            p.requires_grad_(False)

        # Pre-compute flat buffer layout for batched weight sync (1 broadcast vs 398).
        self._compute_wsync_layout(self._embed_model)

        # vLLM engine already created in _init_vllm_early_dedicated — verify it exists.
        assert self._vllm_llm is not None, (
            "vLLM LLM should have been initialized in _init_vllm_early_dedicated"
        )

        # Generic PG setup (training_pg + wsync_pg) + gen param seeding.
        # Must be called in same new_group order as _setup_bioreason_models on training ranks.
        self._setup_dedicated_vllm_rank(cfg)

        log.info(
            "Rank %d (vLLM server): setup complete — embed_model loaded, wsync_pg created, "
            "num_steps=%d",
            self.rank, self._total_steps,
        )

    def _setup_bioreason_models(self, cfg: DictConfig) -> None:
        """Instantiate BioReasonModel for policy and ref — no FSDP/checkpointer needed.

        BioReason loads ESM3 + GO graph encoder + projectors + Qwen3-4B backbone
        from a single checkpoint directory. The 4B model fits on 1-2 XPU tiles
        without FSDP sharding at the batch sizes used for GRPO RL training.
        """
        from torchtune.dev.bioreason.model import BioReasonModel

        ckpt_dir = cfg.base_model_path
        # PEFT-LoRA knobs (matches published BioReason-Pro RL recipe defaults).
        # When enable_lora=True the policy backbone is frozen + adapter-trained;
        # the ref model is ALWAYS full (enable_lora=False) — it must stay the
        # frozen SFT base the KL is measured against.
        self._enable_lora = bool(cfg.get("enable_lora", False))
        _lora_rank = int(cfg.get("lora_rank", 16))
        _lora_alpha = int(cfg.get("lora_alpha", 32))
        _lora_dropout = float(cfg.get("lora_dropout", 0.05))
        # ESM3 pre-encode cache (optional): when set, neither policy nor ref builds
        # the ESM3 encoder — they look up cached per-residue features. Frees
        # ~5.5 GiB/tile and removes the per-step encoder forward.
        _esm3_cache_path = cfg.get("esm3_cache_path", None)
        # Fail fast on a stale cache: the cache keys are sha1(sequence[:max_protein_len]),
        # so a cache encoded at a DIFFERENT max_protein_len than the dataset config would
        # KeyError deep in build_prompt_embeds. The model's _load_esm3_cache checks the
        # ESM3 model name but not the length, so cross-check the sidecar here where the
        # dataset's max_protein_len is visible.
        if _esm3_cache_path is not None:
            _ds_cfg = cfg.get("dataset", None)
            _cfg_mpl = _ds_cfg.get("max_protein_len", None) if _ds_cfg is not None else None
            _sidecar = _esm3_cache_path + ".json"
            if _cfg_mpl is not None and os.path.exists(_sidecar):
                with open(_sidecar) as _f:
                    _cache_mpl = json.load(_f).get("max_protein_len")
                if _cache_mpl is not None and int(_cache_mpl) != int(_cfg_mpl):
                    raise ValueError(
                        f"ESM3 cache max_protein_len mismatch: cache sidecar="
                        f"{_cache_mpl} vs dataset.max_protein_len={_cfg_mpl}. "
                        f"Re-encode with precompute_esm3_cache.py --max_protein_len "
                        f"{_cfg_mpl}, or point esm3_cache_path at the matching cache."
                    )
        _r = int(os.environ.get("RANK", "?"))
        def _mark(tag):
            print(f"[BIOMARK r{_r}] {tag}", file=sys.stderr, flush=True)
        # Resume a trained LoRA adapter (e.g. continue a 4N run at 8N). Points at a
        # dir with adapter_model.safetensors (what save_checkpoint writes). Only the
        # POLICY loads it; the ref stays the frozen full SFT model. None = fresh init.
        _adapter_path = cfg.get("lora_adapter_path", None)
        # On resume, the TRAINED projections live next to the adapter (save_checkpoint
        # writes adapter/ + protein_projection.pt + go_projection.pt into the SAME epoch
        # dir). adapter_path points at the adapter/ subdir, so its parent is the proj
        # dir. Overlay them so resume continues the trained projectors, not the SFT base
        # (without this, the LoRA adapter resumes but the trainable projectors silently
        # restart from SFT init). Override via cfg.proj_resume_dir if the layout differs.
        _proj_resume_dir = cfg.get("proj_resume_dir", None)
        if _proj_resume_dir is None and _adapter_path is not None:
            import os as _os
            _proj_resume_dir = _os.path.dirname(_adapter_path.rstrip("/"))
        _mark("policy:start")
        log.info(
            "BioReason: loading policy model from %s (enable_lora=%s, adapter=%s, proj_resume=%s)",
            ckpt_dir, self._enable_lora, _adapter_path, _proj_resume_dir,
        )
        self._model = BioReasonModel(
            ckpt_dir=ckpt_dir,
            device=self._device,
            dtype=self._dtype,
            enable_lora=self._enable_lora,
            lora_rank=_lora_rank,
            lora_alpha=_lora_alpha,
            lora_dropout=_lora_dropout,
            esm3_cache_path=_esm3_cache_path,
            adapter_path=_adapter_path,
            proj_resume_dir=_proj_resume_dir,
        )
        _mark("policy:loaded")
        self._model.train()
        if self._enable_activation_checkpointing:
            self._model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            log.info("BioReason: gradient checkpointing enabled on backbone")
            _mark("policy:ac_enabled")

        ref_device = torch.device("cpu") if self._ref_cpu_offload else self._device
        _mark(f"ref:start dev={ref_device}")
        log.info("BioReason: loading ref model from %s (device=%s)", ckpt_dir, ref_device)
        self._ref_model = BioReasonModel(
            ckpt_dir=ckpt_dir,
            device=ref_device,
            dtype=self._dtype,
            esm3_cache_path=_esm3_cache_path,
        )
        _mark("ref:loaded")
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        self._ref_model_device = ref_device

        # BioReasonHFTokenizer exposes pad_id, eos_id, stop_tokens (missing on raw HF tok).
        from torchtune.dev.bioreason.dataset import BioReasonHFTokenizer
        self._tokenizer = BioReasonHFTokenizer(ckpt_dir=ckpt_dir)

        self._use_fsdp1 = False
        self._fsdp2_param_groups_meta = []
        self._tune_to_hf_map = {}
        self._vocab_size = self._model.vocab_size
        self._checkpointer = None
        self._is_bioreason = True
        # Move ref model to XPU only during ref forward, then back to CPU.
        # Saves ~8 GiB HBM during backward while keeping XPU ref forward speed.
        self._bioreason_dynamic_ref_offload = True

        if self._is_rank_zero:
            trainable = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
            log.info(
                "BioReason setup: vocab=%d, trainable=%.3fB params",
                self._vocab_size,
                trainable / 1e9,
            )

        # Dedicated vLLM mode: wrap policy in FSDP1 SHARD_GRAD_OP (ZeRO-2) over
        # training ranks (0..N-2). Rank N-1 is the vLLM server and does not reach this.
        # SHARD_GRAD_OP shards gradients and optimizer states (ZeRO-2); params are
        # AllGathered during forward/backward (replicated during compute) and sharded
        # at rest. For 11 ranks this reduces gradient memory from 8 GiB to 0.73 GiB
        # and optimizer moments from 16 GiB to 1.45 GiB — eliminating the DDP bucket
        # pinning that forced forward_batch_size=4 in earlier runs.
        # FSDP2 (fully_shard) is NOT used — it deadlocks with oneCCL per-layer comms.
        # COLOCATE also FSDP-wraps (NEW 2026-06-18): each tile holds an in-process
        # vLLM engine (~24 GiB) AND the policy. Without FSDP the full 4B model is
        # replicated per tile and collides with vLLM → banned:1 (NotPresent) at
        # step-1 generation. Wrap with FULL_SHARD (ZeRO-3) + reshard_after_forward
        # so params are SHARDED at rest (freed after fwd/bwd), leaving room for the
        # resident vLLM weights — exactly the dense LoRA-colocate fix (its config:
        # "reshard_after_forward MANDATORY for colocate; ZeRO-2 default OOMs").
        _is_colocate = self._vllm_mode in ("colocate", "colocate_sleep")
        _wrap_fsdp1 = (
            (self._vllm_mode == "dedicated_rank" and self._vllm_dedicated_rank is not None)
            or (self._vllm_mode == "server")
            or _is_colocate
        )
        if _wrap_fsdp1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
            if self._vllm_mode == "dedicated_rank":
                # Generic PG setup: training_pg (xccl, [0..N-2]) + wsync_pg (gloo, [0, N-1]).
                # new_group order must match _setup_dedicated_vllm_rank on the vLLM rank.
                self._setup_dedicated_training_pgs(cfg)
            else:
                # server / colocate: all WORLD ranks are training ranks. server's vLLM
                # is on a separate node; colocate's vLLM is in-process per rank. Either
                # way no wsync PG (server ships over HTTP; colocate loads in-process).
                _training_ranks = list(range(self.world_size))
                self._training_pg = torch.distributed.new_group(_training_ranks, backend="xccl")
                self._wsync_pg = None
            _pre_wrap = self._model
            # Freeze the embed copy — replicated convenience tensor (not backbone's
            # embed_tokens), so FSDP should NOT shard it. With requires_grad=False,
            # FSDP excludes it from the flat param and keeps it replicated on each rank.
            # This lets build_full_embeds() work correctly outside the FSDP forward context.
            _pre_wrap._embed.requires_grad_(False)
            _mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            # HSDP (server mode, data_parallel_replicate_dim>1): replicate the model
            # across nodes, FSDP-shard within each node — distinct prompts per replica
            # in PARALLEL (the throughput lever; batch_size only adds SEQUENTIAL prompts).
            # FSDP1 expresses this as _HYBRID_SHARD_ZERO2 over the 2D dp_mesh
            # (dp_replicate × dp_shard) the base __init__ built (grpo_full...:435). The
            # cross-replica grad all-reduce is NATIVE to HYBRID_SHARD, so replicas stay
            # in sync and weight-sync from rank 0 is correct. ignored_modules + the
            # frozen _embed handling are identical to the single-replica path.
            _is_hsdp = (
                self._vllm_mode == "server" and getattr(self, "_dp_replicate", 1) > 1
            )
            _ignored = [m for m in [_pre_wrap._embed,
                                    _pre_wrap.protein_encoder,
                                    _pre_wrap.go_encoder]
                        if m is not None and isinstance(m, torch.nn.Module)]
            if _is_hsdp:
                try:
                    _shard_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
                except AttributeError:
                    _shard_strategy = ShardingStrategy.HYBRID_SHARD
                # Route the inter-node grad all-reduce over gloo (XCCL cross-node leaks
                # CXI MR handles -> banned:1 ~step10); base helper, validated on AGPT-2B.
                try:
                    from torchtune.dev.rl.distributed import enable_fsdp1_hsdp_inter_node_gloo
                    enable_fsdp1_hsdp_inter_node_gloo()
                except Exception as _e:
                    log.warning("enable_fsdp1_hsdp_inter_node_gloo unavailable: %s", _e)
                self._model = FSDP(
                    _pre_wrap,
                    sharding_strategy=_shard_strategy,
                    mixed_precision=_mp_policy,
                    device_mesh=self._dp_mesh,
                    ignored_modules=_ignored,
                    use_orig_params=True,
                    limit_all_gathers=True,
                )
                log.info(
                    "Rank %d: FSDP1 HSDP (%s) over dp_mesh (replicate=%d x shard=%d)",
                    self.rank, _shard_strategy.name,
                    self._dp_replicate, self._dp_shard,
                )
            else:
                # colocate: FULL_SHARD (ZeRO-3) shards params at rest → frees ~11/12 of
                # the 4B footprint for the co-resident vLLM engine. server/dedicated
                # single-replica keep the validated SHARD_GRAD_OP (ZeRO-2; vLLM off-tile).
                _shard_strategy = (
                    ShardingStrategy.FULL_SHARD if _is_colocate
                    else ShardingStrategy.SHARD_GRAD_OP
                )
                self._model = FSDP(
                    _pre_wrap,
                    sharding_strategy=_shard_strategy,
                    mixed_precision=_mp_policy,
                    process_group=self._training_pg,
                    ignored_modules=_ignored,
                    use_orig_params=True,
                    device_id=self._device,
                )
            self._use_fsdp1 = True
            # Pre-compute chunked broadcast layout inside summon_full_params.
            # Outside summon_full_params, use_orig_params=True params reflect SHARD sizes
            # (not full), so chunk boundaries would be wrong. With rank0_only=True, all
            # training ranks see the correct full shapes; numel() is consistent.
            with FSDP.summon_full_params(self._model, writeback=False, rank0_only=True):
                self._compute_wsync_layout(self._model)
            _wsync_desc = (
                f"wsync_pg=[0,{self._vllm_dedicated_rank}]"
                if self._vllm_mode == "dedicated_rank"
                else "wsync=HTTP raw_bytes (no PG)"
            )
            log.info(
                "Rank %d: FSDP1 " + _shard_strategy.name + " wrapped over training_pg (%d ranks), "
                "ignored=[_embed, protein_encoder, go_encoder], %s",
                self.rank, len(_training_ranks), _wsync_desc,
            )
        else:
            self._training_pg = None
            self._wsync_pg = None

    # ── vLLM generation override ───────────────────────────────────────────────

    def _http_generate_from_embeds_cpu(
        self,
        embeds_list: list,
        batch_input_ids_cpu: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Pure vLLM HTTP round-trip from a pre-built CPU embeds list.

        THREAD-SAFE / XPU-FREE: this is the only part of BioReason generation
        that the async rollout producer thread may run. It touches NO XPU device
        and NO distributed collective — just HTTP POSTs to the vLLM server pool
        and CPU tensor assembly. The returned query_responses lives on CPU; the
        caller (sync path or async consumer) is responsible for moving it to the
        device and broadcasting it to the other ranks (see
        :meth:`_broadcast_query_responses`).

        Args:
            embeds_list: list of ``bsz`` CPU bf16 ``[P, H]`` prompt-embed tensors
                (already detached + contiguous on CPU by the caller, on the main
                thread, since slicing the FSDP-gathered ``prompt_embeds`` requires
                the gather to have happened on a training rank).
            batch_input_ids_cpu: ``[bsz, context_length]`` prompt token IDs on CPU
                (used only to fill the prompt prefix of the output tensor).
            context_length: prompt length.

        Returns:
            query_responses on CPU: ``[bsz, context_length + max_generated_tokens]``.
        """
        bsz = len(embeds_list)
        total_len = context_length + self._max_generated_tokens
        gen_kwargs = dict(
            max_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k or 0,
            top_p=getattr(self, "_top_p", 1.0),
            stop_token_ids=getattr(self, "_stop_token_ids_list", None),
        )
        t0 = time.perf_counter()
        num_clients = len(self._vllm_clients)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        _seqs_per_engine = int(os.environ.get("TORCHTUNE_VLLM_SEQS_PER_ENGINE", "4"))
        _seqs_per_engine = max(1, _seqs_per_engine)
        _want_engines = max(1, (bsz + _seqs_per_engine - 1) // _seqs_per_engine)
        # Async lookahead is single-replica only (dp_replicate==1 is enforced in
        # setup), so the engine-band partitioning collapses to the validated
        # single-leader [0..) assignment. Keep the formula for parity.
        _n_rep = max(1, getattr(self, "_dp_replicate", 1))
        # TORCHTUNE_VLLM_REPLICA_BANDS (default 1 = fixed): partition the engine pool
        # into dp_replicate disjoint bands so each replica's shard-leader hits its own
        # engines (all 12 used, zero cross-leader contention). =0 restores the OLD
        # buggy behavior (every leader starts at engine 0 -> piles onto 0..3, idles
        # 4..11) for same-node A/B measurement. No effect when dp_replicate<=1.
        _bands_on = os.environ.get("TORCHTUNE_VLLM_REPLICA_BANDS", "1") != "0"
        _replica_idx = (self.rank // self._dp_shard) if _n_rep > 1 else 0
        if _bands_on:
            _band = max(1, num_clients // _n_rep)
            _eng_base = (_replica_idx % _n_rep) * _band
            _is_last_band = (_replica_idx == _n_rep - 1)
            _band_size = (num_clients - _eng_base) if _is_last_band else _band
        else:
            _eng_base = 0
            _band_size = num_clients
        _n_engines = max(1, min(_want_engines, _band_size))
        _engine_ids = [(_eng_base + e) % num_clients for e in range(_n_engines)]
        _groups: list[list[int]] = [[] for _ in range(_n_engines)]
        for _i in range(bsz):
            _groups[_i % _n_engines].append(_i)

        def _call_group(client, idxs):
            embeds = [embeds_list[j] for j in idxs]
            out = client.generate_from_embeds(prompt_embeds=embeds, **gen_kwargs)
            return {idxs[k]: (out[k] if out and k < len(out) else []) for k in range(len(idxs))}

        completions = [None] * bsz
        with ThreadPoolExecutor(max_workers=_n_engines) as pool:
            futures = [
                pool.submit(_call_group, self._vllm_clients[_engine_ids[g]], _groups[g])
                for g in range(_n_engines) if _groups[g]
            ]
            for future in as_completed(futures):
                for _gi, _comp in future.result().items():
                    completions[_gi] = _comp
        gen_time = time.perf_counter() - t0

        # CPU assembly — NO XPU. Consumer moves to device.
        query_responses = torch.full(
            (bsz, total_len), self._tokenizer.pad_id, dtype=batch_input_ids_cpu.dtype
        )
        query_responses[:, :context_length] = batch_input_ids_cpu
        for i, comp in enumerate(completions):
            length = min(len(comp), self._max_generated_tokens)
            if length:
                query_responses[i, context_length : context_length + length] = torch.tensor(
                    comp[:length], dtype=batch_input_ids_cpu.dtype
                )
        total_tokens = sum(len(c) for c in completions)
        log.info(
            "Rank %d: vLLM-embeds HTTP: %d seqs over %d engines (ids=%s), %d tokens in "
            "%.1fs (%.1f tok/s)",
            self.rank, bsz, _n_engines, _engine_ids, total_tokens, gen_time,
            total_tokens / max(gen_time, 0.01),
        )
        return query_responses

    def _generate_with_vllm_server_embeds(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """vLLM server mode for multimodal (BioReason): POST prompt_embeds.

        Differs from _generate_with_vllm: instead of token IDs, sends per-prompt
        bf16 embedding tensors (built from ESM3+GO+projectors on the train side).
        Each replica handles a round-robin slice of the batch in parallel.

        Returns:
            query_responses: ``[B*G, context_length + max_generated_tokens]``
        """
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        # HSDP (dp_replicate>1): each replica's SHARD LEADER generates its own distinct
        # prompt slice and broadcasts to its node-local followers via _gloo_dp_shard_pg
        # (in _broadcast_query_responses). Single-replica: _is_shard_leader == _is_rank_zero
        # (base __init__), so this is byte-identical to the validated path.
        _generates = getattr(self, "_is_shard_leader", self._is_rank_zero)
        if _generates:
            assert prompt_embeds is not None and prompt_embeds.shape[0] == bsz, (
                f"prompt_embeds required for vllm_server_embeds; got "
                f"{None if prompt_embeds is None else prompt_embeds.shape}, bsz={bsz}"
            )
            embeds_list = [prompt_embeds[i].detach().cpu().contiguous() for i in range(bsz)]
            gen_kwargs = dict(
                max_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k or 0,
                top_p=getattr(self, "_top_p", 1.0),
                stop_token_ids=getattr(self, "_stop_token_ids_list", None),
            )

            t0 = time.perf_counter()
            num_clients = len(self._vllm_clients)
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # GENERATION BATCHING (2026-06-22): the old path submitted ONE request per
            # prompt round-robin'd across all 12 engines -> ~1 seq/engine -> SINGLE-STREAM
            # decode (~50 tok/s). vLLM engines batch concurrent seqs at ~175 tok/s
            # (Running:3-4) — 3-4x faster — but only if each engine gets MULTIPLE seqs.
            # KV cache sits at ~4% so there's huge headroom. Fix: GROUP the bsz embeds
            # into per-engine batches (target ~TORCHTUNE_VLLM_SEQS_PER_ENGINE seqs each)
            # and submit ONE multi-embed call per engine (the client already accepts a
            # list -> vLLM batches them on that tile). This also REDUCES concurrent POSTs
            # (fewer in-flight HTTP -> fewer simultaneous IPC handles, the old banned:1
            # risk at G>=16). Set TORCHTUNE_VLLM_SEQS_PER_ENGINE=1 to restore the old
            # spread-thin behavior.
            _seqs_per_engine = int(os.environ.get("TORCHTUNE_VLLM_SEQS_PER_ENGINE", "4"))
            _seqs_per_engine = max(1, _seqs_per_engine)
            # Number of engines THIS leader wants = ceil(bsz / seqs_per_engine).
            _want_engines = max(1, (bsz + _seqs_per_engine - 1) // _seqs_per_engine)

            # REPLICA-DISJOINT ENGINE ASSIGNMENT (2026-06-23 straggler fix):
            # Under HSDP (dp_replicate>1) every shard leader (ranks 0, 12, 24 at
            # dp_replicate=3) runs THIS method concurrently and they ALL point at the
            # SAME `num_clients` vLLM URLs. The previous code used clients[g % num_clients]
            # starting at g=0 for every leader, so all R leaders piled onto engines
            # [0.._want_engines) — at the prod envelope (bsz=16, spe=4) that is engines
            # 0-3 carrying 3×4=12 concurrent seqs each while engines 4-11 sat 100% IDLE.
            # Measured cost: gen ~92s/step, 27s mean spread across the 3 leaders, only
            # 4 of 12 engines used. Fix: partition the engine pool into `n_rep` disjoint
            # contiguous bands and give each replica its own band, so the 48 seqs/step
            # (R×bsz) spread uniformly over all `num_clients` engines with no
            # cross-leader contention. Falls back to the old [0..) base when
            # dp_replicate<=1 (single leader → byte-identical to the validated path).
            _n_rep = max(1, getattr(self, "_dp_replicate", 1))
            _replica_idx = (self.rank // self._dp_shard) if _n_rep > 1 else 0
            # Engines available to THIS replica: an even contiguous band of the pool.
            _band = max(1, num_clients // _n_rep)
            _eng_base = (_replica_idx % _n_rep) * _band
            # This leader uses min(want, band) engines from its band (capped so two
            # replicas never share an engine; the last band absorbs the remainder).
            _is_last_band = (_replica_idx == _n_rep - 1)
            _band_size = (num_clients - _eng_base) if _is_last_band else _band
            _n_engines = max(1, min(_want_engines, _band_size))
            # Global engine indices this leader will hit (disjoint across replicas).
            _engine_ids = [(_eng_base + e) % num_clients for e in range(_n_engines)]
            # Contiguous groups so each engine call carries ~_seqs_per_engine embeds.
            _groups: list[list[int]] = [[] for _ in range(_n_engines)]
            for _i in range(bsz):
                _groups[_i % _n_engines].append(_i)

            def _call_group(client, idxs):
                embeds = [embeds_list[j] for j in idxs]
                out = client.generate_from_embeds(prompt_embeds=embeds, **gen_kwargs)
                # out is list aligned with embeds; map back to global indices.
                return {idxs[k]: (out[k] if out and k < len(out) else []) for k in range(len(idxs))}

            completions = [None] * bsz
            with ThreadPoolExecutor(max_workers=_n_engines) as pool:
                futures = [
                    pool.submit(_call_group, self._vllm_clients[_engine_ids[g]], _groups[g])
                    for g in range(_n_engines) if _groups[g]
                ]
                for future in as_completed(futures):
                    for _gi, _comp in future.result().items():
                        completions[_gi] = _comp
            gen_time = time.perf_counter() - t0
            if self._is_rank_zero:
                log.info(
                    "Rank 0: gen fan-out: bsz=%d over %d engines (ids=%s, %d seqs/engine "
                    "target, replica=%d/%d) -> batched decode",
                    bsz, _n_engines, _engine_ids, _seqs_per_engine, _replica_idx, _n_rep,
                )

            query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
            query_responses[:, :context_length] = batch_input_ids
            for i, comp in enumerate(completions):
                length = min(len(comp), self._max_generated_tokens)
                query_responses[i, context_length : context_length + length] = torch.tensor(
                    comp[:length], dtype=batch_input_ids.dtype, device=self._device
                )

            total_tokens = sum(len(c) for c in completions)
            log.info(
                "Rank %d: vLLM-embeds generation: %d sequences (%d clients), %d tokens in "
                "%.1fs (%.1f tok/s)",
                self.rank, bsz, num_clients, total_tokens, gen_time,
                total_tokens / max(gen_time, 0.01),
            )
        else:
            query_responses = batch_input_ids.new_empty(bsz, total_len)

        return self._broadcast_query_responses(query_responses)

    # ── Async generation lookahead (BioReason) ─────────────────────────────────

    def _build_prompt_embeds_for_batch(self, batch: dict):
        """Collective: build the expanded ``[B*G, P, H]`` CPU prompt_embeds for a
        raw dataloader ``batch``. MUST be called on EVERY training rank together
        (it runs ``FSDP.summon_full_params``, a world collective, when the model
        is FSDP-wrapped). Returns ``None`` if the batch carries no proteins (no
        embeds path) so the caller can fall back to the token path.

        This is the collective half of BioReason generation. The async lookahead
        runs it on the main thread (all ranks) one step ahead; only the rank-0 CPU
        slice it produces is then handed to the pure-HTTP producer thread.
        """
        protein_sequences = batch.get("protein_sequences", None)
        if protein_sequences is None or not hasattr(self._policy, "build_prompt_embeds"):
            return None
        input_ids = batch["tokens"].to(self._device)
        batch_size = input_ids.shape[0]
        grpo_size = self.grpo_samples
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        import contextlib
        _gather_ctx = (
            FSDP.summon_full_params(self._model, writeback=False)
            if isinstance(self._model, FSDP) else contextlib.nullcontext()
        )
        with torch.no_grad(), _gather_ctx:
            pe_base = self._policy.build_prompt_embeds(input_ids, protein_sequences)  # [B,P,H] CPU
        prompt_embeds = (
            pe_base.unsqueeze(1)
            .expand(-1, grpo_size, -1, -1)
            .reshape(batch_size * grpo_size, pe_base.shape[1], pe_base.shape[2])
            .contiguous()
        )  # [B*G, P, H] CPU
        return prompt_embeds

    def _async_lookahead_iter_impl(self, dataloader):
        """BioReason async generation/training overlap (server mode, staleness=1).

        Pipeline boundary (see CLAUDE.md async constraints):
          * COLLECTIVE / XPU work — built on the MAIN thread on every rank, one
            step ahead: ``prompt_embeds`` (ESM3 cache + GO + trainable projectors
            under ``summon_full_params``). All ranks resume the generator's
            ``next()`` together, so the collective is safe.
          * PURE HTTP — run on the per-shard-leader :class:`RolloutProducer`
            thread: ``generate_from_embeds`` over the vLLM pool + CPU assembly.
            No XPU, no collective (the load-bearing thread-safety property).
          * CONSUMER (main thread, all ranks) — broadcasts the query_responses and
            runs ref/policy fwd + bwd + optimizer + weight-sync synchronously.

        HSDP (dp_replicate>1) — PER-REPLICA generation. Each replica's SHARD
        LEADER (global ranks 0, dp_shard, 2*dp_shard, ...) runs its OWN producer
        thread and POSTs its replica's DISTINCT prompt slice to the shared vLLM
        pool. The consume-time broadcast is NODE-LOCAL over ``_gloo_dp_shard_pg``
        (from each shard-leader's GLOBAL rank), exactly like the validated SYNC
        path (:meth:`_generate_with_vllm_server_embeds` ->
        :meth:`_broadcast_query_responses`). The single-replica path
        (dp_replicate==1) is the SPECIAL CASE: ``_is_shard_leader`` is rank 0 and
        the broadcast group is the world group, so it stays byte-identical to the
        previously-validated 2N single-replica async path.

        Followers (non-shard-leaders) DO build embeds (collective) and DO take the
        node-local broadcast, but issue no HTTP — they pre-allocate an empty
        query_responses in generate_trajectory and receive the leader's tensor.
        The producer thread runs only on shard leaders.

        STALENESS PIN (=1). The work item is tagged with the weight version that
        is live at the MAIN-THREAD post point (before the consumer trains/bumps for
        any intervening batch), passed to the producer via the ``_weight_version``
        key so the producer does NOT re-snapshot at pickup. Without this pin the
        tag drifts by the queue depth and the consume-time lag plateaus at 2 (one
        extra weight sync slips in between pickup and consume). With it, rollout i
        is generated under the weights live one sync before it is trained on
        (lag == 1) deterministically. See WeightVersionTracker / RolloutProducer.
        """
        from torchtune.dev.rl.async_rollout import RolloutProducer

        # This rank GENERATES iff it is its replica's shard-leader. Single-replica:
        # _is_shard_leader == _is_rank_zero (set in base __init__). HSDP: ranks
        # 0, dp_shard, 2*dp_shard, ... each lead their replica.
        _is_leader = getattr(self, "_is_shard_leader", self._is_rank_zero)

        # The MAIN thread is the SOLE dataloader driver (the embeds build is a
        # collective, so iteration must stay aligned across ranks). The producer
        # does NOT iterate the dataloader; its "batch source" is a mailbox the main
        # thread fills with pre-built CPU work items. This keeps RolloutProducer's
        # bounded-queue + weight-version-tagging machinery while moving the data
        # iteration to the (collective-safe) main thread.
        from queue import Queue as _Q

        # Mailbox: main thread -> producer. Each item is a dict carrying the CPU
        # embeds work plus the main-thread weight-version snapshot (_weight_version,
        # used by RolloutProducer to pin the tag); a None sentinel signals
        # end-of-data. Bounded by max_staleness for back-pressure.
        _http_inbox: _Q = _Q(maxsize=self._async_generation_max_staleness)

        def _next_batch():
            # Runs in the producer thread. Blocks on the mailbox the main thread
            # fills. Returns None at end-of-data so RolloutProducer exits cleanly.
            return _http_inbox.get()

        def _produce_one(work):
            # Runs in the producer thread. `work` is the mailbox dict posted by the
            # main thread. Pure HTTP + CPU assembly — NO XPU, NO collective.
            qr_cpu = self._http_generate_from_embeds_cpu(
                work["embeds_list"], work["bii_cpu"], work["ctx"]
            )
            return qr_cpu, {}

        producer = RolloutProducer(
            produce_fn=_produce_one,
            batch_iter_fn=_next_batch,
            weight_versions=self._weight_versions,
            max_staleness=self._async_generation_max_staleness,
            name="bioreason_rollout_producer",
        )
        self._rollout_producer = producer if _is_leader else None

        # The producer thread (shard leaders only) drives HTTP. Followers do NOT
        # start it (they never POST), but DO run the collective embeds build + the
        # node-local broadcast inline in generate_trajectory. To keep the
        # generator's control flow identical across ranks (so collectives stay
        # aligned), the embeds build for the lookahead batch is driven HERE on
        # every rank.
        if _is_leader:
            producer.start()
            log.info(
                "Rank %d (replica leader): BioReason rollout producer started "
                "(max_staleness=%d, HTTP-only overlap; embeds built on main "
                "thread; dp_replicate=%d).",
                self.rank, self._async_generation_max_staleness,
                getattr(self, "_dp_replicate", 1),
            )

        # Lookahead pipeline (one-step-ahead, staleness=1):
        #   For each batch i, on EVERY rank:
        #     1. build batch i's prompt_embeds (collective, all ranks aligned)
        #     2. leader: post batch i's CPU embeds_list to its producer mailbox
        #        (tagged with the version live NOW, before training i-1) so the
        #        producer starts batch i's HTTP IMMEDIATELY (overlapping the
        #        consumer's training on batch i-1)
        #     3. if a previous batch is pending: stash its prompt_embeds, pull its
        #        finished HTTP result (leader) into _pending_async_query_responses,
        #        and yield it to the train loop (which broadcasts + trains)
        # The producer runs batch i's HTTP while the trainer processes batch i-1 →
        # the overlap. A one-slot lag buffer keeps embeds and query_responses for
        # the SAME batch together.
        _pending = None  # (batch, prompt_embeds) awaiting its HTTP result

        def _consume(prev_batch, prev_pe):
            # Set on EVERY rank so generate_trajectory's server branch is symmetric
            # (leaders have the HTTP result; followers broadcast-receive). Reset by
            # the no-protein fallback below for batches async cannot overlap.
            self._async_consume_active = True
            self._pending_async_prompt_embeds = prev_pe
            if _is_leader:
                item = producer.get()
                if item is None:
                    return False  # producer exhausted unexpectedly
                self._pending_async_query_responses = (
                    item.batch_meta["rollout_payload"].to(self._device)
                )
                self._last_rollout_item = item
                _w_now = self._weight_versions.version
                log.info(
                    "Rank %d: async consume (producer_latency=%.1fs, qsize=%d, "
                    "rollout_w_ver=%d, cur_w_ver=%d, lag=%d)",
                    self.rank, item.produce_latency_s, producer.qsize(),
                    item.weight_version, _w_now,
                    max(0, _w_now - item.weight_version),
                )
            else:
                self._pending_async_query_responses = None
            return True

        try:
            for batch in dataloader:
                prompt_embeds = self._build_prompt_embeds_for_batch(batch)
                if prompt_embeds is None:
                    # No protein path for this batch — async cannot overlap it.
                    # Fall back to fully synchronous generation for this batch:
                    # leave both stashes None so generate_trajectory takes the
                    # inline embeds/token path. Drain any pending slot first.
                    if _pending is not None:
                        if not _consume(*_pending):
                            break
                        yield _pending[0]
                        _pending = None
                    # This batch has no protein path — generate it fully
                    # synchronously (inline token path). Async OFF for this one.
                    self._async_consume_active = False
                    self._pending_async_prompt_embeds = None
                    self._pending_async_query_responses = None
                    yield batch
                    continue
                if _is_leader:
                    bsz = prompt_embeds.shape[0]
                    bii_cpu = batch["tokens"][:, None, :].expand(
                        -1, self.grpo_samples, -1
                    ).reshape(bsz, -1).cpu()
                    ctx = batch["tokens"].shape[1]
                    embeds_list = [prompt_embeds[i].contiguous() for i in range(bsz)]
                    # STALENESS PIN: snapshot the weight version on the MAIN thread
                    # at post time (deterministically ordered w.r.t. the per-step
                    # bumps) and hand it to the producer so it does NOT drift at
                    # pickup. Lag == 1 at consume. See RolloutProducer._run.
                    _http_inbox.put({
                        "embeds_list": embeds_list,
                        "bii_cpu": bii_cpu,
                        "ctx": ctx,
                        "_weight_version": self._weight_versions.version,
                    })

                if _pending is not None:
                    if not _consume(*_pending):
                        break
                    yield _pending[0]
                _pending = (batch, prompt_embeds)

            # Drain the final pending slot (its HTTP was dispatched above).
            if _pending is not None:
                if _consume(*_pending):
                    yield _pending[0]
        finally:
            if _is_leader:
                # Unblock the producer if it is still waiting on the mailbox, then
                # stop it. (stop() also drains the result queue.)
                try:
                    _http_inbox.put_nowait(None)
                except Exception:
                    pass
                producer.stop()
            self._rollout_producer = None
            self._pending_async_query_responses = None
            self._pending_async_prompt_embeds = None
            self._async_consume_active = False

    def _generate_with_colocated_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate using this rank's colocated vLLM engine.

        Args:
            prompt_embeds: ``[B*G, ctx_len, H]`` CPU tensor for multimodal inputs.
                When provided, passes embeddings to vLLM instead of token IDs.
                Requires vLLM initialised with ``enable_prompt_embeds=True``.
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

        if prompt_embeds is not None:
            # Multimodal: pass pre-computed embeddings to vLLM (CPU tensors required).
            vllm_prompts = [{"prompt_embeds": prompt_embeds[i]} for i in range(bsz)]
        else:
            # Text-only: strip padding and pass token ID lists.
            raw_prompts = []
            for i in range(bsz):
                ids = batch_input_ids[i].cpu().tolist()
                ids = [t for t in ids if t != self._tokenizer.pad_id]
                raw_prompts.append(ids)
            vllm_prompts = [{"prompt_token_ids": p} for p in raw_prompts]

        t0 = time.perf_counter()
        outputs = self._vllm_llm.generate(vllm_prompts, sampling_params=sampling_params)
        gen_time = time.perf_counter() - t0

        query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
        query_responses[:, :context_length] = batch_input_ids
        for i, out in enumerate(outputs):
            ids = out.outputs[0].token_ids
            length = min(len(ids), self._max_generated_tokens)
            query_responses[i, context_length : context_length + length] = torch.tensor(
                ids[:length], dtype=batch_input_ids.dtype, device=self._device
            )

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        log.info(
            "Rank %d: colocated vLLM generation: %d sequences, %d tokens in %.1fs (%.1f tok/s)",
            self.rank, bsz, total_tokens, gen_time, total_tokens / max(gen_time, 0.01),
        )
        return query_responses

    # ── Trajectory generation override ────────────────────────────────────────

    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
        protein_sequences: Optional[list] = None,
    ) -> GRPOTrajectory:
        """
        Generates a trajectory, with BioReason multimodal support.

        When protein_sequences is provided, pre-computes ESM3+GO prompt embeddings
        and uses the inputs_embeds path for policy/ref forward passes.
        """
        from torchtune import generation as torchtune_generation
        from torchtune.modules import local_kv_cache
        from torchtune.dev.rl.generation import generate

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if not _colocate_vllm_mode:
            device_empty_cache(self._device)
        elif self._vllm_mode == "colocate_sleep" and self._vllm_llm is not None and hasattr(self, '_vllm_is_sleeping') and self._vllm_is_sleeping:
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            log.info("Rank %d: waking up vLLM for generation", self.rank)
            t_wake = time.perf_counter()
            self._vllm_llm.wake_up(tags=["weights"])
            # colocate_sleep syncs weights here (NOT the train loop). _sync_colocated_
            # weights is overridden to route to the LoRA merge when _has_lora. Runs
            # after wake(weights) (engine live to receive load_weights) and BEFORE
            # wake(kv_cache) so the merge transient doesn't co-reside with the KV pool.
            self._sync_colocated_weights()
            self._vllm_llm.wake_up(tags=["kv_cache"])
            self._vllm_is_sleeping = False
            log.info("Rank %d: vLLM wake_up + weight sync completed in %.2fs",
                     self.rank, time.perf_counter() - t_wake)
        elif self._vllm_mode == "colocate" and self._vllm_llm is not None:
            # Plain colocate (vLLM resident, no sleep): the WEIGHT sync runs in the
            # base train() loop (_run_wsync_block → _sync_colocated_weights, which
            # BioReason overrides to the LoRA merge). Here we only restore KV cache
            # shapes (if a prior step zeroed them) + reset prefix cache before gen.
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            if hasattr(self, '_vllm_kv_cache_shapes'):
                kv_caches = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
                for i, (shape, dtype) in enumerate(self._vllm_kv_cache_shapes):
                    kv_caches[i] = torch.zeros(shape, dtype=dtype, device=self._device)
                del self._vllm_kv_cache_shapes
            self._vllm_llm.llm_engine.reset_prefix_cache()

        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        # ASYNC FAST PATH: when the async lookahead consumed a rollout for THIS
        # batch, it ALREADY built the prompt_embeds one step earlier (and stashed
        # them) and the producer ALREADY ran the HTTP. Reuse the stash and SKIP the
        # inline collective embeds build below — rebuilding here would run a second
        # summon_full_params per step (the exact cost the lookahead exists to hide)
        # and is wasteful even though it would produce the same tensor. The stash is
        # set on every rank by _async_lookahead_iter_impl._consume; on non-rank-0
        # ranks _pending_async_query_responses stays None but the embeds stash and
        # the consume-active flag are still set so the path stays symmetric.
        _async_consume = getattr(self, "_async_consume_active", False)
        prompt_embeds = None
        if _async_consume and getattr(self, "_pending_async_prompt_embeds", None) is not None:
            prompt_embeds = self._pending_async_prompt_embeds
            self._pending_async_prompt_embeds = None
        elif protein_sequences is not None and hasattr(self._policy, 'build_prompt_embeds'):
            # Multimodal: build prompt embeddings once per unique prompt, then expand
            # to B*G. build_prompt_embeds(input_ids [B,P], protein_sequences [B]) ->
            # [B,P,H] on CPU. protein_projection and go_projection are trainable ->
            # FSDP-sharded at rest; summon_full_params gathers them so the projector
            # forward sees complete weights. (SYNC path — byte-identical to the
            # validated baseline when async is disabled.)
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            import contextlib
            _gather_ctx = (
                FSDP.summon_full_params(self._model, writeback=False)
                if isinstance(self._model, FSDP) else contextlib.nullcontext()
            )
            with torch.no_grad(), _gather_ctx:
                pe_base = self._policy.build_prompt_embeds(
                    input_ids.to(self._device), protein_sequences
                )  # [B, P, H] CPU
            prompt_embeds = (
                pe_base.unsqueeze(1)
                .expand(-1, grpo_size, -1, -1)
                .reshape(batch_size * grpo_size, pe_base.shape[1], pe_base.shape[2])
                .contiguous()
            )  # [B*G, P, H] CPU

        # step 1: generate responses
        _vllm_t0 = time.perf_counter()
        if self._vllm_mode in ("colocate", "colocate_sleep"):
            query_responses = self._generate_with_colocated_vllm(
                batch_input_ids, context_length, prompt_embeds=prompt_embeds
            )
        elif self._vllm_mode == "dedicated_rank":
            bsz = batch_input_ids.shape[0]
            total_len = context_length + self._max_generated_tokens
            if self._is_rank_zero:
                query_responses_cpu = self._generate_with_dedicated_vllm(
                    batch_input_ids, context_length, protein_sequences
                )
                query_responses = query_responses_cpu.to(self._device)
            else:
                query_responses = batch_input_ids.new_empty(bsz, total_len)
            torch.distributed.broadcast(query_responses, src=0, group=self._training_pg)
        elif self._vllm_mode == "server":
            # ASYNC LOOKAHEAD (checked BEFORE the inline embeds path). When async
            # generation is engaged for THIS batch (_async_consume_active is set on
            # EVERY rank by _async_lookahead_iter_impl._consume), the prompt_embeds
            # for this batch were already built one step earlier (and reused above)
            # and the rank-0 producer thread already ran the HTTP, stashing the
            # query_responses in _pending_async_query_responses. The consumer here
            # ONLY runs the world broadcast (all ranks together) — it does NOT
            # rebuild embeds or re-issue the HTTP. Gated on the all-ranks
            # _async_consume flag (NOT on _pending_async_query_responses, which is
            # None on non-rank-0) so every rank takes the SAME branch and the
            # broadcast collective stays aligned. The inline (synchronous) path
            # below is byte-identical to the validated baseline when async is off
            # (_async_consume is always False then).
            _async_consume = getattr(self, "_async_consume_active", False)
            if _async_consume:
                bsz = batch_input_ids.shape[0]
                total_len = context_length + self._max_generated_tokens
                # Each replica's SHARD LEADER holds its replica's HTTP result;
                # followers pre-allocate the empty buffer and receive it over the
                # NODE-LOCAL broadcast in _broadcast_query_responses (which uses
                # _shard_leader_global_rank + _gloo_dp_shard_pg under HSDP). Single-
                # replica: _is_shard_leader == rank 0 and the broadcast is the world
                # group, so this is byte-identical to the validated 2N async path.
                _is_leader = getattr(self, "_is_shard_leader", self._is_rank_zero)
                if _is_leader:
                    query_responses = self._pending_async_query_responses
                    assert query_responses is not None, (
                        "async consume active but no shard-leader query_responses stashed"
                    )
                    assert query_responses.shape == (bsz, total_len), (
                        f"async qr shape mismatch: got {tuple(query_responses.shape)}, "
                        f"expected ({bsz}, {total_len})"
                    )
                else:
                    query_responses = batch_input_ids.new_empty(bsz, total_len)
                self._pending_async_query_responses = None
                query_responses = self._broadcast_query_responses(query_responses)
            elif getattr(self, "_is_bioreason", False) and prompt_embeds is not None:
                query_responses = self._generate_with_vllm_server_embeds(
                    batch_input_ids, context_length, prompt_embeds
                )
            else:
                query_responses = self._generate_with_vllm(batch_input_ids, context_length)
        else:
            _stop_tokens = (
                None if self._dp_replicate > 1
                else self._tokenizer.stop_tokens
            )
            with local_kv_cache(
                model=self._model,
                batch_size=batch_size * grpo_size,
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
                    rng=self._rng if self._device.type == "cuda" else None,
                    stop_tokens=_stop_tokens,
                    return_logits=False,
                )

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _vllm_time = time.perf_counter() - _vllm_t0

        if self._vllm_mode not in ("server", "dedicated_rank") and not self._production_mode:
            torch.distributed.barrier()

        # Free vLLM GPU memory for training forward/backward passes.
        if _colocate_vllm_mode and self._vllm_llm is not None:
            if torch.xpu.is_available():
                mem_before = torch.xpu.memory_allocated(self._device) / 1024**3
            if self._vllm_mode == "colocate_sleep":
                log.info("Rank %d: sleeping vLLM (weights + KV cache) for training", self.rank)
                t_free = time.perf_counter()
                self._vllm_llm.sleep(level=1)
                self._vllm_is_sleeping = True
            else:
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
                log.info("Rank %d: vLLM memory freed in %.1fs", self.rank,
                         time.perf_counter() - t_free)

        responses = query_responses[:, context_length:].clone()

        vocab_size = getattr(self, '_vocab_size', None)
        if vocab_size is not None and vocab_size > 0:
            oob_mask = responses >= vocab_size
            if oob_mask.any():
                log.warning("Clamping %d OOB token IDs (max=%d, vocab=%d)",
                            oob_mask.sum().item(), responses.max().item(), vocab_size)
                responses = responses.clamp(max=vocab_size - 1)
                query_responses = torch.cat([query_responses[:, :context_length], responses], dim=1)

        query_response_padding_masks = query_responses != self._tokenizer.pad_id
        masks = torchtune_generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = torchtune_generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        num_seqs = query_responses.shape[0]
        fwd_bs = self._forward_batch_size

        # step 2: rollout-time policy logprobs (only when needed for IS ratios)
        if self._ppo_epochs > 1 or self._compute_rollout_logprobs_required:
            _policy_fwd_t0 = time.perf_counter()
            with torch.no_grad():
                if fwd_bs >= num_seqs:
                    log.info("Rank %d: policy forward start (shape=%s)",
                             self.rank, list(query_responses.shape))
                    if prompt_embeds is not None:
                        _full_emb = self._policy.build_full_embeds(prompt_embeds, responses)
                        _attn_mask = (query_responses != self._tokenizer.pad_id).long()
                        logits = self._model(
                            inputs_embeds=_full_emb, attention_mask=_attn_mask,
                            position_ids=position_ids,
                        )
                        del _full_emb, _attn_mask
                    else:
                        logits = self._model(query_responses, input_pos=position_ids, mask=masks)
                    log.info("Rank %d: policy forward done", self.rank)
                    logits = logits[:, context_length - 1:]
                    logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
                    del logits
                else:
                    log.info("Rank %d: policy forward start CHUNKED (total=%d, chunk=%d)",
                             self.rank, num_seqs, fwd_bs)
                    logprobs_chunks = []
                    for cs in range(0, num_seqs, fwd_bs):
                        ce = min(cs + fwd_bs, num_seqs)
                        if prompt_embeds is not None:
                            _full_emb = self._policy.build_full_embeds(
                                prompt_embeds[cs:ce], responses[cs:ce]
                            )
                            _attn_mask = (query_responses[cs:ce] != self._tokenizer.pad_id).long()
                            chunk_logits = self._model(
                                inputs_embeds=_full_emb, attention_mask=_attn_mask,
                                position_ids=position_ids[cs:ce],
                            )
                            del _full_emb, _attn_mask
                        else:
                            chunk_logits = self._model(
                                query_responses[cs:ce],
                                input_pos=position_ids[cs:ce],
                                mask=masks[cs:ce],
                            )
                        chunk_logits = chunk_logits[:, context_length - 1:]
                        logprobs_chunks.append(
                            rlhf.batched_logits_to_logprobs(
                                chunk_logits, responses[cs:ce], self._temperature
                            )
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

        # step 2.1: ref model logprobs
        _ref_fwd_t0 = time.perf_counter()
        log.info("Rank %d: pre-ref forward", self.rank)
        if not self._production_mode:
            self._training_barrier()

        # Dynamic ref offload: move ref model to XPU for fast ref forward.
        if getattr(self, '_bioreason_dynamic_ref_offload', False):
            self._ref_model.to(self._device)
            log.info("Rank %d: ref model → XPU for ref forward", self.rank)

        _ref_dev = next(self._ref_model.parameters()).device
        log.info("Rank %d: ref model device=%s, position_ids.device=%s",
                 self.rank, _ref_dev, position_ids.device)
        if fwd_bs >= num_seqs:
            log.info("Rank %d: ref forward start", self.rank)
            if prompt_embeds is not None:
                _full_emb = self._ref_model.build_full_embeds(prompt_embeds, responses)
                _attn_mask = (query_responses != self._tokenizer.pad_id).long().to(_ref_dev)
                ref_logits = self._ref_model(
                    inputs_embeds=_full_emb, attention_mask=_attn_mask,
                    position_ids=position_ids.to(_ref_dev),
                ).to(self._device)
                del _full_emb, _attn_mask
            else:
                ref_logits = self._ref_model(
                    query_responses, input_pos=position_ids, mask=masks
                )
            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, responses, self._temperature
            )
            del ref_logits
        else:
            log.info("Rank %d: ref forward start CHUNKED (total=%d, chunk=%d)",
                     self.rank, num_seqs, fwd_bs)
            ref_logprobs_chunks = []
            for cs in range(0, num_seqs, fwd_bs):
                ce = min(cs + fwd_bs, num_seqs)
                if prompt_embeds is not None:
                    _full_emb = self._ref_model.build_full_embeds(
                        prompt_embeds[cs:ce], responses[cs:ce]
                    )
                    _attn_mask = (
                        query_responses[cs:ce] != self._tokenizer.pad_id
                    ).long().to(_ref_dev)
                    chunk_ref_logits = self._ref_model(
                        inputs_embeds=_full_emb, attention_mask=_attn_mask,
                        position_ids=position_ids[cs:ce].to(_ref_dev),
                    ).to(self._device)
                    del _full_emb, _attn_mask
                else:
                    chunk_ref_logits = self._ref_model(
                        query_responses[cs:ce],
                        input_pos=position_ids[cs:ce],
                        mask=masks[cs:ce],
                    )
                chunk_ref_logits = rlhf.truncate_sequence_for_logprobs(
                    chunk_ref_logits, context_length
                )
                ref_logprobs_chunks.append(
                    rlhf.batched_logits_to_logprobs(
                        chunk_ref_logits, responses[cs:ce], self._temperature
                    )
                )
                del chunk_ref_logits
                # empty_cache leaks UR handles under FSDP + in-process vLLM
                # (colocate) → banned:1. Safe in server/dedicated modes.
                if not _colocate_vllm_mode:
                    device_empty_cache(self._device)
            ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
            del ref_logprobs_chunks
            log.info("Rank %d: ref forward done (chunked)", self.rank)
        if not _colocate_vllm_mode:
            device_empty_cache(self._device)

        # Dynamic ref offload: move ref model back to CPU to free XPU HBM for backward.
        if getattr(self, '_bioreason_dynamic_ref_offload', False):
            self._ref_model.to('cpu')
            log.info("Rank %d: ref model → CPU after ref forward (freed ~8 GiB XPU)", self.rank)
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if self._is_rank_zero:
            log.info(
                "Rank 0: post-ref-fwd alloc=%.2f GiB resv=%.2f GiB",
                torch.xpu.memory_allocated(self._device) / 1e9,
                torch.xpu.memory_reserved(self._device) / 1e9,
            )
        _ref_fwd_time = time.perf_counter() - _ref_fwd_t0

        log.info(
            "Rank %d: GENTIMING vllm=%.1fs policy_fwd=%.1fs ref_fwd=%.1fs",
            self.rank, _vllm_time, _policy_fwd_time, _ref_fwd_time,
        )

        (response_padding_masks, responses) = rlhf.truncate_sequence_at_first_stop_token(
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
        elif self._reward_mode == "bioreason":
            from torchtune.dev.bioreason.reward import bioreason_reward_fn as _br_reward
            _decoded, _expanded_answers = [], []
            _resp_lens = []
            _has_eos = []
            for _b in range(batch_size):
                for _g in range(grpo_size):
                    _ids = responses[_b, _g]
                    _non_pad = _ids[_ids != self._tokenizer.pad_id]
                    _decoded.append(self._tokenizer.decode(_non_pad.cpu().tolist()))
                    _expanded_answers.append(answers[_b])
                    _rlen = int(_non_pad.numel())
                    _resp_lens.append(_rlen)
                    # Stop detection (for the stop_rate DIAG). Two cases count as
                    # "model chose to stop" (i.e. NOT a max_gen truncation):
                    #   (a) a configured stop token is present in the returned ids, OR
                    #   (b) the sequence ended BEFORE max_generated_tokens. vLLM with
                    #       stop_token_ids stops at EOS but by default does NOT include
                    #       the stop token in the output (include_stop_str_in_output=
                    #       False), so case (a) alone reads stop_rate=0.000 even when
                    #       vLLM correctly stopped — case (b) catches that. A seq that
                    #       hit the cap has _rlen >= max_gen and no stop token => not a stop.
                    _tok_present = bool(torch.isin(
                        _non_pad, torch.tensor(self._stop_token_ids,
                                               device=_non_pad.device)).any().item())
                    _under_cap = _rlen < int(self._max_generated_tokens)
                    _has_eos.append(_tok_present or _under_cap)
            _rw, _succ, _br_diag = _br_reward(
                _decoded, _expanded_answers, return_diagnostics=True,
                propagate_hierarchy=self._reward_propagate_hierarchy,
                obo_path=self._reward_obo_path,
            )
            rewards = _rw.view(batch_size, grpo_size, 1)
            successes = _succ.float().view(batch_size, grpo_size, 1)
            metadata = {}
            # Aggregate BioReason-specific diagnostics across ranks.
            self._log_bioreason_diagnostics(
                _br_diag,
                response_lens=_resp_lens,
                has_eos=_has_eos,
                rewards_bg=_rw.view(batch_size, grpo_size),
            )
        else:
            rewards, successes, metadata = batched_rewards(
                self._tokenizer, responses, answers, device=self._device
            )
        rewards = rewards.to(self._device)
        successes = successes.to(self._device)

        rewards = rewards.sum(dim=-1)
        successes = successes.sum(dim=-1)

        if self._is_rank_zero:
            try:
                sample_resp = responses[0, 0]
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

        # BioReason-Pro authors' fix for low-variance reward collapse: pool
        # mean/std across the full B*G batch instead of per-prompt-group, so a
        # single non-zero reward anywhere in the batch yields signal for every
        # rollout. Without this, when all G rollouts of a prompt get reward=0
        # (the common case early in training), advantages collapse to 0 and
        # gradient is exactly zero — the kl_loss-flat-at-0.003 failure mode.
        # Default on for bioreason reward; opt-out via batch_level_advantages: false.
        if self._batch_level_advantages:
            from torchtune.dev.bioreason.reward import batch_level_advantages
            advantages = batch_level_advantages(
                rewards.reshape(batch_size * grpo_size), group_size=grpo_size,
            )
        else:
            advantages = (rewards - rewards.mean(1, keepdim=True)) / (
                rewards.std(1, keepdim=True) + 1e-4
            )
            advantages = advantages.reshape(batch_size * grpo_size)
        # Log advantage stats so log parsers can confirm whether group_std=0 batches
        # produced zero advantages (pure KL update) or nonzero advantages (real policy
        # gradient).  Cross-reference with BIOREASON_DIAG group_std and METRICS kl_loss.
        if self._is_rank_zero:
            log.info(
                "BIOREASON_ADV step=%d adv_abs_max=%.4f adv_std=%.4f",
                self._steps_run, advantages.abs().max().item(), advantages.std().item(),
            )
        # Zero-signal skip lever: decide COLLECTIVELY whether to skip the optimizer
        # step. The base train loop gates optimizer.step() on _skip_optimizer_step
        # (only when skip_zero_advantage_step is enabled). All training ranks must
        # agree or FSDP collectives desync — so we all-reduce MAX of the local
        # advantage magnitude over _training_pg and skip only when EVERY rank's
        # advantages are zero (global no-signal step). A single rank with signal
        # keeps the step (grads are all-reduced across ranks anyway).
        if getattr(self, "_skip_zero_advantage_step", False):
            import torch.distributed as _dist
            _pg = getattr(self, "_training_pg", None)
            _local_max = advantages.abs().max().detach().to(self._device).reshape(1)
            if _dist.is_initialized():
                _dist.all_reduce(_local_max, op=_dist.ReduceOp.MAX, group=_pg)
            self._skip_optimizer_step = bool(_local_max.item() <= 1e-8)
            if self._skip_optimizer_step and self._is_rank_zero:
                log.info(
                    "BIOREASON_SKIP step=%d: global advantage ~0 — optimizer step "
                    "will be skipped", self._steps_run,
                )
        del responses
        if not _colocate_vllm_mode:  # empty_cache leaks UR handles under colocate
            device_empty_cache(self._device)

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
            prompt_embeds=prompt_embeds,  # None for text-only; [B*G, P, H] CPU for multimodal
        )

    def _log_bioreason_diagnostics(
        self,
        diag: dict,
        response_lens: list[int],
        has_eos: list[bool],
        rewards_bg: torch.Tensor,
    ) -> None:
        """All-reduce BioReason rollout diagnostics and emit one line on rank 0.

        BIOREASON_DIAG step=N n=… go_emit=… nonzero_rew=…
            mean_pred=… mean_tp=… len_mean=… len_p95=…
            trunc_rate=… stop_rate=… group_std=… batch_std=…

        Reasoning vs throughput trade-off: rollouts are ~32-34s and many of these
        counters are tiny (B*G ≤ 32 typically), so building int32/float32 tensors
        and one all-reduce adds <1ms per step.
        """
        try:
            import torch.distributed as _dist
            # In dedicated_rank mode the vLLM rank is in _run_vllm_generation_server()
            # and never joins training collectives.  Use _training_pg so only training
            # ranks participate; None falls back to the default world group (correct for
            # server/colocate modes where every world rank is a training rank).
            pg = getattr(self, "_training_pg", None)
            ws = _dist.get_world_size(group=pg) if _dist.is_initialized() else 1
            dev = self._device

            pred = diag["pred_count"].to(dev).float()
            tp = diag["tp_count"].to(dev).float()
            has_pred = diag["has_pred"].to(dev).float()
            lens = torch.tensor(response_lens, dtype=torch.float32, device=dev)
            stops = torch.tensor(has_eos, dtype=torch.float32, device=dev)
            rb = rewards_bg.to(dev).float()
            nonzero = (rb > 0).float()
            # Per-prompt-group reward std (mean over groups), and overall batch std.
            if rb.shape[1] > 1:
                group_stds = rb.std(dim=1, unbiased=False)
            else:
                group_stds = torch.zeros(rb.shape[0], device=dev)
            # Truncation: response reached max_generated_tokens AND no stop token.
            max_gen = float(self._max_generated_tokens)
            trunc = ((lens >= max_gen) & (stops == 0)).float()

            # Reduce sums + sum-of-squares for variance, plus a single scan
            # tensor for length percentile (approximated as max).
            local_n = float(lens.numel())
            sums = torch.stack([
                pred.sum(), tp.sum(), has_pred.sum(), nonzero.sum(),
                lens.sum(), stops.sum(), trunc.sum(),
                group_stds.sum(), torch.tensor(float(rb.shape[0]), device=dev),
                rb.sum(), (rb * rb).sum(),
            ])
            count = torch.tensor([local_n], device=dev)
            len_max = lens.max().unsqueeze(0) if lens.numel() else torch.tensor([0.0], device=dev)
            if ws > 1:
                _dist.all_reduce(sums, op=_dist.ReduceOp.SUM, group=pg)
                _dist.all_reduce(count, op=_dist.ReduceOp.SUM, group=pg)
                _dist.all_reduce(len_max, op=_dist.ReduceOp.MAX, group=pg)
            n = count.item()
            if n <= 0:
                return
            n_groups = sums[8].item() or 1.0
            r_sum = sums[9].item()
            r_sqsum = sums[10].item()
            r_mean = r_sum / n
            batch_var = max(r_sqsum / n - r_mean * r_mean, 0.0)

            if self._is_rank_zero:
                # step= is the pre-increment count (steps completed before this
                # trajectory).  Parsers that cross-reference with BATCH_REWARD or
                # training logs should use the same 0-based convention: step N here
                # corresponds to base-recipe step N before _steps_run += 1.
                log.info(
                    "BIOREASON_DIAG step=%d n=%d go_emit=%.3f nonzero_rew=%.3f "
                    "mean_pred=%.2f mean_tp=%.2f len_mean=%.1f len_max=%.0f "
                    "trunc_rate=%.3f stop_rate=%.3f group_std=%.4f batch_std=%.4f",
                    self._steps_run, int(n),
                    sums[2].item() / n,        # go_emit
                    sums[3].item() / n,        # nonzero_rew
                    sums[0].item() / n,        # mean_pred
                    sums[1].item() / n,        # mean_tp
                    sums[4].item() / n,        # len_mean
                    len_max.item(),
                    sums[6].item() / n,        # trunc_rate
                    sums[5].item() / n,        # stop_rate
                    sums[7].item() / n_groups, # group_std (mean over groups)
                    batch_var ** 0.5,          # batch_std
                )
        except Exception as e:
            log.warning("BIOREASON_DIAG log failed: %s", e)

    def generate_trajectory_batched(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
        protein_sequences: Optional[list] = None,
    ) -> GRPOTrajectory:
        """Generates trajectories in gen_batch_size vLLM-generation chunks.

        Chunk by gen_batch_size (NOT forward_batch_size): generation is a vLLM HTTP
        round, decoupled from the training/ref micro-batch. With fbs=1 the old code
        chunked here by 1 → `batch_size` SEQUENTIAL vLLM calls/step; at batch_size>1
        ~half the prompts' calls returned EMPTY completions (vLLM server state across
        rapid repeated calls). gen_batch_size defaults to batch_size, so all prompts
        go in ONE generation call — matching the validated batch_size=1 path (one call).
        Set gen_batch_size < batch_size only if a single vLLM call OOMs the engine.
        """
        trajectories: list[GRPOTrajectory] = []
        _gen_bs = getattr(self, "_gen_batch_size", self.batch_size)
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, _gen_bs):
                batch_input_ids = input_ids[batch_start : batch_start + _gen_bs]
                batch_answers = answers[batch_start : batch_start + _gen_bs]
                batch_proteins = (
                    protein_sequences[batch_start : batch_start + _gen_bs]
                    if protein_sequences is not None else None
                )
                # empty_cache leaks UR handles under colocate (FSDP + in-process
                # vLLM); the wake path's gc.collect()+synchronize is the safe sub.
                if not _colocate_vllm_mode:
                    device_empty_cache(self._device)
                trajectories.append(
                    self.generate_trajectory(batch_input_ids, batch_answers, batch_proteins)
                )
                if not _colocate_vllm_mode:
                    device_empty_cache(self._device)

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

    # ── GRPO step override ────────────────────────────────────────────────────

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """
        GRPO optimization step with BioReason inputs_embeds support.

        When trajectory.prompt_embeds is set, uses build_full_embeds() for the
        policy forward instead of token IDs (inputs_embeds path).
        """
        if self._device.type == "xpu":
            torch.xpu.synchronize()

        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "pre_forward", log=log)
            if self._device.type == "xpu":
                try:
                    torch.xpu.reset_peak_memory_stats()
                except RuntimeError:
                    pass

        _fwd_t0 = time.perf_counter()
        _multimodal = trajectory.prompt_embeds is not None

        if self._enable_packing and not _multimodal:
            from torchtune.dev.rl.packing import pack_trajectory_for_training, unpack_tensor
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
            # Single forward + single backward (non-EP only; includes multimodal).
            total_seqs = trajectory.query_responses.shape[0]
            grad_scale = max(1, self._gradient_accumulation_steps)

            log.info("Rank %d: single-backward forward start (total=%d seqs)",
                     self.rank, total_seqs)
            _fwd_t0_sb = time.perf_counter()
            if _multimodal:
                _comp_ids = trajectory.query_responses[:, context_length:]
                _full_emb = self._policy.build_full_embeds(trajectory.prompt_embeds, _comp_ids)
                _attn_mask = (trajectory.query_responses != self._tokenizer.pad_id).long()
                pi_logits = self._model(
                    inputs_embeds=_full_emb,
                    attention_mask=_attn_mask,
                    position_ids=trajectory.position_ids,
                )
                del _full_emb, _attn_mask, _comp_ids
            else:
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
                    "trajectory.logprobs is None"
                )
            old_logprobs = (
                trajectory.logprobs if trajectory.logprobs is not None else pi_logprobs.detach()
            )
            loss, policy_loss, kl_loss, ratios, clipfrac = self._loss_fn(
                old_logprobs,
                pi_logprobs,
                trajectory.ref_logprobs,
                trajectory.advantages,
                padding_masks=~trajectory.response_padding_masks,
            )

            log.info("Rank %d: single-backward backward start", self.rank)
            _bwd_t0_sb = time.perf_counter()
            from torchtune.dev.rl.distributed import _orig_reduce_scatter_tensor
            import torch.distributed as _tdist_sb_fix
            _rsc_patch_saved = _tdist_sb_fix.reduce_scatter_tensor
            _tdist_sb_fix.reduce_scatter_tensor = _orig_reduce_scatter_tensor
            try:
                (loss / grad_scale).backward()
            finally:
                _tdist_sb_fix.reduce_scatter_tensor = _rsc_patch_saved
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _bwd_total = time.perf_counter() - _bwd_t0_sb
            log.info("Rank %d: single-backward backward=%.1fs", self.rank, _bwd_total)
            _fwd_time = _fwd_time_sb

        else:
            # Chunked training forward+backward.
            total_seqs = trajectory.query_responses.shape[0]
            fwd_bs = self._forward_batch_size
            num_fwd_chunks = (total_seqs + fwd_bs - 1) // fwd_bs
            grad_scale = num_fwd_chunks * max(1, self._gradient_accumulation_steps)

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
                    log.info(
                        "Rank 0: PRE-train-fwd[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce,
                        torch.xpu.memory_allocated() / 1024**3,
                        torch.xpu.memory_reserved() / 1024**3,
                    )
                log.info("Rank %d: grpo_step chunk[%d:%d] fwd", self.rank, _cs, _ce)
                if _multimodal:
                    _chunk_comp_ids = trajectory.query_responses[_cs:_ce, context_length:]
                    _chunk_full_emb = self._policy.build_full_embeds(
                        trajectory.prompt_embeds[_cs:_ce], _chunk_comp_ids
                    )
                    _chunk_attn_mask = (
                        trajectory.query_responses[_cs:_ce] != self._tokenizer.pad_id
                    ).long()
                    _c_logits = self._model(
                        inputs_embeds=_chunk_full_emb,
                        attention_mask=_chunk_attn_mask,
                        position_ids=trajectory.position_ids[_cs:_ce],
                    )
                    del _chunk_full_emb, _chunk_attn_mask, _chunk_comp_ids
                else:
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
                _c_pi_lp.masked_fill_(trajectory.response_padding_masks[_cs:_ce], 1.0)
                del _c_logits
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                if self._device.type == "xpu" and self._is_rank_zero:
                    log.info(
                        "Rank 0: POST-train-fwd[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce,
                        torch.xpu.memory_allocated() / 1024**3,
                        torch.xpu.memory_reserved() / 1024**3,
                    )

                if self._compute_rollout_logprobs_required:
                    assert trajectory.logprobs is not None, (
                        "async_generation / always_compute_rollout_logprobs is set but "
                        "trajectory.logprobs is None"
                    )
                _c_old_lp = (
                    trajectory.logprobs[_cs:_ce]
                    if trajectory.logprobs is not None
                    else _c_pi_lp.detach()
                )
                _c_loss, _c_pol, _c_kl, _c_rat, _c_clip = self._loss_fn(
                    _c_old_lp,
                    _c_pi_lp,
                    trajectory.ref_logprobs[_cs:_ce],
                    trajectory.advantages[_cs:_ce],
                    padding_masks=~trajectory.response_padding_masks[_cs:_ce],
                )
                _chunk_losses.append(_c_loss.detach())
                _chunk_policy_losses.append(_c_pol.detach())
                _chunk_kl_losses.append(_c_kl.detach())
                _chunk_ratios.append(_c_rat.detach())
                _chunk_clipfracs.append(_c_clip.detach())
                _chunk_pi_logprobs.append(_c_pi_lp.detach())

                _bwd_t0 = time.perf_counter()
                if _use_fsdp2_grad_sync and not _is_last_chunk:
                    self._model.set_requires_gradient_sync(False)
                if _use_fsdp1_no_sync and not _is_last_chunk:
                    _bwd_ctx = self._model.no_sync()
                elif _use_ddp_no_sync and not _is_last_chunk:
                    _bwd_ctx = self._model.no_sync()
                else:
                    import contextlib
                    _bwd_ctx = contextlib.nullcontext()
                with _bwd_ctx:
                    (_c_loss / grad_scale).backward()
                if _use_fsdp2_grad_sync and _is_last_chunk:
                    self._model.set_requires_gradient_sync(True)
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _bwd_total += time.perf_counter() - _bwd_t0

            loss = torch.stack(_chunk_losses).mean()
            policy_loss = torch.stack(_chunk_policy_losses).mean()
            kl_loss = torch.stack(_chunk_kl_losses).mean()
            # stack().mean() not cat(): GRPOSimpleLoss returns ratios as a 0-dim
            # scalar (torch.tensor(1.0)); torch.cat can't concatenate 0-dim tensors
            # (crashes only on the chunked path, fbs < num_seqs). Mirrors the base
            # recipe (grpo_full_finetune_distributed_xpu.py:4294).
            ratios = torch.stack(_chunk_ratios).mean()
            clipfrac = torch.stack(_chunk_clipfracs).mean()
            pi_logprobs = torch.cat(_chunk_pi_logprobs)
            _fwd_time = time.perf_counter() - _fwd_t0 - _bwd_total

        log.info("Rank %d: grpo_step bwd=%.1fs", self.rank, _bwd_total)

        with torch.no_grad():
            _old_lp = trajectory.logprobs if trajectory.logprobs is not None else pi_logprobs
            approx_policy_kls = (0.5 * (pi_logprobs - _old_lp).pow(2)).mean()

        return GRPOStats(
            loss,
            policy_loss,
            kl_loss,
            ratios,
            clipfrac,
            approx_policy_kls,
            None,  # metadata
        )

    # ── Hooks (subclass extension points; train() lives in base) ──────────────

    def _extract_batch_kwargs(self, batch: dict) -> dict:
        """Forward multimodal protein_sequences into ``generate_trajectory_batched``.

        Replaces a 180-line train() override that previously caused the missing
        weight-sync regression (project_bioreason_train_missing_wsync). The base
        train() now calls ``self._extract_batch_kwargs(batch)`` and splat-applies
        the result, so all sync/clip/optim/log behavior stays in one place.
        """
        if not getattr(self, "_is_bioreason", False):
            return {}
        return {"protein_sequences": batch.get("protein_sequences", None)}


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for BioReason GRPO recipe."""
    recipe = GRPOBioReasonDistributedXPU(cfg=cfg)
    config.log_config(recipe_name="GRPOBioReasonDistributedXPU", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

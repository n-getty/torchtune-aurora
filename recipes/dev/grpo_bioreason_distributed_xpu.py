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

import json
import logging
import os
import sys
import time
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

# Import the base recipe — it handles all the XPU/XCCL shim setup at import time.
# `recipes/__init__.py` deliberately raises on import (to keep tests from picking
# up the recipes package), so we load the sibling base recipe by file path.
import importlib.util as _importlib_util

import torch
from omegaconf import DictConfig

from torchtune import config, rlhf, training, utils
from torchtune.dev.bioreason.rollout_dump import (
    dump_rollout_groups,
    rollout_dump_path,
)
from torchtune.dev.rl.distributed import _slice_trajectory, device_empty_cache
from torchtune.dev.rl.generation import (
    compact_prompt_completion_batch,
    finish_response_logits,
    gather_response_logits,
    get_descending_response_chunk_ranges,
    get_length_sorted_response_chunks,
    get_right_padded_response_length,
    pad_response_logprobs,
    response_only_logits_kwargs,
    trim_query_responses_to_global_max,
)
from torchtune.dev.rl.ref_prefix_share import (
    prefix_share_supported,
    ref_prefix_share_enabled,
    shared_prefix_ref_logprobs,
)
from torchtune.dev.rl.rewards import batched_rewards, gene_recall_batched_rewards
from torchtune.dev.rl.behavior_logprobs import (
    audit_behavior_logprobs,
    broadcast_behavior_logprobs,
    build_behavior_logprobs,
    fit_behavior_logprobs_width,
    require_processed_mode,
    rows_needing_fallback,
)
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory

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


def _split_prompt_n_request_plan(
    request_plan: list[tuple[int, list[int], int]],
    num_clients: int,
    engine_stride: int,
    engine_phase: int = 0,
    prompts_per_request: int = 1,
) -> list[tuple[int, list[int], int]]:
    prompts_per_request = max(1, prompts_per_request)
    return [
        (
            (engine_id + engine_phase + split_idx * engine_stride) % num_clients,
            indices[start : start + request_n * prompts_per_request],
            request_n,
        )
        for engine_id, indices, request_n in request_plan
        for split_idx, start in enumerate(
            range(0, len(indices), request_n * prompts_per_request)
        )
    ]


def _split_choice_request_plan(
    request_plan: list[tuple[int, list[int], int]],
    num_clients: int,
    engine_stride: int,
) -> list[tuple[int, list[int], int]]:
    if engine_stride == 0:
        return request_plan
    return [
        (
            (engine_id + (choice_idx % request_n) * engine_stride) % num_clients,
            [index],
            1,
        )
        for engine_id, indices, request_n in request_plan
        for choice_idx, index in enumerate(indices)
    ]


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
        self._ref_forward_batch_size = cfg.get(
            "ref_forward_batch_size", cfg.forward_batch_size
        )
        self._trim_chunk_width = (
            os.environ.get("TORCHTUNE_TRIM_CHUNK_WIDTH", "0") == "1"
        )
        self._sort_policy_chunks_by_length = (
            os.environ.get("TORCHTUNE_SORT_POLICY_CHUNKS_BY_LENGTH", "0") == "1"
        )
        self._compact_prompt_chunks = (
            os.environ.get("TORCHTUNE_COMPACT_PROMPT_CHUNKS", "0") == "1"
        )
        self._skip_nonfinite_grad_step = cfg.get("skip_nonfinite_grad_step", True)
        if self._trim_chunk_width and self._is_rank_zero:
            log.info("BioReason per-microbatch sequence-width trimming ENABLED")
        if self._sort_policy_chunks_by_length and self._is_rank_zero:
            log.info("BioReason policy microbatches length sorting ENABLED")
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
            "reward_propagate_hierarchy",
            self._reward_mode == "bioreason",
        )
        # obo for reward propagation: explicit config, else the checkpoint dir (each
        # bioreason ckpt ships go-basic.obo), else reward.py's env/source fallback.
        self._reward_obo_path = cfg.get(
            "reward_obo_path",
            cfg.get("base_model_path", None),
        )
        # Pool advantage normalization across the full batch (BioReason-Pro fix).
        # Default true for bioreason mode (matches the upstream paper's GRPO setup);
        # explicit override possible via config field.
        self._batch_level_advantages = cfg.get(
            "batch_level_advantages",
            self._reward_mode == "bioreason",
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
            self._always_compute_rollout_logprobs or self._async_generation_enabled
        )

        # Use vLLM's own sampler logprobs as pi_old instead of paying a second
        # no-grad forward over the whole [B*G, P+C] batch. Async-only: the policy
        # forward it removes does not run at all when async is off and ppo_epochs==1.
        self._use_vllm_behavior_logprobs = (
            bool(cfg.get("use_vllm_behavior_logprobs", False))
            and self._async_generation_enabled
        )
        if self._use_vllm_behavior_logprobs:
            # WHAT THIS CAN AND CANNOT CHECK. api_server exposes only /load and
            # /version -- neither reports engine config -- so this asserts the mode
            # the LAUNCHER was told to pass, never one the server confirmed. If the
            # spawn line in run_bioreason_32b_Nnode_hsdp.sh lacks
            # --logprobs-mode processed_logprobs, this check still passes and the
            # ratios are silently wrong. Keep the two in sync by hand.
            require_processed_mode(cfg.get("vllm_logprobs_mode"))
        self._pending_async_behavior_logprobs = None

        # TORCHTUNE_BLP_AUDIT=N: for the first N steps, compute pi_old BOTH ways and
        # log the disagreement, then use the trainer's recompute. The substitution's
        # correctness cannot be established by the CPU tests (they pin alignment, not
        # numerics -- vLLM uses different kernels and TP sharding) and its failure is
        # silent, so this is the only instrument that can tell a working substitution
        # from a biased one. Costs the policy forward it exists to remove, hence a
        # step budget rather than a bool. 0 = off.
        try:
            self._blp_audit_steps = int(os.environ.get("TORCHTUNE_BLP_AUDIT", "0"))
        except ValueError:
            self._blp_audit_steps = 0

        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        self._eval_every_n_steps = cfg.get("eval_every_n_steps", 0)
        self._eval_max_examples = cfg.get("eval_max_examples", 50)

        stop_token_ids = (
            list(self._tokenizer.stop_tokens)
            if hasattr(self._tokenizer, "stop_tokens") and self._tokenizer.stop_tokens
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
            log.warning(
                "TORCHTUNE_VLLM_STOP_TOKENS=0: vLLM will NOT stop at EOS "
                "(every rollout decodes to max_tokens). A/B-only setting."
            )

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
        if getattr(self, "_is_bioreason", False):
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
        if hasattr(self._policy, "vllm_param_iter"):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # Step-stamped, NOT bare epoch_{epoch} (2026-09-14 fix). This recipe runs
            # with epochs=1, so curr_epoch stayed 0 for the whole run and EVERY
            # save_every_n_steps save rewrote the same epoch_0/ in place. The write is
            # non-atomic (a ~268 MiB safetensors + two ~70 MiB torch.saves, no
            # temp-then-rename), and this dir is the ONLY artifact — BioReason's
            # save_checkpoint short-circuits the base, so there is no recipe_state.pt
            # to fall back on (no optimizer/step/dataloader state is saved at all).
            # Consequences that bit us: (1) a kill mid-write leaves the sole copy torn;
            # (2) no rollback to an earlier step if the policy degrades; (3) a trend
            # eval reading the live dir races the trainer and cannot tell which step it
            # got. Embedding _steps_run gives one directory per save. Same root cause as
            # memory/feedback_epoch_dir_overwritten_within_same_epoch_20260817.md, but
            # strictly worse here because that path at least kept resume_state.pt.
            # Pinned by tests/torchtune/dev/rl/test_bioreason_checkpoint_step_stamped.py.
            save_dir = os.path.join(
                self._output_dir, f"epoch_{epoch}_step{getattr(self, '_steps_run', 0)}"
            )
            _fsdp = (
                getattr(self, "_use_fsdp1", False)
                and torch.distributed.is_initialized()
            )
            _has_lora = getattr(self._model, "_has_lora", False)

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
            if _fsdp and _has_lora:
                _param_to_name = {id(p): n for n, p in self._model.named_parameters()}
                _full_sd = {} if self._is_rank_zero else None
                _fsdp_units = [m for m in self._model.modules() if isinstance(m, FSDP)]
                for _unit in _fsdp_units:
                    with torch.no_grad(), FSDP.summon_full_params(
                        _unit, recurse=False, writeback=False, rank0_only=False
                    ):
                        if not self._is_rank_zero:
                            continue
                        for (
                            _param
                        ) in _base_module._weight_sync_module._fsdp1_own_params(
                            _unit, FSDP
                        ):
                            if not _param.requires_grad:
                                continue
                            _name = _param_to_name.get(id(_param))
                            if _name is not None and _name not in _full_sd:
                                _full_sd[_name] = _param.detach().cpu().contiguous()
                if self._is_rank_zero:
                    for _name, _param in self._model.named_parameters():
                        _clean_name = _name.replace("_fsdp_wrapped_module.", "")
                        _clean_name = _clean_name.replace(
                            "_checkpoint_wrapped_module.", ""
                        )
                        if (
                            _param.requires_grad
                            and _name not in _full_sd
                            and _clean_name.startswith(
                                ("protein_projection.", "go_projection.")
                            )
                        ):
                            _full_sd[_name] = _param.detach().cpu().contiguous()
                utils.log_rank_zero(
                    log,
                    f"BioReason checkpoint gathered {len(_full_sd or {})} trainable "
                    "tensors without materializing the frozen 32B backbone",
                )
            elif _fsdp:
                from torch.distributed.fsdp import StateDictType

                with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
                    _full_sd = self._model.state_dict()
            else:
                _full_sd = self._model.state_dict()

            if self._is_rank_zero:
                os.makedirs(save_dir, exist_ok=True)

                def _strip(name):
                    return name.replace("_fsdp_wrapped_module.", "").replace(
                        "_checkpoint_wrapped_module.", ""
                    )

                # Projections: pull protein_projection.* / go_projection.* (already full
                # tensors in the gathered CPU dict — clone to detach from any shared store).
                for _pname in ("protein_projection", "go_projection"):
                    _sub = {}
                    for k, v in _full_sd.items():
                        ck = _strip(k)
                        if ck.startswith(_pname + "."):
                            _sub[ck[len(_pname) + 1 :]] = v.detach().clone()
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
                                ck = ck[len("backbone.") :]
                            _adapter[ck] = v.detach().clone().contiguous()
                    save_file(
                        _adapter, os.path.join(_adir, "adapter_model.safetensors")
                    )
                    # adapter_config.json (PEFT loader needs it; mirror the ctor config).
                    try:
                        self._policy.backbone.peft_config["default"].save_pretrained(
                            _adir
                        )
                    except Exception:
                        import json as _json

                        with open(
                            os.path.join(_adir, "adapter_config.json"), "w"
                        ) as _f:
                            _json.dump({"peft_type": "LORA"}, _f)
                    log.info(
                        "BioReason checkpoint saved to %s (adapter, %d lora tensors)",
                        save_dir,
                        len(_adapter),
                    )
                else:
                    # Full backbone: save the (stripped) backbone.* tensors.
                    _bk = {
                        _strip(k)[len("backbone.") :]: v.detach().clone()
                        for k, v in _full_sd.items()
                        if _strip(k).startswith("backbone.")
                    }
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
        import contextlib
        import gc

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
            if isinstance(self._model, FSDP)
            else contextlib.nullcontext()
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
            self.rank,
            n_synced,
            time.perf_counter() - t0,
        )

    # Bind the base (inherited) colocate sync under a private name so the LoRA
    # override below can fall back to it for the non-LoRA (full-FT) path. The base
    # method is a class attribute (bound from the weight_sync module at the base
    # class body), so reference it via the base CLASS, not the module.
    _sync_colocated_weights_base = (
        GRPOFullFinetuneDistributedXPU._sync_colocated_weights
    )

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
        log.info(
            "Rank %d (vLLM server): loading embed model from %s", self.rank, ckpt_dir
        )
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
        assert (
            self._vllm_llm is not None
        ), "vLLM LLM should have been initialized in _init_vllm_early_dedicated"

        # Generic PG setup (training_pg + wsync_pg) + gen param seeding.
        # Must be called in same new_group order as _setup_bioreason_models on training ranks.
        self._setup_dedicated_vllm_rank(cfg)

        log.info(
            "Rank %d (vLLM server): setup complete — embed_model loaded, wsync_pg created, "
            "num_steps=%d",
            self.rank,
            self._total_steps,
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
            _cfg_mpl = (
                _ds_cfg.get("max_protein_len", None) if _ds_cfg is not None else None
            )
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
            ckpt_dir,
            self._enable_lora,
            _adapter_path,
            _proj_resume_dir,
        )
        # This policy model is about to be FSDP1-wrapped below (server/dedicated_rank/
        # colocate all set _wrap_fsdp1) — FSDP(..., device_id=self._device) shards and
        # places each rank's slice on GPU itself. Loading the backbone onto GPU HERE
        # first (the default path, kept byte-identical for the validated 4B configs)
        # would materialize the FULL backbone on every rank's single tile before FSDP
        # ever runs: fine at 4B (~8 GiB), fatal at 32B (~65.6 GiB > one 64 GiB tile —
        # confirmed on HW, every rank OOM'd inside `.to()` at construction). Gated
        # behind fsdp_full_shard_at_rest (already the 32B-only opt-in for the FSDP
        # sharding-strategy override below) rather than made the default everywhere,
        # so no existing 4B run's load path changes.
        _force_full_shard = cfg.get("fsdp_full_shard_at_rest", False)
        self._fsdp_full_shard_at_rest = bool(_force_full_shard)
        _will_wrap_fsdp1 = _force_full_shard and (
            (
                self._vllm_mode == "dedicated_rank"
                and self._vllm_dedicated_rank is not None
            )
            or (self._vllm_mode == "server")
            or (self._vllm_mode in ("colocate", "colocate_sleep"))
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
            backbone_cpu_init=_will_wrap_fsdp1,
        )
        _mark("policy:loaded")
        self._model.train()
        if self._enable_activation_checkpointing:
            self._model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            log.info("BioReason: gradient checkpointing enabled on backbone")
            _mark("policy:ac_enabled")

        # 32B-scale: FSDP2-shard the ref model instead of CPU-offloading it. The ref
        # model is frozen and forward-only, so per-layer FSDP2 sharding (fully_shard,
        # the SAME mechanism the validated 32B SFT path and the base recipe's 32B
        # dense-Qwen3 GRPO path already use in production) is a strictly better fit
        # than dragging a 65.6 GiB model through a CPU forward pass every step (10s of
        # minutes per step on HW, confirmed) or the FSDP1 flatten-buffer OOM a naive
        # top-level wrap hits (see the policy's own auto_wrap_policy fix above for that
        # failure mode). ref_cpu_offload stays available as an explicit opt-out for
        # anyone who wants the old (correct, just slow) behavior.
        _shard_ref_fsdp2 = bool(_force_full_shard) and not self._ref_cpu_offload
        ref_device = torch.device("cpu") if self._ref_cpu_offload else self._device
        _mark(f"ref:start dev={ref_device}")
        log.info(
            "BioReason: loading ref model from %s (device=%s, fsdp2_shard=%s)",
            ckpt_dir,
            ref_device,
            _shard_ref_fsdp2,
        )
        self._ref_model = BioReasonModel(
            ckpt_dir=ckpt_dir,
            device=ref_device,
            dtype=self._dtype,
            esm3_cache_path=_esm3_cache_path,
            # Same reasoning as the policy: if this backbone is about to be FSDP2-sharded,
            # don't materialize the full ~65.6 GiB backbone onto one GPU first — let
            # fully_shard's meta-init + broadcast path place shards directly.
            backbone_cpu_init=_shard_ref_fsdp2,
        )
        _mark("ref:loaded")
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        if _shard_ref_fsdp2:
            from torch.distributed._composable.fsdp import fully_shard

            # HSDP (dp_replicate>1): scope the ref model's per-layer all-gather to
            # THIS REPLICA's own dp_shard group, not the world default group.
            # mesh=None resolves to the default process group, which at
            # dp_replicate=1 (every 32B smoke test to date: 2N/4N/8N) trivially
            # equals the dp_shard group (there's only one replica, so "world" and
            # "shard" are the same ranks) — the bug was invisible until the first
            # dp_replicate>1 run. At dp_replicate>1 (16N production, dp_replicate=15)
            # mesh=None makes EVERY layer's unshard a 180-rank collective spanning
            # all replicas, even though each replica's ref model is independent and
            # never needs cross-replica sync. Root-caused on job 8813804/8813949:
            # `torch.distributed.DistStoreError: wait timeout after 600000ms` inside
            # `self._ref_model(...)`'s first forward, hitting one full replica's
            # ranks at a time (node 13 then node 14 across two separate runs) —
            # consistent with the LAST-connecting replica's ranks always being the
            # ones left waiting on a world-scoped barrier the others already passed.
            _ref_fsdp2_mesh = (
                self._dp_mesh["dp_shard"]
                if getattr(self, "_dp_replicate", 1) > 1
                else None
            )
            _ref_decoder_layer_cls = None
            for _rn, _rm in self._ref_model.backbone.named_modules():
                if _rn.endswith(".layers.0") or _rn.endswith("layers.0"):
                    _ref_decoder_layer_cls = type(_rm)
                    break
            if _ref_decoder_layer_cls is None:
                raise RuntimeError(
                    "fsdp_full_shard_at_rest=true but could not find a '...layers.0' "
                    "module on the ref model's backbone to FSDP2-shard."
                )
            _n_ref_layers_sharded = 0
            for _rn, _rm in reversed(list(self._ref_model.backbone.named_modules())):
                if isinstance(_rm, _ref_decoder_layer_cls):
                    fully_shard(_rm, mesh=_ref_fsdp2_mesh, reshard_after_forward=True)
                    _n_ref_layers_sharded += 1
            # Root wrap: mirrors torchtune.training.shard_model's final "shard the
            # entire model to account for stragglers" step — the ref model has no
            # optimizer/backward, so a single reshard_after_forward=True root unit is
            # sufficient (no need for the policy's ignored_modules dance; the ref
            # model's projections are never trained and never need to be reachable
            # via a `summon_full_params`-style gather — build_full_embeds only calls
            # them, it doesn't need to write to them).
            fully_shard(
                self._ref_model.backbone,
                mesh=_ref_fsdp2_mesh,
                reshard_after_forward=True,
            )
            log.info(
                "BioReason: ref model FSDP2-sharded (%d decoder layers + root unit)",
                _n_ref_layers_sharded,
            )
            ref_device = self._device
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
        # 32B-scale exception: this round-trip materializes the ENTIRE ref model on
        # one tile via a bare .to(device) call — ~65.6 GiB at 32B (confirmed OOM on
        # HW, "Tried to allocate 250 MiB" with 63.16 GiB already resident). The "~8
        # GiB" saving in the comment above is a 4B-scale number; at 32B the ref
        # model is never FSDP-sharded (ref_cpu_offload keeps it CPU-resident, full
        # stop), so this optimization is actively harmful, not just unnecessary.
        # Disabled whenever fsdp_full_shard_at_rest is set (the 32B-only opt-in);
        # the ref forward instead runs directly on CPU (see ref_cpu_offload).
        self._bioreason_dynamic_ref_offload = not cfg.get(
            "fsdp_full_shard_at_rest", False
        )

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
            (
                self._vllm_mode == "dedicated_rank"
                and self._vllm_dedicated_rank is not None
            )
            or (self._vllm_mode == "server")
            or _is_colocate
        )
        if _wrap_fsdp1:
            from torch.distributed.fsdp import (
                BackwardPrefetch,
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
            )

            if self._vllm_mode == "dedicated_rank":
                # Generic PG setup: training_pg (xccl, [0..N-2]) + wsync_pg (gloo, [0, N-1]).
                # new_group order must match _setup_dedicated_vllm_rank on the vLLM rank.
                self._setup_dedicated_training_pgs(cfg)
            else:
                # server / colocate: all WORLD ranks are training ranks. server's vLLM
                # is on a separate node; colocate's vLLM is in-process per rank. Either
                # way no wsync PG (server ships over HTTP; colocate loads in-process).
                _training_ranks = list(range(self.world_size))
                self._training_pg = torch.distributed.new_group(
                    _training_ranks, backend="xccl"
                )
                self._wsync_pg = None
                # LOAD-BEARING: a gloo-backed barrier group, used ONLY to order
                # _init_sender_pool's post-init barrier against rank 0's
                # concurrent _init_xccl_weight_sync() communicator construction.
                # Root-caused 2026-09-07 (jobs 8810079/8810147/8810186, v40-v43,
                # 4/4 reproductions across TWO different physical node pairs,
                # ruling out node-specific hardware corruption): the WORLD
                # default process group on XPU is ALSO xccl-backed (confirmed —
                # torchtune.training.xpu_utils.get_xpu_distributed_backend()
                # returns "xccl" absent CPU offload), so the "plain
                # torch.distributed.barrier()" that v29 tried and found
                # ineffective was STILL an XCCL collective under the hood — it
                # never actually tested a barrier that couldn't contend with
                # concurrent Level-Zero/XCCL communicator construction. Two
                # concurrent XCCL-touching operations on the same rank/device
                # (this barrier on an XCCL group + rank 0's fresh
                # ProcessGroupXCCL construction inside _init_xccl_weight_sync)
                # appear unsafe to interleave — `ur_die` reproduced identically
                # on rank 1 every time. A gloo barrier can never touch
                # Level-Zero state, so it can safely order ranks without racing
                # rank 0's XCCL construction. Mirrors the existing
                # `_wsync_pg`/`TORCHTUNE_WSYNC_BACKEND=gloo` pattern already
                # used for the dedicated_rank path's cross-PG (vllm_backend.py).
                import torch.distributed.distributed_c10d as _dc10d

                _default_pg = _dc10d._get_default_group()
                _orig_bound = _default_pg.bound_device_id
                _default_pg.bound_device_id = None
                try:
                    self._wsync_barrier_pg = torch.distributed.new_group(
                        _training_ranks, backend="gloo"
                    )
                finally:
                    _default_pg.bound_device_id = _orig_bound
            _pre_wrap = self._model
            # _embed is already frozen unconditionally in BioReasonModel.__init__
            # (_freeze_embed_copy) regardless of LoRA/full-FT. Re-assert it here
            # for the FSDP-specific reason: it's a replicated convenience tensor
            # (not backbone's embed_tokens), so FSDP should NOT shard it. With
            # requires_grad=False, FSDP excludes it from the flat param and keeps
            # it replicated on each rank, letting build_full_embeds() work
            # correctly outside the FSDP forward context.
            _pre_wrap._embed.requires_grad_(False)
            _mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            # 32B-scale override: a top-level-only FSDP1 wrap must flatten the ENTIRE
            # wrapped module into one contiguous buffer on one GPU before it can shard
            # (FlatParamHandle.flatten_tensors_into_flat_param -> torch.cat) — ~61 GiB for
            # the 32B backbone, confirmed OOM on HW even with backbone_cpu_init=True and
            # device_id set (CLAUDE.md's "per-module wrapping causes catastrophic overhead"
            # warning is about STEP-TIME at 4B scale, not a correctness constraint — at 32B,
            # a working-but-slower per-layer wrap is required just to fit at all). Only the
            # transformer decoder layers are wrapped as separate FSDP units (each ~1.85 GiB,
            # not ~61 GiB) via transformer_auto_wrap_policy; everything else in the backbone
            # stays inside the outer top-level unit exactly as before. Gated behind
            # fsdp_full_shard_at_rest so the validated 4B top-level-only path never changes.
            _auto_wrap_policy = None
            if _force_full_shard:
                import functools

                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

                # Discover the decoder layer class by scanning named_modules() rather than
                # calling backbone.get_decoder() directly — PEFT's PeftModel wraps the HF
                # model and proxies attribute access via __getattr__, which is reliable for
                # plain attributes but not verified here for a bound-method call chain
                # (get_decoder().layers[0]); scanning modules is robust regardless of
                # PEFT/LoRA wrapping.
                _decoder_layer_cls = None
                for _name, _mod in _pre_wrap.backbone.named_modules():
                    if _name.endswith(".layers.0") or _name.endswith("layers.0"):
                        _decoder_layer_cls = type(_mod)
                        break
                if _decoder_layer_cls is None:
                    raise RuntimeError(
                        "fsdp_full_shard_at_rest=true but could not find a '...layers.0' "
                        "module to derive the transformer decoder layer class for "
                        "auto_wrap_policy — backbone architecture may not match the "
                        "expected HF decoder-layer-list convention."
                    )
                _auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={_decoder_layer_cls},
                )
                log.info(
                    "fsdp_full_shard_at_rest: per-layer auto_wrap_policy on %s "
                    "(avoids the single ~61 GiB flatten-buffer OOM at 32B scale)",
                    _decoder_layer_cls,
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
            _ignore_candidates = [
                _pre_wrap._embed,
                _pre_wrap.protein_encoder,
                _pre_wrap.go_encoder,
            ]
            if _force_full_shard:
                # 32B-scale: also ignore the trainable projectors. At 4B, several call
                # sites (generate_trajectory's prompt_embeds build, LoRA delta sync) reach
                # these via a per-step `summon_full_params(self._model)` — cheap there
                # because it re-gathers the whole ~8 GiB backbone anyway. At 32B, with the
                # backbone split into 64 per-layer FSDP units (auto_wrap_policy), that same
                # call cascades into unsharding ALL 64 units simultaneously on every rank —
                # confirmed OOM on HW (62.5 GiB/rank, "Tried to allocate 936 MiB" inside
                # FSDP's _alloc_padded_unsharded_flat_param). The projectors are tiny
                # (tens of MB) and don't need FSDP sharding at all; ignoring them means
                # they're always fully replicated per rank, so those call sites' `summon`
                # becomes unnecessary for THIS purpose (it's still needed elsewhere for
                # backbone params, which callers gate separately).
                _ignore_candidates += [
                    _pre_wrap.protein_projection,
                    _pre_wrap.go_projection,
                ]
            _ignored = [
                m
                for m in _ignore_candidates
                if m is not None and isinstance(m, torch.nn.Module)
            ]
            _replicate_trainables = bool(
                _force_full_shard
                and self._enable_lora
                and os.environ.get("TORCHTUNE_FSDP_REPLICATE_TRAINABLES", "0") == "1"
            )
            _ignored_states = None
            if _replicate_trainables:
                _ignored_states = list(
                    {
                        *(
                            param
                            for module in _ignored
                            for param in module.parameters()
                        ),
                        *(
                            param
                            for param in _pre_wrap.parameters()
                            if param.requires_grad
                        ),
                    }
                )
                for param in _ignored_states:
                    if param.requires_grad and param.device != self._device:
                        param.data = param.data.to(self._device)
                # Says BUILT, not ACTIVE. The previous wording ("ignored_states ACTIVE
                # for N trainable parameters") was emitted here, at list-construction
                # time, and read as proof that FSDP had received the list — which is
                # exactly how a correct diagnosis got discarded on 2026-09-15 (the
                # non-HSDP branch built the list and then passed `ignored_modules`
                # instead). Whether it is actually honored is logged after the wrap.
                log.info(
                    "Rank %d: FSDP parameter-level ignored_states BUILT for %d "
                    "trainable parameters on %s (not yet passed to FSDP — see the "
                    "post-wrap ignored_states check)",
                    self.rank,
                    sum(param.requires_grad for param in _ignored_states),
                    self._device,
                )
            # Recorded so per-step call sites (generate_trajectory's prompt_embeds build,
            # LoRA delta sync) can skip an otherwise-unnecessary summon_full_params(self._model)
            # when the modules they actually need are already fully-replicated (ignored),
            # not FSDP-sharded — see the 32B-scale note above for why summon becomes
            # catastrophically expensive (not just "free but pointless") once the backbone
            # is split into many per-layer FSDP units.
            self._projectors_fsdp_ignored = bool(_force_full_shard)
            self._ignored_trainable_params = (
                [
                    param
                    for param in _pre_wrap.parameters()
                    if param.requires_grad
                    and (
                        _replicate_trainables
                        or any(
                            param is projector_param
                            for module in (
                                _pre_wrap.protein_projection,
                                _pre_wrap.go_projection,
                            )
                            for projector_param in module.parameters()
                        )
                    )
                ]
                if _force_full_shard
                else []
            )
            # 32B-scale override: SHARD_GRAD_OP/_HYBRID_SHARD_ZERO2 keep the FULL frozen
            # backbone resident on every rank (only grad/optimizer state is sharded) — fine
            # at 4B (~8 GiB bf16) but the 32B backbone alone is ~65.6 GiB bf16, exceeding a
            # single 64 GiB tile before any activations/KV/LoRA overhead (same failure class
            # as vLLM TP=1 SEGFAULTing on the 32B eval path, launch_vllm_http_32b_tp2.sh).
            # fsdp_full_shard_at_rest=true forces true params-at-rest sharding (ZeRO-3
            # equivalent) so each tile only holds its 1/dp_shard slice of the backbone.
            # (_force_full_shard is computed once, earlier, near the policy model's
            # construction — reused here rather than redefined.)
            if _is_hsdp:
                if _force_full_shard:
                    _shard_strategy = ShardingStrategy.HYBRID_SHARD
                else:
                    try:
                        _shard_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
                    except AttributeError:
                        _shard_strategy = ShardingStrategy.HYBRID_SHARD
                # Route the inter-node grad all-reduce over gloo (XCCL cross-node leaks
                # CXI MR handles -> banned:1 ~step10); base helper, validated on AGPT-2B.
                try:
                    from torchtune.dev.rl.distributed import (
                        enable_fsdp1_hsdp_inter_node_gloo,
                    )

                    enable_fsdp1_hsdp_inter_node_gloo()
                except Exception as _e:
                    log.warning("enable_fsdp1_hsdp_inter_node_gloo unavailable: %s", _e)
                _fsdp_ignore_kwargs = (
                    {"ignored_states": _ignored_states}
                    if _ignored_states is not None
                    else {"ignored_modules": _ignored}
                )
                self._model = FSDP(
                    _pre_wrap,
                    sharding_strategy=_shard_strategy,
                    mixed_precision=_mp_policy,
                    device_mesh=self._dp_mesh,
                    **_fsdp_ignore_kwargs,
                    use_orig_params=True,
                    device_id=self._device,
                    limit_all_gathers=True,
                    auto_wrap_policy=_auto_wrap_policy,
                    backward_prefetch=(
                        BackwardPrefetch.BACKWARD_POST
                        if _force_full_shard
                        else BackwardPrefetch.BACKWARD_PRE
                    ),
                )
                if (
                    _force_full_shard
                    and os.environ.get("TORCHTUNE_FSDP_POSTDIVIDE_ONLY", "0") == "1"
                ):
                    _adjusted_states = 0
                    _original_factors = set()
                    for _fsdp_state in FSDP.fsdp_modules(self._model):
                        _predivide = float(_fsdp_state._gradient_predivide_factor)
                        _postdivide = float(_fsdp_state._gradient_postdivide_factor)
                        _original_factors.add((_predivide, _postdivide))
                        _fsdp_state._gradient_predivide_factor = 1.0
                        _fsdp_state._gradient_postdivide_factor = (
                            _predivide * _postdivide
                        )
                        _adjusted_states += 1
                    if _adjusted_states == 0:
                        raise RuntimeError(
                            "TORCHTUNE_FSDP_POSTDIVIDE_ONLY=1 found no FSDP1 states"
                        )
                    log.info(
                        "Rank %d: FSDP postdivide-only ACTIVE for %d states "
                        "(original factors=%s; avoids full-gradient div_ on XPU)",
                        self.rank,
                        _adjusted_states,
                        sorted(_original_factors),
                    )
                if (
                    _force_full_shard
                    and os.environ.get("TORCHTUNE_FSDP_CPU_POSTDIVIDE", "0") == "1"
                ):
                    if os.environ.get("TORCHTUNE_FSDP_POSTDIVIDE_ONLY", "0") == "1":
                        raise RuntimeError(
                            "TORCHTUNE_FSDP_CPU_POSTDIVIDE and "
                            "TORCHTUNE_FSDP_POSTDIVIDE_ONLY are mutually exclusive"
                        )
                    _adjusted_states = 0
                    _total_divisors = set()
                    for _fsdp_state in FSDP.fsdp_modules(self._model):
                        _predivide = float(_fsdp_state._gradient_predivide_factor)
                        _postdivide = float(_fsdp_state._gradient_postdivide_factor)
                        _total_divisors.add(_predivide * _postdivide)
                        _fsdp_state._gradient_predivide_factor = 1.0
                        _fsdp_state._gradient_postdivide_factor = 1.0
                        _adjusted_states += 1
                    if _adjusted_states == 0 or len(_total_divisors) != 1:
                        raise RuntimeError(
                            "TORCHTUNE_FSDP_CPU_POSTDIVIDE=1 requires one consistent "
                            "FSDP1 gradient divisor"
                        )
                    _cpu_postdivide = _total_divisors.pop()
                    from torchtune.dev.rl.distributed import (
                        set_fsdp1_hsdp_cpu_postdivide,
                    )

                    set_fsdp1_hsdp_cpu_postdivide(_cpu_postdivide)
                    log.info(
                        "Rank %d: FSDP CPU postdivide ACTIVE for %d states "
                        "(CPU divisor=%s; XPU pre/post factors=1)",
                        self.rank,
                        _adjusted_states,
                        _cpu_postdivide,
                    )
                log.info(
                    "Rank %d: FSDP1 HSDP (%s) over dp_mesh (replicate=%d x shard=%d)",
                    self.rank,
                    _shard_strategy.name,
                    self._dp_replicate,
                    self._dp_shard,
                )
            else:
                # colocate: FULL_SHARD (ZeRO-3) shards params at rest → frees ~11/12 of
                # the 4B footprint for the co-resident vLLM engine. server/dedicated
                # single-replica keep the validated SHARD_GRAD_OP (ZeRO-2; vLLM off-tile),
                # unless fsdp_full_shard_at_rest=true (32B — see the HSDP branch above for
                # why SHARD_GRAD_OP cannot hold the full backbone on one tile at this scale).
                _shard_strategy = (
                    ShardingStrategy.FULL_SHARD
                    if (_is_colocate or _force_full_shard)
                    else ShardingStrategy.SHARD_GRAD_OP
                )
                # FIX 2026-09-15 (jobs 8827075 / 8827354 / 8827618). This branch used a
                # hardcoded `ignored_modules=_ignored`, which excludes only the ENCODER
                # MODULES (_embed, protein_encoder, go_encoder). The HSDP branch above
                # instead passes `**_fsdp_ignore_kwargs`, which becomes
                # `ignored_states=_ignored_states` (parameter-level) whenever
                # `_replicate_trainables` is set — so at 16N the LoRA adapters were kept
                # OUT of the per-layer flat params and stayed replicated, while at
                # dp_replicate=1 they were swept INTO the wrapped decoder layers.
                #
                # Symptom: the named grad census showed each rank holding grads for
                # exactly one or three PROJECTION TYPES across all 64 layers
                # (rank1: q,k,v | rank2: o | rank5: gate | rank8: up | rank11: down;
                # 7 ranks with none), counts 0/128/384 summing to exactly 896. The param
                # names carry `_fsdp_wrapped_module` TWICE —
                # `..._fsdp_wrapped_module.backbone...layers.N._fsdp_wrapped_module.mlp.down_proj.lora_A...`
                # — proving the adapters sat inside an FSDP-wrapped decoder layer.
                #
                # The `ignored_states ACTIVE for 904` log line does NOT prove the list was
                # passed: it fires when the list is BUILT. That line is why an earlier,
                # correct hypothesis was wrongly discarded — the log was necessary but not
                # sufficient evidence.
                #
                # Reusing the same kwargs construction as the HSDP branch keeps the two
                # paths honest and restores replicated adapters at dp_replicate=1.
                _fsdp_ignore_kwargs_nonhsdp = (
                    {"ignored_states": _ignored_states}
                    if _ignored_states is not None
                    else {"ignored_modules": _ignored}
                )
                self._model = FSDP(
                    _pre_wrap,
                    sharding_strategy=_shard_strategy,
                    mixed_precision=_mp_policy,
                    process_group=self._training_pg,
                    **_fsdp_ignore_kwargs_nonhsdp,
                    use_orig_params=True,
                    device_id=self._device,
                    auto_wrap_policy=_auto_wrap_policy,
                    limit_all_gathers=True,
                    backward_prefetch=(
                        BackwardPrefetch.BACKWARD_POST
                        if _force_full_shard
                        else BackwardPrefetch.BACKWARD_PRE
                    ),
                )
            self._use_fsdp1 = True

            # POST-WRAP VERIFICATION (2026-09-15). The pre-wrap log says the ignored
            # list was BUILT; this says whether FSDP actually honored it. A LoRA param
            # that ended up inside a wrapped decoder layer has `_fsdp_wrapped_module`
            # appearing a SECOND time after its `layers.N` segment — that is precisely
            # the signature the named grad census found when the non-HSDP branch was
            # passing `ignored_modules` instead of `ignored_states` (each rank then
            # held grads for only 1-3 projection types; 7 of 12 ranks had none).
            # Cheap (one pass over names at setup) and it converts a silent,
            # 3-job-to-diagnose misconfiguration into an immediate warning.
            # Ask FSDP DIRECTLY. An earlier version of this check inferred "swept in"
            # from the qualified name containing `_fsdp_wrapped_module` after the
            # `layers.N` segment — that is a FALSE POSITIVE generator: `ignored_states`
            # excludes a param from FLATTENING, not from the module TREE, so an ignored
            # adapter still lives under a wrapped decoder layer and its qualified name
            # picks up `_fsdp_wrapped_module` either way. It cannot distinguish
            # "flattened into the layer's FlatParameter" from "ignored but nested".
            #
            # `_ignored_params` is the set FSDP actually resolved and honored
            # (torch/distributed/fsdp/_init_utils.py: `state._ignored_params = ...`,
            # then `managed_params = _get_orig_params(module, state._ignored_params)`),
            # and FSDP propagates it to auto-wrapped children via root_kwargs
            # (fully_sharded_data_parallel.py: `"ignored_states": self._ignored_params`).
            # Membership in that set is ground truth; names are not.
            if getattr(self, "_projectors_fsdp_ignored", False):
                try:
                    _ign = set()
                    for _mod in self._model.modules():
                        _ign |= {
                            id(_p) for _p in getattr(_mod, "_ignored_params", set())
                        }
                    _lora = [
                        (_n, _p)
                        for _n, _p in self._model.named_parameters()
                        if _p.requires_grad and ".lora_" in _n
                    ]
                    _missed = [_n for _n, _p in _lora if id(_p) not in _ign]
                    if _missed:
                        log.warning(
                            "FSDP ignored_states NOT honored for %d/%d trainable LoRA "
                            "params (e.g. %s): they are absent from FSDP's resolved "
                            "_ignored_params, so they are flattened and sharded. "
                            "Per-rank grads will then cover only a subset of "
                            "projections and _sync_ignored_trainable_grads will average "
                            "disjoint shards. Check this FSDP() call passes "
                            "ignored_states, not ignored_modules.",
                            len(_missed),
                            len(_lora),
                            _missed[0],
                        )
                    else:
                        log.info(
                            "FSDP ignored_states honored: all %d trainable LoRA params "
                            "are in FSDP's _ignored_params (replicated, not flattened).",
                            len(_lora),
                        )
                except Exception:  # diagnostics must never break setup
                    pass
            # Pre-compute chunked broadcast layout. Two paths:
            #   - Default (4B, validated): summon_full_params(rank0_only=True) — outside
            #     it, use_orig_params=True params reflect SHARD sizes (not full), so chunk
            #     boundaries would be wrong. summon_full_params re-gathers the WHOLE model
            #     onto rank 0 to read correct full shapes — free at 4B (~8 GiB) but at 32B
            #     this alone OOMs (confirmed on HW: 60.69 GiB already allocated by the
            #     per-layer-sharded model, +936 MiB for the gather tips it over on a 64 GiB
            #     tile). _compute_wsync_layout only reads shape/numel/dtype metadata, never
            #     tensor data — so at 32B, read it from `_pre_wrap` BEFORE FSDP wrapping
            #     instead, while the backbone is still whole on CPU (backbone_cpu_init):
            #     shapes/dtypes are identical whether read pre-wrap or via summon_full_params
            #     (merge-state and device placement don't affect vllm_param_iter's name/shape
            #     translation), so this produces a byte-identical layout without ever
            #     re-gathering the sharded model onto one GPU.
            if _force_full_shard:
                self._compute_wsync_layout(_pre_wrap)
            else:
                with FSDP.summon_full_params(
                    self._model, writeback=False, rank0_only=True
                ):
                    self._compute_wsync_layout(self._model)
            _wsync_desc = (
                f"wsync_pg=[0,{self._vllm_dedicated_rank}]"
                if self._vllm_mode == "dedicated_rank"
                else "wsync=HTTP raw_bytes (no PG)"
            )
            log.info(
                "Rank %d: FSDP1 "
                + _shard_strategy.name
                + " wrapped over training_pg (%d ranks), "
                "ignored=[_embed, protein_encoder, go_encoder], %s",
                self.rank,
                len(_training_ranks),
                _wsync_desc,
            )
            # lora_wsync_mode: "merged" is the HW-validated fallback. "delta"
            # ships an unsharded base once; "delta_tp" snapshots vLLM's resident
            # TP-local base and ships only LoRA A/B factors. Delta modes are only
            # exercised alongside fsdp_full_shard_at_rest (32B). See
            # memory/project_bioreason_32b_grpo_fsdp1_per_layer_first_working_path_20260906.md
            # for why the merged path is slow at 32B (~65s/step wsync, job
            # 8809198/v32: ~31s gather + ~33s broadcast of a ~61 GiB payload).
            self._lora_wsync_mode = cfg.get("lora_wsync_mode", "merged")
            if self._lora_wsync_mode not in ("merged", "delta", "delta_tp"):
                raise ValueError(
                    f"lora_wsync_mode must be 'merged', 'delta', or 'delta_tp', got "
                    f"{self._lora_wsync_mode!r}"
                )
            if (
                self._lora_wsync_mode == "delta"
                and _force_full_shard
                and getattr(self._model, "_has_lora", False)
            ):
                self._cache_bioreason_lora_base_per_unit()
                # LOAD-BEARING: this is a 64-unit summon_full_params sweep run
                # during setup(), well before the first _init_sender_pool/
                # _init_xccl_weight_sync call. Every other call site in this
                # file that runs a summon_full_params/PG-construction
                # collective on XPU is followed by a synchronize() to flush
                # pending device ops (see _init_sender_pool's and
                # _xccl_gather_fsdp1's own comments: "ProcessGroupXCCL
                # constructor may leave pending GPU ops that deadlock/crash
                # the subsequent collective"). This call site was missing
                # that flush — confirmed on HW (jobs 8810079/v36, v37):
                # `ur_die: urEventWait must not be called for an internal
                # event` on rank 1, at the FIRST _init_xccl_weight_sync call
                # several minutes later in the training loop, reproducing
                # identically even after the barrier fix was added directly
                # inside _publish_bioreason_lora_delta and _init_sender_pool
                # (both already correct) — the corruption predates either of
                # those functions and comes from this unflushed setup-time
                # sweep instead.
                if self._device.type == "xpu" and torch.xpu.is_available():
                    torch.xpu.synchronize()
                # Use the gloo barrier group for consistency with the fix
                # applied to _init_sender_pool (see weight_sync.py) — this
                # call site runs at setup() time, before _init_xccl_weight_sync
                # is ever called, so it was never actually implicated in the
                # ur_die race (0 crashes here across v38-v43), but there's no
                # reason to leave it on an XCCL-backed group either.
                torch.distributed.barrier(
                    group=getattr(self, "_wsync_barrier_pg", None)
                    or getattr(self, "_training_pg", None)
                )
            elif (
                self._lora_wsync_mode == "delta_tp"
                and _force_full_shard
                and getattr(self._model, "_has_lora", False)
            ):
                self._bior_lora_delta_ready = True
        else:
            self._training_pg = None
            self._wsync_pg = None

    # ── vLLM generation override ───────────────────────────────────────────────

    def _http_generate_from_embeds_cpu(
        self,
        embeds_list: list,
        batch_input_ids_cpu: torch.Tensor,
        context_length: int,
        return_logprobs: bool = False,
    ):
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
        from concurrent.futures import as_completed, ThreadPoolExecutor

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
            if num_clients < _n_rep:
                _eng_base = _replica_idx % num_clients
                _band_size = 1
            else:
                _band = num_clients // _n_rep
                _eng_base = _replica_idx * _band
                _is_last_band = _replica_idx == _n_rep - 1
                _band_size = (num_clients - _eng_base) if _is_last_band else _band
        else:
            _eng_base = 0
            _band_size = num_clients
        _n_engines = max(1, min(_want_engines, _band_size))
        _engine_ids = [(_eng_base + e) % num_clients for e in range(_n_engines)]
        _groups: list[list[int]] = [[] for _ in range(_n_engines)]
        for _i in range(bsz):
            _groups[_i % _n_engines].append(_i)

        _prompt_n = (
            self.grpo_samples
            if os.environ.get("TORCHTUNE_VLLM_PROMPT_N", "0") == "1"
            and _n_engines == 1
            and bsz % self.grpo_samples == 0
            else 1
        )

        request_plan = [
            (_engine_ids[group_idx], indices, _prompt_n)
            for group_idx, indices in enumerate(_groups)
            if indices
        ]
        if (
            os.environ.get("TORCHTUNE_VLLM_SPLIT_PROMPT_N", "0") == "1"
            and _prompt_n > 1
        ):
            request_plan = _split_prompt_n_request_plan(
                request_plan,
                num_clients,
                int(os.environ.get("TORCHTUNE_VLLM_SPLIT_ENGINE_STRIDE", "0")),
                _replica_idx
                * int(
                    os.environ.get("TORCHTUNE_VLLM_SPLIT_ENGINE_PHASE_PER_REPLICA", "0")
                ),
            )
        request_plan = _split_choice_request_plan(
            request_plan,
            num_clients,
            int(os.environ.get("TORCHTUNE_VLLM_SPLIT_CHOICE_ENGINE_STRIDE", "0")),
        )

        def _call_group(engine_id, client, idxs, request_n):
            request_idxs = idxs[::request_n]
            embeds = [embeds_list[j] for j in request_idxs]
            request_t0 = time.perf_counter()
            out = client.generate_from_embeds(
                prompt_embeds=embeds,
                n=request_n,
                return_logprobs=return_logprobs,
                **gen_kwargs,
            )
            # With return_logprobs the client returns a TUPLE. Unpacking it
            # explicitly matters: the old `out[k]` indexing would silently take the
            # token-id list as the whole result and the logprobs would vanish
            # without an error.
            if return_logprobs:
                out, out_lp = out
            else:
                out_lp = None
            # Emit the SAME "request done" record the synchronous generation path
            # emits (see the sibling _call_group in _generate_with_vllm_fanout).
            # This path is the ASYNC one: it runs on the RolloutProducer thread and
            # used to log nothing at all, so `rolog_token_totals.sh` exited 3 BLIND
            # on every async log and the async-vs-sync A/B had exactly ONE token
            # estimator (BIOREASON_DIAG) with no independent cross-check -- the
            # single-estimator setup that produced the arity burn. The grammar is
            # copied verbatim from the sync site so one parser serves both arms; do
            # not "improve" the wording without updating rolog_token_totals.sh.
            _out_lens = [len(tokens) for tokens in (out or [])]
            _out_tokens = sum(_out_lens)
            _elapsed = time.perf_counter() - request_t0
            log.info(
                "Rank %d: vLLM engine=%d request done sequences=%d "
                "output_tokens=%d output_length_min=%d output_length_max=%d "
                "elapsed=%.1fs tok/s=%.1f",
                self.rank,
                engine_id,
                len(idxs),
                _out_tokens,
                min(_out_lens, default=0),
                max(_out_lens, default=0),
                _elapsed,
                _out_tokens / max(_elapsed, 0.01),
            )
            return {
                idxs[k]: (
                    out[k] if out and k < len(out) else [],
                    out_lp[k] if out_lp and k < len(out_lp) else None,
                )
                for k in range(len(idxs))
            }

        completions = [None] * bsz
        completion_lps = [None] * bsz
        with ThreadPoolExecutor(max_workers=len(request_plan)) as pool:
            futures = [
                pool.submit(
                    _call_group,
                    engine_id,
                    self._vllm_clients[engine_id],
                    indices,
                    request_n,
                )
                for engine_id, indices, request_n in request_plan
            ]
            for future in as_completed(futures):
                for _gi, (_comp, _lp) in future.result().items():
                    completions[_gi] = _comp
                    completion_lps[_gi] = _lp
        gen_time = time.perf_counter() - t0

        # CPU assembly — NO XPU. Consumer moves to device.
        query_responses = torch.full(
            (bsz, total_len), self._tokenizer.pad_id, dtype=batch_input_ids_cpu.dtype
        )
        query_responses[:, :context_length] = batch_input_ids_cpu
        for i, comp in enumerate(completions):
            length = min(len(comp), self._max_generated_tokens)
            if length:
                query_responses[
                    i, context_length : context_length + length
                ] = torch.tensor(comp[:length], dtype=batch_input_ids_cpu.dtype)
        total_tokens = sum(len(c) for c in completions)
        log.info(
            "Rank %d: vLLM-embeds HTTP: %d seqs over %d engines (ids=%s), %d tokens in "
            "%.1fs (%.1f tok/s)",
            self.rank,
            bsz,
            _n_engines,
            _engine_ids,
            total_tokens,
            gen_time,
            total_tokens / max(gen_time, 0.01),
        )
        if not return_logprobs:
            return query_responses, None
        # WHOLE-BATCH fallback, never per-row repair. vLLM logprobs and a recomputed
        # forward differ by recompute noise (ratios up to 1.0739 on a healthy run),
        # so a batch mixing the two sources carries a per-row systematic difference
        # correlated with exactly the rows that failed -- a confound no downstream
        # statistic could detect.
        _bad = rows_needing_fallback(
            completion_lps, completions, self._max_generated_tokens
        )
        if _bad:
            log.warning(
                "Rank %d: %d/%d rows lack usable vLLM logprobs (first few: %s); "
                "falling back to the policy forward for the WHOLE batch",
                self.rank,
                len(_bad),
                bsz,
                _bad[:8],
            )
            return query_responses, None
        return query_responses, build_behavior_logprobs(
            completion_lps, completions, bsz, self._max_generated_tokens
        )

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
            embeds_list = [
                prompt_embeds[i].detach().cpu().contiguous() for i in range(bsz)
            ]
            gen_kwargs = dict(
                max_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k or 0,
                top_p=getattr(self, "_top_p", 1.0),
                stop_token_ids=getattr(self, "_stop_token_ids_list", None),
            )

            t0 = time.perf_counter()
            num_clients = len(self._vllm_clients)
            from concurrent.futures import as_completed, ThreadPoolExecutor

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
            _seqs_per_engine = int(
                os.environ.get("TORCHTUNE_VLLM_SEQS_PER_ENGINE", "4")
            )
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
            if num_clients < _n_rep:
                _eng_base = _replica_idx % num_clients
                _band_size = 1
            else:
                _band = num_clients // _n_rep
                _eng_base = _replica_idx * _band
                # This leader uses min(want, band) engines from its band (capped so two
                # replicas never share an engine; the last band absorbs the remainder).
                _is_last_band = _replica_idx == _n_rep - 1
                _band_size = (num_clients - _eng_base) if _is_last_band else _band
            _n_engines = max(1, min(_want_engines, _band_size))
            # Global engine indices this leader will hit (disjoint across replicas).
            _engine_ids = [(_eng_base + e) % num_clients for e in range(_n_engines)]
            # Contiguous groups so each engine call carries ~_seqs_per_engine embeds.
            _groups: list[list[int]] = [[] for _ in range(_n_engines)]
            for _i in range(bsz):
                _groups[_i % _n_engines].append(_i)

            _prompt_n = (
                self.grpo_samples
                if os.environ.get("TORCHTUNE_VLLM_PROMPT_N", "0") == "1"
                and _n_engines == 1
                and bsz % self.grpo_samples == 0
                else 1
            )

            request_plan = [
                (_engine_ids[group_idx], indices, _prompt_n)
                for group_idx, indices in enumerate(_groups)
                if indices
            ]
            if (
                os.environ.get("TORCHTUNE_VLLM_SPLIT_PROMPT_N", "0") == "1"
                and _prompt_n > 1
            ):
                request_plan = _split_prompt_n_request_plan(
                    request_plan,
                    num_clients,
                    int(os.environ.get("TORCHTUNE_VLLM_SPLIT_ENGINE_STRIDE", "0")),
                    _replica_idx
                    * int(
                        os.environ.get(
                            "TORCHTUNE_VLLM_SPLIT_ENGINE_PHASE_PER_REPLICA", "0"
                        )
                    ),
                    int(
                        os.environ.get("TORCHTUNE_VLLM_SPLIT_PROMPTS_PER_REQUEST", "1")
                    ),
                )
            request_plan = _split_choice_request_plan(
                request_plan,
                num_clients,
                int(os.environ.get("TORCHTUNE_VLLM_SPLIT_CHOICE_ENGINE_STRIDE", "0")),
            )

            log.info(
                "Rank %d: vLLM-embeds submit start: %d sequences requests=%s "
                "max_tokens=%d",
                self.rank,
                bsz,
                [
                    (engine_id, len(indices), request_n)
                    for engine_id, indices, request_n in request_plan
                ],
                self._max_generated_tokens,
            )

            def _call_group(engine_id, client, idxs, request_n):
                request_idxs = idxs[::request_n]
                embeds = [embeds_list[j] for j in request_idxs]
                request_t0 = time.perf_counter()
                log.info(
                    "Rank %d: vLLM engine=%d request start prompts=%d n=%d sequences=%d",
                    self.rank,
                    engine_id,
                    len(request_idxs),
                    request_n,
                    len(request_idxs) * request_n,
                )
                out = client.generate_from_embeds(
                    prompt_embeds=embeds,
                    n=request_n,
                    **gen_kwargs,
                )
                output_lengths = [len(tokens) for tokens in (out or [])]
                output_tokens = sum(output_lengths)
                request_elapsed = time.perf_counter() - request_t0
                log.info(
                    "Rank %d: vLLM engine=%d request done sequences=%d "
                    "output_tokens=%d output_length_min=%d output_length_max=%d "
                    "elapsed=%.1fs tok/s=%.1f",
                    self.rank,
                    engine_id,
                    len(idxs),
                    output_tokens,
                    min(output_lengths, default=0),
                    max(output_lengths, default=0),
                    request_elapsed,
                    output_tokens / max(request_elapsed, 0.01),
                )
                # out is list aligned with embeds; map back to global indices.
                return {
                    idxs[k]: (out[k] if out and k < len(out) else [])
                    for k in range(len(idxs))
                }

            completions = [None] * bsz
            with ThreadPoolExecutor(max_workers=len(request_plan)) as pool:
                futures = [
                    pool.submit(
                        _call_group,
                        engine_id,
                        self._vllm_clients[engine_id],
                        indices,
                        request_n,
                    )
                    for engine_id, indices, request_n in request_plan
                ]
                for future in as_completed(futures):
                    for _gi, _comp in future.result().items():
                        completions[_gi] = _comp
            gen_time = time.perf_counter() - t0
            if self._is_rank_zero:
                log.info(
                    "Rank 0: gen fan-out: bsz=%d over %d engines (ids=%s, %d seqs/engine "
                    "target, replica=%d/%d) -> batched decode",
                    bsz,
                    _n_engines,
                    _engine_ids,
                    _seqs_per_engine,
                    _replica_idx,
                    _n_rep,
                )

            query_responses = batch_input_ids.new_full(
                (bsz, total_len), self._tokenizer.pad_id
            )
            query_responses[:, :context_length] = batch_input_ids
            for i, comp in enumerate(completions):
                length = min(len(comp), self._max_generated_tokens)
                query_responses[
                    i, context_length : context_length + length
                ] = torch.tensor(
                    comp[:length], dtype=batch_input_ids.dtype, device=self._device
                )

            total_tokens = sum(len(c) for c in completions)
            log.info(
                "Rank %d: vLLM-embeds generation: %d sequences (%d clients), %d tokens in "
                "%.1fs (%.1f tok/s)",
                self.rank,
                bsz,
                num_clients,
                total_tokens,
                gen_time,
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
        if protein_sequences is None or not hasattr(
            self._policy, "build_prompt_embeds"
        ):
            return None
        input_ids = batch["tokens"].to(self._device)
        batch_size = input_ids.shape[0]
        grpo_size = self.grpo_samples
        import contextlib

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # See generate_trajectory's identical exception for why 32B skips this gather
        # entirely when the projectors are already FSDP-ignored (fully replicated).
        if getattr(self, "_projectors_fsdp_ignored", False):
            _gather_ctx = contextlib.nullcontext()
        elif isinstance(self._model, FSDP):
            _gather_ctx = FSDP.summon_full_params(self._model, writeback=False)
        else:
            _gather_ctx = contextlib.nullcontext()
        with torch.no_grad(), _gather_ctx:
            pe_base = self._policy.build_prompt_embeds(
                input_ids, protein_sequences
            )  # [B,P,H] CPU
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
            #
            # The behavior logprobs ride back in the TELEMETRY dict, which
            # RolloutProducer merges into item.batch_meta. They must not be written
            # onto `self` here: this runs on the producer thread, one step ahead of
            # the consumer, so a `self` attribute would be overwritten by the NEXT
            # batch before the current one is consumed.
            qr_cpu, blp_cpu = self._http_generate_from_embeds_cpu(
                work["embeds_list"],
                work["bii_cpu"],
                work["ctx"],
                return_logprobs=self._use_vllm_behavior_logprobs,
            )
            return qr_cpu, {"behavior_logprobs": blp_cpu}

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
                self.rank,
                self._async_generation_max_staleness,
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
                self._pending_async_query_responses = item.batch_meta[
                    "rollout_payload"
                ].to(self._device)
                # May be None (feature off, or a row needed the fallback). The
                # follower ranks cannot know which, so the decision is broadcast
                # with the data -- see broadcast_behavior_logprobs.
                self._pending_async_behavior_logprobs = item.batch_meta.get(
                    "behavior_logprobs"
                )
                self._last_rollout_item = item
                _w_now = self._weight_versions.version
                # Stash the CONSUME-time version for the METRICS async tail,
                # which would otherwise read the counter after this step's own
                # publish and over-report the lag by one (2026-09-16).
                self._last_rollout_consume_wver = _w_now
                log.info(
                    "Rank %d: async consume (producer_latency=%.1fs, qsize=%d, "
                    "rollout_w_ver=%d, cur_w_ver=%d, lag=%d)",
                    self.rank,
                    item.produce_latency_s,
                    producer.qsize(),
                    item.weight_version,
                    _w_now,
                    max(0, _w_now - item.weight_version),
                )
            else:
                self._pending_async_query_responses = None
                self._pending_async_behavior_logprobs = None
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
                    bii_cpu = (
                        batch["tokens"][:, None, :]
                        .expand(-1, self.grpo_samples, -1)
                        .reshape(bsz, -1)
                        .cpu()
                    )
                    ctx = batch["tokens"].shape[1]
                    embeds_list = [prompt_embeds[i].contiguous() for i in range(bsz)]
                    # STALENESS PIN: snapshot the weight version on the MAIN thread
                    # at post time (deterministically ordered w.r.t. the per-step
                    # bumps) and hand it to the producer so it does NOT drift at
                    # pickup. Lag == 1 at consume. See RolloutProducer._run.
                    _http_inbox.put(
                        {
                            "embeds_list": embeds_list,
                            "bii_cpu": bii_cpu,
                            "ctx": ctx,
                            "_weight_version": self._weight_versions.version,
                        }
                    )

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

        query_responses = batch_input_ids.new_full(
            (bsz, total_len), self._tokenizer.pad_id
        )
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
            self.rank,
            bsz,
            total_tokens,
            gen_time,
            total_tokens / max(gen_time, 0.01),
        )
        return query_responses

    def _blp_audit_active(self) -> bool:
        """True while the behavior-logprobs audit should still run.

        Gating on the step counter rather than a bool is deliberate: the audit pays
        exactly the policy forward the feature exists to remove, so leaving it on for
        a whole run would hide the speedup it is meant to qualify.
        """
        return (
            getattr(self, "_blp_audit_steps", 0) > 0
            and getattr(self, "_steps_run", 0) < self._blp_audit_steps
        )

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
        from torchtune.dev.rl.generation import generate
        from torchtune.modules import local_kv_cache

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if not _colocate_vllm_mode:
            device_empty_cache(self._device)
        elif (
            self._vllm_mode == "colocate_sleep"
            and self._vllm_llm is not None
            and hasattr(self, "_vllm_is_sleeping")
            and self._vllm_is_sleeping
        ):
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
            log.info(
                "Rank %d: vLLM wake_up + weight sync completed in %.2fs",
                self.rank,
                time.perf_counter() - t_wake,
            )
        elif self._vllm_mode == "colocate" and self._vllm_llm is not None:
            # Plain colocate (vLLM resident, no sleep): the WEIGHT sync runs in the
            # base train() loop (_run_wsync_block → _sync_colocated_weights, which
            # BioReason overrides to the LoRA merge). Here we only restore KV cache
            # shapes (if a prior step zeroed them) + reset prefix cache before gen.
            import gc

            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            if hasattr(self, "_vllm_kv_cache_shapes"):
                kv_caches = (
                    self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
                )
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
        if (
            _async_consume
            and getattr(self, "_pending_async_prompt_embeds", None) is not None
        ):
            prompt_embeds = self._pending_async_prompt_embeds
            self._pending_async_prompt_embeds = None
        elif protein_sequences is not None and hasattr(
            self._policy, "build_prompt_embeds"
        ):
            import contextlib

            # Multimodal: build prompt embeddings once per unique prompt, then expand
            # to B*G. build_prompt_embeds(input_ids [B,P], protein_sequences [B]) ->
            # [B,P,H] on CPU. protein_projection and go_projection are trainable ->
            # FSDP-sharded at rest; summon_full_params gathers them so the projector
            # forward sees complete weights. (SYNC path — byte-identical to the
            # validated baseline when async is disabled.)
            # 32B-scale exception: when the projectors are already FSDP-ignored (fully
            # replicated per rank, not sharded — see the FSDP-wrap block), no gather is
            # needed to reach them, and summon_full_params(self._model) would instead
            # cascade into unsharding all 64 per-layer FSDP units at once (OOM, see the
            # same block's comment). Skip it entirely in that case.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if getattr(self, "_projectors_fsdp_ignored", False):
                _gather_ctx = contextlib.nullcontext()
            elif isinstance(self._model, FSDP):
                _gather_ctx = FSDP.summon_full_params(self._model, writeback=False)
            else:
                _gather_ctx = contextlib.nullcontext()
            _embed_t0 = time.perf_counter()
            if getattr(self, "_is_shard_leader", self._is_rank_zero):
                log.info(
                    "Rank %d: prompt-embeds build start batch=%d grpo=%d "
                    "projectors_ignored=%s",
                    self.rank,
                    batch_size,
                    grpo_size,
                    getattr(self, "_projectors_fsdp_ignored", False),
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
            if getattr(self, "_is_shard_leader", self._is_rank_zero):
                log.info(
                    "Rank %d: prompt-embeds build done base_shape=%s expanded_shape=%s "
                    "elapsed=%.1fs",
                    self.rank,
                    tuple(pe_base.shape),
                    tuple(prompt_embeds.shape),
                    time.perf_counter() - _embed_t0,
                )

        # step 1: generate responses
        # Declared ABOVE the vllm_mode dispatch, not inside the "server" branch:
        # the site-4 gate below reads this name unconditionally, so a declaration
        # scoped to one mode would raise UnboundLocalError under colocate /
        # dedicated_rank. Only the async-consume path ever sets it to a tensor.
        _behavior_logprobs = None
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
                    assert (
                        query_responses is not None
                    ), "async consume active but no shard-leader query_responses stashed"
                    assert query_responses.shape == (bsz, total_len), (
                        f"async qr shape mismatch: got {tuple(query_responses.shape)}, "
                        f"expected ({bsz}, {total_len})"
                    )
                else:
                    query_responses = batch_input_ids.new_empty(bsz, total_len)
                self._pending_async_query_responses = None
                query_responses = self._broadcast_query_responses(query_responses)
                # SAME PROVENANCE, SAME BROADCAST. Only the leader issued the HTTP,
                # so only the leader can hold pi_old. Skipping this broadcast is not
                # a crash: followers would take the policy-forward branch while the
                # leader skipped it -- divergent collective participation, i.e. a
                # hang. broadcast_fn is _broadcast_query_responses ITSELF so both of
                # its branches (node-local gloo under HSDP, world/_training_pg
                # otherwise) are mirrored exactly rather than re-derived.
                if self._use_vllm_behavior_logprobs:
                    _blp = self._pending_async_behavior_logprobs
                    self._pending_async_behavior_logprobs = None
                    _behavior_logprobs = broadcast_behavior_logprobs(
                        _blp.to(self._device) if _blp is not None else None,
                        num_seqs=bsz,
                        max_generated_tokens=self._max_generated_tokens,
                        is_leader=_is_leader,
                        broadcast_fn=self._broadcast_query_responses,
                        device=self._device,
                    )
            elif getattr(self, "_is_bioreason", False) and prompt_embeds is not None:
                query_responses = self._generate_with_vllm_server_embeds(
                    batch_input_ids, context_length, prompt_embeds
                )
            else:
                query_responses = self._generate_with_vllm(
                    batch_input_ids, context_length
                )
        else:
            _stop_tokens = (
                None if self._dp_replicate > 1 else self._tokenizer.stop_tokens
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

        if (
            self._vllm_mode not in ("server", "dedicated_rank")
            and not self._production_mode
        ):
            torch.distributed.barrier()

        if getattr(self, "_dp_replicate", 1) > 1:
            _trim_pg = self._gloo_dp_shard_pg
        elif self._vllm_mode == "dedicated_rank":
            _trim_pg = self._training_pg
        else:
            _trim_pg = None
        _untrimmed_response_length = query_responses.shape[1] - context_length
        query_responses, _active_response_length = trim_query_responses_to_global_max(
            query_responses,
            context_length,
            self._tokenizer.pad_id,
            process_group=_trim_pg,
        )
        if self._is_rank_zero and _active_response_length < _untrimmed_response_length:
            log.info(
                "BIOREASON_TRIM response_tokens=%d->%d sequence_tokens=%d",
                _untrimmed_response_length,
                _active_response_length,
                query_responses.shape[1],
            )

        # Free vLLM GPU memory for training forward/backward passes.
        if _colocate_vllm_mode and self._vllm_llm is not None:
            if torch.xpu.is_available():
                mem_before = torch.xpu.memory_allocated(self._device) / 1024**3
            if self._vllm_mode == "colocate_sleep":
                log.info(
                    "Rank %d: sleeping vLLM (weights + KV cache) for training",
                    self.rank,
                )
                t_free = time.perf_counter()
                self._vllm_llm.sleep(level=1)
                self._vllm_is_sleeping = True
            else:
                log.info("Rank %d: freeing vLLM KV cache for training", self.rank)
                t_free = time.perf_counter()
                kv_caches = (
                    self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
                )
                self._vllm_kv_cache_shapes = []
                for i, cache in enumerate(kv_caches):
                    self._vllm_kv_cache_shapes.append((cache.shape, cache.dtype))
                    kv_caches[i] = torch.empty(0, device="cpu")
            if torch.xpu.is_available():
                mem_after = torch.xpu.memory_allocated(self._device) / 1024**3
                log.info(
                    "Rank %d: vLLM memory freed in %.1fs (%.2f -> %.2f GiB, freed %.2f GiB)",
                    self.rank,
                    time.perf_counter() - t_free,
                    mem_before,
                    mem_after,
                    mem_before - mem_after,
                )
            else:
                log.info(
                    "Rank %d: vLLM memory freed in %.1fs",
                    self.rank,
                    time.perf_counter() - t_free,
                )

        responses = query_responses[:, context_length:].clone()

        vocab_size = getattr(self, "_vocab_size", None)
        if vocab_size is not None and vocab_size > 0:
            oob_mask = responses >= vocab_size
            if oob_mask.any():
                log.warning(
                    "Clamping %d OOB token IDs (max=%d, vocab=%d)",
                    oob_mask.sum().item(),
                    responses.max().item(),
                    vocab_size,
                )
                responses = responses.clamp(max=vocab_size - 1)
                query_responses = torch.cat(
                    [query_responses[:, :context_length], responses], dim=1
                )

        query_response_padding_masks = query_responses != self._tokenizer.pad_id
        masks = torchtune_generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = torchtune_generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        num_seqs = query_responses.shape[0]
        ref_fwd_bs = self._ref_forward_batch_size
        _response_padding_masks_for_width = None
        if self._trim_chunk_width and ref_fwd_bs < num_seqs:
            (
                _response_padding_masks_for_width,
                _,
            ) = rlhf.truncate_sequence_at_first_stop_token(
                responses.clone(),
                self._stop_token_ids,
                self._tokenizer.pad_id,
            )

        # step 2: rollout-time policy logprobs (only when needed for IS ratios)
        #
        # ppo_epochs > 1 deliberately still recomputes: later epochs need pi_old under
        # the CURRENT weights, not the behavior policy that produced the rollout.
        # BioReason runs ppo_epochs=1 so this does not bite today, but the gate must
        # not silently change semantics if it ever does.
        if (
            _behavior_logprobs is not None
            and self._ppo_epochs == 1
            and not self._blp_audit_active()
        ):
            # WIDTH, not just values. build_behavior_logprobs pads to the CAP
            # (max_generated_tokens); `responses` is the batch's ACTUAL longest
            # generation. They coincide only when some row hit the limit, so a
            # mismatch appears at the first step where none did -- job 8833972
            # survived steps 0-1 at width 3072 and died at step 2's 1743.
            logprobs = fit_behavior_logprobs_width(
                _behavior_logprobs, responses.shape[1]
            )
            # Both sibling branches set _policy_fwd_time and the GENTIMING line at
            # the end of this method reads it unconditionally. Omitting it here
            # would UnboundLocalError on exactly the path this feature enables.
            _policy_fwd_time = 0.0
            log.info(
                "Rank %d: pi_old from vLLM sampler (%s) — policy forward skipped",
                self.rank,
                list(logprobs.shape),
            )
        elif self._ppo_epochs > 1 or self._compute_rollout_logprobs_required:
            _policy_fwd_t0 = time.perf_counter()
            with torch.no_grad():
                if ref_fwd_bs >= num_seqs:
                    log.info(
                        "Rank %d: policy forward start (shape=%s)",
                        self.rank,
                        list(query_responses.shape),
                    )
                    _forward_context_length = context_length
                    if prompt_embeds is not None:
                        _forward_prompt_embeds = prompt_embeds
                        _attn_mask = (query_responses != self._tokenizer.pad_id).long()
                        _forward_position_ids = position_ids
                        if self._compact_prompt_chunks:
                            (
                                _forward_prompt_embeds,
                                _attn_mask,
                                _forward_position_ids,
                                _forward_context_length,
                            ) = compact_prompt_completion_batch(
                                prompt_embeds,
                                query_responses[:, :context_length],
                                responses,
                                self._tokenizer.pad_id,
                            )
                        _full_emb = self._policy.build_full_embeds(
                            _forward_prompt_embeds,
                            responses,
                            _forward_context_length
                            if isinstance(_forward_context_length, torch.Tensor)
                            else None,
                            _attn_mask.shape[1]
                            if isinstance(_forward_context_length, torch.Tensor)
                            else None,
                        )
                        _ro_kwargs = response_only_logits_kwargs(
                            self._model,
                            _forward_context_length,
                            responses.shape[1],
                        )
                        logits = self._model(
                            inputs_embeds=_full_emb,
                            attention_mask=_attn_mask,
                            position_ids=_forward_position_ids,
                            **_ro_kwargs,
                        )
                        del _full_emb, _attn_mask
                    else:
                        _ro_kwargs = {}
                        logits = self._model(
                            query_responses, input_pos=position_ids, mask=masks
                        )
                    log.info("Rank %d: policy forward done", self.rank)
                    logits = finish_response_logits(
                        logits,
                        _ro_kwargs,
                        _forward_context_length,
                        responses.shape[1],
                    )
                    logprobs = rlhf.batched_logits_to_logprobs(
                        logits, responses, self._temperature
                    )
                    del logits
                else:
                    log.info(
                        "Rank %d: policy forward start CHUNKED (total=%d, chunk=%d)",
                        self.rank,
                        num_seqs,
                        ref_fwd_bs,
                    )
                    _chunk_ranges = [
                        (cs, min(cs + ref_fwd_bs, num_seqs))
                        for cs in range(0, num_seqs, ref_fwd_bs)
                    ]
                    if self._trim_chunk_width:
                        _chunk_ranges = get_descending_response_chunk_ranges(
                            _response_padding_masks_for_width, ref_fwd_bs
                        )
                    logprobs_chunks = [None] * len(_chunk_ranges)
                    for cs, ce in _chunk_ranges:
                        _chunk_response_length = (
                            get_right_padded_response_length(
                                _response_padding_masks_for_width[cs:ce]
                            )
                            if self._trim_chunk_width
                            else responses.shape[1]
                        )
                        _chunk_total_length = context_length + _chunk_response_length
                        if self._trim_chunk_width and self._is_rank_zero:
                            log.info(
                                "BIOREASON_CHUNK_WIDTH phase=rollout_policy chunk=%d:%d "
                                "response=%d/%d total=%d",
                                cs,
                                ce,
                                _chunk_response_length,
                                responses.shape[1],
                                _chunk_total_length,
                            )
                        _chunk_query_responses = query_responses[
                            cs:ce, :_chunk_total_length
                        ]
                        _chunk_responses = responses[cs:ce, :_chunk_response_length]
                        _forward_context_length = context_length
                        if prompt_embeds is not None:
                            _forward_prompt_embeds = prompt_embeds[cs:ce]
                            _attn_mask = (
                                _chunk_query_responses != self._tokenizer.pad_id
                            ).long()
                            _forward_position_ids = position_ids[
                                cs:ce, :_chunk_total_length
                            ]
                            if self._compact_prompt_chunks:
                                (
                                    _forward_prompt_embeds,
                                    _attn_mask,
                                    _forward_position_ids,
                                    _forward_context_length,
                                ) = compact_prompt_completion_batch(
                                    _forward_prompt_embeds,
                                    _chunk_query_responses[:, :context_length],
                                    _chunk_responses,
                                    self._tokenizer.pad_id,
                                )
                            _full_emb = self._policy.build_full_embeds(
                                _forward_prompt_embeds,
                                _chunk_responses,
                                _forward_context_length
                                if isinstance(_forward_context_length, torch.Tensor)
                                else None,
                                _attn_mask.shape[1]
                                if isinstance(_forward_context_length, torch.Tensor)
                                else None,
                            )
                            _ro_kwargs = response_only_logits_kwargs(
                                self._model,
                                _forward_context_length,
                                _chunk_responses.shape[1],
                            )
                            chunk_logits = self._model(
                                inputs_embeds=_full_emb,
                                attention_mask=_attn_mask,
                                position_ids=_forward_position_ids,
                                **_ro_kwargs,
                            )
                            del _full_emb, _attn_mask
                        else:
                            _ro_kwargs = {}
                            chunk_logits = self._model(
                                _chunk_query_responses,
                                input_pos=position_ids[cs:ce, :_chunk_total_length],
                                mask=masks[
                                    cs:ce,
                                    :_chunk_total_length,
                                    :_chunk_total_length,
                                ],
                            )
                        chunk_logits = finish_response_logits(
                            chunk_logits,
                            _ro_kwargs,
                            _forward_context_length,
                            _chunk_responses.shape[1],
                        )
                        logprobs_chunks[cs // ref_fwd_bs] = pad_response_logprobs(
                            rlhf.batched_logits_to_logprobs(
                                chunk_logits,
                                _chunk_responses,
                                self._temperature,
                            ),
                            responses.shape[1],
                        )
                        del (
                            chunk_logits,
                            _chunk_query_responses,
                            _chunk_responses,
                        )
                    logprobs = torch.cat(logprobs_chunks, dim=0)
                    del logprobs_chunks
                    log.info("Rank %d: policy forward done (chunked)", self.rank)
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _policy_fwd_time = time.perf_counter() - _policy_fwd_t0
        else:
            logprobs = None
            _policy_fwd_time = 0.0

        # step 2.05: behavior-logprobs audit (diagnostic; off unless TORCHTUNE_BLP_AUDIT>0)
        #
        # Deliberately reports and does NOT act: pi_old stays the trainer's recompute
        # for every audited step. A threshold that silently switched sources would make
        # the audited steps differ from the unaudited ones in a way no log records.
        # Wrapped whole because a diagnostic must never be able to kill a 36h arm.
        if (
            self._blp_audit_active()
            and _behavior_logprobs is not None
            and logprobs is not None
        ):
            try:
                # Derive the exclusion mask from the response tokens directly rather
                # than reusing _response_padding_masks_for_width: that one is only
                # populated under (trim_chunk_width and ref_fwd_bs < num_seqs), so a
                # reference to it would silently pass None on the common path and let
                # PAD_FILL sentinels into the statistics. True = exclude.
                _blp_pad = None
                if responses.shape == logprobs.shape:
                    _blp_pad = responses == self._tokenizer.pad_id
                # Same cap-vs-actual width mismatch the substitution branch fixes. Here
                # it would not crash the run (the try/except below swallows it) -- it
                # would silently produce "BLP_AUDIT failed: shape mismatch" on exactly
                # the steps where no row hit the cap, i.e. an instrument that goes blind
                # without saying so. Fit first so the audit reads on every step.
                _blp_stats = audit_behavior_logprobs(
                    fit_behavior_logprobs_width(
                        _behavior_logprobs.to(logprobs.device), logprobs.shape[1]
                    ),
                    logprobs,
                    padding_mask=_blp_pad,
                )
                log.info(
                    "Rank %d: BLP_AUDIT step=%d n=%d mean_abs=%.6f p99_abs=%.6f "
                    "max_abs=%.6f ratio_p99=%.4f ratio_max=%.4f bias=%+.6f",
                    self.rank,
                    getattr(self, "_steps_run", -1),
                    _blp_stats["n_compared"],
                    _blp_stats["mean_abs"],
                    _blp_stats["p99_abs"],
                    _blp_stats["max_abs"],
                    _blp_stats["ratio_p99"],
                    _blp_stats["ratio_max"],
                    _blp_stats["bias"],
                )
            except Exception as _blp_exc:  # noqa: BLE001 - diagnostic must not kill a run
                log.warning("Rank %d: BLP_AUDIT failed: %s", self.rank, _blp_exc)

        # step 2.1: ref model logprobs
        _ref_fwd_t0 = time.perf_counter()
        log.info("Rank %d: pre-ref forward", self.rank)
        if not self._production_mode:
            self._training_barrier()

        # Dynamic ref offload: move ref model to XPU for fast ref forward.
        if getattr(self, "_bioreason_dynamic_ref_offload", False):
            self._ref_model.to(self._device)
            log.info("Rank %d: ref model → XPU for ref forward", self.rank)

        _ref_dev = next(self._ref_model.parameters()).device
        log.info(
            "Rank %d: ref model device=%s, position_ids.device=%s",
            self.rank,
            _ref_dev,
            position_ids.device,
        )
        # ── EXACT shared-prefix ref forward (opt-in, TORCHTUNE_REF_PREFIX_SHARE=1) ──
        # The ref model is frozen and this whole block is no-grad, so reusing one
        # prompt's KV cache across its G continuations is exact rather than an
        # approximation. Runs each distinct prompt's ~4096-token prefix ONCE instead
        # of grpo_samples times. Falls through to the unmodified full-recompute path
        # whenever the preconditions don't hold. See torchtune/dev/rl/ref_prefix_share.py.
        _prefix_share_done = False
        if ref_prefix_share_enabled() and prompt_embeds is not None:
            _ps_ok, _ps_reason = prefix_share_supported(
                prompt_embeds=prompt_embeds,
                num_seqs=num_seqs,
                group_size=self.grpo_samples,
                # This path uses the uniform full prompt width; row-compaction is a
                # width optimisation the cached path does not (yet) reproduce, so it
                # is deliberately not combined with it.
                compacted_prompt_lengths=None,
            )
            if not _ps_ok:
                if self._is_rank_zero:
                    log.info(
                        "REF_PREFIX_SHARE requested-but-skipped: %s", _ps_reason
                    )
            elif not hasattr(self._ref_model, "forward_cached"):
                if self._is_rank_zero:
                    log.info(
                        "REF_PREFIX_SHARE requested-but-skipped: ref model %s has no "
                        "forward_cached()",
                        type(self._ref_model).__name__,
                    )
            else:
                _ps_mask = (
                    (query_responses != self._tokenizer.pad_id).long().to(_ref_dev)
                )
                ref_logprobs = shared_prefix_ref_logprobs(
                    self._ref_model,
                    prompt_embeds,
                    responses,
                    group_size=self.grpo_samples,
                    temperature=self._temperature,
                    prompt_length=context_length,
                    attention_mask=_ps_mask,
                    position_ids=position_ids.to(_ref_dev),
                    logprob_fn=rlhf.batched_logits_to_logprobs,
                    device=self._device,
                )
                del _ps_mask
                _prefix_share_done = True
                if self._is_rank_zero:
                    log.info(
                        "REF_PREFIX_SHARE engaged: %d groups x G=%d, prefix=%d "
                        "computed once per group (was %d times)",
                        num_seqs // self.grpo_samples,
                        self.grpo_samples,
                        context_length,
                        self.grpo_samples,
                    )
        if _prefix_share_done:
            pass
        elif ref_fwd_bs >= num_seqs:
            log.info("Rank %d: ref forward start", self.rank)
            _forward_context_length = context_length
            if prompt_embeds is not None:
                _forward_prompt_embeds = prompt_embeds
                _attn_mask = (
                    (query_responses != self._tokenizer.pad_id).long().to(_ref_dev)
                )
                _forward_position_ids = position_ids.to(_ref_dev)
                if self._compact_prompt_chunks:
                    (
                        _forward_prompt_embeds,
                        _attn_mask,
                        _forward_position_ids,
                        _forward_context_length,
                    ) = compact_prompt_completion_batch(
                        prompt_embeds,
                        query_responses[:, :context_length],
                        responses,
                        self._tokenizer.pad_id,
                    )
                _full_emb = self._ref_model.build_full_embeds(
                    _forward_prompt_embeds,
                    responses,
                    _forward_context_length
                    if isinstance(_forward_context_length, torch.Tensor)
                    else None,
                    _attn_mask.shape[1]
                    if isinstance(_forward_context_length, torch.Tensor)
                    else None,
                )
                _ro_kwargs = response_only_logits_kwargs(
                    self._ref_model, _forward_context_length, responses.shape[1]
                )
                ref_logits = self._ref_model(
                    inputs_embeds=_full_emb,
                    attention_mask=_attn_mask,
                    position_ids=_forward_position_ids,
                    **_ro_kwargs,
                ).to(self._device)
                del _full_emb, _attn_mask
            else:
                _ro_kwargs = {}
                ref_logits = self._ref_model(
                    query_responses, input_pos=position_ids, mask=masks
                )
            ref_logits = finish_response_logits(
                ref_logits, _ro_kwargs, _forward_context_length, responses.shape[1]
            )
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, responses, self._temperature
            )
            del ref_logits
        else:
            log.info(
                "Rank %d: ref forward start CHUNKED (total=%d, chunk=%d)",
                self.rank,
                num_seqs,
                ref_fwd_bs,
            )
            _chunk_ranges = [
                (cs, min(cs + ref_fwd_bs, num_seqs))
                for cs in range(0, num_seqs, ref_fwd_bs)
            ]
            if self._trim_chunk_width:
                _chunk_ranges = get_descending_response_chunk_ranges(
                    _response_padding_masks_for_width, ref_fwd_bs
                )
            ref_logprobs_chunks = [None] * len(_chunk_ranges)
            for cs, ce in _chunk_ranges:
                _chunk_response_length = (
                    get_right_padded_response_length(
                        _response_padding_masks_for_width[cs:ce]
                    )
                    if self._trim_chunk_width
                    else responses.shape[1]
                )
                _chunk_total_length = context_length + _chunk_response_length
                if self._trim_chunk_width and self._is_rank_zero:
                    log.info(
                        "BIOREASON_CHUNK_WIDTH phase=reference chunk=%d:%d "
                        "response=%d/%d total=%d",
                        cs,
                        ce,
                        _chunk_response_length,
                        responses.shape[1],
                        _chunk_total_length,
                    )
                _chunk_query_responses = query_responses[cs:ce, :_chunk_total_length]
                _chunk_responses = responses[cs:ce, :_chunk_response_length]
                _forward_context_length = context_length
                if prompt_embeds is not None:
                    _forward_prompt_embeds = prompt_embeds[cs:ce]
                    _attn_mask = (
                        (_chunk_query_responses != self._tokenizer.pad_id)
                        .long()
                        .to(_ref_dev)
                    )
                    _forward_position_ids = position_ids[
                        cs:ce, :_chunk_total_length
                    ].to(_ref_dev)
                    if self._compact_prompt_chunks:
                        (
                            _forward_prompt_embeds,
                            _attn_mask,
                            _forward_position_ids,
                            _forward_context_length,
                        ) = compact_prompt_completion_batch(
                            _forward_prompt_embeds,
                            _chunk_query_responses[:, :context_length],
                            _chunk_responses,
                            self._tokenizer.pad_id,
                        )
                        if self._is_rank_zero and isinstance(
                            _forward_context_length, torch.Tensor
                        ):
                            log.info(
                                "BIOREASON_ROW_COMPACT phase=reference "
                                "prompt_lengths=%s packed_width=%d",
                                _forward_context_length.tolist(),
                                _attn_mask.shape[1],
                            )
                    _full_emb = self._ref_model.build_full_embeds(
                        _forward_prompt_embeds,
                        _chunk_responses,
                        _forward_context_length
                        if isinstance(_forward_context_length, torch.Tensor)
                        else None,
                        _attn_mask.shape[1]
                        if isinstance(_forward_context_length, torch.Tensor)
                        else None,
                    )
                    _ro_kwargs = response_only_logits_kwargs(
                        self._ref_model,
                        _forward_context_length,
                        _chunk_responses.shape[1],
                    )
                    chunk_ref_logits = self._ref_model(
                        inputs_embeds=_full_emb,
                        attention_mask=_attn_mask,
                        position_ids=_forward_position_ids,
                        **_ro_kwargs,
                    ).to(self._device)
                    del _full_emb, _attn_mask
                else:
                    _ro_kwargs = {}
                    chunk_ref_logits = self._ref_model(
                        _chunk_query_responses,
                        input_pos=position_ids[cs:ce, :_chunk_total_length],
                        mask=masks[
                            cs:ce,
                            :_chunk_total_length,
                            :_chunk_total_length,
                        ],
                    )
                chunk_ref_logits = finish_response_logits(
                    chunk_ref_logits,
                    _ro_kwargs,
                    _forward_context_length,
                    _chunk_responses.shape[1],
                )
                ref_logprobs_chunks[cs // ref_fwd_bs] = pad_response_logprobs(
                    rlhf.batched_logits_to_logprobs(
                        chunk_ref_logits,
                        _chunk_responses,
                        self._temperature,
                    ),
                    responses.shape[1],
                )
                del chunk_ref_logits, _chunk_query_responses, _chunk_responses
                # empty_cache leaks UR handles under FSDP + in-process vLLM
                # (colocate) → banned:1. Safe in server/dedicated modes.
                if not _colocate_vllm_mode:
                    device_empty_cache(self._device)
            ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
            del ref_logprobs_chunks
            log.info("Rank %d: ref forward done (chunked)", self.rank)
        del _response_padding_masks_for_width
        if not _colocate_vllm_mode:
            device_empty_cache(self._device)

        # Dynamic ref offload: move ref model back to CPU to free XPU HBM for backward.
        if getattr(self, "_bioreason_dynamic_ref_offload", False):
            self._ref_model.to("cpu")
            log.info(
                "Rank %d: ref model → CPU after ref forward (freed ~8 GiB XPU)",
                self.rank,
            )
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
            self.rank,
            _vllm_time,
            _policy_fwd_time,
            _ref_fwd_time,
        )

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
                self._tokenizer,
                responses,
                answers,
                device=self._device,
                reward_metric=self._gene_reward_metric,
            )
        elif self._reward_mode == "sum_digits":
            from torchtune.dev.rl.rewards import sum_digits_batched_rewards

            rewards, successes, metadata = sum_digits_batched_rewards(
                self._tokenizer,
                responses,
                answers,
                device=self._device,
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
                    _non_pad_ids = _non_pad.cpu().tolist()
                    _decoded.append(self._tokenizer.decode(_non_pad_ids))
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
                    _tok_present = any(
                        token_id in self._stop_token_ids_list
                        for token_id in _non_pad_ids
                    )
                    _under_cap = _rlen < int(self._max_generated_tokens)
                    _has_eos.append(_tok_present or _under_cap)
            _rw, _succ, _br_diag = _br_reward(
                _decoded,
                _expanded_answers,
                return_diagnostics=True,
                propagate_hierarchy=self._reward_propagate_hierarchy,
                obo_path=self._reward_obo_path,
            )
            rewards = _rw.view(batch_size, grpo_size, 1)
            successes = _succ.float().view(batch_size, grpo_size, 1)
            metadata = {}
            # Persist the G rollouts per prompt before _decoded goes out of scope.
            # Without this the completions are unrecoverable: only a single 200-char
            # truncated SAMPLE_RESPONSE per step survives in the log. That gap forced
            # the group-frequency F_max test to run against a PROXY (four repeated
            # evals of one checkpoint) instead of real temperature-sampled rollouts
            # -- see memory/project_bioreason_freq_ranking_beats_flat_confidence_20260915.
            # Rank 0 only (all ranks hold identical data at dp_replicate=1), OFF unless
            # TORCHTUNE_DUMP_ROLLOUTS=1, and the helper swallows every exception.
            if self._is_rank_zero:
                dump_rollout_groups(
                    rollout_dump_path(getattr(self, "_output_dir", None)),
                    step=self._steps_run,
                    batch_size=batch_size,
                    grpo_size=grpo_size,
                    decoded=_decoded,
                    answers=answers,
                    rewards=_rw.reshape(-1).tolist(),
                    successes=_succ.float().reshape(-1).tolist(),
                    proteins=protein_sequences,
                )
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
                rewards.reshape(batch_size * grpo_size),
                group_size=grpo_size,
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
                self._steps_run,
                advantages.abs().max().item(),
                advantages.std().item(),
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
                    "will be skipped",
                    self._steps_run,
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

            # Under HSDP, reduce across the dp_replicate group for this shard index.
            # Using the world group races unrelated FSDP collectives across replicas;
            # reducing over dp_shard would only duplicate one replica's diagnostics.
            if getattr(self, "_dp_replicate", 1) > 1:
                pg = self._dp_mesh.get_group("dp_replicate")
            else:
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
            length_thresholds = torch.tensor(
                [512, 1024, 1536, 2048, 2560, 3072],
                dtype=torch.float32,
                device=dev,
            )
            length_cdf_counts = (lens[:, None] <= length_thresholds).sum(dim=0).float()

            # Reduce sums + sum-of-squares for variance, plus a single scan
            # tensor for length percentile (approximated as max).
            local_n = float(lens.numel())
            sums = torch.stack(
                [
                    pred.sum(),
                    tp.sum(),
                    has_pred.sum(),
                    nonzero.sum(),
                    lens.sum(),
                    stops.sum(),
                    trunc.sum(),
                    group_stds.sum(),
                    torch.tensor(float(rb.shape[0]), device=dev),
                    rb.sum(),
                    (rb * rb).sum(),
                ]
            )
            count = torch.tensor([local_n], device=dev)
            len_max = (
                lens.max().unsqueeze(0)
                if lens.numel()
                else torch.tensor([0.0], device=dev)
            )
            if ws > 1:
                _dist.all_reduce(sums, op=_dist.ReduceOp.SUM, group=pg)
                _dist.all_reduce(count, op=_dist.ReduceOp.SUM, group=pg)
                _dist.all_reduce(len_max, op=_dist.ReduceOp.MAX, group=pg)
                _dist.all_reduce(length_cdf_counts, op=_dist.ReduceOp.SUM, group=pg)
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
                    self._steps_run,
                    int(n),
                    sums[2].item() / n,  # go_emit
                    sums[3].item() / n,  # nonzero_rew
                    sums[0].item() / n,  # mean_pred
                    sums[1].item() / n,  # mean_tp
                    sums[4].item() / n,  # len_mean
                    len_max.item(),
                    sums[6].item() / n,  # trunc_rate
                    sums[5].item() / n,  # stop_rate
                    sums[7].item() / n_groups,  # group_std (mean over groups)
                    batch_var**0.5,  # batch_std
                )
                log.info(
                    "BIOREASON_LENGTH_CDF step=%d le512=%.3f le1024=%.3f "
                    "le1536=%.3f le2048=%.3f le2560=%.3f le3072=%.3f",
                    self._steps_run,
                    *(length_cdf_counts / n).tolist(),
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
                    if protein_sequences is not None
                    else None
                )
                # empty_cache leaks UR handles under colocate (FSDP + in-process
                # vLLM); the wake path's gc.collect()+synchronize is the safe sub.
                if not _colocate_vllm_mode:
                    device_empty_cache(self._device)
                trajectories.append(
                    self.generate_trajectory(
                        batch_input_ids, batch_answers, batch_proteins
                    )
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

        # One-shot per-run rank-0 diagnostic naming the grpo_step path actually
        # taken. The base recipe emits this (grpo_full_finetune_distributed_xpu.py
        # ~4150) but this override replaces that code, so every BioReason run was
        # silently reporting "grpo_step path: NOT EMITTED" to
        # scripts/check_run_health.sh -- disabling the one discriminator
        # docs/RESULTS_DISCIPLINE.md relies on to tell a chunked run from a
        # single-backward one, and disabling --compare's path-match assertion
        # entirely. Branch conditions below MUST mirror the if/elif chain that
        # follows; the extra `_multimodal` term is real (packing is skipped on the
        # embeds path, so enable_packing=true still lands on CHUNKED here).
        if self._is_rank_zero and not getattr(self, "_grpo_path_logged", False):
            _env_val = os.environ.get("TORCHTUNE_USE_CHUNKED_LOSS", "<unset>")
            _num_seqs_init = trajectory.query_responses.shape[0]
            if self._enable_packing and not _multimodal:
                _path, _num_chunks = "PACKED", 1
            elif _env_val == "1" and self._expert_parallel_degree <= 1:
                _path, _num_chunks = "SINGLE_BACKWARD", 1
            else:
                _path = "CHUNKED_BACKWARD"
                _fbs_init = max(1, self._forward_batch_size)
                _num_chunks = (_num_seqs_init + _fbs_init - 1) // _fbs_init
            log.info(
                "grpo_step path: %s (TORCHTUNE_USE_CHUNKED_LOSS=%s, fbs=%d, "
                "num_seqs=%d, num_chunks=%d, ep_degree=%d, multimodal=%s, "
                "enable_packing=%s)",
                _path,
                _env_val,
                self._forward_batch_size,
                _num_seqs_init,
                _num_chunks,
                self._expert_parallel_degree,
                _multimodal,
                self._enable_packing,
            )
            self._grpo_path_logged = True

        if self._enable_packing and not _multimodal:
            from torchtune.dev.rl.packing import (
                pack_trajectory_for_training,
                unpack_tensor,
            )

            (
                packed_tokens,
                packed_positions,
                packed_masks,
                bins,
                actual_lens,
            ) = pack_trajectory_for_training(
                trajectory.query_responses,
                trajectory.position_ids,
                self._tokenizer.pad_id,
            )
            log.info(
                "Rank %d: grpo_step packed forward start (%d seqs -> %d packs)",
                self.rank,
                trajectory.query_responses.shape[0],
                packed_tokens.shape[0],
            )
            packed_logits = self._model(
                packed_tokens,
                input_pos=packed_positions,
                mask=packed_masks,
            )
            del packed_tokens, packed_positions, packed_masks
            pi_logits = unpack_tensor(
                packed_logits,
                bins,
                actual_lens,
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

            log.info(
                "Rank %d: single-backward forward start (total=%d seqs)",
                self.rank,
                total_seqs,
            )
            _fwd_t0_sb = time.perf_counter()
            if _multimodal:
                _comp_ids = trajectory.query_responses[:, context_length:]
                _full_emb = self._policy.build_full_embeds(
                    trajectory.prompt_embeds, _comp_ids
                )
                _attn_mask = (
                    trajectory.query_responses != self._tokenizer.pad_id
                ).long()
                # truncate_sequence_for_logprobs([:, ctx-1:-1]) is exactly the
                # scalar-prompt-length case of gather_response_logits, so the
                # response-only projection applies here unchanged.
                _sb_response_length = _comp_ids.shape[1]
                _ro_kwargs = response_only_logits_kwargs(
                    self._model, context_length, _sb_response_length
                )
                pi_logits = self._model(
                    inputs_embeds=_full_emb,
                    attention_mask=_attn_mask,
                    position_ids=trajectory.position_ids,
                    **_ro_kwargs,
                )
                del _full_emb, _attn_mask, _comp_ids
            else:
                _ro_kwargs = {}
                _sb_response_length = (
                    trajectory.query_responses.shape[1] - context_length
                )
                pi_logits = self._model(
                    trajectory.query_responses,
                    input_pos=trajectory.position_ids,
                    mask=trajectory.masks,
                )
            pi_logits = finish_response_logits(
                pi_logits, _ro_kwargs, context_length, _sb_response_length
            )
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
                trajectory.logprobs
                if trajectory.logprobs is not None
                else pi_logprobs.detach()
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
            import torch.distributed as _tdist_sb_fix
            from torchtune.dev.rl.distributed import _orig_reduce_scatter_tensor

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

            _use_fsdp2_grad_sync = (
                num_fwd_chunks > 1
                and hasattr(self._model, "set_requires_gradient_sync")
                and not self._use_fsdp1
            )
            _use_fsdp1_no_sync = (
                num_fwd_chunks > 1
                and self._use_fsdp1
                and hasattr(self._model, "no_sync")
                # 32B per-layer auto_wrap_policy: each FlatParamHandle mixes frozen
                # base weights with trainable LoRA adapter weights. FSDP1's no_sync()
                # accumulates the FULL unsharded gradient for every no_sync'd handle
                # until the eventual synced chunk (see FSDP.no_sync's own docstring:
                # "accumulate the full model gradients ... until the eventual sync").
                # With grpo_samples/forward_batch_size forcing 3+ no_sync'd chunks
                # before the final synced one, that's ~65.6 GiB/64 layers held live
                # across all 64 layers simultaneously by the last chunk — matches the
                # observed ~58 GiB backward OOM being CONSTANT regardless of
                # max_generated_tokens (256 vs 1024 gave nearly identical OOM points,
                # ruling out activation/sequence-length as the driver). Sync every
                # chunk instead at this scale — extra reduce-scatters cost time, not
                # a correctness or memory risk.
                and not self._fsdp_full_shard_at_rest
            )
            _use_ddp_no_sync = (
                num_fwd_chunks > 1
                and not self._use_fsdp1
                and not hasattr(self._model, "set_requires_gradient_sync")
                and isinstance(self._model, torch.nn.parallel.DistributedDataParallel)
            )

            _chunk_losses, _chunk_policy_losses, _chunk_kl_losses = [], [], []
            _chunk_ratios, _chunk_clipfracs = [], []
            _chunk_weights = []
            _chunk_pi_logprobs = [None] * total_seqs
            _bwd_total = 0.0

            if self._sort_policy_chunks_by_length:
                _chunk_rows = get_length_sorted_response_chunks(
                    trajectory.response_padding_masks, fwd_bs
                )
            else:
                _chunk_ranges = [
                    (_cs, min(_cs + fwd_bs, total_seqs))
                    for _cs in range(0, total_seqs, fwd_bs)
                ]
                if self._trim_chunk_width:
                    _chunk_ranges = get_descending_response_chunk_ranges(
                        trajectory.response_padding_masks, fwd_bs
                    )
                _chunk_rows = [list(range(_cs, _ce)) for _cs, _ce in _chunk_ranges]
            for _chunk_index, _rows in enumerate(_chunk_rows):
                _is_last_chunk = _chunk_index + 1 == num_fwd_chunks
                _trajectory_response_length = trajectory.response_padding_masks.shape[1]
                _chunk_response_length = (
                    get_right_padded_response_length(
                        trajectory.response_padding_masks[_rows]
                    )
                    if self._trim_chunk_width
                    else _trajectory_response_length
                )
                _chunk_total_length = context_length + _chunk_response_length
                if self._trim_chunk_width and self._is_rank_zero:
                    log.info(
                        "BIOREASON_CHUNK_WIDTH phase=train_policy rows=%s "
                        "response=%d/%d total=%d",
                        _rows,
                        _chunk_response_length,
                        _trajectory_response_length,
                        _chunk_total_length,
                    )
                _chunk_query_responses = trajectory.query_responses[
                    _rows, :_chunk_total_length
                ]
                _chunk_padding_masks = trajectory.response_padding_masks[
                    _rows, :_chunk_response_length
                ]
                if self._device.type == "xpu" and self._is_rank_zero:
                    _chunk_token_counts = (
                        ~trajectory.response_padding_masks[_rows]
                    ).sum(dim=-1)
                    log.info(
                        "Rank 0: PRE-train-fwd rows=%s response_tokens=%s "
                        "alloc=%.2f GiB, resv=%.2f GiB",
                        _rows,
                        _chunk_token_counts.tolist(),
                        torch.xpu.memory_allocated() / 1024**3,
                        torch.xpu.memory_reserved() / 1024**3,
                    )
                log.info("Rank %d: grpo_step rows=%s fwd", self.rank, _rows)
                _forward_context_length = context_length
                if _multimodal:
                    _chunk_comp_ids = _chunk_query_responses[:, context_length:]
                    _forward_prompt_embeds = trajectory.prompt_embeds[_rows]
                    _chunk_attn_mask = (
                        _chunk_query_responses != self._tokenizer.pad_id
                    ).long()
                    _forward_position_ids = trajectory.position_ids[
                        _rows, :_chunk_total_length
                    ]
                    if self._compact_prompt_chunks:
                        (
                            _forward_prompt_embeds,
                            _chunk_attn_mask,
                            _forward_position_ids,
                            _forward_context_length,
                        ) = compact_prompt_completion_batch(
                            _forward_prompt_embeds,
                            _chunk_query_responses[:, :context_length],
                            _chunk_comp_ids,
                            self._tokenizer.pad_id,
                        )
                    _chunk_full_emb = self._policy.build_full_embeds(
                        _forward_prompt_embeds,
                        _chunk_comp_ids,
                        _forward_context_length
                        if isinstance(_forward_context_length, torch.Tensor)
                        else None,
                        _chunk_attn_mask.shape[1]
                        if isinstance(_forward_context_length, torch.Tensor)
                        else None,
                    )
                    _ro_kwargs = response_only_logits_kwargs(
                        self._model,
                        _forward_context_length,
                        _chunk_comp_ids.shape[1],
                    )
                    _c_response_length = _chunk_comp_ids.shape[1]
                    _c_logits = self._model(
                        inputs_embeds=_chunk_full_emb,
                        attention_mask=_chunk_attn_mask,
                        position_ids=_forward_position_ids,
                        **_ro_kwargs,
                    )
                    del _chunk_full_emb, _chunk_attn_mask
                else:
                    _ro_kwargs = {}
                    _c_response_length = _chunk_response_length
                    _c_logits = self._model(
                        _chunk_query_responses,
                        input_pos=trajectory.position_ids[
                            _rows, :_chunk_total_length
                        ],
                        mask=trajectory.masks[
                            _rows,
                            :_chunk_total_length,
                            :_chunk_total_length,
                        ],
                    )
                _c_logits = finish_response_logits(
                    _c_logits,
                    _ro_kwargs,
                    _forward_context_length,
                    _c_response_length,
                )
                if _multimodal:
                    del _chunk_comp_ids
                _c_pi_lp = rlhf.batched_logits_to_logprobs(
                    _c_logits,
                    _chunk_query_responses[:, context_length:],
                    self._temperature,
                    chunk_size=1,
                )
                _c_pi_lp.masked_fill_(_chunk_padding_masks, 1.0)
                del _c_logits
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                if self._device.type == "xpu" and self._is_rank_zero:
                    log.info(
                        "Rank 0: POST-train-fwd rows=%s alloc=%.2f GiB, resv=%.2f GiB",
                        _rows,
                        torch.xpu.memory_allocated() / 1024**3,
                        torch.xpu.memory_reserved() / 1024**3,
                    )

                if self._compute_rollout_logprobs_required:
                    assert trajectory.logprobs is not None, (
                        "async_generation / always_compute_rollout_logprobs is set but "
                        "trajectory.logprobs is None"
                    )
                _c_old_lp = (
                    trajectory.logprobs[_rows, :_chunk_response_length]
                    if trajectory.logprobs is not None
                    else _c_pi_lp.detach()
                )
                _c_loss, _c_pol, _c_kl, _c_rat, _c_clip = self._loss_fn(
                    _c_old_lp,
                    _c_pi_lp,
                    trajectory.ref_logprobs[_rows, :_chunk_response_length],
                    trajectory.advantages[_rows],
                    padding_masks=~_chunk_padding_masks,
                )
                _chunk_losses.append(_c_loss.detach())
                _chunk_policy_losses.append(_c_pol.detach())
                _chunk_kl_losses.append(_c_kl.detach())
                _chunk_ratios.append(_c_rat.detach())
                _chunk_clipfracs.append(_c_clip.detach())
                _chunk_weight = len(_rows) / total_seqs
                _chunk_weights.append(_chunk_weight)
                _padded_pi_logprobs = pad_response_logprobs(
                    _c_pi_lp.detach(), _trajectory_response_length
                )
                for _row_offset, _row in enumerate(_rows):
                    _chunk_pi_logprobs[_row] = _padded_pi_logprobs[
                        _row_offset : _row_offset + 1
                    ]
                del _chunk_query_responses, _chunk_padding_masks

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
                _rsc_bypass_chunk = self._expert_parallel_degree <= 1
                _rsc_patch_saved_ck = None
                import torch.distributed as _tdist_ck_fix
                from torchtune.dev.rl.distributed import _orig_reduce_scatter_tensor

                if _rsc_bypass_chunk:
                    _rsc_patch_saved_ck = _tdist_ck_fix.reduce_scatter_tensor
                    _tdist_ck_fix.reduce_scatter_tensor = _orig_reduce_scatter_tensor
                    if self._is_rank_zero and not getattr(
                        self, "_chunked_rsc_bypass_logged", False
                    ):
                        log.info(
                            "chunked backward: non-EP reduce_scatter bypass ACTIVE "
                            "(native XCCL; avoids gloo CPU-bounce)"
                        )
                        self._chunked_rsc_bypass_logged = True
                try:
                    with _bwd_ctx:
                        (
                            _c_loss
                            * _chunk_weight
                            / max(1, self._gradient_accumulation_steps)
                        ).backward()
                finally:
                    if _rsc_bypass_chunk:
                        _tdist_ck_fix.reduce_scatter_tensor = _rsc_patch_saved_ck
                if _use_fsdp2_grad_sync and _is_last_chunk:
                    self._model.set_requires_gradient_sync(True)
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _bwd_total += time.perf_counter() - _bwd_t0

            # Grad census IMMEDIATELY after the last backward, before anything else can
            # touch .grad. Paired with the census in _sync_ignored_trainable_grads, this
            # brackets the window in which grads go missing at dp_replicate=1 (job
            # 8827075: 7 of 12 ranks reported 0/904 there, counts summing to exactly 896).
            # If the two counts AGREE, the grads never existed and the cause is in
            # backward/FSDP; if they DIFFER, something between the two call sites clears
            # them. Costs one int comparison per step; only logs on the anomaly.
            if getattr(self, "_projectors_fsdp_ignored", False) and getattr(
                self, "_ignored_trainable_params", None
            ):
                _pb = sum(
                    1 for p in self._ignored_trainable_params if p.grad is not None
                )
                # Same calibration as the sync-site census: the 8 projector params never
                # get grads (embeds built under no_grad and cached), so the healthy count
                # is 896/904 at 32B, not 904/904. Alarm only on MISSING LORA grads.
                _npj = sum(
                    1
                    for _m in (
                        self._model.protein_projection,
                        self._model.go_projection,
                    )
                    for _ in _m.parameters()
                ) if hasattr(self._model, "protein_projection") else 0
                if _pb < len(self._ignored_trainable_params) - _npj:
                    log.warning(
                        "IGNORED_GRAD_CENSUS_POSTBWD rank=%d present=%d/%d "
                        "(measured at end of grpo_step backward; compare with the "
                        "IGNORED_GRAD_CENSUS line from _sync_ignored_trainable_grads)",
                        self.rank,
                        _pb,
                        len(self._ignored_trainable_params),
                    )

            loss = torch.stack(
                [value * weight for value, weight in zip(_chunk_losses, _chunk_weights)]
            ).sum()
            policy_loss = torch.stack(
                [
                    value * weight
                    for value, weight in zip(_chunk_policy_losses, _chunk_weights)
                ]
            ).sum()
            kl_loss = torch.stack(
                [
                    value * weight
                    for value, weight in zip(_chunk_kl_losses, _chunk_weights)
                ]
            ).sum()
            # stack().mean() not cat(): GRPOSimpleLoss returns ratios as a 0-dim
            # scalar (torch.tensor(1.0)); torch.cat can't concatenate 0-dim tensors
            # (crashes only on the chunked path, fbs < num_seqs). Mirrors the base
            # recipe (grpo_full_finetune_distributed_xpu.py:4294).
            ratios = torch.stack(
                [value * weight for value, weight in zip(_chunk_ratios, _chunk_weights)]
            ).sum()
            clipfrac = torch.stack(
                [
                    value * weight
                    for value, weight in zip(_chunk_clipfracs, _chunk_weights)
                ]
            ).sum()
            pi_logprobs = torch.cat(_chunk_pi_logprobs)
            _fwd_time = time.perf_counter() - _fwd_t0 - _bwd_total

        log.info("Rank %d: grpo_step bwd=%.1fs", self.rank, _bwd_total)

        with torch.no_grad():
            _old_lp = (
                trajectory.logprobs if trajectory.logprobs is not None else pi_logprobs
            )
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

    def _sync_ignored_trainable_grads(self) -> None:
        if not getattr(self, "_projectors_fsdp_ignored", False):
            return
        shard_pg = getattr(self, "_gloo_dp_shard_pg", None)
        replicate_pg = getattr(self, "_gloo_dp_replicate_pg", None)
        if shard_pg is None:
            return

        # DEADLOCK FIX (2026-09-14, job 8826889). The previous implementation bucketed
        # by `param.grad.dtype` and looped over the resulting dict, calling all_reduce
        # INSIDE that loop. Both the bucket set and the iteration count were therefore
        # functions of LOCAL state (`param.grad is not None`), so a rank with no grads
        # made ZERO collective calls while its peers made one -- a guaranteed hang.
        #
        # Observed at dp_replicate=1: rank 0 logged "Averaged 0 ignored trainable
        # gradients" (empty dict -> zero iterations -> fell through and continued) while
        # ranks 1-11 blocked in all_reduce until the 1800s gloo timeout; PBS
        # Exit_status=143, zero steps completed. It never fired at 16N only because rank
        # 0 happened to have the same 896 grads as everyone else.
        #
        # Fix: iterate a FIXED, rank-independent list. `_ignored_trainable_params` is
        # built from model structure and is identical on every rank, and `param.dtype`
        # is structural too (unlike `param.grad.dtype`, which does not exist when the
        # grad is missing). Every rank therefore performs exactly the same number of
        # collectives in the same order, whatever its local grads look like.
        params_by_dtype: dict = {}
        for param in self._ignored_trainable_params:
            params_by_dtype.setdefault(param.dtype, []).append(param)

        # Per-rank grad-presence census, logged BEFORE any collective. Job 8826889 left
        # an open question the logs could not answer: rank 0 reported 0 grads while its
        # peers had some, but nothing recorded WHICH ranks were missing WHAT. The fixed
        # collective no longer hangs, so a recurrence would otherwise pass silently as a
        # GRAD PRESENCE DISAGREEMENT with no forensic detail. One line per rank, once
        # per step, is cheap and makes the next occurrence self-diagnosing.
        _local_present = sum(
            1 for p in self._ignored_trainable_params if p.grad is not None
        )
        _local_total = len(self._ignored_trainable_params)
        # THRESHOLD CALIBRATION (2026-09-15): the healthy state is NOT total/total.
        # `_ignored_trainable_params` = 896 LoRA + 8 projector at 32B, and the 8
        # projector params NEVER receive grads: build_prompt_embeds runs under
        # torch.no_grad() and the training forward consumes the CACHED
        # trajectory.prompt_embeds, so the projectors are not in the autograd graph.
        # Confirmed on the healthy 16N run, which logs exactly "Averaged 896".
        # Warning on `present != total` would therefore fire on every healthy run.
        # Alarm only when LoRA grads are missing, which is the real defect
        # (2N: rank 0 had 0/904, rank 1 had 384/904).
        _n_projector = sum(
            1
            for _m in (self._model.protein_projection, self._model.go_projection)
            for _ in _m.parameters()
        ) if hasattr(self._model, "protein_projection") else 0
        _expected = _local_total - _n_projector
        if _local_present < _expected:
            # NAME the params that DO have grads. Two runs (8827075, 8827354) produced
            # byte-identical per-rank counts (0/128/384, summing to exactly 896), and the
            # POSTBWD census matched the sync-site census exactly -- so nothing clears
            # them, backward simply never produces them, deterministically. Counts alone
            # cannot say WHICH params those are; 128 == 1 projection type x 2 x 64 layers
            # is consistent with a projection-type partition but that is inference, not
            # evidence. This logs the actual names so the next run settles it.
            _named = []
            try:
                _id2name = {
                    id(p): n for n, p in self._model.named_parameters()
                }
                _named = sorted(
                    {
                        _id2name.get(id(p), "<unmapped>")
                        for p in self._ignored_trainable_params
                        if p.grad is not None
                    }
                )
            except Exception:  # never let diagnostics break the step
                pass
            # Collapse "...layers.N.<rest>" -> "<rest>" so 64 layers of one projection
            # show up as ONE entry; that is exactly the distinction we need.
            import re as _re

            _kinds = sorted({_re.sub(r"\.layers\.\d+\.", ".layers.N.", n) for n in _named})
            log.warning(
                "IGNORED_GRAD_CENSUS rank=%d present=%d/%d distinct_kinds=%d "
                "kinds=%s (missing grads on this rank; if ranks disagree the update "
                "is averaged over contributors only)",
                self.rank,
                _local_present,
                _local_total,
                len(_kinds),
                _kinds[:12],
            )

        _n_present_total = 0
        _n_total = 0
        _disagree: list = []
        # sorted() for a deterministic bucket order across ranks; insertion order is
        # already identical, but an explicit total order costs nothing and removes any
        # dependence on dict-ordering subtleties.
        for dtype in sorted(params_by_dtype, key=str):
            params = params_by_dtype[dtype]
            sizes = [param.numel() for param in params]

            # Zero-fill missing grads so the buffer shape is rank-independent. A missing
            # grad contributes nothing to the sum, which is the correct neutral element.
            flat_cpu = torch.cat(
                [
                    (
                        param.grad.detach()
                        .reshape(-1)
                        .to(device="cpu", dtype=dtype)  # no-op when grad.dtype == dtype
                        if param.grad is not None
                        else torch.zeros(param.numel(), dtype=dtype, device="cpu")
                    )
                    for param in params
                ]
            )
            # Per-param presence, reduced alongside the grads: this is what lets us tell
            # "all ranks contributed" (the validated case) from "some ranks silently had
            # no grad" (a real bug that used to present only as a hang).
            present = torch.tensor(
                [1.0 if param.grad is not None else 0.0 for param in params],
                dtype=torch.float32,
                device="cpu",
            )

            torch.distributed.all_reduce(flat_cpu, group=shard_pg)
            torch.distributed.all_reduce(present, group=shard_pg)
            _divisor = float(self._dp_shard)
            if replicate_pg is not None and self._dp_replicate > 1:
                torch.distributed.all_reduce(flat_cpu, group=replicate_pg)
                torch.distributed.all_reduce(present, group=replicate_pg)
                _divisor *= float(self._dp_replicate)

            _n_total += len(params)
            _n_present_total += int(present.eq(_divisor).sum().item())

            offset = 0
            for param, size, n_present in zip(params, sizes, present.tolist()):
                chunk = flat_cpu[offset : offset + size]
                offset += size
                if n_present == _divisor:
                    # Every rank contributed: plain mean. BYTE-IDENTICAL to the
                    # pre-fix path, so validated 16N numerics are unchanged.
                    chunk = chunk / _divisor
                elif n_present > 0:
                    # Partial: mean over the ranks that actually had a grad. Dividing by
                    # the full world here would silently scale the update down by
                    # n_present/_divisor -- a wrong-but-plausible training run.
                    chunk = chunk / n_present
                    _disagree.append((param.shape, int(n_present), int(_divisor)))
                else:
                    # No rank had this grad (e.g. the 8 projector params, whose embeds
                    # are built under no_grad and cached). Nothing to write back.
                    continue
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.copy_(chunk.view_as(param.grad))

        if _disagree and self._is_rank_zero:
            # Loud: rank-dependent grad presence is the signature of a real upstream
            # bug (it is what made the old code deadlock). Surface it instead of
            # quietly training on a diluted or partial gradient.
            log.error(
                "GRAD PRESENCE DISAGREEMENT across ranks for %d/%d ignored trainable "
                "params (first 3: %s). Averaged over the contributing ranks only. This "
                "indicates some ranks lost grads for replicated params -- investigate; "
                "do not trust this run's updates.",
                len(_disagree),
                _n_total,
                _disagree[:3],
            )

        if self._is_rank_zero:
            log.info(
                "Averaged %d/%d ignored trainable gradients across %d x %d HSDP ranks",
                _n_present_total,
                _n_total,
                self._dp_replicate,
                self._dp_shard,
            )

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

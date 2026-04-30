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
from typing import Any, Optional

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
        self._enable_packing = cfg.get("enable_packing", False)
        self._expert_parallel_degree = cfg.get("expert_parallel_degree", 1)
        self._shard_pg = None
        self._compute_rollout_logprobs_required = cfg.get("always_compute_rollout_logprobs", False)
        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        self._eval_every_n_steps = cfg.get("eval_every_n_steps", 0)
        self._eval_max_examples = cfg.get("eval_max_examples", 50)

        stop_token_ids = (
            list(self._tokenizer.stop_tokens)
            if hasattr(self._tokenizer, 'stop_tokens') and self._tokenizer.stop_tokens
            else [self._tokenizer.eos_id]
        )
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

        # Optimizer, loss, dataloader
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=None,
        )
        self._loss_fn = config.instantiate(cfg.loss)
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
        # Bypasses FSDP gather (not applicable for DDP/plain model).
        if hasattr(self._policy, 'vllm_param_iter'):
            if self._is_rank_zero:
                save_dir = os.path.join(self._output_dir, f"epoch_{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    self._policy.protein_projection.state_dict(),
                    os.path.join(save_dir, "protein_projection.pt"),
                )
                torch.save(
                    self._policy.go_projection.state_dict(),
                    os.path.join(save_dir, "go_projection.pt"),
                )
                self._policy.backbone.save_pretrained(
                    os.path.join(save_dir, "backbone")
                )
                log.info("BioReason checkpoint saved to %s", save_dir)
            # In dedicated_rank mode, rank 11 is in server loop — use training_pg only.
            if torch.distributed.is_initialized():
                pg = self._training_pg if self._vllm_mode == "dedicated_rank" else None
                torch.distributed.barrier(group=pg)
            return

        super().save_checkpoint(epoch)

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
        log.info("BioReason: loading policy model from %s", ckpt_dir)
        self._model = BioReasonModel(
            ckpt_dir=ckpt_dir,
            device=self._device,
            dtype=self._dtype,
        )
        self._model.train()
        if self._enable_activation_checkpointing:
            self._model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            log.info("BioReason: gradient checkpointing enabled on backbone")

        ref_device = torch.device("cpu") if self._ref_cpu_offload else self._device
        log.info("BioReason: loading ref model from %s (device=%s)", ckpt_dir, ref_device)
        self._ref_model = BioReasonModel(
            ckpt_dir=ckpt_dir,
            device=ref_device,
            dtype=self._dtype,
        )
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
        _wrap_fsdp1 = (
            self._vllm_mode == "dedicated_rank" and self._vllm_dedicated_rank is not None
        ) or (self._vllm_mode == "server")
        if _wrap_fsdp1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
            if self._vllm_mode == "dedicated_rank":
                # Generic PG setup: training_pg (xccl, [0..N-2]) + wsync_pg (gloo, [0, N-1]).
                # new_group order must match _setup_dedicated_vllm_rank on the vLLM rank.
                self._setup_dedicated_training_pgs(cfg)
            else:
                # server mode: all ranks are training ranks; vLLM lives on a separate
                # node (asymmetric launcher) and is not part of WORLD. No wsync PG —
                # weights ship over HTTP (raw_bytes path → /dev/shm + /collective_rpc).
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
            self._model = FSDP(
                _pre_wrap,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                mixed_precision=_mp_policy,
                process_group=self._training_pg,
                ignored_modules=[m for m in [_pre_wrap._embed,
                                             _pre_wrap.protein_encoder,
                                             _pre_wrap.go_encoder]
                                 if m is not None and isinstance(m, torch.nn.Module)],
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
                "Rank %d: FSDP1 SHARD_GRAD_OP (ZeRO-2) wrapped over training_pg (%d ranks), "
                "ignored=[_embed, protein_encoder, go_encoder], %s",
                self.rank, len(_training_ranks), _wsync_desc,
            )
        else:
            self._training_pg = None
            self._wsync_pg = None

    # ── vLLM generation override ───────────────────────────────────────────────

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

        if self._is_rank_zero:
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
            )

            t0 = time.perf_counter()
            num_clients = len(self._vllm_clients)
            # Submit one /v1/completions request per prompt so the vLLM scheduler
            # on each tile sees concurrent in-flight requests and can batch their
            # decode kernels. Round-robin assignment keeps load balanced.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _call_one(client, embed):
                out = client.generate_from_embeds(prompt_embeds=[embed], **gen_kwargs)
                return out[0] if out else []

            completions = [None] * bsz
            max_workers = max(1, min(bsz, num_clients * 8))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_call_one, self._vllm_clients[i % num_clients], embeds_list[i]): i
                    for i in range(bsz)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    completions[i] = future.result()
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
                "Rank %d: vLLM-embeds generation: %d sequences (%d clients), %d tokens in "
                "%.1fs (%.1f tok/s)",
                self.rank, bsz, num_clients, total_tokens, gen_time,
                total_tokens / max(gen_time, 0.01),
            )
        else:
            query_responses = batch_input_ids.new_empty(bsz, total_len)

        torch.distributed.broadcast(query_responses, src=0)
        return query_responses

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
            self._sync_colocated_weights()
            self._vllm_llm.wake_up(tags=["kv_cache"])
            self._vllm_is_sleeping = False
            log.info("Rank %d: vLLM wake_up + weight sync completed in %.2fs",
                     self.rank, time.perf_counter() - t_wake)
        elif self._vllm_llm is not None and hasattr(self, '_vllm_kv_cache_shapes'):
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

        # Multimodal: build prompt embeddings once per unique prompt, then expand to B*G.
        # BioReasonModel.build_prompt_embeds(input_ids [B, P], protein_sequences [B])
        # returns [B, P, H] on CPU. Expand to [B*G, P, H] for rollouts.
        prompt_embeds = None
        if protein_sequences is not None and hasattr(self._policy, 'build_prompt_embeds'):
            # protein_projection and go_projection are trainable → FSDP-sharded at rest.
            # summon_full_params gathers them so build_prompt_embeds sees complete weights.
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
            if getattr(self, "_is_bioreason", False) and prompt_embeds is not None:
                query_responses = self._generate_with_vllm_server_embeds(
                    batch_input_ids, context_length, prompt_embeds
                )
            elif getattr(self, "_pending_async_query_responses", None) is not None:
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
                device_empty_cache(self._device)
            ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
            del ref_logprobs_chunks
            log.info("Rank %d: ref forward done (chunked)", self.rank)
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
            for _b in range(batch_size):
                for _g in range(grpo_size):
                    _ids = responses[_b, _g]
                    _non_pad = _ids[_ids != self._tokenizer.pad_id]
                    _decoded.append(self._tokenizer.decode(_non_pad.cpu().tolist()))
                    _expanded_answers.append(answers[_b])
            _rw, _succ = _br_reward(_decoded, _expanded_answers)
            rewards = _rw.view(batch_size, grpo_size, 1)
            successes = _succ.float().view(batch_size, grpo_size, 1)
            metadata = {}
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

        advantages = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.reshape(batch_size * grpo_size)
        del responses
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

    def generate_trajectory_batched(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
        protein_sequences: Optional[list] = None,
    ) -> GRPOTrajectory:
        """Generates trajectories using forward_batch_size micro-batches."""
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_answers = answers[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_proteins = (
                    protein_sequences[batch_start : batch_start + self._forward_batch_size]
                    if protein_sequences is not None else None
                )
                device_empty_cache(self._device)
                trajectories.append(
                    self.generate_trajectory(batch_input_ids, batch_answers, batch_proteins)
                )
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
            ratios = torch.cat(_chunk_ratios)
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

    # ── Train override ─────────────────────────────────────────────────────────

    def train(self) -> None:
        """Override train() to extract protein_sequences and pass through."""
        if not getattr(self, '_is_bioreason', False):
            super().train()
            return

        # Dedicated vLLM rank: run generation server loop, then exit.
        if self._is_vllm_rank:
            self._run_vllm_generation_server()
            return

        try:
            training.cleanup_before_training()
        except RuntimeError:
            pass

        self._optimizer.zero_grad(set_to_none=True)

        training_complete = False
        curr_epoch = 0
        for curr_epoch in range(self.total_epochs):
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                tokens = batch["tokens"]
                answers = batch["answers"]
                protein_sequences = batch.get("protein_sequences", None)
                tokens = tokens.to(self._device)

                _, context_length = tokens.shape

                if self._device.type == "xpu" and self.rank == 0:
                    _alloc = torch.xpu.memory_allocated() / 1024**3
                    _resv = torch.xpu.memory_reserved() / 1024**3
                    log.info("Rank 0: PRE-STEP %d memory: allocated=%.2f GiB, reserved=%.2f GiB",
                             self._steps_run, _alloc, _resv)

                _step_t0 = time.perf_counter()
                trajectory = self.generate_trajectory_batched(
                    tokens, answers, protein_sequences
                )
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _gen_time = time.perf_counter() - _step_t0
                if not self._production_mode:
                    self._training_barrier()

                grpo_stats: list[GRPOStats] = []
                _grpo_t0 = time.perf_counter()

                for _ in range(self._ppo_epochs):
                    total_samples = trajectory.query_responses.shape[0]
                    if self._gradient_accumulation_steps > 1:
                        micro_batch_size = total_samples // self._gradient_accumulation_steps
                        for ga_step in range(self._gradient_accumulation_steps):
                            start_idx = ga_step * micro_batch_size
                            end_idx = start_idx + micro_batch_size
                            micro_traj = _slice_trajectory(trajectory, start_idx, end_idx)
                            is_last = (ga_step == self._gradient_accumulation_steps - 1)
                            if not is_last and self._use_fsdp1 and hasattr(self._model, 'no_sync'):
                                with self._model.no_sync():
                                    step_stats = self.grpo_step(micro_traj, context_length)
                            else:
                                step_stats = self.grpo_step(micro_traj, context_length)
                            grpo_stats.append(step_stats)
                            if not is_last and self._device.type == "xpu":
                                import gc
                                gc.collect()
                    else:
                        step_stats = self.grpo_step(trajectory, context_length)
                        grpo_stats.append(step_stats)

                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _grpo_time = time.perf_counter() - _grpo_t0

                    # Clip gradients.
                    if self._clip_grad_norm is not None:
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        if self._use_fsdp1 and isinstance(self._model, FSDP):
                            # FSDP1 SHARD_GRAD_OP: use FSDP.clip_grad_norm_ which AllReduces
                            # norm² across training_pg before clipping.
                            self._model.clip_grad_norm_(float(self._clip_grad_norm))
                        else:
                            # Plain DDP / no FSDP: compute norm on-device, single .item().
                            _local_norm_sq = torch.zeros(
                                (), device=self._device, dtype=torch.float32
                            )
                            _grads_to_clip = []
                            for _p in self._model.parameters():
                                if _p.grad is not None:
                                    _local_norm_sq = _local_norm_sq + _p.grad.float().pow(2).sum()
                                    _grads_to_clip.append(_p.grad)
                            _grad_norm = _local_norm_sq.sqrt()
                            _clip_coef = (
                                float(self._clip_grad_norm) / (_grad_norm + 1e-6)
                            ).clamp(max=1.0)
                            for _g in _grads_to_clip:
                                _g.detach().mul_(_clip_coef)

                    if self._device.type == "xpu":
                        torch.xpu.synchronize()

                    if not self._production_mode:
                        self._training_barrier()

                    log.info("Rank %d: optimizer.step()", self.rank)
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    if not self._production_mode:
                        self._training_barrier()

                self._steps_run += 1
                _step_time = time.perf_counter() - _step_t0

                if self._is_rank_zero and self._steps_run % self._log_every_n_steps == 0:
                    _stats = grpo_stats[-1]
                    log.info(
                        "Step %d: loss=%.4f policy_loss=%.4f kl_loss=%.4f "
                        "gen=%.1fs grpo=%.1fs total=%.1fs",
                        self._steps_run,
                        _stats.loss.item(),
                        _stats.policy_loss.item(),
                        _stats.kl_loss.item(),
                        _gen_time,
                        _grpo_time,
                        _step_time,
                    )
                    if hasattr(self, '_metric_logger'):
                        self._metric_logger.log_dict(
                            {
                                "loss": _stats.loss.item(),
                                "policy_loss": _stats.policy_loss.item(),
                                "kl_loss": _stats.kl_loss.item(),
                                "approx_policy_kls": _stats.approx_policy_kls.item(),
                                "ratios_mean": _stats.ratios.mean().item(),
                                "clipfrac": _stats.clipfrac.item(),
                                "gen_time": _gen_time,
                                "step_time": _step_time,
                            },
                            step=self._steps_run,
                        )

                self.cleanup_after_step(trajectory, grpo_stats)

                if self._steps_run >= self._total_steps:
                    training_complete = True
                    break

            if training_complete:
                break

        if self._is_rank_zero:
            log.info("BioReason GRPO training complete after %d steps.", self._steps_run)
        self.save_checkpoint(epoch=curr_epoch)


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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# BioReason multimodal SFT on a native torchtune Gemma 4 backbone (Aurora / XPU).
#
# Subclass of FullFinetuneRecipeDistributedXPU. It reuses the parent's FSDP2 setup,
# no_sync/ZeRO-2 gradient accumulation, async dataloader, time_per_step metric, and
# checkpointing. It overrides only what the multimodal forward requires:
#   - _setup_model:  build BioReasonNativeModel (gemma4_31b/lora_gemma4_31b) and load
#       the GEMMA4 backbone weights, instead of a bare TransformerDecoder.
#   - _setup_data:   wrap the dataloader so the per-batch protein/go string side-inputs
#       are stashed on the recipe (batch_to_device only accepts tensors) and a
#       tensor-only batch flows through the unchanged parent train loop.
#   - _loss_step:    splice multimodal embeds (grad-enabled) and run the native decoder.

import os
import sys
import time
from functools import partial
from typing import Any, Optional

# Reuse the parent recipe's module (it installs the torchtune sys.modules shim + XPU
# patches at import time). Load it by file path so this recipe is import-order safe.
import importlib.util as _imp_util

_PARENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "full_finetune_distributed_xpu.py")
_spec = _imp_util.spec_from_file_location(
    "_sft_full_finetune_distributed_xpu", _PARENT_PATH
)
_parent_mod = _imp_util.module_from_spec(_spec)
_spec.loader.exec_module(_parent_mod)

FullFinetuneRecipeDistributedXPU = _parent_mod.FullFinetuneRecipeDistributedXPU

import torch
from torch import nn
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.dev.bioreason.model_native import BioReasonNativeModel
from torchtune.modules.loss import SFTLoss


class _SideInputDataLoader:
    """Thin wrapper over a StatefulDataLoader that pops the non-tensor multimodal side
    inputs (protein_sequences / go_aspects) out of each batch and stashes them on the
    recipe, yielding a tensor-only batch so the parent train loop's batch_to_device and
    fingerprint paths work unchanged. Proxies sampler/state_dict for the parent."""

    # protein/go strings + (when packing) the per-doc position ids, block-mask seq_lens,
    # and doc->row map. All are consumed by _loss_step, not by the parent's token/label loop.
    SIDE_KEYS = ("protein_sequences", "go_aspects", "input_pos", "seq_lens", "batch_idx_map")

    def __init__(
        self,
        dl: StatefulDataLoader,
        recipe: "BioReasonSFTRecipeDistributedXPU",
        epoch_sampler=None,
    ):
        self._dl = dl
        self._recipe = recipe
        # The epoch-aware sampler whose set_epoch the train loop must call. With a plain
        # DataLoader this is dl.sampler (StatefulDistributedSampler); with a custom
        # batch_sampler (per-bucket sizing) dl.sampler is None, so we hold it explicitly.
        self._epoch_sampler = epoch_sampler if epoch_sampler is not None else dl.sampler

    @property
    def sampler(self):
        return self._epoch_sampler

    def __len__(self):
        return len(self._dl)

    def state_dict(self):
        return self._dl.state_dict()

    def load_state_dict(self, sd):
        return self._dl.load_state_dict(sd)

    def __iter__(self):
        for batch in self._dl:
            side = {k: batch.pop(k) for k in self.SIDE_KEYS if k in batch}
            self._recipe._current_side_inputs = side
            yield batch


class BioReasonSFTRecipeDistributedXPU(FullFinetuneRecipeDistributedXPU):
    """Native-Gemma4 multimodal SFT recipe."""

    def __init__(self, cfg: DictConfig):
        # ── Custom USM caching allocator (MUST run before any XPU allocation, i.e.
        # before super().__init__ touches the device). The parent SFT recipe has no
        # allocator hook, so it gets the DEFAULT XPU allocator: a FRESH VA per step for
        # each FSDP AllGather param buffer. OFI/Slingshot registers each new VA for DMA
        # and never deregisters → receive-side AllGather tiles accumulate external,
        # PyTorch-invisible fabric memory until l0_free→0 → a write to a recycled VA
        # → NotPresent PDE → banned:1 (observed at step ~3 on 12-tile FSDP2; rank0
        # looked fine at 4.85 GiB because the leak is external/uncounted). The custom
        # allocator (usm_caching_alloc.so) pools the AllGather buffer at a STABLE VA so
        # OFI registers once → flat memory. This is the same hook the GRPO recipe uses
        # to run 32B; see bugs/project_ccl_ipc_handle_cache.md (CONFIRMED fix).
        _usm_so = os.environ.get("XPU_USM_ALLOC_SO")
        if _usm_so:
            from torch.xpu.memory import (
                XPUPluggableAllocator,
                change_current_allocator,
            )

            _usm_alloc = XPUPluggableAllocator(
                _usm_so, "xpu_usm_malloc", "xpu_usm_free"
            )
            change_current_allocator(_usm_alloc)
            # Pluggable allocator doesn't implement getDeviceStats/emptyCache —
            # monkeypatch the memory-query API to no-ops so logging doesn't crash.
            torch.xpu.memory_allocated = lambda device=None: 0
            torch.xpu.memory_reserved = lambda device=None: 0
            torch.xpu.max_memory_allocated = lambda device=None: 0
            torch.xpu.max_memory_reserved = lambda device=None: 0
            torch.xpu.reset_peak_memory_stats = lambda device=None: None
            torch.xpu.empty_cache = lambda: None
            torch.xpu.memory_stats = lambda device=None: {}

        super().__init__(cfg)
        self._current_side_inputs: dict = {}
        # Stash the LoRA rank/alpha for checkpoint-time merge (parent doesn't keep cfg).
        _m = cfg.get("model", {})
        self._lora_rank = int(_m.get("lora_rank", 32))
        self._lora_alpha = float(_m.get("lora_alpha", 64.0))
        # Placeholder ids (used to keep token padding from colliding with them — see _setup_data).
        self._protein_token_id = int(_m.get("protein_token_id", 151643))
        self._go_token_id = int(_m.get("go_token_id", 151644))
        # Constant per-step batch shape (XPU 32B banned:1 fix — see _setup_data). Default ON.
        self._pad_to_fixed = bool(cfg.get("pad_to_fixed", True))
        # Optional bucketed fixed shapes (finite shape set, less padding waste than a
        # single max_seq_len shape). None => fall back to pad_to_fixed. See _setup_data.
        _buckets = cfg.get("pad_buckets", None)
        self._pad_buckets = [int(b) for b in _buckets] if _buckets else None
        # Per-bucket batch sizing (throughput lever): when set alongside pad_buckets, a
        # length-grouped distributed batch sampler gives each length bucket its own batch
        # size (bigger for short seqs) so the 65 GiB FSDP weight-gather amortizes over more
        # samples WITHOUT dropping any of the corpus. Same length/order as pad_buckets, e.g.
        # pad_buckets=[2048,4096,6144] + bucket_batch_sizes=[4,2,1]. None => uniform
        # batch_size (prior behavior). See _setup_data + LengthGroupedDistributedBatchSampler.
        _bbs = cfg.get("bucket_batch_sizes", None)
        self._bucket_batch_sizes = [int(b) for b in _bbs] if _bbs else None
        # Token PACKING (throughput lever for the ~65% GEMM floor): concatenate several
        # short examples into one fixed-max_seq_len pack (block-diagonal doc mask so docs
        # don't cross-attend) so the step carries REAL tokens instead of ~43% pad. Routes
        # attention through compiled flex (block-diag BlockMask); requires
        # TORCHTUNE_USE_XPU_FLEX=1 (flash is causal-only). Mutually exclusive with per-bucket
        # batch sizing (packing IS the amortization). See BioReasonPackedSFTDataset +
        # project_bioreason_sft_packing_scope_20260715.
        self._packing = bool(cfg.get("packing", False))
        if self._packing and self._bucket_batch_sizes is not None:
            raise ValueError(
                "packing and bucket_batch_sizes are mutually exclusive (packing already "
                "amortizes the gather by filling the sequence)."
            )
        if self._bucket_batch_sizes is not None and self._pad_buckets is None:
            raise ValueError(
                "bucket_batch_sizes requires pad_buckets (the length buckets it sizes)."
            )
        if (
            self._bucket_batch_sizes is not None
            and len(self._bucket_batch_sizes) != len(self._pad_buckets)
        ):
            raise ValueError(
                f"bucket_batch_sizes ({self._bucket_batch_sizes}) must match pad_buckets "
                f"({self._pad_buckets}) in length/order."
            )
        # Self-contained resume (decoupled from the parent's HF-checkpointer recipe_state
        # path, which is base-model-oriented and would error on this LoRA subclass). When
        # `bioreason_resume: true`, the base still loads normally from base_model_path
        # (parent resume_from_checkpoint stays effectively off for the checkpointer) and we
        # restore adapters+projections+optimizer+dataloader+step from <output_dir>/
        # resume_state.pt. Force the parent flag off so its checkpointer doesn't look for a
        # recipe_state.pt we never write.
        self._bioreason_resume = bool(cfg.get("bioreason_resume", False))
        if self._bioreason_resume:
            self._resume_from_checkpoint = False  # parent flag — keep its HF loader on base
        # Stage-2 handoff: load the Stage-1-aligned projector weights (protein_projection.pt /
        # go_projection.pt) from a Stage-1 epoch dir AFTER the meta-build re-init, so the LoRA
        # backbone fine-tunes on top of an already-aligned projector (the published two-stage
        # recipe). This is DISTINCT from bioreason_resume (same-run resume) and from the model's
        # proj_resume_dir (which is wiped by the recipe's post-meta to_empty/reset_parameters).
        # Empty/None = fresh projector (Stage 1). Set to a Stage-1 epoch_N dir for Stage 2.
        self._stage1_proj_dir = cfg.get("stage1_proj_dir", None)

    def setup(self, cfg: DictConfig) -> None:
        """Parent setup builds model/optimizer/dataloader; _setup_model already restored
        the adapters+projections on resume. Here we additionally restore the optimizer
        moments, dataloader position, and global_step/epoch from our self-contained
        resume_state (the parent's HF-checkpointer recipe-state path does not carry them
        for this multimodal-LoRA subclass)."""
        super().setup(cfg=cfg)
        blob = getattr(self, "_resume_blob", None)
        if not self._bioreason_resume or blob is None:
            return
        # Optimizer moments (trainable params only; frozen base has none).
        if training.OPT_KEY in blob and self._optimizer is not None:
            training.load_from_full_optimizer_state_dict(
                self._model, self._optimizer, blob[training.OPT_KEY], self._device
            )
            utils.log_rank_zero(self._logger, "RESUMED optimizer state.")
        # Dataloader position (StatefulDataLoader). The wrapper proxies load_state_dict.
        if blob.get("dataloader") is not None:
            try:
                self._dataloader.load_state_dict(blob["dataloader"])
                utils.log_rank_zero(self._logger, "RESUMED dataloader position.")
            except Exception as e:  # noqa: BLE001
                utils.log_rank_zero(
                    self._logger,
                    f"WARNING: could not restore dataloader position ({e}); "
                    f"continuing from dataset start (step counter still correct).",
                )
        # Training progress.
        self.global_step = int(blob.get(training.STEPS_KEY, self.global_step))
        self.epochs_run = int(blob.get(training.EPOCHS_KEY, self.epochs_run))
        utils.log_rank_zero(
            self._logger,
            f"RESUMED progress: global_step={self.global_step} epochs_run={self.epochs_run}",
        )

    # ── model: build the multimodal wrapper + shard it ────────────────────────
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
    ) -> nn.Module:
        utils.log_rank_zero(
            self._logger,
            "Instantiating BioReasonNativeModel (Gemma 4 backbone) ...",
        )
        init_start = time.perf_counter()

        # Build the wrapper on META device so the 31B backbone never materializes full
        # on one tile (mirrors the parent's meta-init + sharded-load pattern; a direct
        # on-device build would OOM before sharding). The ESM3 cache (torch.load CPU
        # dict) and GO-embedding load inside __init__ are unaffected by the default
        # device; we relocate the GO cache to the real device after sharding.
        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model: BioReasonNativeModel = config.instantiate(
                cfg_model, device=self._device, dtype=self._dtype
            )

        # Activation checkpointing on the decoder layers.
        #   - full AC (enable_activation_checkpointing=True, ac_mode=None): checkpoint
        #     every transformer layer.
        #   - selective AC (enable_activation_checkpointing=False, ac_mode="selective",
        #     ac_option=N|"op"): checkpoint every Nth layer / selective-op — trades the
        #     memory headroom we have for less recompute. Mirrors the parent recipe.
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            from torchtune.training.activations import (
                apply_selective_activation_checkpointing,
            )

            apply_selective_activation_checkpointing(
                model.backbone, ac_mode, ac_option
            )
        elif enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model.backbone,
                auto_wrap_policy={_parent_mod.modules.TransformerSelfAttentionLayer},
            )

        # Optional: swap eager RMSNorm -> fused Triton RMSNorm on XPU (gated by
        # TORCHTUNE_USE_FUSED_RMSNORM=1). MUST run before shard_model so the fused
        # module's transplanted .scale parameter is the one that gets sharded.
        # No-op (returns 0) when the flag is off or Triton is unavailable.
        from torchtune.modules._fused_rmsnorm_xpu import maybe_swap_rmsnorm_for_fused

        _n_fused = maybe_swap_rmsnorm_for_fused(model)
        if _n_fused:
            utils.log_rank_zero(
                self._logger, f"fused RMSNorm engaged: {_n_fused} modules swapped"
            )

        # Optional: fused Triton RoPE on XPU (gated by TORCHTUNE_USE_FUSED_ROPE=1).
        from torchtune.modules._fused_rope_xpu import maybe_swap_rope_for_fused

        _n_rope = maybe_swap_rope_for_fused(model)
        if _n_rope:
            utils.log_rank_zero(
                self._logger, f"fused RoPE engaged: {_n_rope} modules swapped"
            )

        # FSDP2 shard the whole wrapper (backbone + projections). The parent's
        # _loss_step/no_sync paths call set_requires_gradient_sync on self._model, so the
        # wrapper must be the fully_shard root.
        if self.parallel_dims.dp_shard_enabled or self.parallel_dims.cp_enabled:
            if self.parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_dim_names = ("dp_shard_cp",)
            training.shard_model(
                model=model,
                shard_conditions=[
                    partial(
                        training.get_shard_conditions,
                        names_to_match=custom_sharded_layers,
                    )
                ],
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=self.world_mesh[dp_mesh_dim_names],
            )

        # Initialize the meta-built non-base modules BEFORE loading the base weights
        # (canonical torchtune LoRA order, recipes/lora_finetune_distributed.py:566-584):
        #   - LoRA adapters (AdapterModule): to_empty + initialize_parameters
        #   - the from-scratch protein/GO projections (plain nn.Sequential): to_empty +
        #     reset_parameters per Linear
        #   - RoPE buffers: rope_init
        # to_empty is applied PER-MODULE (never on the whole model — that would wipe the
        # base weights once loaded; here base is still meta so order also matters).
        from torchtune.modules.peft import AdapterModule

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                if isinstance(m, AdapterModule):
                    m.to_empty(device=self._device)
                    m.initialize_parameters()
                if hasattr(m, "rope_init"):
                    m.rope_init()
            for proj in (model.protein_projection, model.go_projection):
                proj.to_empty(device=self._device)
                for layer in proj:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

        # Load the GEMMA4 base weights into the sharded model via the FSDP2-aware loader
        # (full CPU state dict -> sharded DTensors per rank, no full materialization).
        # strict=False: adapters/projections are not in the base checkpoint (already
        # initialized above); keys are backbone.* in the wrapper, so prefix them.
        _prefixed = {f"backbone.{k}": v for k, v in model_state_dict.items()}
        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            _prefixed,
            self._device,
            strict=False,
            cpu_offload=fsdp_cpu_offload,
        )
        _real_unexpected = list(base_unexpected or [])
        if _real_unexpected:
            raise RuntimeError(
                f"Base load: {len(_real_unexpected)} UNEXPECTED keys "
                f"(first few: {_real_unexpected[:5]}). Checkpoint/arch mismatch."
            )
        utils.log_rank_zero(
            self._logger,
            f"Loaded GEMMA4 base ({len(model_state_dict)} tensors) into sharded backbone.",
        )

        # RESUME: overwrite the freshly-initialized adapters + projections with the saved
        # trainable state (frozen base already loaded above; optimizer/step/dataloader are
        # restored in setup()). Without this, resume_from_checkpoint would silently restart
        # the LoRA adapters from scratch and lose all training progress.
        if self._bioreason_resume:
            rpath = self._resume_state_path()
            if not os.path.exists(rpath):
                raise FileNotFoundError(
                    f"bioreason_resume=True but no resume_state at {rpath}. "
                    f"(Eval checkpoints alone cannot resume training.)"
                )
            self._resume_blob = torch.load(rpath, map_location="cpu", weights_only=False)
            trainable = self._resume_blob["trainable"]  # stripped keys
            # Re-prefix to match the wrapper's (unwrapped) module names and load as a full
            # state dict into the sharded model (DTensor-aware), strict=False (base+buffers
            # are not in this subset).
            missing, unexpected = training.load_from_full_model_state_dict(
                model, trainable, self._device, strict=False,
                cpu_offload=fsdp_cpu_offload,
            )
            if unexpected:
                raise RuntimeError(
                    f"resume trainable load: {len(unexpected)} unexpected keys "
                    f"(first: {list(unexpected)[:5]}) — adapter/projection name mismatch."
                )
            utils.log_rank_zero(
                self._logger,
                f"RESUMED {len(trainable)} trainable tensors (adapters+projections) "
                f"from {rpath} (step={self._resume_blob.get(training.STEPS_KEY)}).",
            )

        # STAGE-2 HANDOFF: load the Stage-1-aligned projector. Must happen AFTER the
        # post-meta to_empty/reset_parameters (which would otherwise wipe it) and is skipped
        # when resuming the same run (bioreason_resume already restored the projector). The
        # projections are sharded DTensors here, so load the full CPU state dict via the
        # FSDP2-aware loader under the wrapper-relative keys (protein_projection.* / go_*).
        if self._stage1_proj_dir and not self._bioreason_resume:
            proj_full = {}
            for pname in ("protein_projection", "go_projection"):
                ppath = os.path.join(self._stage1_proj_dir, f"{pname}.pt")
                if not os.path.exists(ppath):
                    raise FileNotFoundError(
                        f"stage1_proj_dir set but {ppath} missing — point it at a Stage-1 "
                        f"epoch_N dir holding protein_projection.pt/go_projection.pt."
                    )
                sub = torch.load(ppath, map_location="cpu")
                for k, v in sub.items():
                    proj_full[f"{pname}.{k}"] = v
            missing, unexpected = training.load_from_full_model_state_dict(
                model, proj_full, self._device, strict=False,
                cpu_offload=fsdp_cpu_offload,
            )
            if unexpected:
                raise RuntimeError(
                    f"Stage-1 projector load: {len(unexpected)} unexpected keys "
                    f"(first: {list(unexpected)[:5]}) — projection name mismatch."
                )
            utils.log_rank_zero(
                self._logger,
                f"STAGE-2: loaded {len(proj_full)} aligned projector tensors from "
                f"{self._stage1_proj_dir}.",
            )

        # GO embedding cache must be a real tensor (loaded with map_location=device in
        # __init__, so unaffected by the meta context).
        if (
            "all" in model._go_embed_cache
            and model._go_embed_cache["all"].device.type == "meta"
        ):
            raise RuntimeError(
                "GO embedding cache landed on meta — set go_embedding_path to a real file."
            )

        # Optional per-layer torch.compile of the backbone decoder layers (config
        # `compile: {model: true}`). The parent's _setup_model calls training.compile_model,
        # but this subclass reimplements _setup_model, so it must be invoked here too.
        # compile_model targets only TransformerSelfAttentionLayer modules (inside
        # model.backbone) — the multimodal splice (BioReasonNativeModel.forward /
        # _splice_embeds) is NOT a layer module and stays eager, so the dynamic Python
        # ESM3-cache loop + scatter/clone are never traced. The MFU profile (2026-07-09)
        # showed 26% of the step in un-fused elementwise (norms/activations/casts) +
        # slow softmax; per-layer compile fuses those epilogues. Default OFF
        # (compile:false -> _compile_model=False -> no-op; all prior runs unaffected).
        # dynamic follows the parent's _compile_dynamic (default True on XPU).
        if getattr(self, "_compile_model", False):
            utils.log_rank_zero(
                self._logger,
                "Compiling backbone decoder layers (per-layer torch.compile, dynamic=%s)."
                % getattr(self, "_compile_dynamic", True),
            )
            training.compile_model(
                model, verbose=self._is_rank_zero,
                dynamic=getattr(self, "_compile_dynamic", True),
            )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading, activation_offloading_use_streams
        )
        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            rotate_method=self.context_parallel_rotate_method,
            world_mesh=self.world_mesh,
            model=model,
        )
        self.train_context = training.get_train_context(
            enable_loss_parallel=self.use_loss_parallel_ctx_manager,
        )

        training.validate_no_params_on_meta_device(model)
        utils.log_rank_zero(
            self._logger,
            f"Model setup took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            mem_stats = training.get_memory_stats(device=self._device)
            if mem_stats:
                training.log_memory_stats(mem_stats)

        if self._device.type == "xpu":
            torch.distributed.barrier()
        else:
            torch.distributed.barrier(device_ids=[self._device.index])
        return model

    # ── data: stash string side inputs, yield tensor-only batches ─────────────
    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> _SideInputDataLoader:
        from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

        ds = config.instantiate(cfg_dataset, self._tokenizer)
        collate = _get_component_from_path(collate_fn)
        # Token packing: wrap the base dataset in the multimodal-aware packer and swap in the
        # packed collate. Every pack is a fixed max_seq_len row (banned:1-safe); the sampler
        # runs over PACKS. Must precede the sampler build (sampler length = number of packs).
        if getattr(self, "_packing", False):
            from torchtune.dev.bioreason.dataset_sft import (
                BioReasonPackedSFTDataset,
                bioreason_sft_packed_collate_fn,
            )

            _fixed = int(cfg_dataset.get("max_seq_len", 0)) or None
            if _fixed is None:
                raise ValueError("packing requires dataset.max_seq_len set.")
            _tok_pad = self._tokenizer.pad_id
            if _tok_pad in (self._protein_token_id, self._go_token_id):
                _tok_pad = 0
            _t0 = time.perf_counter()
            ds = BioReasonPackedSFTDataset(
                ds,
                max_seq_len=_fixed,
                padding_idx=_tok_pad,
                ignore_idx=self._loss_fn.ignore_index,
            )
            collate = bioreason_sft_packed_collate_fn
            utils.log_rank_zero(
                self._logger,
                "Token PACKING ENABLED: %d examples -> %d packs (fixed seq=%d, block-diag "
                "flex attn) [plan %.1fs]."
                % (len(ds.ds), len(ds), _fixed, time.perf_counter() - _t0),
            )
        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=shuffle, seed=0
        )
        # Per-bucket batch sizing (throughput): a length-grouped distributed BATCH sampler
        # replaces the uniform (sampler + batch_size) pair. Each length bucket gets its own
        # batch size so short seqs amortize the FSDP gather over more samples, over the FULL
        # corpus. Requires pad_buckets (the length bins) + bucket_batch_sizes (per-bin bs).
        batch_bucket_sizes = getattr(self, "_bucket_batch_sizes", None)
        _pad_buckets_cfg = getattr(self, "_pad_buckets", None)
        # Constant per-step shape (XPU 32B stability — isolation sweep proved fixed-shape
        # seq=4096 trains clean while variable LARGE shapes churn VAs -> banned:1).
        # Pads every batch to the dataset's max_seq_len. Default ON for this recipe.
        pad_fixed = bool(getattr(self, "_pad_to_fixed", True))
        fixed_len = int(cfg_dataset.get("max_seq_len", 0)) or None
        # Bucketed fixed shapes: keep the per-step shape set FINITE (preserves the
        # banned:1 fix) but pad to the smallest bucket that holds the batch instead of
        # always to max_seq_len. Cuts the ~71% padding waste at p50~1770 / seq6144.
        pad_buckets = getattr(self, "_pad_buckets", None)
        # CRITICAL: token pad id MUST NOT collide with the protein/GO placeholder ids.
        # On Qwen3 tokenizer.pad_id == 151643 == protein_token_id, so padding tokens with
        # pad_id turns every pad position into a protein placeholder -> the splice's
        # placeholder count inflates by the pad amount (differs per rank) -> "Protein
        # token count N != features M" crash. Pad positions are masked in labels and are
        # right-side (causal: never attended by real tokens), so a neutral id 0 is safe.
        tok_pad = self._tokenizer.pad_id
        if tok_pad in (self._protein_token_id, self._go_token_id):
            tok_pad = 0
        collate_partial = partial(
            collate,
            padding_idx=tok_pad,
            ignore_idx=self._loss_fn.ignore_index,
            max_seq_len=fixed_len,
            pad_to_fixed=pad_fixed,
            pad_buckets=pad_buckets,
        )
        common_kwargs = dict(
            dataset=ds,
            collate_fn=collate_partial,
            num_workers=self._dataloader_num_workers,
            pin_memory=self._dataloader_pin_memory,
            **(
                {
                    "persistent_workers": True,
                    "prefetch_factor": self._dataloader_prefetch_factor,
                }
                if self._dataloader_num_workers > 0
                else {}
            ),
        )
        epoch_sampler = sampler
        if batch_bucket_sizes is not None:
            # Per-bucket batch sizing: length-grouped distributed BATCH sampler. Buckets
            # come from pad_buckets; the collate's own pad_buckets then pads each
            # homogeneous-length batch to its own bucket. batch_size / drop_last are
            # subsumed by the batch_sampler (mutually exclusive with them in DataLoader).
            from torchtune.dev.bioreason.dataset_sft import (
                LengthGroupedDistributedBatchSampler,
            )

            if fixed_len is None:
                raise ValueError("bucket_batch_sizes requires dataset.max_seq_len set.")
            buckets = sorted(int(b) for b in _pad_buckets_cfg if int(b) <= fixed_len)
            if not buckets or buckets[-1] < fixed_len:
                buckets.append(fixed_len)  # top bucket must hold every (capped) length
            # bucket_batch_sizes aligns with the CONFIGURED pad_buckets order; re-map onto
            # the clamped/appended `buckets` list by ceiling value (defaults to 1).
            cfg_map = {int(b): int(s) for b, s in zip(_pad_buckets_cfg, batch_bucket_sizes)}
            bbs = [cfg_map.get(b, 1) for b in buckets]
            t0 = time.perf_counter()
            lengths = ds.compute_lengths()
            batch_sampler = LengthGroupedDistributedBatchSampler(
                lengths=lengths,
                buckets=buckets,
                bucket_batch_sizes=bbs,
                num_replicas=self.dp_degree,
                rank=self.dp_rank,
                shuffle=shuffle,
                seed=0,
            )
            epoch_sampler = batch_sampler
            utils.log_rank_zero(
                self._logger,
                "Per-bucket batch sizing ENABLED: buckets=%s batch_sizes=%s -> %d batches/"
                "rank/epoch (length scan %.1fs over %d examples)."
                % (buckets, bbs, len(batch_sampler), time.perf_counter() - t0, len(ds)),
            )
            dl = StatefulDataLoader(batch_sampler=batch_sampler, **common_kwargs)
        else:
            dl = StatefulDataLoader(
                batch_size=batch_size, sampler=sampler, drop_last=True, **common_kwargs
            )
        if dataloader_state_dict is not None:
            dl.load_state_dict(dataloader_state_dict)
        return _SideInputDataLoader(dl, self, epoch_sampler=epoch_sampler)

    # ── checkpoint: merged backbone (HF) + projections ────────────────────────
    # ── resumable training state (self-contained; frozen-base LoRA) ───────────
    # The base never trains, so a resumable checkpoint only needs the TRAINABLE state:
    # LoRA adapters + the from-scratch protein/GO projections + optimizer moments +
    # dataloader position + global_step/epoch. The base reloads from base_model_path as
    # always. This is separate from save_checkpoint()'s eval artifact (merged W_eff HF
    # safetensors), which cannot resume training (optimizer/step/adapters folded away).
    # Written to <output_dir>/resume_state.pt (rank0 only); loaded when
    # resume_from_checkpoint=True. See docs/reports/bioreason_sft_oom_diagnosis_20260627.md.
    def _resume_state_path(self) -> str:
        return os.path.join(self._output_dir, "resume_state.pt")

    @staticmethod
    def _trainable_keys(sd: dict) -> dict:
        """Adapter (lora_a/lora_b/magnitude) + projection params, stripped of FSDP/AC
        wrapper prefixes. These are exactly the params with requires_grad=True."""
        def strip(name):
            return (name.replace("_fsdp_wrapped_module.", "")
                        .replace("_checkpoint_wrapped_module.", ""))
        out = {}
        for k, v in sd.items():
            ck = strip(k)
            if (("lora_a" in ck) or ("lora_b" in ck) or ("magnitude" in ck)
                    or ck.startswith("protein_projection.")
                    or ck.startswith("go_projection.")):
                out[ck] = v
        return out

    def _save_resume_state(self, *, epoch: int) -> None:
        """Persist trainable + optimizer + dataloader + progress for exact resume."""
        full_sd = training.gather_cpu_state_dict(
            self._model, self._is_rank_zero, device=self._device
        )
        # Optimizer full state dict (trainable params only — frozen base has no moments).
        opt_sd = training.get_full_optimizer_state_dict(
            self._model, self._optimizer, self._is_rank_zero, device=self._device
        )
        if not self._is_rank_zero:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return
        state = {
            "trainable": self._trainable_keys(full_sd),
            training.OPT_KEY: opt_sd,
            training.STEPS_KEY: self.global_step,
            training.EPOCHS_KEY: epoch,
            "dataloader": self._dataloader.state_dict(),
            "lora_rank": self._lora_rank,
            "lora_alpha": self._lora_alpha,
        }
        path = self._resume_state_path()
        os.makedirs(self._output_dir, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        utils.log_rank_zero(
            self._logger,
            f"Saved resume_state (trainable+opt+dataloader, step={self.global_step}) -> {path}",
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def save_checkpoint(self, *, epoch: int, full_tensors: bool) -> None:
        """Save BOTH a resumable training state AND the eval merged-backbone checkpoint."""
        # 1) resumable trainable state (so resume_from_checkpoint can continue exactly).
        self._save_resume_state(epoch=epoch)
        # 2) the eval artifact (merged W_eff + projections), as before.
        self._save_eval_checkpoint(epoch=epoch, full_tensors=full_tensors)

    def _save_eval_checkpoint(self, *, epoch: int, full_tensors: bool) -> None:
        """Save a self-contained Gemma4 checkpoint for eval.

        The parent's CheckpointClient path assumes a bare TransformerDecoder; our
        wrapper has backbone.* + projection.* and (under LoRA) lora_a/lora_b. So we
        gather the full state dict, MERGE LoRA into the backbone (W_eff), and write:
          - the merged backbone via the configured GEMMA4 checkpointer (-> HF
            safetensors, loadable by the vLLM eval), and
          - protein_projection.pt / go_projection.pt alongside it.
        """
        from torchtune.dev.bioreason.model_native import BioReasonNativeModel

        full_sd = training.gather_cpu_state_dict(
            self._model, self._is_rank_zero, device=self._device
        )
        if not self._is_rank_zero:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        # Merge LoRA -> bare backbone tune-format state dict.
        if self._model._has_lora:
            backbone_sd = self._model.merged_backbone_for_save(
                full_sd, lora_rank=self._lora_rank, lora_alpha=self._lora_alpha
            )
        else:
            backbone_sd = BioReasonNativeModel.merge_backbone_state_dict(full_sd)

        checkpointer = self._checkpoint_client._get_checkpointer()
        checkpointer.save_checkpoint(
            {training.MODEL_KEY: backbone_sd},
            epoch=epoch,
        )

        # Projections (stripped) alongside the HF backbone output dir.
        out_dir = os.path.join(checkpointer._output_dir, f"epoch_{epoch}") \
            if hasattr(checkpointer, "_output_dir") else self._output_dir

        def _strip(name):
            return (name.replace("_fsdp_wrapped_module.", "")
                        .replace("_checkpoint_wrapped_module.", ""))

        os.makedirs(out_dir, exist_ok=True)
        for pname in ("protein_projection", "go_projection"):
            sub = {}
            for k, v in full_sd.items():
                ck = _strip(k)
                if ck.startswith(pname + "."):
                    sub[ck[len(pname) + 1:]] = v.detach().clone()
            if sub:
                torch.save(sub, os.path.join(out_dir, f"{pname}.pt"))
        utils.log_rank_zero(
            self._logger,
            f"Saved BioReason SFT checkpoint (merged backbone + projections) to {out_dir}",
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ── loss: native decoder forward (splice runs INSIDE forward) ─────────────
    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch.pop("labels")
        tokens = batch["tokens"]
        side = self._current_side_inputs
        protein_sequences = side.get("protein_sequences", [])
        go_aspects = side.get("go_aspects", None)

        # Token-packing extras (present only when packing=true): per-doc position ids, the
        # block-diagonal mask input (per-row doc lengths), and the doc->row map for the
        # splice. Build the flex BlockMask here (doc A must not attend doc B).
        pack_mask = None
        input_pos = None
        batch_idx_map = None
        if getattr(self, "_packing", False):
            from torchtune.modules.attention_utils import xpu_packed_block_causal_mask

            input_pos = side.get("input_pos")
            if input_pos is not None:
                input_pos = input_pos.to(self._device)
            batch_idx_map = side.get("batch_idx_map")
            seq_lens = side.get("seq_lens")
            if seq_lens is not None:
                pack_mask = xpu_packed_block_causal_mask(
                    seq_lens, tokens.shape[1], self._device
                )

        # The embed-splice MUST run inside model.forward (under the root FSDP forward
        # hook) so tok_embeddings.weight is unsharded — building embeds out here leaves
        # it a DTensor and aten.embedding errors on mixed Tensor/DTensor.
        _timing = os.environ.get("TORCHTUNE_BIOREASON_TIMING") == "1"
        if _timing and self._is_rank_zero:
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _fwd_t0 = time.perf_counter()

        with self.activations_handling_ctx:
            outputs = self._model(
                tokens,
                protein_sequences=protein_sequences,
                go_aspects=go_aspects,
                batch_idx_map=batch_idx_map,
                mask=pack_mask,
                input_pos=input_pos,
            )

        if _timing and self._is_rank_zero:
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            self._logger.info(
                "[bioreason-timing] fwd (incl splice) %.3fs bs=%d seq=%d"
                % (time.perf_counter() - _fwd_t0, tokens.shape[0], tokens.shape[1])
            )

        # SFTLoss (chunked) consumes the list of logit chunks directly. Non-SFTLoss
        # (plain CE) needs flat [N, vocab] / [N] — mirror the parent.
        if not isinstance(self._loss_fn, SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
        loss = self._loss_fn(outputs, labels)
        del outputs
        return loss


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point. Mirrors the parent's recipe_main but constructs the BioReason SFT
    subclass."""
    config.log_config(recipe_name="BioReasonSFTRecipeDistributedXPU", cfg=cfg)
    recipe = BioReasonSFTRecipeDistributedXPU(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

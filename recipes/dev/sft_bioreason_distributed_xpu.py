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

    SIDE_KEYS = ("protein_sequences", "go_aspects")

    def __init__(self, dl: StatefulDataLoader, recipe: "BioReasonSFTRecipeDistributedXPU"):
        self._dl = dl
        self._recipe = recipe

    @property
    def sampler(self):
        return self._dl.sampler

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
        super().__init__(cfg)
        self._current_side_inputs: dict = {}
        # Stash the LoRA rank/alpha for checkpoint-time merge (parent doesn't keep cfg).
        _m = cfg.get("model", {})
        self._lora_rank = int(_m.get("lora_rank", 32))
        self._lora_alpha = float(_m.get("lora_alpha", 64.0))

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

        # Activation checkpointing on the decoder layers (wrap by the transformer
        # self-attention layer type, which subsumes Gemma4TransformerLayer).
        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model.backbone,
                auto_wrap_policy={_parent_mod.modules.TransformerSelfAttentionLayer},
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

        # GO embedding cache must be a real tensor (loaded with map_location=device in
        # __init__, so unaffected by the meta context).
        if (
            "all" in model._go_embed_cache
            and model._go_embed_cache["all"].device.type == "meta"
        ):
            raise RuntimeError(
                "GO embedding cache landed on meta — set go_embedding_path to a real file."
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
        sampler = StatefulDistributedSampler(
            ds, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=shuffle, seed=0
        )
        dl = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(
                collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            ),
            drop_last=True,
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
        if dataloader_state_dict is not None:
            dl.load_state_dict(dataloader_state_dict)
        return _SideInputDataLoader(dl, self)

    # ── checkpoint: merged backbone (HF) + projections ────────────────────────
    def save_checkpoint(self, *, epoch: int, full_tensors: bool) -> None:
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

    # ── loss: splice multimodal embeds (grad on) + native decoder forward ─────
    def _loss_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch.pop("labels")
        tokens = batch["tokens"]
        side = self._current_side_inputs
        protein_sequences = side.get("protein_sequences", [])
        go_aspects = side.get("go_aspects", None)

        with self.activations_handling_ctx:
            input_embeds = self._model.build_full_embeds_train(
                tokens,
                protein_sequences=protein_sequences,
                go_aspects=go_aspects,
            )
            outputs = self._model(input_embeds=input_embeds)

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

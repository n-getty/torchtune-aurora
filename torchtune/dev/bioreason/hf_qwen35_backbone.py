"""
HF-wrapper adapter for the Qwen3.5-family hybrid linear-attention backbone
(``model_type: qwen3_5``, HF class ``Qwen3_5ForCausalLM``) — Qwen3.6-27B.

Unlike :mod:`torchtune.dev.bioreason.model_native` (native torchtune
``TransformerDecoder``: Gemma4/Qwen3), this backbone is architecturally a hybrid of
48 Gated-DeltaNet linear-attention layers + 16 full-attention layers (interval-4
pattern), with no native torchtune port. Porting the architecture natively was
explicitly rejected (see the Qwen3.6-27B integration plan) in favor of wrapping the
real HF ``Qwen3_5ForCausalLM`` as-is behind a thin adapter exposing the same small
attribute/method contract :class:`~torchtune.dev.bioreason.model_native.BioReasonNativeModel`
expects from its backbone (``tok_embeddings``, ``output``/``skip_output_layer``,
``forward(tokens=..., input_embeds=...)``, ``set_num_output_chunks``).

Checkpoint layout note: the real Qwen3.6-27B checkpoint on disk uses the
multimodal convention — weights under ``model.language_model.*`` (text backbone),
``model.visual.*`` (vision tower, unused — text-only splice), ``mtp.*``
(multi-token-prediction head, unused), and a top-level ``lm_head.weight``
(untied: ``tie_word_embeddings: false``). :func:`remap_qwen35_checkpoint_keys`
converts this into the flat ``Qwen3_5ForCausalLM`` state-dict layout
(``model.*`` / ``lm_head.weight``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Shared with model_native.py's qwen3_5_27b_hf + enable_lora branch (construction)
# and sft_bioreason_distributed_xpu.py's eval-checkpoint save (LoRA merge) — a single
# source of truth so the two never drift apart. "conv1d" is appended separately by
# callers when include_conv1d_lora=True (see model_native.py's docstring for why it's
# opt-in: PEFT's Conv1d LoRA layer requires rank % groups == 0, unsatisfiable at
# practical ranks against this module's groups=128).
HF_QWEN35_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]


class HFQwen35Backbone(nn.Module):
    """Thin adapter wrapping a real HF ``Qwen3_5ForCausalLM``.

    Args:
        config_path (str): path to the checkpoint's ``config.json`` (the top-level
            multimodal config; only its ``text_config`` sub-dict is used).
        dtype (torch.dtype): parameter dtype. Default: bfloat16.
        skip_init_weights (bool): if True, skip HF's ``post_init()`` weight
            initialization (real weights always overwrite it downstream via
            ``load_qwen35_safetensors`` + ``load_from_full_model_state_dict``).
            Kept as an escape hatch, but Step-0 validation confirmed
            ``Qwen3_5ForCausalLM`` constructs cleanly under ``torch.device("meta")``
            (the flagged ``A_log`` in-place-uniform-fill in
            ``Qwen3_5GatedDeltaNet._init_weights`` does not raise on meta tensors),
            so the default here is False — it is not needed for meta-device FSDP2
            builds. Default: False.
    """

    def __init__(
        self,
        *,
        config_path: str,
        dtype: torch.dtype = torch.bfloat16,
        skip_init_weights: bool = False,
    ):
        super().__init__()
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForCausalLM,
        )

        with open(config_path) as f:
            full_config = json.load(f)
        text_config = dict(full_config["text_config"])
        text_config["use_cache"] = False
        config = Qwen3_5TextConfig(**text_config)

        if skip_init_weights:
            # HF's PreTrainedModel.__init__ calls post_init() -> init_weights()
            # unconditionally. Real weights always overwrite this via a full
            # state-dict load downstream, so the init pass is pure waste (and,
            # per the constructor's docstring, was a pre-emptive workaround for a
            # meta-device failure mode that Step-0 validation showed does not
            # actually occur — this path is untested against that; only use it if
            # a future HF/transformers version regresses the meta-device case).
            _orig_init_weights = Qwen3_5ForCausalLM.init_weights
            Qwen3_5ForCausalLM.init_weights = lambda self: None
            try:
                self.model = Qwen3_5ForCausalLM(config)
            finally:
                Qwen3_5ForCausalLM.init_weights = _orig_init_weights
        else:
            self.model = Qwen3_5ForCausalLM(config)

        self.model = self.model.to(dtype)
        self.config = config
        self.skip_output_layer: bool = False
        self._num_output_chunks = 0

    @property
    def _causal_lm(self):
        """The underlying ``Qwen3_5ForCausalLM``, unwrapped from any PEFT adapter.

        ``self.model`` is reassigned in-place to a ``PeftModel`` when LoRA is
        applied (see ``model_native.py``'s Stage-2 wiring). PEFT's own
        ``__getattr__`` delegation chain (``PeftModel`` -> ``LoraModel`` ->
        wrapped model) does NOT line up with this adapter's plain-model attribute
        paths (e.g. ``peft_model.model`` resolves to the CausalLM directly, one
        traversal short of ``plain_model.model`` == the inner ``Qwen3_5TextModel``)
        — using PEFT's own ``get_base_model()`` sidesteps that mismatch instead of
        hand-deriving the nesting depth. Note the LoRA-injected Linear layers live
        IN-PLACE inside this same returned object (PEFT swaps submodules on the
        original model, it does not copy the graph), so driving forward directly
        through the unwrapped CausalLM still runs the LoRA-adapted path.
        """
        model = self.model
        if hasattr(model, "get_base_model"):
            return model.get_base_model()
        return model

    @property
    def tok_embeddings(self) -> nn.Embedding:
        return self._causal_lm.model.embed_tokens

    @property
    def output(self) -> nn.Linear:
        return self._causal_lm.lm_head

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        # No-op: HF's lm_head is a single dense Linear; there is no native
        # chunked-logits path to configure. LinearCrossEntropyLoss drives the
        # projection itself via skip_output_layer + .output, which this adapter
        # already supports — chunk count is irrelevant here.
        self._num_output_chunks = num_output_chunks

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Matches BioReasonNativeModel's backbone call signature.

        ``encoder_input``/``encoder_mask``/``input_pos`` are accepted for
        signature compatibility but unused (no cross-attention / native rope
        cache in this backbone — HF computes its own RoPE internally).
        """
        if mask is not None:
            raise NotImplementedError(
                "HFQwen35Backbone.forward: explicit attention masks (packing) are "
                "not supported in v1 — got mask is not None. Use unpacked, "
                "single-document batches."
            )
        causal_lm = self._causal_lm
        hidden_states = causal_lm.model(
            input_ids=tokens if input_embeds is None else None,
            inputs_embeds=input_embeds,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
        ).last_hidden_state
        if self.skip_output_layer:
            return hidden_states
        return causal_lm.lm_head(hidden_states)


def remap_qwen35_checkpoint_keys(raw_sd: dict) -> dict:
    """Remap a real Qwen3.6-27B checkpoint's multimodal key layout to the flat
    ``Qwen3_5ForCausalLM`` state-dict layout.

    Drops ``mtp.*`` (multi-token-prediction head) and ``model.visual.*`` (vision
    tower) — both unused by the text-only BioReason splice. Strips the
    ``model.language_model.`` prefix to ``model.``. Passes ``lm_head.weight``
    through unchanged (already top-level, untied). Raises on any key matching
    none of these — a real Qwen3.6-27B checkpoint should have every key sorted by
    one of these three fates; an unrecognized layout is silently ignoring
    weights rather than something to warn past.
    """
    out: dict = {}
    for k, v in raw_sd.items():
        if k.startswith("mtp.") or k.startswith("model.visual."):
            continue
        elif k.startswith("model.language_model."):
            out["model." + k[len("model.language_model.") :]] = v
        elif k == "lm_head.weight":
            out[k] = v
        else:
            raise ValueError(
                f"remap_qwen35_checkpoint_keys: unrecognized checkpoint key {k!r} "
                "(expected a prefix of 'mtp.', 'model.visual.', "
                "'model.language_model.', or exactly 'lm_head.weight')."
            )
    return out


def load_qwen35_safetensors(checkpoint_dir: str) -> dict:
    """Load a real Qwen3.6-27B checkpoint's sharded safetensors into a single CPU
    state dict, keyed by the RAW (multimodal) checkpoint layout — pass the result
    through :func:`remap_qwen35_checkpoint_keys` before loading into
    ``HFQwen35Backbone.model``.
    """
    from safetensors.torch import load_file

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_names = sorted(set(index["weight_map"].values()))

    merged: dict = {}
    for shard_name in shard_names:
        shard_path = os.path.join(checkpoint_dir, shard_name)
        shard = load_file(shard_path, device="cpu")
        merged.update(shard)
    return merged


def save_qwen35_checkpoint(backbone: HFQwen35Backbone, output_dir: str) -> None:
    """Save ``backbone.model`` (a real ``Qwen3_5ForCausalLM``, or a merged copy of
    one — see ``merge_peft_lora_state_dict`` for the LoRA/Stage-2 case) via HF's
    own sharded-safetensors writer, so the result is directly loadable by
    ``AutoModelForCausalLM.from_pretrained`` / vLLM for eval.
    """
    backbone.model.save_pretrained(output_dir, safe_serialization=True)


def merge_peft_lora_state_dict(
    backbone_state_dict: dict,
    config_path: str,
    *,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    target_modules: list[str],
    dtype: torch.dtype = torch.bfloat16,
) -> "Qwen3_5ForCausalLM":  # noqa: F821 - imported lazily below
    """Reconstruct a fresh CPU ``Qwen3_5ForCausalLM`` + PEFT wrapper matching the
    exact training-time ``LoraConfig``, load a gathered (base+adapter) state dict
    into it, and merge the adapters via PEFT's own ``merge_and_unload()``.

    Deliberately reuses PEFT's tested merge implementation rather than
    hand-deriving its internal key-nesting convention (``base_model.model.*.
    lora_A/lora_B`` vs. a hand-rolled scheme) — see the integration plan's
    checkpoint-save section for why this was preferred over reimplementing the
    ``W_eff = W_base + scale * (B @ A)`` merge directly.

    Args:
        backbone_state_dict: gathered CPU state dict with keys matching the
            PEFT-wrapped module's own naming (i.e. exactly what
            ``get_peft_model(Qwen3_5ForCausalLM(...), lora_config).state_dict()``
            would produce) — NOT the raw HF layout.

    Returns:
        The merged, plain (no adapter) ``Qwen3_5ForCausalLM`` — pass directly to
        :func:`save_qwen35_checkpoint`.
    """
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    with open(config_path) as f:
        full_config = json.load(f)
    text_config = dict(full_config["text_config"])
    text_config["use_cache"] = False
    config = Qwen3_5TextConfig(**text_config)

    base_model = Qwen3_5ForCausalLM(config).to(dtype)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)
    missing, unexpected = peft_model.load_state_dict(backbone_state_dict, strict=False)
    if unexpected:
        raise ValueError(
            f"merge_peft_lora_state_dict: {len(unexpected)} unexpected keys not "
            f"matched in the reconstructed PEFT model, e.g. {unexpected[:5]}"
        )
    merged = peft_model.merge_and_unload()
    return merged

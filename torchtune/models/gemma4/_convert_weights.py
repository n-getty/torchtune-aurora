# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtune.models.convert_weights import get_mapped_key

"""
Weight conversion for Gemma 4 models.

Key differences from Gemma 2:
- HF prefix is "model.language_model." (multimodal wrapper), not "model."
- Global attention layers have NO v_proj (k_eq_v is structural)
- q_norm and k_norm per layer
- layer_scalar per layer (post-residual scaling)
- Uses Qwen2-style RoPE (no weight permutation needed)
"""

_GEMMA4_FROM_HF = {
    "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
    "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.sa_scale.scale",
    "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.mlp_scale.scale",
    "model.language_model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
    "model.language_model.norm.weight": "norm.rms_norm.scale",
}

# Keys to skip entirely (vision encoder, embedding projection, etc.)
_SKIP_PREFIXES = [
    "model.vision_tower.",
    "model.embed_vision.",
    "model.multi_modal_projector.",
]


def gemma4_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 16,
    dim: int = 5376,
    head_dim: int = None,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to torchtune's format.

    Handles the multimodal Gemma4 checkpoint by:
    - Extracting only language model weights (skipping vision encoder)
    - Mapping the "model.language_model." prefix to torchtune's flat format
    - Skipping v_proj on global layers (they don't have it)
    - No weight permutation (Qwen2-style RoPE matches HF layout)

    Args:
        state_dict: State dict in HF's format.
        num_heads: Number of query heads (unused, kept for interface consistency).
        num_kv_heads: Number of KV heads (unused, kept for interface consistency).
        dim: Model dimension (unused, kept for interface consistency).
        head_dim: Head dimension (unused, kept for interface consistency).

    Returns:
        State dict in torchtune's format.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        # Skip vision tower and multimodal projection weights
        if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        # Skip rotary embedding inverse frequency
        if "rotary_emb.inv_freq" in key:
            continue
        new_key = get_mapped_key(key, _GEMMA4_FROM_HF)
        converted_state_dict[new_key] = value
    return converted_state_dict


def gemma4_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 16,
    dim: int = 5376,
    head_dim: int = None,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to HF's format.

    Args:
        state_dict: State dict in torchtune's format.
        num_heads: Number of query heads (unused, kept for interface consistency).
        num_kv_heads: Number of KV heads (unused, kept for interface consistency).
        dim: Model dimension (unused, kept for interface consistency).
        head_dim: Head dimension (unused, kept for interface consistency).

    Returns:
        State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _GEMMA4_FROM_HF.items() if v is not None}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

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
    # --- 26B-A4B MoE additions ---
    # Extra per-layer norms for additive MoE path
    "model.language_model.layers.{}.post_feedforward_layernorm_1.weight": "layers.{}.post_mlp_norm.scale",
    "model.language_model.layers.{}.pre_feedforward_layernorm_2.weight": "layers.{}.pre_moe_norm.scale",
    "model.language_model.layers.{}.post_feedforward_layernorm_2.weight": "layers.{}.post_moe_norm.scale",
    # Router weights — HF stores directly under layer (no "moe_block." prefix)
    # router.norm has no weight in HF checkpoint (with_scale=False); torchtune default (zeros=identity) is used.
    "model.language_model.layers.{}.router.proj.weight": "layers.{}.moe_block.router.proj.weight",
    "model.language_model.layers.{}.router.scale": "layers.{}.moe_block.router.scale",
    "model.language_model.layers.{}.router.per_expert_scale": "layers.{}.moe_block.router.per_expert_scale",
    # Expert down_proj: HF [E, hidden=2816, intermediate=704] → GroupedExperts [E, intermediate=704, hidden=2816]
    # Transposed in gemma4_hf_to_tune(); entry here signals it should be converted (not skipped).
    "model.language_model.layers.{}.experts.down_proj": "layers.{}.moe_block.experts.down_proj",
    # experts.gate_up_proj is handled specially in gemma4_hf_to_tune() — no entry needed here.
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

        # Special case: fused gate_up_proj [E, 2*intermediate, hidden] → split + transpose
        # HF key: "model.language_model.layers.N.experts.gate_up_proj" [128, 1408, 2816]
        # GroupedExperts: gate_proj [E, hidden, moe_intermediate], up_proj [E, hidden, moe_intermediate]
        if key.endswith(".experts.gate_up_proj"):
            m = re.search(r"layers\.(\d+)\.", key)
            if m is None:
                raise ValueError(f"Could not extract layer index from key: {key}")
            layer_idx = m.group(1)
            gate_hf, up_hf = value.chunk(2, dim=1)   # each [E, moe_intermediate, hidden]
            converted_state_dict[f"layers.{layer_idx}.moe_block.experts.gate_proj"] = gate_hf.transpose(1, 2).contiguous()
            converted_state_dict[f"layers.{layer_idx}.moe_block.experts.up_proj"] = up_hf.transpose(1, 2).contiguous()
            continue

        # Special case: down_proj transpose
        # HF: [E, hidden=2816, moe_intermediate=704] → GroupedExperts: [E, moe_intermediate=704, hidden=2816]
        # (matmul in _forward_no_grouped_mm: h @ w2 where h=[T, moe_intermediate], w2=[moe_intermediate, hidden])
        if key.endswith(".experts.down_proj"):
            m = re.search(r"layers\.(\d+)\.", key)
            if m is None:
                raise ValueError(f"Could not extract layer index from key: {key}")
            layer_idx = m.group(1)
            converted_state_dict[f"layers.{layer_idx}.moe_block.experts.down_proj"] = value.transpose(1, 2).contiguous()
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
    # Remove the placeholder entry for down_proj (handled specially below)
    inverted_mapping_dict.pop("layers.{}.moe_block.experts.down_proj", None)

    for key, value in state_dict.items():
        # Special case: gate_proj + up_proj → re-fuse into HF gate_up_proj
        if key.endswith(".moe_block.experts.gate_proj"):
            m = re.search(r"layers\.(\d+)\.", key)
            if m is None:
                raise ValueError(f"Could not extract layer index from key: {key}")
            layer_idx = m.group(1)
            up_key = f"layers.{layer_idx}.moe_block.experts.up_proj"
            if up_key not in state_dict:
                raise ValueError(f"Missing {up_key} when inverting gate_proj for layer {layer_idx}")
            gate = value.transpose(1, 2)              # [E, moe_intermediate, hidden]
            up = state_dict[up_key].transpose(1, 2)  # [E, moe_intermediate, hidden]
            gate_up = torch.cat([gate, up], dim=1)   # [E, 2*moe_intermediate, hidden]
            hf_key = f"model.language_model.layers.{layer_idx}.experts.gate_up_proj"
            converted_state_dict[hf_key] = gate_up.contiguous()
            continue
        if key.endswith(".moe_block.experts.up_proj"):
            # Already handled above alongside gate_proj
            continue

        # Special case: down_proj inverse transpose
        if key.endswith(".moe_block.experts.down_proj"):
            m = re.search(r"layers\.(\d+)\.", key)
            if m is None:
                raise ValueError(f"Could not extract layer index from key: {key}")
            layer_idx = m.group(1)
            hf_key = f"model.language_model.layers.{layer_idx}.experts.down_proj"
            converted_state_dict[hf_key] = value.transpose(1, 2).contiguous()
            continue

        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict

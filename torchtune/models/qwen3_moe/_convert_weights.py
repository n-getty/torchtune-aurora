import re
from collections import defaultdict

import torch

from torchtune.models.convert_weights import get_mapped_key

# Non-expert HF-to-torchtune key mappings. Expert weights are handled
# directly in the conversion functions (stacking/unstacking per-expert tensors).
_QWEN3_MOE_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.router.gate.weight",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}

_EXPERT_KEY_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)


def qwen3_moe_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 4,
    dim: int = 2048,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
    num_experts: int = 128,
) -> dict[str, torch.Tensor]:
    """Convert HF Qwen3 MoE state dict to torchtune format.

    HF stores experts as individual tensors per expert per projection:
      model.layers.N.mlp.experts.E.gate_proj.weight  [768, 2048]

    GroupedExpertsHF stores them stacked in HF-native layout (no transpose):
      layers.N.mlp.experts.gate_proj  [128, 768, 2048]
    """
    converted = {}
    if head_dim is None:
        head_dim = dim // num_heads

    # Collect per-expert tensors: experts_data[layer][proj][expert_idx] = tensor
    experts_data: dict[int, dict[str, dict[int, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" in key:
            continue
        if tie_word_embeddings and key == "lm_head.weight":
            continue

        m = _EXPERT_KEY_RE.match(key)
        if m:
            layer_idx = int(m.group(1))
            expert_idx = int(m.group(2))
            proj_name = m.group(3)
            experts_data[layer_idx][proj_name][expert_idx] = value
        else:
            new_key = get_mapped_key(key, _QWEN3_MOE_FROM_HF)
            converted[new_key] = value

    # Stack expert tensors into GroupedExpertsHF format (HF-native layout)
    for layer_idx in sorted(experts_data.keys()):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            expert_dict = experts_data[layer_idx][proj_name]
            stacked = torch.stack(
                [expert_dict[e] for e in range(num_experts)], dim=0
            )
            # HF layout [E, out_features, in_features] matches GroupedExpertsHF
            # directly — no transpose needed
            tune_key = f"layers.{layer_idx}.mlp.experts.{proj_name}"
            converted[tune_key] = stacked

    return converted


def qwen3_moe_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 4,
    dim: int = 2048,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
    num_experts: int = 128,
) -> dict[str, torch.Tensor]:
    """Convert torchtune Qwen3 MoE state dict back to HF format."""
    converted = {}
    inverted = {v: k for k, v in _QWEN3_MOE_FROM_HF.items() if v is not None}
    if head_dim is None:
        head_dim = dim // num_heads

    _expert_tune_re = re.compile(
        r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)"
    )

    for key, value in state_dict.items():
        m = _expert_tune_re.match(key)
        if m:
            layer_idx = m.group(1)
            proj_name = m.group(2)
            # HF-native layout: already [E, out_features, in_features] — just unstack
            for e in range(value.shape[0]):
                hf_key = f"model.layers.{layer_idx}.mlp.experts.{e}.{proj_name}.weight"
                converted[hf_key] = value[e].clone()
        else:
            new_key = get_mapped_key(key, inverted)
            converted[new_key] = value
            if key == "tok_embeddings.weight" and tie_word_embeddings:
                converted["lm_head.weight"] = value.detach().clone()

    return converted


def build_tune_to_hf_map_moe(named_params) -> dict[str, str]:
    """Build torchtune→HF param name mapping for weight sync.

    Non-expert params use the standard _QWEN3_MOE_FROM_HF inversion.
    Expert params (stacked 3D tensors) are identity-mapped here and expanded
    to per-expert tensors at sync time by expand_experts_for_vllm().
    """
    inverted = {v: k for k, v in _QWEN3_MOE_FROM_HF.items() if v is not None}
    tune_to_hf = {}
    for tune_name, _ in named_params:
        clean = tune_name.replace("_fsdp_wrapped_module.", "")
        clean = clean.replace("_checkpoint_wrapped_module.", "")
        if ".mlp.experts." in clean:
            tune_to_hf[clean] = clean
        else:
            tune_to_hf[clean] = get_mapped_key(clean, inverted)
    return tune_to_hf


def expand_experts_for_vllm(hf_state_dict: dict) -> dict:
    """Expand stacked expert tensors into per-expert HF format for vLLM.

    Called as a post-processing step in the SHM weight sync path. Detects
    stacked 3D expert params (identity-mapped by build_tune_to_hf_map_moe)
    and unstacks them into individual per-expert 2D tensors with HF key names.
    No transpose needed — GroupedExpertsHF stores in HF-native layout.
    """
    _expert_pattern = re.compile(
        r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)"
    )
    expanded = {}
    for key, value in hf_state_dict.items():
        m = _expert_pattern.match(key)
        if m and value.dim() == 3:
            layer_idx = m.group(1)
            proj_name = m.group(2)
            # HF-native layout: already [E, out, in] — just unstack
            for e in range(value.shape[0]):
                hf_key = f"model.layers.{layer_idx}.mlp.experts.{e}.{proj_name}.weight"
                expanded[hf_key] = value[e].contiguous()
        else:
            expanded[key] = value
    return expanded


def fuse_experts_for_vllm(hf_state_dict: dict) -> dict:
    """Fuse stacked expert tensors into vLLM's w13/w2 format for SHM sync.

    Produces 96 pre-fused 3D tensors (48 layers x {w13, w2}) instead of
    18,432 per-expert 2D tensors. The vLLM side only needs to TP-shard,
    transpose, and copy — eliminating the stacking/fusing bottleneck.

    Input expert keys (identity-mapped from torchtune):
      layers.N.mlp.experts.gate_proj  [E, intermediate, hidden]
      layers.N.mlp.experts.up_proj    [E, intermediate, hidden]
      layers.N.mlp.experts.down_proj  [E, hidden, intermediate]

    Output keys (vLLM param names):
      model.layers.N.mlp.experts.w13_weight  [E, 2*intermediate, hidden]
      model.layers.N.mlp.experts.w2_weight   [E, hidden, intermediate]
    """
    _expert_pattern = re.compile(
        r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)"
    )
    layer_experts: dict[str, dict[str, torch.Tensor]] = {}
    result = {}
    for key, value in hf_state_dict.items():
        m = _expert_pattern.match(key)
        if m and value.dim() == 3:
            layer_idx = m.group(1)
            proj = m.group(2)
            layer_experts.setdefault(layer_idx, {})[proj] = value
        else:
            result[key] = value

    for layer_idx in sorted(layer_experts.keys(), key=int):
        gate = layer_experts[layer_idx]["gate_proj"]
        up = layer_experts[layer_idx]["up_proj"]
        down = layer_experts[layer_idx]["down_proj"]
        w13 = torch.cat([gate, up], dim=1)
        result[f"model.layers.{layer_idx}.mlp.experts.w13_weight"] = w13
        result[f"model.layers.{layer_idx}.mlp.experts.w2_weight"] = down
    return result

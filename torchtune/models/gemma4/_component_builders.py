# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import nn

from torchtune.models.gemma._component_builders import gemma_mlp
from torchtune.models.gemma.gemma_norm_embedding import GemmaNormEmbeddings
from torchtune.models.gemma.rms_norm import GemmaRMSNorm
from torchtune.models.gemma2._component_builders import Gemma2FinalNorm
from torchtune.models.gemma4._attention import Gemma4Attention
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings
from torchtune.modules import TransformerDecoder, TransformerSelfAttentionLayer, TiedLinear
from torchtune.modules.attention_utils import _MaskType


# Default Gemma 4 layer pattern: 5 sliding + 1 full, repeated 10 times
_GEMMA4_LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 10


class Gemma4TransformerLayer(TransformerSelfAttentionLayer):
    """
    Gemma 4 transformer layer that extends TransformerSelfAttentionLayer
    with a per-layer scalar buffer applied after attention + MLP.

    The layer_scalar is a non-trainable buffer loaded from the checkpoint
    that scales the full layer output.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        out = super().forward(x, mask=mask, input_pos=input_pos, **kwargs)
        out = out * self.layer_scalar
        return out


def gemma4(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    local_head_dim: int,
    local_num_kv_heads: int,
    local_rope_base: float,
    sliding_window_size: int,
    global_head_dim: int,
    global_num_kv_heads: int,
    global_rope_base: float,
    global_partial_rotary_factor: float,
    global_k_eq_v: bool,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    final_capping_value: float = 30.0,
    query_pre_attn_scalar: Optional[int] = None,
    layer_types: Optional[list[str]] = None,
) -> TransformerDecoder:
    """
    Build Gemma 4 TransformerDecoder with heterogeneous attention layers.

    Gemma 4 alternates between sliding/local attention layers (5:1 ratio) and
    full/global attention layers. The two layer types differ in head dimension,
    KV head count, and RoPE configuration.

    Args:
        vocab_size: number of tokens in vocabulary
        num_layers: number of transformer layers
        num_heads: number of query heads (same for all layers)
        embed_dim: embedding dimension
        intermediate_dim: MLP intermediate dimension
        max_seq_len: maximum sequence length
        local_head_dim: head dimension for sliding attention layers
        local_num_kv_heads: number of KV heads for sliding attention layers
        local_rope_base: RoPE base frequency for sliding layers
        sliding_window_size: sliding window size for local attention
        global_head_dim: head dimension for full attention layers
        global_num_kv_heads: number of KV heads for full attention layers
        global_rope_base: RoPE base frequency for global layers
        global_partial_rotary_factor: fraction of head_dim receiving RoPE on global layers
        global_k_eq_v: whether keys equal values on global layers
        attn_dropout: attention dropout rate. Default: 0.0
        norm_eps: RMS norm epsilon. Default: 1e-6
        final_capping_value: logit soft capping value. Default: 30.0
        query_pre_attn_scalar: pre-attention scalar. Default: None (uses head_dim)
        layer_types: list of "sliding_attention" or "full_attention" per layer.
            Default: 5 sliding + 1 full repeated 10 times.

    Returns:
        TransformerDecoder: Instantiation of Gemma 4 model.
    """
    if layer_types is None:
        layer_types = _GEMMA4_LAYER_TYPES

    assert len(layer_types) == num_layers, (
        f"layer_types length ({len(layer_types)}) must match num_layers ({num_layers})"
    )

    # Create RoPE instances for each layer type
    # Local: full rotary on local_head_dim
    local_rope = Qwen2RotaryPositionalEmbeddings(
        dim=local_head_dim,
        max_seq_len=max_seq_len,
        base=local_rope_base,
    )
    # Global: partial rotary (25% of global_head_dim)
    global_rotary_dim = int(global_head_dim * global_partial_rotary_factor)
    global_rope = Qwen2RotaryPositionalEmbeddings(
        dim=global_rotary_dim,
        max_seq_len=max_seq_len,
        base=global_rope_base,
    )

    layers = nn.ModuleList()
    for layer_idx in range(num_layers):
        is_global = layer_types[layer_idx] == "full_attention"

        if is_global:
            head_dim = global_head_dim
            num_kv_heads = global_num_kv_heads
            rope = global_rope
            sliding_window = None
            partial_rotary = global_partial_rotary_factor
            k_eq_v = global_k_eq_v
        else:
            head_dim = local_head_dim
            num_kv_heads = local_num_kv_heads
            rope = local_rope
            sliding_window = sliding_window_size
            partial_rotary = 1.0
            k_eq_v = False

        mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

        # Global layers have no v_proj (k_eq_v is structural)
        if k_eq_v:
            v_proj = nn.Identity()
        else:
            v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)

        self_att = Gemma4Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=v_proj,
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            q_norm=GemmaRMSNorm(head_dim, eps=norm_eps),
            k_norm=GemmaRMSNorm(head_dim, eps=norm_eps),
            kv_cache=None,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            sliding_window_size=sliding_window,
            softcapping=None,  # Gemma4 has no per-layer attention soft capping
            query_pre_attn_scalar=query_pre_attn_scalar,
            partial_rotary_factor=partial_rotary,
            k_eq_v=k_eq_v,
        )

        layer = Gemma4TransformerLayer(
            attn=self_att,
            mlp=mlp,
            sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            sa_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = GemmaNormEmbeddings(vocab_size, embed_dim)
    output_proj = TiedLinear(tok_embeddings)
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        output=output_proj,
        head_dim=local_head_dim,  # Use local head_dim as default (50 of 60 layers)
        norm=Gemma2FinalNorm(final_capping_value, embed_dim, eps=norm_eps),
    )
    return model

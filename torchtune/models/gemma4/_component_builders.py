# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchtune.models.gemma._component_builders import gemma_mlp
from torchtune.models.gemma.gemma_norm_embedding import GemmaNormEmbeddings
from torchtune.models.gemma.rms_norm import GemmaRMSNorm
from torchtune.models.gemma2._component_builders import Gemma2FinalNorm
from torchtune.models.gemma4._attention import Gemma4Attention
from torchtune.models.gemma4._moe import Gemma4MoeRouter
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings
from torchtune.modules import TransformerDecoder, TransformerSelfAttentionLayer, TiedLinear
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.moe import MoE
from torchtune.modules.moe.experts import GroupedExperts


# Default Gemma 4 layer pattern: 5 sliding + 1 full, repeated 10 times
_GEMMA4_LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 10


class Gemma4TransformerLayer(TransformerSelfAttentionLayer):
    """
    Gemma 4 transformer layer extending TransformerSelfAttentionLayer with:
    - A per-layer scalar buffer (layer_scalar) applied after the FFN residual.
    - Optional additive MoE block alongside the dense MLP (for the 26B-A4B model).

    When ``moe_block`` is provided, the FFN output is:
        dense_out = post_mlp_norm( mlp( mlp_norm(h) ) )
        moe_out   = post_moe_norm( moe_block( pre_moe_norm(h) ) )
        h = h + mlp_scale( dense_out + moe_out )

    When ``moe_block`` is None (31B dense), the standard forward is used:
        h = h + mlp_scale( mlp( mlp_norm(h) ) )

    Args:
        moe_block (Optional[nn.Module]): MoE module (``MoE`` instance). Default: None.
        pre_moe_norm (Optional[nn.Module]): Norm applied to h before MoE. Default: None.
        post_moe_norm (Optional[nn.Module]): Norm applied to MoE output. Default: None.
        post_mlp_norm (Optional[nn.Module]): Norm applied to dense MLP output before
            adding to MoE output. Default: None.
        *args, **kwargs: Passed to TransformerSelfAttentionLayer.
    """

    def __init__(
        self,
        *args,
        moe_block: Optional[nn.Module] = None,
        pre_moe_norm: Optional[nn.Module] = None,
        post_moe_norm: Optional[nn.Module] = None,
        post_mlp_norm: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer("layer_scalar", torch.ones(1))
        self.moe_block = moe_block
        self.pre_moe_norm = pre_moe_norm
        self.post_moe_norm = post_moe_norm
        self.post_mlp_norm = post_mlp_norm
        # v158: when True, attention+dense run inside an explicit non-reentrant
        # checkpoint and MoE runs OUTSIDE it. Set by the recipe in lieu of
        # apply_activation_checkpointing for MoE-bearing layers, so the router
        # is never recomputed (recompute would re-derive a different
        # num_tokens_per_expert under tie-flips → ScatterAddBackward0 mismatch).
        self._ac_enabled = False

    def _attn_and_dense(
        self,
        x: torch.Tensor,
        mask: Optional[_MaskType],
        input_pos: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.sa_norm(x)
        if self.mask_mod is not None:
            bsz, seq_len, *_ = h.shape
            mask = self.mask_mod(mask=mask, bsz=bsz, seq_len=seq_len)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)
        h = self.sa_scale(attn_out) + x
        if self.moe_block is not None:
            dense_part = self.post_mlp_norm(self.mlp(self.mlp_norm(h)))
        else:
            dense_part = self.mlp(self.mlp_norm(h))
        return h, dense_part

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        if self._ac_enabled:
            h, dense_part = torch.utils.checkpoint.checkpoint(
                self._attn_and_dense,
                x,
                mask,
                input_pos,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            h, dense_part = self._attn_and_dense(x, mask, input_pos)

        if self.moe_block is not None:
            moe_part = self.post_moe_norm(self.moe_block(self.pre_moe_norm(h)))
            out = h + self.mlp_scale(dense_part + moe_part)
        else:
            out = h + self.mlp_scale(dense_part)

        out = out * self.layer_scalar.to(out.device)
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


def gemma4_moe_block(
    embed_dim: int,
    moe_intermediate_dim: int,
    num_experts: int,
    top_k: int,
    norm_eps: float = 1e-6,
) -> MoE:
    """
    Build the MoE block for Gemma 4 26B-A4B.

    Uses a custom Gemma4MoeRouter (RMSNorm + proj + scale + per_expert_scale)
    and GroupedExperts with GELU activation. Falls through to the loop-based
    expert computation path on XPU (grouped_mm requires SM90/CUDA).

    Args:
        embed_dim: Model hidden dimension.
        moe_intermediate_dim: Each expert's intermediate dimension (704 for 26B-A4B).
        num_experts: Total number of experts (128 for 26B-A4B).
        top_k: Number of experts each token is routed to (8 for 26B-A4B).
        norm_eps: RMSNorm epsilon. Default: 1e-6.

    Returns:
        MoE: Configured MoE layer.
    """
    router = Gemma4MoeRouter(
        hidden_dim=embed_dim,
        num_experts=num_experts,
        top_k=top_k,
        norm_eps=norm_eps,
    )
    experts = GroupedExperts(
        dim=embed_dim,
        hidden_dim=moe_intermediate_dim,
        num_experts=num_experts,
        activation=partial(F.gelu, approximate="tanh"),
    )
    return MoE(router=router, experts=experts, shared_expert=None)


# Default 26B-A4B layer pattern: 5 sliding + 1 full, repeated 5 times (30 layers)
_GEMMA4_26B_LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 5


def gemma4_26b_a4b(
    vocab_size: int = 262_144,
    num_layers: int = 30,
    num_heads: int = 16,
    embed_dim: int = 2816,
    intermediate_dim: int = 2112,
    moe_intermediate_dim: int = 704,
    num_experts: int = 128,
    top_k: int = 8,
    max_seq_len: int = 8192,
    local_head_dim: int = 256,
    local_num_kv_heads: int = 8,
    local_rope_base: float = 10_000.0,
    sliding_window_size: int = 1024,
    global_head_dim: int = 512,
    global_num_kv_heads: int = 2,
    global_rope_base: float = 1_000_000.0,
    global_partial_rotary_factor: float = 0.25,
    global_k_eq_v: bool = True,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    final_capping_value: float = 30.0,
    query_pre_attn_scalar: Optional[int] = None,
    layer_types: Optional[list] = None,
) -> TransformerDecoder:
    """
    Build the Gemma 4 26B-A4B MoE TransformerDecoder.

    Architecture: 30 layers, each with a dense MLP (intermediate_dim=2112) plus an
    additive MoE block (128 experts, top-8, moe_intermediate_dim=704). Same hybrid
    sliding/global attention pattern as the 31B dense model (5:1 ratio), but with
    30 layers instead of 60.

    The MoE block output is added to the dense MLP output (not a replacement):
        out = post_feedforward_layernorm(
            post_mlp_norm(dense_mlp(x)) + post_moe_norm(moe_block(pre_moe_norm(x)))
        )

    Args:
        vocab_size: Vocabulary size. Default: 262144.
        num_layers: Number of transformer layers. Default: 30.
        num_heads: Number of query attention heads. Default: 16.
        embed_dim: Model hidden dimension. Default: 2816.
        intermediate_dim: Dense MLP intermediate dimension. Default: 2112.
        moe_intermediate_dim: Per-expert MLP intermediate dimension. Default: 704.
        num_experts: Total number of MoE experts. Default: 128.
        top_k: Number of experts activated per token. Default: 8.
        max_seq_len: Maximum sequence length. Default: 8192.
        local_head_dim: Head dimension for sliding attention layers. Default: 256.
        local_num_kv_heads: KV heads for sliding attention layers. Default: 8.
        local_rope_base: RoPE base for sliding attention. Default: 10000.
        sliding_window_size: Sliding window size. Default: 1024.
        global_head_dim: Head dimension for global attention layers. Default: 512.
        global_num_kv_heads: KV heads for global attention layers. Default: 8.
        global_rope_base: RoPE base for global attention. Default: 1000000.
        global_partial_rotary_factor: Fraction of head_dim with RoPE on global layers.
            Default: 0.25.
        global_k_eq_v: Whether global layers share K and V projections. Default: True.
        attn_dropout: Attention dropout rate. Default: 0.0.
        norm_eps: RMSNorm epsilon. Default: 1e-6.
        final_capping_value: Logit soft-capping value. Default: 30.0.
        query_pre_attn_scalar: Pre-attention scalar. Default: None (uses head_dim).
        layer_types: List of "sliding_attention" or "full_attention" per layer.
            Default: 5 sliding + 1 full, repeated 5 times.

    Returns:
        TransformerDecoder: Gemma 4 26B-A4B model.
    """
    if layer_types is None:
        layer_types = _GEMMA4_26B_LAYER_TYPES

    assert len(layer_types) == num_layers, (
        f"layer_types length ({len(layer_types)}) must match num_layers ({num_layers})"
    )

    local_rope = Qwen2RotaryPositionalEmbeddings(
        dim=local_head_dim,
        max_seq_len=max_seq_len,
        base=local_rope_base,
    )
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
            softcapping=None,
            query_pre_attn_scalar=query_pre_attn_scalar,
            partial_rotary_factor=partial_rotary,
            k_eq_v=k_eq_v,
        )

        moe = gemma4_moe_block(
            embed_dim=embed_dim,
            moe_intermediate_dim=moe_intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            norm_eps=norm_eps,
        )

        layer = Gemma4TransformerLayer(
            attn=self_att,
            mlp=mlp,
            sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            sa_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            moe_block=moe,
            pre_moe_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            post_moe_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            post_mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
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
        head_dim=local_head_dim,  # Use local head_dim (25 of 30 layers are sliding)
        norm=Gemma2FinalNorm(final_capping_value, embed_dim, eps=norm_eps),
    )
    return model

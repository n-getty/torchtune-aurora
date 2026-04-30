from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from torchtune.models.qwen2._positional_embeddings import (
    Qwen2RotaryPositionalEmbeddings,
)
from torchtune.models.qwen3._attention import Qwen3Attention
from torchtune.models.qwen3_moe._experts import GroupedExpertsHF
from torchtune.models.qwen3_moe._router import Qwen3MoeRouter
from torchtune.modules import RMSNorm, TiedLinear, TransformerSelfAttentionLayer
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.moe.moe import MoE
from torchtune.modules.transformer import TransformerDecoder


class Qwen3MoeTransformerLayer(TransformerSelfAttentionLayer):
    """Qwen3 MoE layer that supports split activation checkpointing.

    When ``self._ac_enabled`` is set to True (by the recipe's split-AC helper),
    the attention block (sa_norm + attn + residual) runs inside an explicit
    non-reentrant ``torch.utils.checkpoint`` and the MoE block runs OUTSIDE it.
    This mirrors the v158 fix in ``Gemma4TransformerLayer``: it prevents the
    router from being re-executed during backward, which would otherwise
    re-derive ``num_tokens_per_expert`` under tie-flips and cause a
    ScatterAddBackward0 ±1 row-count mismatch on EP ranks.

    When ``_ac_enabled`` is False (the default), behavior is identical to
    ``TransformerSelfAttentionLayer.forward``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set to True by the recipe's split-AC helper for MoE-bearing layers.
        self._ac_enabled = False

    def _attn_block(
        self,
        x: torch.Tensor,
        mask: Optional[_MaskType],
        input_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.sa_norm(x)
        if self.mask_mod is not None:
            bsz, seq_len, *_ = h.shape
            mask = self.mask_mod(mask=mask, bsz=bsz, seq_len=seq_len)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)
        return self.sa_scale(attn_out) + x

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        if self._ac_enabled:
            h = torch.utils.checkpoint.checkpoint(
                self._attn_block,
                x,
                mask,
                input_pos,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            h = self._attn_block(x, mask, input_pos)

        # MoE always runs OUTSIDE the AC region — router executes exactly once,
        # so num_tokens_per_expert and gather_idx are stable across FWD/BWD.
        mlp_out = self.mlp(self.mlp_norm(h))
        return h + self.mlp_scale(mlp_out)


def qwen3_moe_block(
    embed_dim: int,
    moe_intermediate_dim: int,
    num_experts: int,
    experts_per_token: int,
    norm_topk_prob: bool = True,
) -> MoE:
    """Build a single MoE block for Qwen3 MoE."""
    router = Qwen3MoeRouter(
        gate=nn.Linear(embed_dim, num_experts, bias=False),
        dim=embed_dim,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        norm_topk_prob=norm_topk_prob,
    )
    experts = GroupedExpertsHF(
        dim=embed_dim,
        hidden_dim=moe_intermediate_dim,
        num_experts=num_experts,
        activation=F.silu,
    )
    return MoE(router=router, experts=experts, shared_expert=None)


def qwen3_moe(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    moe_intermediate_dim: int,
    num_experts: int,
    experts_per_token: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
    norm_topk_prob: bool = True,
) -> TransformerDecoder:
    """Build the Qwen3 MoE decoder.

    Each layer uses standard Qwen3 attention (with QK-norm) and an MoE block
    in place of the dense MLP. The MoE block is passed as the ``mlp`` argument
    to ``TransformerSelfAttentionLayer`` — no custom layer class needed.
    """
    head_dim = head_dim or embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = Qwen2RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = Qwen3Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            q_norm=RMSNorm(dim=head_dim, eps=norm_eps),
            k_norm=RMSNorm(dim=head_dim, eps=norm_eps),
            kv_cache=None,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        moe = qwen3_moe_block(
            embed_dim=embed_dim,
            moe_intermediate_dim=moe_intermediate_dim,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            norm_topk_prob=norm_topk_prob,
        )
        layer = Qwen3MoeTransformerLayer(
            attn=self_attn,
            mlp=moe,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    if tie_word_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

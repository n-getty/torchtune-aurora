# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Gemma4Attention(nn.Module):
    """
    Attention module for Gemma 4 models, supporting:
    - Per-layer sliding window vs full attention
    - Per-layer head dimensions (local=256, global=512)
    - Partial rotary embeddings (global layers use 25% of head_dim)
    - K=V sharing on global layers (attention_k_eq_v)
    - Logit soft capping on attention weights

    Based on Gemma2Attention with extensions for Gemma4's heterogeneous layer architecture.

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads
        num_kv_heads (int): number of key and value heads
        head_dim (int): dimension of each head
        q_proj (nn.Module): projection layer for query
        k_proj (nn.Module): projection layer for key
        v_proj (nn.Module): projection layer for value
        output_proj (nn.Module): projection layer for output
        pos_embeddings (Optional[nn.Module]): positional embeddings layer
        q_norm (Optional[nn.Module]): normalization layer for query
        k_norm (Optional[nn.Module]): normalization layer for key
        kv_cache (Optional[KVCache]): KVCache object
        max_seq_len (int): maximum sequence length. Default: 4096
        is_causal (bool): sets the default mask to causal. Default: True
        attn_dropout (float): dropout value. Default: 0.0
        sliding_window_size (Optional[int]): size of sliding window, None for full attention
        softcapping (Optional[float]): soft capping value for attention weights
        query_pre_attn_scalar (Optional[int]): pre-attention scaling value
        partial_rotary_factor (float): fraction of head_dim receiving RoPE. Default: 1.0
        k_eq_v (bool): when True, value = key (keys equal values). Default: False
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        sliding_window_size: Optional[int] = None,
        softcapping: Optional[float] = None,
        query_pre_attn_scalar: Optional[int] = None,
        partial_rotary_factor: float = 1.0,
        k_eq_v: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({attn_dropout}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.partial_rotary_factor = partial_rotary_factor
        self.k_eq_v = k_eq_v

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        # gemma related parameters
        self.sliding_window_size = sliding_window_size
        self.softcapping = softcapping
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def _apply_rotary(self, t: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply RoPE with partial rotary factor support."""
        if self.pos_embeddings is None:
            return t
        if self.partial_rotary_factor < 1.0:
            rotary_dim = int(self.head_dim * self.partial_rotary_factor)
            t_rot, t_pass = t[..., :rotary_dim], t[..., rotary_dim:]
            t_rot = self.pos_embeddings(t_rot, input_pos=input_pos)
            return torch.cat([t_rot, t_pass], dim=-1)
        else:
            return self.pos_embeddings(t, input_pos=input_pos)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None and (not isinstance(mask, torch.Tensor)):
            raise NotImplementedError(
                "Block masks are not implemented yet, use packed=False."
            )

        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Gemma4 order: norm → RoPE → transpose
        if self.q_norm is not None:
            q = self.q_norm(q)

        q = self._apply_rotary(q, input_pos=input_pos)

        # [b, n_h, s_x, h_d]
        q = q.transpose(1, 2)

        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # k has shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)

            # K=V: on global layers, value = raw k_proj output (no norm, no RoPE)
            # Must capture before k_norm and RoPE are applied to k
            if self.k_eq_v:
                v = k  # reference — k will be reassigned by norm/RoPE below
            else:
                v = self.v_proj(y)

            # Reshape k for norm and RoPE: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)

            # Apply k_norm BEFORE RoPE (Gemma4 order)
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Apply partial rotary positional embeddings to k (NOT v)
            k = self._apply_rotary(k, input_pos=input_pos)

            # Expand KV heads to match query heads
            k = k.view(b, s_y, self.num_kv_heads, 1, self.head_dim)
            v = v.view(b, s_y, self.num_kv_heads, 1, self.head_dim)

            if self.num_heads != self.num_kv_heads:
                k = k.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)
                v = v.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)

            k = k.reshape(b, s_y, -1, self.head_dim)
            v = v.reshape(b, s_y, -1, self.head_dim)

            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        q.mul_(self.scaling)
        output = torch.matmul(q, k.transpose(2, 3))

        if mask is None:
            mask = torch.tril(
                torch.ones(
                    size=(s_x, s_x),
                    dtype=torch.bool,
                ).to(x.device)
            )

        if mask.dtype == torch.bool:
            mask = torch.where(mask.logical_not(), -2.3819763e38, 0)

        if self.sliding_window_size is not None:
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        if self.softcapping is not None:
            output = output / self.softcapping
            output = torch.tanh(output)
            output = output * self.softcapping

        output = output + mask
        output = F.softmax(output.float(), dim=-1).type_as(q)

        output = torch.matmul(output, v)

        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)

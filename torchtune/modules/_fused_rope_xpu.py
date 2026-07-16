# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Fused Triton RoPE (forward + backward) for Intel PVC / Aurora XPU.

Drop-in for ``torchtune.models.qwen2._positional_embeddings.Qwen2RotaryPositionalEmbeddings``
(the RoPE Qwen3 uses). Collapses the eager rotate-half RoPE — which materializes ``rotated =
cat((-x2, x1))`` plus two elementwise products and an add, i.e. several passes over the
[b, s, n_h, h_d] activation — into a single Triton launch. Backward is hand-written (a Triton
kernel is not auto-differentiable).

SEMANTICS (must match Qwen2RotaryPositionalEmbeddings exactly):
    cache[pos] = cat([cos(freqs), sin(freqs)])   # freqs = cat([idx_theta, idx_theta]) -> each
                                                 #   half of cos/sin is duplicated, fp32
    x1, x2 = x[..., :h_d/2], x[..., h_d/2:]
    out    = x * cos + cat((-x2, x1)) * sin       # rotate-half (GPT-NeoX), NOT interleaved
Per (half-index j) pair (x1_j, x2_j) this is a 2D rotation by angle a_j:
    y1_j = x1_j * c_j - x2_j * s_j
    y2_j = x2_j * c_j + x1_j * s_j     (c_j = cos, s_j = sin at position/angle j)

BACKWARD: the map is an orthogonal rotation, so dL/dx = J^T @ dL/dy where J = [[c,-s],[s,c]]
=> J^T = [[c, s],[-s, c]] = rotation by -angle. i.e. backward is the forward with sin negated:
    dx1_j = c_j * dy1_j + s_j * dy2_j
    dx2_j = -s_j * dy1_j + c_j * dy2_j
RoPE has NO learnable params (theta/cache are buffers), so backward returns only dx.

input_pos: during packed training torchtune passes per-token position ids [b, s]; we gather
cos/sin = cache[input_pos] in Python (cheap slice/gather) and expand to [b*s, h_d], so the
kernel just indexes cos/sin by (row // n_h). input_pos=None -> cache[:seq_len] broadcast over b.

Gated by TORCHTUNE_USE_FUSED_ROPE=1; swapped via maybe_swap_rope_for_fused(model) which
TRANSPLANTS the existing theta/cache buffers (meta-safe, zero recompute). Clean eager fallback
on non-XPU / no-Triton / odd head_dim.
"""
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from torchtune.utils import get_logger

_log: logging.Logger = get_logger()

_USE_FUSED_ROPE = os.environ.get("TORCHTUNE_USE_FUSED_ROPE", "0") == "1"

_triton = None
_tl = None
if _USE_FUSED_ROPE:
    try:
        import triton as _triton
        import triton.language as _tl
    except ImportError:
        _triton = None
        _tl = None

# Imported lazily inside the swap helper to avoid a hard dep at import time.
try:
    from torchtune.models.qwen2._positional_embeddings import (
        Qwen2RotaryPositionalEmbeddings,
    )
except Exception:  # pragma: no cover
    Qwen2RotaryPositionalEmbeddings = None

_SWAP_LOG_DONE = False


if _triton is not None:

    def _warp_configs():
        return [
            _triton.Config({}, num_warps=w, num_stages=1)
            for w in (1, 2, 4, 8, 16)
        ]

    @_triton.autotune(configs=_warp_configs(), key=["HALF"])
    @_triton.jit
    def _rope_fwd_kernel(
        X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
        n_heads, HALF,
        stride_xr, stride_cr,
        NEG_SIN: _tl.constexpr,          # False=forward, True=backward (sin negated)
        BLOCK: _tl.constexpr,
    ):
        # One program per row = one (b, s, head) slice of head_dim = 2*HALF elements.
        row = _tl.program_id(0)
        pos = row // n_heads             # (b,s) flat position -> cos/sin row
        offs = _tl.arange(0, BLOCK)
        mask = offs < HALF

        x_base = X_ptr + row * stride_xr
        x1 = _tl.load(x_base + offs, mask=mask, other=0.0).to(_tl.float32)
        x2 = _tl.load(x_base + HALF + offs, mask=mask, other=0.0).to(_tl.float32)

        c = _tl.load(COS_ptr + pos * stride_cr + offs, mask=mask, other=1.0).to(_tl.float32)
        s = _tl.load(SIN_ptr + pos * stride_cr + offs, mask=mask, other=0.0).to(_tl.float32)
        if NEG_SIN:
            s = -s

        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s

        o_base = OUT_ptr + row * stride_xr
        _tl.store(o_base + offs, y1, mask=mask)
        _tl.store(o_base + HALF + offs, y2, mask=mask)

    def _launch(x2d, cos2d, sin2d, n_heads, half, neg_sin):
        out = torch.empty_like(x2d)
        BLOCK = _triton.next_power_of_2(half)
        _rope_fwd_kernel[(x2d.shape[0],)](
            x2d, cos2d, sin2d, out,
            n_heads, half,
            x2d.stride(0), cos2d.stride(0),
            NEG_SIN=neg_sin, BLOCK=BLOCK,
        )
        return out

    class _FusedRoPEFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos2d, sin2d, n_heads):
            # x: [b, s, n_h, h_d] contiguous; cos2d/sin2d: [b*s, h_d] (halves duplicated).
            b, s, nh, hd = x.shape
            half = hd // 2
            x2d = x.reshape(-1, hd).contiguous()
            out = _launch(x2d, cos2d, sin2d, nh, half, neg_sin=False)
            ctx.save_for_backward(cos2d, sin2d)
            ctx.shape = (b, s, nh, hd)
            ctx.n_heads = nh
            return out.reshape(b, s, nh, hd)

        @staticmethod
        def backward(ctx, grad_out):
            cos2d, sin2d = ctx.saved_tensors
            b, s, nh, hd = ctx.shape
            half = hd // 2
            g2d = grad_out.reshape(-1, hd).contiguous()
            dx = _launch(g2d, cos2d, sin2d, nh, half, neg_sin=True)
            return dx.reshape(b, s, nh, hd), None, None, None


class FusedQwen2RoPE(nn.Module):
    """Drop-in for Qwen2RotaryPositionalEmbeddings. Same buffers (theta, cache), same
    forward signature. Uses the fused Triton kernel on XPU; exact eager fallback otherwise."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 1_000_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def _eager(self, x: torch.Tensor, input_pos: Optional[torch.Tensor]) -> torch.Tensor:
        seq_len = x.size(1)
        head_dim = x.size(-1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        rope_cache = rope_cache.view(-1, seq_len, 1, head_dim * 2)
        cos = rope_cache[..., :head_dim].to(x.dtype)
        sin = rope_cache[..., head_dim:].to(x.dtype)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return ((x * cos) + (rotated * sin)).type_as(x)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [b, s, n_h, h_d]
        b, s, nh, hd = x.shape
        if _triton is None or x.device.type != "xpu" or hd % 2 != 0:
            return self._eager(x, input_pos)

        # Build cos/sin as [b*s, h_d] to match the kernel's (row // n_h) indexing.
        rope_cache = self.cache[:s] if input_pos is None else self.cache[input_pos]
        # rope_cache: [s, 2*hd] (input_pos=None) or [b, s, 2*hd] (packed) -> [., s, 2*hd]
        rope_cache = rope_cache.view(-1, s, hd * 2)
        cos = rope_cache[..., :hd].to(x.dtype)         # [nb, s, hd], nb in {1, b}
        sin = rope_cache[..., hd:].to(x.dtype)
        if cos.shape[0] == 1 and b > 1:
            cos = cos.expand(b, s, hd)
            sin = sin.expand(b, s, hd)
        cos2d = cos.reshape(-1, hd).contiguous()       # [b*s, hd]
        sin2d = sin.reshape(-1, hd).contiguous()
        return _FusedRoPEFn.apply(x.contiguous(), cos2d, sin2d, nh)

    @classmethod
    def from_rope(cls, rope: nn.Module) -> "FusedQwen2RoPE":
        """Build from an existing Qwen2RotaryPositionalEmbeddings, TRANSPLANTING the theta/
        cache buffers (no recompute, meta-safe)."""
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.dim = rope.dim
        m.base = rope.base
        m.max_seq_len = rope.max_seq_len
        # transplant buffers (register on the new module so .to()/sharding track them)
        m.register_buffer("theta", rope.theta, persistent=False)
        m.register_buffer("cache", rope.cache, persistent=False)
        return m


def maybe_swap_rope_for_fused(model: nn.Module) -> int:
    """If TORCHTUNE_USE_FUSED_ROPE=1 and Triton is available, replace every
    Qwen2RotaryPositionalEmbeddings with FusedQwen2RoPE (buffers transplanted). Returns count
    swapped (0 if off / unavailable). Safe to call before or after FSDP (RoPE has no params)."""
    global _SWAP_LOG_DONE
    if not _USE_FUSED_ROPE:
        return 0
    if _triton is None or Qwen2RotaryPositionalEmbeddings is None:
        if not _SWAP_LOG_DONE:
            _log.info("fused_rope=disabled (triton or Qwen2RoPE import failed)")
            _SWAP_LOG_DONE = True
        return 0

    to_swap = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, Qwen2RotaryPositionalEmbeddings)
    ]
    for name, mod in to_swap:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], FusedQwen2RoPE.from_rope(mod))

    if not _SWAP_LOG_DONE:
        _log.info("fused_rope=engaged (%d RoPE modules swapped)", len(to_swap))
        _SWAP_LOG_DONE = True
    return len(to_swap)


def _reset_swap_log_for_testing() -> None:
    global _SWAP_LOG_DONE
    _SWAP_LOG_DONE = False

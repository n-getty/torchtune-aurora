# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Differentiable fused RMSNorm (forward + backward Triton kernels) for Intel PVC / Aurora XPU.

Drop-in for ``torchtune.modules.rms_norm.RMSNorm``. Collapses the 3-pass eager RMSNorm
(upcast+square+mean / rsqrt+mul / cast+scale) into a single Triton launch, and hand-writes
the backward (a Triton kernel is not auto-differentiable) so it works in training.

Gated behind ``TORCHTUNE_USE_FUSED_RMSNORM=1`` and swapped in via
``maybe_swap_rmsnorm_for_fused(model)`` BEFORE FSDP sharding. The swap TRANSPLANTS the
existing ``.scale`` nn.Parameter object into the fused module, so it is safe on meta-device
models (no reallocation) and preserves the state_dict / sharding / weight-load wiring.

Provenance: ported from autokernel/kernels/fused_rmsnorm_autograd.py, validated on one PVC
tile — forward rel-err 1.7e-3, dx 2.9e-3, dscale 1.7e-3 (norm-wise), fp32 gradcheck PASS;
swapping all norm sites in a real Qwen3-32B layer gave +8.5% fwd+bwd step (22.2%->24.1% of
420 TFLOP/s) with end-to-end grad rel-err 7.8e-4. See autokernel docs/kernel_catalogue.md §9.

Falls back cleanly (import-guarded) to eager RMSNorm on any non-XPU device, if Triton is
unavailable, or if the last-dim block would exceed the Triton element limit.
"""
import logging
import os

import torch
import torch.nn as nn

from torchtune.modules.rms_norm import RMSNorm
from torchtune.utils import get_logger

_log: logging.Logger = get_logger()

_USE_FUSED_RMSNORM = os.environ.get("TORCHTUNE_USE_FUSED_RMSNORM", "0") == "1"

# Triton is optional; guard the import so importing this module never hard-fails.
# NOTE: the kernels MUST use the conventional aliases ``triton`` / ``tl`` (not private
# ``_triton``/``_tl``). Under torch.compile, Inductor re-codegens any @triton.jit kernel
# reached during tracing into a fresh module that imports triton as ``triton`` and
# triton.language as ``tl``; private aliases in the kernel source then raise
# ``NameError: name '_tl' is not defined`` at Inductor compile time. (HW-caught: fused+compile
# A/B job 8676886, 2026-07-16 — autokernel's own tests passed only because they ran compile-OFF.)
triton = None
tl = None
if _USE_FUSED_RMSNORM:
    try:
        import triton
        import triton.language as tl
    except ImportError:
        triton = None
        tl = None

_SWAP_LOG_DONE = False


# ---------------------------------------------------------------------------
# Triton kernels (defined only if triton imported). num_stages=1 is an XPU
# requirement (multi-stage pipelining is unsupported on PVC).
# ---------------------------------------------------------------------------
if triton is not None:

    def _warp_configs():
        return [
            triton.Config({}, num_warps=w, num_stages=1)
            for w in (1, 2, 4, 8, 16, 32)
        ]

    @triton.autotune(configs=_warp_configs(), key=["N"])
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, W_ptr, Y_ptr, R_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + row * stride_xm + offs * stride_xn,
                     mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / N
        r = 1.0 / tl.sqrt(ms + eps)
        tl.store(R_ptr + row, r)
        xn = x * r
        xn_cast = xn.to(Y_ptr.dtype.element_ty)          # torchtune casts BEFORE scale
        w = tl.load(W_ptr + offs, mask=mask, other=0.0)
        tl.store(Y_ptr + row * stride_ym + offs * stride_yn, xn_cast * w, mask=mask)

    @triton.autotune(configs=_warp_configs(), key=["N"])
    @triton.jit
    def _rmsnorm_bwd_kernel(
        X_ptr, W_ptr, G_ptr, R_ptr,
        DX_ptr, DSP_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_gm, stride_gn,
        stride_dxm, stride_dxn,
        stride_dspm, stride_dspn,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + row * stride_xm + offs * stride_xn,
                     mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + row * stride_gm + offs * stride_gn,
                     mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R_ptr + row)
        xn = x * r
        c = tl.sum(g * w * x, axis=0)
        dx = r * w * g - x * (r * r * r) * c / N
        tl.store(DX_ptr + row * stride_dxm + offs * stride_dxn, dx, mask=mask)
        tl.store(DSP_ptr + row * stride_dspm + offs * stride_dspn, g * xn, mask=mask)

    class _FusedRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            orig_shape = x.shape
            N = orig_shape[-1]
            x2d = x.reshape(-1, N).contiguous()
            M = x2d.shape[0]
            weight_c = weight.contiguous()
            y2d = torch.empty_like(x2d)
            r = torch.empty(M, device=x2d.device, dtype=torch.float32)
            BLOCK_SIZE = triton.next_power_of_2(N)
            _rmsnorm_fwd_kernel[(M,)](
                x2d, weight_c, y2d, r,
                M, N,
                x2d.stride(0), x2d.stride(1),
                y2d.stride(0), y2d.stride(1),
                eps, BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.save_for_backward(x2d, weight_c, r)
            ctx.orig_shape = orig_shape
            ctx.N = N
            ctx.M = M
            return y2d.reshape(orig_shape)

        @staticmethod
        def backward(ctx, grad_out):
            x2d, weight_c, r = ctx.saved_tensors
            N, M = ctx.N, ctx.M
            g2d = grad_out.reshape(-1, N).contiguous()
            dx2d = torch.empty_like(x2d)
            dscale_partial = torch.empty(M, N, device=x2d.device, dtype=torch.float32)
            BLOCK_SIZE = triton.next_power_of_2(N)
            _rmsnorm_bwd_kernel[(M,)](
                x2d, weight_c, g2d, r,
                dx2d, dscale_partial,
                M, N,
                x2d.stride(0), x2d.stride(1),
                g2d.stride(0), g2d.stride(1),
                dx2d.stride(0), dx2d.stride(1),
                dscale_partial.stride(0), dscale_partial.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
            dscale = dscale_partial.sum(dim=0).to(weight_c.dtype)
            dx = dx2d.reshape(ctx.orig_shape).to(grad_out.dtype)
            return dx, dscale, None


class FusedRMSNorm(nn.Module):
    """Drop-in for :class:`torchtune.modules.rms_norm.RMSNorm`.

    Same ``__init__(dim, eps)`` and same learnable parameter name (``self.scale``) so
    state_dicts are byte-compatible with the eager module.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to the exact eager formula when the fused path can't run:
        # non-XPU device, triton missing, or a last dim too large for one block.
        if (
            triton is None
            or x.device.type != "xpu"
            or self.normalized_shape[0] > 131072
        ):
            x_fp32 = x.float()
            x_normed = (
                x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
            ).type_as(x)
            return x_normed * self.scale
        return _FusedRMSNormFn.apply(x, self.scale, self.eps)

    @classmethod
    def from_rms_norm(cls, rms: RMSNorm) -> "FusedRMSNorm":
        """Build a FusedRMSNorm from an existing RMSNorm, TRANSPLANTING the existing
        ``.scale`` nn.Parameter object (not a copy). This is meta-device safe and keeps
        the parameter's identity so FSDP sharding / state_dict load wiring is preserved."""
        dim = rms.scale.shape[0]
        eps = getattr(rms, "eps", 1e-6)
        m = cls.__new__(cls)          # bypass __init__ so we don't allocate a new scale
        nn.Module.__init__(m)
        m.normalized_shape = (dim,)
        m.eps = eps
        m.scale = rms.scale           # transplant the SAME Parameter object
        return m


def maybe_swap_rmsnorm_for_fused(model: nn.Module) -> int:
    """If ``TORCHTUNE_USE_FUSED_RMSNORM=1`` and Triton is available, recursively replace
    every :class:`RMSNorm` submodule in ``model`` with :class:`FusedRMSNorm`, transplanting
    each module's ``.scale`` parameter. Returns the number of modules swapped (0 if the flag
    is off or the fused path is unavailable). Call this BEFORE FSDP ``shard_model`` so the
    fused module's (transplanted) parameter is the one that gets sharded.
    """
    global _SWAP_LOG_DONE
    if not _USE_FUSED_RMSNORM:
        return 0
    if triton is None:
        if not _SWAP_LOG_DONE:
            _log.info("fused_rmsnorm=disabled (triton import failed)")
            _SWAP_LOG_DONE = True
        return 0

    to_swap = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, RMSNorm)
    ]
    for name, mod in to_swap:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], FusedRMSNorm.from_rms_norm(mod))

    if not _SWAP_LOG_DONE:
        _log.info("fused_rmsnorm=engaged (%d RMSNorm modules swapped)", len(to_swap))
        _SWAP_LOG_DONE = True
    return len(to_swap)


def _reset_swap_log_for_testing() -> None:
    global _SWAP_LOG_DONE
    _SWAP_LOG_DONE = False

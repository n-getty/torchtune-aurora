# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Split attention over a shared prefix, with an exact ring-style backward.

GRPO generates ``G`` continuations per prompt, so the policy forward recomputes each
prompt ``G`` times. At BioReason's production shape (prompt ~4096 tok, response ~1152
tok, G=8) that duplicated prefix work is roughly **69% of training compute**
(``memory/project_bioreason_grpo_prompt_recompute_dominates_20260915.md``).

This module provides the attention primitive needed to hoist it: run the prefix once,
then have every response token attend to the shared prefix plus its own response
history::

    cross block:  resp -> prefix    q=[B,H,R,D]  kv=[B,H,P,D]   is_causal=False
    self  block:  resp -> resp      q=[B,H,R,D]  kv=[B,H,R,D]   is_causal=True
    merge by log-sum-exp

Bottom-right causal alignment is not an extra correction here -- it falls out of the
decomposition, because "every response token sees all of the prefix" **is**
``is_causal=False``. That is precisely what the cached-KV reference variant could not
achieve: it needed an explicit bottom-right mask, which disqualified it from the fused
kernel and OOMed at 21 GiB inside SDPA
(``memory/project_bioreason_ref_prefix_share_oom_flash_ineligible_suffix_20260915.md``).

## Why the backward is not a custom kernel

The fused flash op returns a log-sum-exp that is **not differentiable**
(``lse.grad_fn is None`` while ``out.grad_fn`` is set, HW-confirmed job 8830106).
Merging with detached weights produces an *exact forward* with order-1 gradient error --
the worst possible failure shape, because every forward check passes.

The way out is that ``_scaled_dot_product_flash_attention_backward`` accepts ``out`` and
``logsumexp`` as **inputs**. Hand each block's backward the **merged** pair and the
gradients are exact (the ring-attention identity): the kernel reconstructs
``P = exp(qk*scale - lse)``, so a global ``lse`` makes each block's ``P`` the global
softmax restricted to that block's keys, and ``D = rowsum(grad_out * out)`` from the
global ``out`` is the correct global correction term. The merge is therefore never
differentiated through.

HW-validated on Aurora (job 8830160): ring gradients land at the bf16 noise floor
(worst rel 6.7e-3 against a measured floor of 5.65e-3) while a detached-LSE control is
13x worse; fwd+bwd is 1.74x faster than G full-length passes at G=8. See
``memory/project_bioreason_ring_backward_exact_dedup_unblocked_20260916.md``.

**These are kernel-level numbers.** End-to-end payoff is fbs-dependent (1.64x at
production fbs=2, requires ``G % fbs == 0``) and bounded by the fact that this backward
is measured *not* to be FLOP-bound. See :mod:`torchtune.dev.rl.group_chunking`.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch

log = logging.getLogger(__name__)

# head_dim values the Aurora fused flash kernel accepts. 256 is HW-CONFIRMED
# UNSUPPORTED -- sdpa_kernel([FLASH_ATTENTION]) raises "No available kernel" there
# regardless of GQA/layout/seqlen. See
# memory/project_qwen38_27b_head_dim256_flash_kernel_gap_20260902.md.
_FLASH_SUPPORTED_HEAD_DIMS = (64, 96, 128, 192)

_fused_unavailable_logged = False


def is_bshd(t: torch.Tensor) -> bool:
    """Whether ``t`` (logical ``[B, H, S, D]``) is stored BSHD, i.e. ``[B, S, H, D]``.

    A hard precondition of the Aurora fused flash kernel, and the one that is easiest to
    lose: ``expand``/``repeat_interleave``/``permute``/``reshape`` and even
    ``.contiguous()`` on an expanded view all produce BHSD-contiguous output, which the
    kernel rejects. It says so only as a stderr ``UserWarning``
    (``FlashAttentionXPU requires query, key, and value to be in BSHD layout``); the
    exception that reaches Python is the causeless ``No available kernel``. Cost three
    HW runs of the gate #3b probe to localize. See
    memory/feedback_replay_must_copy_the_helper_not_just_the_call_20260916.md.
    """
    return t.transpose(1, 2).is_contiguous()


def to_bshd(t: torch.Tensor) -> torch.Tensor:
    """Re-lay ``t`` into BSHD memory, a no-op when it already is.

    Mirrors ``torchtune/dev/bioreason/model.py::_to_bshd_memory``, which production's
    flash wrapper applies to q/k/v *after* its GQA KV repeat -- the step whose omission
    caused the probe failures above.
    """
    if t.transpose(1, 2).is_contiguous():
        return t
    return t.transpose(1, 2).contiguous().transpose(1, 2)


def fused_split_supported(q: torch.Tensor) -> tuple[bool, str]:
    """Whether the fused flash path can serve this tensor.

    Reports *every* failing condition rather than short-circuiting on the first. Device
    is usually the first thing to fail off-XPU, and short-circuiting there would make
    the dtype and head_dim guards unreachable — including from CPU tests, which is the
    only place they can be exercised cheaply.

    Args:
        q (torch.Tensor): query tensor, ``[B, H, S, D]``.

    Returns:
        tuple[bool, str]: ``(supported, reason)``; ``reason`` is empty when supported
        and otherwise lists all failing conditions, ``;``-separated.
    """
    reasons = []
    if q.device.type != "xpu":
        reasons.append(f"device={q.device.type} (fused flash path is XPU-only)")
    if q.dtype not in (torch.bfloat16, torch.float16):
        reasons.append(f"dtype={q.dtype} (fused flash needs bf16/fp16)")
    if q.shape[-1] not in _FLASH_SUPPORTED_HEAD_DIMS:
        reasons.append(
            f"head_dim={q.shape[-1]} not in {_FLASH_SUPPORTED_HEAD_DIMS} "
            "(head_dim=256 is HW-confirmed unsupported)"
        )
    return (not reasons), "; ".join(reasons)


def _merge_lse(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two attention blocks by log-sum-exp, accumulating in fp32.

    Args:
        out_a (torch.Tensor): first block output, ``[B, H, S, D]``.
        lse_a (torch.Tensor): first block log-sum-exp, ``[B, H, S]``.
        out_b (torch.Tensor): second block output, ``[B, H, S, D]``.
        lse_b (torch.Tensor): second block log-sum-exp, ``[B, H, S]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: merged ``(out, lse)``. ``lse`` is fp32 for
        reduced-precision inputs and keeps its own dtype at fp32 or above, so a float64
        reference path stays float64 rather than being silently truncated.
    """
    # Accumulate in at least fp32: the weights are exponentials of differences, and
    # bf16 here would cost more than the kernel's own error. Never DOWNcast an input
    # that is already wider.
    acc = lse_a.dtype if lse_a.dtype.itemsize >= 4 else torch.float32
    lse = torch.logaddexp(lse_a.to(acc), lse_b.to(acc))
    w_a = (lse_a.to(acc) - lse).exp().unsqueeze(-1)
    w_b = (lse_b.to(acc) - lse).exp().unsqueeze(-1)
    merged = out_a.to(acc) * w_a + out_b.to(acc) * w_b
    return merged.to(out_a.dtype), lse


class _SplitPrefixAttention(torch.autograd.Function):
    """Fused two-block attention whose backward reuses the stock flash backward op.

    Forward runs the cross and self blocks through the fused kernel and merges them.
    Backward calls the fused backward **twice with the merged out/lse**, then sums the
    query gradients. Nothing differentiates the merge itself, which is what makes the
    non-differentiable fused LSE a non-issue rather than a correctness hazard.
    """

    @staticmethod
    def forward(ctx, q, k_prefix, v_prefix, k_resp, v_resp, scale):
        flash = torch.ops.aten._scaled_dot_product_flash_attention
        with torch.no_grad():
            out_c, lse_c, cq_c, ck_c, mq_c, mk_c, seed_c, off_c, _ = flash(
                q, k_prefix, v_prefix, 0.0, False, False, scale=scale
            )
            out_s, lse_s, cq_s, ck_s, mq_s, mk_s, seed_s, off_s, _ = flash(
                q, k_resp, v_resp, 0.0, True, False, scale=scale
            )
            merged, lse = _merge_lse(out_c, lse_c, out_s, lse_s)

        ctx.save_for_backward(q, k_prefix, v_prefix, k_resp, v_resp, merged, lse)
        # cum_seq/max/philox are returned per block and must be replayed exactly.
        ctx.aux_cross = (cq_c, ck_c, mq_c, mk_c, seed_c, off_c)
        ctx.aux_self = (cq_s, ck_s, mq_s, mk_s, seed_s, off_s)
        ctx.scale = scale
        return merged, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        # grad_lse is intentionally ignored: the merged lse is a forward-only artifact
        # threaded into the backward calls, never a differentiable output. Callers must
        # not build loss terms on it -- see the module docstring.
        q, k_prefix, v_prefix, k_resp, v_resp, merged, lse = ctx.saved_tensors
        bwd = torch.ops.aten._scaled_dot_product_flash_attention_backward
        grad_out = grad_out.contiguous()

        cq_c, ck_c, mq_c, mk_c, seed_c, off_c = ctx.aux_cross
        cq_s, ck_s, mq_s, mk_s, seed_s, off_s = ctx.aux_self

        # THE identity: both calls receive the MERGED out and MERGED lse, not their own
        # block's. Passing per-block values silently yields wrong gradients.
        gq_c, gk_p, gv_p = bwd(
            grad_out, q, k_prefix, v_prefix, merged, lse,
            cq_c, ck_c, mq_c, mk_c, 0.0, False, seed_c, off_c, scale=ctx.scale,
        )
        gq_s, gk_r, gv_r = bwd(
            grad_out, q, k_resp, v_resp, merged, lse,
            cq_s, ck_s, mq_s, mk_s, 0.0, True, seed_s, off_s, scale=ctx.scale,
        )
        return gq_c + gq_s, gk_p, gv_p, gk_r, gv_r, None


def _reference_split_attention(q, k_prefix, v_prefix, k_resp, v_resp, scale):
    """Math fallback: the same split, expressed with autograd-visible ops.

    Used off-XPU and for any shape the fused kernel rejects. Mathematically identical
    to the fused path (both compute bottom-right-aligned causal attention over
    ``[prefix; resp]``), but materializes the score tensors, so it is only appropriate
    for tests and small shapes.
    """
    resp_len = q.shape[-2]
    prefix_len = k_prefix.shape[-2]

    scores_c = (q @ k_prefix.transpose(-1, -2)) * scale
    scores_s = (q @ k_resp.transpose(-1, -2)) * scale
    i = torch.arange(resp_len, device=q.device).unsqueeze(-1)
    j = torch.arange(resp_len, device=q.device).unsqueeze(0)
    scores_s = scores_s.masked_fill(j > i, float("-inf"))

    scores = torch.cat([scores_c, scores_s], dim=-1)
    # Match _merge_lse: promote to fp32, but never truncate an already-wider dtype.
    acc = scores.dtype if scores.dtype.itemsize >= 4 else torch.float32
    lse = torch.logsumexp(scores.to(acc), dim=-1)
    probs = scores.softmax(dim=-1)
    out = probs[..., :prefix_len] @ v_prefix + probs[..., prefix_len:] @ v_resp
    return out, lse


def split_prefix_attention(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_resp: torch.Tensor,
    v_resp: torch.Tensor,
    *,
    scale: Optional[float] = None,
    force_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention from response queries over a shared prefix plus causal response history.

    Equivalent to full attention over ``[prefix; resp]`` with a **bottom-right-aligned**
    causal mask, i.e. response token ``i`` attends to every prefix key and to response
    keys ``0..i``. The prefix KV is supplied separately so a caller can compute it once
    per prompt and broadcast it across that prompt's ``G`` continuations.

    Args:
        q (torch.Tensor): response queries, ``[B, H, R, D]``.
        k_prefix (torch.Tensor): prefix keys, ``[B, H, P, D]``.
        v_prefix (torch.Tensor): prefix values, ``[B, H, P, D]``.
        k_resp (torch.Tensor): response keys, ``[B, H, R, D]``.
        v_resp (torch.Tensor): response values, ``[B, H, R, D]``.
        scale (Optional[float]): softmax scale; defaults to ``1/sqrt(head_dim)``.
        force_reference (bool): use the math fallback even when the fused path is
            available. For tests and A/B equivalence checks.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(out, lse)`` with ``out`` shaped like ``q``
        and ``lse`` shaped ``[B, H, R]`` in fp32. **``lse`` is not differentiable** --
        it is exposed for diagnostics and for chaining further merges, never as a loss
        input.

    Raises:
        ValueError: if the GQA head counts of the prefix and response KV disagree, or
            if the query and response KV lengths differ (the self block must be square
            for the fused causal kernel to accept it).
    """
    if k_prefix.shape[1] != k_resp.shape[1]:
        raise ValueError(
            f"prefix/response KV head counts differ: {k_prefix.shape[1]} vs "
            f"{k_resp.shape[1]}; expand GQA heads before calling"
        )
    if q.shape[-2] != k_resp.shape[-2]:
        raise ValueError(
            f"query length {q.shape[-2]} != response KV length {k_resp.shape[-2]}; "
            "the self block must be square (the fused kernel rejects non-square "
            "is_causal=True, which is exactly the control this design relies on)"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    supported, reason = fused_split_supported(q)
    if force_reference or not supported:
        if not force_reference and q.device.type == "xpu":
            global _fused_unavailable_logged
            if not _fused_unavailable_logged:
                log.info("split_prefix_attention fused=skipped reason=%s", reason)
                _fused_unavailable_logged = True
        return _reference_split_attention(q, k_prefix, v_prefix, k_resp, v_resp, scale)

    # Re-lay-out to BSHD before the fused call. Callers reach this function through
    # expand/reshape/permute plumbing (the prefix broadcast, the GQA head fold), every
    # step of which produces BHSD-contiguous output that the kernel rejects. Doing it
    # here rather than at each call site means a new caller cannot reintroduce the bug,
    # and to_bshd is a no-op when the tensor is already correct so it costs nothing on
    # the path that was already right. See is_bshd for why this is not optional.
    q = to_bshd(q)
    k_prefix, v_prefix = to_bshd(k_prefix), to_bshd(v_prefix)
    k_resp, v_resp = to_bshd(k_resp), to_bshd(v_resp)

    return _SplitPrefixAttention.apply(q, k_prefix, v_prefix, k_resp, v_resp, scale)


def fold_groups_into_heads(q_rows: torch.Tensor, group_size: int) -> torch.Tensor:
    """Fold the rollout group ``G`` out of the batch axis and into the query-head axis.

    Turns ``[B*G, H, R, D]`` into ``[B, H*G, R, D]`` so the prefix KV can stay at ``B``
    rows and be consumed by native GQA (``enable_gqa=True``) instead of being
    materialized ``G`` times. At BioReason's production shape that is the difference
    between 8.0 MiB and 64.0 MiB of prefix KV per layer (HW-measured, job 8830340).

    **The head order is load-bearing and must be ``h*G + g``, not ``g*H + h``.** GQA maps
    query head ``i`` to KV head ``i // (n_q_total // n_kv)``; with ``h*G + g`` every
    continuation ``g`` of query head ``h`` lands in the same KV group as ``h`` did
    before the fold, so group ``g`` of prompt ``b`` reads prompt ``b``'s prefix. The
    natural ``reshape`` of a ``[B, G, H, ...]`` layout gives ``g*H + h`` instead, which
    routes every rollout to *another rollout's prompt*: HW-measured wrong by
    ``rel=8.5e-01`` -- yet finite, so it trains to garbage rather than crashing. A
    permute between the two reshapes is what fixes it, and it is a real copy
    (~604 MiB/layer at production shape, transient, freed per layer) -- ~8x cheaper than
    the ~34 GiB of duplicated prefix KV it avoids.

    Args:
        q_rows (torch.Tensor): per-rollout queries, ``[B*G, H, R, D]``, group-contiguous
            (the ``G`` continuations of prompt ``b`` occupy rows ``[b*G, (b+1)*G)``).
        group_size (int): ``G``, continuations per prompt.

    Returns:
        torch.Tensor: ``[B, H*G, R, D]``, query head ``h*G + g``.

    Raises:
        ValueError: if the leading dimension is not divisible by ``group_size``.
    """
    bg, h, r, d = q_rows.shape
    if bg % group_size:
        raise ValueError(
            f"leading dim {bg} is not divisible by group_size {group_size}; "
            "the fold assumes group-contiguous rollout layout"
        )
    b = bg // group_size
    # reshape, not view: q_rows is BSHD-strided on the fused path, so the B and G axes
    # are not adjacent in memory and view() raises. See
    # memory/feedback_probe_gates_need_independent_exception_boundaries_20260916.md.
    return (
        q_rows.reshape(b, group_size, h, r, d)
        .permute(0, 2, 1, 3, 4)  # (b, g, h, ...) -> (b, h, g, ...) == head h*G + g
        .reshape(b, h * group_size, r, d)
    )


def unfold_heads_into_groups(out: torch.Tensor, group_size: int) -> torch.Tensor:
    """Inverse of :func:`fold_groups_into_heads`: ``[B, H*G, R, D]`` -> ``[B*G, H, R, D]``.

    Args:
        out (torch.Tensor): folded attention output, ``[B, H*G, R, D]``.
        group_size (int): ``G``, the same value passed to the fold.

    Returns:
        torch.Tensor: ``[B*G, H, R, D]``, group-contiguous.

    Raises:
        ValueError: if the head dimension is not divisible by ``group_size``.
    """
    b, hg, r, d = out.shape
    if hg % group_size:
        raise ValueError(
            f"head dim {hg} is not divisible by group_size {group_size}; "
            "this tensor did not come from fold_groups_into_heads"
        )
    h = hg // group_size
    return (
        out.reshape(b, h, group_size, r, d)
        .permute(0, 2, 1, 3, 4)
        .reshape(b * group_size, h, r, d)
    )


def broadcast_prefix_kv(kv: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand per-prompt prefix KV across that prompt's continuations.

    GRPO lays rollouts out group-contiguously (the ``G`` continuations of prompt ``b``
    occupy rows ``[b*G, (b+1)*G)``), so this uses ``expand`` + ``reshape`` to match.

    Note the result is materialized: ``reshape`` on an expanded tensor copies, because
    the fused kernel needs real strides. The saving is that the prefix was *computed*
    once, not that it occupies one row of memory.

    **Prefer :func:`fold_groups_into_heads`** on the fused path: folding ``G`` into the
    query heads lets the prefix KV stay at ``B`` rows entirely (8.0 MiB vs 64.0 MiB at
    G=8, HW-measured). This function remains for the reference path and for callers that
    genuinely need ``B*G`` KV rows.

    Args:
        kv (torch.Tensor): prefix keys or values, ``[B, H, P, D]``.
        group_size (int): ``G``, continuations per prompt.

    Returns:
        torch.Tensor: ``[B*G, H, P, D]``, group-contiguous.
    """
    b, h, p, d = kv.shape
    return kv.unsqueeze(1).expand(b, group_size, h, p, d).reshape(b * group_size, h, p, d)

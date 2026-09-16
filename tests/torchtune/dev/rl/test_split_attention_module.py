# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU tests for `torchtune.dev.rl.split_attention`, the prefix-dedup primitive.

The algebra itself is pinned separately in ``test_split_attention_merge.py`` (float64,
from first principles, including the two negative controls). This file pins the
**module's contract**: that the public entry point computes bottom-right causal
attention, that it degrades to the math fallback instead of crashing on shapes the
fused kernel rejects, that its shape guards fire, and that the GQA/group broadcast
helper lays rows out the way GRPO does.

The fused XPU path cannot run here -- `fused_split_supported` returns False off-XPU, so
these exercise the reference branch. The fused branch's numerics are a hardware
question, gated by ``experiments/bioreason/probe_split_sdpa_ring_backward.py``
(job 8830160: ring gradients at the bf16 noise floor, detached-LSE control 13x worse).
"""
import math

import pytest
import torch

from torchtune.dev.rl.split_attention import (
    _merge_lse,
    broadcast_prefix_kv,
    fused_split_supported,
    split_prefix_attention,
)

B, H, PREFIX, RESP, D = 2, 3, 12, 5, 8
SCALE = D**-0.5


def _qkv(seed: int, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda s: torch.randn(B, H, s, D, generator=g, dtype=dtype)  # noqa: E731
    return mk(RESP), mk(PREFIX), mk(PREFIX), mk(RESP), mk(RESP), mk(RESP)


def _full_bottom_right(q, k_prefix, v_prefix, k_resp, v_resp):
    """Ground truth: one monolithic attention over [prefix; resp], bottom-right causal."""
    k = torch.cat([k_prefix, k_resp], dim=-2)
    v = torch.cat([v_prefix, v_resp], dim=-2)
    scores = (q @ k.transpose(-1, -2)) * SCALE
    i = torch.arange(RESP).unsqueeze(-1)
    j = torch.arange(PREFIX + RESP).unsqueeze(0)
    scores = scores.masked_fill(j > i + PREFIX, float("-inf"))
    return scores.softmax(dim=-1) @ v, torch.logsumexp(scores, dim=-1)


class TestForwardMatchesMonolithic:
    def test_out_and_lse_match_bottom_right_attention(self):
        q, kp, vp, kr, vr, _ = _qkv(0)
        out, lse = split_prefix_attention(q, kp, vp, kr, vr, scale=SCALE)
        ref_out, ref_lse = _full_bottom_right(q, kp, vp, kr, vr)
        assert torch.allclose(out, ref_out, atol=1e-12), (
            f"split forward differs from monolithic bottom-right causal attention "
            f"by {(out - ref_out).abs().max().item():.3e}"
        )
        assert torch.allclose(lse, ref_lse, atol=1e-12)

    def test_is_not_top_left_aligned(self):
        """Guard: top-left alignment is the classic wrong answer and must not pass.

        With ``is_causal=True`` over the concatenated sequence the kernel would align
        top-left, letting response token i see only keys 0..i of the *prefix* -- a
        silently plausible but wrong result. Assert the two genuinely differ so a
        refactor that reintroduces top-left cannot slip past the test above.
        """
        q, kp, vp, kr, vr, _ = _qkv(1)
        out, _ = split_prefix_attention(q, kp, vp, kr, vr, scale=SCALE)
        k = torch.cat([kp, kr], dim=-2)
        v = torch.cat([vp, vr], dim=-2)
        scores = (q @ k.transpose(-1, -2)) * SCALE
        i = torch.arange(RESP).unsqueeze(-1)
        j = torch.arange(PREFIX + RESP).unsqueeze(0)
        top_left = (scores.masked_fill(j > i, float("-inf"))).softmax(dim=-1) @ v
        assert not torch.allclose(out, top_left, atol=1e-6), (
            "split output matches TOP-LEFT causal attention; the test fixture is "
            "degenerate and would not catch a real alignment regression"
        )

    def test_default_scale_is_inverse_sqrt_head_dim(self):
        q, kp, vp, kr, vr, _ = _qkv(2)
        a, _ = split_prefix_attention(q, kp, vp, kr, vr)
        b, _ = split_prefix_attention(q, kp, vp, kr, vr, scale=1.0 / math.sqrt(D))
        assert torch.equal(a, b)

    @pytest.mark.parametrize("prefix_len", [1, 4, 17])
    def test_varying_prefix_lengths(self, prefix_len):
        g = torch.Generator().manual_seed(3)
        mk = lambda s: torch.randn(B, H, s, D, generator=g, dtype=torch.float64)  # noqa: E731
        q, kr, vr = mk(RESP), mk(RESP), mk(RESP)
        kp, vp = mk(prefix_len), mk(prefix_len)
        out, _ = split_prefix_attention(q, kp, vp, kr, vr, scale=SCALE)
        k = torch.cat([kp, kr], dim=-2)
        v = torch.cat([vp, vr], dim=-2)
        scores = (q @ k.transpose(-1, -2)) * SCALE
        i = torch.arange(RESP).unsqueeze(-1)
        j = torch.arange(prefix_len + RESP).unsqueeze(0)
        ref = (scores.masked_fill(j > i + prefix_len, float("-inf"))).softmax(-1) @ v
        assert torch.allclose(out, ref, atol=1e-12)


class TestGradientsFlow:
    def test_reference_branch_is_differentiable_and_correct(self):
        """The fallback must produce real gradients, not merely a matching forward.

        A forward-only check would pass even for a detached merge -- exactly the trap
        documented in the module docstring -- so compare gradients against the
        monolithic reference.
        """
        q, kp, vp, kr, vr, g_out = _qkv(4)
        tensors = [t.clone().requires_grad_(True) for t in (q, kp, vp, kr, vr)]
        out, _ = split_prefix_attention(*tensors, scale=SCALE)
        out.backward(g_out)

        ref_tensors = [t.clone().requires_grad_(True) for t in (q, kp, vp, kr, vr)]
        ref_out, _ = _full_bottom_right(*ref_tensors)
        ref_out.backward(g_out)

        for name, got, want in zip("q kp vp kr vr".split(), tensors, ref_tensors):
            assert got.grad is not None, f"{name} received no gradient"
            assert torch.allclose(got.grad, want.grad, atol=1e-10), (
                f"grad_{name} differs by {(got.grad - want.grad).abs().max().item():.3e}"
            )


class TestGuards:
    def test_mismatched_kv_head_counts_raise(self):
        q, kp, vp, kr, vr, _ = _qkv(5)
        kp = kp[:, :1]
        vp = vp[:, :1]
        with pytest.raises(ValueError, match="head counts differ"):
            split_prefix_attention(q, kp, vp, kr, vr, scale=SCALE)

    def test_non_square_self_block_raises(self):
        """The self block must be square: the fused kernel rejects non-square causal.

        That rejection is load-bearing -- it is the negative control that proves the
        cross block reaches flash only because it is is_causal=False. A silent
        fallback here would hide a caller bug.
        """
        q, kp, vp, kr, vr, _ = _qkv(6)
        with pytest.raises(ValueError, match="must be square"):
            split_prefix_attention(q, kp, vp, kr[:, :, :-1], vr[:, :, :-1], scale=SCALE)

    def test_cpu_is_not_fused_supported(self):
        q, *_ = _qkv(7)
        supported, reason = fused_split_supported(q)
        assert not supported and "XPU-only" in reason

    def test_head_dim_256_is_rejected(self):
        """head_dim=256 is HW-confirmed unsupported; the guard must not claim otherwise."""
        q = torch.zeros(1, 1, 4, 256, dtype=torch.bfloat16)
        supported, reason = fused_split_supported(q)
        assert not supported and "head_dim=256" in reason

    def test_fp32_is_rejected_by_the_fused_guard(self):
        q = torch.zeros(1, 1, 4, 128, dtype=torch.float32)
        supported, reason = fused_split_supported(q)
        assert not supported and "bf16" in reason

    def test_force_reference_matches_default_cpu_path(self):
        q, kp, vp, kr, vr, _ = _qkv(8)
        a, _ = split_prefix_attention(q, kp, vp, kr, vr, scale=SCALE)
        b, _ = split_prefix_attention(
            q, kp, vp, kr, vr, scale=SCALE, force_reference=True
        )
        assert torch.equal(a, b)


class TestMergeAndBroadcast:
    def test_merge_of_disjoint_blocks_equals_joint_softmax(self):
        g = torch.Generator().manual_seed(9)
        q = torch.randn(B, H, RESP, D, generator=g, dtype=torch.float64)
        k1 = torch.randn(B, H, 6, D, generator=g, dtype=torch.float64)
        v1 = torch.randn(B, H, 6, D, generator=g, dtype=torch.float64)
        k2 = torch.randn(B, H, 4, D, generator=g, dtype=torch.float64)
        v2 = torch.randn(B, H, 4, D, generator=g, dtype=torch.float64)

        def attn(k, v):
            s = (q @ k.transpose(-1, -2)) * SCALE
            return s.softmax(-1) @ v, torch.logsumexp(s, dim=-1)

        o1, l1 = attn(k1, v1)
        o2, l2 = attn(k2, v2)
        merged, lse = _merge_lse(o1, l1, o2, l2)
        ref_out, ref_lse = attn(torch.cat([k1, k2], -2), torch.cat([v1, v2], -2))
        assert torch.allclose(merged, ref_out, atol=1e-12)
        assert torch.allclose(lse, ref_lse, atol=1e-12)

    def test_broadcast_is_group_contiguous(self):
        """GRPO puts prompt b's G continuations at rows [b*G, (b+1)*G).

        If this were group-STRIDED instead, every rollout would attend to the wrong
        prompt's prefix -- a correctness failure that produces finite, plausible
        losses, so it must be pinned by layout rather than by a smoke test.
        """
        kv = torch.arange(3 * 2 * 4 * 2, dtype=torch.float32).reshape(3, 2, 4, 2)
        out = broadcast_prefix_kv(kv, group_size=5)
        assert out.shape == (15, 2, 4, 2)
        for b in range(3):
            for g in range(5):
                assert torch.equal(out[b * 5 + g], kv[b]), (
                    f"row {b * 5 + g} should carry prompt {b}'s prefix; layout is not "
                    "group-contiguous"
                )

    def test_broadcast_group_size_one_is_identity(self):
        kv = torch.randn(4, 2, 3, 8)
        assert torch.equal(broadcast_prefix_kv(kv, 1), kv)

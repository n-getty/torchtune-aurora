# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU/float64 tests for the split-attention (prefix dedup) math.

Two separate claims are pinned here, and they fail for different reasons:

1. **Forward**: splitting attention into a cross block (``resp -> prefix``,
   ``is_causal=False``) and a self block (``resp -> resp``, ``is_causal=True``) and
   merging by log-sum-exp reproduces full **bottom-right-aligned** causal attention.
   Bottom-right alignment is not an extra correction -- it falls out of the split,
   because "every response token attends to all of the prefix" IS ``is_causal=False``.

2. **Backward (the ring identity)**: reconstructing each block's gradient from the
   **merged** ``out``/``lse`` -- rather than the block's own -- gives the true gradient.
   This is what lets the implementation call the stock fused backward op twice and sum,
   with no autograd through the merge weights. The fused LSE is non-differentiable
   (``lse.grad_fn is None`` while ``out.grad_fn`` is set, HW-confirmed job 8830106), so
   a naive detached-weight merge yields an *exact forward* and badly wrong gradients --
   the worst failure shape, since every forward check passes. ``test_detached_lse_merge_
   is_wrong`` asserts that trap is real, so a future refactor cannot quietly reintroduce it.

These use float64 on CPU: the point is the algebra, not kernel behavior. Whether the XPU
kernel actually honors a caller-supplied lse is a hardware question, gated separately by
``experiments/bioreason/probe_split_sdpa_ring_backward.py``.
"""
import pytest
import torch

B, H, PREFIX, RESP, D = 2, 3, 16, 7, 8
SCALE = D**-0.5


def _qkv(seed: int):
    g = torch.Generator().manual_seed(seed)
    mk = lambda s: torch.randn(B, H, s, D, generator=g, dtype=torch.float64)  # noqa: E731
    return mk(RESP), mk(PREFIX), mk(PREFIX), mk(RESP), mk(RESP), mk(RESP)


def _attn(q, k, v, mask=None):
    """Reference attention returning (out, lse) in float64."""
    scores = (q @ k.transpose(-1, -2)) * SCALE
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    return scores.softmax(dim=-1) @ v, lse


def _causal_mask(q_len, kv_len):
    i = torch.arange(q_len).unsqueeze(-1)
    j = torch.arange(kv_len).unsqueeze(0)
    return j > i


def _bottom_right_mask():
    """Response token i may see all PREFIX keys plus response keys up to i."""
    i = torch.arange(RESP).unsqueeze(-1)
    j = torch.arange(PREFIX + RESP).unsqueeze(0)
    return j > i + PREFIX


def _merge(out_a, lse_a, out_b, lse_b):
    lse = torch.logaddexp(lse_a, lse_b)
    wa = (lse_a - lse).exp().unsqueeze(-1)
    wb = (lse_b - lse).exp().unsqueeze(-1)
    return out_a * wa + out_b * wb, lse


def _split_forward(q, k_p, v_p, k_r, v_r):
    out_c, lse_c = _attn(q, k_p, v_p)  # cross: no mask == is_causal=False
    out_s, lse_s = _attn(q, k_r, v_r, _causal_mask(RESP, RESP))
    return _merge(out_c, lse_c, out_s, lse_s)


def _monolithic_forward(q, k_p, v_p, k_r, v_r):
    return _attn(
        q,
        torch.cat([k_p, k_r], dim=2),
        torch.cat([v_p, v_r], dim=2),
        _bottom_right_mask(),
    )


class TestSplitForward:
    def test_split_merge_equals_bottom_right_causal(self):
        q, k_p, v_p, k_r, v_r, _ = _qkv(0)
        got, got_lse = _split_forward(q, k_p, v_p, k_r, v_r)
        ref, ref_lse = _monolithic_forward(q, k_p, v_p, k_r, v_r)
        assert torch.allclose(got, ref, atol=1e-12), (got - ref).abs().max()
        assert torch.allclose(got_lse, ref_lse, atol=1e-12)

    def test_split_is_not_top_left_causal(self):
        """Guard: the split must NOT reproduce top-left alignment.

        ``is_causal=True`` over the concatenated sequence is top-left-aligned and is the
        wrong mask here. If a refactor makes these agree, the split silently lost the
        bottom-right semantics -- which is exactly the drift that corrupted the cached
        reference path (~7e-3 on a float64 model).
        """
        q_full = torch.randn(B, H, PREFIX + RESP, D, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        q, k_p, v_p, k_r, v_r, _ = _qkv(0)
        top_left, _ = _attn(
            q_full,
            torch.cat([k_p, k_r], dim=2),
            torch.cat([v_p, v_r], dim=2),
            _causal_mask(PREFIX + RESP, PREFIX + RESP),
        )
        split, _ = _split_forward(q, k_p, v_p, k_r, v_r)
        assert top_left.shape[2] != split.shape[2]

    @pytest.mark.parametrize("prefix_len", [1, 5, 16, 33])
    def test_merge_holds_across_prefix_lengths(self, prefix_len):
        global PREFIX
        original, PREFIX = PREFIX, prefix_len
        try:
            q, k_p, v_p, k_r, v_r, _ = _qkv(2)
            got, _ = _split_forward(q, k_p, v_p, k_r, v_r)
            ref, _ = _monolithic_forward(q, k_p, v_p, k_r, v_r)
            assert torch.allclose(got, ref, atol=1e-12)
        finally:
            PREFIX = original


class TestRingBackward:
    """The identity that lets us reuse the stock fused backward op."""

    @staticmethod
    def _reference_grads(q, k_p, v_p, k_r, v_r, grad_out):
        leaves = [t.clone().requires_grad_(True) for t in (q, k_p, v_p, k_r, v_r)]
        out, _ = _monolithic_forward(*leaves)
        (out * grad_out).sum().backward()
        return [t.grad for t in leaves]

    @staticmethod
    def _block_grads(q, k, v, grad_out, merged_out, merged_lse, mask):
        """Gradient of one block, reconstructed from the MERGED out/lse.

        This mirrors what ``_scaled_dot_product_flash_attention_backward`` does when
        handed a caller-supplied lse: P = exp(scores - lse) is the *global* softmax
        restricted to this block's keys, and D = rowsum(grad_out * merged_out) is the
        global correction term.
        """
        scores = (q @ k.transpose(-1, -2)) * SCALE
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        p = (scores - merged_lse.unsqueeze(-1)).exp()
        grad_v = p.transpose(-1, -2) @ grad_out
        dp = grad_out @ v.transpose(-1, -2)
        d = (grad_out * merged_out).sum(dim=-1, keepdim=True)
        ds = p * (dp - d) * SCALE
        return ds @ k, ds.transpose(-1, -2) @ q, grad_v

    def test_ring_backward_is_exact(self):
        q, k_p, v_p, k_r, v_r, grad_out = _qkv(3)
        merged_out, merged_lse = _split_forward(q, k_p, v_p, k_r, v_r)

        gq_c, gk_p, gv_p = self._block_grads(
            q, k_p, v_p, grad_out, merged_out, merged_lse, None
        )
        gq_s, gk_r, gv_r = self._block_grads(
            q, k_r, v_r, grad_out, merged_out, merged_lse, _causal_mask(RESP, RESP)
        )
        got = [gq_c + gq_s, gk_p, gv_p, gk_r, gv_r]

        ref = self._reference_grads(q, k_p, v_p, k_r, v_r, grad_out)
        for name, g, r in zip(("q", "k_p", "v_p", "k_r", "v_r"), got, ref):
            assert torch.allclose(g, r, atol=1e-10), f"{name}: {(g - r).abs().max()}"

    def test_per_block_lse_backward_is_wrong(self):
        """Using each block's OWN lse (instead of the merged one) must be wrong.

        Pins why the merged lse has to be threaded into both backward calls: it is not a
        cosmetic detail, it is the entire correctness argument.
        """
        q, k_p, v_p, k_r, v_r, grad_out = _qkv(4)
        out_c, lse_c = _attn(q, k_p, v_p)
        out_s, lse_s = _attn(q, k_r, v_r, _causal_mask(RESP, RESP))

        gq_c, _, _ = self._block_grads(q, k_p, v_p, grad_out, out_c, lse_c, None)
        gq_s, _, _ = self._block_grads(
            q, k_r, v_r, grad_out, out_s, lse_s, _causal_mask(RESP, RESP)
        )
        ref = self._reference_grads(q, k_p, v_p, k_r, v_r, grad_out)
        assert not torch.allclose(gq_c + gq_s, ref[0], atol=1e-6)

    def test_detached_lse_merge_is_wrong(self):
        """The documented trap: exact forward, badly wrong gradients.

        Treating the log-sum-exp merge weights as constants gives a forward that matches
        to float64 precision while the gradient is off by order 1. HW-confirmed at ~0.6
        absolute error; this pins the mechanism on CPU so it cannot silently return.
        """
        q, k_p, v_p, k_r, v_r, grad_out = _qkv(5)
        leaves = [t.clone().requires_grad_(True) for t in (q, k_p, v_p, k_r, v_r)]
        lq, lk_p, lv_p, lk_r, lv_r = leaves

        out_c, lse_c = _attn(lq, lk_p, lv_p)
        out_s, lse_s = _attn(lq, lk_r, lv_r, _causal_mask(RESP, RESP))
        # The trap: weights computed under no_grad, so autograd sees them as constants.
        with torch.no_grad():
            lse = torch.logaddexp(lse_c, lse_s)
            wa = (lse_c - lse).exp().unsqueeze(-1)
            wb = (lse_s - lse).exp().unsqueeze(-1)
        merged = out_c * wa + out_s * wb

        ref_out, _ = _monolithic_forward(q, k_p, v_p, k_r, v_r)
        assert torch.allclose(merged, ref_out, atol=1e-12), "forward must still be exact"

        (merged * grad_out).sum().backward()
        ref = self._reference_grads(q, k_p, v_p, k_r, v_r, grad_out)
        assert not torch.allclose(lq.grad, ref[0], atol=1e-6), (
            "detached-LSE merge must produce WRONG gradients -- if this passes, the "
            "trap has been silently fixed and the guard needs revisiting"
        )

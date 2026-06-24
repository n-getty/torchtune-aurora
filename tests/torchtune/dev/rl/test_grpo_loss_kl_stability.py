# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""KL-estimator numerical stability for GRPO losses.

The k3 KL estimator exp(d) - d - 1 with d = ref_logprobs - pi_logprobs blows up
to Inf/NaN when d is large. logprobs come from log_softmax (always finite, never
raw -inf), but over long generations (2048 tokens) a rare token can get a very
negative policy logprob (e.g. -80) while the ref is normal → d ≈ +80 → exp(d) =
Inf → the masked_mean is Inf/NaN and poisons the whole step. This was hit on a
BioReason 2048-gen run (2026-06-18, step 4 → loss NaN, persisted). The pre-fix
code clamped only the max side but a NaN (e.g. inf-inf from two extreme tokens)
slips through clamp. The fix scrubs NaN and clamps both sides.

These tests use FINITE-BUT-EXTREME logprobs (what log_softmax actually produces),
not raw -inf, and pin that the loss + KL stay finite.
"""
import pytest

torch = pytest.importorskip("torch")

from torchtune.dev.rl.loss import GRPOSimpleLoss, GRPOLoss


def _inputs():
    B, L = 4, 8
    pi_old = torch.randn(B, L) * 0.1
    pi = torch.randn(B, L) * 0.1
    ref = torch.randn(B, L) * 0.1
    adv = torch.randn(B)
    mask = torch.ones(B, L, dtype=torch.bool)
    return pi_old, pi, ref, adv, mask


@pytest.mark.parametrize("loss_cls", [GRPOSimpleLoss, GRPOLoss])
def test_kl_finite_on_extreme_negative_pi_logprob(loss_cls):
    """pi very negative (e.g. -80, a near-zero-prob token) while ref normal →
    d ≈ +80 → exp(d) overflows to Inf pre-fix. Must stay finite."""
    pi_old, pi, ref, adv, mask = _inputs()
    pi[0, 3] = -80.0   # finite, but log_softmax-plausible on a long sequence
    loss_fn = loss_cls(kl_coeff=1e-3)
    out = loss_fn(pi_old, pi, ref, adv, padding_masks=mask)
    loss, _policy, kl = out[0], out[1], out[2]
    assert torch.isfinite(loss).all(), f"{loss_cls.__name__}: loss not finite"
    assert torch.isfinite(kl).all(), f"{loss_cls.__name__}: kl not finite"


@pytest.mark.parametrize("loss_cls", [GRPOSimpleLoss, GRPOLoss])
def test_kl_finite_on_extreme_both_directions(loss_cls):
    """Extreme d in both signs across tokens — the masked_mean must be finite."""
    pi_old, pi, ref, adv, mask = _inputs()
    pi[1, 2] = -90.0
    ref[1, 5] = -90.0
    loss_fn = loss_cls(kl_coeff=1e-3)
    out = loss_fn(pi_old, pi, ref, adv, padding_masks=mask)
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[2]).all()


@pytest.mark.parametrize("loss_cls", [GRPOSimpleLoss, GRPOLoss])
def test_kl_matches_baseline_on_normal_inputs(loss_cls):
    """Hardening must NOT change the KL on well-behaved inputs: a normal d in
    ~[-1,1] is far inside the [-20,20] clamp and has no NaN/Inf, so the result
    equals the plain k3 estimator."""
    pi_old, pi, ref, adv, mask = _inputs()
    loss_fn = loss_cls(kl_coeff=1e-3)
    _loss, _policy, kl = loss_fn(pi_old, pi, ref, adv, padding_masks=mask)[:3]
    # reference k3 on the same (finite) inputs
    d = (ref - pi)
    expected_kl = ((torch.exp(d) - d - 1) * mask).sum() / mask.sum()
    torch.testing.assert_close(kl, expected_kl, rtol=1e-5, atol=1e-5)

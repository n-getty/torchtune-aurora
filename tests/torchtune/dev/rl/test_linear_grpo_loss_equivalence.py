# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU-safe equivalence tests for the chunked-vocab LinearGRPOLoss path used by the
LoRA-GRPO XPU recipe (memory-efficient training forward).

Pins down that LinearGRPOLoss (hidden -> per-seq-chunk projection -> CE logprobs ->
GRPOSimple formulation) is numerically equivalent to the reference full-logit path
(proj(hidden) -> batched_logits_to_logprobs -> GRPOSimpleLoss), including gradients,
while never materializing the full [B, S, vocab] logit tensor.

No XPU / distributed / IPEX required. Tiny shapes. Runs in <1 s on CPU.
"""
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtune import rlhf
from torchtune.dev.rl.linear_grpo_loss import LinearGRPOLoss
from torchtune.dev.rl.loss import GRPOSimpleLoss


# Tiny dims. S=12 with k=8 chunks exercises uneven tensor_split (12 = 2,2,2,2,1,1,1,1).
EMB = 8
VOCAB = 16
B = 4          # B*G flattened
S = 12         # response length
KL_COEFF = 0.1


def _make_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, S, EMB, generator=g, dtype=torch.float32)
    targets = torch.randint(0, VOCAB, (B, S), generator=g)
    ref_logprobs = torch.randn(B, S, generator=g) * 0.5
    advantages = torch.randn(B, generator=g)
    padding_masks = torch.ones(B, S, dtype=torch.bool)
    # mark a couple trailing tokens as padding (excluded from loss) on some rows
    padding_masks[0, -2:] = False
    padding_masks[2, -1:] = False
    return hidden, targets, ref_logprobs, advantages, padding_masks


def _reference(proj, hidden, targets, ref_logprobs, advantages, padding_masks, temperature=1.0):
    """Full-logit reference: proj(hidden) -> logprobs(/T) -> GRPOSimpleLoss."""
    logits = proj(hidden)  # [B, S, vocab] FP32 — the tensor we want to AVOID at scale
    # batched_logits_to_logprobs == log_softmax(logits / T) + gather
    pi_logprobs = rlhf.batched_logits_to_logprobs(logits, targets, temperature)
    loss_fn = GRPOSimpleLoss(kl_coeff=KL_COEFF)
    loss, policy_loss, kl_loss, _, _ = loss_fn(
        pi_logprobs.detach(),   # pi_old (unused by GRPOSimpleLoss)
        pi_logprobs,
        ref_logprobs,
        advantages,
        padding_masks=padding_masks,
    )
    return loss, policy_loss, kl_loss, pi_logprobs


class TestLinearGRPOLossEquivalence(unittest.TestCase):
    def _proj(self, seed: int = 1):
        torch.manual_seed(seed)
        return nn.Linear(EMB, VOCAB, bias=False)

    def test_forward_equivalence_across_chunk_counts(self):
        """loss / policy / kl / pi_logprobs match the full-logit reference for k in {1,2,4,8}."""
        for k in (1, 2, 4, 8):
            with self.subTest(num_output_chunks=k):
                proj = self._proj()
                hidden, targets, ref_lp, adv, pmask = _make_inputs()

                ref_loss, ref_pol, ref_kl, ref_pi = _reference(
                    proj, hidden, targets, ref_lp, adv, pmask
                )

                loss_fn = LinearGRPOLoss(num_output_chunks=k, kl_coeff=KL_COEFF)
                # mimic set_model_output without a full model
                loss_fn.linear_projection = proj
                lin_loss, lin_pol, lin_kl, _, _, lin_pi = loss_fn(
                    hidden, targets, ref_lp, adv, padding_masks=pmask
                )

                torch.testing.assert_close(lin_loss, ref_loss, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(lin_pol, ref_pol, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(lin_kl, ref_kl, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(lin_pi, ref_pi, atol=1e-5, rtol=1e-4)

    def test_temperature_equivalence(self):
        """With matched temperature, LinearGRPOLoss matches the /T reference (T in {0.7, 0.8, 1.3})."""
        for temp in (0.7, 0.8, 1.3):
            with self.subTest(temperature=temp):
                proj = self._proj()
                hidden, targets, ref_lp, adv, pmask = _make_inputs()

                ref_loss, ref_pol, ref_kl, ref_pi = _reference(
                    proj, hidden, targets, ref_lp, adv, pmask, temperature=temp
                )

                loss_fn = LinearGRPOLoss(
                    num_output_chunks=4, kl_coeff=KL_COEFF, temperature=temp
                )
                loss_fn.linear_projection = proj
                lin_loss, _, lin_kl, _, _, lin_pi = loss_fn(
                    hidden, targets, ref_lp, adv, padding_masks=pmask
                )

                torch.testing.assert_close(lin_pi, ref_pi, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(lin_loss, ref_loss, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(lin_kl, ref_kl, atol=1e-5, rtol=1e-4)

    def test_temperature_default_is_one(self):
        """Default temperature is 1.0 (no scaling) so existing behavior is unchanged."""
        loss_fn = LinearGRPOLoss(num_output_chunks=4, kl_coeff=KL_COEFF)
        self.assertEqual(loss_fn.temperature, 1.0)

    def test_gradient_equivalence(self):
        """Backward through hidden: hidden.grad matches the reference (grad path preserved)."""
        proj = self._proj()
        hidden, targets, ref_lp, adv, pmask = _make_inputs()

        h_ref = hidden.clone().requires_grad_(True)
        ref_loss, *_ = _reference(proj, h_ref, targets, ref_lp, adv, pmask)
        ref_loss.backward()

        h_lin = hidden.clone().requires_grad_(True)
        loss_fn = LinearGRPOLoss(num_output_chunks=4, kl_coeff=KL_COEFF)
        loss_fn.linear_projection = proj
        lin_loss, *_ = loss_fn(h_lin, targets, ref_lp, adv, padding_masks=pmask)
        lin_loss.backward()

        self.assertIsNotNone(h_lin.grad)
        torch.testing.assert_close(h_lin.grad, h_ref.grad, atol=1e-5, rtol=1e-4)

    def test_never_materializes_full_seq_vocab(self):
        """The per-chunk projection input seq-dim must be <= ceil(S/k), never full S."""
        k = 4
        proj = self._proj()
        hidden, targets, ref_lp, adv, pmask = _make_inputs()
        loss_fn = LinearGRPOLoss(num_output_chunks=k, kl_coeff=KL_COEFF)
        loss_fn.linear_projection = proj

        seen_seq_dims = []
        real_proj_forward = proj.forward

        def _spy(x):
            # x is [chunk_b, chunk_s, EMB]
            seen_seq_dims.append(x.shape[1])
            return real_proj_forward(x)

        with mock.patch.object(proj, "forward", side_effect=_spy):
            loss_fn(hidden, targets, ref_lp, adv, padding_masks=pmask)

        self.assertTrue(seen_seq_dims, "projection was never called")
        import math
        max_allowed = math.ceil(S / k)
        self.assertLessEqual(
            max(seen_seq_dims), max_allowed,
            f"a chunk projected seq-dim {max(seen_seq_dims)} > ceil(S/k)={max_allowed} "
            "(full-vocab materialization not actually chunked)",
        )

    def test_kl_hardening_finite_on_inf_diff(self):
        """A token with ref-pi -> +inf must not NaN/Inf the loss (KL clamp+nan_to_num)."""
        proj = self._proj()
        hidden, targets, ref_lp, adv, pmask = _make_inputs()
        # Force a huge positive ref logprob on one token -> d = ref - pi explodes.
        ref_lp = ref_lp.clone()
        ref_lp[1, 3] = 1e9
        loss_fn = LinearGRPOLoss(num_output_chunks=4, kl_coeff=KL_COEFF)
        loss_fn.linear_projection = proj
        lin_loss, _, lin_kl, _, _, _ = loss_fn(
            hidden, targets, ref_lp, adv, padding_masks=pmask
        )
        self.assertTrue(torch.isfinite(lin_loss).all(), "loss not finite under inf KL diff")
        self.assertTrue(torch.isfinite(lin_kl).all(), "kl not finite under inf KL diff")


if __name__ == "__main__":
    unittest.main()

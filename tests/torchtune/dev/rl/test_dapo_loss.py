# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe tests for the DAPO/dr_grpo aggregation in GRPOSimpleLoss.

Mirrors trl/trainer/grpo_trainer.py:2387-2407 so that switching
``loss_type`` between {"grpo", "dapo", "bnpo", "dr_grpo"} gives the same
numerical result TRL would on the same inputs.

DAPO is the default TRL/ezpz loss type since 2025; matching it is the
prerequisite for closing the convergence-rate gap to ezpz that the
post-Q/K and post-IS-correction bake-offs left open. See
`memory/project_vllm_is_correction_phase4_results.md`.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

from torchtune.dev.rl.loss import GRPOSimpleLoss


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_RECIPE = REPO_ROOT / "recipes/dev/grpo_full_finetune_distributed_xpu.py"


def _trl_grpo_loss(per_token_loss, mask):
    """TRL `loss_type="grpo"` reduction at trl/.../grpo_trainer.py:2389."""
    return ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()


def _trl_bnpo_loss(per_token_loss, mask):
    """TRL `loss_type="bnpo"` reduction (grpo_trainer.py:2393)."""
    return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)


def _trl_dr_grpo_loss(per_token_loss, mask, max_completion_length):
    """TRL `loss_type="dr_grpo"` reduction (grpo_trainer.py:2397)."""
    return (per_token_loss * mask).sum() / (
        per_token_loss.size(0) * max_completion_length
    )


def _trl_dapo_loss(per_token_loss, mask, num_items_in_batch):
    """TRL `loss_type="dapo"` reduction (grpo_trainer.py:2401-2402).

    Single-process equivalent (num_processes=1 → normalizer == num_items).
    """
    return (per_token_loss * mask).sum() / num_items_in_batch


class TestLossTypeMatchesTRL(unittest.TestCase):
    """For each loss_type, our GRPOSimpleLoss must produce a result
    numerically equal to TRL's same-name reduction on identical inputs.
    With kl_coeff=0 and num_iterations=1 the per_token_loss reduces to
    -(exp(0) * advantages) = -advantages, so we can build the same
    per_token_loss tensor on both sides and compare reductions.
    """

    def _make_inputs(self, bsz=4, T=8, seed=0):
        torch.manual_seed(seed)
        # Variable per-row lengths to exercise the length-bias differences
        # between grpo and dapo.
        pi = torch.randn(bsz, T, requires_grad=True)
        pi_old = pi.detach().clone()
        ref = pi.detach().clone()
        adv = torch.tensor([1.0, -0.5, 0.7, -0.3])
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],  # T=8
            [1, 1, 1, 1, 0, 0, 0, 0],  # T=4
            [1, 1, 0, 0, 0, 0, 0, 0],  # T=2
            [1, 1, 1, 1, 1, 1, 0, 0],  # T=6
        ], dtype=torch.bool)
        return pi, pi_old, ref, adv, mask

    def _trl_per_token_loss(self, advantages, mask):
        """Reproduce TRL's per_token_loss at num_iterations=1, beta=0:
        coef_1 = exp(per_token_logps - per_token_logps.detach()) ≡ 1
        per_token_loss = -min(coef_1*adv, coef_2*adv) = -adv (broadcast)
        """
        return -(advantages[:, None].expand_as(mask).float())

    def test_grpo_matches(self):
        loss_fn = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0, loss_type="grpo")
        pi, pi_old, ref, adv, mask = self._make_inputs()
        ours, _, _, _, _ = loss_fn(pi_old, pi, ref, adv, padding_masks=mask)
        trl = _trl_grpo_loss(self._trl_per_token_loss(adv, mask), mask.float())
        torch.testing.assert_close(ours, trl, rtol=1e-5, atol=1e-6)

    def test_bnpo_matches(self):
        loss_fn = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0, loss_type="bnpo")
        pi, pi_old, ref, adv, mask = self._make_inputs()
        ours, _, _, _, _ = loss_fn(pi_old, pi, ref, adv, padding_masks=mask)
        trl = _trl_bnpo_loss(self._trl_per_token_loss(adv, mask), mask.float())
        torch.testing.assert_close(ours, trl, rtol=1e-5, atol=1e-6)

    def test_dapo_matches_with_normalizer(self):
        loss_fn = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0, loss_type="dapo")
        pi, pi_old, ref, adv, mask = self._make_inputs()
        n_items = float(mask.sum().item())
        ours, _, _, _, _ = loss_fn(
            pi_old, pi, ref, adv,
            padding_masks=mask, num_items_in_batch=n_items,
        )
        trl = _trl_dapo_loss(
            self._trl_per_token_loss(adv, mask), mask.float(), n_items,
        )
        torch.testing.assert_close(ours, trl, rtol=1e-5, atol=1e-6)

    def test_dapo_local_fallback_matches_bnpo(self):
        """Without `num_items_in_batch`, dapo falls back to local mask.sum()
        — which is exactly bnpo. Useful as a sanity check that the
        fallback math agrees with the dedicated bnpo path."""
        dapo = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0, loss_type="dapo")
        bnpo = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0, loss_type="bnpo")
        pi, pi_old, ref, adv, mask = self._make_inputs()
        l_dapo, _, _, _, _ = dapo(pi_old, pi, ref, adv, padding_masks=mask)
        l_bnpo, _, _, _, _ = bnpo(pi_old, pi, ref, adv, padding_masks=mask)
        torch.testing.assert_close(l_dapo, l_bnpo, rtol=1e-5, atol=1e-6)

    def test_dr_grpo_matches(self):
        loss_fn = GRPOSimpleLoss(epsilon=0.1, kl_coeff=0.0, loss_type="dr_grpo")
        pi, pi_old, ref, adv, mask = self._make_inputs()
        max_len = mask.size(1)
        ours, _, _, _, _ = loss_fn(
            pi_old, pi, ref, adv,
            padding_masks=mask, num_items_in_batch=mask.size(0) * max_len,
        )
        trl = _trl_dr_grpo_loss(
            self._trl_per_token_loss(adv, mask), mask.float(), max_len,
        )
        torch.testing.assert_close(ours, trl, rtol=1e-5, atol=1e-6)

    def test_length_bias_grpo_vs_dapo(self):
        """The whole point of DAPO: long sequences with the same per-token
        signal must contribute proportionally more in DAPO than in GRPO.
        Build a batch where all per-token-losses are 1.0 and vary only
        the active-token mask; DAPO weights by token count, GRPO doesn't.
        """
        # 2 sequences: lengths 8 and 2. Both have per_token_loss=1.0.
        bsz, T = 2, 8
        per_token_loss = torch.ones(bsz, T)
        mask = torch.tensor([
            [1] * 8,
            [1] * 2 + [0] * 6,
        ], dtype=torch.float32)
        grpo_l = _trl_grpo_loss(per_token_loss, mask)
        dapo_l = _trl_dapo_loss(per_token_loss, mask, float(mask.sum()))
        # GRPO averages per-seq means then mean: (1.0 + 1.0)/2 = 1.0
        # DAPO sums and divides by total tokens: (8 + 2) / 10 = 1.0
        # When the per-token signal is uniform, both agree.
        torch.testing.assert_close(grpo_l, dapo_l, rtol=1e-5, atol=1e-6)

        # But when the long sequence has a STRONGER per-token signal,
        # DAPO weights it correspondingly more.
        per_token_loss = torch.tensor([
            [2.0] * 8,            # long sequence with adv=2
            [-1.0] * 2 + [0] * 6, # short sequence with adv=-1
        ])
        # GRPO: (2.0 + (-1.0))/2 = 0.5
        # DAPO: (2*8 + (-1)*2) / 10 = 14/10 = 1.4
        grpo_l = _trl_grpo_loss(per_token_loss, mask)
        dapo_l = _trl_dapo_loss(per_token_loss, mask, float(mask.sum()))
        self.assertAlmostEqual(grpo_l.item(), 0.5, places=5)
        self.assertAlmostEqual(dapo_l.item(), 1.4, places=5)


class TestLossTypeValidation(unittest.TestCase):
    def test_invalid_loss_type_raises(self):
        with self.assertRaises(ValueError):
            GRPOSimpleLoss(loss_type="nope")

    def test_grpo_default(self):
        loss = GRPOSimpleLoss()
        self.assertEqual(loss.loss_type, "grpo")


class TestRecipeWiresDapoNormalizer(unittest.TestCase):
    def test_recipe_computes_dapo_normalizer(self):
        src = BASE_RECIPE.read_text()
        # The recipe must compute a normalizer (active token count or B*L)
        # and pass it as num_items_in_batch to BOTH loss invocation sites.
        self.assertIn("_dapo_active", src,
                      "recipe must gate DAPO normalizer computation on a "
                      "_dapo_active flag derived from cfg.loss.loss_type")
        self.assertIn("num_items_in_batch", src,
                      "recipe must pass num_items_in_batch into the loss")
        # Count loss call sites that receive the DAPO normalizer.
        self.assertGreaterEqual(
            src.count("num_items_in_batch\"] = _dapo_normalizer"),
            2,
            "both grpo_step loss invocations (chunked + single-bwd) must "
            "thread the DAPO normalizer",
        )

    def test_grad_scale_bypass_for_dapo(self):
        """For DAPO, each chunk returns a normalized SUM (denom = global
        token count). The chunked-loss grad_scale must NOT multiply by
        num_fwd_chunks for the DAPO path — that would extra-divide the
        already-normalized chunk sum.
        """
        src = BASE_RECIPE.read_text()
        # Look for the conditional grad_scale.
        self.assertRegex(
            src,
            r"if _dapo_active:\s*\n\s*grad_scale\s*=\s*max\(1,\s*self\._gradient_accumulation_steps\)",
        )


if __name__ == "__main__":
    unittest.main()

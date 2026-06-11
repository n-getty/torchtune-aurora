# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test for the base GRPO advantage normalization path.

Per-prompt-group normalization silently zeros the policy gradient whenever
all G rollouts of a prompt earn the same reward (common early in training:
all wrong, or all earning only the format bonus). The BioReason recipe fixed
this with batch-level normalization; the base recipe got the same lift in
the iterative-sniffing-blanket plan. This test locks in:

 1) `batch_level_advantages` produces a non-zero advantage even for the
    rollouts of a collapsed prompt group, as long as some other prompt in
    the batch has variance.
 2) The base recipe defaults `batch_level_advantages` to True. A future
    accidental flip to per-prompt would re-introduce the silent zero-grad
    failure mode and this test must fail loudly.
 3) The bioreason re-export still points at the canonical implementation.
"""
import re
import unittest
from pathlib import Path

import torch

from torchtune.dev.rl.rewards import batch_level_advantages
from torchtune.dev.bioreason import reward as bioreason_reward


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_RECIPE = REPO_ROOT / "recipes/dev/grpo_full_finetune_distributed_xpu.py"


class TestBatchLevelAdvantages(unittest.TestCase):
    def test_batch_level_advantages_nonzero_when_group_collapses(self):
        # Group 0: all rollouts get the same reward → per-prompt std = 0.
        # Group 1: has variance.
        rewards = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # collapsed
                [0.0, 0.0, 1.0, 0.0],  # has variance
            ]
        )
        batch_size, grpo_size = rewards.shape

        # Per-prompt path (legacy): group 0 → exactly zero advantages.
        per_prompt = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True, unbiased=False) + 1e-4
        )
        self.assertTrue(
            torch.all(per_prompt[0].abs() < 1e-6),
            f"per-prompt path should zero group 0, got {per_prompt[0]}",
        )

        # Batch-level path: group 0 rollouts get a nonzero (negative) advantage
        # because they are *below* the batch mean; group 1's winning rollout
        # gets a nonzero positive advantage.
        flat = batch_level_advantages(
            rewards.reshape(batch_size * grpo_size), group_size=grpo_size
        )
        adv = flat.reshape(batch_size, grpo_size)
        self.assertGreater(
            adv[0].abs().min().item(),
            0.1,
            f"batch-level should give nonzero adv even for collapsed group, got {adv[0]}",
        )
        self.assertGreater(
            adv[1].abs().max().item(),
            0.1,
            f"batch-level should give nonzero adv for varied group, got {adv[1]}",
        )

    def test_batch_level_advantages_handles_degenerate_single_element(self):
        # Single-element batch: std (unbiased=False) is 0 → advantage is 0, NOT NaN.
        rewards = torch.tensor([0.5])
        adv = batch_level_advantages(rewards, group_size=1)
        self.assertTrue(torch.isfinite(adv).all().item(), f"got non-finite: {adv}")
        torch.testing.assert_close(adv, torch.zeros_like(adv))

    def test_batch_level_advantages_zero_mean(self):
        torch.manual_seed(0)
        rewards = torch.randn(16)
        adv = batch_level_advantages(rewards, group_size=4)
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=4)
        # std ~= 1 within eps tolerance for a non-degenerate batch.
        self.assertAlmostEqual(adv.std(unbiased=False).item(), 1.0, places=3)


class TestBaseRecipeDefaultsToBatchLevel(unittest.TestCase):
    def test_default_is_true(self):
        """The base recipe must read cfg.get('batch_level_advantages', True).

        Anything else (no read, or default=False) re-opens the silent zero-grad
        failure mode for users running the base recipe against simple tasks
        where most prompt groups collapse early in training.
        """
        src = BASE_RECIPE.read_text()
        # Look for `cfg.get("batch_level_advantages", <default>)`. Accept either
        # quote style; require the default literal to be True.
        pat = re.compile(
            r"cfg\.get\(\s*['\"]batch_level_advantages['\"]\s*,\s*(True|False)\s*\)"
        )
        matches = pat.findall(src)
        self.assertEqual(
            len(matches),
            1,
            "expected exactly one cfg.get('batch_level_advantages', ...) read in "
            f"the base recipe, found {len(matches)}",
        )
        self.assertEqual(
            matches[0],
            "True",
            "base recipe MUST default batch_level_advantages to True; "
            "per-prompt normalization silently zeros the policy gradient when "
            "any prompt's G rollouts get equal rewards.",
        )

    def test_recipe_uses_batch_level_branch(self):
        """The advantage block must dispatch on self._batch_level_advantages
        and import the canonical helper from torchtune.dev.rl.rewards."""
        src = BASE_RECIPE.read_text()
        self.assertIn("self._batch_level_advantages", src)
        self.assertIn(
            "from torchtune.dev.rl.rewards import batch_level_advantages",
            src,
            "advantage block must import from the canonical location "
            "(torchtune.dev.rl.rewards), not the bioreason re-export.",
        )


class TestBioreasonReExport(unittest.TestCase):
    def test_bioreason_reexport_is_canonical(self):
        # The bioreason module must expose the SAME function object as the
        # canonical location, so existing BioReason imports keep working.
        self.assertIs(
            bioreason_reward.batch_level_advantages,
            batch_level_advantages,
        )


if __name__ == "__main__":
    unittest.main()

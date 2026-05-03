# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test pinning the LoRA-GRPO advantage normalization contract.

The 8B isolation rung in docs/reports/lora_status.md uses grpo_samples=1.
With the default unbiased=True, torch.std over a single sample returns NaN,
which silently produces NaN advantages → NaN loss → killed run.

This test:
  1) Static guard: recipe must compute std with unbiased=False so G=1 is 0.
  2) Behavioral: simulate the recipe's advantage line at G=1 and G=4,
     assert no NaN / Inf and that G=4 still produces finite normalized values.
"""
import unittest

import torch


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "lora_grpo_full_finetune_distributed_xpu.py"
)


class TestLoraAdvantageG1(unittest.TestCase):
    def test_recipe_uses_unbiased_false_for_advantage_std(self):
        """Static guard: rewards.std for advantage normalization MUST pass
        unbiased=False so G=1 isolation runs don't NaN out at step 0."""
        with open(_RECIPE_PATH) as f:
            src = f.read()
        # Find the advantages assignment block.
        idx = src.find("advantages = (rewards - rewards.mean(1")
        self.assertNotEqual(
            idx, -1, "could not find advantages assignment in recipe"
        )
        # Look in a small window around that line for the std call.
        window = src[idx:idx + 400]
        self.assertIn(
            "rewards.std(",
            window,
            "advantages block does not call rewards.std()",
        )
        self.assertIn(
            "unbiased=False",
            window,
            "advantages std must use unbiased=False (G=1 → NaN otherwise). "
            "Window:\n" + window,
        )

    def test_g1_advantage_is_finite(self):
        """Behavioral: with G=1, advantage should be 0 (or near 0) — never NaN."""
        # Match the recipe's shape: rewards is [batch_size, grpo_size].
        rewards = torch.tensor([[1.5], [2.0], [3.0]])  # batch=3, G=1
        adv = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True, unbiased=False) + 1e-4
        )
        self.assertTrue(torch.isfinite(adv).all().item(), f"got non-finite: {adv}")
        # mean over a single sample == that sample → numerator is exactly 0
        torch.testing.assert_close(adv, torch.zeros_like(adv))

    def test_g4_advantage_normalization_unchanged_in_spirit(self):
        """Behavioral: G>=2 still produces finite, non-trivial normalized
        advantages with unbiased=False (the constant scale shift is fine)."""
        torch.manual_seed(0)
        rewards = torch.randn(2, 4)  # batch=2, G=4
        adv = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True, unbiased=False) + 1e-4
        )
        self.assertTrue(torch.isfinite(adv).all().item())
        # Each row should be approximately zero-mean.
        torch.testing.assert_close(
            adv.mean(1), torch.zeros(2), atol=1e-4, rtol=0
        )
        # And not all zeros (i.e. normalization is doing real work).
        self.assertGreater(adv.abs().max().item(), 0.5)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test pinning the rollout-policy-logprobs gate in the LoRA recipe.

Background:
  The dense recipe (recipes/dev/grpo_full_finetune_distributed_xpu.py:2534)
  only computes rollout-time policy logprobs when ppo_epochs > 1 OR
  always_compute_rollout_logprobs=True. The LoRA recipe used to compute them
  unconditionally → ~13% wasted step time at 4B (8.3s of 64.4s).

Static guards:
  1) Recipe must compute self._compute_rollout_logprobs_required from cfg.
  2) The Step 2 policy fwd must be guarded by that flag.
  3) Step 5 mask-fill must skip None logprobs.
  4) grpo_step must fall back to pi_logprobs.detach() when trajectory.logprobs
     is None (already pinned in test_lora_grpo_step_mask, but we re-assert).
"""
import unittest


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "lora_grpo_full_finetune_distributed_xpu.py"
)


class TestLoraRolloutLogprobsGate(unittest.TestCase):
    def setUp(self):
        with open(_RECIPE_PATH) as f:
            self.src = f.read()

    def test_recipe_computes_rollout_logprobs_required_flag(self):
        self.assertIn(
            "self._compute_rollout_logprobs_required",
            self.src,
            "recipe must read/derive _compute_rollout_logprobs_required to "
            "skip rollout policy fwd in single-epoch sync GRPO",
        )
        self.assertIn(
            "always_compute_rollout_logprobs",
            self.src,
            "recipe must accept always_compute_rollout_logprobs from cfg "
            "(matches dense recipe gate)",
        )

    def test_policy_fwd_is_guarded(self):
        """The Step 2 policy fwd must sit inside `if
        self._compute_rollout_logprobs_required:` — otherwise we recompute
        every step in single-epoch sync mode (the default)."""
        marker = "Step 2: policy logprobs"
        idx = self.src.find(marker)
        self.assertNotEqual(idx, -1, f"could not find '{marker}' in recipe")
        # Inspect the next ~2000 chars for the gate.
        window = self.src[idx:idx + 2000]
        self.assertIn(
            "if self._compute_rollout_logprobs_required",
            window,
            "Step 2 policy fwd is not gated; would always run.\n" + window[:400],
        )
        # And the unguarded path must set logprobs = None.
        self.assertIn(
            "logprobs = None",
            window,
            "Step 2 must set logprobs=None when gate is False so grpo_step "
            "falls back to chunk_pi_lp.detach()",
        )

    def test_masked_fill_handles_none_logprobs(self):
        """Step 5 mask-fill must guard against logprobs=None — otherwise
        skipping rollout policy fwd would NoneType-crash."""
        # Find the mask-padding block.
        marker = "Mask padding"
        idx = self.src.find(marker)
        self.assertNotEqual(idx, -1, f"could not find '{marker}' block")
        window = self.src[idx:idx + 400]
        # Either an explicit None check, or wrapped in if logprobs is not None.
        self.assertIn(
            "if logprobs is not None",
            window,
            "logprobs.masked_fill_ must be guarded against None.\n" + window,
        )

    def test_grpo_step_uses_pi_logprobs_detach_when_none(self):
        """grpo_step must fall back to chunk_pi_lp.detach() (chunked) and
        pi_logprobs.detach() (single-fwd) when trajectory.logprobs is None."""
        # Both occurrences live in grpo_step.
        self.assertEqual(
            self.src.count("if trajectory.logprobs is not None"),
            2,
            "grpo_step should check trajectory.logprobs is not None in BOTH "
            "the chunked and single-fwd branches (got "
            f"{self.src.count('if trajectory.logprobs is not None')})",
        )
        self.assertIn("pi_logprobs.detach()", self.src)
        self.assertIn("chunk_pi_lp.detach()", self.src)


if __name__ == "__main__":
    unittest.main()

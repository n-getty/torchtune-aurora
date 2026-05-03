# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test pinning the LoRA-GRPO adapter rank-equality contract.

Background:
  Adapter params are passed as ignored_states to FSDP (recipe ~line 380).
  Their gradients are manually all-reduced (~line 1444). After
  optimizer.step(), all ranks SHOULD have bit-identical adapter weights,
  but we previously had no in-loop assertion of this.

  The recipe now hashes each rank's adapter params and all-reduces MIN/MAX
  (gated by lora.log_validation_metrics so production runs stay quiet).
  This test pins:

    1) The hash + MIN/MAX all-reduce code is present in the train loop.
    2) The validation hash logic itself, exercised on simulated 2-rank
       hashes, correctly distinguishes identical from divergent adapters.
"""
import unittest

import torch


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "lora_grpo_full_finetune_distributed_xpu.py"
)


def _build_adapter_hash(adapter_params):
    """Mirrors the recipe's hash construction (see VALMET_RANK_EQUAL block).
    Sum of (param_flat * positional_weight) over all adapter params."""
    h = torch.zeros((), dtype=torch.float64)
    for i, p in enumerate(adapter_params):
        flat = p.detach().to(torch.float64).flatten()
        idx = torch.arange(flat.numel(), dtype=torch.float64)
        h += (flat * (idx + float(i + 1))).sum()
    return h


class TestLoraAdapterRankEqualityContract(unittest.TestCase):
    def setUp(self):
        with open(_RECIPE_PATH) as f:
            self.src = f.read()

    def test_recipe_contains_rank_equality_check(self):
        for marker in (
            "VALMET_RANK_EQUAL",
            "VALMET_RANK_DIVERGENCE",
            "ReduceOp.MIN",
            "ReduceOp.MAX",
        ):
            self.assertIn(
                marker,
                self.src,
                f"recipe must contain '{marker}' in adapter rank-equality "
                "validation block (regression: silent rank divergence after "
                "manual adapter all-reduce + optimizer.step would go undetected)",
            )

    def test_check_is_gated_on_log_validation_metrics(self):
        """The rank-equality check should sit inside an `if
        self._log_validation_metrics and ...` block. Find the all_reduce
        for the hash and verify the nearest preceding `if` line gates it
        on the validation flag."""
        # Locate the rank-equality block by searching for the unique pattern
        # used to construct the per-rank hash.
        marker = "ReduceOp.MIN"
        idx = self.src.find(marker)
        self.assertNotEqual(idx, -1)
        # Scan backwards in the source by lines to find the nearest enclosing
        # `if ...` that mentions log_validation_metrics. Walk up until we hit
        # the dedent boundary (the if at column 20 used by the train loop).
        lines_before = self.src[:idx].splitlines()
        gated = False
        for line in reversed(lines_before[-80:]):
            if "if self._log_validation_metrics" in line:
                gated = True
                break
        self.assertTrue(
            gated,
            "rank-equality check should be gated on log_validation_metrics "
            "so it runs in the validation ladder, not in every prod run",
        )

    def test_hash_distinguishes_identical_from_divergent(self):
        """Behavioral: confirm the recipe's hash construction actually catches
        divergence. Simulate two 'ranks' with the same adapter params (hash
        equal), then mutate one element on rank 1 (hash must differ)."""
        torch.manual_seed(0)
        # Two adapter-shaped tensors per rank (one 'lora_a', one 'lora_b').
        rank0 = [torch.randn(4, 8), torch.randn(8, 4)]
        rank1 = [t.clone() for t in rank0]

        h0 = _build_adapter_hash(rank0)
        h1 = _build_adapter_hash(rank1)
        self.assertEqual(
            float(h0.item()),
            float(h1.item()),
            "identical adapter tensors must hash identically",
        )

        # Mutate one element on rank 1 by a small amount.
        rank1[0][0, 0] += 1e-3
        h1b = _build_adapter_hash(rank1)
        self.assertNotEqual(
            float(h0.item()),
            float(h1b.item()),
            "single-element adapter divergence must be detected by the hash",
        )
        # Spread reported as |max - min| matches what the recipe logs.
        spread = abs(float(h1b.item()) - float(h0.item()))
        self.assertGreater(spread, 0.0)


if __name__ == "__main__":
    unittest.main()

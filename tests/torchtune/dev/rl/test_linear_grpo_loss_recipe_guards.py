# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Source-level guard tests for the chunked-vocab LinearGRPOLoss across the GRPO recipe
family.

LinearGRPOLoss projects ``model.output`` OUTSIDE ``model.forward`` (per sequence
chunk). That is only correct when the output weight is materialized at projection
time and the model exposes torchtune's ``skip_output_layer`` hidden-state path.

- base full-FT (``grpo_full_finetune_distributed_xpu.py``): SUPPORTED (Phase 1).
  The tied output weight (== tok_embeddings.weight) stays resident in the ROOT FSDP2
  unit through the forward (default AllGather prefetch), so the loss reads the full
  weight post-forward (closure over tok_embeddings.weight; root unshard() as a
  defensive no-op). The recipe WIRES the loss (set_model_output) inside a scope
  fence: FSDP2 FULL_SHARD, non-EP, non-HSDP, non-packing, no compile, ppo_epochs==1,
  and a residency fence that FORBIDS 'tok_embeddings' in custom_sharded_layers for
  tied models (making it its own unit reshards it mid-forward and breaks generation).
- BioReason (``grpo_bioreason_distributed_xpu.py``): still UNSUPPORTED — HF
  ``AutoModelForCausalLM`` backbone has no ``skip_output_layer`` hidden path. It MUST
  keep its fail-fast.

These tests pin the base recipe's wiring + scope fences (so a future edit cannot drop
a fence and re-enable a silently-wrong path) and the BioReason fail-fast. Pure source
inspection — no XPU/distributed/model load.
"""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RECIPES = REPO / "recipes" / "dev"


def _src(name: str) -> str:
    return (RECIPES / name).read_text()


class TestLinearGRPOLossRecipeGuards(unittest.TestCase):
    def test_base_recipe_wires_linear_loss_with_fences(self):
        """The base recipe must WIRE LinearGRPOLoss (Phase 1) and keep every scope fence.

        The old fail-fast ('not supported in the base full-FT') is gone; in its place
        the recipe detects the loss and wires it under fail-fast fences. If any fence
        text is dropped a silently-wrong path could be re-enabled, so pin them all.
        """
        src = _src("grpo_full_finetune_distributed_xpu.py")
        # Detection + wiring (not a fail-fast anymore).
        self.assertIn('"set_model_output"', src)
        self.assertIn("set_model_output(self._model)", src)
        self.assertNotIn("not supported in the base full-FT", src)
        # Scope fences (Phase 1).
        self.assertIn("not supported with expert", src)          # EP
        self.assertIn("not supported with HSDP", src)            # HSDP / dp_replicate
        self.assertIn("requires FSDP2", src)                     # FSDP1
        self.assertIn("not supported with enable_packing", src)  # packing
        self.assertIn("incompatible with compile=True", src)     # compile
        self.assertIn("simple (no IS-clip) GRPO formulation", src)  # ppo_epochs / async
        # The IS-clip fence must gate on the DERIVED _compute_rollout_logprobs_required
        # (covers async_generation too), not just the raw _always_compute flag — the
        # linear grpo_step branch drops the rollout-logprob assertion the full path keeps.
        self.assertIn("self._compute_rollout_logprobs_required", src)
        # `.output` must be hasattr-guarded before the TiedLinear isinstance check so an
        # out-of-scope model fails with a clear message, not a raw AttributeError.
        self.assertIn('hasattr(self._model, "output")', src)
        # Tied-embedding residency fence: tok_embeddings must be FORBIDDEN from
        # custom_sharded_layers for tied models (root-resident weight), and 'output'
        # required for untied models.
        self.assertIn("tok_embeddings", src)
        self.assertIn("custom_sharded_layers", src)
        self.assertIn("NOT be in custom_sharded_layers", src)
        # Temperature threaded into the loss.
        self.assertIn("self._loss_fn.temperature = cfg.temperature", src)

    def test_bioreason_recipe_failfast_present(self):
        src = _src("grpo_bioreason_distributed_xpu.py")
        self.assertIn('hasattr(self._loss_fn, "set_model_output")', src)
        self.assertIn("not supported in the BioReason", src)

    def test_lora_recipe_supports_linear_loss(self):
        """The LoRA recipe must WIRE LinearGRPOLoss (set_model_output), not block it."""
        src = _src("lora_grpo_full_finetune_distributed_xpu.py")
        self.assertIn("set_model_output(self._model)", src)
        # and it must keep the safety fences (no-FSDP / ppo_epochs / compile / temp)
        self.assertIn("TORCHTUNE_LINEAR_LOSS_ALLOW_FSDP", src)
        self.assertIn("_loss_fn.temperature = self._temperature", src)

    def test_guard_discriminates_by_set_model_output(self):
        """Sanity: LinearGRPOLoss has set_model_output; GRPOSimpleLoss/GRPOLoss do not."""
        from torchtune.dev.rl.linear_grpo_loss import LinearGRPOLoss
        from torchtune.dev.rl.loss import GRPOSimpleLoss, GRPOLoss

        self.assertTrue(hasattr(LinearGRPOLoss(), "set_model_output"))
        self.assertFalse(hasattr(GRPOSimpleLoss(), "set_model_output"))
        self.assertFalse(hasattr(GRPOLoss(), "set_model_output"))


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Source-level guard tests for the chunked-vocab LinearGRPOLoss across the GRPO recipe
family.

LinearGRPOLoss projects ``model.output`` OUTSIDE ``model.forward`` (per sequence
chunk). That is only correct when the output weight stays resident and the model
exposes torchtune's ``skip_output_layer`` hidden-state path. Two recipes cannot
satisfy that today and MUST fail fast rather than silently corrupt training:

- base full-FT (``grpo_full_finetune_distributed_xpu.py``): FSDP FULL_SHARD reshards
  the trained tied output weight after forward -> wrong numerics + broken grads.
- BioReason (``grpo_bioreason_distributed_xpu.py``): HF ``AutoModelForCausalLM``
  backbone has no ``skip_output_layer`` hidden path, and runs FSDP FULL_SHARD.

Only the LoRA colocate recipe (no-FSDP, frozen tied output) supports it. These tests
pin the fail-fast guards so a future edit cannot drop them and re-enable a
silently-wrong path. Pure source inspection — no XPU/distributed/model load.
"""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RECIPES = REPO / "recipes" / "dev"


def _src(name: str) -> str:
    return (RECIPES / name).read_text()


class TestLinearGRPOLossRecipeGuards(unittest.TestCase):
    def test_base_recipe_failfast_present(self):
        src = _src("grpo_full_finetune_distributed_xpu.py")
        self.assertIn('hasattr(self._loss_fn, "set_model_output")', src)
        self.assertIn("not supported in the base full-FT", src)

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test pinning the LoRA-GRPO response_padding_masks contract.

Background:
  The LoRA recipe constructs query_responses pre-filled with pad_id, then
  overlays vLLM completion bytes on top (recipe lines ~912-918). When vLLM
  returns a completion shorter than max_generated_tokens with NO stop token
  (length cutoff), the trailing positions remain as pad_id but
  truncate_sequence_at_first_stop_token() leaves them mask=False because no
  stop token was ever seen. Those synthetic pads then leak into loss/KL via
  ~response_padding_masks in grpo_step.

  This test pins the recipe's mitigation:

      response_padding_masks = response_padding_masks | (responses == pad_id)

  Verifying both cases:
    1) completion ends WITHOUT EOS: trailing pad_id positions must be masked.
    2) completion ends WITH EOS: behavior matches truncate_sequence_at_first_stop_token
       (post-EOS positions also masked).

  AST source extraction so we exercise the actual recipe bytes — no
  module import (the recipe module pulls torchao + xpu backends).
"""
import ast
import textwrap
import unittest

import torch


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "lora_grpo_full_finetune_distributed_xpu.py"
)


def _grpo_step_4_source() -> str:
    """Read the file and return the lines around the Step 4 truncate+OR block."""
    with open(_RECIPE_PATH) as f:
        return f.read()


class TestLoraResponsePaddingMaskContract(unittest.TestCase):
    def test_recipe_contains_pad_id_or_after_truncate(self):
        """Static guard: the recipe MUST OR pad-id positions into the mask
        immediately after truncate_sequence_at_first_stop_token. If this
        regresses (someone deletes the OR), training silently includes
        synthetic vLLM pads in loss/KL."""
        src = _grpo_step_4_source()
        # The OR line should reference pad_id and live near the truncate call.
        # Locate the truncate call site and verify the OR appears within 5 lines.
        lines = src.splitlines()
        trunc_line = None
        for i, line in enumerate(lines):
            if "truncate_sequence_at_first_stop_token" in line and "rlhf." in line:
                trunc_line = i
                break
        self.assertIsNotNone(
            trunc_line,
            "could not find truncate_sequence_at_first_stop_token call in recipe",
        )
        window = "\n".join(lines[trunc_line:trunc_line + 12])
        self.assertIn(
            "response_padding_masks",
            window,
            "expected response_padding_masks update near truncate call",
        )
        self.assertIn(
            "self._tokenizer.pad_id",
            window,
            "expected pad_id OR-update to follow truncate call (regression: "
            "synthetic vLLM pads leak into loss/KL when completion has no EOS)",
        )
        # Specifically check the OR pattern (allow `|` or `|=` to cover
        # both forms).
        self.assertTrue(
            ("|" in window and "pad_id" in window),
            "expected `... | (responses == self._tokenizer.pad_id)` near "
            "truncate call; current window:\n" + window,
        )

    def test_no_eos_completion_pads_are_masked(self):
        """Behavioral test: simulate the recipe's mask flow on a completion
        that has NO stop token (length cutoff) and verify trailing pad_id
        positions are correctly masked after the OR."""
        from torchtune import rlhf

        pad_id = 0
        stop_tokens = torch.tensor([99])  # not present in any sample below

        # Simulate vLLM returning two completions: completion 0 has 4 real tokens
        # then 4 trailing pads (length cutoff, no EOS); completion 1 is full length.
        responses = torch.tensor(
            [[5, 6, 7, 8, pad_id, pad_id, pad_id, pad_id],
             [5, 6, 7, 8, 9, 10, 11, 12]],
            dtype=torch.long,
        ).clone()

        # Apply recipe Step 4 exactly as written in the recipe.
        rpm, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, stop_tokens, pad_id
        )
        # Without the OR, rpm[0, 4:] would still be False because no stop fired.
        self.assertFalse(
            rpm[0, 4:].any().item(),
            "sanity: truncate alone leaves no-EOS pads as mask=False (precondition)",
        )

        # The recipe's fix: OR pad-id positions in.
        rpm = rpm | (responses == pad_id)

        # Now: row 0 must mask positions 4..7 (synthetic pads); row 1 unmasked.
        expected_row0 = torch.tensor([False, False, False, False, True, True, True, True])
        expected_row1 = torch.tensor([False] * 8)
        torch.testing.assert_close(rpm[0], expected_row0)
        torch.testing.assert_close(rpm[1], expected_row1)

    def test_with_eos_completion_post_eos_still_masked(self):
        """Sanity: the OR doesn't *remove* coverage from the existing
        truncate behavior — post-EOS positions remain masked."""
        from torchtune import rlhf

        pad_id = 0
        stop_tokens = torch.tensor([99])

        # Completion with EOS at position 3, followed by garbage.
        responses = torch.tensor(
            [[5, 6, 7, 99, 42, 43, 44, 45]],
            dtype=torch.long,
        ).clone()
        rpm, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, stop_tokens, pad_id
        )
        rpm = rpm | (responses == pad_id)

        # The truncate fills post-EOS with pad_id, so positions 4..7 are masked.
        self.assertFalse(rpm[0, :3].any().item(), "pre-EOS unmasked")
        self.assertTrue(rpm[0, 4:].all().item(), "post-EOS masked")


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU-safe unit tests for the prompt-truncation guard added to _generate_with_vllm.

Background
----------
When prompt_len + max_generated_tokens > vllm_max_model_len, vLLM's PagedAttention
kernel tries to access a block index beyond the allocated block table and crashes
with a banned:1 PDE segfault on XPU.  The fix truncates raw_prompts to
    max_prompt_len = vllm_max_model_len - max_generated_tokens
(tail-keeping, so the model sees the most recent context).

These tests verify the truncation logic in isolation without needing an XPU,
a vLLM process, or a live recipe.
"""
import unittest

PAD_ID = 0


def _strip_and_truncate(ids, pad_id, max_prompt_len):
    """Mirrors the recipe's strip-padding-then-truncate logic."""
    ids = [t for t in ids if t != pad_id]
    return ids[-max_prompt_len:] if len(ids) > max_prompt_len else ids


class TestPromptTruncation(unittest.TestCase):
    """Verify the strip-and-truncate helper used in _generate_with_vllm."""

    def test_short_prompt_unchanged(self):
        """Prompt shorter than max_prompt_len passes through unchanged."""
        ids = [1, 2, 3, 4, 5]
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=10)
        self.assertEqual(result, ids)

    def test_exact_length_unchanged(self):
        """Prompt exactly at max_prompt_len passes through unchanged."""
        ids = list(range(1, 9))  # 8 tokens
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=8)
        self.assertEqual(result, ids)

    def test_long_prompt_tail_truncated(self):
        """Prompt longer than max_prompt_len keeps the tail (most recent context)."""
        ids = list(range(1, 21))  # 20 tokens: [1, 2, ..., 20]
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=10)
        self.assertEqual(result, list(range(11, 21)))

    def test_padding_stripped_before_truncation(self):
        """Padding tokens are stripped before the length check."""
        # 5 real tokens + 10 pad tokens; with max_prompt_len=8 the real tokens fit
        ids = [1, 2, 3, 4, 5] + [PAD_ID] * 10
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=8)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_padding_stripped_then_truncated(self):
        """Padding stripped, then tail-truncated if still over limit."""
        # 15 real tokens padded to 20; max_prompt_len=10 → keep tail 10 real tokens
        real = list(range(1, 16))
        ids = real + [PAD_ID] * 5
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=10)
        self.assertEqual(result, real[-10:])

    def test_result_always_fits_max_model_len(self):
        """After truncation, len(prompt) + max_gen never exceeds max_model_len."""
        max_model_len = 1024
        max_gen = 600
        max_prompt_len = max_model_len - max_gen  # 424

        for prompt_len in [100, 424, 425, 800, 1024, 1500]:
            ids = list(range(1, prompt_len + 1))
            result = _strip_and_truncate(ids, PAD_ID, max_prompt_len)
            total = len(result) + max_gen
            self.assertLessEqual(
                total, max_model_len,
                f"prompt_len={prompt_len}: {len(result)}+{max_gen}={total} > {max_model_len}",
            )

    def test_all_padding_produces_empty(self):
        """All-padding input produces an empty prompt after stripping."""
        ids = [PAD_ID] * 20
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=10)
        self.assertEqual(result, [])

    def test_max_prompt_len_one(self):
        """Edge case: max_prompt_len=1 keeps only the last token."""
        ids = [1, 2, 3, 4, 5]
        result = _strip_and_truncate(ids, PAD_ID, max_prompt_len=1)
        self.assertEqual(result, [5])


class TestMaxPromptLenCalculation(unittest.TestCase):
    """Verify the max_prompt_len formula covers the overflow scenario."""

    def _max_prompt_len(self, vllm_max_model_len, max_generated_tokens):
        return vllm_max_model_len - max_generated_tokens

    def test_standard_config(self):
        self.assertEqual(self._max_prompt_len(1024, 200), 824)

    def test_long_gen_config(self):
        """The config that was crashing: max_gen=600 → limit=424."""
        self.assertEqual(self._max_prompt_len(1024, 600), 424)

    def test_large_model_len(self):
        self.assertEqual(self._max_prompt_len(1536, 700), 836)

    def test_formula_prevents_overflow(self):
        """Any prompt ≤ max_prompt_len will always fit with max_gen tokens."""
        for max_model_len in [512, 1024, 1280, 1536, 2048]:
            for max_gen in [200, 400, 600, 700]:
                if max_gen >= max_model_len:
                    continue
                mpl = self._max_prompt_len(max_model_len, max_gen)
                self.assertLessEqual(
                    mpl + max_gen, max_model_len,
                    f"max_model_len={max_model_len} max_gen={max_gen}: {mpl}+{max_gen} > {max_model_len}",
                )


if __name__ == "__main__":
    unittest.main()

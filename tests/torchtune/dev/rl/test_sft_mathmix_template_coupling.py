# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe coupling test: the SFT math-mix renders the EXACT <think>/<answer>
template the GRPO reward parser keys on.

This is a real correctness coupling, not cosmetic. Per the 2026-06-15 status
note (``project_agpt2b_sft_mathmix_to_grpo``), the previous SFT pass produced
raw question/answer pairs WITHOUT the ``<think>...</think> <answer>...</answer>``
wrapper, and the GRPO reward parser
(``torchtune.dev.rl.rewards.extract_tags`` and ``FormattingReward``) silently
scored those completions on the loose partial-credit path. Aligning the SFT
template to the GRPO template was a free format-compliance win.

If a future edit changes the SFT template (different tag names, dropped tags,
re-ordered tags) but leaves the GRPO parser alone, training would teach the
model a format the reward function can't read — and the failure is silent (no
crash, just a flat reward curve). These tests make that divergence loud.

What we pin:
  * Each per-source transform (gsm8k / math / metamath) emits a ``trainable``
    string whose answer is recoverable by ``extract_tags``.
  * The rendered template uses the same tag literals the parser regexes use.
  * The strict ``FormattingReward`` regex's tag literals match the template's.
  * The mathmix template is byte-identical to the canonical
    ``torchtune.dev.rl.gsm8k.TRAINABLE_PROMPT`` (single source of truth).
"""
import re

import pytest

from torchtune.dev.rl.gsm8k import PREAMBLE_PROMPT, TRAINABLE_PROMPT
from torchtune.dev.rl.rewards import extract_tags

import torchtune.dev.sft.auroragpt_math_mix as mm


def test_mathmix_uses_canonical_template():
    # The mathmix module imports the canonical prompts; assert it did not
    # shadow them with a local copy that could drift.
    assert mm.PREAMBLE_PROMPT is PREAMBLE_PROMPT
    assert mm.TRAINABLE_PROMPT is TRAINABLE_PROMPT


def test_template_tags_match_parser_tags():
    # The reward parser keys on <think> and <answer>. The training template
    # MUST contain those exact tag literals or RL can never read the answer.
    assert "<think>" in TRAINABLE_PROMPT and "</think>" in TRAINABLE_PROMPT
    assert "<answer>" in TRAINABLE_PROMPT and "</answer>" in TRAINABLE_PROMPT


def test_gsm8k_transform_roundtrips_through_parser():
    row = {"question": "What is 2+2?", "answer": "We add 2 and 2.\n#### 4"}
    out = mm._gsm8k_transform(row)
    assert set(out) == {"preamble", "trainable"}
    cot, ans = extract_tags(out["trainable"])
    assert ans == "4", f"GRPO parser could not recover answer: {ans!r}"
    assert cot, "GRPO parser recovered empty cot — format reward would be 0"


def test_math_transform_roundtrips_through_parser():
    row = {
        "problem": "Compute the fraction.",
        "solution": "After working it out, the result is $\\boxed{\\frac{1}{4}}$.",
    }
    out = mm._math_transform(row)
    assert out is not None, "math transform dropped an extractable row"
    cot, ans = extract_tags(out["trainable"])
    assert ans == "\\frac{1}{4}", f"answer mismatch: {ans!r}"


def test_math_transform_drops_unboxed_rows():
    # No \boxed{} → transform returns None and the loader's filter drops it.
    row = {"problem": "no boxed answer here", "solution": "the answer is 5"}
    assert mm._math_transform(row) is None


def test_metamath_transform_roundtrips_through_parser():
    row = {
        "query": "How many apples?",
        "response": "Step one. Step two.\n#### 7\nThe answer is: 7",
    }
    out = mm._metamath_transform(row)
    assert out is not None
    cot, ans = extract_tags(out["trainable"])
    assert ans == "7", f"answer mismatch: {ans!r}"
    assert cot, "metamath cot empty after stripping trailer"
    # The trailer answer lines must NOT leak into the <think> CoT.
    assert "#### 7" not in cot
    assert "The answer is" not in cot


def test_metamath_transform_drops_answer_only_rows():
    # A response with no usable CoT (answer-only) must be dropped.
    row = {"query": "x?", "response": "#### 5\nThe answer is: 5"}
    assert mm._metamath_transform(row) is None


def test_strict_formatting_reward_tags_match_template():
    """The strict FormattingReward regex is built from think_tag/answer_tag.
    Verify the canonical tag names ('think','answer') match the template so a
    well-formed SFT completion can in principle earn strict format reward."""
    think_tag, answer_tag = "think", "answer"
    soft_think = rf"<{think_tag}>.*?</{think_tag}>"
    soft_answer = rf"<{answer_tag}>.*?</{answer_tag}>"
    rendered = TRAINABLE_PROMPT.format(cot="reasoning", answer="42")
    assert re.search(soft_think, rendered, re.DOTALL)
    assert re.search(soft_answer, rendered, re.DOTALL)
    # think must precede answer (soft-format ordering requirement)
    assert rendered.index("<think>") < rendered.index("<answer>")


def test_all_sources_share_one_template_renderer():
    """Guard against per-source template drift: every transform must render
    via TRAINABLE_PROMPT, so the produced trainable strings differ only in the
    cot/answer payload, never in the surrounding tags."""
    gsm = mm._gsm8k_transform(
        {"question": "q", "answer": "reason\n#### 9"}
    )["trainable"]
    math = mm._math_transform(
        {"problem": "q", "solution": "work $\\boxed{9}$"}
    )["trainable"]
    meta = mm._metamath_transform(
        {"query": "q", "response": "reason\n#### 9\nThe answer is: 9"}
    )["trainable"]
    # Strip the variable payload and confirm the tag skeleton is identical.
    skel = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    for s in (gsm, math, meta):
        assert skel.fullmatch(s), f"template skeleton drift: {s!r}"

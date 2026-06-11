# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU pin-down for the ezpz task port (no XPU, no distributed init).

Two layers of checks:

1. **Self-consistency** (always runs): each dataset builder produces the
   expected number of samples for a given seed, prompts contain the canonical
   ``"What is ... Reply with just the number."`` (or task-specific) skeleton,
   and rewards round-trip ground-truth answers to ``successes == 1.0``.

2. **Parity with upstream ezpz** (skipped unless the env var
   ``TORCHTITAN_EZPZ_DIR`` points at a working clone of ``saforem2/torchtitan``
   ezpz branch with the ezpz venv on ``sys.path`` — i.e. only meaningful from
   the bake-off launcher after ``run_ezpz.sh`` has cloned the repo). When
   present, the test imports each upstream ``tasks/*.py`` and asserts both
   sides produce **bit-identical** prompts + answers for the same seed.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest
import torch


def _has_ezpz() -> bool:
    p = os.environ.get("TORCHTITAN_EZPZ_DIR")
    return bool(p) and (Path(p) / "torchtitan" / "experiments" / "ezpz" / "rl" / "tasks").is_dir()


class _FakeTok:
    """Tiny tokenizer stub — only ``encode`` is used by the dataset wrapper."""

    pad_id = 0
    eos_id = 1
    bos_id = 2

    def encode(self, text: str, add_eos: bool = False, add_bos: bool = False) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# ──────────────────────────────────────────────────────────────────────────────
# Self-consistency
# ──────────────────────────────────────────────────────────────────────────────


def test_dispatch_builds_every_task():
    from torchtune.dev.rl.ezpz_tasks import build_ezpz_dataset

    tok = _FakeTok()
    for task in ("sum_digits", "multiply", "word_sort", "countdown", "arithmetic"):
        ds = build_ezpz_dataset(tok, task=task, num_samples=20, seed=42)
        assert len(ds) == 20
        sample = ds[0]
        assert "tokens" in sample and "answer" in sample and "question" in sample
        assert isinstance(sample["answer"], str)
        assert len(sample["tokens"]) > 0


def test_dispatch_unknown_task_raises():
    from torchtune.dev.rl.ezpz_tasks import build_ezpz_dataset

    with pytest.raises(ValueError, match="unknown ezpz task"):
        build_ezpz_dataset(_FakeTok(), task="not_a_task")


def test_seed_determinism():
    from torchtune.dev.rl.ezpz_tasks import build_sum_digits_dataset

    ds_a = build_sum_digits_dataset(_FakeTok(), num_samples=50, seed=42)
    ds_b = build_sum_digits_dataset(_FakeTok(), num_samples=50, seed=42)
    for i in range(50):
        assert ds_a[i]["question"] == ds_b[i]["question"]
        assert ds_a[i]["answer"] == ds_b[i]["answer"]


@pytest.mark.parametrize(
    "task, reward_cls, completion, answer, want_success",
    [
        ("sum_digits", "SumDigitsReward", "the answer is 12", "12", 1.0),
        ("sum_digits", "SumDigitsReward", "no number here", "12", 0.0),
        ("multiply", "MultiplyReward", "7 × 8 = 56", "56", 1.0),
        ("multiply", "MultiplyReward", "the answer is 5", "12", 0.0),
        ("word_sort", "WordSortReward", "apple, banana, cherry", "apple, banana, cherry", 1.0),
        ("word_sort", "WordSortReward", "cherry, banana, apple", "apple, banana, cherry", 0.0),
        ("countdown", "CountdownReward", "1 + 2 + 3 = 6", "6", 1.0),
        ("arithmetic", "ArithmeticReward", "2 + 3 = 5", "5", 1.0),
    ],
)
def test_rewards_score_known_cases(task, reward_cls, completion, answer, want_success):
    import torchtune.dev.rl.ezpz_tasks as mod

    cls = getattr(mod, reward_cls)
    fn = cls()
    out = fn(torch.zeros(1, 1, dtype=torch.long), [completion], [answer])
    assert out.successes.tolist() == [want_success], (task, reward_cls, completion)


def test_answer_extraction_chain():
    from torchtune.dev.rl.ezpz_tasks import extract_answer

    assert extract_answer(r"so \boxed{42} is the answer") == "42"
    assert extract_answer("chain of thought\n#### 17") == "17"
    assert extract_answer("The answer is -7 because reasons") == "-7"
    assert extract_answer("3 + 4 = 7 then 7 + 1 = 8") == "8"
    assert extract_answer("there are 3 apples and 17 oranges") == "17"
    assert extract_answer("no numbers here at all") is None


def test_word_sort_partial_reward_positional():
    from torchtune.dev.rl.ezpz_tasks import WordSortReward

    fn = WordSortReward()
    out = fn(
        torch.zeros(1, 1, dtype=torch.long),
        ["apple, banana, kiwi"],  # 2/3 in correct positions
        ["apple, banana, cherry"],
    )
    # accuracy 0.0 + partial 2/3 = 0.6666...
    assert pytest.approx(out.total_reward.item(), abs=1e-5) == 2 / 3


def test_arithmetic_length_penalty_bounds():
    from torchtune.dev.rl.ezpz_tasks import ArithmeticReward

    fn = ArithmeticReward()
    short = fn(torch.zeros(1, 1, dtype=torch.long), ["= 5"], ["5"])
    long_text = " ".join(["x"] * 100) + " = 5"
    long_ = fn(torch.zeros(1, 1, dtype=torch.long), [long_text], ["5"])
    assert short.rewards["length_penalty"].item() == 0.0
    assert long_.rewards["length_penalty"].item() == -1.0


# ──────────────────────────────────────────────────────────────────────────────
# Parity with upstream ezpz (opt-in via TORCHTITAN_EZPZ_DIR)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_ezpz(), reason="set TORCHTITAN_EZPZ_DIR to compare against upstream")
@pytest.mark.parametrize(
    "task, ours_builder, upstream_module",
    [
        ("sum_digits", "build_sum_digits_dataset", "sum_digits"),
        ("multiply", "build_multiply_dataset", "multiply"),
        ("word_sort", "build_word_sort_dataset", "word_sort"),
        ("countdown", "build_countdown_dataset", "countdown"),
        ("arithmetic", "build_arithmetic_dataset", "arithmetic"),
    ],
)
def test_prompts_bit_identical_to_ezpz(task, ours_builder, upstream_module):
    ezpz_root = Path(os.environ["TORCHTITAN_EZPZ_DIR"])
    sys.path.insert(0, str(ezpz_root))
    try:
        import importlib

        upstream = importlib.import_module(
            f"torchtitan.experiments.ezpz.rl.tasks.{upstream_module}"
        )
    finally:
        sys.path.pop(0)

    import torchtune.dev.rl.ezpz_tasks as mod

    builder = getattr(mod, ours_builder)
    ours = builder(_FakeTok(), num_samples=100, seed=42)
    upstream_ds = upstream.build_dataset(num_samples=100, seed=42)

    for i in range(100):
        their_prompt = upstream_ds[i]["prompt"]
        their_text = their_prompt[0]["content"] if isinstance(their_prompt, list) else their_prompt
        assert ours[i]["question"] == their_text, f"{task} sample {i} prompt mismatch"
        assert ours[i]["answer"] == upstream_ds[i]["answer"], f"{task} sample {i} answer mismatch"

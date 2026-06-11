# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""ezpz/rl task ports for apples-to-apples GRPO comparison.

Ports prompt generation and reward functions bit-for-bit from
``saforem2/torchtitan`` (BSD-licensed) branch ``ezpz``, path
``torchtitan/experiments/ezpz/rl/tasks/{sum_digits,multiply,word_sort,
countdown,arithmetic,common}.py``. Datasets are generated in-process from
a seeded ``random.Random`` so no external download or HF Hub call is made.

Each task exposes:
  * ``build_<task>_dataset(tokenizer, ...)`` — torchtune Dataset usable with
    ``torchtune.dev.rl.data.padded_collate_rl``.
  * A ``Reward`` subclass that sums the ezpz reward functions for that task
    (so the recipe's single-call reward path matches ezpz's TRL summation).
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from torchtune.dev.rl.rewards import Reward, RewardOutput
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
    SentencePieceBaseTokenizer,
)


# ──────────────────────────────────────────────────────────────────────────────
# AuroraGPT-2B tokenizer (sentencepiece wrapper exposing the attributes the
# GRPO recipe consumes — pad_id, eos_id, encode(text, add_eos=...), decode(ids)).
# ──────────────────────────────────────────────────────────────────────────────


class AuroraGPTTokenizer(SentencePieceBaseTokenizer):
    """Sentencepiece tokenizer compatible with the GRPO recipe's expectations.

    Inherits ``encode`` / ``decode`` / ``pad_id`` / ``eos_id`` / ``bos_id``
    from :class:`SentencePieceBaseTokenizer`. ``stop_tokens`` (used by the
    recipe to assemble ``_stop_token_ids``) defaults to ``[eos_id]``.
    """

    def __init__(self, path: str, *, max_seq_len: Optional[int] = None) -> None:
        super().__init__(path)
        self.max_seq_len = max_seq_len
        # The recipe falls back to cfg.stop_token_ids when this isn't set,
        # but exposing it keeps things simple.
        self.stop_tokens = [self.eos_id] if self.eos_id is not None else []


def auroragpt_tokenizer(path: str, *, max_seq_len: Optional[int] = None) -> AuroraGPTTokenizer:
    return AuroraGPTTokenizer(path=path, max_seq_len=max_seq_len)


# ──────────────────────────────────────────────────────────────────────────────
# Shared extraction helpers (ezpz tasks/common.py)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_LARGE_POOL = 100_000


def get_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(
            (msg.get("content", "") if isinstance(msg, dict) else str(msg))
            for msg in completion
        )
    return str(completion)


def extract_answer(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:the\s+)?answer\s+is\s+(-?\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    matches = re.findall(r"=\s*(-?\d+)", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"\b(-?\d+)\b", text)
    if matches:
        return matches[-1]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset scaffolding
# ──────────────────────────────────────────────────────────────────────────────


# Gemma-style chat template — kept for the opt-in case. Empirical 2026-06-10:
# wrapping AGPT-2B base prompts with this template made the model *slower*
# and didn't improve accuracy (the checkpoint is pretraining-only — it treats
# the gemma turn markers as ordinary tokens, not as a chat schema). Default
# is raw prompts to match the original behavior.
def _gemma_prompt_wrap(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


class _EzpzPromptDataset(Dataset):
    """Wraps a list of (prompt_str, answer_str) into the torchtune RL dict shape."""

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        samples: list[dict[str, Any]],
        max_seq_len: int = 1024,
        chat_template: Optional[str] = None,
    ):
        self._tokenizer = tokenizer
        self._samples = samples
        self._max_seq_len = max_seq_len
        self._chat_template = chat_template

    def __len__(self) -> int:
        return len(self._samples)

    def _wrap(self, prompt: str) -> str:
        if self._chat_template == "gemma":
            return _gemma_prompt_wrap(prompt)
        return prompt

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        prompt = self._wrap(sample["prompt"])
        answer = sample["answer"]
        tokens = self._tokenizer.encode(prompt, add_eos=False)
        if len(tokens) > self._max_seq_len:
            tokens = tokens[: self._max_seq_len]
        return {
            "question": prompt,
            "tokens": tokens,
            "mask": [1] * len(tokens),
            "answer": answer,
        }


def _materialize(sample_fn, num_samples: int) -> list[dict[str, Any]]:
    n = num_samples if num_samples and num_samples > 0 else _DEFAULT_LARGE_POOL
    return [sample_fn() for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# sum_digits
# ──────────────────────────────────────────────────────────────────────────────


def _sample_sum_digits(rng: random.Random, min_addends: int, max_addends: int, max_digit: int) -> dict:
    n = rng.randint(min_addends, max_addends)
    digits = [rng.randint(0, max_digit) for _ in range(n)]
    expr = " + ".join(str(d) for d in digits)
    return {
        "prompt": f"What is {expr}? Reply with just the number.",
        "answer": str(sum(digits)),
    }


def build_sum_digits_dataset(
    tokenizer: ModelTokenizer,
    *,
    num_samples: int = 0,
    min_addends: int = 2,
    max_addends: int = 5,
    max_digit: int = 9,
    seed: int = 42,
    max_seq_len: int = 512,
    chat_template: Optional[str] = None,
) -> _EzpzPromptDataset:
    rng = random.Random(seed)
    samples = _materialize(
        lambda: _sample_sum_digits(rng, min_addends, max_addends, max_digit),
        num_samples,
    )
    return _EzpzPromptDataset(tokenizer, samples, max_seq_len=max_seq_len, chat_template=chat_template)


class SumDigitsReward(Reward):
    """ezpz sum_digits: 1.0 accuracy + 0.5 format bonus (max 1.5)."""

    def __init__(self) -> None:
        self._fmt = re.compile(r"\d+\s*\+\s*\d+")

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        acc, fmt = [], []
        for comp, ans in zip(completions, answers):
            text = get_completion_text(comp)
            ext = extract_answer(text)
            acc.append(1.0 if (ext is not None and ext == str(ans)) else 0.0)
            fmt.append(0.5 if self._fmt.search(text) else 0.0)
        acc_t = torch.tensor(acc)
        fmt_t = torch.tensor(fmt)
        return RewardOutput(
            reward_base_name="sum_digits",
            total_reward=acc_t + fmt_t,
            successes=acc_t,
            rewards={"accuracy": acc_t, "format": fmt_t},
        )


# ──────────────────────────────────────────────────────────────────────────────
# multiply
# ──────────────────────────────────────────────────────────────────────────────


def _sample_multiply(rng: random.Random, num_factors: int, max_factor: int) -> dict:
    factors = [rng.randint(2, max_factor) for _ in range(num_factors)]
    product = 1
    for f in factors:
        product *= f
    expr = " × ".join(str(f) for f in factors)  # ×
    return {
        "prompt": f"What is {expr}? Reply with just the number.",
        "answer": str(product),
    }


def build_multiply_dataset(
    tokenizer: ModelTokenizer,
    *,
    num_samples: int = 0,
    num_factors: int = 2,
    max_factor: int = 12,
    seed: int = 42,
    max_seq_len: int = 512,
    chat_template: Optional[str] = None,
) -> _EzpzPromptDataset:
    rng = random.Random(seed)
    samples = _materialize(
        lambda: _sample_multiply(rng, num_factors, max_factor),
        num_samples,
    )
    return _EzpzPromptDataset(tokenizer, samples, max_seq_len=max_seq_len, chat_template=chat_template)


class MultiplyReward(Reward):
    """ezpz multiply: 1.0 accuracy + 0.5 format (regex matches ``\\d+\\s*[×x\\*]\\s*\\d+``)."""

    def __init__(self) -> None:
        self._fmt = re.compile(r"\d+\s*[×x\*]\s*\d+")

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        acc, fmt = [], []
        for comp, ans in zip(completions, answers):
            text = get_completion_text(comp)
            ext = extract_answer(text)
            acc.append(1.0 if (ext is not None and ext == str(ans)) else 0.0)
            fmt.append(0.5 if self._fmt.search(text) else 0.0)
        acc_t = torch.tensor(acc)
        fmt_t = torch.tensor(fmt)
        return RewardOutput(
            reward_base_name="multiply",
            total_reward=acc_t + fmt_t,
            successes=acc_t,
            rewards={"accuracy": acc_t, "format": fmt_t},
        )


# ──────────────────────────────────────────────────────────────────────────────
# word_sort
# ──────────────────────────────────────────────────────────────────────────────

_WORD_POOL = (
    "apple banana cherry date elderberry fig grape honeydew "
    "kiwi lemon mango nectarine orange papaya quince raspberry "
    "strawberry tangerine ugli vanilla walnut xigua yellow zucchini "
    "almond brazil cashew dill ear fennel garlic herb iris jasmine "
    "kale leek mint nutmeg olive parsley rosemary sage thyme "
    "umbrella violet willow xenon yarrow"
).split()


def _sample_word_sort(rng: random.Random, min_words: int, max_words: int) -> dict:
    n = rng.randint(min_words, max_words)
    words = rng.sample(_WORD_POOL, n)
    # reshuffle if already sorted
    while words == sorted(words):
        rng.shuffle(words)
    prompt = (
        f"Sort these words alphabetically: {', '.join(words)}\n"
        "Reply with just the sorted list, separated by commas."
    )
    return {"prompt": prompt, "answer": ", ".join(sorted(words))}


def build_word_sort_dataset(
    tokenizer: ModelTokenizer,
    *,
    num_samples: int = 0,
    min_words: int = 3,
    max_words: int = 6,
    seed: int = 42,
    max_seq_len: int = 512,
    chat_template: Optional[str] = None,
) -> _EzpzPromptDataset:
    rng = random.Random(seed)
    samples = _materialize(
        lambda: _sample_word_sort(rng, min_words, max_words),
        num_samples,
    )
    return _EzpzPromptDataset(tokenizer, samples, max_seq_len=max_seq_len, chat_template=chat_template)


def _normalize_words(text: str) -> list[str]:
    return [m.lower() for m in re.findall(r"[a-zA-Z]+", text)]


class WordSortReward(Reward):
    """ezpz word_sort: 1.0 exact-match + per-position partial match (max ~2.0)."""

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        acc, partial = [], []
        for comp, ans in zip(completions, answers):
            text = get_completion_text(comp)
            extracted = _normalize_words(text)
            expected = _normalize_words(ans)
            if not extracted or not expected:
                acc.append(0.0)
                partial.append(0.0)
                continue
            acc.append(1.0 if extracted == expected else 0.0)
            matches = sum(
                a == b for a, b in zip(extracted, expected)
            )
            partial.append(matches / len(expected))
        acc_t = torch.tensor(acc)
        par_t = torch.tensor(partial)
        return RewardOutput(
            reward_base_name="word_sort",
            total_reward=acc_t + par_t,
            successes=acc_t,
            rewards={"accuracy": acc_t, "partial": par_t},
        )


# ──────────────────────────────────────────────────────────────────────────────
# countdown
# ──────────────────────────────────────────────────────────────────────────────


def _evaluate_left_to_right(nums: list[int], ops: list[str]) -> int:
    acc = nums[0]
    for n, op in zip(nums[1:], ops):
        if op == "+":
            acc = acc + n
        elif op == "-":
            acc = acc - n
        elif op == "*":
            acc = acc * n
        else:
            raise ValueError(op)
    return acc


def _find_target(rng: random.Random, numbers: list[int]) -> Optional[int]:
    import itertools

    candidates = []
    ops_pool = ["+", "-", "*"]
    for perm in itertools.permutations(numbers):
        for _ in range(20):
            ops = [rng.choice(ops_pool) for _ in range(len(perm) - 1)]
            try:
                val = _evaluate_left_to_right(list(perm), ops)
            except Exception:
                continue
            if 1 <= val <= 999:
                candidates.append(val)
    if not candidates:
        return None
    return rng.choice(candidates)


def _sample_countdown(
    rng: random.Random,
    min_nums: int,
    max_nums: int,
    max_value: int,
) -> dict:
    for _ in range(50):
        n = rng.randint(min_nums, max_nums)
        numbers = [rng.randint(1, max_value) for _ in range(n)]
        target = _find_target(rng, numbers)
        if target is not None:
            nums_str = ", ".join(str(x) for x in numbers)
            prompt = (
                f"Using the numbers {nums_str}, create an arithmetic expression "
                f"using +, -, * that equals {target}. Show your expression and result."
            )
            return {
                "prompt": prompt,
                "answer": str(target),
                "numbers": nums_str,
            }
    # graceful fallback
    nums_str = ", ".join(str(x) for x in numbers)
    return {
        "prompt": f"Using the numbers {nums_str}, create an arithmetic expression equal to {sum(numbers)}.",
        "answer": str(sum(numbers)),
        "numbers": nums_str,
    }


def build_countdown_dataset(
    tokenizer: ModelTokenizer,
    *,
    num_samples: int = 0,
    min_nums: int = 3,
    max_nums: int = 5,
    max_value: int = 12,
    seed: int = 42,
    max_seq_len: int = 512,
    chat_template: Optional[str] = None,
) -> _EzpzPromptDataset:
    rng = random.Random(seed)
    samples = _materialize(
        lambda: _sample_countdown(rng, min_nums, max_nums, max_value),
        num_samples,
    )
    return _EzpzPromptDataset(tokenizer, samples, max_seq_len=max_seq_len, chat_template=chat_template)


def _extract_countdown_result(text: str) -> Optional[int]:
    matches = re.findall(r"=\s*(-?\d+)", text)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            return None
    nums = re.findall(r"-?\d+", text)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


class CountdownReward(Reward):
    """ezpz countdown: 1.0 acc + 0.5 format + 0.5 uses-given-numbers (max 2.0)."""

    def __init__(self) -> None:
        self._fmt = re.compile(r"\d+\s*[+\-\*]\s*\d+")
        self._lhs = re.compile(r"([\d\s+\-\*]+)=")

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        acc, fmt, used = [], [], []
        for comp, ans in zip(completions, answers):
            text = get_completion_text(comp)
            extracted = _extract_countdown_result(text)
            try:
                expected = int(ans)
            except ValueError:
                expected = None
            acc.append(1.0 if (extracted is not None and extracted == expected) else 0.0)
            fmt.append(0.5 if self._fmt.search(text) else 0.0)
            # uses_given_numbers — we don't pass numbers list at reward time,
            # so this rewards the structural presence of allowed-form numbers
            # on the LHS; same scoring as ezpz's permissive check on a single
            # answer per problem (numbers field unavailable through the
            # recipe's reward interface).
            lhs = self._lhs.search(text)
            used.append(0.5 if (lhs and re.findall(r"\d+", lhs.group(1))) else 0.0)
        acc_t = torch.tensor(acc)
        fmt_t = torch.tensor(fmt)
        used_t = torch.tensor(used)
        return RewardOutput(
            reward_base_name="countdown",
            total_reward=acc_t + fmt_t + used_t,
            successes=acc_t,
            rewards={"accuracy": acc_t, "format": fmt_t, "used_numbers": used_t},
        )


# ──────────────────────────────────────────────────────────────────────────────
# arithmetic (mixed +, -, ×, ÷)
# ──────────────────────────────────────────────────────────────────────────────


def _sample_arithmetic(
    rng: random.Random,
    min_operands: int,
    max_operands: int,
    max_value: int,
) -> dict:
    op = rng.choice(["add", "sub", "mul", "div"])
    n = rng.randint(min_operands, max_operands)
    if op == "add":
        nums = [rng.randint(0, max_value) for _ in range(n)]
        ans = sum(nums)
        symbol = "+"
    elif op == "sub":
        nums = [rng.randint(max_value // 2, max_value)]
        running = nums[0]
        for _ in range(n - 1):
            x = rng.randint(0, running)
            nums.append(x)
            running -= x
        ans = running
        symbol = "-"
    elif op == "mul":
        nums = [rng.randint(2, max(2, max_value)) for _ in range(n)]
        ans = 1
        for x in nums:
            ans *= x
        symbol = "×"
    else:  # div
        # build dividend = quotient * divisor for clean integer division.
        n = 2
        quotient = rng.randint(1, max_value)
        divisor = rng.randint(2, max(2, max_value))
        nums = [quotient * divisor, divisor]
        ans = quotient
        symbol = "÷"
    expr = f" {symbol} ".join(str(x) for x in nums)
    return {
        "prompt": f"What is {expr}? Reply with just the number.",
        "answer": str(ans),
        "op": op,
    }


def build_arithmetic_dataset(
    tokenizer: ModelTokenizer,
    *,
    num_samples: int = 0,
    min_operands: int = 2,
    max_operands: int = 4,
    max_value: int = 20,
    seed: int = 42,
    max_seq_len: int = 512,
    chat_template: Optional[str] = None,
) -> _EzpzPromptDataset:
    rng = random.Random(seed)
    samples = _materialize(
        lambda: _sample_arithmetic(rng, min_operands, max_operands, max_value),
        num_samples,
    )
    return _EzpzPromptDataset(tokenizer, samples, max_seq_len=max_seq_len, chat_template=chat_template)


class ArithmeticReward(Reward):
    """ezpz arithmetic: 1.0 acc + 0.5 format + length_penalty in [-1.0, 0.0]."""

    _LENGTH_PENALTY_TARGET = 8
    _LENGTH_PENALTY_HARD = 64

    def __init__(self) -> None:
        self._fmt = re.compile(r"\d+\s*[+\-×x\*÷/]\s*\d+")

    def _length_penalty(self, text: str) -> float:
        n = len(text.split())
        if n <= self._LENGTH_PENALTY_TARGET:
            return 0.0
        if n >= self._LENGTH_PENALTY_HARD:
            return -1.0
        span = self._LENGTH_PENALTY_HARD - self._LENGTH_PENALTY_TARGET
        return -((n - self._LENGTH_PENALTY_TARGET) / span)

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: list[str],
        answers: list[str],
    ) -> RewardOutput:
        acc, fmt, lp = [], [], []
        for comp, ans in zip(completions, answers):
            text = get_completion_text(comp)
            ext = extract_answer(text)
            acc.append(1.0 if (ext is not None and ext == str(ans)) else 0.0)
            fmt.append(0.5 if self._fmt.search(text) else 0.0)
            lp.append(self._length_penalty(text))
        acc_t = torch.tensor(acc)
        fmt_t = torch.tensor(fmt)
        lp_t = torch.tensor(lp)
        return RewardOutput(
            reward_base_name="arithmetic",
            total_reward=acc_t + fmt_t + lp_t,
            successes=acc_t,
            rewards={"accuracy": acc_t, "format": fmt_t, "length_penalty": lp_t},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch helper for the YAML config
# ──────────────────────────────────────────────────────────────────────────────


_BUILDERS = {
    "sum_digits": build_sum_digits_dataset,
    "multiply": build_multiply_dataset,
    "word_sort": build_word_sort_dataset,
    "countdown": build_countdown_dataset,
    "arithmetic": build_arithmetic_dataset,
}


def build_ezpz_dataset(
    tokenizer: ModelTokenizer,
    *,
    task: str,
    num_samples: int = 0,
    seed: int = 42,
    max_seq_len: int = 512,
    chat_template: Optional[str] = None,
    **kwargs: Any,
) -> _EzpzPromptDataset:
    """One-call dispatch used by the YAML ``dataset._component_`` field.

    ``chat_template``: when ``"gemma"``, wraps each prompt with the Gemma
    conversation template (``<start_of_turn>user\\n{}<end_of_turn>\\n<start_of_turn>model\\n``)
    so AGPT-2B sees its native pretrain turn markers. Default ``None`` keeps
    raw prompts (the historical behavior). Only the sum_digits / multiply /
    word_sort / countdown / arithmetic builders consume this kwarg today —
    others ignore it via ``**kwargs``.
    """
    if task not in _BUILDERS:
        raise ValueError(f"unknown ezpz task: {task!r}; choices: {list(_BUILDERS)}")
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    return _BUILDERS[task](
        tokenizer,
        num_samples=num_samples,
        seed=seed,
        max_seq_len=max_seq_len,
        **kwargs,
    )

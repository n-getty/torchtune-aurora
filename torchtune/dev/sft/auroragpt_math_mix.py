# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-corpus math SFT mix for AuroraGPT-2B.

Produces a ``ConcatDataset`` of GSM8K + MATH + MetaMathQA-subset, every example
rendered into the exact prompt template the GRPO recipe expects at eval time
(``torchtune.dev.rl.gsm8k.PREAMBLE_PROMPT`` / ``TRAINABLE_PROMPT``).

Why this exists
---------------
The previous SFT pass used ``torchtune.datasets.instruct_dataset`` on GSM8K
only. That pass produced raw "question/answer" pairs without the
``<think>...</think> <answer>...</answer>`` wrapper the GRPO reward parser
keys on (``FormattedMathCorrectnessReward`` requires the answer inside
``<answer>``; ``ThinkingAnswerFormattingReward`` requires both tags in
strict order). The +28% mean-reward lift observed in the 2026-06-14 handoff
came entirely from the loose ``answer in completion`` partial-credit path
(0.5 of the 1.0 ``FormattedMathCorrectnessReward``), not the strict
formatted path — success rate stayed flat at ~2.8%.

This module:
  1. Aligns SFT format to the GRPO template (free format-compliance win).
  2. Broadens the corpus from 7.5K GSM8K only to ~30-50K mixed math
     examples, so the SFT'd model has seen a wider variety of math
     reasoning patterns before RL begins.
  3. Upsamples GSM8K to ~30% of effective examples (concat replication)
     so the format and difficulty distribution stay anchored on the
     downstream eval task.

Per the 2026-06-14 handoff report, the SFT-to-RL bottleneck is success-rate
sparsity (~3%), not the loss path. Lifting success rate is the lever; broader
corpora + format alignment are the two cheapest interventions.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from torchtune.datasets import SFTDataset
from torchtune.datasets._concat import ConcatDataset
from torchtune.dev.rl.gsm8k import PREAMBLE_PROMPT, TRAINABLE_PROMPT
from torchtune.modules.tokenizers import ModelTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Answer extractors
# ──────────────────────────────────────────────────────────────────────────────


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the contents of the LAST top-level ``\\boxed{...}`` in text.

    Handles nested braces (e.g. ``\\boxed{\\frac{1}{4}}``) by tracking depth.
    Returns None if no ``\\boxed{...}`` is present. We take the LAST occurrence
    because MATH solutions sometimes mention an earlier intermediate ``\\boxed``
    before the final answer.
    """
    last = None
    i = 0
    needle = r"\boxed{"
    while True:
        idx = text.find(needle, i)
        if idx == -1:
            break
        start = idx + len(needle)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[start : j - 1]
            i = j
        else:
            break  # unbalanced; give up
    return last


_HASH_ANS_RE = re.compile(r"####\s*([^\n]+)")
_THE_ANSWER_RE = re.compile(r"[Tt]he\s+answer\s+is:?\s*([^\n]+?)\.?\s*$", re.MULTILINE)


def _extract_metamath_answer(response: str) -> Optional[str]:
    """Extract the final answer from a MetaMathQA response.

    GSM-style examples end with ``#### N\\nThe answer is: N``;
    MATH-style examples end with ``... = \\boxed{X}$\\nThe answer is: X``.
    Prefer ``####`` (cleanest), then ``The answer is:``, then ``\\boxed{}``.
    """
    m = _HASH_ANS_RE.search(response)
    if m:
        return m.group(1).strip()
    m = _THE_ANSWER_RE.search(response)
    if m:
        return m.group(1).strip().rstrip(".")
    boxed = _extract_boxed(response)
    if boxed is not None:
        return boxed.strip()
    return None


def _split_metamath_cot(response: str, answer: str) -> str:
    """Strip the trailing answer line(s) from a MetaMathQA response so the
    remainder is a clean CoT we can wrap in ``<think>...</think>``.

    Removes both ``#### N`` and ``The answer is: ...`` trailers if present.
    """
    cot = response
    cot = _HASH_ANS_RE.sub("", cot)
    cot = _THE_ANSWER_RE.sub("", cot)
    return cot.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Per-source message transforms
#
# Each transform maps a raw source row → {"preamble": ..., "trainable": ...}
# matching torchtune.dev.rl.gsm8k.sft_gsm_transform's contract. The shared
# model_transform below tokenizes both halves and masks the preamble.
# ──────────────────────────────────────────────────────────────────────────────


def _gsm8k_transform(row: dict[str, str]) -> dict[str, str]:
    """GSM8K: question / answer with embedded ``#### N`` final answer."""
    question = row["question"]
    solution = row["answer"]
    cot, answer = solution.split("#### ")
    return {
        "preamble": PREAMBLE_PROMPT.format(question=question),
        "trainable": TRAINABLE_PROMPT.format(cot=cot.strip(), answer=answer.strip()),
    }


def _math_transform(row: dict[str, str]) -> Optional[dict[str, str]]:
    """MATH (competition_math mirror): problem / solution with ``\\boxed{}``.

    Returns None when the boxed answer can't be extracted; the caller's
    filter_fn drops these rows so downstream tokenization can assume both
    fields are present.
    """
    problem = row["problem"]
    solution = row["solution"]
    answer = _extract_boxed(solution)
    if answer is None:
        return None
    # Keep the boxed expression in the CoT (often part of the proof flow)
    # but the <answer> tag holds the extracted scalar.
    return {
        "preamble": PREAMBLE_PROMPT.format(question=problem),
        "trainable": TRAINABLE_PROMPT.format(cot=solution.strip(), answer=answer.strip()),
    }


def _metamath_transform(row: dict[str, str]) -> Optional[dict[str, str]]:
    """MetaMathQA: query / response with ``####`` or ``The answer is:`` trailer."""
    query = row["query"]
    response = row["response"]
    answer = _extract_metamath_answer(response)
    if answer is None:
        return None
    cot = _split_metamath_cot(response, answer)
    if not cot:  # answer-only rows aren't useful for CoT-style SFT
        return None
    return {
        "preamble": PREAMBLE_PROMPT.format(question=query),
        "trainable": TRAINABLE_PROMPT.format(cot=cot, answer=answer),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shared model transform (tokenization + preamble masking)
# Mirrors torchtune.dev.rl.gsm8k.gsm8k_sft.model_transform exactly.
# ──────────────────────────────────────────────────────────────────────────────


def _make_model_transform(tokenizer: ModelTokenizer) -> Callable:
    # Hard cap each sample at the tokenizer's max_seq_len. Defensive: the
    # 2026-06-15 first-attempt SFT crashed at step ~10 because the loader
    # emitted samples up to 3079 tokens (MATH long solutions, MetaMathQA
    # outliers) while the model was built with max_seq_len=2048. Truncating
    # at tokenization time guarantees no batch ever exceeds the model cap.
    # When truncating, we preserve the trainable suffix (the answer + final
    # CoT steps) over the preamble — drop preamble tokens first since they
    # are loss-masked anyway.
    max_seq = getattr(tokenizer, "max_seq_len", None)

    def model_transform(problem: dict[str, str]) -> dict[str, list[int]]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False)
        if max_seq is not None and len(pre_tokens) + len(trainable_tokens) > max_seq:
            # First, hard-cap the trainable span at max_seq (rare; only if a
            # single answer alone overflows). Then trim preamble from the
            # head to fit the remainder.
            trainable_tokens = trainable_tokens[:max_seq]
            keep_pre = max_seq - len(trainable_tokens)
            pre_tokens = pre_tokens[-keep_pre:] if keep_pre > 0 else []
        # 1 == mask out of loss (preamble); 0 == contributes to loss
        mask = [1] * len(pre_tokens) + [0] * len(trainable_tokens)
        return {"tokens": pre_tokens + trainable_tokens, "mask": mask}

    return model_transform


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────


def auroragpt_math_mix_sft(
    tokenizer: ModelTokenizer,
    *,
    gsm8k_source: str = "openai/gsm8k",
    gsm8k_name: str = "main",
    math_source: str = "qwedsacf/competition_math",
    metamath_source: str = "meta-math/MetaMathQA",
    metamath_subset: int = 25000,
    metamath_seed: int = 42,
    gsm8k_replicas: int = 3,
    math_replicas: int = 1,
    metamath_replicas: int = 1,
) -> ConcatDataset:
    """Build the GSM8K + MATH + MetaMathQA mix used by the multi-corpus SFT.

    Replica counts shape the per-epoch token mix. Defaults (gsm8k=3, math=1,
    metamath_subset=25K, metamath=1) produce:

        GSM8K:      7.5K * 3 = 22.5K  (39% of ~57.5K rows; format anchor)
        MATH:      12.5K * 1 = 12.5K  (22%)
        MetaMathQA: 25.0K * 1 = 25.0K  (43%)

    GSM8K examples are short (avg ~150 tokens of CoT); MATH and MetaMathQA
    are longer, so the GSM8K *token* share lands around 25-30%, which keeps
    the format and difficulty distribution anchored on the downstream eval
    while exposing the model to genuinely harder problems and a wider
    distribution of reasoning patterns.

    Args:
        tokenizer: the AuroraGPT sentencepiece tokenizer; must expose
            ``encode(text, add_bos=..., add_eos=...)``.
        gsm8k_source / gsm8k_name: HF dataset id / config for GSM8K.
        math_source: HF dataset id for the MATH mirror (the original
            ``hendrycks/competition_math`` is DMCA'd; the ``qwedsacf`` mirror
            is bit-identical and accessible).
        metamath_source: HF dataset id for MetaMathQA.
        metamath_subset: how many MetaMathQA rows to use per replica. Drawn
            with seeded shuffle so the same subset reloads deterministically.
            Set to 0 to disable MetaMathQA entirely.
        metamath_seed: shuffle seed for the MetaMathQA subsample.
        gsm8k_replicas / math_replicas / metamath_replicas: per-source
            replication count (concat-style upsampling). 0 disables a source.

    Returns:
        ``ConcatDataset`` ready to feed into ``full_finetune_distributed_xpu``.
    """
    parts = []

    if gsm8k_replicas > 0:
        gsm8k_one = SFTDataset(
            source=gsm8k_source,
            message_transform=_gsm8k_transform,
            model_transform=_make_model_transform(tokenizer),
            split="train",
            name=gsm8k_name,
        )
        parts.extend([gsm8k_one] * gsm8k_replicas)

    if math_replicas > 0:
        # filter_fn runs BEFORE message_transform; the message transform
        # returns None on un-extractable rows, but SFTDataset doesn't filter
        # on None. Use the inline guard in __getitem__ via a pre-pass filter
        # by checking the boxed extraction.
        def _math_filter(row, _idx=None):
            return _extract_boxed(row.get("solution", "")) is not None

        math_one = SFTDataset(
            source=math_source,
            message_transform=_math_transform,
            model_transform=_make_model_transform(tokenizer),
            filter_fn=_math_filter,
            split="train",
        )
        parts.extend([math_one] * math_replicas)

    if metamath_replicas > 0 and metamath_subset > 0:
        def _metamath_filter(row, _idx=None):
            resp = row.get("response", "")
            ans = _extract_metamath_answer(resp)
            if ans is None:
                return False
            # Drop answer-only rows (no usable CoT). Mirrors _metamath_transform's
            # `if not cot: return None` guard so SFTDataset never sees a None sample.
            return bool(_split_metamath_cot(resp, ans))

        # Subsample via HF split slicing after a seeded shuffle. The shuffle
        # is deterministic; the slice keeps memory bounded.
        # We use HF's built-in `split` slicing combined with `.shuffle(seed)`
        # would require a second pass — instead pass `split` with no slice
        # and let `filter_fn` thin it, then we accept the natural order;
        # to actually subsample by count we wrap the SFTDataset post-hoc.
        metamath_full = SFTDataset(
            source=metamath_source,
            message_transform=_metamath_transform,
            model_transform=_make_model_transform(tokenizer),
            filter_fn=_metamath_filter,
            split="train",
        )
        metamath_sub = _SubsampledDataset(
            metamath_full, num_samples=metamath_subset, seed=metamath_seed
        )
        parts.extend([metamath_sub] * metamath_replicas)

    if not parts:
        raise ValueError("auroragpt_math_mix_sft: all sources disabled")

    return ConcatDataset(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Subsampler — deterministic index permutation, no data copy.
# ──────────────────────────────────────────────────────────────────────────────

import random
from torch.utils.data import Dataset


class _SubsampledDataset(Dataset):
    """Deterministic random subset of a wrapped Dataset.

    Stores only the index permutation, not the data. Re-creates the same
    permutation on every process (seeded), so DistributedSampler still sees
    a consistent global view across ranks.
    """

    def __init__(self, base: Dataset, *, num_samples: int, seed: int) -> None:
        self._base = base
        n = len(base)
        k = min(num_samples, n)
        rng = random.Random(seed)
        # sample without replacement
        self._indices = rng.sample(range(n), k)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._base[self._indices[idx]]

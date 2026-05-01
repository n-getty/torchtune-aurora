# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import re
from typing import Any, Optional

from datasets import load_from_disk
from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer

from .data import BASE_PROMPT


def _extract_boxed_answer(solution: str) -> str:
    """Extract the last \\boxed{...} answer from a MATH solution, handling nested braces."""
    idx = solution.rfind(r"\boxed{")
    if idx == -1:
        return solution.strip().split("\n")[-1]
    depth = 0
    for i, c in enumerate(solution[idx + 7:], start=idx + 7):
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return solution[idx + 7 : i]
            depth -= 1
    return solution[idx + 7:]


class CompetitionMathDataset(Dataset):
    """
    Competition math dataset (MATH benchmark) loaded from disk, prepared for GRPO.
    Problems are hard enough to require 500-1500 token reasoning chains from Qwen3.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        dataset_path: str,
        min_level: int = 3,
        max_level: int = 5,
        partition: Optional[str] = None,
    ) -> None:
        self._tokenizer = tokenizer
        ds = load_from_disk(dataset_path)

        def _keep(example, idx):
            level = example.get("level", "")
            # level field is like "Level 4" or an int
            try:
                lvl = int(str(level).replace("Level", "").strip())
            except ValueError:
                lvl = 3
            if not (min_level <= lvl <= max_level):
                return False
            if partition is not None:
                match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
                if match:
                    start, end, total = map(int, match.groups())
                    return start <= (idx % total) <= end
            return True

        self._data = ds.filter(_keep, with_indices=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        question = sample["problem"]
        answer = _extract_boxed_answer(sample["solution"])

        prompt = BASE_PROMPT % question
        tokens = self._tokenizer.encode(prompt, add_eos=False)
        mask = [1] * len(tokens)

        return {
            "question": prompt,
            "tokens": tokens,
            "mask": mask,
            "answer": answer,
        }


def competition_math_dataset(
    tokenizer: ModelTokenizer,
    *,
    dataset_path: str = "/lus/flare/projects/ModCon/ngetty/datasets/competition_math_train",
    min_level: int = 3,
    max_level: int = 5,
    partition: Optional[str] = None,
) -> CompetitionMathDataset:
    """
    Competition math dataset for GRPO varlen benchmarking.
    Uses level 3-5 problems to force 500-1500 token reasoning chains.
    Load from pre-downloaded Lustre path (no HF download at training time).
    """
    return CompetitionMathDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        min_level=min_level,
        max_level=max_level,
        partition=partition,
    )

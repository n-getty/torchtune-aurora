# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Sum-of-digits task for GRPO RL training.

Replicates the sum_digits task from torchtitan/ezpz for apples-to-apples
comparison. Each sample is a random addition problem (2-5 addends, digits 0-9).

Prompt: "What is 3 + 7 + 2? Reply with just the number."
Answer: "12"
"""

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer


def generate_sum_digits_data(
    output_path: str,
    num_samples: int = 1000,
    min_addends: int = 2,
    max_addends: int = 5,
    max_digit: int = 9,
    seed: int = 42,
) -> None:
    """Generate sum-of-digits JSONL dataset matching torchtitan/ezpz format."""
    rng = random.Random(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for _ in range(num_samples):
            n_addends = rng.randint(min_addends, max_addends)
            digits = [rng.randint(0, max_digit) for _ in range(n_addends)]
            expr = " + ".join(str(d) for d in digits)
            answer = str(sum(digits))
            prompt = f"What is {expr}? Reply with just the number."
            f.write(json.dumps({"input": prompt, "output": answer}) + "\n")


class SumDigitsDataset(Dataset):
    """Dataset for sum-of-digits GRPO training from a local JSONL file."""

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        data_files: str,
        max_seq_len: int = 512,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._data = []
        with open(data_files, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        prompt = sample["input"]
        answer = sample["output"]

        tokens = self._tokenizer.encode(prompt, add_eos=False)
        if len(tokens) > self._max_seq_len:
            tokens = tokens[: self._max_seq_len]

        return {
            "question": prompt,
            "tokens": tokens,
            "mask": [1] * len(tokens),
            "answer": answer,
        }


def sum_digits_dataset(
    tokenizer: ModelTokenizer,
    *,
    data_files: str,
    max_seq_len: int = 512,
) -> SumDigitsDataset:
    """Factory function for config instantiation."""
    return SumDigitsDataset(
        tokenizer=tokenizer,
        data_files=data_files,
        max_seq_len=max_seq_len,
    )

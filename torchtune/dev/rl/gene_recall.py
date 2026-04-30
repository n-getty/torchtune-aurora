# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Gene recall dataset for RL training.

Reads JSONL files produced by prepare_gene_recall_data.py.  Each line has:
  {"input": "<full prompt>", "output": "GENE1, GENE2, ..."}

The narrative is extracted from the "input" field and re-formatted with a
reasoning-style prompt that instructs the model to think through the biology
before outputting a structured gene list inside <genes>...</genes> tags.
The reward parser reads from the tag, with a fallback to full-text scan.
"""

import json
import re
from typing import Any

from torch.utils.data import Dataset

from torchtune.modules.tokenizers import ModelTokenizer


# Reasoning-style prompt: instructs the model to think first, then
# produce a structured final answer inside <genes> tags.
_PROMPT_TEMPLATE = (
    "You are a molecular biology expert. Given a description of a cancer "
    "biology process, reason step by step about which genes are core "
    "players, then provide your final answer.\n\n"
    "Process description:\n{narrative}\n\n"
    "Think through the key pathways and regulators involved, then output "
    "a comma-separated list of HGNC gene symbols inside <genes> tags.\n\n"
    "Example format:\n"
    "[Your reasoning here...]\n"
    "<genes>TP53, BRCA1, MYC</genes>"
)

# Regex to pull the narrative out of the original JSONL prompt format.
_NARRATIVE_RE = re.compile(
    r"Process description:\n(.*?)\n\nGene symbols:", re.DOTALL
)


def _build_prompt(raw_input: str) -> str:
    """Extract the narrative from the original prompt and reformat it."""
    m = _NARRATIVE_RE.search(raw_input)
    if m:
        narrative = m.group(1).strip()
    else:
        # Fallback: use the whole input as the narrative
        narrative = raw_input.strip()
    return _PROMPT_TEMPLATE.format(narrative=narrative)


class GeneRecallDataset(Dataset):
    """Dataset for gene recall RL training from a local JSONL file."""

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        data_files: str,
        max_seq_len: int = 2048,
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
        prompt = _build_prompt(sample["input"])
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


def gene_recall_dataset(
    tokenizer: ModelTokenizer,
    *,
    data_files: str,
    max_seq_len: int = 2048,
) -> GeneRecallDataset:
    """
    Gene recall dataset loaded from a local JSONL file.

    Each sample's narrative is extracted from the original prompt and
    reformatted to encourage reasoning before a structured <genes>...</genes>
    final answer. The reward parser extracts genes from the tag, with a
    fallback to full-text scanning during early training.

    Args:
        tokenizer (ModelTokenizer): tokenizer for encoding prompts
        data_files (str): path to the JSONL file
        max_seq_len (int): maximum prompt token length; longer prompts are truncated

    Returns:
        GeneRecallDataset
    """
    return GeneRecallDataset(
        tokenizer=tokenizer,
        data_files=data_files,
        max_seq_len=max_seq_len,
    )

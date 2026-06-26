"""
SFT dataset for BioReason multimodal fine-tuning on a native Gemma 4 backbone.

Differs from the RL dataset (:mod:`torchtune.dev.bioreason.dataset`) in two ways that
matter for the native Gemma4 path:

1. **Placeholder tokens are RESERVED Gemma vocab ids, spliced as integers.** The native
   :class:`Gemma4Tokenizer` has no ``add_special_tokens`` / chat-template path, and the
   ``<|protein_pad|>`` / ``<|go_graph_pad|>`` strings shatter under Gemma's BPE. So we
   reserve two ``<unused*>`` vocab ids and insert them directly into the token id list at
   the right counts (``len(seq)+2`` protein, ``num_go_tokens`` GO). The model fills those
   positions from the trainable projections.

2. **It returns labels (CE targets) with the prompt span masked.** SFT supervises the
   ``reasoning`` (optional) + ``final_answer`` assistant span only; the prompt — including
   the placeholder runs — is masked with ``CROSS_ENTROPY_IGNORE_IDX``.

Default placeholder ids are the top two unused Gemma vocab entries
(``<unused6225>``=262142 protein, ``<unused6226>``=262143 GO). They are single tokens by
construction and never produced by normal prompt text (pinned by a CPU test).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX

logger = logging.getLogger(__name__)

# Reserved Gemma-4 vocab ids used as protein / GO placeholder positions.
DEFAULT_PROTEIN_TOKEN_ID = 262142  # <unused6225>
DEFAULT_GO_TOKEN_ID = 262143  # <unused6226>


def _nonempty(v) -> bool:
    if v is None:
        return False
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


# Verbatim from the RL dataset's _SYS_WITH_CONTEXT (the paper's
# CAFA5_REASONING_TEMPLATE_WITH_CONTEXT system block). Kept in sync by a CPU test that
# asserts byte-equality against torchtune.dev.bioreason.dataset.
_SYS_WITH_CONTEXT = (
    "You are a scientific assistant specialized in protein function prediction. "
    "Given a protein sequence, organism information, and additional context (InterPro "
    "domain annotations and/or initial GO term speculations), step-by-step reason about "
    "the InterPro terms, Gene Ontology (GO) terms regarding molecular function, "
    "biological process, and cellular component, protein-protein interactions (PPI), and "
    "overall function. Use the provided information as a starting point and improve upon "
    "it with deeper analysis. Provide a summary of your findings in your final answer."
)


class BioReasonSFTDataset(Dataset):
    """Multimodal SFT dataset for the native-Gemma4 BioReason model.

    Each example yields:
        tokens:  [S] full token stream (prompt with spliced placeholder ids + target)
        labels:  [S] CE targets; prompt span (incl. placeholders) = IGNORE
        protein_sequence: str — raw (truncated) AA sequence
        go_aspect: str

    Args:
        data_files (str): directory or comma-separated parquet/jsonl paths.
        tokenizer: a Gemma4Tokenizer (encode(text, add_bos, add_eos) -> list[int]).
        max_seq_len (int): truncate the full stream to this length.
        max_protein_len (int): truncate AA sequence (must match the ESM3 cache key).
        num_go_tokens (int): GO placeholder count (must match go_embedding.pt slice).
        protein_token_id (int): reserved id for protein placeholders.
        go_token_id (int): reserved id for GO placeholders.
        train_on_reasoning (bool): supervise reasoning+final_answer (True) or
            final_answer only (False). Default: True (matches published SFT).
        inject_go_pred (bool): use the paper-faithful go_speculations prompt. Default: True.
    """

    def __init__(
        self,
        data_files: str,
        tokenizer,
        max_seq_len: int = 4096,
        max_protein_len: int = 2048,
        num_go_tokens: int = 200,
        protein_token_id: int = DEFAULT_PROTEIN_TOKEN_ID,
        go_token_id: int = DEFAULT_GO_TOKEN_ID,
        train_on_reasoning: bool = True,
        inject_go_pred: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_protein_len = max_protein_len
        self.num_go_tokens = num_go_tokens
        self.protein_token_id = protein_token_id
        self.go_token_id = go_token_id
        self.train_on_reasoning = train_on_reasoning
        self.inject_go_pred = inject_go_pred
        self.examples = self._load(data_files)
        logger.info(
            "Loaded %d BioReason SFT examples from %s", len(self.examples), data_files
        )

    def _load(self, data_files: str) -> list[dict]:
        # os.walk + extension filter (NOT stdlib glob — hangs on DAOS/dfuse).
        if os.path.isdir(data_files):
            parquets, jsonls = [], []
            for root, _dirs, files in os.walk(data_files):
                for fn in files:
                    if fn.endswith(".parquet"):
                        parquets.append(os.path.join(root, fn))
                    elif fn.endswith(".jsonl"):
                        jsonls.append(os.path.join(root, fn))
            paths = sorted(parquets) + sorted(jsonls)
        else:
            paths = [p.strip() for p in data_files.split(",")]

        examples: list[dict] = []
        for path in paths:
            if not os.path.exists(path):
                logger.warning("Data file not found: %s", path)
                continue
            if path.endswith(".parquet"):
                import pandas as pd

                df = pd.read_parquet(path)
                examples.extend(df.to_dict("records"))
                logger.info("Loaded %d rows from %s", len(df), path)
            else:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                examples.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning("Skipping malformed line in %s", path)
        if not examples:
            raise RuntimeError(f"No examples loaded from {data_files}.")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _build_prompt_text(self, ex: dict) -> str:
        """Paper-faithful go_pred prompt (system+user folded into one text block).
        Mirrors BioReasonRLDataset._build_go_pred_prompt_text exactly (pinned by test)."""
        org = ex.get("organism", "") or "Unknown"
        interpro_data = ex.get("interpro_formatted", "") or ""
        ppi_data = ex.get("ppi_formatted", "") or ""
        go_spec = ex.get("go_pred", "") or ""

        aspects = []
        if _nonempty(ex.get("go_mf")):
            aspects.append("Molecular Function")
        if _nonempty(ex.get("go_cc")):
            aspects.append("Cellular Component")
        if _nonempty(ex.get("go_bp")):
            aspects.append("Biological Process")
        go_aspects_suffix = (
            f" and focus more on its {', '.join(aspects)}." if aspects else "."
        )

        if ppi_data and (interpro_data or go_spec):
            user = (
                f"Given the protein above from organism {org} with the following InterPro "
                f"annotations:\n{interpro_data if interpro_data else 'None'}\n\n"
                f"And the following protein-protein interaction partners:\n"
                f"{ppi_data if ppi_data else 'None'}\n\n"
                f"And the following initial GO term speculations:\n"
                f"{go_spec if go_spec else 'None'}\n\n"
                f"Reason about the function of the protein{go_aspects_suffix}"
            )
        else:
            user = (
                f"Given the protein above from organism {org} with the following InterPro "
                f"annotations:\n{interpro_data}\n\n"
                f"And the following initial GO term speculations:\n{go_spec}\n\n"
                f"Reason about the function of the protein."
            )
        return f"{_SYS_WITH_CONTEXT.strip()}\n\n{user.strip()}"

    @staticmethod
    def _strip_bos(ids: list[int], bos_id: int) -> list[int]:
        """The Gemma HF tokenizer always prepends BOS; strip it from non-initial
        segments so the assembled stream has exactly one leading BOS."""
        if ids and ids[0] == bos_id:
            return ids[1:]
        return ids

    def _build_prompt_ids(self, ex: dict, protein_seq: str) -> list[int]:
        """Assemble prompt token ids with placeholder ids spliced in.

        Layout: [BOS] <prompt_text> "\nProtein: " [PROT]*(L+2) "\nGO graph: " [GO]*N
                "\nReasoning:\n"
        The protein/GO marker text is plain (the native tokenizer has no special-token
        path); the model fills the reserved-id runs from the projections.
        """
        tok = self.tokenizer
        bos = tok.bos_id
        text = self._build_prompt_text(ex)

        ids: list[int] = list(tok.encode(text, add_bos=True, add_eos=False))
        ids += self._strip_bos(
            tok.encode("\nProtein: ", add_bos=False, add_eos=False), bos
        )
        ids += [self.protein_token_id] * (len(protein_seq) + 2)
        ids += self._strip_bos(
            tok.encode("\nGO graph: ", add_bos=False, add_eos=False), bos
        )
        ids += [self.go_token_id] * self.num_go_tokens
        ids += self._strip_bos(
            tok.encode("\nReasoning:\n", add_bos=False, add_eos=False), bos
        )
        return ids

    def _build_target_ids(self, ex: dict) -> list[int]:
        reasoning = ex.get("reasoning", "") or ""
        final = ex.get("final_answer", "") or ""
        if self.train_on_reasoning and reasoning:
            target = f"{reasoning}\n{final}"
        else:
            target = final
        ids = self.tokenizer.encode(target, add_bos=False, add_eos=True)
        return self._strip_bos(ids, self.tokenizer.bos_id)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        protein_seq = (ex.get("sequence", "") or "")[: self.max_protein_len]
        go_aspect = ex.get("go_aspect", "all") or "all"

        prompt_ids = self._build_prompt_ids(ex, protein_seq)
        target_ids = self._build_target_ids(ex)

        tokens = prompt_ids + target_ids
        labels = [CROSS_ENTROPY_IGNORE_IDX] * len(prompt_ids) + list(target_ids)

        # Truncate (right) to max_seq_len, keeping tokens/labels aligned.
        tokens = tokens[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        # Fail-fast data-pipeline check: placeholder counts survived assembly.
        n_prot = sum(1 for t in tokens if t == self.protein_token_id)
        n_go = sum(1 for t in tokens if t == self.go_token_id)
        # (counts can be < expected only if truncation cut into the prompt — that would
        # break the embed-splice contract, so surface it.)
        expected_prot = len(protein_seq) + 2
        if n_prot != expected_prot or n_go != self.num_go_tokens:
            raise ValueError(
                f"Placeholder count mismatch after assembly/truncation: "
                f"protein {n_prot}!={expected_prot} or GO {n_go}!={self.num_go_tokens}. "
                f"Increase max_seq_len (prompt too long for the budget)."
            )

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "protein_sequence": protein_seq,
            "go_aspect": go_aspect,
        }


def bioreason_sft_collate_fn(
    batch: list[dict],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    max_seq_len: Optional[int] = None,
) -> dict:
    """Pad tokens (padding_idx) and labels (ignore_idx) to a common length; attach the
    raw protein/go string lists (not padded). Mirrors padded_collate_sft for the
    token/label tensors while carrying the multimodal side inputs."""
    seqs = [ex["tokens"] for ex in batch]
    lbls = [ex["labels"] for ex in batch]
    max_len = max(s.shape[0] for s in seqs)
    if max_seq_len is not None:
        max_len = min(max_len, max_seq_len)

    tok_out = torch.full((len(batch), max_len), padding_idx, dtype=torch.long)
    lbl_out = torch.full((len(batch), max_len), ignore_idx, dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, lbls)):
        n = min(s.shape[0], max_len)
        tok_out[i, :n] = s[:n]
        lbl_out[i, :n] = l[:n]

    return {
        "tokens": tok_out,
        "labels": lbl_out,
        "protein_sequences": [ex["protein_sequence"] for ex in batch],
        "go_aspects": [ex["go_aspect"] for ex in batch],
    }


def bioreason_sft_dataset(
    tokenizer,
    data_files: str,
    max_seq_len: int = 4096,
    max_protein_len: int = 2048,
    num_go_tokens: int = 200,
    protein_token_id: int = DEFAULT_PROTEIN_TOKEN_ID,
    go_token_id: int = DEFAULT_GO_TOKEN_ID,
    train_on_reasoning: bool = True,
    inject_go_pred: bool = True,
) -> BioReasonSFTDataset:
    """TorchTune component factory (YAML config entry point)."""
    return BioReasonSFTDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_protein_len=max_protein_len,
        num_go_tokens=num_go_tokens,
        protein_token_id=protein_token_id,
        go_token_id=go_token_id,
        train_on_reasoning=train_on_reasoning,
        inject_go_pred=inject_go_pred,
    )

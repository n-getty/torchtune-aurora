"""
Dataset loader for BioReason-Pro RL training.

Source: wanglab/bioreason-pro-rl-reasoning-data (9.2k examples)
Schema (to be confirmed after download):
  - protein_sequence: str  — amino acid sequence
  - go_aspect: str         — "all" / "bp" / "mf" / "cc"
  - prompt: str            — formatted input text (protein description + GO context)
  - go_ground_truth: str   — comma-separated GO:XXXXXXX terms (reward signal)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Placeholder token strings (must match the tokenizer's extended vocab)
PROTEIN_PAD = "<|protein_pad|>"
GO_PAD = "<|go_graph_pad|>"


def _nonempty(v) -> bool:
    """True if a parquet cell (str / list / ndarray / None) holds real content.
    go_{bp,mf,cc} are stored as ndarrays; len() works but `if v` is ambiguous for arrays."""
    if v is None:
        return False
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


class BioReasonRLDataset(Dataset):
    """
    Dataset for BioReason-Pro GRPO RL fine-tuning.

    Each example yields:
        tokens: [ctx_len] — tokenized prompt with protein_pad and go_graph_pad tokens
        protein_sequence: str — raw amino acid sequence
        go_aspect: str — GO namespace
        answer: str — comma-separated ground truth GO terms
    """

    def __init__(
        self,
        data_files: str,
        tokenizer,
        max_seq_len: int = 2048,
        max_protein_len: int = 512,
        num_go_tokens: int = 200,
        answer_column: str = "go_ids",
        inject_go_pred: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_protein_len = max_protein_len
        self.num_go_tokens = num_go_tokens
        # RL reward target column. MUST be the experimental ground truth (`go_ids`,
        # == go_bp∪go_mf∪go_cc == the eval F_max target). The old default `go_pred`
        # was a BUG: it's the GO-GPT submodel's noisy PREDICTIONS (~50% recall of GT),
        # so RL optimized toward another model's guesses, not truth — flat reward +
        # can degrade eval F_max. Kept configurable only for A/B against the old runs.
        self.answer_column = answer_column
        # Train/eval/SFT prompt-distribution match. The SFT ckpt was TRAINED to REFINE
        # GO-GPT's predictions (the `go_pred` column, injected as "go_speculations"), and
        # the fixed eval injects them too (eval_cafa_fmax.py --inject_go_pred, which lifted
        # the published SFT from 0.41 → ~0.65 F_max). But RL training used a COLD prompt
        # with NO go_pred → RL optimized an off-distribution prompt that differs from both
        # SFT and eval. inject_go_pred=True builds the SAME paper-faithful prompt the eval
        # uses (paper's _format_reasoning_prompt + format_cafa5_for_protein_llm), so the
        # train, SFT, and eval prompt distributions all match. Default False keeps the
        # legacy cold-prompt path byte-identical for A/B. See
        # memory/project_bioreason_eval_fixed_rl_flat_vs_sft_20260626.
        self.inject_go_pred = inject_go_pred

        self.examples = self._load(data_files)
        logger.info(f"Loaded {len(self.examples)} BioReason RL examples from {data_files}")

    def _load(self, data_files: str) -> list[dict]:
        # Collect paths: directory or comma-separated list.
        # Use os.walk + extension filter (NOT the stdlib glob module) — recursive
        # glob hangs on DAOS/dfuse mounts (see CLAUDE.md "Critical Platform
        # Constraints"). The regression test asserts the substring is absent.
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

        examples = []
        for path in paths:
            if not os.path.exists(path):
                logger.warning(f"Data file not found: {path}")
                continue
            if path.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(path)
                examples.extend(df.to_dict("records"))
                logger.info(f"Loaded {len(df)} rows from {path}")
            else:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                examples.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping malformed line in {path}")

        if not examples:
            raise RuntimeError(
                f"No examples loaded from {data_files}. "
                "Download: huggingface-cli download --repo-type dataset "
                "wanglab/bioreason-pro-rl-reasoning-data "
                "--local-dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl"
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    # Verbatim from BioReason-Pro/bioreason2/dataset/prompts/cafa5.py — the
    # CAFA5_REASONING_TEMPLATE_WITH_CONTEXT* the published SFT was trained on and the
    # fixed eval uses. Reproduced here (rather than imported) to keep the dataloader hot
    # path free of the heavy bioreason2 import; test_bioreason_go_pred_prompt asserts byte
    # equality against the paper module so any upstream drift is caught.
    _SYS_WITH_CONTEXT = (
        "You are a scientific assistant specialized in protein function prediction. "
        "Given a protein sequence, organism information, and additional context (InterPro "
        "domain annotations and/or initial GO term speculations), step-by-step reason about "
        "the InterPro terms, Gene Ontology (GO) terms regarding molecular function, "
        "biological process, and cellular component, protein-protein interactions (PPI), and "
        "overall function. Use the provided information as a starting point and improve upon "
        "it with deeper analysis. Provide a summary of your findings in your final answer."
    )

    def _build_go_pred_prompt_text(self, ex: dict) -> str:
        """Fold the paper's system+user WITH_CONTEXT* prompt into a single text block,
        injecting go_pred as go_speculations. Selects the PPI variant when ppi_formatted is
        present (matches _format_reasoning_prompt's branch: ppi_in_prompt and (interpro or
        go_speculations)). go_aspects_suffix is built from which go_{mf,cc,bp} are present."""
        org = ex.get("organism", "") or "Unknown"
        interpro_data = ex.get("interpro_formatted", "") or ""
        ppi_data = ex.get("ppi_formatted", "") or ""
        go_spec = ex.get("go_pred", "") or ""

        # go_aspects_suffix (paper order: MF, CC, BP)
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
        # format_cafa5_for_protein_llm folds system + user into the single text block.
        return f"{self._SYS_WITH_CONTEXT.strip()}\n\n{user.strip()}"

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        # Schema from wanglab/bioreason-pro-rl-reasoning-data (confirmed after download):
        #   sequence, go_pred, protein_names, protein_function, organism,
        #   ppi_formatted, interpro_formatted, go_bp, go_mf, go_cc, go_ids
        protein_seq = ex.get("sequence", "")
        go_aspect = ex.get("go_aspect", "all") or "all"
        # Reward target: the experimental ground-truth GO terms (self.answer_column,
        # default "go_ids" == eval F_max target). go_ids is stored as a list/ndarray;
        # join to a comma-separated string (the reward fn regex-extracts GO:####### so
        # the exact separator is cosmetic, but a clean string keeps logs readable).
        _ans = ex.get(self.answer_column, "")
        if _ans is None:
            _ans = ""
        if not isinstance(_ans, str):  # list / ndarray of GO terms
            _ans = ", ".join(str(t) for t in _ans)
        answer = _ans

        # Build the context text from available fields
        name = ex.get("protein_names", "")
        func = ex.get("protein_function", "")
        org = ex.get("organism", "")
        ppi = ex.get("ppi_formatted", "")
        interpro = ex.get("interpro_formatted", "")
        if self.inject_go_pred:
            # Paper-faithful prompt (system+user folded into the text block), injecting
            # go_pred as "go_speculations" exactly as the fixed eval does. We build the
            # text here (NOT via the heavy bioreason2 import in the dataloader hot path)
            # but mirror _format_reasoning_prompt's CAFA5_REASONING_TEMPLATE_WITH_CONTEXT*
            # verbatim so train == eval == SFT. Pinned by test_bioreason_go_pred_prompt.
            prompt_text = self._build_go_pred_prompt_text(ex)
        else:
            prompt_text = (
                f"Protein: {name} ({org})\n"
                f"Function: {func}\n"
                + (f"Domains: {interpro}\n" if interpro else "")
                + (f"Interactions: {ppi}\n" if ppi else "")
                + "Predict the GO terms for this protein."
            )

        # Truncate protein sequence to max length
        protein_seq = protein_seq[:self.max_protein_len]

        # Render the SFT-canonical chat template (chat_template.jinja). The user
        # block is a content list so the template emits "Protein: <|protein_pad|>\n\n"
        # and "GO graph: <|go_graph_pad|>\n\n" (matches what bioreason-pro-sft saw).
        # add_generation_prompt=True ends the string at "<|im_start|>assistant\n"
        # and lets the model decide whether to emit <think> itself — never inject
        # a hardcoded "<think>\n" suffix here, that forces every completion into
        # reasoning mode and zeros out the reward signal under any practical
        # max_generated_tokens budget.
        messages = [
            {"role": "system", "content": "You are an expert in protein function prediction."},
            {"role": "user", "content": [
                {"type": "protein"},
                {"type": "go_graph"},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        # Some torchtune tokenizers don't expose apply_chat_template; reach the
        # underlying HF tokenizer. BioReasonHFTokenizer stores it as `_tok`;
        # torchtune wrappers expose it as `tokenizer`.
        _hf_tok = getattr(self.tokenizer, "_tok", None) \
                  or getattr(self.tokenizer, "tokenizer", self.tokenizer)
        full_prompt = _hf_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Expand the single placeholder tokens to the true counts the embedding
        # pipeline expects. Match upstream PLProcessor (processing_pl.py:184-185):
        # protein placeholders = len(seq) + 2 (ESM3 BOS/EOS) so build_prompt_embeds
        # can fill them directly with the unstripped ESM3 per-residue features.
        protein_placeholders_count = len(protein_seq) + 2
        full_prompt = full_prompt.replace(
            PROTEIN_PAD, PROTEIN_PAD * protein_placeholders_count, 1,
        )
        full_prompt = full_prompt.replace(
            GO_PAD, GO_PAD * self.num_go_tokens, 1,
        )

        # HF tokenizer: encode returns list[int]; TorchTune tokenizer may differ
        encoded = self.tokenizer.encode(full_prompt)
        if isinstance(encoded, dict):
            tokens = encoded["input_ids"]
        else:
            tokens = encoded
        tokens = tokens[:self.max_seq_len]

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "protein_sequence": protein_seq,
            "go_aspect": go_aspect,
            "answer": answer,
        }


def bioreason_collate_fn(
    batch: list[dict],
    padding_idx: int,
    max_seq_len: Optional[int] = None,
) -> dict:
    """
    Collate BioReason examples into a padded batch.

    Returns a dict with:
        tokens: [B, ctx_len] — padded token IDs
        protein_sequences: List[str] — raw sequences (variable length, not padded)
        go_aspects: List[str]
        answers: List[str]
    """
    # Pad tokens to same length
    seqs = [ex["tokens"] for ex in batch]
    max_len = max(s.shape[0] for s in seqs)
    if max_seq_len is not None:
        max_len = min(max_len, max_seq_len)

    padded = torch.full((len(batch), max_len), padding_idx, dtype=torch.long)
    for i, seq in enumerate(seqs):
        n = min(seq.shape[0], max_len)
        padded[i, :n] = seq[:n]

    return {
        "tokens": padded,
        "protein_sequences": [ex["protein_sequence"] for ex in batch],
        "go_aspects": [ex["go_aspect"] for ex in batch],
        "answers": [ex["answer"] for ex in batch],
    }


def bioreason_rl_dataset(
    tokenizer,
    data_files: str,
    max_seq_len: int = 2048,
    max_protein_len: int = 512,
    num_go_tokens: int = 200,
    answer_column: str = "go_ids",
    inject_go_pred: bool = False,
) -> BioReasonRLDataset:
    """TorchTune component factory for use in YAML configs."""
    return BioReasonRLDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_protein_len=max_protein_len,
        num_go_tokens=num_go_tokens,
        answer_column=answer_column,
        inject_go_pred=inject_go_pred,
    )


class BioReasonHFTokenizer:
    """
    Thin wrapper around HuggingFace AutoTokenizer for BioReason-Pro.

    Presents the subset of the TorchTune tokenizer interface used by the recipe:
      - encode(text) -> list[int]
      - decode(ids) -> str
      - pad_id, eos_id, stop_tokens

    Uses HF because BioReason extends the Qwen3 vocab with special tokens
    (<|protein_pad|>, <|go_graph_pad|>) that are not in the base vocab.json.
    """

    def __init__(self, ckpt_dir: str):
        import sys, types, os
        _BIOREASON_SRC = os.environ.get(
            "BIOREASON_SRC", "/flare/ModCon/ngetty/BioReason-Pro"
        )
        for pkg_name, pkg_path in [
            ("bioreason2", f"{_BIOREASON_SRC}/bioreason2"),
            ("bioreason2.models", f"{_BIOREASON_SRC}/bioreason2/models"),
        ]:
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = [pkg_path]
                pkg.__package__ = pkg_name
                sys.modules[pkg_name] = pkg

        from transformers import AutoTokenizer
        from bioreason2.models.special_tokens import get_all_special_tokens
        self._tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        self._tok.add_special_tokens(
            {"additional_special_tokens": get_all_special_tokens()}
        )
        self.pad_id = self._tok.pad_token_id or self._tok.eos_token_id
        self.eos_id = self._tok.eos_token_id
        self.stop_tokens = [self.eos_id]  # recipe uses this for truncation

    def encode(self, text: str, **kwargs) -> list:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids, **kwargs) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def __len__(self) -> int:
        return len(self._tok)


def bioreason_tokenizer(ckpt_dir: str) -> "BioReasonHFTokenizer":
    """TorchTune component factory for BioReason tokenizer (YAML config entry point)."""
    return BioReasonHFTokenizer(ckpt_dir=ckpt_dir)

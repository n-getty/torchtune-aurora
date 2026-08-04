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
import re
from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler

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
        go_pred_dropout (float): training-only probability of blanking go_pred per sample
            (deterministic in (seed, idx), resume-safe) to force the protein/GO features to
            carry the GO signal instead of the prompt-copy shortcut. Default: 0.0 (off).
        go_pred_dropout_seed (int): seed for the per-sample dropout decision. Default: 0.
        exhaustive_target (bool): append the full GT GO-term list (go_bp/mf/cc) to the SFT
            target so the model learns breadth (CAFA F_max is ancestor-propagated). Default:
            False (curated reasoning+final_answer trace only). See Exp 1. LEAKS ground truth
            into the target (two independent Exp-1-style attempts BROKE generation: 21-45%
            empty outputs) — prefer append_gopred_target below.
        append_gopred_target (bool): append the GO terms already PRESENT IN THE PROMPT's
            go_pred field to the SFT target (verbatim, extracted via the same GO:\\d{7}
            regex the eval scorer uses), instead of the full GT list. NOT a leak — go_pred is
            already visible to the model in the prompt; this only teaches the model to
            reproduce that in-context breadth in its own OUTPUT (which the frozen-backbone
            model does not do on its own — a post-hoc union of model-output + go_pred
            recovered +0.067 F_max on held-out eval with ZERO training, see
            memory/project_bioreason_32b_capability_push_20260718.md). The curated
            reasoning+final_answer trace is kept BEFORE the appended list (coherence +
            reasoning-derived terms not in go_pred are preserved). Mutually exclusive with
            exhaustive_target (only one may be True). Default: False.
        bp_oversample_factor (float): duplicate examples whose ``go_bp`` column is
            non-empty this many times in ``self.examples`` (e.g. 2.0 = each BP-containing
            row appears twice, giving it ~2x its natural per-epoch sampling frequency).
            Targets a persistent weak point: across every BioReason checkpoint evaluated
            to date, the "biological_process" CAFA namespace scores lowest of the three
            (BP/CC/MF) and regresses first under overfitting (see
            docs/reports/bioreason_ablations_headline_findings_20260730.md, gap #8).
            Namespace membership is read directly from the existing ``go_bp`` column — no
            new data processing. Applied AFTER ``_filter_over_length`` (duplicates are
            plain dict references, so length/bucket computation and packing all see them
            as ordinary additional examples — no new sampler class needed). Default: 1.0
            (off; must be >= 1.0).
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
        drop_over_length: bool = True,
        go_pred_dropout: float = 0.0,
        go_pred_dropout_seed: int = 0,
        exhaustive_target: bool = False,
        append_gopred_target: bool = False,
        bp_oversample_factor: float = 1.0,
    ):
        if exhaustive_target and append_gopred_target:
            raise ValueError(
                "exhaustive_target and append_gopred_target are mutually exclusive "
                "(both append a term list to the target; pick one)."
            )
        if bp_oversample_factor < 1.0:
            raise ValueError(
                f"bp_oversample_factor must be >= 1.0 (1.0 = off); got "
                f"{bp_oversample_factor}."
            )
        self.bp_oversample_factor = float(bp_oversample_factor)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_protein_len = max_protein_len
        self.num_go_tokens = num_go_tokens
        self.protein_token_id = protein_token_id
        self.go_token_id = go_token_id
        self.train_on_reasoning = train_on_reasoning
        self.inject_go_pred = inject_go_pred
        self.drop_over_length = drop_over_length
        # Training-only leak reduction: with probability `go_pred_dropout`, blank the
        # go_pred "speculations" for a sample so the protein/GO features (not a prompt
        # copy) must carry the GO signal. The decision is a deterministic function of
        # (seed, idx) — NOT global RNG — so it is bit-reproducible across a
        # StatefulDataLoader resume. Eval never sets this (always injects go_pred).
        self.go_pred_dropout = float(go_pred_dropout)
        self.go_pred_dropout_seed = int(go_pred_dropout_seed)
        # Exhaustive-target mode (Exp 1): append the full GT GO-term list to the SFT target
        # so the model learns breadth (F_max is ancestor-propagated). Default off = the
        # curated reasoning+final_answer trace only (byte-identical to prior behavior).
        self.exhaustive_target = bool(exhaustive_target)
        # Approach B: append the IN-PROMPT go_pred terms (no GT leak) to the target so the
        # model learns to natively preserve go_pred's breadth on top of its own reasoning.
        self.append_gopred_target = bool(append_gopred_target)
        self.examples = self._load(data_files)
        logger.info(
            "Loaded %d BioReason SFT examples from %s", len(self.examples), data_files
        )
        if self.drop_over_length:
            self._filter_over_length()
        if self.bp_oversample_factor > 1.0:
            self._oversample_bp()

    def _oversample_bp(self) -> None:
        """Duplicate BP-containing rows to give them ~bp_oversample_factor x their
        natural per-epoch sampling frequency.

        Runs AFTER _filter_over_length so duplicated rows are guaranteed already
        length-valid. Duplicates are plain references to the same dict (examples are
        read-only downstream), so the length-grouped bucketing sampler and the packed-
        dataset path both see them as ordinary additional rows with no special-casing —
        this deliberately reuses the existing sampling machinery instead of adding a new
        sampler class. A fractional factor (e.g. 1.5) duplicates a deterministic prefix
        of the BP rows (by original index) so the run is reproducible across resumes.
        """
        bp_idxs = [i for i, ex in enumerate(self.examples) if _nonempty(ex.get("go_bp"))]
        if not bp_idxs:
            logger.warning(
                "bp_oversample_factor=%.2f set but no examples have a non-empty go_bp "
                "column — oversampling is a no-op.", self.bp_oversample_factor,
            )
            return
        full_copies = int(self.bp_oversample_factor) - 1  # -1: the row already counts once
        frac = self.bp_oversample_factor - int(self.bp_oversample_factor)
        n_frac = round(frac * len(bp_idxs))

        extra = []
        for _ in range(full_copies):
            extra.extend(self.examples[i] for i in bp_idxs)
        extra.extend(self.examples[i] for i in bp_idxs[:n_frac])

        logger.info(
            "bp_oversample_factor=%.2f: %d/%d examples have non-empty go_bp; added %d "
            "duplicate rows (total %d -> %d).",
            self.bp_oversample_factor, len(bp_idxs), len(self.examples), len(extra),
            len(self.examples), len(self.examples) + len(extra),
        )
        self.examples.extend(extra)

    def _filter_over_length(self) -> None:
        """Drop examples whose PROMPT alone exceeds max_seq_len.

        The XPU training attention uses the MATH SDPA backend (no flash backward on
        XPU), so the per-call fwd+bwd transient is O(S^2): seq=5120 -> ~19 GiB,
        seq=6144 -> ~27 GiB, which sets a hard per-tile seq ceiling. The prompt is
        never truncated (placeholder runs are load-bearing for the embed-splice), so
        an example whose prompt exceeds the budget cannot train under this max_seq_len.
        These are a <1% long tail (job 8572239 census: >5120 = 0.68%, >6144 = 0.11% at
        max_protein_len=2048; the prompt is text-dominated so lowering max_protein_len
        barely helps). Drop them rather than fail-fast on the whole run. See
        docs/reports/bioreason_sft_oom_diagnosis_20260627.md."""
        kept, dropped = [], 0
        for ex in self.examples:
            protein_seq = (ex.get("sequence", "") or "")[: self.max_protein_len]
            prompt_len = len(self._build_prompt_ids(ex, protein_seq))
            if prompt_len >= self.max_seq_len:
                dropped += 1
            else:
                kept.append(ex)
        if dropped:
            logger.warning(
                "Dropped %d/%d examples (%.2f%%) whose prompt >= max_seq_len=%d "
                "(prompt never truncated; the XPU math-SDPA O(S^2) seq ceiling). "
                "Set drop_over_length=False to fail-fast instead.",
                dropped, dropped + len(kept),
                100.0 * dropped / max(1, dropped + len(kept)), self.max_seq_len,
            )
        self.examples = kept

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

    def compute_lengths(self) -> list[int]:
        """Full token-stream length per example (prompt + target), capped at
        ``max_seq_len``. This is exactly the length ``__getitem__`` produces and the
        length the collate pads to, so a bucketed sampler keyed on it groups examples by
        their real per-step tensor shape. Computed once (tokenizer pass over the corpus);
        cache it on the caller — this is the length-grouped-sampler input.

        The cap matters: the target is right-truncated to ``max_seq_len - len(prompt)``
        in ``__getitem__``, so an example whose prompt+target exceeds ``max_seq_len``
        trains at exactly ``max_seq_len`` and must bucket there, not above."""
        lengths: list[int] = []
        for ex in self.examples:
            protein_seq = (ex.get("sequence", "") or "")[: self.max_protein_len]
            prompt_len = len(self._build_prompt_ids(ex, protein_seq))
            target_len = len(self._build_target_ids(ex))
            lengths.append(min(prompt_len + target_len, self.max_seq_len))
        return lengths

    def _build_prompt_text(
        self, ex: dict, interpro_in_prompt: bool = True, ppi_in_prompt: bool = True,
    ) -> str:
        """Paper-faithful go_pred prompt (system+user folded into one text block).
        Mirrors BioReasonRLDataset._build_go_pred_prompt_text exactly (pinned by test).

        interpro_in_prompt / ppi_in_prompt: eval-time-only text ablation (mirrors
        eval_cafa_fmax.py's flags of the same name, which only applied to the
        non-native _format_reasoning_prompt path — this checkpoint's native
        BioReasonSFTDataset._build_prompt_text had no equivalent knob until now,
        closing gap #6 in docs/reports/bioreason_ablations_headline_findings_20260730.md).
        Default True = unchanged prior behavior. There is no
        include_protein_function_summary equivalent here — this prompt format
        never includes a protein_function/UniProt-summary field at all, unlike the
        published paper's format_cafa5_for_protein_llm.
        """
        org = ex.get("organism", "") or "Unknown"
        interpro_data = (ex.get("interpro_formatted", "") or "") if interpro_in_prompt else ""
        ppi_data = (ex.get("ppi_formatted", "") or "") if ppi_in_prompt else ""
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
        text = self._build_prompt_text(
            ex,
            interpro_in_prompt=getattr(self, "interpro_in_prompt", True),
            ppi_in_prompt=getattr(self, "ppi_in_prompt", True),
        )

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

    @staticmethod
    def _gt_terms(ex: dict) -> list[str]:
        """Full ground-truth GO term list from go_bp/go_mf/go_cc (the columns the CAFA
        eval scores against). Deduped, stable order (MF, CC, BP — paper aspect order)."""
        seen: set[str] = set()
        out: list[str] = []
        for col in ("go_mf", "go_cc", "go_bp"):
            v = ex.get(col)
            if not _nonempty(v):  # handles numpy arrays (pandas to_dict) + lists + None
                continue
            # A single scalar string is one term; anything else iterable (list/tuple/ndarray)
            # is a collection of terms. Avoid `bool(array)` / `isinstance(list)` (arrays fail
            # both — the step-0 crash on the real dataset, 2026-07-02).
            items = [v] if isinstance(v, str) else list(v)
            for t in items:
                t = str(t).strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    _GO_ID_RE = re.compile(r"GO:\d{7}")

    @classmethod
    def _gopred_terms(cls, ex: dict) -> list[str]:
        """GO terms already present in the prompt's go_pred field (dedup, first-seen order).
        NOT ground truth — this is in-context text the model can already see; appending it
        to the target teaches native reproduction, it does not leak unseen information."""
        go_pred = ex.get("go_pred", "") or ""
        seen: set[str] = set()
        out: list[str] = []
        for t in cls._GO_ID_RE.findall(str(go_pred)):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _build_target_ids(self, ex: dict) -> list[int]:
        reasoning = ex.get("reasoning", "") or ""
        final = ex.get("final_answer", "") or ""
        if self.train_on_reasoning and reasoning:
            target = f"{reasoning}\n{final}"
        else:
            target = final
        # Exhaustive-target mode (Exp 1): append the FULL GT GO-term list after the trace so
        # the model learns BREADTH (CAFA F_max is ancestor-propagated -> rewards emitting all
        # true terms). The curated GPT-5 trace alone lists only ~7 terms; the eval extracts
        # GO:\d{7} over the whole response, so literal term strings here are scored. The
        # reasoning trace is kept (coherence + interpretability); the list is appended.
        if self.exhaustive_target:
            terms = self._gt_terms(ex)
            if terms:
                target = f"{target}\n\nGO terms: " + ", ".join(terms)
        # Approach B (append_gopred_target): append the go_pred terms ALREADY IN THE PROMPT
        # (not GT) so the model learns to natively preserve that breadth in its own output,
        # on top of the curated reasoning trace's own (partly novel) terms. See
        # memory/project_bioreason_32b_capability_push_20260718.md for the post-hoc-union
        # diagnostic (+0.067 F_max, zero training) that motivated this.
        elif self.append_gopred_target:
            terms = self._gopred_terms(ex)
            if terms:
                target = f"{target}\n\nGO terms: " + ", ".join(terms)
        ids = self.tokenizer.encode(target, add_bos=False, add_eos=True)
        return self._strip_bos(ids, self.tokenizer.bos_id)

    def _drop_go_pred(self, idx: int) -> bool:
        """Deterministic per-sample go_pred-dropout decision (resume-safe).

        A hash of (seed, idx) mapped to [0, 1) — no global RNG, so the same idx yields
        the same decision on every epoch and across a dataloader resume."""
        if self.go_pred_dropout <= 0.0:
            return False
        if self.go_pred_dropout >= 1.0:
            return True
        # Stable 64-bit mix of (seed, idx) -> uniform in [0, 1).
        h = (self.go_pred_dropout_seed * 0x9E3779B97F4A7C15 + idx * 0xBF58476D1CE4E5B9) & (
            (1 << 64) - 1
        )
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        h ^= h >> 27
        u = (h & ((1 << 53) - 1)) / float(1 << 53)
        return u < self.go_pred_dropout

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        protein_seq = (ex.get("sequence", "") or "")[: self.max_protein_len]
        go_aspect = ex.get("go_aspect", "all") or "all"

        # Leak reduction: blank go_pred for this sample (shallow copy — never mutate the
        # cached row) so the prompt-copy shortcut is unavailable and the protein/GO
        # features must carry the signal. Reuses the existing empty-go_pred prompt path.
        if self._drop_go_pred(idx):
            ex = {**ex, "go_pred": ""}

        prompt_ids = self._build_prompt_ids(ex, protein_seq)
        target_ids = self._build_target_ids(ex)

        # Budget: keep the FULL prompt (the placeholder runs are load-bearing for the
        # embed-splice and must never be cut) and truncate the TARGET from the right to
        # fit max_seq_len. Truncating the prompt would desync the placeholder count from
        # the ESM3/GO features. If the prompt alone exceeds the budget, the example can't
        # train under this max_seq_len — fail loudly with actionable guidance.
        room_for_target = self.max_seq_len - len(prompt_ids)
        if room_for_target <= 0:
            raise ValueError(
                f"Prompt ({len(prompt_ids)} tokens: {len(protein_seq) + 2} protein + "
                f"{self.num_go_tokens} GO + text) exceeds max_seq_len={self.max_seq_len}. "
                f"Raise max_seq_len or lower max_protein_len/num_go_tokens."
            )
        target_ids = target_ids[:room_for_target]

        tokens = prompt_ids + target_ids
        labels = [CROSS_ENTROPY_IGNORE_IDX] * len(prompt_ids) + list(target_ids)

        # Invariant: placeholder counts intact (prompt was never truncated).
        n_prot = sum(1 for t in tokens if t == self.protein_token_id)
        n_go = sum(1 for t in tokens if t == self.go_token_id)
        expected_prot = len(protein_seq) + 2
        assert n_prot == expected_prot and n_go == self.num_go_tokens, (
            f"placeholder count desync (protein {n_prot}/{expected_prot}, "
            f"GO {n_go}/{self.num_go_tokens}) — prompt assembly bug"
        )

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "protein_sequence": protein_seq,
            "go_aspect": go_aspect,
        }


class BioReasonPackedSFTDataset(Dataset):
    """Multimodal-aware token packing for BioReason SFT. Concatenates several short
    examples (from a base :class:`BioReasonSFTDataset`) into ONE fixed-``max_seq_len``
    pack so the packed step carries real tokens instead of ~43% padding — the throughput
    lever for the ~65% GEMM floor (see project_bioreason_sft_packing_scope_20260715).

    WHY a bespoke packer (not torchtune ``PackedDataset``): the stock packer only carries
    ``tokens``/``labels``; BioReason needs the per-document ``protein_sequence`` /
    ``go_aspect`` side-inputs carried through IN DOCUMENT ORDER so the model's
    ``_splice_embeds`` can fill the reserved placeholder ids left-to-right. The splice is
    ALREADY N-docs-per-row correct (``per_item[batch_idx_map[seq_idx]].append(...)`` in
    model_native.py) — it just needs the ordered side-input lists + a ``batch_idx_map`` that
    maps every doc in a pack to that pack's row. No model change required.

    Each pack yields (all fixed length ``max_seq_len``):
      - ``tokens`` / ``labels``: concatenated docs, right-padded (padding_idx / ignore_idx).
      - ``input_pos``: per-document position ids reset to 0 at each doc boundary (RoPE must
        see each doc starting at position 0, not a global offset — else doc 2 is encoded as
        if it were N tokens into doc 1). Pad positions get an arbitrary in-range id.
      - ``seq_lens``: list of per-doc lengths (incl. the trailing pad "doc"), the input to
        the block-diagonal (document) attention mask so doc A never attends doc B.
      - ``protein_sequences`` / ``go_aspects``: per-doc, IN PACK ORDER.
      - ``doc_batch_idx``: [0]*n_docs (all docs in this pack live in row 0 of a bs=1 pack);
        the collate offsets these per batch row.

    Fixed shape: every pack is exactly ``max_seq_len`` (banned:1-safe — one VA, like
    ``pad_to_fixed``). Greedy first-fit packing: a doc that would overflow the current pack
    starts a new one; a single doc longer than ``max_seq_len`` is impossible (the base
    dataset truncates the target to fit, and drops prompt-over-length examples).

    Packing is deterministic given the base dataset order (which is shuffled once per epoch
    by the sampler upstream). ``set_epoch`` re-packs against the new permutation.
    """

    def __init__(
        self,
        ds: "BioReasonSFTDataset",
        max_seq_len: int,
        padding_idx: int = 0,
        ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    ) -> None:
        self.ds = ds
        self.max_seq_len = int(max_seq_len)
        self.padding_idx = int(padding_idx)
        self.ignore_idx = int(ignore_idx)
        # Precompute per-example token lengths (capped) for the greedy pack plan. Reuses the
        # dataset's own length accounting so packs match exactly what __getitem__ produces.
        self._lengths = [min(l, self.max_seq_len) for l in ds.compute_lengths()]
        self._plan: list[list[int]] = self._build_plan(list(range(len(ds))))

    def _build_plan(self, order: list[int]) -> list[list[int]]:
        """Greedy first-fit: walk examples in ``order``, start a new pack whenever the next
        doc would overflow ``max_seq_len``. Returns a list of packs, each a list of example
        indices."""
        packs: list[list[int]] = []
        cur: list[int] = []
        cur_len = 0
        for idx in order:
            L = self._lengths[idx]
            if cur and cur_len + L > self.max_seq_len:
                packs.append(cur)
                cur = []
                cur_len = 0
            cur.append(idx)
            cur_len += L
        if cur:
            packs.append(cur)
        return packs

    def set_epoch_order(self, order: list[int]) -> None:
        """Re-pack against a new (shuffled) example order. Called per epoch so packing
        composition varies while staying deterministic within an epoch."""
        self._plan = self._build_plan(order)

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, pack_idx: int) -> dict:
        idxs = self._plan[pack_idx]
        toks: list[int] = []
        lbls: list[int] = []
        pos: list[int] = []
        seq_lens: list[int] = []
        prot: list[str] = []
        go: list[str] = []
        for j in idxs:
            item = self.ds[j]
            t = item["tokens"].tolist()
            l = item["labels"].tolist()
            n = len(t)
            # Truncate the LAST doc if the pack would exceed max_seq_len (rare; the plan
            # prevents it except for accumulated rounding). Keep placeholder runs intact by
            # only truncating when there is genuine overflow of the TARGET tail.
            room = self.max_seq_len - len(toks)
            if n > room:
                n = room
                t = t[:n]
                l = l[:n]
            toks.extend(t)
            lbls.extend(l)
            pos.extend(range(n))          # per-doc position reset
            seq_lens.append(n)
            prot.append(item["protein_sequence"])
            go.append(item["go_aspect"])
            if len(toks) >= self.max_seq_len:
                break
        # Right-pad to fixed length.
        pad = self.max_seq_len - len(toks)
        if pad > 0:
            toks.extend([self.padding_idx] * pad)
            lbls.extend([self.ignore_idx] * pad)
            pos.extend(range(pad))        # pad positions: any in-range ids (masked out)
            seq_lens.append(pad)          # trailing pad "document" for the block mask
        return {
            "tokens": torch.tensor(toks, dtype=torch.long),
            "labels": torch.tensor(lbls, dtype=torch.long),
            "input_pos": torch.tensor(pos, dtype=torch.long),
            "seq_lens": seq_lens,
            "protein_sequences": prot,
            "go_aspects": go,
        }


def bioreason_sft_packed_collate_fn(
    batch: list[dict],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    max_seq_len: Optional[int] = None,
    pad_to_fixed: bool = True,
    pad_buckets: Optional[list[int]] = None,
) -> dict:
    """Collate for :class:`BioReasonPackedSFTDataset` packs. Each element is ALREADY a
    fixed-length pack, so this just stacks rows and concatenates the ordered side-inputs,
    building ``batch_idx_map`` so every doc maps to its own batch row (the splice groups by
    it). ``pad_to_fixed`` / ``pad_buckets`` accepted for signature-compat with the recipe's
    partial but are no-ops here (packs are already fixed length)."""
    tok_out = torch.stack([ex["tokens"] for ex in batch], dim=0)
    lbl_out = torch.stack([ex["labels"] for ex in batch], dim=0)
    input_pos = torch.stack([ex["input_pos"] for ex in batch], dim=0)
    protein_sequences: list[str] = []
    go_aspects: list[str] = []
    batch_idx_map: list[int] = []
    seq_lens: list[list[int]] = []
    for row, ex in enumerate(batch):
        protein_sequences.extend(ex["protein_sequences"])
        go_aspects.extend(ex["go_aspects"])
        batch_idx_map.extend([row] * len(ex["protein_sequences"]))
        seq_lens.append(ex["seq_lens"])
    return {
        "tokens": tok_out,
        "labels": lbl_out,
        "input_pos": input_pos,
        "seq_lens": seq_lens,          # per-row list of doc lengths -> block-diagonal mask
        "protein_sequences": protein_sequences,
        "go_aspects": go_aspects,
        "batch_idx_map": batch_idx_map,
    }


class LengthGroupedDistributedBatchSampler(Sampler):
    """Distributed batch sampler that groups examples by length bucket and gives each
    bucket its own batch size — so short sequences train in bigger microbatches and the
    65 GiB FSDP weight-gather amortizes over more samples, WITHOUT dropping any of the
    corpus to a short max_seq_len.

    The throughput lever this session established: the 32B SFT step is FSDP-gather-bound,
    not compute-bound, and ``batch_size>1`` is blocked only by the O(S^2) attention
    transient. Short sequences have plenty of headroom for a bigger batch; long ones do
    not. This sampler realizes per-bucket batch sizing (e.g. 2048->bs4, 4096->bs2,
    6144->bs1) over the FULL corpus.

    FSDP correctness contract (why this is safe):
      * Every DP rank yields the **same number of batches** per epoch. FSDP2
        all-gather (fwd) / reduce-scatter (bwd) are collective over the shard group and
        fire once per microbatch; a rank with fewer microbatches would leave its peers
        hanging on a collective. Equal batch COUNT is guaranteed by construction (a
        bucket contributes ``floor(n_bucket / (bs_bucket * num_replicas))`` slots, same
        for all ranks).
      * At each batch slot ALL ranks draw from the SAME bucket, so their sequence lengths
        match — the per-microbatch reduce-scatter has no straggler. (Correctness would
        hold even if lengths differed; matching them just removes a stall.)
      * The training loop token-weights the loss (``loss * num_tokens``) and all-reduces
        ``num_tokens`` and ``running_loss`` before the optimizer step, so unequal
        samples-per-rank within a step is already numerically correct — no change needed.

    Bucketing key = the FULL token-stream length (prompt + target, capped at
    ``max_seq_len``), matching what the collate pads to. Pair with ``pad_buckets`` in the
    collate so each homogeneous-length batch pads to its own bucket (near-zero waste).

    Args:
        lengths (list[int]): per-example full token length (from
            :meth:`BioReasonSFTDataset.compute_lengths`).
        buckets (list[int]): ascending bucket ceilings, e.g. ``[2048, 4096, 6144]``. An
            example lands in the smallest bucket ``>=`` its length; the top bucket must be
            ``>=`` every length (append ``max_seq_len`` if unsure).
        bucket_batch_sizes (list[int]): batch size per bucket, same length/order as
            ``buckets``. Larger for shorter buckets.
        num_replicas (int): DP world size (``dp_degree``).
        rank (int): this process's DP rank (``dp_rank``).
        shuffle (bool): shuffle within-bucket assignment and slot order per epoch.
        seed (int): base RNG seed (combined with epoch via ``set_epoch``).
    """

    def __init__(
        self,
        lengths: list[int],
        buckets: list[int],
        bucket_batch_sizes: list[int],
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if len(buckets) != len(bucket_batch_sizes):
            raise ValueError(
                f"buckets ({len(buckets)}) and bucket_batch_sizes "
                f"({len(bucket_batch_sizes)}) must have equal length."
            )
        if list(buckets) != sorted(buckets):
            raise ValueError(f"buckets must be ascending; got {buckets}.")
        if any(bs < 1 for bs in bucket_batch_sizes):
            raise ValueError(f"bucket_batch_sizes must be >=1; got {bucket_batch_sizes}.")
        self.lengths = list(lengths)
        self.buckets = list(buckets)
        self.bucket_batch_sizes = list(bucket_batch_sizes)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        # Static assignment of every example to a bucket index (smallest ceiling >= len).
        # An example longer than the top bucket is a config error — compute_lengths caps
        # at max_seq_len, so set the top bucket >= max_seq_len.
        self._bucket_of: list[int] = []
        over = 0
        for L in self.lengths:
            bi = next((i for i, b in enumerate(self.buckets) if L <= b), None)
            if bi is None:
                bi = len(self.buckets) - 1  # clamp to top bucket
                over += 1
            self._bucket_of.append(bi)
        if over:
            logger.warning(
                "%d/%d examples exceed the top bucket %d and were clamped (they will be "
                "truncated to the bucket length by the collate). Set the top bucket >= "
                "the dataset max_seq_len to avoid this.",
                over, len(self.lengths), self.buckets[-1],
            )
        self._indices_by_bucket: list[list[int]] = [[] for _ in self.buckets]
        for idx, bi in enumerate(self._bucket_of):
            self._indices_by_bucket[bi].append(idx)

        # Number of batch slots per bucket is epoch-invariant (shuffle only reorders
        # within a bucket), so __len__ is stable.
        self._num_batches = sum(
            len(idxs) // (bs * self.num_replicas)
            for idxs, bs in zip(self._indices_by_bucket, self.bucket_batch_sizes)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def _build_slots(self) -> list[list[int]]:
        """Build this rank's ordered list of batches (each a list of example indices)."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        slots: list[list[int]] = []
        for bi, (idxs, bs) in enumerate(
            zip(self._indices_by_bucket, self.bucket_batch_sizes)
        ):
            idxs = list(idxs)
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[p] for p in perm]
            group = bs * self.num_replicas
            n_slots = len(idxs) // group  # drop_last within bucket
            for k in range(n_slots):
                base = k * group + self.rank * bs
                slots.append(idxs[base : base + bs])
        # Interleave buckets across the epoch (deterministic, same order on every rank so
        # slot i is the same bucket for all ranks -> matched seq length per microbatch).
        if self.shuffle and slots:
            order = torch.randperm(len(slots), generator=g).tolist()
            slots = [slots[o] for o in order]
        return slots

    def __iter__(self):
        yield from self._build_slots()


def bioreason_sft_collate_fn(
    batch: list[dict],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    max_seq_len: Optional[int] = None,
    pad_to_fixed: bool = False,
    pad_buckets: Optional[list[int]] = None,
) -> dict:
    """Pad tokens (padding_idx) and labels (ignore_idx) to a common length; attach the
    raw protein/go string lists (not padded). Mirrors padded_collate_sft for the
    token/label tensors while carrying the multimodal side inputs.

    pad_to_fixed (requires max_seq_len): pad EVERY batch to exactly max_seq_len so the
    per-step tensor shape is CONSTANT. On XPU this is the fix for the 32B SFT banned:1:
    isolation sweep (docs/reports/bioreason_sft_oom_diagnosis_20260627.md) showed
    fixed-shape seq=4096 trains 10/10 clean while variable LARGE shapes churn allocator
    VAs (no empty_cache under FSDP + OFI MR monitor) -> stale-VA write -> banned:1.
    Padding with padding_idx (0) does NOT change protein/GO placeholder counts (id 0 is
    neither), so the multimodal splice is unaffected.

    pad_buckets (overrides pad_to_fixed when set): pad each batch UP to the smallest
    bucket >= the batch's real max length, instead of always to max_seq_len. This keeps
    the set of per-step tensor shapes FINITE (the property that fixes banned:1 — a bounded
    number of distinct VAs the allocator reuses) while avoiding the ~71% padding waste of
    always padding a p50~1770 prompt to 6144. Buckets are clamped to <= max_seq_len and the
    top bucket must be >= max_seq_len so nothing overflows. e.g. [2048, 4096, 6144] on a
    p50~1770/p99~4618 corpus puts the median batch on the 2048 shape (~3x less compute)."""
    seqs = [ex["tokens"] for ex in batch]
    lbls = [ex["labels"] for ex in batch]
    if pad_buckets:
        if max_seq_len is None:
            raise ValueError("pad_buckets requires max_seq_len")
        buckets = sorted(b for b in pad_buckets if b <= max_seq_len)
        if not buckets or buckets[-1] < max_seq_len:
            buckets.append(max_seq_len)
        real_max = max(min(s.shape[0], max_seq_len) for s in seqs)
        # smallest bucket that holds the longest real sequence in this batch
        max_len = next((b for b in buckets if b >= real_max), max_seq_len)
    elif pad_to_fixed:
        if max_seq_len is None:
            raise ValueError("pad_to_fixed=True requires max_seq_len")
        max_len = max_seq_len
    else:
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
    drop_over_length: bool = True,
    go_pred_dropout: float = 0.0,
    go_pred_dropout_seed: int = 0,
    exhaustive_target: bool = False,
    append_gopred_target: bool = False,
    bp_oversample_factor: float = 1.0,
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
        drop_over_length=drop_over_length,
        go_pred_dropout=go_pred_dropout,
        go_pred_dropout_seed=go_pred_dropout_seed,
        exhaustive_target=exhaustive_target,
        append_gopred_target=append_gopred_target,
        bp_oversample_factor=bp_oversample_factor,
    )

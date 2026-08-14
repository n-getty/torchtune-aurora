# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression guard: BioReason SFT labels must be SHIFTED relative to tokens.

Before 2026-08-05 ``BioReasonSFTDataset.__getitem__`` built
``labels[i] == tokens[i]`` (unshifted). Under causal attention position ``i`` has
already attended to token ``i``, so that objective is "copy the token you can already
see" — an identity map the trainable projector solves on its own. Observed effect:
stage1norm training loss fell 16.54 -> 0.00275 in 110 steps with a FROZEN backbone,
which was misread as convergence. Every 32B SFT checkpoint was trained this way.

These tests pin the shift at the dataset boundary and, separately, demonstrate the
learning-dynamics consequence so the "why" survives even if the dataset is rewritten.

CPU-only, no XPU, no distributed — safe on a login node.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset

IGN = CROSS_ENTROPY_IGNORE_IDX


class _StubTok:
    """Minimal tokenizer: 1 id per whitespace token, deterministic, no specials."""

    def encode(self, text, add_bos=False, add_eos=False, **kwargs):
        ids = [(abs(hash(w)) % 1000) + 10 for w in str(text).split()]
        if add_bos:
            ids = [1] + ids
        if add_eos:
            ids = ids + [2]
        return ids


def _make_ds(prompt_len=12, target_len=5, max_seq_len=64):
    """Build a dataset instance without touching disk, with stubbed prompt/target ids."""
    ds = BioReasonSFTDataset.__new__(BioReasonSFTDataset)
    ds.tokenizer = _StubTok()
    ds.max_seq_len = max_seq_len
    ds.max_protein_len = 8
    ds.num_go_tokens = 2
    ds.protein_token_id = 900
    ds.go_token_id = 901
    ds.train_on_reasoning = True
    ds.inject_go_pred = True
    ds.go_pred_dropout = 0.0
    ds.go_pred_dropout_seed = 0
    ds.exhaustive_target = False
    ds.append_gopred_target = False
    ds.interpro_in_prompt = True
    ds.ppi_in_prompt = True

    seq = "AAAA"  # -> len(seq)+2 == 6 protein placeholders
    ds.examples = [{"sequence": seq, "go_aspect": "all"}]

    # Stub prompt/target assembly: a real prompt ending in text, with the exact
    # placeholder counts __getitem__ asserts on.
    n_text = prompt_len - (len(seq) + 2) - ds.num_go_tokens
    assert n_text >= 1, "prompt_len too small for placeholder runs"
    prompt = (
        [500] * n_text
        + [ds.protein_token_id] * (len(seq) + 2)
        + [ds.go_token_id] * ds.num_go_tokens
    )
    target = [700 + i for i in range(target_len)]
    ds._build_prompt_ids = lambda ex, s: list(prompt)
    ds._build_target_ids = lambda ex, room=None: list(target)
    return ds, prompt, target


def test_labels_are_shifted_one_position_left():
    """labels[i] must equal tokens[i+1] over the supervised span."""
    ds, prompt, target = _make_ds()
    item = ds[0]
    tokens = item["tokens"].tolist()
    labels = item["labels"].tolist()

    assert len(tokens) == len(labels)

    supervised = [i for i, l in enumerate(labels) if l != IGN]
    assert supervised, "no supervised positions — masking is broken"

    for i in supervised:
        assert i + 1 < len(tokens), f"supervised position {i} has no successor token"
        assert labels[i] == tokens[i + 1], (
            f"label at {i} is {labels[i]}, expected tokens[{i+1}]={tokens[i+1]}. "
            "Labels are UNSHIFTED — this is the copy-the-input bug."
        )


def test_labels_are_not_unshifted_copy():
    """Explicitly reject the old behavior: labels[i] == tokens[i] over the target."""
    ds, prompt, target = _make_ds()
    item = ds[0]
    tokens = item["tokens"].tolist()
    labels = item["labels"].tolist()

    identity_matches = sum(
        1 for i, l in enumerate(labels) if l != IGN and l == tokens[i]
    )
    assert identity_matches == 0, (
        f"{identity_matches} supervised labels equal their own token — the model would "
        "be trained to copy the input it can already see."
    )


def test_first_supervised_position_is_last_prompt_token():
    """The last prompt token must be the position that predicts target[0]."""
    ds, prompt, target = _make_ds()
    item = ds[0]
    labels = item["labels"].tolist()

    first = next(i for i, l in enumerate(labels) if l != IGN)
    assert first == len(prompt) - 1, (
        f"first supervised position is {first}, expected {len(prompt)-1} "
        "(the last prompt token predicts the first target token)"
    )
    assert labels[first] == target[0]


def test_final_position_is_ignored():
    """The last position has no successor, so it must not be supervised."""
    ds, _, _ = _make_ds()
    labels = ds[0]["labels"].tolist()
    assert labels[-1] == IGN


def test_prompt_interior_is_masked():
    """Everything before the last prompt token stays IGNORE."""
    ds, prompt, _ = _make_ds()
    labels = ds[0]["labels"].tolist()
    assert all(l == IGN for l in labels[: len(prompt) - 1])


def test_supervised_token_count_equals_target_length():
    ds, _, target = _make_ds()
    labels = ds[0]["labels"].tolist()
    assert sum(1 for l in labels if l != IGN) == len(target)


@pytest.mark.parametrize("prompt_len,target_len", [(9, 1), (12, 3), (20, 12)])
def test_shift_holds_across_shapes(prompt_len, target_len):
    ds, prompt, target = _make_ds(prompt_len=prompt_len, target_len=target_len)
    item = ds[0]
    tokens, labels = item["tokens"].tolist(), item["labels"].tolist()
    assert len(tokens) == len(labels)
    for i, l in enumerate(labels):
        if l != IGN:
            assert l == tokens[i + 1]


def test_unshifted_labels_collapse_loss_with_frozen_backbone():
    """Mechanistic proof of WHY the shift matters — the bug's signature was near-zero loss.

    Frozen causal backbone + a single trainable head (mirrors stage1: frozen backbone,
    trainable projector). With unshifted labels the task is solvable to ~0 loss; with
    correctly shifted labels it is not. This is what produced the 16.54 -> 0.00275
    stage1norm trace that was misread as convergence.
    """
    torch.manual_seed(0)
    V, D, S, B = 64, 32, 24, 16

    emb = nn.Embedding(V, D)
    attn = nn.MultiheadAttention(D, 4, batch_first=True)
    for p in list(emb.parameters()) + list(attn.parameters()):
        p.requires_grad = False

    head = nn.Linear(D, V)
    toks = torch.randint(0, V, (B, S))
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

    def hidden(t):
        x = emb(t)
        out, _ = attn(x, x, x, attn_mask=causal, need_weights=False)
        return out + x

    def train(shifted, steps=400):
        torch.manual_seed(1)
        head.reset_parameters()
        opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
        loss = None
        for _ in range(steps):
            logits = head(hidden(toks))
            if shifted:
                lg, lb = logits[:, :-1], toks[:, 1:]
            else:
                lg, lb = logits, toks
            loss = F.cross_entropy(lg.reshape(-1, V), lb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    unshifted_loss = train(shifted=False)
    shifted_loss = train(shifted=True)

    assert unshifted_loss < 0.1, (
        f"expected the unshifted (copy) task to collapse toward 0, got {unshifted_loss:.4f}"
    )
    assert shifted_loss > 1.0, (
        f"expected the real next-token task to stay high-loss, got {shifted_loss:.4f}"
    )
    assert shifted_loss > 10 * unshifted_loss

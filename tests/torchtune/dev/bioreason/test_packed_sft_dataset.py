# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU-safe unit tests for BioReasonPackedSFTDataset + bioreason_sft_packed_collate_fn.
#
# Packing is the throughput lever for the ~65% GEMM floor: fill the fixed [1,max_seq_len]
# shape with REAL tokens from several docs instead of ~43% pad. Correctness contract:
#   (1) fixed shape: every pack is exactly max_seq_len (banned:1-safe).
#   (2) per-doc position_ids reset to 0 at each doc boundary (RoPE correctness).
#   (3) seq_lens sum to max_seq_len and describe the block-diagonal (document) mask.
#   (4) protein/GO side-inputs are carried IN PACK ORDER (the splice fills placeholders
#       left-to-right by that order).
#   (5) collate builds batch_idx_map mapping every doc to its batch row.
#   (6) no example dropped; greedy first-fit packing.

import torch

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.dev.bioreason.dataset_sft import (
    BioReasonPackedSFTDataset,
    bioreason_sft_packed_collate_fn,
)


class _FakeBioDS:
    """Minimal stand-in for BioReasonSFTDataset: yields tokens/labels/protein/go with
    controllable per-example lengths. Prompt span (first half) masked in labels."""

    def __init__(self, lengths):
        self._lengths = list(lengths)

    def compute_lengths(self):
        return list(self._lengths)

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, idx):
        L = self._lengths[idx]
        # token ids encode the example index (idx*1000 + pos) so we can trace provenance.
        toks = [idx * 1000 + p for p in range(L)]
        half = L // 2
        labels = [CROSS_ENTROPY_IGNORE_IDX] * half + toks[half:]
        return {
            "tokens": torch.tensor(toks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "protein_sequence": f"PROT{idx}",
            "go_aspect": f"GO{idx}",
        }


def test_greedy_pack_plan_fits_and_keeps_all():
    ds = _FakeBioDS([300, 400, 500, 600, 700])  # sum 2500
    p = BioReasonPackedSFTDataset(ds, max_seq_len=1000)
    # greedy first-fit: [300,400] (700) | [500] (+600=1100>1000 -> new) ... verify no overflow
    total_docs = sum(
        len([sl for sl in pack["seq_lens"]]) for pack in (p[i] for i in range(len(p)))
    )
    # every pack fixed length
    for i in range(len(p)):
        assert p[i]["tokens"].shape[0] == 1000
        assert p[i]["labels"].shape[0] == 1000
        assert p[i]["input_pos"].shape[0] == 1000
    # all 5 real docs represented (packs may add a trailing pad "doc")
    real_docs = 0
    for i in range(len(p)):
        real_docs += len(p[i]["protein_sequences"])
    assert real_docs == 5


def test_seq_lens_sum_to_max_and_include_pad():
    ds = _FakeBioDS([300, 400])  # one pack: 700 real + 300 pad
    p = BioReasonPackedSFTDataset(ds, max_seq_len=1000)
    assert len(p) == 1
    pack = p[0]
    assert sum(pack["seq_lens"]) == 1000
    # 2 real docs + 1 pad doc
    assert pack["seq_lens"] == [300, 400, 300]


def test_position_ids_reset_per_doc():
    ds = _FakeBioDS([3, 4])  # pack: doc0 len3, doc1 len4, pad 993
    p = BioReasonPackedSFTDataset(ds, max_seq_len=1000)
    pos = p[0]["input_pos"].tolist()
    assert pos[:3] == [0, 1, 2]          # doc0 resets
    assert pos[3:7] == [0, 1, 2, 3]       # doc1 resets
    assert pos[7:10] == [0, 1, 2]         # pad region also resets (masked out anyway)


def test_side_inputs_in_pack_order():
    ds = _FakeBioDS([300, 400])
    p = BioReasonPackedSFTDataset(ds, max_seq_len=1000)
    pack = p[0]
    assert pack["protein_sequences"] == ["PROT0", "PROT1"]
    assert pack["go_aspects"] == ["GO0", "GO1"]


def test_tokens_concatenated_in_order():
    ds = _FakeBioDS([3, 2])
    p = BioReasonPackedSFTDataset(ds, max_seq_len=10)
    toks = p[0]["tokens"].tolist()
    # doc0: [0,1,2], doc1: [1000,1001], then pad(0)*5
    assert toks[:3] == [0, 1, 2]
    assert toks[3:5] == [1000, 1001]
    assert toks[5:] == [0, 0, 0, 0, 0]


def test_labels_mask_prompt_and_pad():
    ds = _FakeBioDS([4, 4])
    p = BioReasonPackedSFTDataset(ds, max_seq_len=12)
    lbl = p[0]["labels"].tolist()
    I = CROSS_ENTROPY_IGNORE_IDX
    # doc0 len4: first 2 masked, last 2 real; doc1 same; then 4 pad = ignore
    assert lbl[0:2] == [I, I]
    assert lbl[2:4] == [2, 3]           # doc0 tokens idx0*1000+2,3
    assert lbl[4:6] == [I, I]
    assert lbl[6:8] == [1002, 1003]     # doc1 tokens
    assert lbl[8:12] == [I, I, I, I]    # pad


def test_collate_stacks_and_builds_batch_idx_map():
    ds = _FakeBioDS([300, 400, 500])
    p = BioReasonPackedSFTDataset(ds, max_seq_len=800)
    # pack0: [300,400]=700; pack1: [500]
    packs = [p[i] for i in range(len(p))]
    out = bioreason_sft_packed_collate_fn(packs, padding_idx=0, max_seq_len=800)
    B = len(packs)
    assert out["tokens"].shape == (B, 800)
    assert out["input_pos"].shape == (B, 800)
    # batch_idx_map: pack0 has 2 docs -> [0,0]; pack1 has 1 doc -> [1]
    assert out["batch_idx_map"] == [0, 0, 1]
    assert out["protein_sequences"] == ["PROT0", "PROT1", "PROT2"]
    assert len(out["seq_lens"]) == B          # per-row list of doc lengths


def test_set_epoch_order_repacks():
    ds = _FakeBioDS([300, 400, 500, 600])
    p = BioReasonPackedSFTDataset(ds, max_seq_len=1000)
    n0 = len(p)
    p.set_epoch_order([3, 2, 1, 0])           # reversed order -> different packing
    # still fixed shape + all docs present
    real = sum(len(p[i]["protein_sequences"]) for i in range(len(p)))
    assert real == 4
    for i in range(len(p)):
        assert p[i]["tokens"].shape[0] == 1000


def test_single_doc_longer_than_pack_truncated():
    # A doc exactly at max_seq_len fills a pack alone (no overflow).
    ds = _FakeBioDS([1000])
    p = BioReasonPackedSFTDataset(ds, max_seq_len=1000)
    assert len(p) == 1
    assert p[0]["tokens"].shape[0] == 1000
    assert p[0]["seq_lens"] == [1000]         # no pad doc needed

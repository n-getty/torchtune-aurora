# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU drift-guard for torchtune.dev.rl.packing.

Sequence packing folds many short sequences into a few dense packs with
block-diagonal attention masks. A bug here is the worst kind: it does not
crash, it silently corrupts training — e.g. a mask that lets one packed
sequence attend to another, or an unpack that places hidden states at the
wrong offset. The production base recipe uses this when ``enable_packing``
is set, so it is on a real path.

These tests pin the load-bearing invariants:
  * round-trip identity: pack then unpack reproduces the per-sequence values
    at every non-padding position;
  * block-diagonal masks never allow cross-sequence attention;
  * actual-length computation ignores right padding;
  * every input sequence is placed exactly once.

Pure CPU torch — no XPU, no distributed init. Verified against the real
``greedy_bin_pack`` contract (oversized sequences are placed ALONE and may
exceed ``pack_capacity`` — the test does not assert no-overflow because the
function deliberately allows it for a single too-long sequence).
"""
from __future__ import annotations

import torch

from torchtune.dev.rl.packing import (
    compute_actual_seq_lens,
    greedy_bin_pack,
    pack_trajectory_for_training,
    unpack_tensor,
)


# --- compute_actual_seq_lens ---

def test_actual_seq_lens_ignores_right_pad():
    pad_id = 0
    qr = torch.tensor([[1, 2, 3, 0, 0], [1, 0, 0, 0, 0], [1, 2, 0, 0, 0]])
    assert compute_actual_seq_lens(qr, pad_id).tolist() == [3, 1, 2]


def test_actual_seq_lens_all_pad_is_zero():
    qr = torch.zeros(2, 5, dtype=torch.long)
    assert compute_actual_seq_lens(qr, pad_id=0).tolist() == [0, 0]


def test_actual_seq_lens_full_row():
    qr = torch.tensor([[1, 2, 3, 4, 5]])
    assert compute_actual_seq_lens(qr, pad_id=0).tolist() == [5]


# --- greedy_bin_pack ---

def test_bin_pack_places_every_sequence_once():
    seq_lens = torch.tensor([10, 20, 30, 40, 50])
    bins = greedy_bin_pack(seq_lens, pack_capacity=100)
    placed = sorted(idx for b in bins for idx in b)
    assert placed == list(range(len(seq_lens)))


def test_bin_pack_respects_capacity_for_fitting_seqs():
    # All sequences fit individually; no bin of fitting seqs may exceed capacity.
    seq_lens = torch.tensor([100, 200, 150, 50, 300, 75])
    cap = 350
    bins = greedy_bin_pack(seq_lens, cap)
    for b in bins:
        total = sum(int(seq_lens[i]) for i in b)
        # Every member fits (<= cap), so the bin total must too.
        assert total <= cap


def test_bin_pack_zero_len_excluded():
    seq_lens = torch.tensor([0, 10, 0, 20])
    bins = greedy_bin_pack(seq_lens, pack_capacity=50)
    for b in bins:
        for idx in b:
            assert int(seq_lens[idx]) > 0


def test_bin_pack_oversized_placed_alone():
    # A sequence longer than capacity is placed in its own bin (documented
    # behavior — the function logs a warning and does not split it).
    seq_lens = torch.tensor([500, 10, 20])
    bins = greedy_bin_pack(seq_lens, pack_capacity=100)
    big_bin = [b for b in bins if 0 in b][0]
    assert big_bin == [0]


# --- pack / unpack round-trip ---

def _make_tokens(lengths, total_len, pad_id=0):
    n = len(lengths)
    tok = torch.full((n, total_len), pad_id, dtype=torch.long)
    for i, ln in enumerate(lengths):
        tok[i, :ln] = torch.arange(1, ln + 1)
    return tok


def test_pack_unpack_roundtrip_identity():
    lengths = [12, 8, 14, 6]
    total_len = 16
    pad_id = 0
    tokens = _make_tokens(lengths, total_len, pad_id)
    pos_ids = torch.arange(total_len).unsqueeze(0).expand(len(lengths), -1).contiguous()

    packed_tok, packed_pos, packed_mask, bins, actual_lens = (
        pack_trajectory_for_training(tokens, pos_ids, pad_id=pad_id)
    )
    assert actual_lens.tolist() == lengths

    # Simulate per-position model hidden states, unpack, compare non-pad slices.
    num_packs, pack_seq_len = packed_tok.shape
    H = 4
    fake_hidden = torch.randn(num_packs, pack_seq_len, H)
    unpacked = unpack_tensor(fake_hidden, bins, actual_lens, len(lengths), total_len)
    assert unpacked.shape == (len(lengths), total_len, H)

    for pack_idx, bin_indices in enumerate(bins):
        offset = 0
        for seq_idx in bin_indices:
            sl = int(actual_lens[seq_idx])
            torch.testing.assert_close(
                unpacked[seq_idx, :sl],
                fake_hidden[pack_idx, offset:offset + sl],
            )
            offset += sl


def test_packed_tokens_match_source():
    lengths = [5, 3]
    total_len = 8
    pad_id = 0
    tokens = _make_tokens(lengths, total_len, pad_id)
    pos_ids = torch.arange(total_len).unsqueeze(0).expand(len(lengths), -1).contiguous()
    packed_tok, _, _, bins, actual_lens = pack_trajectory_for_training(
        tokens, pos_ids, pad_id=pad_id
    )
    # Reconstruct each source row from the packed layout and compare.
    for pack_idx, bin_indices in enumerate(bins):
        offset = 0
        for seq_idx in bin_indices:
            sl = int(actual_lens[seq_idx])
            assert packed_tok[pack_idx, offset:offset + sl].tolist() == \
                tokens[seq_idx, :sl].tolist()
            offset += sl


def test_block_diagonal_mask_no_cross_sequence_attention():
    # Force >1 sequence into a single pack, then assert the mask zeros all
    # cross-sequence blocks (a leak here silently corrupts the forward pass).
    lengths = [4, 6, 8]
    total_len = 12
    pad_id = 0
    tokens = _make_tokens(lengths, total_len, pad_id)
    pos_ids = torch.arange(total_len).unsqueeze(0).expand(len(lengths), -1).contiguous()
    _, _, masks, bins, actual_lens = pack_trajectory_for_training(
        tokens, pos_ids, pad_id=pad_id
    )
    for pack_idx, bin_indices in enumerate(bins):
        if len(bin_indices) < 2:
            continue
        boundaries = []
        offset = 0
        for seq_idx in bin_indices:
            sl = int(actual_lens[seq_idx])
            boundaries.append((offset, offset + sl))
            offset += sl
        mask = masks[pack_idx]
        for i, (r0, r1) in enumerate(boundaries):
            for j, (c0, c1) in enumerate(boundaries):
                if i == j:
                    continue
                cross = mask[r0:r1, c0:c1]
                assert int(cross.sum()) == 0, (
                    f"pack {pack_idx}: seq {i} attends to seq {j}"
                )


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

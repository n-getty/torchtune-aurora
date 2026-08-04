# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU pin-down for SyntheticFixedLengthDataset — the isoFLOPs-style benchmark
dataset used by the seq4096 MoE-vs-dense MFU comparison (see
memory/project_moe_sft_profiling_gradrelease_bottleneck_20260724.md).

The load-bearing claim this file pins: every sample is EXACTLY `seq_len`
tokens, so running a batch through the standard (non-packed) SFT collate
(`padded_collate_sft`) is a true no-op on padding — no padding is ever added,
and the collate never constructs a `"mask"` key — meaning `mask=None` reaches
`TransformerDecoder.forward()` by construction, with zero new masking code.
This is what lets native XPU flash attention engage on this dataset without
any `TORCHTUNE_MASKFREE_CAUSAL`-style bypass flag.
"""
from unittest.mock import MagicMock

import pytest
import torch

from torchtune.data._collate import padded_collate_sft
from torchtune.datasets import SyntheticFixedLengthDataset, synthetic_fixed_length_dataset


def _mock_tokenizer(vocab_size: int = 32000, pad_id: int = 0):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.pad_id = pad_id
    return tok


class _EncoderOnlyTokenizer:
    """Mimics Qwen3Tokenizer/the Qwen2-family BPE tokenizers, which expose
    only `encoder` (a dict), not `vocab_size` directly — the exact shape
    that caused an AttributeError on real HW before _infer_vocab_size was
    added."""

    def __init__(self, vocab_size: int, pad_id: int = 0):
        self.encoder = {f"tok_{i}": i for i in range(vocab_size)}
        self.pad_id = pad_id


class TestSyntheticFixedLengthDataset:
    @pytest.mark.parametrize("seq_len", [128, 1536, 4096])
    def test_every_sample_is_exactly_seq_len(self, seq_len):
        ds = SyntheticFixedLengthDataset(_mock_tokenizer(), seq_len=seq_len, num_samples=5)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["tokens"].shape == (seq_len,)
            assert sample["labels"].shape == (seq_len,)

    def test_labels_mirror_tokens(self):
        ds = SyntheticFixedLengthDataset(_mock_tokenizer(), seq_len=64, num_samples=3)
        for i in range(len(ds)):
            sample = ds[i]
            torch.testing.assert_close(sample["tokens"], sample["labels"])
            # labels must be a distinct tensor (clone), not an aliasing view —
            # a collate or loss-fn mutation on one must not silently affect
            # the other.
            assert sample["tokens"].data_ptr() != sample["labels"].data_ptr()

    def test_vocab_size_inferred_from_tokenizer_when_not_passed(self):
        tok = _mock_tokenizer(vocab_size=100)
        ds = SyntheticFixedLengthDataset(tok, seq_len=32, num_samples=1)
        sample = ds[0]
        assert sample["tokens"].max().item() < 100
        assert sample["tokens"].min().item() >= 0

    def test_explicit_vocab_size_overrides_tokenizer(self):
        tok = _mock_tokenizer(vocab_size=100000)
        ds = SyntheticFixedLengthDataset(tok, seq_len=32, vocab_size=50, num_samples=1)
        sample = ds[0]
        assert sample["tokens"].max().item() < 50

    def test_vocab_size_falls_back_to_len_encoder(self):
        """Qwen3Tokenizer/the Qwen2-family BPE tokenizers have no `vocab_size`
        attribute at all — only `encoder` (a dict). This is the exact
        AttributeError this fallback fixes (HW-caught on the real recipe)."""
        tok = _EncoderOnlyTokenizer(vocab_size=777)
        assert not hasattr(tok, "vocab_size")
        ds = SyntheticFixedLengthDataset(tok, seq_len=32, num_samples=3)
        for i in range(3):
            sample = ds[i]
            assert sample["tokens"].max().item() < 777
            assert sample["tokens"].min().item() >= 0

    def test_vocab_size_inference_raises_clear_error_when_neither_available(self):
        class _NoVocabInfoTokenizer:
            pass

        with pytest.raises(AttributeError, match="Could not infer vocab_size"):
            SyntheticFixedLengthDataset(_NoVocabInfoTokenizer(), seq_len=32, num_samples=1)

    def test_deterministic_across_construction_with_same_seed(self):
        tok = _mock_tokenizer()
        ds_a = SyntheticFixedLengthDataset(tok, seq_len=64, num_samples=5, seed=42)
        ds_b = SyntheticFixedLengthDataset(tok, seq_len=64, num_samples=5, seed=42)
        for i in range(5):
            torch.testing.assert_close(ds_a[i]["tokens"], ds_b[i]["tokens"])

    def test_different_seeds_produce_different_data(self):
        tok = _mock_tokenizer()
        ds_a = SyntheticFixedLengthDataset(tok, seq_len=64, num_samples=1, seed=0)
        ds_b = SyntheticFixedLengthDataset(tok, seq_len=64, num_samples=1, seed=1)
        assert not torch.equal(ds_a[0]["tokens"], ds_b[0]["tokens"])

    def test_different_indices_produce_different_samples(self):
        ds = SyntheticFixedLengthDataset(_mock_tokenizer(), seq_len=64, num_samples=2, seed=0)
        assert not torch.equal(ds[0]["tokens"], ds[1]["tokens"])

    def test_len_matches_num_samples(self):
        ds = SyntheticFixedLengthDataset(_mock_tokenizer(), seq_len=32, num_samples=17)
        assert len(ds) == 17

    def test_builder_function_matches_class(self):
        tok = _mock_tokenizer()
        ds = synthetic_fixed_length_dataset(tok, seq_len=32, num_samples=1, seed=0)
        assert isinstance(ds, SyntheticFixedLengthDataset)
        assert ds[0]["tokens"].shape == (32,)


class TestSyntheticFixedLengthDatasetCollateNoOp:
    """Pins the load-bearing claim: fixed-length samples make padded_collate_sft
    a true no-op, so mask=None reaches attention with zero new masking code."""

    @pytest.mark.parametrize("seq_len", [128, 1536, 4096])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_collate_output_shape_equals_seq_len_no_padding(self, seq_len, batch_size):
        tok = _mock_tokenizer(pad_id=999)  # distinctive padding_idx to detect any real padding
        ds = SyntheticFixedLengthDataset(tok, seq_len=seq_len, num_samples=batch_size)
        batch = [ds[i] for i in range(batch_size)]
        collated = padded_collate_sft(
            [{"tokens": s["tokens"].tolist(), "labels": s["labels"].tolist()} for s in batch],
            padding_idx=tok.pad_id,
            ignore_idx=-100,
        )
        # Every sample was already seq_len long, so pad_sequence (which pads
        # to the batch's own max) must be a true no-op: output shape equals
        # seq_len exactly, not seq_len + any extra padding.
        assert collated["tokens"].shape == (batch_size, seq_len)
        assert collated["labels"].shape == (batch_size, seq_len)
        # No padding_idx value should appear anywhere in the collated tokens
        # — since the synthetic generator itself never emits it (vocab_size
        # is well below pad_id=999) and no padding was added.
        assert not (collated["tokens"] == tok.pad_id).any()

    def test_collate_output_contains_no_mask_key(self):
        tok = _mock_tokenizer()
        ds = SyntheticFixedLengthDataset(tok, seq_len=64, num_samples=2)
        batch = [ds[i] for i in range(2)]
        collated = padded_collate_sft(
            [{"tokens": s["tokens"].tolist(), "labels": s["labels"].tolist()} for s in batch],
            padding_idx=tok.pad_id,
            ignore_idx=-100,
        )
        # padded_collate_sft never constructs a "mask" key for plain (non-cp,
        # non-encoder-input) batches — this is what lets mask=None reach
        # TransformerDecoder.forward() by construction, not a special-cased
        # bypass flag.
        assert "mask" not in collated

    @pytest.mark.parametrize("cp_degree", [1, 2, 4])
    def test_collate_never_constructs_mask_key_even_with_cp_degree(self, cp_degree):
        """The mask=None claim must hold regardless of tensor/context
        parallelism config (codex-review flagged this as untested — the only
        thing cp_degree>1 adds to the collated batch is `input_pos`, never
        a `mask` key; verify directly rather than assume)."""
        tok = _mock_tokenizer()
        ds = SyntheticFixedLengthDataset(tok, seq_len=64, num_samples=2)
        batch = [ds[i] for i in range(2)]
        collated = padded_collate_sft(
            [{"tokens": s["tokens"].tolist(), "labels": s["labels"].tolist()} for s in batch],
            padding_idx=tok.pad_id,
            ignore_idx=-100,
            cp_degree=cp_degree,
        )
        assert "mask" not in collated
        if cp_degree > 1:
            assert "input_pos" in collated

    def test_pad_to_multiple_of_gt_1_DOES_add_padding_narrower_claim(self):
        """CORRECTED SCOPE (codex-review finding): the dataset's "no padding
        is ever added" claim only holds when pad_to_multiple_of == 1
        (ParallelDims.min_seq_len_divisor returns 1 unless tp>1 or cp>1 —
        true for every config this dataset ships with today, but NOT an
        unconditional property of the dataset itself). This test pins the
        actual (weaker) boundary: at pad_to_multiple_of>1, padded_collate_sft
        WILL pad an already-multiple-of-N sequence by a further full block
        (see torchtune/data/_collate.py's `pad_to_multiple_of -
        (seq_len % pad_to_multiple_of)` — evaluates to `pad_to_multiple_of`
        itself, not 0, when seq_len is already a multiple). This does NOT
        affect the mask=None claim (independent of padding), but it DOES
        mean tp>1/cp>1 configs using this dataset see more real tokens/step
        than tokenizer.max_seq_len alone would suggest."""
        tok = _mock_tokenizer(pad_id=999)
        seq_len = 64  # already a multiple of 8
        ds = SyntheticFixedLengthDataset(tok, seq_len=seq_len, num_samples=2)
        batch = [ds[i] for i in range(2)]
        collated = padded_collate_sft(
            [{"tokens": s["tokens"].tolist(), "labels": s["labels"].tolist()} for s in batch],
            padding_idx=tok.pad_id,
            ignore_idx=-100,
            pad_to_multiple_of=8,
        )
        assert collated["tokens"].shape == (2, seq_len + 8)
        assert (collated["tokens"][:, seq_len:] == tok.pad_id).all()

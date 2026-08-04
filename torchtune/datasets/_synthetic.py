# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.utils.data import Dataset

from torchtune.modules.transforms.tokenizers import ModelTokenizer


def _infer_vocab_size(tokenizer: ModelTokenizer) -> int:
    """Best-effort ``vocab_size`` lookup across tokenizer implementations.

    Not every ``ModelTokenizer`` exposes ``vocab_size`` directly (e.g.
    ``Qwen3Tokenizer``/the Qwen2-family BPE tokenizers only expose
    ``encoder``, a dict of token string -> id). Falls back to
    ``len(tokenizer.encoder)`` when available, otherwise raises with a clear
    message pointing at the explicit ``vocab_size=`` override.
    """
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    if hasattr(tokenizer, "encoder"):
        return len(tokenizer.encoder)
    raise AttributeError(
        f"Could not infer vocab_size from tokenizer of type {type(tokenizer).__name__} "
        "(no `vocab_size` or `encoder` attribute found). Pass `vocab_size=` explicitly "
        "to SyntheticFixedLengthDataset/synthetic_fixed_length_dataset."
    )


class SyntheticFixedLengthDataset(Dataset):
    """Synthetic dataset of fixed-length random token sequences, for isoFLOPs-style
    throughput/MFU benchmarks that need to decouple measurement from any real
    corpus's length distribution.

    Every sample is exactly ``seq_len`` tokens. ``padded_collate_sft`` never
    constructs a ``"mask"`` key for a plain (non-packed) batch regardless of
    config, so ``mask=None`` reaches attention unconditionally with this
    dataset (packing's block-diagonal mask, which blocks the native XPU flash
    kernel, is unnecessary and must not be enabled — see ``dataset.packed``
    in configs using this dataset, which must be left unset/``false``).

    NOTE the weaker "no padding is ever added" claim only holds when
    ``pad_to_multiple_of == 1`` (``ParallelDims.min_seq_len_divisor`` returns
    1 unless tensor/context parallelism are enabled — true for every config
    this dataset currently ships with, but NOT true unconditionally). At
    ``tp>1`` or ``cp>1``, ``padded_collate_sft`` pads every sample by a
    further ``pad_to_multiple_of`` block even though every sample is already
    ``seq_len`` long — this changes the effective per-step token count but
    does NOT affect the ``mask=None`` claim above, which is independent of
    padding.

    Labels mirror tokens (every position contributes to the loss), matching
    the ``train_on_input: true`` convention used by this project's other
    throughput benchmarks for token-accounting parity.

    Args:
        tokenizer (ModelTokenizer): Unused for token generation (ids are drawn
            directly from ``[0, vocab_size)``, not real text), but accepted
            positionally to match the standard torchtune dataset builder
            signature (``config.instantiate(cfg.dataset, self._tokenizer)``).
            ``vocab_size`` is read from this tokenizer when not passed
            explicitly, so generated ids stay in-range for the model being
            benchmarked without hardcoding a vocab size per model.
        seq_len (int): Exact length of every sample, in tokens.
        vocab_size (Optional[int]): Upper bound (exclusive) for generated
            token ids. Defaults to ``tokenizer.vocab_size``.
        num_samples (int): Number of samples the dataset reports via
            ``__len__``. Default 10_000 (larger than any realistic
            few-hundred-step throughput benchmark needs).
        seed (int): Base seed. Each sample is generated deterministically
            from ``seed + index``, so samples are reproducible without
            pre-materializing the full dataset in memory.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        seq_len: int,
        vocab_size: Optional[int] = None,
        num_samples: int = 10_000,
        seed: int = 0,
    ):
        self._seq_len = seq_len
        self._vocab_size = vocab_size if vocab_size is not None else _infer_vocab_size(tokenizer)
        self._num_samples = num_samples
        self._seed = seed

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self._seed + index)
        tokens = torch.randint(
            0, self._vocab_size, (self._seq_len,), generator=generator, dtype=torch.long
        )
        return {"tokens": tokens, "labels": tokens.clone()}


def synthetic_fixed_length_dataset(
    tokenizer: ModelTokenizer,
    *,
    seq_len: int,
    vocab_size: Optional[int] = None,
    num_samples: int = 10_000,
    seed: int = 0,
) -> SyntheticFixedLengthDataset:
    """Builder for :class:`SyntheticFixedLengthDataset`, for use as a
    ``dataset._component_`` in a recipe config. See the class docstring for
    the full contract (every sample exactly ``seq_len`` tokens, labels mirror
    tokens, ``dataset.packed`` must be left unset/``false``).
    """
    return SyntheticFixedLengthDataset(
        tokenizer,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
        seed=seed,
    )

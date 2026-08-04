# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic length-grouped distributed batch sampler.

Extracted from ``torchtune.dev.bioreason.dataset_sft.LengthGroupedDistributedBatchSampler``
(BioReason 32B SFT, validated ~1.49x throughput after XPU flash made per-sample compute
cheap enough that batching, not sequence length, became the throughput lever — see
``docs/status.md`` 2026-07-15 and ``memory/project_bioreason_sft_flash_native_levers_20260715``).

Unlike dataset packing (which requires a block-diagonal document mask incompatible with
the native XPU flash kernel's ``mask=None``-only fast path, and was HW-tested ~4.9x SLOWER
via flex — see ``memory/project_bioreason_sft_packing_scope_20260715``), bucketing keeps
every sample's own causal boundary intact: each batch is homogeneous-length (one bucket),
so ``padded_collate_sft`` naturally pads to the batch's own max length with no cross-sample
mask needed, and ``TORCHTUNE_USE_XPU_FLASH=1`` engages directly.
"""

import logging
from typing import Sequence

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


def compute_dataset_lengths(dataset: Dataset, max_seq_len: int) -> list[int]:
    """Full tokenized length per example, capped at ``max_seq_len``.

    Generic across any ``Dataset`` whose ``__getitem__`` returns a dict with a ``tokens``
    key (the torchtune SFT dataset contract) — this is exactly the length the collate pads
    to, so a bucketed sampler keyed on it groups examples by their real per-step tensor
    shape. Requires one full pass over the dataset (tokenizes every example); cache the
    result on the caller across epochs.
    """
    lengths: list[int] = []
    for i in range(len(dataset)):
        n = len(dataset[i]["tokens"])
        lengths.append(min(n, max_seq_len))
    return lengths


class LengthGroupedDistributedBatchSampler(Sampler):
    """Distributed batch sampler that groups examples by length bucket and gives each
    bucket its own batch size — so short sequences train in bigger microbatches and
    per-microbatch FSDP overhead amortizes over more samples, without needing dataset
    packing's block-diagonal mask (which blocks the native XPU flash kernel).

    FSDP correctness contract (why this is safe):
      * Every DP rank yields the same number of batches per epoch (equal batch COUNT is
        guaranteed by construction — a bucket contributes
        ``floor(n_bucket / (bs_bucket * num_replicas))`` slots, same for all ranks).
      * At each batch slot ALL ranks draw from the SAME bucket, so their sequence lengths
        match — no straggler on the per-microbatch collective.
      * Token-weighted loss (``loss * num_tokens``, all-reduced) already handles unequal
        samples-per-rank within a step correctly — no change needed there.

    Args:
        lengths (Sequence[int]): per-example token length (see :func:`compute_dataset_lengths`).
        buckets (Sequence[int]): ascending bucket ceilings, e.g. ``[512, 1024, 1536]``. An
            example lands in the smallest bucket ``>=`` its length; the top bucket must be
            ``>=`` every length (append ``max_seq_len`` if unsure).
        bucket_batch_sizes (Sequence[int]): batch size per bucket, same length/order as
            ``buckets``. Larger for shorter buckets.
        num_replicas (int): DP world size.
        rank (int): this process's DP rank.
        shuffle (bool): shuffle within-bucket assignment and slot order per epoch.
        seed (int): base RNG seed (combined with epoch via ``set_epoch``).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        buckets: Sequence[int],
        bucket_batch_sizes: Sequence[int],
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

        self._num_batches = sum(
            len(idxs) // (bs * self.num_replicas)
            for idxs, bs in zip(self._indices_by_bucket, self.bucket_batch_sizes)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def _build_slots(self) -> list[list[int]]:
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
        if self.shuffle and slots:
            order = torch.randperm(len(slots), generator=g).tolist()
            slots = [slots[o] for o in order]
        return slots

    def __iter__(self):
        yield from self._build_slots()

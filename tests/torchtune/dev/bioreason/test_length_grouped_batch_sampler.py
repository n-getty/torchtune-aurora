# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU-safe unit tests for LengthGroupedDistributedBatchSampler (per-bucket batch sizing).
#
# The sampler is the throughput lever that lets short sequences train in bigger microbatches
# WITHOUT dropping any corpus. Its FSDP-correctness contract is:
#   (1) every DP rank yields the SAME NUMBER of batches (else a collective hangs), and
#   (2) at each batch slot all ranks draw from the SAME bucket (matched seq length -> no
#       reduce-scatter straggler).
# These tests pin both, plus bucket assignment, per-bucket batch size, drop-last, no
# cross-rank example overlap, and epoch-determinism.

import pytest

from torchtune.dev.bioreason.dataset_sft import LengthGroupedDistributedBatchSampler


def _make(lengths, buckets, bbs, num_replicas, rank, shuffle=True, seed=0, epoch=0):
    s = LengthGroupedDistributedBatchSampler(
        lengths=lengths,
        buckets=buckets,
        bucket_batch_sizes=bbs,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )
    s.set_epoch(epoch)
    return s


def test_bucket_assignment_smallest_ceiling():
    # length -> smallest bucket >= length
    lengths = [10, 2048, 2049, 4096, 4097, 6144]
    s = _make(lengths, [2048, 4096, 6144], [4, 2, 1], num_replicas=1, rank=0)
    assert s._bucket_of == [0, 0, 1, 1, 2, 2]


def test_per_bucket_batch_size_and_shapes():
    # 8 short (bucket0 bs4), 4 mid (bucket1 bs2), 3 long (bucket2 bs1), 1 replica.
    lengths = [100] * 8 + [3000] * 4 + [5000] * 3
    s = _make(lengths, [2048, 4096, 6144], [4, 2, 1], num_replicas=1, rank=0, shuffle=False)
    batches = list(iter(s))
    sizes = sorted(len(b) for b in batches)
    # bucket0: 8//4=2 batches of 4; bucket1: 4//2=2 batches of 2; bucket2: 3//1=3 of 1
    assert sizes == [1, 1, 1, 2, 2, 4, 4]
    assert len(s) == len(batches) == 7


def test_equal_batch_count_across_ranks():
    # The core FSDP invariant: every rank yields the same number of batches.
    lengths = [100] * 37 + [3000] * 15 + [5000] * 9  # deliberately non-divisible
    kw = dict(buckets=[2048, 4096, 6144], bbs=[4, 2, 1], num_replicas=4)
    counts = [len(list(iter(_make(lengths, rank=r, **kw)))) for r in range(4)]
    assert len(set(counts)) == 1, f"ranks disagree on batch count: {counts}"
    # __len__ must match the realized count too.
    assert all(len(_make(lengths, rank=r, **kw)) == counts[0] for r in range(4))


def test_matched_bucket_per_slot_across_ranks():
    # At slot i, every rank must draw the same bucket (matched seq length). We infer the
    # bucket from the example lengths in each rank's slot i.
    lengths = [100] * 40 + [3000] * 16 + [5000] * 8
    kw = dict(buckets=[2048, 4096, 6144], bbs=[4, 2, 1], num_replicas=4, seed=7, epoch=2)
    per_rank = [list(iter(_make(lengths, rank=r, **kw))) for r in range(4)]
    n = len(per_rank[0])
    for i in range(n):
        bucket_lens = set()
        for r in range(4):
            slot = per_rank[r][i]
            # all examples in a slot share a bucket; use the representative length
            bucket_lens.add(lengths[slot[0]])
        assert len(bucket_lens) == 1, f"slot {i}: ranks drew different buckets {bucket_lens}"


def test_no_cross_rank_overlap_within_epoch():
    # A given example index must not appear on two ranks in the same epoch (would double-
    # count it and desync the token-weighted loss).
    lengths = [100] * 40 + [3000] * 16 + [5000] * 8
    kw = dict(buckets=[2048, 4096, 6144], bbs=[4, 2, 1], num_replicas=4, seed=3, epoch=1)
    seen = set()
    for r in range(4):
        for batch in iter(_make(lengths, rank=r, **kw)):
            for idx in batch:
                assert idx not in seen, f"index {idx} appears on multiple ranks"
                seen.add(idx)


def test_epoch_determinism_and_variation():
    lengths = [100] * 20 + [3000] * 8
    kw = dict(buckets=[2048, 4096], bbs=[4, 2], num_replicas=2, rank=0, seed=11)
    e0a = list(iter(_make(lengths, epoch=0, **kw)))
    e0b = list(iter(_make(lengths, epoch=0, **kw)))
    e1 = list(iter(_make(lengths, epoch=1, **kw)))
    assert e0a == e0b, "same (seed, epoch) must be deterministic"
    assert e0a != e1, "different epoch should reshuffle"


def test_clamp_over_top_bucket():
    # lengths above the top bucket clamp into it (compute_lengths caps at max_seq_len, but
    # guard anyway); they must still be batched, not dropped.
    lengths = [100] * 4 + [9999] * 4
    s = _make(lengths, [2048, 4096], [4, 1], num_replicas=1, rank=0, shuffle=False)
    assert s._bucket_of == [0, 0, 0, 0, 1, 1, 1, 1]
    all_idx = [i for b in iter(s) for i in b]
    assert sorted(all_idx) == list(range(8))


def test_validation_errors():
    with pytest.raises(ValueError):  # mismatched lengths
        LengthGroupedDistributedBatchSampler([1], [2048, 4096], [4], 1, 0)
    with pytest.raises(ValueError):  # non-ascending buckets
        LengthGroupedDistributedBatchSampler([1], [4096, 2048], [1, 4], 1, 0)
    with pytest.raises(ValueError):  # bs < 1
        LengthGroupedDistributedBatchSampler([1], [2048], [0], 1, 0)

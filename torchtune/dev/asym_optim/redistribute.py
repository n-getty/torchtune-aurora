# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Pure index-math helpers for the 12->4 / 4->12 redistribute used by
# AsymAdamWXPU. The helpers are factored out of the collectives so they
# can be unit-tested on a login node without dist.init_process_group.

from typing import List, Sequence, Tuple

import torch


def compute_overlap_matrix(
    n_src: int, src_shard_size: int, n_dst: int
) -> Tuple[List[List[int]], int]:
    """Return overlap[i][j] = number of elements trainer-shard i contributes to
    spare-shard j, plus the per-spare receive size (n_dst splits the global
    numel evenly after FSDP2's pad-to-divisible-by-n_src convention).

    The "global numel" for the param is ``n_src * src_shard_size`` (i.e.
    FSDP2's padded total). Spare partitioning is contiguous along the same
    flat axis with shard size ``ceil(total / n_dst)`` rounded up to keep
    spare splits equal — last spare may receive a smaller real slice if
    ``total`` is not divisible by ``n_dst``.

    Returns:
        overlap: List[List[int]]  shape (n_src, n_dst)
        dst_split_size: int       per-spare receive size

    The caller is responsible for tracking any extra padding inside the last
    spare's shard so the inverse scatter restores trainer shards exactly.
    """
    if src_shard_size < 0 or n_src <= 0 or n_dst <= 0:
        raise ValueError("n_src, n_dst, src_shard_size must be positive ints")
    total = n_src * src_shard_size
    # Pad total up to a multiple of n_dst so every spare gets the same chunk
    # (matches what we'd do at the wire — pad on the spare side, strip on
    # scatter back).
    if total % n_dst == 0:
        dst_total = total
    else:
        dst_total = total + (n_dst - total % n_dst)
    dst_split_size = dst_total // n_dst

    overlap = [[0] * n_dst for _ in range(n_src)]
    for i in range(n_src):
        src_lo = i * src_shard_size
        src_hi = src_lo + src_shard_size
        for j in range(n_dst):
            dst_lo = j * dst_split_size
            dst_hi = dst_lo + dst_split_size
            o = max(0, min(src_hi, dst_hi) - max(src_lo, dst_lo))
            overlap[i][j] = o
    return overlap, dst_split_size


def build_a2a_splits(
    overlap: List[List[int]],
    pg_ranks: Sequence[int],
    train_ranks: Sequence[int],
    spare_ranks: Sequence[int],
    my_rank: int,
    direction: str,
) -> Tuple[List[int], List[int]]:
    """Produce (input_split_sizes, output_split_sizes) ordered by ``pg_ranks``
    for ``dist.all_to_all_single`` on a PG that contains every rank in
    ``train_ranks ∪ spare_ranks``.

    direction ``"gather"`` (12->4): trainers send their shard fragments to
    spares; spares send nothing.

    direction ``"scatter"`` (4->12): spares send fragments back to trainers;
    trainers send nothing.
    """
    if direction not in ("gather", "scatter"):
        raise ValueError("direction must be 'gather' or 'scatter'")

    n_pg = len(pg_ranks)
    rank_to_pg = {r: i for i, r in enumerate(pg_ranks)}
    in_splits = [0] * n_pg
    out_splits = [0] * n_pg

    if direction == "gather":
        if my_rank in train_ranks:
            i = list(train_ranks).index(my_rank)
            for j, sp in enumerate(spare_ranks):
                in_splits[rank_to_pg[sp]] = overlap[i][j]
        if my_rank in spare_ranks:
            j = list(spare_ranks).index(my_rank)
            for i, tr in enumerate(train_ranks):
                out_splits[rank_to_pg[tr]] = overlap[i][j]
    else:
        if my_rank in spare_ranks:
            j = list(spare_ranks).index(my_rank)
            for i, tr in enumerate(train_ranks):
                in_splits[rank_to_pg[tr]] = overlap[i][j]
        if my_rank in train_ranks:
            i = list(train_ranks).index(my_rank)
            for j, sp in enumerate(spare_ranks):
                out_splits[rank_to_pg[sp]] = overlap[i][j]
    return in_splits, out_splits


def cpu_simulate_round_trip(
    src_shards: List[torch.Tensor], n_dst: int
) -> List[torch.Tensor]:
    """Pure-CPU simulation of the 12->4 gather followed by the 4->12 scatter,
    used by tests to assert round-trip identity without any collectives.

    Expects ``src_shards`` to be a list of length ``n_src`` (e.g. 12) of
    1-D tensors of equal size ``src_shard_size`` (already FSDP2-padded).
    Returns the round-tripped trainer shards.
    """
    n_src = len(src_shards)
    src_shard_size = src_shards[0].numel()
    for s in src_shards:
        if s.numel() != src_shard_size:
            raise ValueError("all src shards must be the same size")
        if s.dim() != 1:
            raise ValueError("src shards must be flat 1-D")

    overlap, dst_split_size = compute_overlap_matrix(
        n_src, src_shard_size, n_dst
    )

    # Build dst shards.
    dst_shards = [
        torch.zeros(dst_split_size, dtype=src_shards[0].dtype) for _ in range(n_dst)
    ]
    for j in range(n_dst):
        dst_lo = j * dst_split_size
        dst_offset = 0
        for i in range(n_src):
            o = overlap[i][j]
            if o == 0:
                continue
            src_lo = i * src_shard_size
            # Source slice [a:b) from the trainer shard.
            a = max(0, dst_lo + dst_offset - src_lo)
            b = a + o
            dst_shards[j][dst_offset : dst_offset + o] = src_shards[i][a:b]
            dst_offset += o

    # Round-trip back.
    out = [torch.zeros_like(s) for s in src_shards]
    for i in range(n_src):
        src_lo = i * src_shard_size
        src_offset = 0
        for j in range(n_dst):
            o = overlap[i][j]
            if o == 0:
                continue
            dst_lo = j * dst_split_size
            a = max(0, src_lo + src_offset - dst_lo)
            b = a + o
            out[i][src_offset : src_offset + o] = dst_shards[j][a:b]
            src_offset += o
    return out

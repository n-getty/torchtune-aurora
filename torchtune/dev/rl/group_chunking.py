# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Group-aligned forward chunking for GRPO prefix dedup.

GRPO lays its rollouts out **group-contiguous**: the ``G`` continuations of prompt
``b`` occupy rows ``[b*G, (b+1)*G)``. BioReason builds them that way explicitly
(``pe_base.unsqueeze(1).expand(-1, G, -1, -1).reshape(B*G, ...)`` in
``grpo_bioreason_distributed_xpu.py``), and ``ref_prefix_share`` already depends on
the same layout.

A prompt prefix can only be deduplicated among rows that are **in the same forward
chunk** -- the chunk is the unit that gets one forward call. So dedup requires
``group_size % forward_batch_size == 0``; otherwise a chunk straddles a group
boundary, holds a partial group, and there is no complete shared prefix to hoist.

Production today runs ``G=8, fbs=2``, which *is* aligned, but the payoff depends
sharply on how much of a group a chunk covers::

    B=4 G=8, prefix~4096 tok, response~1152 tok, num_seqs=32
      fbs=2 (prod) : 16 chunks, 102,400 tok  -> 1.64x
      fbs=3        : NOT aligned (8 % 3 != 0) -- dedup impossible
      fbs=4        :  8 chunks,  69,632 tok  -> 2.41x
      fbs=8        :  4 chunks,  53,248 tok  -> 3.15x   (whole group per chunk)

Baseline is ``num_seqs * (prefix + response) = 167,936`` tokens regardless of fbs.

The non-obvious consequence: a **deduplicated** ``fbs=8`` chunk holds ``4096 +
8*1152 = 13,312`` tokens against today's ``2*(4096+1152) = 10,496`` -- only 1.27x --
because dedup removes seven of the eight prefix copies that would otherwise make
``fbs=8`` a 4x chunk. Dedup and a larger ``fbs`` are therefore complements, not
competing levers.

These are **token counts, not time predictions.** The backward on this workload is
measured to be *not* FLOP-bound (length-sorted chunking removed 25.6% of padded
FLOPs for +0% wall clock), so treat the ratios as upper bounds. See
``memory/project_bioreason_dedup_payoff_depends_on_fbs_20260916.md``.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def group_aligned_supported(
    *, num_seqs: int, group_size: int, forward_batch_size: int
) -> tuple[bool, str]:
    """Check whether chunks of ``forward_batch_size`` rows respect group boundaries.

    Args:
        num_seqs (int): total rows in the batch (``B*G``).
        group_size (int): ``G``, continuations per prompt.
        forward_batch_size (int): rows per forward chunk.

    Returns:
        tuple[bool, str]: ``(supported, reason)``; ``reason`` is empty when supported.
    """
    if group_size <= 1:
        return False, f"group_size={group_size}; nothing to share"
    if forward_batch_size <= 0:
        return False, f"forward_batch_size={forward_batch_size} must be positive"
    if num_seqs % group_size != 0:
        return False, f"num_seqs={num_seqs} not divisible by group_size={group_size}"
    if forward_batch_size > group_size:
        # A chunk spanning >1 group is fine for correctness but this helper's
        # contract is one shared prefix per chunk; multi-group chunks need the
        # per-group loop instead. Reject rather than silently dedup only part.
        return (
            False,
            f"forward_batch_size={forward_batch_size} exceeds group_size="
            f"{group_size}; a chunk would span multiple prompts",
        )
    if group_size % forward_batch_size != 0:
        return (
            False,
            f"group_size={group_size} not divisible by forward_batch_size="
            f"{forward_batch_size}; chunks would straddle group boundaries",
        )
    return True, ""


def group_aligned_chunk_ranges(
    *, num_seqs: int, group_size: int, forward_batch_size: int
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` chunks that never straddle a group boundary.

    Every returned chunk lies entirely within one group, so all of its rows share a
    single prompt prefix.

    Args:
        num_seqs (int): total rows in the batch (``B*G``).
        group_size (int): ``G``, continuations per prompt.
        forward_batch_size (int): rows per forward chunk.

    Returns:
        list[tuple[int, int]]: half-open row ranges covering ``[0, num_seqs)`` in order.

    Raises:
        ValueError: if the geometry is not group-aligned; call
            :func:`group_aligned_supported` first to branch instead of raising.
    """
    ok, reason = group_aligned_supported(
        num_seqs=num_seqs,
        group_size=group_size,
        forward_batch_size=forward_batch_size,
    )
    if not ok:
        raise ValueError(f"not group-aligned: {reason}")
    ranges: list[tuple[int, int]] = []
    for gs in range(0, num_seqs, group_size):
        for cs in range(gs, gs + group_size, forward_batch_size):
            ranges.append((cs, cs + forward_batch_size))
    return ranges


def dedup_token_counts(
    *,
    num_seqs: int,
    group_size: int,
    forward_batch_size: int,
    prefix_len: int,
    response_len: int,
) -> tuple[int, int]:
    """Token counts with and without within-chunk prefix dedup.

    Args:
        num_seqs (int): total rows (``B*G``).
        group_size (int): ``G``.
        forward_batch_size (int): rows per forward chunk.
        prefix_len (int): prompt tokens per row.
        response_len (int): response tokens per row.

    Returns:
        tuple[int, int]: ``(baseline_tokens, dedup_tokens)``. When the geometry is
        not group-aligned the two are equal -- dedup is unavailable, not free.
    """
    baseline = num_seqs * (prefix_len + response_len)
    ok, _ = group_aligned_supported(
        num_seqs=num_seqs,
        group_size=group_size,
        forward_batch_size=forward_batch_size,
    )
    if not ok:
        return baseline, baseline
    n_chunks = num_seqs // forward_batch_size
    dedup = n_chunks * (prefix_len + forward_batch_size * response_len)
    return baseline, dedup

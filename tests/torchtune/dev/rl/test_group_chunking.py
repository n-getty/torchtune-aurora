# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Group-aligned chunking for GRPO prefix dedup.

Pins the preconditions a within-chunk prefix dedup depends on. The load-bearing one
is that ``group_size % forward_batch_size == 0``: production runs ``G=8, fbs=2``
(aligned), but ``fbs=3`` was a live candidate from the throughput sweep and is
**not** aligned, so it silently cannot dedup at all.
"""
import pytest

from torchtune.dev.rl.group_chunking import (
    dedup_token_counts,
    group_aligned_chunk_ranges,
    group_aligned_supported,
)

# BioReason 32B production shapes (B=4 G=8, ~4096-token prompt, ~1152-token response).
PREFIX, RESP, G, NUM_SEQS = 4096, 1152, 8, 32


class TestGroupAlignedSupported:
    @pytest.mark.parametrize("fbs", [1, 2, 4, 8])
    def test_divisors_of_group_size_are_aligned(self, fbs):
        ok, reason = group_aligned_supported(
            num_seqs=NUM_SEQS, group_size=G, forward_batch_size=fbs
        )
        assert ok, reason

    @pytest.mark.parametrize("fbs", [3, 5, 6, 7])
    def test_non_divisors_are_rejected(self, fbs):
        ok, reason = group_aligned_supported(
            num_seqs=NUM_SEQS, group_size=G, forward_batch_size=fbs
        )
        assert not ok
        assert "straddle" in reason

    def test_chunk_spanning_multiple_groups_is_rejected(self):
        # fbs > G would put two prompts' rows in one chunk: correct to compute, but
        # it is not the one-shared-prefix-per-chunk contract this helper offers.
        ok, reason = group_aligned_supported(
            num_seqs=NUM_SEQS, group_size=G, forward_batch_size=16
        )
        assert not ok
        assert "multiple prompts" in reason

    def test_group_size_one_has_nothing_to_share(self):
        ok, reason = group_aligned_supported(
            num_seqs=4, group_size=1, forward_batch_size=1
        )
        assert not ok
        assert "nothing to share" in reason

    def test_ragged_batch_rejected(self):
        ok, reason = group_aligned_supported(
            num_seqs=30, group_size=G, forward_batch_size=2
        )
        assert not ok
        assert "not divisible" in reason


class TestChunkRanges:
    @pytest.mark.parametrize("fbs", [1, 2, 4, 8])
    def test_ranges_tile_the_batch_exactly(self, fbs):
        ranges = group_aligned_chunk_ranges(
            num_seqs=NUM_SEQS, group_size=G, forward_batch_size=fbs
        )
        assert ranges[0][0] == 0
        assert ranges[-1][1] == NUM_SEQS
        for (_, prev_end), (nxt_start, _) in zip(ranges, ranges[1:]):
            assert prev_end == nxt_start, "chunks must be contiguous and non-overlapping"
        assert sum(e - s for s, e in ranges) == NUM_SEQS

    @pytest.mark.parametrize("fbs", [1, 2, 4, 8])
    def test_no_chunk_straddles_a_group_boundary(self, fbs):
        """The property the whole dedup rests on: one chunk sees one prompt."""
        for start, end in group_aligned_chunk_ranges(
            num_seqs=NUM_SEQS, group_size=G, forward_batch_size=fbs
        ):
            assert start // G == (end - 1) // G, (
                f"chunk [{start},{end}) spans groups "
                f"{start // G}..{(end - 1) // G}"
            )

    def test_matches_naive_ranges_when_aligned(self):
        # When aligned, group-aware chunking must not perturb the row order or the
        # chunk boundaries the recipe already uses -- otherwise enabling dedup would
        # silently change batching alongside the intended optimization.
        fbs = 2
        naive = [
            (cs, min(cs + fbs, NUM_SEQS)) for cs in range(0, NUM_SEQS, fbs)
        ]
        assert (
            group_aligned_chunk_ranges(
                num_seqs=NUM_SEQS, group_size=G, forward_batch_size=fbs
            )
            == naive
        )

    def test_raises_when_not_aligned(self):
        with pytest.raises(ValueError, match="not group-aligned"):
            group_aligned_chunk_ranges(
                num_seqs=NUM_SEQS, group_size=G, forward_batch_size=3
            )


class TestTokenCounts:
    def test_baseline_is_independent_of_fbs(self):
        baselines = {
            dedup_token_counts(
                num_seqs=NUM_SEQS,
                group_size=G,
                forward_batch_size=fbs,
                prefix_len=PREFIX,
                response_len=RESP,
            )[0]
            for fbs in (1, 2, 3, 4, 8)
        }
        assert baselines == {NUM_SEQS * (PREFIX + RESP)} == {167_936}

    @pytest.mark.parametrize(
        "fbs,expected_dedup", [(2, 102_400), (4, 69_632), (8, 53_248)]
    )
    def test_production_payoff_table(self, fbs, expected_dedup):
        """Guards the documented numbers; prod fbs=2 is 1.64x, NOT the ~2.2x once
        recorded (that figure was the unreachable fbs=3 row)."""
        baseline, dedup = dedup_token_counts(
            num_seqs=NUM_SEQS,
            group_size=G,
            forward_batch_size=fbs,
            prefix_len=PREFIX,
            response_len=RESP,
        )
        assert dedup == expected_dedup
        assert baseline > dedup

    def test_payoff_grows_monotonically_with_fbs(self):
        payoffs = []
        for fbs in (1, 2, 4, 8):
            baseline, dedup = dedup_token_counts(
                num_seqs=NUM_SEQS,
                group_size=G,
                forward_batch_size=fbs,
                prefix_len=PREFIX,
                response_len=RESP,
            )
            payoffs.append(baseline / dedup)
        assert payoffs == sorted(payoffs)
        assert payoffs[0] == pytest.approx(1.0), "fbs=1 cannot share anything"
        assert payoffs[-1] == pytest.approx(3.15, abs=0.01)

    def test_unaligned_reports_no_saving_rather_than_a_fake_one(self):
        baseline, dedup = dedup_token_counts(
            num_seqs=NUM_SEQS,
            group_size=G,
            forward_batch_size=3,
            prefix_len=PREFIX,
            response_len=RESP,
        )
        assert baseline == dedup, "fbs=3 is unaligned: dedup is unavailable, not free"

    def test_dedup_makes_a_bigger_chunk_nearly_free(self):
        """A dedup'd fbs=8 chunk is only ~1.27x today's fbs=2 chunk, because dedup
        deletes 7 of 8 prefix copies. Dedup and larger fbs are complements."""
        today_chunk = 2 * (PREFIX + RESP)
        dedup_chunk = PREFIX + 8 * RESP
        assert dedup_chunk / today_chunk == pytest.approx(1.27, abs=0.01)
        naive_chunk = 8 * (PREFIX + RESP)
        assert naive_chunk / today_chunk == pytest.approx(4.0, abs=0.01)

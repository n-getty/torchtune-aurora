# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test for the vLLM sampled-logprob capture path.

Phase 1 of the vLLM importance-sampling correction (plan
~/.claude/plans/linear-singing-cocke.md). Pins down the shape and dtype of
the per-token sampled-logprob tensor we expect to extract from a
vLLM RequestOutput when SamplingParams(logprobs=0) is set, AND the
GRPOTrajectory NamedTuple's new `vllm_sampled_logprobs` slot.

A real vLLM call returns CompletionOutput.logprobs as
``list[Optional[dict[int, Logprob]]]`` (per vllm/sampling_params.py: with
``logprobs=0`` the sampled token's own logprob is always included, so the
dict has at least one entry). We use stub objects here to avoid an XPU/vLLM
dependency at test time — the only contract we need to lock down is that
the extractor reads ``output.outputs[0].logprobs[i][token_id].logprob``
and produces a dense ``[bsz, max_gen]`` float32 tensor padded with 0.0.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from torchtune.dev.rl.types import GRPOTrajectory

# The vLLM sampled-logprob capture (GRPOTrajectory.vllm_sampled_logprobs slot +
# _extract_vllm_sampled_logprobs) these tests pin was written on 2026-06-11 then lost
# as collateral when the recipe files were `git checkout c58acbed`'d to undo the bad
# "Antigravity walkthrough" patches. It is the rollout-side half of the parked
# IS-correction feature (only needed for off-policy/async GRPO, not the current
# on-policy sync recipe). Kept as the executable spec; partial impl preserved at git
# tag `parked/is-dapo-impl`. Delete this skip when re-landing. See
# memory/project_ezpz_gap_was_lr_not_code.md + project_vllm_is_correction_phase4_results.md.
pytestmark = pytest.mark.skip(
    reason="vLLM logprob-capture impl parked (collateral revert 2026-06-11); spec kept, "
    "code at tag parked/is-dapo-impl. See project_ezpz_gap_was_lr_not_code.md."
)


@dataclass
class _StubLogprob:
    logprob: float


@dataclass
class _StubCompletion:
    token_ids: list[int]
    logprobs: Optional[list[Optional[dict[int, _StubLogprob]]]]


@dataclass
class _StubRequest:
    outputs: list[_StubCompletion]


def _extract_vllm_sampled_logprobs(
    outputs, bsz: int, max_generated_tokens: int, device=None
) -> torch.Tensor:
    """Pull the per-position sampled-token logprob into a dense tensor.

    This is the reference shape the recipe-side extractor must match.
    Positions past the actual completion length are padded with 0.0
    (consistent with how `response_padding_masks` zeros them later).
    """
    out = torch.zeros(
        (bsz, max_generated_tokens),
        dtype=torch.float32,
        device=device if device is not None else torch.device("cpu"),
    )
    for i, response in enumerate(outputs):
        comp = response.outputs[0]
        token_ids = comp.token_ids
        per_pos = comp.logprobs
        if per_pos is None:
            continue
        n = min(len(token_ids), max_generated_tokens, len(per_pos))
        for j in range(n):
            tid = token_ids[j]
            slot = per_pos[j]
            if slot is None:
                continue
            entry = slot.get(tid)
            if entry is None:
                # Some vLLM versions key by the top-1 candidate when
                # logprobs=0 — fall back to the first value.
                if len(slot) == 0:
                    continue
                entry = next(iter(slot.values()))
            out[i, j] = float(entry.logprob)
    return out


class TestVllmLogprobsExtractor(unittest.TestCase):
    def test_dense_extraction_padding(self):
        # bsz=2, max_gen=4, first response is 3 tokens, second is 4 tokens
        outs = [
            _StubRequest(outputs=[_StubCompletion(
                token_ids=[10, 11, 12],
                logprobs=[
                    {10: _StubLogprob(-1.0)},
                    {11: _StubLogprob(-2.0)},
                    {12: _StubLogprob(-0.5)},
                ],
            )]),
            _StubRequest(outputs=[_StubCompletion(
                token_ids=[20, 21, 22, 23],
                logprobs=[
                    {20: _StubLogprob(-0.1)},
                    {21: _StubLogprob(-0.2)},
                    {22: _StubLogprob(-0.3)},
                    {23: _StubLogprob(-0.4)},
                ],
            )]),
        ]
        t = _extract_vllm_sampled_logprobs(outs, bsz=2, max_generated_tokens=4)
        self.assertEqual(t.shape, (2, 4))
        self.assertEqual(t.dtype, torch.float32)
        # First row: 3 real entries + 1 zero pad
        torch.testing.assert_close(
            t[0],
            torch.tensor([-1.0, -2.0, -0.5, 0.0], dtype=torch.float32),
        )
        # Second row: 4 real entries
        torch.testing.assert_close(
            t[1],
            torch.tensor([-0.1, -0.2, -0.3, -0.4], dtype=torch.float32),
        )

    def test_none_slot_pads_zero(self):
        outs = [
            _StubRequest(outputs=[_StubCompletion(
                token_ids=[1, 2],
                # vLLM occasionally emits None for a position when the
                # request is canceled mid-generation. Pad as zero.
                logprobs=[{1: _StubLogprob(-3.0)}, None],
            )]),
        ]
        t = _extract_vllm_sampled_logprobs(outs, bsz=1, max_generated_tokens=2)
        torch.testing.assert_close(
            t[0], torch.tensor([-3.0, 0.0], dtype=torch.float32),
        )

    def test_top1_fallback_when_sampled_token_missing(self):
        # Some vLLM builds put only the top-1 candidate in the dict when
        # logprobs=0. Sampled token differs → fall back to the dict's only
        # value rather than crashing.
        outs = [
            _StubRequest(outputs=[_StubCompletion(
                token_ids=[42],
                logprobs=[{99: _StubLogprob(-7.0)}],
            )]),
        ]
        t = _extract_vllm_sampled_logprobs(outs, bsz=1, max_generated_tokens=1)
        torch.testing.assert_close(
            t[0], torch.tensor([-7.0], dtype=torch.float32),
        )

    def test_truncates_to_max_generated_tokens(self):
        # If vLLM returns more than max_generated_tokens (e.g. driver glitch),
        # the extractor must NOT overflow the output tensor.
        outs = [
            _StubRequest(outputs=[_StubCompletion(
                token_ids=[0, 1, 2, 3, 4],
                logprobs=[{i: _StubLogprob(-float(i))} for i in range(5)],
            )]),
        ]
        t = _extract_vllm_sampled_logprobs(outs, bsz=1, max_generated_tokens=3)
        self.assertEqual(t.shape, (1, 3))
        torch.testing.assert_close(
            t[0], torch.tensor([0.0, -1.0, -2.0], dtype=torch.float32),
        )


class TestGRPOTrajectoryHasVllmLogprobsField(unittest.TestCase):
    """Phase 2 contract: trajectory carries the captured logprobs."""

    def test_field_exists_and_defaults_to_none(self):
        self.assertIn("vllm_sampled_logprobs", GRPOTrajectory._fields)
        t = GRPOTrajectory()
        self.assertIsNone(t.vllm_sampled_logprobs)

    def test_slice_trajectory_carries_field(self):
        from torchtune.dev.rl.distributed import _slice_trajectory

        bsz, T = 4, 8
        traj = GRPOTrajectory(
            query_responses=torch.zeros(bsz, T),
            logprobs=torch.zeros(bsz, T),
            ref_logprobs=torch.zeros(bsz, T),
            advantages=torch.zeros(bsz),
            rewards=torch.zeros(bsz),
            successes=torch.zeros(bsz),
            masks=None,
            position_ids=torch.zeros(bsz, T, dtype=torch.long),
            response_padding_masks=torch.zeros(bsz, T, dtype=torch.bool),
            seq_lens=torch.full((bsz,), T, dtype=torch.long),
            answers=["a"] * bsz,
            vllm_sampled_logprobs=torch.arange(bsz * T, dtype=torch.float32).view(bsz, T),
        )
        sliced = _slice_trajectory(traj, 1, 3)
        self.assertIsNotNone(sliced.vllm_sampled_logprobs)
        self.assertEqual(sliced.vllm_sampled_logprobs.shape, (2, T))
        torch.testing.assert_close(
            sliced.vllm_sampled_logprobs,
            torch.arange(bsz * T, dtype=torch.float32).view(bsz, T)[1:3],
        )

    def test_slice_trajectory_handles_none_field(self):
        from torchtune.dev.rl.distributed import _slice_trajectory

        bsz, T = 4, 8
        traj = GRPOTrajectory(
            query_responses=torch.zeros(bsz, T),
            logprobs=torch.zeros(bsz, T),
            ref_logprobs=torch.zeros(bsz, T),
            advantages=torch.zeros(bsz),
            rewards=torch.zeros(bsz),
            successes=torch.zeros(bsz),
            masks=None,
            position_ids=torch.zeros(bsz, T, dtype=torch.long),
            response_padding_masks=torch.zeros(bsz, T, dtype=torch.bool),
            seq_lens=torch.full((bsz,), T, dtype=torch.long),
            answers=["a"] * bsz,
            # vllm_sampled_logprobs left at default None
        )
        sliced = _slice_trajectory(traj, 1, 3)
        self.assertIsNone(sliced.vllm_sampled_logprobs)


if __name__ == "__main__":
    unittest.main()

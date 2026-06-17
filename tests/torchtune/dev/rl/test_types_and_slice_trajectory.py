# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU drift-guard for GRPOTrajectory/GRPOStats fields and _slice_trajectory.

``_slice_trajectory`` (distributed.py) is generic: it iterates
``trajectory._fields`` and slices any Tensor or list field by ``[start:end]``,
passing everything else through. That generality means the contract lives in
the field SET, not in the slice code. Pinning the field list here catches the
silent-drift case where a field is added to (or removed from) GRPOTrajectory:
recipe sites that unpack the NamedTuple, and the per-step trajectory slicing in
gradient accumulation, both depend on the exact shape.

This also gives ``_slice_trajectory`` live CPU coverage — previously it was only
exercised by a now-deleted parked test file.

Pure CPU — no XPU, no distributed init.
"""
from __future__ import annotations

import torch

from torchtune.dev.rl.distributed import _slice_trajectory
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory


# --- field pinning ---

EXPECTED_TRAJECTORY_FIELDS = (
    "query_responses",
    "logprobs",
    "ref_logprobs",
    "advantages",
    "rewards",
    "successes",
    "masks",
    "position_ids",
    "response_padding_masks",
    "seq_lens",
    "answers",
    "prompt_embeds",
)

EXPECTED_STATS_FIELDS = (
    "loss",
    "policy_loss",
    "kl_loss",
    "ratios",
    "clipfrac",
    "approx_policy_kls",
    "metadata",
)


def test_trajectory_fields_exact():
    """If this fails, a field was added/removed/reordered. Verify every recipe
    unpack site and _slice_trajectory still handle the new shape, then update
    EXPECTED_TRAJECTORY_FIELDS."""
    assert GRPOTrajectory._fields == EXPECTED_TRAJECTORY_FIELDS


def test_stats_fields_exact():
    assert GRPOStats._fields == EXPECTED_STATS_FIELDS


def test_trajectory_all_fields_default_none():
    """Callers construct partial trajectories (e.g. before advantages exist),
    so every field must be optional."""
    t = GRPOTrajectory()
    for f in GRPOTrajectory._fields:
        assert getattr(t, f) is None


# --- _slice_trajectory behavior ---

def _make_traj(bsz=4, plen=5, rlen=3):
    total = plen + rlen
    return GRPOTrajectory(
        query_responses=torch.arange(bsz * total).reshape(bsz, total),
        logprobs=torch.zeros(bsz, rlen),
        ref_logprobs=torch.ones(bsz, rlen),
        advantages=torch.arange(bsz, dtype=torch.float),
        rewards=torch.zeros(bsz),
        successes=torch.ones(bsz),
        masks=None,  # often None on the no-mask path
        position_ids=torch.zeros(bsz, total, dtype=torch.long),
        response_padding_masks=torch.zeros(bsz, rlen, dtype=torch.bool),
        seq_lens=torch.full((bsz,), total, dtype=torch.long),
        answers=[str(i) for i in range(bsz)],
        prompt_embeds=None,
    )


def test_slice_tensor_and_list_fields():
    sliced = _slice_trajectory(_make_traj(bsz=4), 1, 3)
    assert sliced.query_responses.shape[0] == 2
    assert sliced.advantages.tolist() == [1.0, 2.0]
    assert sliced.answers == ["1", "2"]  # list sliced too


def test_slice_passes_none_through():
    sliced = _slice_trajectory(_make_traj(), 0, 2)
    assert sliced.masks is None
    assert sliced.prompt_embeds is None


def test_slice_every_field_handled_no_raise():
    """Smoke: slicing must not raise for any field. If a new non-Tensor,
    non-list field type is added, this surfaces it here rather than at runtime
    in the gradient-accumulation loop."""
    traj = _make_traj(bsz=6)
    for start, end in [(0, 1), (2, 5), (0, 6)]:
        s = _slice_trajectory(traj, start, end)
        assert len(s.answers) == end - start
        assert s.query_responses.shape[0] == end - start


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

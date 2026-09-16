# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pin the (currently inert) ``format_penalty`` branch in ``bioreason_reward_fn``.

``reward.py:201-203`` intends to penalize a completion that emits no GO terms::

    if not predicted:
        score = max(0.0, score - format_penalty)

But ``weighted_f_score(predicted=empty, ...)`` is already ``0.0`` — precision and
recall are both zero when nothing is predicted — so the expression is
``max(0.0, 0.0 - format_penalty) == 0.0`` for **every** penalty value. The branch
can never lower the score below what it already is.

Why that matters, measured on run 8829513 (steps 0-42, 1376 rollouts):
a zero-term completion and a completion whose GO terms all miss the ground truth
both score exactly ``0.0``. GRPO's advantage is a within-batch contrast of rewards,
so the two are **indistinguishable to the gradient** — there is no signal that
says "use the output format". Over 42 steps the fraction of rollouts emitting no
GO term at all rose 0.303 -> 0.358, driven by ``prose_no_GO_ids`` (134 -> 167):
the model answers in prose and never emits the machine-readable IDs the reward
parses. See
``memory/project_bioreason_grpo_reward_decay_is_zero_term_fraction_not_term_count_20260916.md``.

These tests do NOT assert the desired behavior — changing the reward mid-campaign
would break comparability with the in-flight run. They pin the *current* semantics
so that if someone later makes the penalty bite, it is a deliberate, visible change
and the affected runs can be dated.
"""

import pytest

torch = pytest.importorskip("torch")

from torchtune.dev.bioreason.reward import (  # noqa: E402
    bioreason_reward_fn,
    weighted_f_score,
)

_GT = "GO:0008152, GO:0036211, GO:0009987"


def test_weighted_f_score_of_empty_prediction_is_zero():
    """The precondition that makes the penalty unreachable."""
    assert weighted_f_score(set(), {"GO:0008152"}, beta=1.0) == 0.0


@pytest.mark.parametrize("penalty", [0.0, 0.1, 0.5, 0.9, 1.0, 10.0])
def test_format_penalty_is_inert_for_every_value(penalty):
    """No value of ``format_penalty`` changes a zero-term completion's reward."""
    rewards, _ = bioreason_reward_fn(
        ["a prose answer that never emits an identifier"],
        [_GT],
        format_penalty=penalty,
    )
    assert rewards.tolist() == [0.0], (
        f"format_penalty={penalty} produced {rewards.tolist()}; the branch is "
        "expected to be inert because max(0.0, 0.0 - p) == 0.0"
    )


def test_no_format_signal_between_wrong_terms_and_no_terms():
    """The gradient-relevant consequence: both collapse to the same reward.

    GRPO advantages are a contrast of rewards within a batch. Two failure modes
    that score identically cannot be told apart by the policy gradient.
    """
    rewards, _ = bioreason_reward_fn(
        [
            "a prose answer that never emits an identifier",
            "Answer: GO:9999998, GO:9999999",  # well-formed, entirely wrong
        ],
        [_GT, _GT],
    )
    no_terms, wrong_terms = rewards.tolist()
    assert no_terms == wrong_terms == 0.0, (
        "expected both to be 0.0 under current semantics; if this now differs, a "
        "format signal was introduced and run comparability must be re-dated"
    )


def test_correct_terms_still_score_above_zero():
    """Guard the test above against passing for the trivial reason."""
    rewards, _ = bioreason_reward_fn(["Answer: " + _GT], [_GT])
    assert rewards.tolist()[0] > 0.5


def test_has_pred_diagnostic_distinguishes_them_even_though_reward_does_not():
    """The information exists in diagnostics — it just never reaches the reward."""
    _, _, diag = bioreason_reward_fn(
        ["prose only", "Answer: GO:9999999"],
        [_GT, _GT],
        return_diagnostics=True,
    )
    assert diag["has_pred"].tolist() == [False, True]
    assert diag["pred_count"].tolist() == [0, 1]

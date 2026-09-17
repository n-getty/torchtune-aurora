# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for ``audit_behavior_logprobs``.

This function is itself an instrument, so the tests are calibration: they feed it
disagreements of KNOWN size -- including the two magnitudes the module docstring cites
as the decision boundaries (0.0739 recompute noise, 4.09x for the raw_logprobs bug) --
and assert it reports them. An instrument that has only ever been run on real data,
where the true answer is unknown, cannot be distinguished from one that is broken.
"""

import ast
import math
import pathlib

import pytest
import torch

from torchtune.dev.rl.behavior_logprobs import audit_behavior_logprobs, PAD_FILL

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
RECIPE = REPO_ROOT / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"


def test_identical_inputs_report_zero_disagreement():
    x = torch.randn(4, 16)
    s = audit_behavior_logprobs(x, x.clone())
    assert s["n_compared"] == 64
    assert s["max_abs"] == 0.0
    assert s["ratio_max"] == pytest.approx(1.0)
    assert s["bias"] == pytest.approx(0.0)


def test_recovers_a_known_constant_offset():
    """A uniform shift is the signature of a systematic bug (e.g. wrong mode)."""
    base = torch.randn(3, 8)
    offset = 0.25
    s = audit_behavior_logprobs(base + offset, base)
    assert s["mean_abs"] == pytest.approx(offset, abs=1e-5)
    assert s["bias"] == pytest.approx(offset, abs=1e-5)
    # bias == mean_abs (not ~0) is what separates a systematic offset from noise.
    assert s["ratio_max"] == pytest.approx(math.exp(offset), rel=1e-4)


def test_bias_is_signed_and_noise_averages_out():
    base = torch.zeros(200, 10)
    noisy = base.clone()
    noisy[::2] += 0.1
    noisy[1::2] -= 0.1
    s = audit_behavior_logprobs(noisy, base)
    assert s["mean_abs"] == pytest.approx(0.1, abs=1e-5)
    assert s["bias"] == pytest.approx(0.0, abs=1e-5)  # symmetric -> no bias


def test_calibrates_against_the_documented_healthy_noise_floor():
    """0.0739 is the recompute spread the module cites from a healthy run."""
    base = torch.zeros(100, 10)
    s = audit_behavior_logprobs(base + math.log(1.0739), base)
    assert s["ratio_max"] == pytest.approx(1.0739, rel=1e-3)


def test_calibrates_against_the_documented_raw_logprobs_bug():
    """4.09x is the worst-case IS error from serving unscaled raw_logprobs."""
    base = torch.zeros(100, 10)
    s = audit_behavior_logprobs(base + math.log(4.09), base)
    assert s["ratio_max"] == pytest.approx(4.09, rel=1e-3)
    # The two regimes must be far apart in the reported units, or the audit could
    # not be used to tell them apart on hardware.
    healthy = audit_behavior_logprobs(base + math.log(1.0739), base)
    assert s["ratio_max"] > 3 * healthy["ratio_max"]


def test_padding_mask_excludes_sentinel_positions():
    """PAD_FILL vs a real logprob is a meaningless difference that would dominate."""
    behavior = torch.zeros(2, 6)
    recomputed = torch.zeros(2, 6)
    behavior[:, 4:] = PAD_FILL  # sentinel
    recomputed[:, 4:] = -7.5  # arbitrary real value at a padded slot

    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 4:] = True

    unmasked = audit_behavior_logprobs(behavior, recomputed)
    masked = audit_behavior_logprobs(behavior, recomputed, padding_mask=mask)

    assert unmasked["max_abs"] > 8.0, "control: padding must dominate when included"
    assert masked["n_compared"] == 8
    assert masked["max_abs"] == pytest.approx(0.0)


def test_non_finite_values_are_dropped_not_propagated():
    behavior = torch.zeros(2, 4)
    recomputed = torch.zeros(2, 4)
    behavior[0, 0] = float("-inf")
    recomputed[1, 3] = float("nan")
    s = audit_behavior_logprobs(behavior, recomputed)
    assert s["n_compared"] == 6
    assert math.isfinite(s["max_abs"]) and s["max_abs"] == pytest.approx(0.0)


def test_fully_masked_batch_reports_zero_not_nan():
    """A fully-padded batch is legitimate; NaN stats would read as catastrophe."""
    x = torch.zeros(2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    s = audit_behavior_logprobs(x, x, padding_mask=mask)
    assert s["n_compared"] == 0
    assert s["ratio_max"] == 1.0
    assert all(math.isfinite(v) for v in s.values())


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        audit_behavior_logprobs(torch.zeros(2, 4), torch.zeros(2, 5))


# ── Recipe wiring guards ──────────────────────────────────────────────────────
# The helper being correct is worth nothing if the recipe never calls it, or calls it
# on a path the audit flag cannot reach. These pin the call site itself.


def _recipe_src() -> str:
    if not RECIPE.exists():
        pytest.skip(f"recipe not present: {RECIPE}")
    return RECIPE.read_text()


def test_recipe_calls_the_audit():
    src = _recipe_src()
    assert "audit_behavior_logprobs(" in src
    assert "BLP_AUDIT" in src, "the audit must emit a greppable log key"


def test_audit_flag_suppresses_the_skip_branch():
    """The audit needs BOTH values, so it must not take the policy-forward shortcut.

    Without this the audit would compare the behavior logprobs against themselves
    (or against None) and report a perfect match no matter how wrong they were --
    an instrument that always reads zero.
    """
    src = _recipe_src()
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.get_source_segment(src, node.test) or ""
        if "_behavior_logprobs is not None" not in test_src:
            continue
        if "_ppo_epochs == 1" not in test_src:
            continue
        found = True
        assert "_blp_audit_active" in test_src, (
            "the pi_old skip branch must be disabled while the audit is active, "
            f"got condition: {test_src}"
        )
    assert found, "could not locate the pi_old substitution branch in the recipe"


def test_audit_is_exception_wrapped():
    """A diagnostic must never be able to kill a multi-hour run."""
    src = _recipe_src()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = ast.get_source_segment(src, node) or ""
        if "audit_behavior_logprobs(" in body:
            assert node.handlers, "audit try block has no except handler"
            return
    pytest.fail("audit_behavior_logprobs call is not wrapped in try/except")

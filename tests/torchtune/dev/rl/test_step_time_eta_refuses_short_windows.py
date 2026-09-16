"""An ETA projection must refuse a window too short to extrapolate from.

THE BURN (2026-09-16). The rebase arm (8831916) had 7 TIMING lines. An inline OLS on its
6 warm steps read slope +17.5 s/step, t=+4.09, projecting ETA_100 at 41.2h against a 36h
wall -- "this job cannot reach step 100". The Phase 2 arm, same config, ran to completion
and reads +28.4 s/step (t=+1.96) on its OWN first 6 warm steps and -0.32 s/step over its
full 99. The early window is STEEPER on the arm that finished.

Per-step jitter on this workload is ~240s peak-to-peak with no trend, so a 6-point fit is
set by where the end points land, and extrapolating it ~94 steps multiplies that error by
~94. The t looked decisive because the residual sd came from the same 6 points that set
the slope. Acting on it would have meant a qalter, kill, or resubmit on noise.

Enforcement, per the burn->enforcement rule (a memory entry alone leaves it open):
step_time_eta.py refuses to call a sub-min-n slope a risk, and fits the SAME window on a
reference arm when given one. These tests are its calibration certificate.

The guard that matters most is NOT "does it refuse" -- a tool that always refuses is
useless and would pass a naive test. It is selftest 3: a REAL drift at n>=min_n must
still alarm. Both directions are asserted here.

Login-node tests: no XPU, no allocation, no job submission.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "experiments" / "bioreason" / "step_time_eta.py"


def _skip_if_absent():
    if not SCRIPT.exists():
        pytest.skip(f"{SCRIPT} not present (experiments/ is gitignored)")


def _run(*args, timeout=120):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def test_script_exists():
    _skip_if_absent()
    assert SCRIPT.exists()


def test_selftest_passes():
    """The calibration must separate short-window noise from real drift."""
    _skip_if_absent()
    r = _run("--selftest")
    assert r.returncode == 0, (
        "step_time_eta calibration FAILED -- do not trust any ETA verdict from it "
        f"until it passes:\n{r.stdout}\n{r.stderr}"
    )
    assert "SELFTEST PASSED" in r.stdout


def test_selftest_keeps_the_real_drift_case():
    """A guard that never alarms is not a guard.

    Deleting selftest 3 (or flipping its expectation) would let min-n swallow a genuine
    wall risk, which is the opposite failure and a more expensive one -- the job dies at
    the wall with no checkpoint past the last save.
    """
    _skip_if_absent()
    text = SCRIPT.read_text(errors="replace")
    assert "must be WALL RISK" in text, (
        "the real-drift selftest case was removed; without it the min-n guard could "
        "suppress every alarm and still pass its own calibration"
    )
    assert "missed a real wall risk" in text


def test_short_window_is_insufficient_not_ok_and_not_risk():
    """Three-valued outcome: 'not enough data' must not read as either verdict."""
    _skip_if_absent()
    r = _run("--selftest")
    assert "VERDICT: INSUFFICIENT" in r.stdout, (
        "a sub-min-n window must report INSUFFICIENT; collapsing it to OK hides a real "
        "drift, and collapsing it to WALL RISK is the burn this tool exists to prevent"
    )
    # All three verdicts must be reachable, or the tool is not actually deciding.
    for verdict in ("VERDICT: INSUFFICIENT", "VERDICT: OK", "VERDICT: WALL RISK"):
        assert verdict in r.stdout, f"{verdict} never occurs across the selftest cases"


def test_short_window_still_prints_the_projection():
    """Refusing to ACT on a number is not the same as hiding it."""
    _skip_if_absent()
    text = SCRIPT.read_text(errors="replace")
    assert "a slope-based projection here would read" in text, (
        "the refused projection must still be printed -- suppressing it invites the "
        "reader to recompute it by hand, which is exactly how the burn happened"
    )


def test_refusal_names_the_recheck_point():
    """A refusal without a next action gets routed around."""
    _skip_if_absent()
    r = _run("--selftest")
    assert "Re-check at step" in r.stdout
    assert "Do not qalter, kill, or resubmit" in r.stdout, (
        "the refusal must name the actions it is forbidding; 'insufficient data' alone "
        "does not stop someone from acting on the number printed above it"
    )


def test_reference_arm_window_is_fitted_on_the_same_n():
    """The comparison is only meaningful at the SAME window length."""
    _skip_if_absent()
    text = SCRIPT.read_text(errors="replace")
    assert "ols(ys[:n])" in text, (
        "the reference arm must be fitted over the treatment arm's first n warm steps; "
        "comparing a 6-point fit to a 99-point fit compares window length, not drift"
    )
    assert "same window" in text


def test_step_zero_is_excluded():
    """Step 0 carries model load and warmup; including it fakes a downward slope."""
    _skip_if_absent()
    text = SCRIPT.read_text(errors="replace")
    assert "if s > 0" in text, "step 0 must be dropped from the warm-step fit"


def test_missing_timing_lines_fails_loudly():
    """A log with no TIMING lines must not report an ETA over zero steps."""
    _skip_if_absent()
    text = SCRIPT.read_text(errors="replace")
    assert "no TIMING lines" in text

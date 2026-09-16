"""The trend-significance test must stay calibrated, and must stay honestly labelled.

THE BURN (2026-09-16). A moving-block bootstrap was written to check whether the
exact-match reward trend on the Phase 2 100-step run survives serial dependence. It
returned P(slope<=0) ~ 0.50 on constructed series with true slopes of 0.000, 0.010,
0.024 AND 0.050 -- the last with a naive OLS t of +14.8. Cause: it resampled blocks and
refit against a RENUMBERED x-axis, destroying the time ordering the statistic is about.
On its say-so the conclusion "the exact-match trend does not survive clustering" was
written down. It was an instrument artifact, and a real finding was nearly discarded.

A significance test is an instrument. These tests are its calibration certificate:

1. `--selftest` must pass -- the test separates constructed series with known answers.
2. The selftest's own expectations must be ACHIEVABLE. The weak case (slope/100=0.010,
   SNR 0.5) reads p=0.057 under naive OLS, which upper-bounds the power of any
   serial-dependence-robust test; demanding significance there would have meant
   loosening the test until the fixture passed. It is pinned as a POWER FLOOR instead:
   if it ever goes significant, the test has become anti-conservative.
3. The banned idiom (refitting against a renumbered axis) must not come back.
4. The output must not call OLS-on-step-means "clustered". Aggregating to one
   observation per step picks the right UNIT; it is not a correction for serial
   dependence. Conflating the two is what produced the mislabel in the first place.

These run on a login node -- no XPU, no allocation, no rollout data required.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "experiments" / "bioreason" / "reward_vs_exact_trend.py"


def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} missing"


def test_selftest_passes():
    """The calibration must actually separate known-signal from known-null."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--selftest"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, (
        "block-permutation calibration FAILED -- do not trust any trend verdict from "
        f"this script until it passes:\n{r.stdout}\n{r.stderr}"
    )
    assert "SELFTEST PASSED" in r.stdout
    # The null and the strong-signal cases are the two that matter most.
    assert "slope/100=0.000" in r.stdout and "slope/100=0.050" in r.stdout


def test_selftest_keeps_the_power_floor_case():
    """The underpowered case must remain, and must remain expected-NS.

    Deleting it (or flipping it to "sig") is how the ruler gets re-priced to save a
    verdict. Its whole job is to fail loudly if the test becomes anti-conservative.
    """
    text = SCRIPT.read_text(errors="replace")
    assert "POWER FLOOR" in text, (
        "the slope=0.010 power-floor case was removed from the selftest; it is the "
        "guard against making the test anti-conservative"
    )
    assert "(0.010, \"NS\"" in text or "(0.010, 'NS'" in text, (
        "the power-floor case must be expected NON-significant (naive OLS reads "
        "p=0.057 there too -- it is an underpowered draw, not a test defect)"
    )


def test_permutation_uses_a_fixed_x_axis():
    """The banned idiom: refitting a permuted series against a renumbered axis."""
    text = SCRIPT.read_text(errors="replace")
    assert "block_perm_p" in text, "the validated block-permutation test is gone"
    assert "FIXED" in text, (
        "block_perm_p must document that it permutes against a FIXED x axis -- "
        "permuting the axis instead is the inert-bootstrap bug"
    )
    # The correct implementation permutes y and reuses the same xs built once.
    assert "rng.shuffle(blocks)" in text
    assert "ols_t(xs, perm)" in text, (
        "the permuted series must be regressed against the ORIGINAL xs; building a "
        "fresh range() inside the loop is the bug this test exists to prevent"
    )


def test_ols_on_step_means_is_not_labelled_clustered():
    """Naming discipline: 'right unit' != 'serial-dependence correction'."""
    text = SCRIPT.read_text(errors="replace")
    assert "clustered to one observation per step" not in text, (
        "the trend header called OLS-on-step-means 'clustered', which is what let a "
        "naive t be reported as if it were corrected for serial dependence"
    )
    assert "OLS on step-means" in text


def test_shape_check_flags_sign_disagreement():
    """A V must not be reported as a trend.

    The real exact-match series declines for 50 steps (t=-2.32) then recovers
    (t=+1.91); its positive fitted slope describes no epoch of the run.
    """
    text = SCRIPT.read_text(errors="replace")
    assert "HALVES DISAGREE IN SIGN" in text, (
        "shape_report must warn when the two half-window fits disagree in sign"
    )
    assert "ta * tb < 0" in text


@pytest.mark.parametrize("flag", ["--selftest", "--acf", "--block"])
def test_documented_flags_exist(flag):
    text = SCRIPT.read_text(errors="replace")
    assert flag in text, f"{flag} is referenced in the docs/tests but not in the script"

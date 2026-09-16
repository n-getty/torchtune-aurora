"""The endpoint-eval feeder must take its arm as a PARAMETER, not a hardcoded path.

THE BURN (2026-09-16). Both `auto_feed_step100_eval.sh` and its v2 successor hardcode
`OUT=.../bioreason_32b_2n_b4g8_phase2_100step` and `JOB=8829513`. By the time two new
GRPO arms were running (rebase, per-group-advantage), neither feeder could ever fire for
them: each would sit in its poll loop waiting on a step-100 checkpoint that the arm it
was nominally watching never writes, and the failure is SILENT -- a feeder waiting and a
feeder watching the wrong directory look identical from outside.

The tempting repair is to copy v2 and edit the path, which is verbatim the banked burn
"audit a copied dispatcher's hardcoded PATHS" (a copied A/B baseline that pointed at the
wrong arm). v3 parameterises instead: ARM_OUT / ARM_JOB / ARM_STEP, all REQUIRED.

Required with no defaults is the load-bearing part. A default is how a feeder points at
last campaign's arm while looking correctly configured, so these tests assert both that
the parameters exist AND that they have no fallback value.

Login-node tests: no XPU, no allocation, no job submission.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
FEEDER = REPO / "experiments" / "bioreason" / "auto_feed_endpoint_eval_v3.sh"


def _skip_if_absent():
    if not FEEDER.exists():
        pytest.skip(f"{FEEDER} not present (experiments/ is gitignored)")


def test_feeder_exists():
    _skip_if_absent()
    assert FEEDER.exists()


def test_syntax_is_valid():
    """bash -n. v3 tripped bash 4.4 mis-parsing '(' inside ${var:?...}."""
    _skip_if_absent()
    r = subprocess.run(["bash", "-n", str(FEEDER)], capture_output=True, text=True)
    assert r.returncode == 0, f"syntax error:\n{r.stderr}"


@pytest.mark.parametrize("var", ["ARM_OUT", "ARM_JOB", "ARM_STEP"])
def test_arm_identity_is_a_required_parameter(var):
    """Each identity field must be required (:?) -- never :- or = with a default."""
    _skip_if_absent()
    text = FEEDER.read_text(errors="replace")
    assert f"${{{var}:?" in text, (
        f"{var} must be REQUIRED via ${{{var}:?...}}. A default silently points the "
        f"feeder at some other campaign's arm while looking configured."
    )
    assert f"${{{var}:-" not in text, f"{var} must not have a :- fallback"


def test_no_hardcoded_arm_output_dir():
    """The v1/v2 defect: the arm baked into the script body."""
    _skip_if_absent()
    body = [
        ln for ln in FEEDER.read_text(errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    text = "\n".join(body)
    assert "phase2_100step" not in text, (
        "a specific arm's output dir is hardcoded in the feeder body again"
    )
    assert "JOB=8829513" not in text, "a specific PBS job id is hardcoded again"


def test_refuses_to_run_without_parameters():
    """Missing identity must exit nonzero, not wait on a wrong/absent path."""
    _skip_if_absent()
    env = {k: v for k, v in os.environ.items()
           if k not in ("ARM_OUT", "ARM_JOB", "ARM_STEP")}
    r = subprocess.run(["bash", str(FEEDER)], capture_output=True, text=True,
                       timeout=60, env=env)
    assert r.returncode != 0, "feeder ran with no arm specified"


def test_fails_fast_on_nonexistent_arm_dir(tmp_path):
    """A typo'd ARM_OUT must fail immediately, not poll for hours."""
    _skip_if_absent()
    env = dict(os.environ)
    env.update({"ARM_OUT": str(tmp_path / "NOPE"), "ARM_JOB": "1", "ARM_STEP": "100"})
    r = subprocess.run(["bash", str(FEEDER)], capture_output=True, text=True,
                       timeout=60, env=env)
    assert r.returncode != 0
    assert "does not exist" in (r.stdout + r.stderr)


def test_still_evaluates_a_snapshot_not_the_live_dir():
    """v2 hardening that must survive: never eval the dir the trainer is writing."""
    _skip_if_absent()
    text = FEEDER.read_text(errors="replace")
    assert "_snapshot" in text
    assert "CKPT=$SNAP" in text, (
        "the submitted arms must reference the SNAPSHOT; pointing them at $LIVE lets "
        "the trainer overwrite the weights mid-eval"
    )


def test_still_tolerates_one_empty_qstat():
    """v2 hardening that must survive: one transient qstat failure != job death."""
    _skip_if_absent()
    text = FEEDER.read_text(errors="replace")
    assert "DEAD_CONFIRMATIONS" in text and "dead_streak" in text, (
        "the consecutive-failure requirement is gone; a single PBS hiccup would "
        "declare the job dead and leave the endpoint arms unsubmitted"
    )


def test_still_verifies_both_projectors():
    """v2 hardening that must survive: an adapter alone evaluates the wrong model."""
    _skip_if_absent()
    text = FEEDER.read_text(errors="replace")
    assert "protein_projection.pt" in text and "go_projection.pt" in text

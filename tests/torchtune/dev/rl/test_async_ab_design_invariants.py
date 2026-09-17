# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Design invariants of the debug-queue async-vs-sync A/B.

WHY THESE ARE TESTS AND NOT COMMENTS. This A/B exists to replace an n=1 number.
Its validity rests on four properties that are invisible in any single run's
output -- a violated one produces a clean-looking result that is simply wrong:

  1. ARMS DIFFER ONLY IN THE TREATMENT. Every topology/routing knob must be
     byte-identical between the arm dispatch and the production reference. One
     drifted knob and the contrast measures that knob
     (memory/feedback_launcher_config_drift).
  2. THE ORDER ALTERNATES. On debug each arm is its own job on its own node
     pair, so if async were always submitted first, "async" and "got the earlier
     node pool" would be the same variable.
  3. THE ROUND IS THE UNIT. Steps within an arm share a node pair and are
     correlated; pooling them as independent quotes a CI ~sqrt(3) too narrow.
  4. THE PRE-REGISTRATION IS NOT RE-TUNED. gen/total stay DIFFERENCE tests and
     grpo stays an EQUIVALENCE check against +/-10%, whatever the data say.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
BR = ROOT / "experiments" / "bioreason"
ARM_DISPATCH = BR / "dispatch_2n_async_arm.sh"
REF_DISPATCH = BR / "dispatch_2n_phase2_100step.sh"
PBS = BR / "pbs_2n_async_arm.sh"
FEEDER = BR / "feed_async_ab_rounds.sh"
CONTRAST = BR / "async_multiround_contrast.py"

# Deliberate per-arm / per-experiment differences. Anything NOT here that
# differs between the arm dispatch and the production reference is drift.
INTENTIONAL = {
    "CONFIG",         # the treatment itself
    "ASYNC",          # the treatment itself
    "OUTPUT_DIR",     # arms must not share a directory
    "NSTEPS",         # 4 here vs 100 in production
    "SAVE_EVERY_N_STEPS",   # a throughput probe writes no 32B checkpoints
    "EXTRA_OVERRIDES",      # reference appends a caller passthrough
    "TORCHTUNE_DUMP_ROLLOUTS",        # off here; identical across arms
    "TORCHTUNE_VLLM_SEQS_PER_ENGINE",  # literal here vs :- default; same value
    "TORCHTUNE_WSYNC_RESET_RUNNING_REQUESTS",  # stated explicitly here
}

EXPORT_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\S*)')


def _exports(path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    out = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s.startswith("export "):
            continue
        for m in EXPORT_RE.finditer(s[len("export "):]):
            out[m.group(1)] = m.group(2).strip('"')
    return out


def test_arm_dispatch_sets_every_knob_the_reference_sets():
    """A new branch must set what its siblings set -- the invariant is the
    sibling INTERSECTION, not whatever this script happens to remember."""
    ref, arm = _exports(REF_DISPATCH), _exports(ARM_DISPATCH)
    missing = sorted(set(ref) - set(arm) - INTENTIONAL)
    assert not missing, (
        "the A/B arm dispatch omits knobs the production reference sets: "
        f"{missing}. An omitted knob falls back to a launcher default and the "
        "contrast silently measures that instead of async."
    )


def test_no_unintended_value_drift_between_arm_and_reference():
    ref, arm = _exports(REF_DISPATCH), _exports(ARM_DISPATCH)
    drift = {
        k: (ref[k], arm[k])
        for k in set(ref) & set(arm)
        if ref[k] != arm[k] and k not in INTENTIONAL
    }
    assert not drift, f"unintended knob drift vs the production reference: {drift}"


def test_submission_order_alternates_across_rounds():
    """If async always went first, arm and node-pool draw would be the same
    variable and no amount of replication would separate them."""
    if not FEEDER.exists():
        pytest.skip("feeder not present")
    text = FEEDER.read_text()
    m = re.search(r"PLAN=\((.*?)\)", text, re.S)
    assert m, "no PLAN array in the feeder"
    entries = re.findall(r'"(\d+):(async|sync)"', m.group(1))
    assert entries, "PLAN parsed but empty"

    first_by_round = {}
    for rnd, arm in entries:
        first_by_round.setdefault(rnd, arm)
    firsts = list(first_by_round.values())
    assert len(set(firsts)) == 2, (
        f"every round submits {firsts[0]} first ({firsts}); arm is confounded "
        "with submission order and node-pool draw."
    )
    # Each round must contain BOTH arms, or it contributes no paired ratio.
    for rnd in first_by_round:
        arms = {a for r, a in entries if r == rnd}
        assert arms == {"async", "sync"}, f"round {rnd} has arms {arms}"


def test_feeder_can_resume_without_duplicating_submitted_arms():
    """The feeder is long-lived and WILL be restarted (it already was, after a
    queue-semantics bug). Restarting from index 0 resubmits arms that are already
    queued, and the index then holds two rows for one round:arm -- which the
    contrast counts as extra rounds, i.e. fabricated replication."""
    if not FEEDER.exists():
        pytest.skip("feeder not present")
    text = FEEDER.read_text()
    assert "FEED_PLAN_START" in text, (
        "the feeder has no resume point; a restart duplicates already-submitted arms."
    )
    body = [l for l in text.splitlines() if not l.strip().startswith("#")]
    assert any("-lt \"${FEED_PLAN_START}\"" in l for l in body), (
        "FEED_PLAN_START is declared but never used to skip submitted entries."
    )


def test_contrast_treats_the_round_as_the_unit():
    """The primary CI must resample ROUNDS. Steps inside one arm share a node
    pair, a server placement and a fabric route."""
    src = CONTRAST.read_text()
    assert "round_ratios" in src and "ci95_round_cluster" in src
    assert "ci95_step_optimistic" in src, (
        "the step-level CI must still be reported, but labelled optimistic -- "
        "hiding it invites someone recomputing it and quoting it as the result."
    )
    tree = ast.parse(src)
    names = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"round_ratios", "step_ratios", "boot_ci"} <= names


def test_preregistration_is_inherited_not_reinvented():
    """grpo must remain an EQUIVALENCE check at +/-10%; gen/total differences."""
    src = CONTRAST.read_text()
    assert "EQUIV_BAND" in src and "from async_paired_contrast import" in src, (
        "the band must be imported from the original tool, not redeclared -- a "
        "second copy is a second thing to quietly widen after seeing the data."
    )
    assert "EQUIV_BAND = " not in src, (
        "async_multiround_contrast redefines EQUIV_BAND; it must import it."
    )
    paired = (BR / "async_paired_contrast.py").read_text()
    m = re.search(r"EQUIV_BAND\s*=\s*([\d.]+)", paired)
    assert m and abs(float(m.group(1)) - 0.10) < 1e-9, (
        f"the pre-registered equivalence band changed to {m and m.group(1)}. It "
        "was fixed at 0.10 BEFORE the data existed and must not be re-tuned."
    )


def test_contrast_verifies_the_two_arms_of_a_round_ran_the_same_config():
    """Pairing is by ROUND NUMBER alone.

    Eight arms are dispatched over hours by a feeder into a shared index, from
    scripts that get edited between campaigns. Nothing in the dispatch compares
    one arm to the other, so an arm that ran on a different base checkpoint or a
    different batch size would be paired, averaged in, and produce a clean
    number for the wrong quantity. The readout must re-derive what each arm
    actually ran (from its own resolved-config dump) and REFUSE rather than pool.
    """
    src = CONTRAST.read_text()
    tree = ast.parse(src)
    names = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"parse_resolved_config", "config_mismatch"} <= names, (
        "the contrast does not read back each arm's resolved config; a round "
        "whose arms ran different configurations would be pooled silently."
    )
    assert "config_problems" in src and "config_verified" in src, (
        "a config mismatch must both REFUSE and be distinguishable from a check "
        "that never ran -- an unchecked gate must not read as a passing one."
    )


def test_config_exemptions_are_computed_not_an_allow_list():
    """An allow-list only covers the drift someone thought of.

    The legitimate per-arm differences (each arm's own output_dir, its own vLLM
    host) must be normalised away by SUBSTITUTION, so that any remaining
    difference is flagged -- including in keys nobody enumerated. The treatment
    keys are the one deliberate exception and are small and named.
    """
    src = CONTRAST.read_text()
    assert "<OUTPUT_DIR>" in src and "<HOST>" in src, (
        "per-arm paths/hosts must be substituted, not skipped by key name."
    )
    # The named exceptions must be the TREATMENT only. If this set grows to
    # include learning knobs, the gate has been widened to fit the data.
    m = re.search(r"TREATMENT_KEYS = \{(.*?)\}", src, re.S)
    assert m, "TREATMENT_KEYS not found"
    keys = set(re.findall(r'"([^"]+)"', m.group(1)))
    forbidden = {
        k for k in keys
        if any(t in k for t in ("batch", "lr", "seed", "temperature", "grpo_samples",
                                "base_model", "epsilon", "kl_coeff", "max_gen"))
    }
    assert not forbidden, (
        f"learning/topology knobs exempted from the drift check: {forbidden}. "
        "Those are exactly the differences that would invalidate the contrast."
    )


def test_contrast_refuses_rather_than_guesses():
    """Three-valued exit: blind must be distinguishable from fail."""
    src = CONTRAST.read_text()
    for token in ("REFUSED", "BLIND", '"OK": 0'):
        assert token in src, f"missing {token} in the contrast exit map"
    assert "MIN_ROUNDS" in src


def test_contrast_selftest_passes():
    """A tool built to fix a bug can be the bug. Calibrate it on constructed
    answers -- including a pure node-variance case it must NOT resolve."""
    r = subprocess.run(
        [sys.executable, str(CONTRAST), "--selftest"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, f"contrast selftest failed:\n{r.stdout}\n{r.stderr}"
    assert "SELFTEST PASS" in r.stdout
    # Pin the specific cases that protect this design, so the selftest cannot be
    # thinned to only the easy cases later.
    for case in (
        "node-variance-only case NOT resolved",
        "a different base checkpoint is caught as drift",
        "a knob present in one arm only is caught as drift",
        "a round whose arms disagree on config is REFUSED",
        "no configs supplied is reported unverified, not passed",
        "parser handles the real dump grammar and stops at its end",
    ):
        assert case in r.stdout, f"the contrast selftest no longer covers: {case}"


def test_pbs_arm_script_targets_debug_two_nodes():
    text = PBS.read_text()
    assert re.search(r"^#PBS -q debug$", text, re.M), "arm job must use the debug queue"
    assert re.search(r"^#PBS -l select=2$", text, re.M)
    assert re.search(r"^#PBS -l walltime=01:00:00$", text, re.M)
    assert "experiments/bioreason/logs/" in text, "PBS -o/-e must land in the subdir"


def test_no_ls_t_log_selection_anywhere_in_the_ab():
    """`ls -t` + concurrent rounds writing one directory = cross-arm mixup."""
    for p in (ARM_DISPATCH, PBS, FEEDER):
        for line in p.read_text().splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue  # prose warning ABOUT ls -t is the point
            assert "ls -t" not in s, f"{p.name} selects a log with `ls -t`: {s}"


def test_arm_identity_is_verified_from_the_run_not_asserted():
    """A mislabeled arm inverts the A/B while every other check stays green."""
    text = ARM_DISPATCH.read_text()
    assert "MISLABELED ARM" in text and "_arm_ok" in text, (
        "the dispatch must confirm from the run's own output which config it "
        "loaded, and refuse to index the row if it disagrees."
    )
    assert "exit 4" in text


def test_exit_status_is_not_read_through_a_pipe():
    """`cmd | tail` then `$?` reads tail's status -- always 0 -- flattening the
    three-valued exit to OK. This is a burn that already happened once."""
    for p in (PBS, FEEDER):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "$?" not in s:
                continue
            assert "|" not in s.split("$?")[0], (
                f"{p.name}:{i} reads $? on a pipeline: {s}"
            )

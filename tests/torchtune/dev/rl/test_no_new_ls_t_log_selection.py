"""Ratchet: no NEW launcher may select a train log with `ls -t`.

The burn (memory/feedback_concurrency_turns_ls_t_into_cross_arm_mixup): two 2N arms
ran concurrently and both graded "the newest train_mpiexec_*_single.log" in a SHARED
directory. Each therefore reported the OTHER arm's verdict under its own name. A
readout that names the wrong run is worse than no readout -- it looks authoritative.

The fix, validated against real runlogs, is to parse the launcher's own echo:

    _train_log=$(grep -oE 'train log: [^ )]+' "$runlog" | tail -1 | sed 's/^train log: //')

`[^ )]+` (not `[^ ]+`) because the launcher also prints "(train log: <path>)" and a
greedy class swallows the ")" -- the earlier fix needed a fix, and that was caught only
because validation asserted the parsed path EXISTS rather than that the parse returned
some string. Hence both checks live here.

Why a ratchet and not a blanket ban: ~37 single-use tune arms from 2026-09-15 still
carry the old idiom. They are dead, rewriting them is churn with its own risk, and a
test that fails on all of them would just be disabled. Freezing them as a known set
means the count can only go DOWN. Adding a new dispatcher with `ls -t` fails CI; the
author is pointed at the validated parse.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "experiments" / "bioreason"

# Matches a LIVE (non-comment) `ls -t` that selects a train/run log.
_LS_T = re.compile(r"^[^#\n]*\bls -t\b[^\n]*(train_mpiexec|run_bioreason)[^\n]*\.log")

# Scripts that predate the fix and are no longer dispatched. FROZEN: this list may
# shrink, never grow. Do not add to it -- use the validated parse instead.
LEGACY_ALLOWED = {
    "ab_readout.sh",
    "batch_go_pred_smoke.sh",
    "chain_gopred_4n_debugscaling.sh",
    "chain_gopred_debugscaling.sh",
    "dispatch_2n_b4g8_100step.sh",
    "dispatch_2n_b4g8_fbs2_ref3_maxseq40_2step.sh",
    "dispatch_2n_b4g8_fbs3_lensort.sh",
    "dispatch_2n_b4g8_fbs3_retest.sh",
    "dispatch_2n_b4g8_fbs4_n6.sh",
    "dispatch_2n_b4g8_gate_v2.sh",
    "dispatch_2n_b4g8_refprefix.sh",
    "dispatch_2n_b4g8_tune_choicesplit.sh",
    "dispatch_2n_b4g8_tune_combo.sh",
    "dispatch_2n_b4g8_tune_concentrate.sh",
    "dispatch_2n_b4g8_tune_fbs3.sh",
    "dispatch_2n_b4g8_tune_ofi.sh",
    "dispatch_2n_b4g8_tune_ofi_fbs3.sh",
    "dispatch_2n_b4g8_tune_reffbs4.sh",
    "dispatch_2n_b4g8_tune_single.sh",
    "dispatch_2n_bsweep_intercept.sh",
    "pbs_2n_async_probe.sh",
    "pbs_2n_bsweep.sh",
    # pbs_2n_fbs3_lensort.sh removed 2026-09-16: both of its `ls -t | head -1` uses
    # (baseline selection AND the log it graded) are now content-derived. The ratchet
    # caught the stale entry on the same run that fixed the script.
    "pbs_2n_fbs3_retest.sh",
    "pbs_2n_fbs4_n6.sh",
    "pbs_2n_grpo_b4g8_canary.sh",
    "pbs_2n_grpo_b4g8_gate.sh",
    "pbs_2n_refprefix.sh",
    "pbs_2n_tune_choicesplit.sh",
    "pbs_2n_tune_combo.sh",
    "pbs_2n_tune_concentrate.sh",
    "pbs_2n_tune_fbs3.sh",
    "pbs_2n_tune_ofi.sh",
    "pbs_2n_tune_ofi_fbs3.sh",
    "pbs_2n_tune_reffbs4.sh",
    "pbs_2n_tune_single.sh",
}

# Scripts fixed to the validated parse. These must STAY fixed.
FIXED = [
    "dispatch_2n_phase2_100step.sh",
    "pbs_2n_phase2_100step.sh",
    "dispatch_2n_phase4_pergroup_adv.sh",
    "dispatch_2n_rebase_step1050.sh",
    "dispatch_2n_async_probe.sh",
]

VALIDATED_PARSE = "grep -oE 'train log: [^ )]+'"


def _offenders():
    out = []
    for path in sorted(EXP.glob("*.sh")):
        text = path.read_text(errors="replace")
        if any(_LS_T.match(line) for line in text.splitlines()):
            out.append(path.name)
    return out


def test_no_new_ls_t_log_selection():
    """A newly added launcher must not select a train log with `ls -t`."""
    new = sorted(set(_offenders()) - LEGACY_ALLOWED)
    assert not new, (
        "These launchers select a train log with `ls -t`, which under any concurrent "
        "2N arm grades the OTHER arm's log under this job's name:\n  "
        + "\n  ".join(new)
        + "\n\nUse the validated parse instead (and assert the path EXISTS):\n"
        f"    _train_log=$({VALIDATED_PARSE} \"$runlog\" | tail -1 | sed 's/^train log: //')\n"
        "    [ -n \"$_train_log\" ] && [ ! -f \"$_train_log\" ] && _train_log=\"\"\n"
        "See memory/feedback_concurrency_turns_ls_t_into_cross_arm_mixup."
    )


def test_legacy_list_is_a_ratchet_not_a_wishlist():
    """Every frozen name must still exist and still offend, so the list can only shrink.

    Without this, a deleted or already-fixed script would linger in LEGACY_ALLOWED and
    silently re-permit the idiom if someone later recreated that filename.
    """
    offenders = set(_offenders())
    stale = sorted(n for n in LEGACY_ALLOWED if n not in offenders)
    assert not stale, (
        "LEGACY_ALLOWED lists scripts that no longer offend (fixed or deleted). "
        "Remove them -- the ratchet must only tighten:\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize("name", FIXED)
def test_fixed_scripts_use_the_validated_parse(name):
    """Regression: the scripts we fixed must keep the parse AND the existence check."""
    path = EXP / name
    assert path.exists(), f"{path} missing"
    text = path.read_text(errors="replace")

    assert VALIDATED_PARSE in text, (
        f"{name} lost the validated train-log parse. Expected: {VALIDATED_PARSE}"
    )
    # `[^ ]+` without the ")" is the bug-in-the-fix: it eats the trailing paren.
    assert "grep -oE 'train log: [^ ]+'" not in text, (
        f"{name} uses the greedy class `[^ ]+`, which swallows the ')' in the "
        "launcher's \"(train log: <path>)\" form. Use `[^ )]+`."
    )
    # A parse that returns a nonexistent string must not be graded.
    assert "! -f" in text, (
        f"{name} parses the train log but never asserts the file EXISTS. That check is "
        "the only reason the greedy-class bug was caught; keep it."
    )


@pytest.mark.parametrize("name", FIXED)
def test_fixed_scripts_are_syntactically_valid(name):
    r = subprocess.run(
        ["bash", "-n", str(EXP / name)], capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, f"bash -n failed for {name}:\n{r.stderr}"


def test_detector_actually_fires(tmp_path):
    """Negative control: the regex must catch the idiom it is meant to catch.

    A ratchet whose detector silently matches nothing passes forever while the burn
    stays open. Construct the offending line and prove it is flagged -- and that the
    validated replacement is NOT.
    """
    bad = '_tl=$(ls -t "$TT"/experiments/bioreason/train_mpiexec_*_single.log 2>/dev/null | head -1)'
    good = "_tl=$(grep -oE 'train log: [^ )]+' \"$runlog\" | tail -1 | sed 's/^train log: //')"
    commented = "# `ls -t train_mpiexec_*_single.log | head -1` grabs the newest log"

    assert _LS_T.match(bad), "detector failed to flag the known-bad idiom"
    assert not _LS_T.match(good), "detector false-positives on the validated parse"
    assert not _LS_T.match(commented), (
        "detector flags a COMMENT describing the idiom; the fixed scripts all document "
        "the burn in prose, so this would fail them for explaining themselves"
    )

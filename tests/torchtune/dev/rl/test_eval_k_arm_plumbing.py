# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU test: the eval ranking arm (`--k`) is plumbed, defaulted safe, and recorded.

`--k` selects how F_max is scored. At k=1 every predicted term carries confidence 1.0
(flat); at k>1 the harness draws k temperature samples and ranks terms by frequency
across them. Frequency ranking is worth **+0.0349 F_max on its own**
(memory/project_bioreason_freq_ranking_beats_flat_confidence_20260915.md), against a
measured step-0 baseline spread of **0.0043**
(memory/project_bioreason_grpo_step0_fmax_baseline_3reps_20260916.md).

So a k mismatch between two endpoints is worth ~8x the noise band: differencing a k=8
step-100 against a k=1 step-0 manufactures a training effect bigger than any plausible
real one, from a scoring change alone. Three things have to hold, and this file pins
each:

1. The shared sub-launcher actually PASSES `--k` through. Until 2026-09-16 it did not,
   so every eval was silently locked at the argparse default and the arm could not be
   varied at all even when a caller asked for it.
2. The default stays 1. It is a shared sub-launcher: a default flipped here flips every
   caller at once, the exact drift class that killed 12/12 eval engines on 09-15
   (memory/feedback_shared_sublauncher_default_is_a_drift_surface_20260916.md).
3. `coverage.txt` RECORDS the k used. An unstamped k cannot be checked after the fact,
   which is how a valid prior F_max was first wrongly rejected and then wrongly
   re-accepted (memory/feedback_fmax_reuse_requires_prompt_distribution_check_20260915.md).

Pure text inspection of the launcher -- no PBS, no XPU, no vLLM.
"""
import os
import re

import pytest

_HERE = os.path.dirname(__file__)
_EXP = os.path.abspath(os.path.join(_HERE, "../../../../experiments/bioreason"))
_LAUNCHER = os.path.join(_EXP, "pbs_2n_eval_vllm_tp2.sh")


@pytest.fixture(scope="module")
def launcher() -> str:
    if not os.path.exists(_LAUNCHER):
        pytest.skip(f"{_LAUNCHER} not present")
    with open(_LAUNCHER) as f:
        return f.read()


class TestKIsPlumbed:
    def test_k_is_passed_to_eval_script(self, launcher):
        """The eval invocation must actually forward --k.

        Without this the EVAL_K variable is decorative: callers set it, nothing reads
        it, and every run silently scores at the argparse default.
        """
        assert re.search(r"--k\s+\"?\$\{?EVAL_K", launcher), (
            "pbs_2n_eval_vllm_tp2.sh does not pass --k $EVAL_K to eval_cafa_fmax.py; "
            "the ranking arm cannot be varied and callers setting EVAL_K are ignored"
        )

    def test_eval_k_has_a_default(self, launcher):
        """Unset EVAL_K under `set -u` would abort the job at the client-launch line."""
        assert re.search(r"EVAL_K=\$\{EVAL_K:-", launcher), (
            "EVAL_K needs a ${EVAL_K:-...} default; the launcher runs under set -u, so "
            "an unset variable aborts at shard launch"
        )

    def test_default_k_is_one(self, launcher):
        """k=1 (flat confidence) is the primary arm. Flipping this is a fleet-wide change."""
        m = re.search(r"EVAL_K=\$\{EVAL_K:-([^}]*)\}", launcher)
        assert m, "EVAL_K default not found"
        assert m.group(1).strip() == "1", (
            f"EVAL_K default is {m.group(1)!r}, expected '1'. This is a SHARED "
            "sub-launcher: changing the default silently re-scores every caller's F_max "
            "by ~+0.0349, which is ~8x the 0.0043 baseline spread. If a k>1 arm is "
            "wanted, pass EVAL_K from the specific caller -- do not move the default."
        )

    def test_default_matches_eval_script_default(self, launcher):
        """Launcher default and script default must agree, or 'unset' means two things."""
        script = os.path.join(_EXP, "eval_cafa_fmax.py")
        if not os.path.exists(script):
            pytest.skip("eval_cafa_fmax.py not present")
        with open(script) as f:
            src = f.read()
        m = re.search(r"add_argument\(\s*[\"']--k[\"'].*?default=(\d+)", src, re.S)
        assert m, "could not find --k default in eval_cafa_fmax.py"
        launcher_default = re.search(r"EVAL_K=\$\{EVAL_K:-([^}]*)\}", launcher).group(1).strip()
        assert launcher_default == m.group(1), (
            f"launcher default k={launcher_default} disagrees with eval_cafa_fmax.py "
            f"default k={m.group(1)}; a run that omits EVAL_K would score differently "
            "depending on which entry point was used"
        )


class TestKIsRecorded:
    def test_coverage_stamps_eval_k(self, launcher):
        """coverage.txt is the only durable record of how a completed run was scored."""
        assert re.search(r"echo\s+\"eval_k:", launcher), (
            "coverage.txt does not stamp eval_k; without it a later reader cannot tell "
            "whether two COMPLETE runs used the same ranking arm, and the launcher flag "
            "alone is not authoritative (it records what was requested, not what ran)"
        )

    def test_eval_k_stamp_is_inside_the_coverage_block(self, launcher):
        """The stamp must land in coverage.txt, not just print to the job log."""
        stamp = launcher.index('echo "eval_k:')
        redirect = launcher.index('} > "$OUT/coverage.txt"')
        provenance = launcher.index("comparability provenance")
        assert provenance < stamp < redirect, (
            "the eval_k echo must sit inside the brace group redirected to "
            "coverage.txt, alongside the other comparability provenance lines"
        )

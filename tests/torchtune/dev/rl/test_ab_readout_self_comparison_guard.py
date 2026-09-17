# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Guards for experiments/bioreason/ab_readout.sh, the shared A/B readout.

Two bugs motivate this file, both found on 2026-09-15:

1. **Self-comparison.** A PBS wrapper that resolves its baseline with
   `ls -t ... | head -1` *after* the dispatch returns picks up the run's own log and
   compares it against itself, reporting a flat 0% delta that looks like a clean
   null result. The helper refuses this.

2. **The guard was defeated by a relative path.** The first version compared the
   caller's string to an absolute glob result; a caller passing
   `experiments/bioreason/x.log` slipped straight through. The guard now
   canonicalizes both sides. This is the case worth pinning: a comparison guard that
   silently fails is worse than none, because the output still looks authoritative.

Shell, not Python, so these drive the real script via bash.
"""
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
HELPER = REPO / "experiments" / "bioreason" / "ab_readout.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash unavailable"
)


def _fake_tree(tmp_path, names):
    """A minimal TT with train logs, newest last in `names`."""
    d = tmp_path / "experiments" / "bioreason"
    d.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    # stub check_run_health so the readout runs without the real checker
    hc = tmp_path / "scripts" / "check_run_health.sh"
    hc.write_text("#!/bin/bash\necho STUB_HEALTH \"$@\"\nexit 0\n")
    hc.chmod(0o755)
    paths = []
    for i, n in enumerate(names):
        p = d / n
        p.write_text("TIMING step=1  total=1.0s  gen=1.0s  grpo=1.0s\n")
        # ls -t ordering follows mtime; make each later file strictly newer
        import os
        os.utime(p, (1_700_000_000 + i * 10, 1_700_000_000 + i * 10))
        paths.append(p)
    return paths


def _run(tmp_path, baseline):
    script = textwrap.dedent(
        f"""
        export TT={tmp_path}
        source {HELPER}
        ab_readout "{baseline}"
        """
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60
    ).stdout


def test_helper_exists_and_parses():
    assert HELPER.exists(), f"{HELPER} missing"
    r = subprocess.run(["bash", "-n", str(HELPER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_absolute_self_comparison_is_refused(tmp_path):
    logs = _fake_tree(tmp_path, ["train_mpiexec_1_single.log", "train_mpiexec_2_single.log"])
    out = _run(tmp_path, str(logs[-1]))  # newest == the run's own log
    assert "self-comparison" in out


def test_relative_self_comparison_is_refused(tmp_path):
    """The regression that motivated ab_canon: same file, different spelling."""
    logs = _fake_tree(tmp_path, ["train_mpiexec_1_single.log", "train_mpiexec_2_single.log"])
    rel = f"{tmp_path}/experiments/bioreason/../bioreason/{logs[-1].name}"
    out = _run(tmp_path, rel)
    assert "self-comparison" in out, (
        "a non-canonical path to the SAME file defeated the guard; the run would be "
        "silently compared against itself"
    )


def test_distinct_baseline_proceeds_to_compare(tmp_path):
    logs = _fake_tree(tmp_path, ["train_mpiexec_1_single.log", "train_mpiexec_2_single.log"])
    out = _run(tmp_path, str(logs[0]))
    assert "self-comparison" not in out
    assert "length-normalized A/B" in out


def test_missing_baseline_says_unverified_not_clean(tmp_path):
    """An uncontrolled cell must not read as a passing one."""
    _fake_tree(tmp_path, ["train_mpiexec_1_single.log"])
    out = _run(tmp_path, "")
    assert "SKIPPED" in out and "UNVERIFIED" in out


def test_no_log_at_all_is_reported(tmp_path):
    (tmp_path / "experiments" / "bioreason").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    out = _run(tmp_path, "")
    assert "never started" in out

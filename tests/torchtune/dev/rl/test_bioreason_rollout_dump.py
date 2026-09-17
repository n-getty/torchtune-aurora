# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe tests for the BioReason GRPO rollout dump.

Motivating gap (2026-09-15): the recipe decodes B*G completions for the reward and
discards them, leaving only one truncated SAMPLE_RESPONSE per step. That made the
group-frequency F_max test impossible to run on real rollouts — it had to use a
proxy built from repeated evals of a single checkpoint. See
memory/project_bioreason_freq_ranking_beats_flat_confidence_20260915.md.

These tests pin the two properties that matter: the dump is OFF unless explicitly
enabled, and it can never raise into the training loop.
"""
import json
import os

import pytest

from torchtune.dev.bioreason.rollout_dump import dump_rollout_groups, rollout_dump_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TORCHTUNE_DUMP_ROLLOUTS", raising=False)
    monkeypatch.delenv("TORCHTUNE_ROLLOUT_DUMP_PATH", raising=False)


def _args(b=2, g=3):
    return dict(
        step=7,
        batch_size=b,
        grpo_size=g,
        decoded=[f"resp b{i // g} g{i % g}" for i in range(b * g)],
        answers=[f"GO:000{i}" for i in range(b)],
        rewards=[float(i) for i in range(b * g)],
        successes=[0.0] * (b * g),
        proteins=["M" * 500 for _ in range(b)],
        go_aspects=["BP"] * b,
    )


# --- gating -----------------------------------------------------------------


def test_disabled_by_default():
    assert rollout_dump_path("/tmp/out") is None


def test_enabled_uses_output_dir(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_DUMP_ROLLOUTS", "1")
    assert rollout_dump_path("/tmp/out") == "/tmp/out/rollouts.jsonl"


def test_explicit_path_wins(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_DUMP_ROLLOUTS", "1")
    monkeypatch.setenv("TORCHTUNE_ROLLOUT_DUMP_PATH", "/tmp/x.jsonl")
    assert rollout_dump_path("/tmp/out") == "/tmp/x.jsonl"


def test_enabled_without_path_disables_rather_than_raises(monkeypatch):
    """Misconfiguration must degrade to 'no dump', not crash a 36h run."""
    monkeypatch.setenv("TORCHTUNE_DUMP_ROLLOUTS", "1")
    assert rollout_dump_path(None) is None


def test_none_path_writes_nothing(tmp_path):
    assert dump_rollout_groups(None, **_args()) is False


# --- content ----------------------------------------------------------------


def test_writes_one_record_per_group(tmp_path):
    p = str(tmp_path / "r.jsonl")
    assert dump_rollout_groups(p, **_args(b=2, g=3)) is True
    recs = [json.loads(ln) for ln in open(p)]
    assert len(recs) == 2
    assert [r["prompt_idx"] for r in recs] == [0, 1]
    assert all(len(r["samples"]) == 3 for r in recs)


def test_group_major_layout_matches_recipe_decode_order(tmp_path):
    """The recipe fills `_decoded` as `for b: for g:`. If the dump indexed it
    differently, samples would be silently attributed to the wrong prompt and the
    whole frequency analysis would be garbage while still looking plausible."""
    p = str(tmp_path / "r.jsonl")
    dump_rollout_groups(p, **_args(b=2, g=3))
    recs = [json.loads(ln) for ln in open(p)]
    assert [s["text"] for s in recs[0]["samples"]] == [
        "resp b0 g0",
        "resp b0 g1",
        "resp b0 g2",
    ]
    assert [s["text"] for s in recs[1]["samples"]] == [
        "resp b1 g0",
        "resp b1 g1",
        "resp b1 g2",
    ]
    # rewards are flat b*G+g -> group 1 must carry 3,4,5 not 0,1,2
    assert [s["reward"] for s in recs[1]["samples"]] == [3.0, 4.0, 5.0]


def test_protein_is_truncated_but_length_preserved(tmp_path):
    p = str(tmp_path / "r.jsonl")
    dump_rollout_groups(p, **_args())
    rec = json.loads(open(p).readline())
    assert len(rec["protein_prefix"]) == 64
    assert rec["protein_len"] == 500


def test_appends_across_steps(tmp_path):
    p = str(tmp_path / "r.jsonl")
    dump_rollout_groups(p, **{**_args(), "step": 1})
    dump_rollout_groups(p, **{**_args(), "step": 2})
    recs = [json.loads(ln) for ln in open(p)]
    assert sorted({r["step"] for r in recs}) == [1, 2]
    assert len(recs) == 4


def test_creates_missing_parent_dir(tmp_path):
    p = str(tmp_path / "nested" / "deep" / "r.jsonl")
    assert dump_rollout_groups(p, **_args()) is True
    assert os.path.exists(p)


# --- never fails the run ----------------------------------------------------


def test_length_mismatch_returns_false_not_raise(tmp_path):
    p = str(tmp_path / "r.jsonl")
    bad = {**_args(b=2, g=3), "decoded": ["only", "three", "items"][:3]}
    bad["batch_size"], bad["grpo_size"] = 2, 3  # expects 6
    assert dump_rollout_groups(p, **bad) is False


def test_unwritable_path_returns_false_not_raise(tmp_path):
    assert dump_rollout_groups("/proc/nope/r.jsonl", **_args()) is False


def test_missing_optional_fields_ok(tmp_path):
    p = str(tmp_path / "r.jsonl")
    a = _args()
    for k in ("rewards", "successes", "proteins", "go_aspects"):
        a.pop(k)
    assert dump_rollout_groups(p, **a) is True
    rec = json.loads(open(p).readline())
    assert rec["protein_prefix"] is None
    assert rec["samples"][0]["reward"] is None


# --- the CALL SITE, not just the helper -------------------------------------
#
# The module tests above all pass whether or not anything ever calls it. A dump
# helper that no recipe invokes is exactly the gap this was written to close, so
# the wiring gets its own guard. AST, not substring: a text match would fire on a
# correct refactor and teach people to ignore the test.

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

_RECIPE = (
    Path(__file__).resolve().parents[4]
    / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
)
_TREE = ast.parse(_RECIPE.read_text())


def _dump_calls():
    return [
        n
        for n in ast.walk(_TREE)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "dump_rollout_groups"
    ]


def test_recipe_calls_dump_rollout_groups():
    assert _dump_calls(), (
        "grpo_bioreason_distributed_xpu.py never calls dump_rollout_groups — the "
        "decoded rollouts are discarded at the end of the reward branch and are "
        "then unrecoverable"
    )


def test_recipe_imports_the_dump_helpers():
    imported = {
        alias.name
        for n in ast.walk(_TREE)
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("rollout_dump")
        for alias in n.names
    }
    assert {"dump_rollout_groups", "rollout_dump_path"} <= imported


def test_dump_call_passes_every_field_the_analysis_needs():
    """Group-frequency ranking needs the per-sample text grouped by prompt, and the
    reward to correlate against. A call that silently dropped `rewards` would still
    write a valid file and still pass every test above."""
    call = _dump_calls()[0]
    kw = {k.arg for k in call.keywords}
    for need in ("step", "batch_size", "grpo_size", "decoded", "answers", "rewards"):
        assert need in kw, f"dump call is missing {need}= (have: {sorted(kw)})"


def test_dump_call_is_rank_zero_guarded():
    """All 12 ranks run this branch with identical data at dp_replicate=1. Ungated,
    the JSONL gets 12 interleaved copies of every record from concurrent appends."""
    call = _dump_calls()[0]
    guards = [
        n
        for n in ast.walk(_TREE)
        if isinstance(n, ast.If)
        and n.lineno <= call.lineno <= (n.end_lineno or n.lineno)
        and "_is_rank_zero" in ast.dump(n.test)
    ]
    assert guards, "dump_rollout_groups call is not inside an `if self._is_rank_zero:`"

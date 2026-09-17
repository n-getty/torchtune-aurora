# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experiments/bioreason/rescore_rollouts_by_group_frequency.py.

The script converts real GRPO rollouts (``rollouts.jsonl``, G=8 completions per prompt)
into frequency-scored predictions for the unmodified F_max scorer. Two of its properties
are load-bearing and would fail silently if they regressed, so they are pinned here:

  - **Every sample votes.** The dump's ``success`` field is the *reward function's*
    correctness verdict, not a liveness flag. Filtering on it would keep only rollouts
    that already agreed with ground truth and turn the measurement into oracle selection
    -- producing a large, entirely fake F_max gain.
  - **Steps are not pooled by default.** Terms from step 3 and step 97 come from
    different policies; unioning them measures drift, not within-group agreement.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _ROOT / "experiments" / "bioreason" / "rescore_rollouts_by_group_frequency.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rescore_rollouts", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    assert _SCRIPT.is_file(), f"missing {_SCRIPT}"
    return _load_module()


def _rec(step, prompt_idx, answer, sample_texts, successes=None, protein="MKV" * 10):
    """One rollout_dump.dump_rollout_groups record."""
    successes = successes if successes is not None else [0.0] * len(sample_texts)
    return {
        "step": step,
        "prompt_idx": prompt_idx,
        "answer": answer,
        "go_aspect": "all",
        "protein_prefix": protein[:64],
        "protein_len": len(protein),
        "samples": [
            {"g": g, "text": t, "reward": 0.5, "success": s}
            for g, (t, s) in enumerate(zip(sample_texts, successes))
        ],
    }


def _write(tmp_path, records):
    p = tmp_path / "rollouts.jsonl"
    with open(p, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def test_frequency_equals_fraction_of_samples_emitting_the_term(mod, tmp_path):
    # 4 samples: GO:0000001 in all 4, GO:0000002 in 2, GO:0000003 in 1.
    texts = [
        "GO:0000001 GO:0000002 GO:0000003",
        "GO:0000001 GO:0000002",
        "GO:0000001",
        "GO:0000001",
    ]
    path = _write(tmp_path, [_rec(5, 0, "GO:0000001", texts)])
    groups, _stats = mod.load_rollouts(path, step=5)
    arms = mod.build_arms(groups)
    scored, _carry, g = next(iter(arms["freq"].values()))
    assert g == 4
    assert scored["GO:0000001"] == pytest.approx(1.0)
    assert scored["GO:0000002"] == pytest.approx(0.5)
    assert scored["GO:0000003"] == pytest.approx(0.25)


def test_union_flat_and_freq_share_a_term_set_so_the_delta_is_pure_ranking(mod, tmp_path):
    texts = ["GO:0000001 GO:0000002", "GO:0000001", "GO:0000003"]
    path = _write(tmp_path, [_rec(1, 0, "GO:0000001", texts)])
    arms = mod.build_arms(mod.load_rollouts(path, step=1)[0])
    flat, _c, _g = next(iter(arms["union_flat"].values()))
    freq, _c2, _g2 = next(iter(arms["freq"].values()))
    assert set(flat) == set(freq)
    assert set(flat.values()) == {1.0}
    assert len(set(freq.values())) > 1  # the sweep has something to sweep over


def test_unsuccessful_samples_still_vote(mod, tmp_path):
    """THE ORACLE-SELECTION GUARD.

    ``success`` in rollouts.jsonl is reward-derived. If the loader ever filters on it,
    only rollouts that already matched ground truth would vote, every frequency would be
    pulled toward the GT, and the arm would post a large fake gain. Here the only sample
    marked successful is the one that happens to be right; a filtering implementation
    would drop the other three and report conf=1.0 for the GT term alone.
    """
    texts = ["GO:0000001", "GO:0000009", "GO:0000009", "GO:0000009"]
    successes = [1.0, 0.0, 0.0, 0.0]
    path = _write(tmp_path, [_rec(3, 0, "GO:0000001", texts, successes=successes)])
    groups, _ = mod.load_rollouts(path, step=3)
    scored, _carry, g = next(iter(mod.build_arms(groups)["freq"].values()))
    assert g == 4, "all four samples must vote, regardless of `success`"
    assert scored["GO:0000001"] == pytest.approx(0.25)
    assert scored["GO:0000009"] == pytest.approx(0.75)


def test_source_does_not_filter_on_success(mod):
    """Belt-and-braces on the same trap: the behavioural test above could be satisfied by
    an implementation that reads `success` for some other purpose and later regresses into
    filtering. Assert the loader never consults the field at all."""
    src = _SCRIPT.read_text()
    loader = src.split("def load_rollouts")[1].split("\ndef ")[0]
    assert 's.get("success"' not in loader and "['success']" not in loader
    assert 'success' in src  # it IS discussed -- in the docstring explaining why not


def test_multi_step_dump_refuses_to_silently_pool(mod, tmp_path):
    path = _write(
        tmp_path,
        [
            _rec(1, 0, "GO:0000001", ["GO:0000001", "GO:0000002"]),
            _rec(2, 0, "GO:0000001", ["GO:0000001", "GO:0000003"]),
        ],
    )
    with pytest.raises(SystemExit) as ei:
        mod.load_rollouts(path)
    assert "--step" in str(ei.value)


def test_last_n_steps_selects_the_trailing_window(mod, tmp_path):
    recs = [_rec(s, 0, "GO:0000001", [f"GO:000000{s}"]) for s in (1, 2, 3)]
    path = _write(tmp_path, recs)
    _groups, stats = mod.load_rollouts(path, last_n_steps=2)
    assert stats["steps_used"] == [2, 3]


def test_ground_truth_lands_where_the_scorer_reads_it(mod, tmp_path):
    """The emitted record must be readable by the UNMODIFIED
    ce.extract_reasoning_ground_truth, which unions go_bp/go_mf/go_cc."""
    path = _write(tmp_path, [_rec(1, 0, "GO:0000001, GO:0000002", ["GO:0000001"])])
    arms = mod.build_arms(mod.load_rollouts(path, step=1)[0])
    out = tmp_path / "arms" / "freq"
    mod.write_arm(arms["freq"], str(out))
    written = list(out.glob("*_k00.json"))
    assert len(written) == 1
    rec = json.loads(written[0].read_text())
    assert set(rec["go_bp"]) == {"GO:0000001", "GO:0000002"}
    assert rec["success"] is True
    assert rec["_scored"]["GO:0000001"] == pytest.approx(1.0)


def test_exactly_one_k_file_per_target_no_oracle_best_of_k(mod, tmp_path):
    """score_fmax_scored.py runs select_best_from_k_samples when it sees _k01+, which
    picks the highest-F1 sample against ground truth. The fold must happen here."""
    path = _write(tmp_path, [_rec(1, 0, "GO:0000001", ["GO:0000001", "GO:0000002"])])
    arms = mod.build_arms(mod.load_rollouts(path, step=1)[0])
    out = tmp_path / "freq"
    mod.write_arm(arms["freq"], str(out))
    names = sorted(p.name for p in out.glob("*.json"))
    assert all(n.endswith("_k00.json") for n in names), names


def test_empty_samples_are_dropped_from_the_denominator_by_default(mod, tmp_path):
    """A sample that emitted no GO term is a parse outcome, not a correctness one, but it
    would otherwise depress every frequency in its group."""
    path = _write(tmp_path, [_rec(1, 0, "GO:0000001", ["GO:0000001", "no terms here"])])
    groups, stats = mod.load_rollouts(path, step=1)
    scored, _c, g = next(iter(mod.build_arms(groups)["freq"].values()))
    assert g == 1 and stats["empty"] == 1
    assert scored["GO:0000001"] == pytest.approx(1.0)


def test_final_answer_only_ignores_terms_considered_mid_trace(mod, tmp_path):
    texts = ["GO:0000009 is plausible </think> GO:0000001"] * 2
    path = _write(tmp_path, [_rec(1, 0, "GO:0000001", texts)])
    groups, _ = mod.load_rollouts(path, step=1, final_answer_only=True)
    scored, _c, _g = next(iter(mod.build_arms(groups)["freq"].values()))
    assert set(scored) == {"GO:0000001"}


def test_torn_final_line_does_not_discard_the_run(mod, tmp_path):
    """A job killed mid-write leaves a partial last line."""
    p = tmp_path / "rollouts.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps(_rec(1, 0, "GO:0000001", ["GO:0000001"])) + "\n")
        fh.write('{"step": 2, "samples": [{"g": 0, "te')
    groups, stats = mod.load_rollouts(str(p), pool_all_steps=True)
    assert stats["records"] == 1 and len(groups) == 1


def test_distinct_proteins_do_not_collide(mod, tmp_path):
    recs = [
        _rec(1, 0, "GO:0000001", ["GO:0000001"], protein="AAA" * 30),
        _rec(1, 1, "GO:0000002", ["GO:0000002"], protein="CCC" * 30),
    ]
    groups, _ = mod.load_rollouts(_write(tmp_path, recs), step=1)
    assert len(groups) == 2

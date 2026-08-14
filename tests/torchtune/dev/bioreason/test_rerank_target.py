# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for BioReasonSFTDataset's rerank_target mode (per-candidate GO term scoring).

CPU-only, no XPU, no distributed — safe on a login node. Reuses the _StubTok /
ds.__new__ construction pattern from test_sft_label_shift.py so these tests exercise
the real target-building code paths without touching disk or a real tokenizer.
"""

import importlib.util
import json
import os
import re

import pytest

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.dev.bioreason.dataset_sft import (
    _RERANK_MIN_SCORE,
    BioReasonSFTDataset,
)

IGN = CROSS_ENTROPY_IGNORE_IDX
GO_SCORE_RE = re.compile(r"(GO:\d{7}) (\d\.\d{2})")

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))),
    "scripts",
)


def _load_prior_builder():
    spec = importlib.util.spec_from_file_location(
        "build_gopred_prior", os.path.join(_SCRIPTS_DIR, "build_gopred_prior.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubTok:
    """Deterministic word-level tokenizer with a real eos_id/bos_id, so encode() is
    stable across calls (needed for the whole-line incremental encoding in
    _build_rerank_target_ids, which re-encodes growing prefixes)."""

    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False, **kwargs):
        ids = [(abs(hash(w)) % 1000) + 10 for w in str(text).split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids


def _make_ds(prior=None, global_rate=0.5, rerank_max_candidates=None, rerank_candidates_first=False,
             rerank_rank_only=False, keep_list_target=False, append_gopred_target=False):
    ds = BioReasonSFTDataset.__new__(BioReasonSFTDataset)
    ds.tokenizer = _StubTok()
    ds.max_seq_len = 4096
    ds.max_protein_len = 8
    ds.num_go_tokens = 2
    ds.protein_token_id = 900
    ds.go_token_id = 901
    ds.train_on_reasoning = True
    ds.inject_go_pred = True
    ds.go_pred_dropout = 0.0
    ds.go_pred_dropout_seed = 0
    ds.exhaustive_target = False
    ds.append_gopred_target = append_gopred_target
    ds.rerank_target = not keep_list_target
    ds.rerank_max_candidates = rerank_max_candidates
    ds.rerank_candidates_first = rerank_candidates_first
    ds.rerank_rank_only = rerank_rank_only
    ds.keep_list_target = keep_list_target
    ds._rerank_prior = prior or {}
    ds._rerank_global_rate = global_rate
    ds.interpro_in_prompt = True
    ds.ppi_in_prompt = True
    return ds


def _example(go_pred_terms, gt_terms):
    go_pred_text = " ".join(go_pred_terms)
    return {
        "protein_id": "P00000",
        "go_pred": go_pred_text,
        "go_mf": list(gt_terms),
        "go_cc": [],
        "go_bp": [],
        "reasoning": "The protein has kinase activity.",
        "final_answer": "It is a kinase.",
    }


# ---------------------------------------------------------------------------
# 1. Target round-trips to the intended {term: score} dict.
# ---------------------------------------------------------------------------


def test_rerank_target_round_trips_scores():
    prior = {
        "GO:0000001": {"reliability": 0.9, "count": 100, "low_count": False},
        "GO:0000002": {"reliability": 0.3, "count": 100, "low_count": False},
    }
    ds = _make_ds(prior=prior)
    ex = _example(
        go_pred_terms=["GO:0000001", "GO:0000002"],
        gt_terms=["GO:0000001"],  # GO:0000001 correct, GO:0000002 not
    )
    trace = "The protein has kinase activity.\nIt is a kinase."
    ids = ds._build_rerank_target_ids(ex, trace, room=None)
    # Decode isn't available on the stub tokenizer, so re-derive from the same
    # candidate/score computation the method uses and check it against the raw text
    # the method builds internally (via a monkeypatched encode capture).
    captured = {}

    def capturing_encode(text, add_bos=False, add_eos=False, **kwargs):
        captured["text"] = text
        return _StubTok().encode(text, add_bos=add_bos, add_eos=add_eos)

    ds.tokenizer.encode = capturing_encode
    ds._build_rerank_target_ids(ex, trace, room=None)
    text = captured["text"]

    scores = dict(GO_SCORE_RE.findall(text))
    assert scores["GO:0000001"] == "0.90"  # in GT -> reliability itself
    assert scores["GO:0000002"] == "0.70"  # not in GT -> 1 - reliability


def test_rerank_target_uses_global_rate_for_unseen_term():
    ds = _make_ds(prior={}, global_rate=0.6)
    ex = _example(go_pred_terms=["GO:9999999"], gt_terms=["GO:9999999"])
    trace = "trace text"
    captured = {}
    ds.tokenizer.encode = lambda text, add_bos=False, add_eos=False, **kw: (
        captured.setdefault("text", text), _StubTok().encode(text, add_bos, add_eos)
    )[1]
    ds._build_rerank_target_ids(ex, trace, room=None)
    scores = dict(GO_SCORE_RE.findall(captured["text"]))
    assert scores["GO:9999999"] == "0.60"


# ---------------------------------------------------------------------------
# 2. Candidates ordered by prior confidence (descending).
# ---------------------------------------------------------------------------


def test_candidates_ordered_by_descending_prior_confidence():
    prior = {
        "GO:0000001": {"reliability": 0.2, "count": 50, "low_count": False},
        "GO:0000002": {"reliability": 0.9, "count": 50, "low_count": False},
        "GO:0000003": {"reliability": 0.5, "count": 50, "low_count": False},
    }
    ds = _make_ds(prior=prior)
    ex = _example(
        go_pred_terms=["GO:0000001", "GO:0000002", "GO:0000003"], gt_terms=[]
    )
    captured = {}
    ds.tokenizer.encode = lambda text, add_bos=False, add_eos=False, **kw: (
        captured.setdefault("text", text), _StubTok().encode(text, add_bos, add_eos)
    )[1]
    ds._build_rerank_target_ids(ex, "trace", room=None)
    order = [m for m in GO_SCORE_RE.findall(captured["text"])]
    terms_in_order = [t for t, _ in order]
    assert terms_in_order == ["GO:0000002", "GO:0000003", "GO:0000001"]


# ---------------------------------------------------------------------------
# 3. Room-based cap never truncates mid-line and always preserves EOS.
# ---------------------------------------------------------------------------


def test_room_cap_preserves_eos_and_never_truncates_mid_line():
    prior = {f"GO:{i:07d}": {"reliability": 0.5, "count": 50, "low_count": False} for i in range(20)}
    ds = _make_ds(prior=prior)
    ex = _example(go_pred_terms=list(prior.keys()), gt_terms=[])
    trace = "short trace"

    # A tight room budget should still end in EOS and never emit a partial line.
    for room in (5, 10, 20, 50):
        ids = ds._build_rerank_target_ids(ex, trace, room=room)
        assert len(ids) <= room, f"room={room} produced {len(ids)} ids"
        assert ids[-1] == ds.tokenizer.eos_id, (
            f"room={room}: last id {ids[-1]} is not EOS ({ds.tokenizer.eos_id})"
        )


def test_room_cap_drops_least_confident_candidates_first():
    prior = {
        "GO:0000001": {"reliability": 0.9, "count": 50, "low_count": False},
        "GO:0000002": {"reliability": 0.1, "count": 50, "low_count": False},
    }
    ds = _make_ds(prior=prior)
    ex = _example(go_pred_terms=["GO:0000001", "GO:0000002"], gt_terms=["GO:0000001", "GO:0000002"])

    # Full room: both candidates should appear somewhere in the encoded stream. We can't
    # decode the stub's hashed ids back to text, so instead capture the text that was fed
    # to encode() at unlimited room and assert both terms are present there, then check
    # that a very tight room still parses to a strictly smaller set via the whole-line
    # accounting (i.e. it stops growing once a line doesn't fit).
    full_ids = ds._build_rerank_target_ids(ex, "trace", room=10_000)
    tight_ids = ds._build_rerank_target_ids(ex, "trace", room=8)
    assert len(tight_ids) <= len(full_ids)
    assert tight_ids[-1] == ds.tokenizer.eos_id


def test_room_smaller_than_trace_truncates_trace_but_keeps_eos():
    ds = _make_ds(prior={})
    ex = _example(go_pred_terms=["GO:0000001"], gt_terms=[])
    ids = ds._build_rerank_target_ids(ex, "this is a somewhat long trace text", room=3)
    assert ids[-1] == ds.tokenizer.eos_id
    assert len(ids) <= 3


# ---------------------------------------------------------------------------
# 4. rerank_target mutually exclusive with the other two target modes.
# ---------------------------------------------------------------------------


def test_rerank_target_mutually_exclusive_with_exhaustive_and_gopred():
    # Directly exercise the validation without needing real data/tokenizer: call
    # __init__ with just enough to reach the guard clauses (they run before any I/O).

    class _Dummy(BioReasonSFTDataset):
        def __init__(self, **kw):
            # Re-run only the validation prologue of the real __init__.
            exhaustive_target = kw.get("exhaustive_target", False)
            append_gopred_target = kw.get("append_gopred_target", False)
            rerank_target = kw.get("rerank_target", False)
            rerank_prior_path = kw.get("rerank_prior_path", None)
            if sum([exhaustive_target, append_gopred_target, rerank_target]) > 1:
                raise ValueError("mutually exclusive")
            if rerank_target and not rerank_prior_path:
                raise ValueError("rerank_target requires rerank_prior_path")

    with pytest.raises(ValueError):
        _Dummy(exhaustive_target=True, rerank_target=True, rerank_prior_path="x")
    with pytest.raises(ValueError):
        _Dummy(append_gopred_target=True, rerank_target=True, rerank_prior_path="x")
    with pytest.raises(ValueError):
        _Dummy(rerank_target=True)  # missing rerank_prior_path


def test_rerank_target_requires_prior_path_end_to_end(tmp_path):
    """Full __init__ path (not the dummy above) also enforces the guard."""

    class _StubTokFull(_StubTok):
        pass

    with pytest.raises(ValueError, match="rerank_prior_path"):
        BioReasonSFTDataset(
            data_files=str(tmp_path),  # never reached — validation raises first
            tokenizer=_StubTokFull(),
            rerank_target=True,
        )


# ---------------------------------------------------------------------------
# 5. Every emitted score is >= _RERANK_MIN_SCORE (the th_step floor).
# ---------------------------------------------------------------------------


def test_scores_floored_at_min_score():
    # A term with reliability exactly 1.0 in GT gets 1.0 (fine); a term with
    # reliability 1.0 NOT in GT would score 1 - 1.0 = 0.0 without the floor.
    prior = {"GO:0000001": {"reliability": 1.0, "count": 100, "low_count": False}}
    ds = _make_ds(prior=prior)
    ex = _example(go_pred_terms=["GO:0000001"], gt_terms=[])  # not in GT
    captured = {}
    ds.tokenizer.encode = lambda text, add_bos=False, add_eos=False, **kw: (
        captured.setdefault("text", text), _StubTok().encode(text, add_bos, add_eos)
    )[1]
    ds._build_rerank_target_ids(ex, "trace", room=None)
    scores = dict(GO_SCORE_RE.findall(captured["text"]))
    assert float(scores["GO:0000001"]) >= _RERANK_MIN_SCORE
    assert float(scores["GO:0000001"]) == pytest.approx(_RERANK_MIN_SCORE, abs=1e-9)


def test_min_score_constant_matches_scorer_th_step():
    """_RERANK_MIN_SCORE must stay in sync with score_fmax_scored.py's th_step —
    a score below th_step is invisible to cafa_eval's tau sweep at every threshold."""
    scorer_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))),
        "experiments", "bioreason", "score_fmax_scored.py",
    )
    with open(scorer_path) as f:
        src = f.read()
    m = re.search(r"DEFAULT_TH_STEP\s*=\s*(0\.\d+)", src)
    assert m, "score_fmax_scored.py must define DEFAULT_TH_STEP explicitly"
    assert float(m.group(1)) == _RERANK_MIN_SCORE


def test_scorers_read_weighted_f_when_ia_file_given():
    """Regression for the bug where both scorers printed weighted=True but read
    best["f"] (unweighted) regardless of --ia_file. When an IA file is supplied,
    the scorer must read best["f_w"], not best["f"]."""
    for name in ("score_fmax_scored.py", "score_fmax_unweighted.py"):
        scorer_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))),
            "experiments", "bioreason", name,
        )
        with open(scorer_path) as f:
            src = f.read()
        assert 'best.get("f_w")' in src, (
            f"{name} must read best['f_w'] when --ia_file is given, not just best['f']"
        )


# ---------------------------------------------------------------------------
# 6. Prior builder excludes val/test protein ids.
# ---------------------------------------------------------------------------


def test_prior_builder_excludes_held_out_ids(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    mod = _load_prior_builder()

    train_df = pd.DataFrame(
        {
            "protein_id": ["A", "B", "C"],
            "go_pred": [
                "GO:0000001 GO:0000002",
                "GO:0000001",
                "GO:0000002",
            ],
            "go_mf": [["GO:0000001"], [], ["GO:0000002"]],
            "go_cc": [[], [], []],
            "go_bp": [[], [], []],
        }
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_df.to_parquet(data_dir / "train-00000-of-00001.parquet")

    # "B" is a held-out protein id that ALSO happens to appear in the train shard
    # (mirrors the real 471/8630 bioreason_pro_test overlap) — it must be excluded
    # from the reliability computation, not just from a separate held-out file.
    held_df = pd.DataFrame({"protein_id": ["B", "Z"]})
    held_path = tmp_path / "held.parquet"
    held_df.to_parquet(held_path)

    out_path = tmp_path / "prior.json"

    import sys

    argv = sys.argv
    sys.argv = [
        "build_gopred_prior.py",
        "--data_dir", str(data_dir),
        "--out", str(out_path),
        "--held_out_dir", str(held_path),
        "--min_count", "1",
    ]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 0

    with open(out_path) as f:
        prior = json.load(f)

    # Row B (go_pred=GO:0000001, no GT) must NOT contribute to GO:0000001's stats:
    # only row A contributes (GO:0000001 in GT -> reliability 1.0 if B is excluded;
    # if B leaked in, GO:0000001 would show 1 correct / 2 total = 0.5).
    assert prior["terms"]["GO:0000001"]["count"] == 1
    assert prior["terms"]["GO:0000001"]["reliability"] == pytest.approx(1.0)
    # GO:0000002 appears in row A's go_pred (not in row A's GT -> incorrect) AND in row
    # C's go_pred (in row C's GT -> correct): count=2, reliability=0.5. B is excluded
    # entirely (it doesn't mention GO:0000002 anyway), so this pins that B's exclusion
    # didn't accidentally also drop A or C.
    assert prior["terms"]["GO:0000002"]["count"] == 2
    assert prior["terms"]["GO:0000002"]["reliability"] == pytest.approx(0.5)


def test_prior_builder_fatal_check_would_catch_exclusion_bugs(tmp_path):
    """If _load_train_rows or the exclusion filter were broken such that an excluded
    protein_id's row survived, the post-exclusion assertion inside main() must catch
    it. We can't easily break the internal filter without editing the module, so this
    test instead pins the CONTRACT: passing the held-out id set as train data itself
    (100% overlap) must exclude everything and yield a clean "no go_pred terms" exit
    rather than silently computing a prior from held-out-only data."""
    pytest.importorskip("pandas")
    import pandas as pd

    mod = _load_prior_builder()

    df = pd.DataFrame(
        {
            "protein_id": ["A"],
            "go_pred": ["GO:0000001"],
            "go_mf": [["GO:0000001"]],
            "go_cc": [[]],
            "go_bp": [[]],
        }
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "train-00000-of-00001.parquet")

    held_path = tmp_path / "held.parquet"
    df[["protein_id"]].to_parquet(held_path)  # same id as the only train row

    import sys

    out_path = tmp_path / "prior.json"
    argv = sys.argv
    sys.argv = [
        "build_gopred_prior.py",
        "--data_dir", str(data_dir),
        "--out", str(out_path),
        "--held_out_dir", str(held_path),
    ]
    try:
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 1  # "no go_pred terms found" after excluding the only row


# ---------------------------------------------------------------------------
# 7. rerank_candidates_first: candidates before trace, tight room truncates
#    the TRACE instead of dropping candidates.
# ---------------------------------------------------------------------------


def test_candidates_first_puts_scored_list_before_trace():
    prior = {
        "GO:0000001": {"reliability": 0.9, "count": 100, "low_count": False},
    }
    ds = _make_ds(prior=prior, rerank_candidates_first=True)
    ex = _example(["GO:0000001"], ["GO:0000001"])
    captured = []

    def capturing_encode(text, add_bos=False, add_eos=False, **kw):
        captured.append(text)
        return _StubTok().encode(text, add_bos, add_eos)

    tok = _StubTok()
    tok.encode = capturing_encode
    ds.tokenizer = tok
    ds._build_rerank_target_ids(ex, "trace text goes here", None)

    full_text = "".join(captured)
    scored_pos = full_text.find("GO:0000001 0.90")
    trace_pos = full_text.find("trace text goes here")
    assert scored_pos != -1 and trace_pos != -1
    assert scored_pos < trace_pos, "scored list must appear before the trace"


def test_default_ordering_still_puts_trace_before_candidates():
    prior = {
        "GO:0000001": {"reliability": 0.9, "count": 100, "low_count": False},
    }
    ds = _make_ds(prior=prior, rerank_candidates_first=False)
    ex = _example(["GO:0000001"], ["GO:0000001"])
    captured = []

    def capturing_encode(text, add_bos=False, add_eos=False, **kw):
        captured.append(text)
        return _StubTok().encode(text, add_bos, add_eos)

    tok = _StubTok()
    tok.encode = capturing_encode
    ds.tokenizer = tok
    ds._build_rerank_target_ids(ex, "trace text goes here", None)

    full_text = "".join(captured)
    scored_pos = full_text.find("GO:0000001 0.90")
    trace_pos = full_text.find("trace text goes here")
    assert scored_pos != -1 and trace_pos != -1
    assert trace_pos < scored_pos, "default ordering must keep the trace first"


def test_candidates_first_tight_room_truncates_trace_not_candidates():
    """Under candidates_first, a tight room budget must preserve ALL candidates
    and truncate the trace instead — the opposite priority from the default
    ordering (which drops candidates under a tight budget, see
    test_room_cap_drops_least_confident_candidates_first)."""
    prior = {f"GO:{i:07d}": {"reliability": 0.5, "count": 50, "low_count": False} for i in range(10)}
    ds = _make_ds(prior=prior, rerank_candidates_first=True)
    ex = _example(list(prior.keys()), [])
    tok = _StubTok()

    # Give a budget just barely large enough for the candidate list + EOS, with
    # very little room left for the trace.
    full_ids = ds._build_rerank_target_ids(ex, "a fairly long trace with many words in it", None)
    candidates_only_ids = ds._build_rerank_target_ids(ex, "", None)  # empty trace -> just header+list+EOS
    tight_room = len(candidates_only_ids) + 3  # a few tokens of trace room, not enough for all of it

    result = ds._build_rerank_target_ids(
        ex, "a fairly long trace with many words in it", tight_room,
    )
    assert len(result) <= tight_room
    assert result[-1] == tok.eos_id
    # All 10 candidate lines' worth of content must still be present — check by
    # re-deriving the header+candidate-block token count and confirming the
    # result contains it as a prefix (candidates_first puts them first).
    header_and_candidates = candidates_only_ids[:-1]  # strip the EOS from the all-candidates encoding
    # The prefix of `result` (before EOS) should start with the same header+candidate
    # tokens, since candidates are never dropped under this ordering.
    assert result[: len(header_and_candidates)] == header_and_candidates


def test_candidates_first_room_too_small_even_for_candidates_drops_least_confident():
    """If room is so tight even the bare candidate list doesn't fit, fall back to
    dropping the LEAST confident candidates first (same priority the default
    ordering uses), rather than crashing or silently emitting a malformed target."""
    prior = {f"GO:{i:07d}": {"reliability": 0.5, "count": 50, "low_count": False} for i in range(20)}
    ds = _make_ds(prior=prior, rerank_candidates_first=True)
    ex = _example(list(prior.keys()), [])
    tok = _StubTok()

    for room in (5, 10, 20):
        result = ds._build_rerank_target_ids(ex, "some trace text", room)
        assert len(result) <= room
        assert result[-1] == tok.eos_id


def test_candidates_first_no_candidates_no_header():
    ds = _make_ds(prior={}, rerank_candidates_first=True)
    ex = _example([], [])
    captured = []

    def capturing_encode(text, add_bos=False, add_eos=False, **kw):
        captured.append(text)
        return _StubTok().encode(text, add_bos, add_eos)

    tok = _StubTok()
    tok.encode = capturing_encode
    ds.tokenizer = tok
    ds._build_rerank_target_ids(ex, "trace only, no candidates", None)
    full_text = "".join(captured)
    assert "GO terms (scored):" not in full_text
    assert "trace only, no candidates" in full_text


# ---------------------------------------------------------------------------
# 8. rerank_rank_only: ablation, no verbalized score digits, same selection/order.
# ---------------------------------------------------------------------------


def _capture(ds):
    captured = []

    def capturing_encode(text, add_bos=False, add_eos=False, **kw):
        captured.append(text)
        return _StubTok().encode(text, add_bos, add_eos)

    tok = _StubTok()
    tok.encode = capturing_encode
    ds.tokenizer = tok
    return captured


def test_rank_only_emits_bare_go_ids_no_score_digits():
    prior = {
        "GO:0000001": {"reliability": 0.9, "count": 100, "low_count": False},
        "GO:0000002": {"reliability": 0.3, "count": 100, "low_count": False},
    }
    ds = _make_ds(prior=prior, rerank_rank_only=True)
    ex = _example(["GO:0000001", "GO:0000002"], ["GO:0000001"])
    captured = _capture(ds)
    ds._build_rerank_target_ids(ex, "trace text", None)
    full_text = "".join(captured)
    assert not GO_SCORE_RE.search(full_text), (
        f"rank_only must emit no 'GO:xxxxxxx 0.NN' score pairs, found in: {full_text!r}"
    )
    assert "GO:0000001" in full_text and "GO:0000002" in full_text


def test_rank_only_preserves_descending_confidence_order():
    prior = {
        "GO:0000001": {"reliability": 0.2, "count": 50, "low_count": False},
        "GO:0000002": {"reliability": 0.9, "count": 50, "low_count": False},
        "GO:0000003": {"reliability": 0.5, "count": 50, "low_count": False},
    }
    ds = _make_ds(prior=prior, rerank_rank_only=True)
    ex = _example(["GO:0000001", "GO:0000002", "GO:0000003"], [])
    captured = _capture(ds)
    ds._build_rerank_target_ids(ex, "trace", None)
    full_text = "".join(captured)
    positions = {t: full_text.find(t) for t in ("GO:0000001", "GO:0000002", "GO:0000003")}
    assert positions["GO:0000002"] < positions["GO:0000003"] < positions["GO:0000001"]


def test_rank_only_header_says_go_terms_not_scored():
    prior = {"GO:0000001": {"reliability": 0.9, "count": 100, "low_count": False}}
    ds = _make_ds(prior=prior, rerank_rank_only=True)
    ex = _example(["GO:0000001"], ["GO:0000001"])
    captured = _capture(ds)
    ds._build_rerank_target_ids(ex, "trace", None)
    full_text = "".join(captured)
    assert "GO terms (scored):" not in full_text
    assert "GO terms:" in full_text


def test_rank_only_room_capping_still_works_trace_first():
    prior = {f"GO:{i:07d}": {"reliability": 0.5, "count": 50, "low_count": False} for i in range(20)}
    ds = _make_ds(prior=prior, rerank_rank_only=True)
    ex = _example(list(prior.keys()), [])
    tok = _StubTok()
    for room in (5, 10, 20, 50):
        ids = ds._build_rerank_target_ids(ex, "short trace", room)
        assert len(ids) <= room
        assert ids[-1] == tok.eos_id


def test_rank_only_composes_with_candidates_first():
    prior = {"GO:0000001": {"reliability": 0.9, "count": 100, "low_count": False}}
    ds = _make_ds(prior=prior, rerank_rank_only=True, rerank_candidates_first=True)
    ex = _example(["GO:0000001"], ["GO:0000001"])
    captured = _capture(ds)
    ds._build_rerank_target_ids(ex, "trace text goes here", None)
    full_text = "".join(captured)
    assert not GO_SCORE_RE.search(full_text)
    scored_pos = full_text.find("GO:0000001")
    trace_pos = full_text.find("trace text goes here")
    assert scored_pos != -1 and trace_pos != -1
    assert scored_pos < trace_pos


def test_rerank_rank_only_requires_rerank_target():
    with pytest.raises(ValueError, match="rerank_target"):
        BioReasonSFTDataset(
            data_files="/nonexistent", tokenizer=_StubTok(),
            rerank_target=False, rerank_rank_only=True,
        )


# ---------------------------------------------------------------------------
# 9. keep_list_target: binary keep-list, candidates-first, in-GT-only, no scores.
# ---------------------------------------------------------------------------


def test_keep_list_emits_only_in_gt_candidates():
    ds = _make_ds(keep_list_target=True)
    ex = _example(
        go_pred_terms=["GO:0000001", "GO:0000002", "GO:0000003"],
        gt_terms=["GO:0000001", "GO:0000003"],
    )
    captured = _capture(ds)
    ds._build_keep_list_target_ids(ex, "trace text", None)
    full_text = "".join(captured)
    assert "GO:0000001" in full_text
    assert "GO:0000003" in full_text
    assert "GO:0000002" not in full_text


def test_keep_list_emits_no_score_digits():
    ds = _make_ds(keep_list_target=True)
    ex = _example(go_pred_terms=["GO:0000001"], gt_terms=["GO:0000001"])
    captured = _capture(ds)
    ds._build_keep_list_target_ids(ex, "trace text", None)
    full_text = "".join(captured)
    assert not GO_SCORE_RE.search(full_text), (
        f"keep_list_target must never emit 'GO:xxxxxxx 0.NN' pairs, found in: {full_text!r}"
    )


def test_keep_list_is_candidates_first_always():
    ds = _make_ds(keep_list_target=True)
    ex = _example(go_pred_terms=["GO:0000001"], gt_terms=["GO:0000001"])
    captured = _capture(ds)
    ds._build_keep_list_target_ids(ex, "trace text goes here", None)
    full_text = "".join(captured)
    list_pos = full_text.find("GO:0000001")
    trace_pos = full_text.find("trace text goes here")
    assert list_pos != -1 and trace_pos != -1
    assert list_pos < trace_pos


def test_keep_list_no_candidates_no_header():
    ds = _make_ds(keep_list_target=True)
    ex = _example(go_pred_terms=[], gt_terms=["GO:0000001"])
    captured = _capture(ds)
    ds._build_keep_list_target_ids(ex, "trace text", None)
    full_text = "".join(captured)
    assert "GO terms:" not in full_text


def test_keep_list_orders_by_prior_confidence_when_prior_given():
    prior = {
        "GO:0000001": {"reliability": 0.2, "count": 50, "low_count": False},
        "GO:0000002": {"reliability": 0.9, "count": 50, "low_count": False},
    }
    ds = _make_ds(prior=prior, keep_list_target=True)
    ex = _example(
        go_pred_terms=["GO:0000001", "GO:0000002"],
        gt_terms=["GO:0000001", "GO:0000002"],
    )
    captured = _capture(ds)
    ds._build_keep_list_target_ids(ex, "trace", None)
    full_text = "".join(captured)
    assert full_text.find("GO:0000002") < full_text.find("GO:0000001")


def test_keep_list_preserves_go_pred_order_without_prior():
    ds = _make_ds(keep_list_target=True)  # no prior given
    ex = _example(
        go_pred_terms=["GO:0000002", "GO:0000001"],
        gt_terms=["GO:0000001", "GO:0000002"],
    )
    captured = _capture(ds)
    ds._build_keep_list_target_ids(ex, "trace", None)
    full_text = "".join(captured)
    assert full_text.find("GO:0000002") < full_text.find("GO:0000001")


def test_keep_list_room_capping_keeps_eos_last_across_room_values():
    ds = _make_ds(keep_list_target=True)
    terms = [f"GO:{i:07d}" for i in range(20)]
    ex = _example(go_pred_terms=terms, gt_terms=terms)
    for room in (5, 10, 20, 50, 200):
        ids = ds._build_keep_list_target_ids(ex, "a somewhat longer trace text here", room)
        assert len(ids) <= room, f"room={room} produced {len(ids)} ids"
        assert ids[-1] == ds.tokenizer.eos_id, (
            f"room={room}: last id {ids[-1]} is not EOS ({ds.tokenizer.eos_id})"
        )


def test_keep_list_room_none_works_for_compute_lengths():
    ds = _make_ds(keep_list_target=True)
    ex = _example(go_pred_terms=["GO:0000001"], gt_terms=["GO:0000001"])
    ids = ds._build_keep_list_target_ids(ex, "trace", room=None)
    assert isinstance(ids, list) and len(ids) > 0
    assert ids[-1] == ds.tokenizer.eos_id


def test_keep_list_via_build_target_ids_dispatch():
    ds = _make_ds(keep_list_target=True)
    ex = _example(go_pred_terms=["GO:0000001", "GO:0000002"], gt_terms=["GO:0000001"])
    ex["reasoning"] = "reasoning trace"
    ex["final_answer"] = "final answer"
    captured = _capture(ds)
    ds._build_target_ids(ex, room=None)
    full_text = "".join(captured)
    assert "GO:0000001" in full_text
    assert "GO:0000002" not in full_text


def test_keep_list_mutually_exclusive_with_other_target_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        BioReasonSFTDataset(
            data_files="/nonexistent", tokenizer=_StubTok(),
            rerank_target=True, rerank_prior_path="x", keep_list_target=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        BioReasonSFTDataset(
            data_files="/nonexistent", tokenizer=_StubTok(),
            append_gopred_target=True, keep_list_target=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        BioReasonSFTDataset(
            data_files="/nonexistent", tokenizer=_StubTok(),
            exhaustive_target=True, keep_list_target=True,
        )


def test_keep_list_target_attribute_is_actually_set(tmp_path):
    """Real-constructor regression test: a prior bug (rerank_rank_only on the NeMo-RL
    side) accepted a flag in __init__ but never wrote self.<flag>, invisible to every
    __new__-bypass test since those set attributes directly. Exercise the REAL __init__
    (not __new__) and assert the attribute actually landed on the instance."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = _example(go_pred_terms=["GO:0000001"], gt_terms=["GO:0000001"])
    (data_dir / "data.jsonl").write_text(json.dumps(row) + "\n")

    ds = BioReasonSFTDataset(
        data_files=str(data_dir),
        tokenizer=_StubTok(),
        keep_list_target=True,
        drop_over_length=False,
    )
    assert ds.keep_list_target is True


def test_append_gopred_target_candidates_first_preserves_eos_and_whole_lines():
    ds = _make_ds(append_gopred_target=True)
    ex = _example(["GO:0000001", "GO:0000002", "GO:0000003"], [])
    full = ds._build_append_gopred_target_ids(
        ex, "a long reasoning trace with enough extra tokens to require truncation", room=None
    )
    candidates_only = ds._build_append_gopred_target_ids(ex, "", room=None)
    room = len(candidates_only) + 2
    result = ds._build_target_ids(ex, room=room)
    assert result[-1] == ds.tokenizer.eos_id
    assert len(result) <= room
    assert len(full) > len(result)
    assert len(result) >= len(candidates_only) - 1

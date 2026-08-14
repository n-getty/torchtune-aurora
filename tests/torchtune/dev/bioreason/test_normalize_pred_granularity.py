# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU-only regression test for experiments/bioreason/normalize_pred_granularity.py.
# Pins the parity-critical behavior: rbdgx3 writes one JSON per (protein, present
# aspect) while Aurora writes one JSON per protein — cafa_evals groups by
# (protein_id, go_aspect_code), so scoring the rbdgx3 layout unnormalized gives the
# same whole-protein generation extra weight in the mean. This test guards the
# collapse-to-one-file logic and its disagreement refusal.

import importlib.util
import json
import os
import sys

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "experiments", "bioreason", "normalize_pred_granularity.py",
)
_spec = importlib.util.spec_from_file_location("normalize_pred_granularity", _SCRIPT_PATH)
npg = importlib.util.module_from_spec(_spec)
sys.modules["normalize_pred_granularity"] = npg
_spec.loader.exec_module(npg)


def _write(dirpath, fn, rec):
    with open(os.path.join(dirpath, fn), "w") as f:
        json.dump(rec, f)


def test_three_aspect_files_collapse_to_one(tmp_path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    src.mkdir()

    common = {
        "protein_id": "P12345",
        "generated_response": "GO:0000001 GO:0000002",
        "go_bp": ["GO:0000001"],
        "go_mf": ["GO:0000002"],
        "go_cc": [],
        "success": True,
        "protein_sequence": "MKT",
    }
    for aspect in ("BP", "CC", "MF"):
        rec = dict(common)
        rec["go_aspect"] = aspect
        _write(str(src), f"P12345_{aspect}_k00.json", rec)

    rc = npg.normalize(str(src), str(dst))
    assert rc == 0

    out_files = os.listdir(str(dst))
    assert out_files == ["P12345_all_k00.json"]

    with open(dst / "P12345_all_k00.json") as f:
        merged = json.load(f)
    assert merged["go_aspect"] == "all"
    assert merged["generated_response"] == common["generated_response"]
    assert merged["go_bp"] == ["GO:0000001"]
    assert sorted(merged["_normalized_from_aspects"]) == ["BP", "CC", "MF"]


def test_already_all_granularity_passes_through(tmp_path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    src.mkdir()

    rec = {
        "protein_id": "P99999",
        "go_aspect": "all",
        "generated_response": "GO:0000003",
        "go_bp": [], "go_mf": ["GO:0000003"], "go_cc": [],
    }
    _write(str(src), "P99999_all_k00.json", rec)

    rc = npg.normalize(str(src), str(dst))
    assert rc == 0
    assert os.listdir(str(dst)) == ["P99999_all_k00.json"]


def test_mismatched_response_across_aspects_is_refused(tmp_path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    src.mkdir()

    base = {
        "protein_id": "P00000",
        "go_bp": ["GO:0000001"], "go_mf": [], "go_cc": [],
    }
    rec_bp = dict(base, go_aspect="BP", generated_response="GO:0000001")
    rec_mf = dict(base, go_aspect="MF", generated_response="DIFFERENT RESPONSE")
    _write(str(src), "P00000_BP_k00.json", rec_bp)
    _write(str(src), "P00000_MF_k00.json", rec_mf)

    rc = npg.normalize(str(src), str(dst))
    assert rc == 1
    assert os.listdir(str(dst)) == []


def test_mismatched_gt_column_across_aspects_is_refused(tmp_path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    src.mkdir()

    base = {
        "protein_id": "P11111",
        "generated_response": "same response",
    }
    rec_bp = dict(base, go_aspect="BP", go_bp=["GO:0000001"], go_mf=[], go_cc=[])
    rec_cc = dict(base, go_aspect="CC", go_bp=["GO:0000002"], go_mf=[], go_cc=[])
    _write(str(src), "P11111_BP_k00.json", rec_bp)
    _write(str(src), "P11111_CC_k00.json", rec_cc)

    rc = npg.normalize(str(src), str(dst))
    assert rc == 1
    assert os.listdir(str(dst)) == []


def test_empty_input_dir_is_fatal(tmp_path):
    src = tmp_path / "in"
    dst = tmp_path / "out"
    src.mkdir()

    rc = npg.normalize(str(src), str(dst))
    assert rc == 1

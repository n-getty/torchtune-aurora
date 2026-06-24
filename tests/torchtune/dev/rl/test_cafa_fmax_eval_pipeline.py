"""End-to-end (data-free, XPU-free) pin-down for the CAFA5 F_max eval pipeline.

The eval driver (experiments/bioreason/eval_cafa_fmax.py) hands per-sample prediction
JSONs to the BioReason-Pro paper's UNMODIFIED scorer (evals/cafa_evals.py -> cafaeval).
The gated wanglab/cafa5 dataset is NOT needed to prove the generate->emit->score wiring
is correct — only the SCHEMA and the metric are exercised here. This test fails loudly
if any of these silently drift before the real-data run:

  1. The driver's placeholder-expansion + GO-regex helpers match dataset.py byte-for-byte
     (a mismatch means the eval prompt differs from what training/rollout feed the model).
  2. make_record() emits exactly the fields the OFFICIAL scorer mode consumes
     (`--reasoning_mode True`: GT from go_bp/go_mf/go_cc lists, preds from generated_response).
  3. The paper's scorer, run unmodified over our emitted JSONs, yields a valid F_max:
     perfect predictions -> ~1.0, empty predictions -> ~0.0. This is the eval-correctness
     gate: if the schema were wrong, cafa_eval would score 0 or crash.

Run on a login node (no XPU):
  PYTHONNOUSERSITE=1 PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC \
    pytest tests/torchtune/dev/rl/test_cafa_fmax_eval_pipeline.py --timeout=120 -v
"""
import importlib.util
import json
import os
import sys

import pytest

_REPO = "/lus/flare/projects/ModCon/ngetty/torchtune"
_BIOREASON_SRC = os.environ.get(
    "BIOREASON_SRC", "/lus/flare/projects/ModCon/ngetty/BioReason-Pro"
)
_OBO = os.path.join(_BIOREASON_SRC, "bioreason2", "dataset", "go-basic.obo")
_DRIVER = os.path.join(_REPO, "experiments", "bioreason", "eval_cafa_fmax.py")


def _load_driver():
    spec = importlib.util.spec_from_file_location("eval_cafa_fmax", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1. Pure-helper contract (no torch, no data) ─────────────────────────────────

def test_aspect_code_matches_paper():
    drv = _load_driver()
    # Verbatim from BioReason-Pro/eval.py:36.
    assert drv.aspect_code("molecular_function") == "MF"
    assert drv.aspect_code("biological_process") == "BP"
    assert drv.aspect_code("cellular_component") == "CC"
    assert drv.aspect_code("all") == "all"  # passthrough for non-split


def test_go_regex_matches_scorer():
    drv = _load_driver()
    text = "Final: <|GO_SUMMARY_START|>GO:0008150, GO:0003674 and junk GO:12 GO:003"
    found = sorted(set(drv._GO_RE.findall(text)))
    # 7-digit only, matching cafa_evals.extract_go_terms' r"GO:\d{7}"
    # (too-short IDs like GO:12 / GO:003 are correctly ignored).
    assert found == ["GO:0003674", "GO:0008150"]


def test_as_list_normalizes_go_columns():
    drv = _load_driver()
    assert drv._as_list(None) == []
    assert drv._as_list([]) == []
    assert drv._as_list(["GO:0000001", "GO:0000002"]) == ["GO:0000001", "GO:0000002"]
    # stringified list (datasets sometimes hands these back)
    assert drv._as_list("['GO:0000001', 'GO:0000002']") == ["GO:0000001", "GO:0000002"]
    # bare GO terms in a string
    assert drv._as_list("GO:0000001 GO:0000002") == ["GO:0000001", "GO:0000002"]


def test_placeholder_expansion_matches_dataset():
    """build_input_ids must expand placeholders by the SAME formula as dataset.py:
    protein = len(seq)+2, GO = num_go_tokens, each replaced exactly once."""
    drv = _load_driver()
    seq = "MKT"  # len 3 -> 5 protein placeholders
    prompt = f"x {drv._PROTEIN_PAD} y {drv._GO_PAD} z"

    class _Tok:
        # encode returns the raw string so we can count placeholders post-expansion.
        def encode(self, s):
            return [ord(c) % 7 for c in s], s  # not used; see monkeypatch below

    # Use a tokenizer that returns the expanded string's placeholder counts.
    captured = {}

    class _CountTok:
        def encode(self, s):
            captured["s"] = s
            return [1, 2, 3]

    import torch  # noqa: F401  (only here to confirm torch is importable in this env)
    out = drv.build_input_ids(prompt, seq, _CountTok(), num_go_tokens=200)
    s = captured["s"]
    assert s.count(drv._PROTEIN_PAD) == len(seq) + 2 == 5
    assert s.count(drv._GO_PAD) == 200
    assert out.tolist() == [1, 2, 3]


def test_placeholder_formula_agrees_with_dataset_source():
    """Cross-check the +2 protein / num_go formula against the actual dataset.py text,
    so a change there without a change here is caught."""
    ds_path = os.path.join(_REPO, "torchtune", "dev", "bioreason", "dataset.py")
    src = open(ds_path).read()
    assert "len(protein_seq) + 2" in src, "dataset.py protein placeholder formula moved"
    assert "PROTEIN_PAD * protein_placeholders_count" in src
    assert "GO_PAD, GO_PAD * self.num_go_tokens" in src


# ── 2. Emitted-record schema ────────────────────────────────────────────────────

def test_make_record_has_reasoning_mode_fields():
    drv = _load_driver()
    sample = {
        "protein_id": "P12345",
        "go_aspect": "molecular_function",
        "sequence": "MKTAYIAK",
        "go_mf": ["GO:0003674"],
        "go_bp": [],
        "go_cc": [],
        "ground_truth": "GO:0003674",
    }
    rec = drv.make_record(sample, "answer <|GO_SUMMARY_START|>GO:0003674<|GO_SUMMARY_END|>")
    # OFFICIAL scorer (reasoning_mode) reads these list columns + generated_response:
    for k in ("protein_id", "go_aspect", "generated_response", "success",
              "go_bp", "go_mf", "go_cc"):
        assert k in rec, f"missing scorer field {k}"
    assert rec["success"] is True
    assert rec["go_mf"] == ["GO:0003674"]
    assert "GO:0003674" in rec["generated_response"]


# ── 2b. Local-parquet eval path (data on hand — no gated wanglab/cafa5) ──────────

_RL_PARQUET = "/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl"


@pytest.mark.skipif(
    not os.path.isdir(_RL_PARQUET) or importlib.util.find_spec("bioreason2") is None,
    reason="RL parquet or BioReason-Pro src not available",
)
def test_local_parquet_path_yields_faithful_samples():
    """The data-on-hand path renders the paper's per-aspect prompt over our LOCAL RL
    parquet (same columns the paper formatter consumes), so eval needs no gated data.
    Pins: one sample per present aspect, real GO-term GT, chat prompt with
    protein+go_graph+text blocks."""
    import argparse
    drv = _load_driver()
    args = argparse.Namespace(
        local_parquet=_RL_PARQUET, seed=23, max_samples=15, max_protein_len=2048,
        interpro_in_prompt=True, ppi_in_prompt=True, include_go_defs=False,
    )
    samples = drv.load_local_parquet_samples(args)
    assert len(samples) > 0
    aspects = {s["go_aspect"] for s in samples}
    assert aspects <= {"molecular_function", "biological_process", "cellular_component"}
    s0 = samples[0]
    assert isinstance(s0["prompt"], list)
    user = [m for m in s0["prompt"] if m.get("role") == "user"][0]
    block_types = [b.get("type") for b in user["content"]]
    assert "protein" in block_types and "go_graph" in block_types
    # GT for the asked aspect is non-empty for at least one sample (the formatter only
    # emits an aspect example when that aspect has labels).
    col = drv.ASPECT_TO_COLUMN[drv.aspect_code(s0["go_aspect"])]
    assert any(len(s[drv.ASPECT_TO_COLUMN[drv.aspect_code(s["go_aspect"])]]) > 0
               for s in samples)


# ── 3. The paper's UNMODIFIED scorer over our emitted JSONs ──────────────────────

def _write_pred_dir(drv, tmp_path, samples_and_responses):
    """Emit eval.py-schema JSONs exactly as the driver's main loop does."""
    out = tmp_path / "preds"
    out.mkdir()
    for s, resp in samples_and_responses:
        rec = drv.make_record(s, resp)
        rec["input_prompt"] = "stub"
        fn = f"{s['protein_id']}_{drv.aspect_code(s['go_aspect'])}_k00.json"
        (out / fn).write_text(json.dumps(rec, indent=2))
    return str(out)


def _score(input_dir):
    """Run the paper's cafa_evals processing + cafaeval over our JSONs, reasoning_mode.

    Imports the scorer from BIOREASON_SRC. go_dag=None (as cafa_evals.main sets it).
    Returns the overall mean F_max (best-threshold F across namespaces).
    """
    pytest.importorskip("cafaeval")
    sys.path.insert(0, os.path.join(_BIOREASON_SRC, "evals"))
    sys.path.insert(0, _BIOREASON_SRC)
    import cafa_evals as ce
    from cafaeval.evaluation import cafa_eval

    preds, gts = ce.process_json_data(
        input_dir, reasoning_mode=True, final_answer_only=False, go_dag=None,
    )
    if not preds or not gts:
        return 0.0, preds, gts
    # Write CAFA-format files exactly like cafa_evals.main.
    pred_dir = os.path.join(input_dir, "_cafa_pred")
    os.makedirs(pred_dir, exist_ok=True)
    pred_tsv = os.path.join(pred_dir, "llm_predictions.tsv")
    gt_tsv = os.path.join(input_dir, "ground_truth.tsv")
    ce.create_cafa_prediction_file(preds, pred_tsv)
    ce.create_cafa_ground_truth_file(gts, gt_tsv)
    # No IA.txt -> unweighted F_max. th_step=0.99 because every score is 1.0.
    # NOTE: the paper's ce.extract_metrics_summary() hard-requires the weighted-F
    # column ('f_w'), which cafaeval only emits when an IA file is passed — so we
    # read the unweighted best-F ('f') df directly here. The real-data run with
    # IA.txt can use extract_metrics_summary unchanged. (Documented gotcha.)
    evaluation_df, best = cafa_eval(_OBO, pred_dir, gt_tsv, th_step=0.99)
    fdf = best.get("f")
    overall = float(fdf.reset_index()["f"].mean()) if fdf is not None else 0.0
    return overall, preds, gts


# Real GO terms from the shipped go-basic.obo (rel 2023-01-01), one set per aspect.
_GT = {
    "molecular_function": ["GO:0000005", "GO:0000006"],
    "biological_process": ["GO:0000001", "GO:0000002"],
    "cellular_component": ["GO:0000015", "GO:0000108"],
}


@pytest.mark.skipif(not os.path.exists(_OBO), reason="go-basic.obo not present")
def test_perfect_predictions_score_high(tmp_path):
    """Emit JSONs whose generated_response contains the exact GT terms -> F_max ~1.0.
    This is the eval-correctness gate: a wrong schema would score 0 or crash."""
    drv = _load_driver()
    sar = []
    for aspect, terms in _GT.items():
        col = drv.ASPECT_TO_COLUMN[drv.aspect_code(aspect)]
        sample = {
            "protein_id": f"PROT_{drv.aspect_code(aspect)}",
            "go_aspect": aspect,
            "sequence": "MKTAYIAKQR",
            "go_bp": [], "go_mf": [], "go_cc": [],
            "ground_truth": " ".join(terms),
        }
        sample[col] = terms
        resp = "reasoning... <|GO_SUMMARY_START|>" + " ".join(terms) + "<|GO_SUMMARY_END|>"
        sar.append((sample, resp))
    input_dir = _write_pred_dir(drv, tmp_path, sar)
    score, preds, gts = _score(input_dir)
    assert len(preds) == 3 and len(gts) == 3, (preds, gts)
    assert score > 0.99, f"perfect predictions should score ~1.0, got {score}"


@pytest.mark.skipif(not os.path.exists(_OBO), reason="go-basic.obo not present")
def test_empty_predictions_score_zero(tmp_path):
    """No GO terms in any response -> no predictions -> F_max 0 (scorer returns early)."""
    drv = _load_driver()
    sar = []
    for aspect, terms in _GT.items():
        col = drv.ASPECT_TO_COLUMN[drv.aspect_code(aspect)]
        sample = {
            "protein_id": f"PROT_{drv.aspect_code(aspect)}",
            "go_aspect": aspect,
            "sequence": "MKTAYIAKQR",
            "go_bp": [], "go_mf": [], "go_cc": [],
            "ground_truth": " ".join(terms),
        }
        sample[col] = terms
        sar.append((sample, "no go terms here at all"))
    input_dir = _write_pred_dir(drv, tmp_path, sar)
    score, preds, gts = _score(input_dir)
    assert len(preds) == 0, "responses with no GO terms must yield no predictions"

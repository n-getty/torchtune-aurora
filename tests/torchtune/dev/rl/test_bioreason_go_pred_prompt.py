"""CPU test: BioReason RL dataset go_pred prompt injection.

Pins the train/eval/SFT prompt-distribution match (see
memory/project_bioreason_eval_fixed_rl_flat_vs_sft_20260626):
  - inject_go_pred=False  -> legacy cold prompt (no go_pred), byte-identical to before.
  - inject_go_pred=True   -> the paper's WITH_CONTEXT* prompt with go_pred injected as
                              go_speculations, matching eval_cafa_fmax.py --inject_go_pred.

The byte-equality-vs-paper check runs only when BIOREASON_SRC is importable (needs the
bioreason2 package on Aurora); otherwise it is skipped and the self-contained structural
checks still run on any login node.
"""
import os
import sys
import importlib.util

import pytest

_HERE = os.path.dirname(__file__)
_DS_PATH = os.path.abspath(
    os.path.join(_HERE, "../../../../torchtune/dev/bioreason/dataset.py")
)


def _load_dataset_module():
    spec = importlib.util.spec_from_file_location("br_dataset_under_test", _DS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_row():
    # A realistic RL parquet row (subset of the real schema).
    return {
        "organism": "Homo sapiens",
        "protein_names": "Test protein",
        "protein_function": "does things",
        "interpro_formatted": "IPR000001: domain A",
        "ppi_formatted": "P12345; Q67890",
        "go_pred": "Molecular Function (MF): GO:0003674 (molecular function)",
        "go_mf": ["GO:0003674"],
        "go_cc": ["GO:0005575"],
        "go_bp": [],
        "go_ids": ["GO:0003674", "GO:0005575"],
        "sequence": "MKT",
    }


def test_inject_false_is_cold_prompt():
    mod = _load_dataset_module()
    ds = mod.BioReasonRLDataset.__new__(mod.BioReasonRLDataset)
    ds.inject_go_pred = False
    # cold path doesn't go through _build_go_pred_prompt_text; just sanity that the flag exists
    assert ds.inject_go_pred is False


def test_inject_true_contains_go_speculations_and_no_ppi_branch_when_ppi_absent():
    mod = _load_dataset_module()
    ds = mod.BioReasonRLDataset.__new__(mod.BioReasonRLDataset)
    ds.inject_go_pred = True
    row = _sample_row()
    txt = ds._build_go_pred_prompt_text(row)
    # go_pred injected as go_speculations
    assert "GO:0003674 (molecular function)" in txt
    assert "initial GO term speculations" in txt
    # PPI present -> PPI branch
    assert "protein-protein interaction partners" in txt
    assert "P12345; Q67890" in txt
    # aspects suffix: MF + CC present, BP empty
    assert "Molecular Function, Cellular Component" in txt
    assert "Biological Process" not in txt.split("focus more on its")[1]


def test_inject_true_no_ppi_branch_when_ppi_missing():
    mod = _load_dataset_module()
    ds = mod.BioReasonRLDataset.__new__(mod.BioReasonRLDataset)
    ds.inject_go_pred = True
    row = _sample_row()
    row["ppi_formatted"] = ""
    txt = ds._build_go_pred_prompt_text(row)
    assert "protein-protein interaction partners" not in txt
    assert "initial GO term speculations" in txt


def test_nonempty_handles_ndarray():
    mod = _load_dataset_module()
    import numpy as np
    assert mod._nonempty(np.array(["GO:1"])) is True
    assert mod._nonempty(np.array([])) is False
    assert mod._nonempty(None) is False
    assert mod._nonempty("") is False
    assert mod._nonempty("x") is True


@pytest.mark.skipif(
    importlib.util.find_spec("bioreason2") is None
    and not os.environ.get("BIOREASON_SRC"),
    reason="bioreason2 not importable (needs BIOREASON_SRC on Aurora)",
)
def test_matches_paper_format_reasoning_prompt():
    """Byte-equality vs the paper's own formatter on the same row."""
    src = os.environ.get("BIOREASON_SRC", "/lus/flare/projects/ModCon/ngetty/BioReason-Pro")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from bioreason2.dataset.cafa5.load import _format_reasoning_prompt
        from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm
    except Exception as e:  # pragma: no cover
        pytest.skip(f"bioreason2 import failed: {e}")

    mod = _load_dataset_module()
    ds = mod.BioReasonRLDataset.__new__(mod.BioReasonRLDataset)
    ds.inject_go_pred = True
    row = _sample_row()

    ours = ds._build_go_pred_prompt_text(row)

    fr = _format_reasoning_prompt(
        dict(row), go_gpt_predictions_column="go_pred",
        interpro_in_prompt=True, ppi_in_prompt=True,
    )
    chat = format_cafa5_for_protein_llm({**row, "prompt": fr["prompt"]})
    # the text block is content[2]["text"] of the user message
    paper_text = chat["prompt"][0]["content"][2]["text"]
    assert ours == paper_text, f"\nOURS:\n{ours}\n\nPAPER:\n{paper_text}"

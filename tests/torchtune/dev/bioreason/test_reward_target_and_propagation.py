"""CPU-safe regression tests for the BioReason RL reward fix (2026-06-22).

Two bugs were found and fixed after the off-the-shelf RL ckpt scored BELOW SFT and our
own LoRA-GRPO run had flat reward / successes=0:

  1. WRONG TARGET: the reward scored vs `go_pred` (the GO-GPT submodel's noisy
     PREDICTIONS, ~50% recall of truth) instead of `go_ids` (== go_bp∪go_mf∪go_cc ==
     the eval F_max ground truth). dataset.py now defaults answer_column="go_ids".

  2. NO HIERARCHY: the reward did EXACT GO-term matching, but the eval metric
     (cafaeval) credits is_a ANCESTORS. Exact-match F1 vs the correct target is still
     ~50% zeros (mean 0.04 on real rollouts) — too flat to learn. Propagating both
     sets to their ancestor closure makes it dense (mean 0.23) AND matches eval.

These tests pin both WITHOUT XPU / a checkpoint / the real ontology (a tiny synthetic
DAG stands in for goatools where needed).
"""
import pytest

torch = pytest.importorskip("torch")

from torchtune.dev.bioreason.reward import (
    weighted_f_score,
    propagate_go_terms,
    bioreason_reward_fn,
)


# ── 1. dataset reward target is go_ids, not go_pred ──────────────────────────────

def test_dataset_default_answer_column_is_go_ids():
    """The reward target MUST default to the clean GT column, never go_pred."""
    import inspect
    from torchtune.dev.bioreason.dataset import BioReasonRLDataset, bioreason_rl_dataset

    assert inspect.signature(BioReasonRLDataset.__init__).parameters[
        "answer_column"].default == "go_ids"
    assert inspect.signature(bioreason_rl_dataset).parameters[
        "answer_column"].default == "go_ids"


def test_dataset_source_has_no_go_pred_default():
    """Guard against silently reverting answer = ex.get('go_pred')."""
    import torchtune.dev.bioreason.dataset as ds
    src = open(ds.__file__).read()
    assert 'ex.get("go_pred"' not in src and "ex.get('go_pred'" not in src, \
        "reward target must not read go_pred (GO-GPT predictions), use go_ids"
    assert "self.answer_column" in src


# ── 2. exact-match F1 baseline (no propagation) is unchanged ─────────────────────

def test_exact_f1_unchanged_without_propagation():
    pred = {"GO:0000001", "GO:0000002"}
    gt = {"GO:0000002", "GO:0000003"}
    # tp=1, p=1/2, r=1/2 -> F1=0.5
    assert weighted_f_score(pred, gt, propagate=False) == pytest.approx(0.5)


def test_both_empty_is_one_no_pred_is_zero():
    assert weighted_f_score(set(), set(), propagate=False) == 1.0
    assert weighted_f_score({"GO:0000001"}, set(), propagate=False) == 0.0
    assert weighted_f_score(set(), {"GO:0000001"}, propagate=False) == 0.0


# ── 3. DAG propagation expands to ancestors and densifies the signal ─────────────

class _FakeTerm:
    def __init__(self, parents):
        self._parents = parents

    def get_all_parents(self):
        return set(self._parents)


class _FakeDag:
    """Minimal stand-in for goatools GODag: a chain
    GO:child -> GO:mid -> GO:root (is_a ancestors)."""
    def __init__(self):
        self._d = {
            "GO:0000003": _FakeTerm({"GO:0000002", "GO:0000001"}),  # child
            "GO:0000002": _FakeTerm({"GO:0000001"}),                # mid
            "GO:0000001": _FakeTerm(set()),                          # root
        }

    def __contains__(self, k):
        return k in self._d

    def __getitem__(self, k):
        return self._d[k]


def test_propagate_includes_self_and_ancestors():
    dag = _FakeDag()
    assert propagate_go_terms({"GO:0000003"}, dag) == {
        "GO:0000003", "GO:0000002", "GO:0000001"}
    # unknown term passes through unchanged
    assert propagate_go_terms({"GO:9999999"}, dag) == {"GO:9999999"}
    # no dag -> identity
    assert propagate_go_terms({"GO:0000003"}, None) == {"GO:0000003"}


def test_propagation_densifies_reward_vs_exact():
    """The core fix: predicting a deep term that shares ancestors with the GT scores
    0 under exact match but >0 under propagation (credit for shared ancestors)."""
    dag = _FakeDag()
    pred = {"GO:0000003"}   # child; propagates to {child, mid, root}
    gt = {"GO:0000002"}     # mid;   propagates to {mid, root}
    exact = weighted_f_score(pred, gt, propagate=False)
    prop = weighted_f_score(pred, gt, propagate=True, dag=dag)
    assert exact == 0.0, "exact match: no shared term -> 0"
    assert prop > 0.0, "propagation: shares {mid, root} -> partial credit"
    # pred_prop={c,m,r}, gt_prop={m,r}: tp=2, p=2/3, r=2/2 -> F1=2*(2/3)/(2/3+1)=0.8
    assert prop == pytest.approx(0.8)


# ── 4. bioreason_reward_fn threads propagate_hierarchy + tolerates missing DAG ───

def test_reward_fn_propagate_flag_off_is_exact():
    # completion text + answer string -> regex-extracted GO terms, exact F1.
    comp = ["the answer is GO:0000001 and GO:0000002"]
    ans = ["GO:0000002, GO:0000003"]
    rw, succ = bioreason_reward_fn(comp, ans, propagate_hierarchy=False)
    assert rw[0].item() == pytest.approx(0.5)


def test_resolve_obo_path_handles_directory(tmp_path):
    """Regression: base_model_path is a DIRECTORY (ckpt dir ships go-basic.obo). A dir
    passed to GODag fails 'COULD NOT READ' — _resolve_obo_path must find the obo file
    inside it, not return the dir. (This bug silently disabled propagation in the first
    2-node smoke: obo_path=/tmp/.../bioreason-pro-sft -> fell back to exact match.)"""
    from torchtune.dev.bioreason.reward import _resolve_obo_path
    # directory containing go-basic.obo -> resolves to the file
    (tmp_path / "go-basic.obo").write_text("format-version: 1.2\n")
    assert _resolve_obo_path(str(tmp_path)) == str(tmp_path / "go-basic.obo")
    # direct file path -> passthrough
    assert _resolve_obo_path(str(tmp_path / "go-basic.obo")) == str(tmp_path / "go-basic.obo")
    # directory WITHOUT an obo -> not resolved to the dir (would crash GODag)
    empty = tmp_path / "empty"; empty.mkdir()
    got = _resolve_obo_path(str(empty))
    assert got is None or got.endswith("go-basic.obo")


def test_reward_fn_propagate_missing_dag_falls_back_to_exact(monkeypatch):
    # If the obo can't be found, load_go_dag returns None and scoring stays exact
    # (never crashes the training step).
    import torchtune.dev.bioreason.reward as R
    monkeypatch.setattr(R, "_resolve_obo_path", lambda p: None)
    comp = ["GO:0000001 GO:0000002"]
    ans = ["GO:0000002 GO:0000003"]
    rw, succ = bioreason_reward_fn(comp, ans, propagate_hierarchy=True,
                                   obo_path="/nonexistent.obo")
    assert rw[0].item() == pytest.approx(0.5)  # fell back to exact

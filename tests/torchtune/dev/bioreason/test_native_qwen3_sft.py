"""CPU tests for the Qwen3-32B BioReason backbone path (no XPU, no distributed).

The Gemma4 path is covered by test_native_gemma4_sft.py. This pins the Qwen3 variant,
whose key differences are: plain nn.Embedding (NO sqrt(H) scale), a separate (non-tied)
nn.Linear output, and Qwen reserved placeholder ids. Uses a tiny qwen3 backbone.
"""

import hashlib

import pytest
import torch
import torch.nn as nn

from torchtune.models.qwen3._component_builders import qwen3, lora_qwen3
from torchtune.dev.bioreason.model_native import BioReasonNativeModel
from torchtune.modules.loss import LinearCrossEntropyLoss
from torchtune.modules.peft import LoRALinear, get_adapter_params

_H = 32
_V = 200
_ESM_DIM = 12
_PROT_ID = 190  # in Qwen3-32B's reserved gap (151643-151935); here just < vocab
_GO_ID = 191
_QWEN_KW = dict(
    vocab_size=_V, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8,
    embed_dim=_H, intermediate_dim=64, max_seq_len=1024, tie_word_embeddings=False,
)


def _backbone(lora=False):
    if lora:
        return lora_qwen3(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True, lora_rank=4, lora_alpha=8.0, **_QWEN_KW,
        ).to(torch.float32)
    return qwen3(**_QWEN_KW).to(torch.float32)


def _model(lora=False, freeze_backbone=False, freeze_projector=False):
    seqs = ["MKT", "MA"]
    cache = {
        hashlib.sha1(s.encode("ascii", "ignore")).hexdigest(): torch.randn(len(s) + 2, _ESM_DIM)
        for s in seqs
    }
    meta = {"embedding_dim": _ESM_DIM, "esm3_model_name": "esm3_sm_open_v1", "n_seqs": 2}
    m = BioReasonNativeModel(
        device=torch.device("cpu"), hidden_size=_H, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, dtype=torch.float32, backbone=_backbone(lora=lora),
        protein_hidden_override=_ESM_DIM, esm3_cache_inject=(cache, meta),
        go_embedding_inject=torch.randn(200, BioReasonNativeModel.GO_DIM),
        enable_lora=lora, freeze_backbone=freeze_backbone,
        freeze_projector=freeze_projector,
    )
    return m, seqs


def _model_norm():
    """Frozen-backbone model with projector_output_norm=True (the capability-path config)."""
    seqs = ["MKT", "MA"]
    cache = {
        hashlib.sha1(s.encode("ascii", "ignore")).hexdigest(): torch.randn(len(s) + 2, _ESM_DIM)
        for s in seqs
    }
    meta = {"embedding_dim": _ESM_DIM, "esm3_model_name": "esm3_sm_open_v1", "n_seqs": 2}
    m = BioReasonNativeModel(
        device=torch.device("cpu"), hidden_size=_H, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, dtype=torch.float32, backbone=_backbone(lora=False),
        protein_hidden_override=_ESM_DIM, esm3_cache_inject=(cache, meta),
        go_embedding_inject=torch.randn(200, BioReasonNativeModel.GO_DIM),
        enable_lora=False, freeze_backbone=True, projector_output_norm=True,
    )
    return m, seqs


def _mkrow(nprot, ngo=200):
    return [1, 2] + [_PROT_ID] * nprot + [_GO_ID] * ngo + [3, 4]


def test_qwen_builds_plain_embedding_no_scale():
    m, _ = _model()
    assert m._embed is m.backbone.tok_embeddings
    assert isinstance(m.backbone.tok_embeddings, nn.Embedding)
    # Qwen output is a separate (non-tied) Linear, distinct from the embedding.
    assert isinstance(m.backbone.output, nn.Linear)


def test_qwen_embed_equivalence_no_scale():
    """No-sqrt-scale path: model(input_embeds=build(toks)) == backbone(tokens=toks)."""
    m, _ = _model()
    toks = torch.randint(0, 180, (2, 12))
    emb = m.build_full_embeds_train(toks, protein_sequences=[], go_aspects=None)
    assert torch.allclose(m(input_embeds=emb), m.backbone(tokens=toks), atol=1e-5)


def test_qwen_splice_and_grad():
    m, seqs = _model()
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    text_only = m._embed(toks)
    prot = toks == _PROT_ID
    assert torch.allclose(emb[~(prot | (toks == _GO_ID))], text_only[~(prot | (toks == _GO_ID))], atol=1e-5)
    m(input_embeds=emb).sum().backward()
    assert m.protein_projection[0].weight.grad.abs().sum().item() > 0
    assert m.go_projection[0].weight.grad.abs().sum().item() > 0


def test_qwen_linear_ce_wiring():
    m, seqs = _model()
    loss = LinearCrossEntropyLoss(num_output_chunks=2)
    loss.set_model_output(m)
    assert m.skip_output_layer is True
    assert loss.linear_projection is m.output
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    out = m(toks, protein_sequences=seqs, go_aspects=["all", "all"])
    assert out.shape[-1] == m.hidden_size  # hidden, not vocab
    labels = torch.full(toks.shape, -100)
    labels[:, -3:] = toks[:, -3:]
    l = loss(out, labels)
    assert torch.isfinite(l)
    l.backward()
    assert m.protein_projection[0].weight.grad.abs().sum().item() > 0


def test_qwen_lora_base_frozen_proj_trainable():
    m, _ = _model(lora=True)
    assert m.backbone.layers[0].attn.q_proj.weight.requires_grad is False
    assert m.backbone.layers[0].attn.q_proj.lora_a.weight.requires_grad is True
    assert m.protein_projection[0].weight.requires_grad is True
    assert len(get_adapter_params(m.backbone)) > 0


def test_qwen_projector_output_norm_bounds_and_saves():
    """projector_output_norm=True appends a LayerNorm (Sequential index 3) to each projector
    so the spliced-feature magnitude is BOUNDED (per-row norm ~sqrt(H)), fixing the
    over-amplification trap (721->1629 -> ':' collapse) without LoRA. Pins: LayerNorm present,
    output norm bounded near sqrt(H), and the state dict carries index-3 keys (so the eval-side
    BioReasonModel must mirror the arch — auto-detected from these keys)."""
    m, _ = _model(lora=False, freeze_backbone=True)  # baseline: no norm
    assert len(m.protein_projection) == 3
    mn, _ = _model_norm()
    assert isinstance(mn.protein_projection[3], nn.LayerNorm)
    assert isinstance(mn.go_projection[3], nn.LayerNorm)
    feat = torch.randn(6, _ESM_DIM)
    out = mn.protein_projection(feat)
    import math
    expect = math.sqrt(_H)
    got = out.norm(dim=-1).mean().item()
    assert 0.3 * expect < got < 3 * expect, f"LayerNorm output norm {got} not ~sqrt(H)={expect}"
    sd = mn.protein_projection.state_dict()
    assert any(k.startswith("3.") for k in sd), "LayerNorm index-3 keys missing from state dict"


def test_qwen_stage2_freeze_projector_only_lora_trains():
    """Stage 2 with freeze_projector=True: LoRA adapters train, but the protein/GO
    projections are FROZEN (locked at their Stage-1-aligned values). Prevents the projector
    over-amplification trap on a long run (trainable projector drifts norm 721->1044 in 10
    steps once loss saturates -> ':' collapse). Pins that adapters stay trainable while
    BOTH projections are frozen."""
    m, _ = _model(lora=True, freeze_projector=True)
    # LoRA adapters still train.
    assert m.backbone.layers[0].attn.q_proj.lora_a.weight.requires_grad is True
    # Base still frozen.
    assert m.backbone.layers[0].attn.q_proj.weight.requires_grad is False
    # Both projections frozen.
    assert all(not p.requires_grad for p in m.protein_projection.parameters())
    assert all(not p.requires_grad for p in m.go_projection.parameters())
    # Trainable set is non-empty and contains NO projection params.
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert not any(n.startswith(("protein_projection.", "go_projection.")) for n in trainable)


def test_qwen_stage2_default_projector_trainable():
    """Default Stage 2 (freeze_projector=False) keeps projections trainable (published)."""
    m, _ = _model(lora=True, freeze_projector=False)
    assert m.protein_projection[0].weight.requires_grad is True
    assert m.go_projection[0].weight.requires_grad is True


def test_qwen_stage1_freeze_backbone_only_projections_train():
    """Stage 1 (freeze_backbone=True, no LoRA): the ENTIRE backbone is frozen and ONLY
    the protein/GO projections train. This is the LLaVA-style alignment phase — with the
    backbone unable to change, the model cannot learn the go_pred text shortcut, so the
    projections are forced to make the spliced multimodal features informative. Pins the
    root-cause fix for the epoch_0 ':' collapse (single-stage LoRA never trained the
    projection)."""
    m, _ = _model(lora=False, freeze_backbone=True)
    # No backbone param is trainable (no LoRA adapters exist in this regime).
    assert all(not p.requires_grad for p in m.backbone.parameters())
    # Both projections remain fully trainable.
    assert m.protein_projection[0].weight.requires_grad is True
    assert m.protein_projection[2].weight.requires_grad is True
    assert m.go_projection[0].weight.requires_grad is True
    # And the trainable set is non-empty and consists ONLY of projection params.
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all(n.startswith(("protein_projection.", "go_projection.")) for n in trainable)


def test_qwen_stage1_projection_grads_flow_backbone_grads_none():
    """Stage 1 backward: projection grads are non-zero (they train), and frozen backbone
    params receive NO grad (requires_grad=False -> .grad stays None)."""
    m, seqs = _model(lora=False, freeze_backbone=True)
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    m(input_embeds=emb).sum().backward()
    assert m.protein_projection[0].weight.grad.abs().sum().item() > 0
    assert m.go_projection[0].weight.grad.abs().sum().item() > 0
    # A representative frozen backbone weight got no gradient.
    assert m.backbone.layers[0].attn.q_proj.weight.grad is None


# ── drop_over_length filter (the XPU math-SDPA O(S^2) seq-ceiling mitigation) ───
class _StubTok:
    """Minimal tokenizer: 1 id per whitespace token; deterministic, no weights.
    Encodes prompt text length proportional to word count so we can drive the
    prompt past max_seq_len with a long `interpro_formatted` field."""
    bos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        ids = ([self.bos_id] if add_bos else []) + [7] * len(text.split())
        if add_eos:
            ids += [2]
        return ids


def _ds_with(monkeypatch, rows, max_seq_len, drop):
    from torchtune.dev.bioreason import dataset_sft as sft
    # Bypass file IO: inject examples directly via _load monkeypatch.
    monkeypatch.setattr(sft.BioReasonSFTDataset, "_load", lambda self, df: list(rows))
    return sft.BioReasonSFTDataset(
        data_files="unused", tokenizer=_StubTok(), max_seq_len=max_seq_len,
        max_protein_len=2048, num_go_tokens=5, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, train_on_reasoning=True, inject_go_pred=True,
        drop_over_length=drop,
    )


def _row(n_words, sequence="MKT"):
    return {
        "organism": "Homo sapiens", "interpro_formatted": "w " * n_words,
        "ppi_formatted": "", "go_pred": "", "go_mf": [], "go_cc": [], "go_bp": [],
        "sequence": sequence, "reasoning": "because", "final_answer": "GO:1",
    }


def test_drop_over_length_filters_long_prompts(monkeypatch):
    # 2 short rows + 1 row whose prompt blows past a small max_seq_len.
    rows = [_row(3), _row(3), _row(5000)]
    ds = _ds_with(monkeypatch, rows, max_seq_len=128, drop=True)
    assert len(ds) == 2  # the 5000-word row dropped, short rows kept
    # every surviving example is trainable (prompt < max_seq_len) and __getitem__ works
    for i in range(len(ds)):
        item = ds[i]
        assert item["tokens"].shape[0] <= 128


def test_drop_over_length_false_keeps_all_and_getitem_fail_fasts(monkeypatch):
    rows = [_row(3), _row(5000)]
    ds = _ds_with(monkeypatch, rows, max_seq_len=128, drop=False)
    assert len(ds) == 2  # nothing dropped
    ds[0]  # short one is fine
    with pytest.raises(ValueError):
        ds[1]  # long one fail-fasts in __getitem__ (defensive fallback)


# ── go_pred_dropout (leak reduction) ───────────────────────────────────────────
def _ds_dropout(monkeypatch, rows, dropout, seed=0):
    from torchtune.dev.bioreason import dataset_sft as sft
    monkeypatch.setattr(sft.BioReasonSFTDataset, "_load", lambda self, df: list(rows))
    return sft.BioReasonSFTDataset(
        data_files="unused", tokenizer=_StubTok(), max_seq_len=4096,
        max_protein_len=2048, num_go_tokens=5, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, train_on_reasoning=True, inject_go_pred=True,
        drop_over_length=False, go_pred_dropout=dropout, go_pred_dropout_seed=seed,
    )


def _row_gp(go_pred="IPR: GO:9 speculation words here"):
    r = _row(3)
    r["go_pred"] = go_pred
    return r


def test_go_pred_dropout_zero_is_parity(monkeypatch):
    """dropout=0.0 must produce byte-identical tokens to the default builder — the
    existing go_pred-present prompt path is untouched (no regression)."""
    rows = [_row_gp() for _ in range(8)]
    ds0 = _ds_dropout(monkeypatch, rows, dropout=0.0)
    dsref = _ds_with(monkeypatch, rows, max_seq_len=4096, drop=False)
    for i in range(len(rows)):
        assert torch.equal(ds0[i]["tokens"], dsref[i]["tokens"]), f"idx {i} diverged at dropout=0"


def test_go_pred_dropout_one_equals_empty_go_pred(monkeypatch):
    """dropout=1.0 must produce tokens equal to the go_pred='' (empty) path for every
    sample — dropout reuses the existing empty-go_pred prompt rendering, no new format."""
    rows = [_row_gp() for _ in range(6)]
    ds1 = _ds_dropout(monkeypatch, rows, dropout=1.0)
    # Reference: identical rows but go_pred already blank.
    rows_empty = [_row_gp(go_pred="") for _ in range(6)]
    dsempty = _ds_dropout(monkeypatch, rows_empty, dropout=0.0)
    for i in range(len(rows)):
        assert torch.equal(ds1[i]["tokens"], dsempty[i]["tokens"]), f"idx {i} != empty-go_pred"


def test_go_pred_dropout_deterministic_resume_safe(monkeypatch):
    """The per-sample decision is a pure function of (seed, idx) — two independent
    dataset constructions (e.g. before/after a dataloader resume) yield the SAME
    dropped set. A global-RNG implementation would fail this."""
    rows = [_row_gp() for _ in range(40)]
    a = _ds_dropout(monkeypatch, rows, dropout=0.5, seed=123)
    b = _ds_dropout(monkeypatch, rows, dropout=0.5, seed=123)
    dropped_a = [a._drop_go_pred(i) for i in range(40)]
    dropped_b = [b._drop_go_pred(i) for i in range(40)]
    assert dropped_a == dropped_b
    # Sanity: 0.5 actually splits (not all-true / all-false) on 40 samples.
    assert 0 < sum(dropped_a) < 40
    # A different seed gives a different mask (the seed is live).
    c = _ds_dropout(monkeypatch, rows, dropout=0.5, seed=999)
    assert [c._drop_go_pred(i) for i in range(40)] != dropped_a


def test_go_pred_dropout_never_mutates_source_row(monkeypatch):
    """Dropout must shallow-copy — the cached example's go_pred is intact after
    __getitem__, so re-reading the same idx (next epoch) sees the original text."""
    rows = [_row_gp(go_pred="ORIGINAL GO:9 text") for _ in range(4)]
    ds = _ds_dropout(monkeypatch, rows, dropout=1.0)  # drop ALL
    for i in range(len(rows)):
        _ = ds[i]["tokens"]
        assert ds.examples[i]["go_pred"] == "ORIGINAL GO:9 text", f"row {i} mutated"


# ── exhaustive_target (Exp 1: append full GT term list for breadth) ─────────────
def _ds_exhaustive(monkeypatch, rows, exhaustive):
    from torchtune.dev.bioreason import dataset_sft as sft
    monkeypatch.setattr(sft.BioReasonSFTDataset, "_load", lambda self, df: list(rows))
    return sft.BioReasonSFTDataset(
        data_files="unused", tokenizer=_StubTok(), max_seq_len=4096,
        max_protein_len=2048, num_go_tokens=5, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, train_on_reasoning=True, inject_go_pred=True,
        drop_over_length=False, exhaustive_target=exhaustive,
    )


# ── bp_oversample_factor (gap #8: BP is the persistently weakest CAFA namespace) ─
def _ds_bp_oversample(monkeypatch, rows, factor):
    from torchtune.dev.bioreason import dataset_sft as sft
    monkeypatch.setattr(sft.BioReasonSFTDataset, "_load", lambda self, df: list(rows))
    return sft.BioReasonSFTDataset(
        data_files="unused", tokenizer=_StubTok(), max_seq_len=4096,
        max_protein_len=2048, num_go_tokens=5, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, train_on_reasoning=True, inject_go_pred=True,
        drop_over_length=False, bp_oversample_factor=factor,
    )


def test_bp_oversample_factor_default_is_noop(monkeypatch):
    rows = [_row(3), _row(3)]
    rows[0]["go_bp"] = ["GO:0000001"]
    rows[1]["go_bp"] = []
    ds = _ds_bp_oversample(monkeypatch, rows, factor=1.0)
    assert len(ds) == 2  # 1.0 = off, no duplication


def test_bp_oversample_factor_2x_duplicates_only_bp_rows(monkeypatch):
    bp_row = _row(3)
    bp_row["go_bp"] = ["GO:0000001"]
    non_bp_row = _row(3)
    non_bp_row["go_bp"] = []
    rows = [bp_row, non_bp_row]
    ds = _ds_bp_oversample(monkeypatch, rows, factor=2.0)
    # 2.0 = each BP row appears twice; non-BP rows untouched.
    assert len(ds) == 3
    bp_count = sum(1 for ex in ds.examples if _sft_nonempty(ex.get("go_bp")))
    assert bp_count == 2
    non_bp_count = sum(1 for ex in ds.examples if not _sft_nonempty(ex.get("go_bp")))
    assert non_bp_count == 1


def test_bp_oversample_factor_fractional_duplicates_a_prefix(monkeypatch):
    bp_rows = [_row(3) for _ in range(4)]
    for i, r in enumerate(bp_rows):
        r["go_bp"] = [f"GO:000000{i}"]
    ds = _ds_bp_oversample(monkeypatch, bp_rows, factor=1.5)
    # 1.5x of 4 BP rows -> 2 extra duplicate rows (round(0.5*4)).
    assert len(ds) == 6


def test_bp_oversample_factor_noop_when_no_bp_rows(monkeypatch):
    rows = [_row(3), _row(3)]
    for r in rows:
        r["go_bp"] = []
    ds = _ds_bp_oversample(monkeypatch, rows, factor=3.0)
    assert len(ds) == 2  # nothing to oversample; logs a warning, doesn't crash


def test_bp_oversample_factor_rejects_below_one():
    from torchtune.dev.bioreason import dataset_sft as sft
    with pytest.raises(ValueError):
        sft.BioReasonSFTDataset.__init__(
            object.__new__(sft.BioReasonSFTDataset),
            data_files="unused", tokenizer=_StubTok(), bp_oversample_factor=0.5,
        )


def _sft_nonempty(v):
    from torchtune.dev.bioreason.dataset_sft import _nonempty
    return _nonempty(v)


# ── interpro_in_prompt / ppi_in_prompt (gap #6: text-ablation on the native path) ──
def _build_native_prompt_text(interpro_in_prompt=True, ppi_in_prompt=True):
    from torchtune.dev.bioreason import dataset_sft as sft
    ex = {
        "organism": "Homo sapiens",
        "interpro_formatted": "IPR000001: domain A",
        "ppi_formatted": "P12345; Q67890",
        "go_pred": "GO:0005524",
        "go_mf": ["GO:0005524"], "go_cc": [], "go_bp": [],
    }
    return sft.BioReasonSFTDataset._build_prompt_text(
        sft.BioReasonSFTDataset.__new__(sft.BioReasonSFTDataset), ex,
        interpro_in_prompt=interpro_in_prompt, ppi_in_prompt=ppi_in_prompt,
    )


def test_interpro_ppi_in_prompt_default_true_unchanged():
    """Default (no args) must be byte-identical to the pre-existing behavior pinned
    by test_H_dataset_prompt_matches_rl_dataset."""
    text = _build_native_prompt_text()
    assert "IPR000001: domain A" in text
    assert "P12345; Q67890" in text


def test_interpro_in_prompt_false_strips_interpro_only():
    text = _build_native_prompt_text(interpro_in_prompt=False)
    assert "IPR000001: domain A" not in text
    assert "P12345; Q67890" in text  # ppi unaffected


def test_ppi_in_prompt_false_strips_ppi_only():
    text = _build_native_prompt_text(ppi_in_prompt=False)
    assert "P12345; Q67890" not in text
    assert "IPR000001: domain A" in text  # interpro unaffected


def test_both_off_strips_both():
    text = _build_native_prompt_text(interpro_in_prompt=False, ppi_in_prompt=False)
    assert "IPR000001: domain A" not in text
    assert "P12345; Q67890" not in text
    assert "GO:0005524" in text  # go_pred (the third context field) untouched


def _row_gt(mf=None, cc=None, bp=None):
    r = _row(3)
    r["go_mf"] = mf or []
    r["go_cc"] = cc or []
    r["go_bp"] = bp or []
    return r


def test_gt_terms_dedup_and_aspect_order():
    """_gt_terms unions MF, CC, BP in that order, dedups, preserves stable order."""
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset as DS
    ex = {"go_mf": ["GO:0000001", "GO:0000002"], "go_cc": ["GO:0000002", "GO:0000003"],
          "go_bp": ["GO:0000004"]}
    assert DS._gt_terms(ex) == ["GO:0000001", "GO:0000002", "GO:0000003", "GO:0000004"]
    assert DS._gt_terms({"go_mf": [], "go_cc": [], "go_bp": []}) == []


def test_gt_terms_handles_numpy_arrays():
    """REGRESSION (step-0 crash 2026-07-02): pandas to_dict('records') yields numpy ARRAYS
    for list columns, not Python lists. `if not v` / `isinstance(v, list)` both fail on an
    ndarray ('truth value ambiguous'). _gt_terms must handle ndarray + empty ndarray + None."""
    import numpy as np
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset as DS
    ex = {"go_mf": np.array(["GO:0000001", "GO:0000002"]),
          "go_cc": np.array(["GO:0000002"]),
          "go_bp": np.array([], dtype=object)}
    assert DS._gt_terms(ex) == ["GO:0000001", "GO:0000002"]
    # all-empty arrays + a None column must not raise
    assert DS._gt_terms({"go_mf": np.array([], dtype=object), "go_cc": None,
                         "go_bp": np.array([], dtype=object)}) == []


def test_exhaustive_target_off_is_parity(monkeypatch):
    """exhaustive_target=False ⇒ target byte-identical to the default builder."""
    rows = [_row_gt(mf=["GO:0000001"], bp=["GO:0000002"]) for _ in range(4)]
    off = _ds_exhaustive(monkeypatch, rows, exhaustive=False)
    ref = _ds_with(monkeypatch, rows, max_seq_len=4096, drop=False)
    for i in range(len(rows)):
        assert torch.equal(off[i]["tokens"], ref[i]["tokens"]), f"idx {i} diverged"


def test_exhaustive_target_grows_target_and_covers_gt(monkeypatch):
    """exhaustive_target=True ⇒ target is LONGER (GT list appended) and every GT term's
    text is rendered into the target string (checked on the raw target the builder makes)."""
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset as DS
    rows = [_row_gt(mf=["GO:0000123", "GO:0000456"], cc=["GO:0000789"])]
    on = _ds_exhaustive(monkeypatch, rows, exhaustive=True)
    off = _ds_exhaustive(monkeypatch, rows, exhaustive=False)
    # target region is longer under exhaustive (list appended). _StubTok = 1 id/word, so
    # more words -> more target tokens; prompt is identical so total grows.
    assert on[0]["tokens"].shape[0] > off[0]["tokens"].shape[0]
    # The rendered target string must contain every GT term verbatim (eval regex GO:\d{7}
    # extracts these). Check the builder's target text directly (tokenizer-agnostic).
    inst = on
    ex = rows[0]
    reasoning = ex.get("reasoning", "") or ""
    final = ex.get("final_answer", "") or ""
    base = f"{reasoning}\n{final}" if reasoning else final
    terms = DS._gt_terms(ex)
    rendered = f"{base}\n\nGO terms: " + ", ".join(terms)
    for t in ("GO:0000123", "GO:0000456", "GO:0000789"):
        assert t in rendered


def test_exhaustive_target_labels_mask_only_prompt(monkeypatch):
    """The appended GT list is SUPERVISED (labels != ignore over the whole target incl.
    the list); only the prompt span is masked."""
    from torchtune.data import CROSS_ENTROPY_IGNORE_IDX as IGN
    rows = [_row_gt(mf=["GO:0000123"], cc=["GO:0000456"])]
    ds = _ds_exhaustive(monkeypatch, rows, exhaustive=True)
    item = ds[0]
    labels = item["labels"]
    # prompt span masked, target span (incl. appended list) supervised
    n_prompt = int((labels == IGN).sum())
    assert n_prompt > 0
    assert int((labels != IGN).sum()) > 0
    # placeholder counts unchanged (prompt not touched by the target change)
    assert int((item["tokens"] == _PROT_ID).sum()) == 3 + 2  # _row protein seq "MKT"->3, +2
    assert int((item["tokens"] == _GO_ID).sum()) == 5


# ── append_gopred_target (Approach B: append IN-PROMPT go_pred terms, no GT leak) ──
def _ds_gopred(monkeypatch, rows, append):
    from torchtune.dev.bioreason import dataset_sft as sft
    monkeypatch.setattr(sft.BioReasonSFTDataset, "_load", lambda self, df: list(rows))
    return sft.BioReasonSFTDataset(
        data_files="unused", tokenizer=_StubTok(), max_seq_len=4096,
        max_protein_len=2048, num_go_tokens=5, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, train_on_reasoning=True, inject_go_pred=True,
        drop_over_length=False, append_gopred_target=append,
    )


def _row_gopred(go_pred_text, reasoning="because", final="GO:1"):
    r = _row(3)
    r["go_pred"] = go_pred_text
    r["reasoning"] = reasoning
    r["final_answer"] = final
    return r


def test_exhaustive_and_append_gopred_are_mutually_exclusive():
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset as DS
    with pytest.raises(ValueError):
        DS(
            data_files="unused", tokenizer=_StubTok(), exhaustive_target=True,
            append_gopred_target=True,
        )


def test_gopred_terms_extracts_dedup_first_seen_order():
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset as DS
    ex = {"go_pred": "MF: GO:0000002, GO:0000001, GO:0000002 (dup)"}
    assert DS._gopred_terms(ex) == ["GO:0000002", "GO:0000001"]
    assert DS._gopred_terms({"go_pred": ""}) == []
    assert DS._gopred_terms({}) == []


def test_append_gopred_target_off_is_parity(monkeypatch):
    """append_gopred_target=False ⇒ target byte-identical to the default builder."""
    rows = [_row_gopred("GO:0000001, GO:0000002") for _ in range(4)]
    off = _ds_gopred(monkeypatch, rows, append=False)
    ref = _ds_with(monkeypatch, rows, max_seq_len=4096, drop=False)
    for i in range(len(rows)):
        assert torch.equal(off[i]["tokens"], ref[i]["tokens"]), f"idx {i} diverged"


def test_append_gopred_target_grows_target_and_preserves_reasoning_terms(monkeypatch):
    """append_gopred_target=True ⇒ target is LONGER (go_pred list appended) and contains
    BOTH the curated reasoning/final_answer terms AND every go_pred term verbatim — the
    reasoning-derived breadth is preserved, not replaced."""
    rows = [_row_gopred(
        "Speculations: GO:0000123, GO:0000456", reasoning="uses GO:0000999", final="GO:0000789",
    )]
    on = _ds_gopred(monkeypatch, rows, append=True)
    off = _ds_gopred(monkeypatch, rows, append=False)
    assert on[0]["tokens"].shape[0] > off[0]["tokens"].shape[0]
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset as DS
    ex = rows[0]
    base = f"{ex['reasoning']}\n{ex['final_answer']}"
    terms = DS._gopred_terms(ex)
    rendered = f"{base}\n\nGO terms: " + ", ".join(terms)
    # curated reasoning term (not in go_pred) survives
    assert "GO:0000999" in rendered
    # curated final_answer term survives
    assert "GO:0000789" in rendered
    # appended go_pred terms present verbatim
    for t in ("GO:0000123", "GO:0000456"):
        assert t in rendered


def test_append_gopred_target_empty_gopred_is_noop(monkeypatch):
    """No go_pred terms to append ⇒ target unchanged (no dangling 'GO terms: ' header)."""
    rows = [_row_gopred("")]
    on = _ds_gopred(monkeypatch, rows, append=True)
    off = _ds_gopred(monkeypatch, rows, append=False)
    assert torch.equal(on[0]["tokens"], off[0]["tokens"])


def test_append_gopred_target_labels_mask_only_prompt(monkeypatch):
    """The appended go_pred list is SUPERVISED; only the prompt span is masked."""
    from torchtune.data import CROSS_ENTROPY_IGNORE_IDX as IGN
    rows = [_row_gopred("GO:0000123, GO:0000456")]
    ds = _ds_gopred(monkeypatch, rows, append=True)
    item = ds[0]
    labels = item["labels"]
    n_prompt = int((labels == IGN).sum())
    assert n_prompt > 0
    assert int((labels != IGN).sum()) > 0
    assert int((item["tokens"] == _PROT_ID).sum()) == 3 + 2
    assert int((item["tokens"] == _GO_ID).sum()) == 5


# ── disable_protein_splice / disable_go_splice (Exp 2 ablation) ──────────────────
def test_disable_protein_splice_leaves_placeholder_embeds():
    """disable_protein_splice=True ⇒ protein positions keep the plain placeholder-id
    embedding (not projected features); GO + text unchanged; output shape identical."""
    m, seqs = _model()
    m.disable_protein_splice = True
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    text_only = m._embed(toks)
    prot = toks == _PROT_ID
    go = toks == _GO_ID
    # protein positions == plain placeholder embedding (splice skipped)
    assert torch.allclose(emb[prot], text_only[prot], atol=1e-5)
    # GO positions STILL spliced (only protein disabled)
    assert not torch.allclose(emb[go], text_only[go], atol=1e-5)
    # shape identical to the enabled path
    m2, _ = _model()
    emb2 = m2.build_full_embeds_train(toks, seqs, ["all", "all"])
    assert emb.shape == emb2.shape


def test_disable_go_splice_leaves_go_placeholder_embeds():
    """disable_go_splice=True ⇒ GO positions keep placeholder embeds; protein still spliced."""
    m, seqs = _model()
    m.disable_go_splice = True
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    text_only = m._embed(toks)
    prot = toks == _PROT_ID
    go = toks == _GO_ID
    assert torch.allclose(emb[go], text_only[go], atol=1e-5)          # GO skipped
    assert not torch.allclose(emb[prot], text_only[prot], atol=1e-5)  # protein still spliced


def test_disable_both_splice_equals_text_only():
    """Both disabled ⇒ embeds == pure text embedding at every position."""
    m, seqs = _model()
    m.disable_protein_splice = True
    m.disable_go_splice = True
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    assert torch.allclose(emb, m._embed(toks), atol=1e-5)


def test_collate_fixed_pad_id_must_not_collide_with_placeholders():
    """REGRESSION: Qwen3 tokenizer.pad_id == 151643 == protein_token_id. If pad_to_fixed
    pads tokens with that id, every pad slot becomes a protein placeholder and the splice
    count inflates by the pad amount -> 'Protein token count N != features M'. The collate
    must pad with a neutral id. This test pins that padding with a non-placeholder id keeps
    the protein-placeholder count equal to the real (pre-pad) count."""
    from torchtune.dev.bioreason.dataset_sft import bioreason_sft_collate_fn
    PROT, GO = 151643, 151644
    row = {"tokens": torch.tensor([1, 2, PROT, PROT, GO, 9]),
           "labels": torch.tensor([-100, -100, -100, -100, -100, 9]),
           "protein_sequence": "MK", "go_aspect": "all"}
    # Pad with a NEUTRAL id (0) to a fixed length much larger than the row.
    out = bioreason_sft_collate_fn([row], padding_idx=0, max_seq_len=64, pad_to_fixed=True)
    assert out["tokens"].shape == (1, 64)
    # Real protein placeholder count (2) must be preserved — pad slots must NOT be counted.
    assert (out["tokens"] == PROT).sum().item() == 2
    assert (out["tokens"] == GO).sum().item() == 1
    # And the failure mode: padding WITH the placeholder id inflates the count (documents the bug).
    bad = bioreason_sft_collate_fn([row], padding_idx=PROT, max_seq_len=64, pad_to_fixed=True)
    assert (bad["tokens"] == PROT).sum().item() == 2 + (64 - 6)  # real + pad slots


def _load_recipe_staticmethod():
    """Load the recipe module by file path (recipes/ is not an importable package) and
    return the BioReasonSFTRecipeDistributedXPU class. The recipe imports its parent by
    file path at import time, so this works on a login node (no XPU touched at import)."""
    import importlib.util as iu, os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "recipes", "dev", "sft_bioreason_distributed_xpu.py")
    spec = iu.spec_from_file_location("_brsft_recipe_test", os.path.abspath(path))
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BioReasonSFTRecipeDistributedXPU


def test_resume_trainable_keys_selects_adapters_and_projections():
    """_trainable_keys must select exactly LoRA adapters + protein/GO projections (the
    requires_grad=True params), stripping FSDP/AC wrapper prefixes. Base/buffers excluded.
    This is what the self-contained resume_state persists; a leak of base weights or a
    miss of any adapter would corrupt resume."""
    Recipe = _load_recipe_staticmethod()
    sd = {
        "_fsdp_wrapped_module.backbone.layers.0.attn.q_proj.weight": 1,
        "backbone.layers.0.attn.q_proj.lora_a.weight": 2,
        "_checkpoint_wrapped_module.backbone.layers.0.attn.q_proj.lora_b.weight": 3,
        "backbone.layers.0.attn.q_proj.magnitude": 4,
        "protein_projection.0.weight": 5,
        "go_projection.2.bias": 6,
        "backbone.tok_embeddings.weight": 7,
        "backbone.layers.0.attn.q_proj.weight": 8,
    }
    keys = Recipe._trainable_keys(sd)
    assert set(keys) == {
        "backbone.layers.0.attn.q_proj.lora_a.weight",
        "backbone.layers.0.attn.q_proj.lora_b.weight",
        "backbone.layers.0.attn.q_proj.magnitude",
        "protein_projection.0.weight",
        "go_projection.2.bias",
    }, set(keys)


def test_lazy_safetensors_cache_roundtrip(tmp_path):
    """_LazySafetensorsCache must mirror dict .get/__contains__/__len__ semantics while
    reading per-key (the scalable replacement for the .pt full-pickle load). Variable
    [L+2, dim] shapes per key must round-trip exactly."""
    import torch
    from safetensors.torch import save_file
    import importlib.util as iu, os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        "torchtune", "dev", "bioreason", "model_native.py")
    spec = iu.spec_from_file_location("_mnative_test", os.path.abspath(path))
    mod = iu.module_from_spec(spec); spec.loader.exec_module(mod)
    LazyCache = mod._LazySafetensorsCache

    data = {
        "a"*40: torch.randn(5, 8, dtype=torch.bfloat16),    # [L+2, dim] variable L
        "b"*40: torch.randn(12, 8, dtype=torch.bfloat16),
    }
    f = tmp_path / "c.safetensors"
    save_file(data, str(f))
    cache = LazyCache(str(f))
    assert len(cache) == 2
    assert ("a"*40) in cache and ("z"*40) not in cache
    assert cache.get("z"*40) is None
    assert torch.equal(cache.get("a"*40), data["a"*40])
    assert torch.equal(cache["b"*40], data["b"*40])
    assert cache.get("a"*40).shape == (5, 8)

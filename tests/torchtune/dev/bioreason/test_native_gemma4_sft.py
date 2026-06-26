"""CPU tests for the native-Gemma4 BioReason SFT path (no XPU, no distributed).

Pins the load-bearing contracts of:
  - torchtune/models/gemma4/_component_builders.py (lora_gemma4)
  - torchtune/dev/bioreason/model_native.py (BioReasonNativeModel)
  - torchtune/dev/bioreason/dataset_sft.py (BioReasonSFTDataset)

Tests (lettered to the plan):
  A  model builds; _embed is backbone.tok_embeddings
  B  splice positions: protein/GO at placeholder ids; text positions unchanged
  C  label/grad flow: projections train through the splice
  D  placeholder-id contract: reserved ids absent from a normal-prompt corpus
  E  ★ embedding-scale equivalence: model(input_embeds=build(toks)) == backbone(tokens)
  F  count fail-fast on protein/feature mismatch
  G  lora_gemma4 builder: adapters trainable + bf16-able, base frozen, output tied,
     global (k_eq_v) v_proj is Identity, base keys == dense keys
  H  recipe wiring: dataset/model placeholder ids agree; recipe references the
     grad-enabled splice (build_full_embeds_train), not the no_grad one
"""

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchtune.models.gemma4._component_builders import gemma4, lora_gemma4
from torchtune.dev.bioreason.model_native import BioReasonNativeModel
from torchtune.modules.peft import LoRALinear, get_adapter_params

_REPO = Path(__file__).resolve().parents[4]
_H = 32
_V = 64
_ESM_DIM = 12
_PROT_ID = 60
_GO_ID = 61
_LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]
_GEMMA_KW = dict(
    vocab_size=_V,
    num_layers=6,
    num_heads=4,
    embed_dim=_H,
    intermediate_dim=64,
    max_seq_len=1024,
    local_head_dim=8,
    local_num_kv_heads=2,
    local_rope_base=10000.0,
    sliding_window_size=16,
    global_head_dim=16,
    global_num_kv_heads=1,
    global_rope_base=1000000.0,
    global_partial_rotary_factor=0.25,
    global_k_eq_v=True,
    layer_types=_LAYER_TYPES,
)


def _tiny_backbone(lora=False):
    if lora:
        return lora_gemma4(
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            lora_rank=4,
            lora_alpha=8.0,
            **_GEMMA_KW,
        ).to(torch.float32)
    return gemma4(**_GEMMA_KW).to(torch.float32)


def _tiny_model(lora=False):
    seqs = ["MKT", "MA"]
    cache = {
        hashlib.sha1(s.encode("ascii", "ignore")).hexdigest(): torch.randn(
            len(s) + 2, _ESM_DIM
        )
        for s in seqs
    }
    meta = {"embedding_dim": _ESM_DIM, "esm3_model_name": "esm3_sm_open_v1", "n_seqs": 2}
    m = BioReasonNativeModel(
        device=torch.device("cpu"),
        hidden_size=_H,
        protein_token_id=_PROT_ID,
        go_token_id=_GO_ID,
        dtype=torch.float32,
        backbone=_tiny_backbone(lora=lora),
        protein_hidden_override=_ESM_DIM,
        esm3_cache_inject=(cache, meta),
        go_embedding_inject=torch.randn(200, BioReasonNativeModel.GO_DIM),
        enable_lora=lora,
    )
    return m, seqs


def _mkrow(nprot, ngo=200):
    return [1, 2] + [_PROT_ID] * nprot + [_GO_ID] * ngo + [3, 4]


# ── A ─────────────────────────────────────────────────────────────────────────
def test_A_model_builds_and_reuses_tok_embeddings():
    m, _ = _tiny_model()
    assert m._embed is m.backbone.tok_embeddings


# ── B ─────────────────────────────────────────────────────────────────────────
def test_B_splice_positions():
    m, seqs = _tiny_model()
    r0 = _mkrow(5)
    r1 = _mkrow(4) + [0]
    toks = torch.tensor([r0, r1])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    text_only = m._embed(toks)
    prot = toks == _PROT_ID
    go = toks == _GO_ID
    text = ~(prot | go)
    assert torch.allclose(emb[text], text_only[text], atol=1e-5)
    assert not torch.allclose(emb[prot], text_only[prot], atol=1e-5)
    assert not torch.allclose(emb[go], text_only[go], atol=1e-5)


# ── C ─────────────────────────────────────────────────────────────────────────
def test_C_projection_grads_flow_through_splice():
    m, seqs = _tiny_model()
    toks = torch.tensor([_mkrow(5), _mkrow(4) + [0]])
    emb = m.build_full_embeds_train(toks, seqs, ["all", "all"])
    m(input_embeds=emb).sum().backward()
    assert m.protein_projection[0].weight.grad.abs().sum().item() > 0
    assert m.go_projection[0].weight.grad.abs().sum().item() > 0


# ── D ─────────────────────────────────────────────────────────────────────────
def test_D_placeholder_ids_absent_from_normal_text():
    """The reserved Gemma vocab ids must never be produced by normal prompt text.
    Uses the on-disk Gemma4 tokenizer if present; otherwise skips (CI without weights)."""
    from torchtune.dev.bioreason.dataset_sft import (
        DEFAULT_PROTEIN_TOKEN_ID,
        DEFAULT_GO_TOKEN_ID,
    )

    base = Path("/lus/flare/projects/ModCon/ngetty/models/gemma-4-31B")
    tj = base / "tokenizer.json"
    tcj = base / "tokenizer_config.json"
    if not (tj.exists() and tcj.exists()):
        pytest.skip("Gemma4 tokenizer files not present")
    from torchtune.models.gemma4 import gemma4_tokenizer

    tok = gemma4_tokenizer(str(tj), str(tcj))
    corpus = (
        "Given the protein above from organism Homo sapiens with the following InterPro "
        "annotations: PF00001. Reason about the function of the protein. Molecular "
        "Function, Cellular Component, Biological Process. GO:0005524 ATP binding."
    )
    ids = tok.encode(corpus, add_bos=False, add_eos=False)
    assert DEFAULT_PROTEIN_TOKEN_ID not in ids
    assert DEFAULT_GO_TOKEN_ID not in ids


# ── E (★) ─────────────────────────────────────────────────────────────────────
def test_E_embedding_scale_equivalence():
    """The whole reuse-tok_embeddings design hinges on this: feeding input_embeds built
    from tokens (no placeholders) must reproduce the bare-token forward exactly, i.e. the
    sqrt(embed_dim) Gemma scale is applied. Catches a silent magnitude mismatch."""
    m, _ = _tiny_model()
    toks = torch.randint(0, 59, (2, 12))  # ids below placeholder range
    emb = m.build_full_embeds_train(toks, protein_sequences=[], go_aspects=None)
    out_embed = m(input_embeds=emb)
    out_tok = m.backbone(tokens=toks)
    assert torch.allclose(out_embed, out_tok, atol=1e-5)


# ── F ─────────────────────────────────────────────────────────────────────────
def test_F_count_mismatch_fail_fast():
    m, _ = _tiny_model()
    # 99 protein placeholders but cache for 'MKT' yields only 5 features → mismatch.
    bad = torch.tensor([_mkrow(99)])
    with pytest.raises(ValueError):
        m.build_full_embeds_train(bad, ["MKT"], ["all"])


# ── G ─────────────────────────────────────────────────────────────────────────
def test_G_lora_builder_contract():
    lora = _tiny_backbone(lora=True)
    dense = _tiny_backbone(lora=False)
    # global (k_eq_v) layer has Identity v_proj; local has a LoRALinear v_proj.
    assert isinstance(lora.layers[5].attn.v_proj, nn.Identity)
    assert isinstance(lora.layers[0].attn.v_proj, LoRALinear)
    assert isinstance(lora.layers[0].attn.q_proj, LoRALinear)
    # adapters exist. NB: the BUILDER does not freeze the base (torchtune convention —
    # the recipe/model does). BioReasonNativeModel freezes it; see test_G_model_freezes_base.
    assert len(get_adapter_params(lora)) > 0
    # base (non-adapter) keys match the dense decoder exactly → checkpoint loads clean
    dk = set(dense.state_dict().keys())
    lk_base = {
        k
        for k in lora.state_dict().keys()
        if not any(s in k for s in ("lora_a", "lora_b", "magnitude"))
    }
    assert dk == lk_base
    # output is tied to tok_embeddings (not separately adaptable)
    from torchtune.modules import TiedLinear

    assert isinstance(lora.output, TiedLinear)


def test_G_lora_adapters_castable_to_bf16():
    lora = _tiny_backbone(lora=True)
    lora.to(torch.bfloat16)
    q = lora.layers[0].attn.q_proj
    assert q.lora_a.weight.dtype == torch.bfloat16


def test_G_model_freezes_base_keeps_projections_trainable():
    """BioReasonNativeModel must freeze the LoRA base but keep adapters AND the
    from-scratch projections trainable."""
    m, _ = _tiny_model(lora=True)
    # base attention weight frozen
    assert m.backbone.layers[0].attn.q_proj.weight.requires_grad is False
    # adapter trainable
    assert m.backbone.layers[0].attn.q_proj.lora_a.weight.requires_grad is True
    # projections trainable
    assert m.protein_projection[0].weight.requires_grad is True
    assert m.go_projection[0].weight.requires_grad is True


# ── H ─────────────────────────────────────────────────────────────────────────
def test_G_lora_merge_for_save():
    """merged_backbone_for_save collapses lora_a/lora_b into W_eff = W + (a/r)(B@A),
    strips the backbone. prefix, drops projections, and preserves non-LoRA weights."""
    H, r = 8, 4
    A = torch.randn(r, H)
    B = torch.randn(H, r)
    Wbase = torch.randn(H, H)
    full = {
        "backbone.layers.0.attn.q_proj.weight": Wbase.clone(),
        "backbone.layers.0.attn.q_proj.lora_a.weight": A.clone(),
        "backbone.layers.0.attn.q_proj.lora_b.weight": B.clone(),
        "backbone.layers.0.attn.k_proj.weight": torch.randn(H, H),
        "protein_projection.0.weight": torch.randn(H, 12),
    }
    m = BioReasonNativeModel.__new__(BioReasonNativeModel)
    out = BioReasonNativeModel.merged_backbone_for_save(m, full, lora_rank=r, lora_alpha=2 * r)
    qkey = "layers.0.attn.q_proj.weight"
    assert torch.allclose(out[qkey].float(), Wbase + 2.0 * (B @ A), atol=1e-4)
    assert not any("lora" in k for k in out)
    assert "layers.0.attn.k_proj.weight" in out
    assert not any("projection" in k for k in out)
    assert all(not k.startswith("backbone.") for k in out)


def test_H_recipe_uses_grad_enabled_splice():
    recipe = (_REPO / "recipes" / "dev" / "sft_bioreason_distributed_xpu.py").read_text()
    # must use the grad-enabled splice, NOT the no_grad build_prompt_embeds
    assert "build_full_embeds_train" in recipe
    assert "build_prompt_embeds" not in recipe


def test_H_config_dataset_model_placeholder_ids_agree():
    import yaml

    for name in (
        "production/sft_bioreason_gemma4_31B_xpu.yaml",
        "smoke/sft_bioreason_gemma4_31B_smoke_xpu.yaml",
    ):
        cfg = yaml.safe_load(
            (_REPO / "recipes" / "configs" / "dev" / name).read_text()
        )
        assert cfg["model"]["protein_token_id"] == cfg["dataset"]["protein_token_id"]
        assert cfg["model"]["go_token_id"] == cfg["dataset"]["go_token_id"]


def test_H_dataset_prompt_matches_rl_dataset():
    """The SFT go_pred prompt must be byte-identical to the RL dataset's (train==eval==SFT
    distribution). Both build from the same _SYS_WITH_CONTEXT + user template."""
    from torchtune.dev.bioreason import dataset_sft as sft
    from torchtune.dev.bioreason import dataset as rl

    assert sft._SYS_WITH_CONTEXT == rl.BioReasonRLDataset._SYS_WITH_CONTEXT

    ex = {
        "organism": "Homo sapiens",
        "interpro_formatted": "PF00001",
        "ppi_formatted": "BRCA1",
        "go_pred": "GO:0005524",
        "go_mf": ["GO:0005524"],
        "go_cc": [],
        "go_bp": [],
    }
    # SFT builds via an instance method; reuse the RL method on a bare object by binding.
    sft_text = sft.BioReasonSFTDataset._build_prompt_text(
        sft.BioReasonSFTDataset.__new__(sft.BioReasonSFTDataset), ex
    )
    rl_text = rl.BioReasonRLDataset._build_go_pred_prompt_text(
        rl.BioReasonRLDataset.__new__(rl.BioReasonRLDataset), ex
    )
    assert sft_text == rl_text

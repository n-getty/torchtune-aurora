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


def _model(lora=False):
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
        enable_lora=lora,
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

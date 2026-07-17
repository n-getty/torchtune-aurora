# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU correctness proof for token PACKING: a packed forward with a block-diagonal document
# mask must produce the SAME per-document logits as running each document SEPARATELY. This is
# the test that catches a broken mask (doc A leaking into doc B's attention). It exercises the
# masking LOGIC on the CPU math path with an explicit 2D bool block-diagonal mask + per-doc
# input_pos — independent of the XPU flex kernel (validated separately on HW). If this passes,
# the packing semantics (block-diag isolation + per-doc RoPE) are correct.

import hashlib

import torch

from torchtune.dev.bioreason.model_native import BioReasonNativeModel
from torchtune.models.qwen3._component_builders import qwen3

_H = 16
_V = 200
_ESM_DIM = 12
_PROT_ID = 190
_GO_ID = 191
_QWEN_KW = dict(
    vocab_size=_V, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8,
    embed_dim=_H, intermediate_dim=64, max_seq_len=1024, tie_word_embeddings=False,
)


def _model():
    seqs = ["MKT", "MA"]
    cache = {
        hashlib.sha1(s.encode("ascii", "ignore")).hexdigest(): torch.randn(len(s) + 2, _ESM_DIM)
        for s in seqs
    }
    meta = {"embedding_dim": _ESM_DIM, "esm3_model_name": "esm3_sm_open_v1", "n_seqs": 2}
    m = BioReasonNativeModel(
        device=torch.device("cpu"), hidden_size=_H, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, dtype=torch.float32, backbone=qwen3(**_QWEN_KW).to(torch.float32),
        protein_hidden_override=_ESM_DIM, esm3_cache_inject=(cache, meta),
        go_embedding_inject=torch.randn(200, BioReasonNativeModel.GO_DIM),
        enable_lora=False, freeze_backbone=True,
    )
    m.eval()
    return m, seqs


def _block_diag_bool_mask(seq_lens: list[int], S: int) -> torch.Tensor:
    """[1, S, S] bool: position i attends j iff same-doc AND causal (i>=j)."""
    doc = torch.zeros(S, dtype=torch.long)
    off = 0
    for d, L in enumerate(seq_lens):
        doc[off:off + L] = d
        off += L
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    same = doc[:, None] == doc[None, :]
    return (causal & same).unsqueeze(0)


def test_packed_equals_separate_docs_text_only():
    """Two text-only docs packed into one row (block-diag mask + per-doc input_pos) must give
    the same logits per doc as forwarding each doc alone."""
    m, _ = _model()
    torch.manual_seed(0)
    # Two docs, no protein/GO placeholders (isolate the mask+position logic).
    doc0 = [1, 5, 9, 12]          # len 4
    doc1 = [2, 7, 3]              # len 3
    S = len(doc0) + len(doc1)     # 7 (no pad for a clean equality)

    # Separate forwards (each doc starts at position 0, plain causal).
    with torch.no_grad():
        out0 = m(torch.tensor([doc0]), protein_sequences=[], go_aspects=None)
        out1 = m(torch.tensor([doc1]), protein_sequences=[], go_aspects=None)

    # Packed forward: one row, block-diagonal mask, per-doc input_pos reset.
    packed = torch.tensor([doc0 + doc1])
    mask = _block_diag_bool_mask([len(doc0), len(doc1)], S)
    input_pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
    with torch.no_grad():
        outp = m(packed, protein_sequences=[], go_aspects=None, mask=mask, input_pos=input_pos)

    # doc0 logits (positions 0..3) must match the standalone doc0 forward; likewise doc1.
    assert torch.allclose(outp[:, :4], out0, atol=1e-4), "doc0 logits diverged under packing"
    assert torch.allclose(outp[:, 4:7], out1, atol=1e-4), "doc1 logits diverged under packing"


def test_block_diag_mask_blocks_cross_doc_attention():
    """Sanity: with a block-diag mask, changing doc1's tokens must NOT change doc0's logits
    (proves no cross-doc leakage). A plain causal mask WOULD change them."""
    m, _ = _model()
    doc0 = [1, 5, 9, 12]
    doc1a = [2, 7, 3]
    doc1b = [99, 40, 150]     # different doc1 content
    S = 7
    mask = _block_diag_bool_mask([4, 3], S)
    ipos = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
    with torch.no_grad():
        oa = m(torch.tensor([doc0 + doc1a]), protein_sequences=[], go_aspects=None, mask=mask, input_pos=ipos)
        ob = m(torch.tensor([doc0 + doc1b]), protein_sequences=[], go_aspects=None, mask=mask, input_pos=ipos)
    # doc0 region identical regardless of doc1 content (block-diag isolation).
    assert torch.allclose(oa[:, :4], ob[:, :4], atol=1e-5), "cross-doc leak: doc0 changed with doc1"
    # doc1 region DID change (sanity that the test actually varied something).
    assert not torch.allclose(oa[:, 4:7], ob[:, 4:7], atol=1e-3)


def test_packed_go_splice_two_docs_one_row():
    """Two docs WITH GO placeholders packed in one row. Regression for the HW bug (job
    8673929): the GO splice path ignored batch_idx_map and assumed one GO block per row, so a
    2-doc pack (2*num_go_tokens placeholders) crashed 'GO token count 400 != GO features 200'.
    The GO features must be assembled per-doc and grouped by batch_idx_map like protein."""
    m, _ = _model()
    NGO = 200  # the go cache is [200, GO_DIM]; each doc consumes up to 200 GO tokens
    # Two docs, each with NGO GO placeholders (no protein, to isolate GO).
    doc0 = [1] + [_GO_ID] * NGO + [8]
    doc1 = [2] + [_GO_ID] * NGO + [9]
    S = len(doc0) + len(doc1)
    packed = torch.tensor([doc0 + doc1])
    mask = _block_diag_bool_mask([len(doc0), len(doc1)], S)
    ipos = torch.tensor([list(range(len(doc0))) + list(range(len(doc1)))])
    with torch.no_grad():
        # batch_idx_map=[0,0]: both docs' GO features -> row 0, in order. Must NOT crash.
        outp = m(packed, protein_sequences=[], go_aspects=["all", "all"],
                 batch_idx_map=[0, 0], mask=mask, input_pos=ipos)
        out0 = m(torch.tensor([doc0]), protein_sequences=[], go_aspects=["all"])
        out1 = m(torch.tensor([doc1]), protein_sequences=[], go_aspects=["all"])
    assert torch.allclose(outp[:, :len(doc0)], out0, atol=1e-4), "packed doc0 GO splice diverged"
    assert torch.allclose(outp[:, len(doc0):], out1, atol=1e-4), "packed doc1 GO splice diverged"


def test_packed_multimodal_splice_two_docs_one_row():
    """Two docs WITH protein placeholders packed in one row: the splice must fill BOTH docs'
    placeholders in order (batch_idx_map=[0,0]) and the block-diag mask isolates them."""
    m, seqs = _model()  # seqs = ["MKT"(len3->5 placeholders), "MA"(len2->4 placeholders)]
    # doc0 uses seqs[0] (5 protein tokens), doc1 uses seqs[1] (4 protein tokens).
    doc0 = [1] + [_PROT_ID] * 5 + [8]        # len 7
    doc1 = [2] + [_PROT_ID] * 4 + [9]        # len 6
    S = len(doc0) + len(doc1)                 # 13
    packed = torch.tensor([doc0 + doc1])
    mask = _block_diag_bool_mask([len(doc0), len(doc1)], S)
    ipos = torch.tensor([list(range(len(doc0))) + list(range(len(doc1)))])
    with torch.no_grad():
        # batch_idx_map=[0,0]: both docs' protein features go to row 0, in order.
        outp = m(packed, protein_sequences=seqs, go_aspects=None,
                 batch_idx_map=[0, 0], mask=mask, input_pos=ipos)
        # standalone doc0 (seqs[0] only, its own row)
        out0 = m(torch.tensor([doc0]), protein_sequences=[seqs[0]], go_aspects=None)
        out1 = m(torch.tensor([doc1]), protein_sequences=[seqs[1]], go_aspects=None)
    assert torch.allclose(outp[:, :len(doc0)], out0, atol=1e-4), "packed doc0 splice diverged"
    assert torch.allclose(outp[:, len(doc0):], out1, atol=1e-4), "packed doc1 splice diverged"

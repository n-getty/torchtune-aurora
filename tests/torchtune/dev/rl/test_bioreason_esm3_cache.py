# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression tests for the BioReason ESM3 pre-encode cache.

ESM3 is frozen and the dataset's sequences are fixed, so the ~1.4B fp32 encoder
is pre-run offline into a keyed cache and NOT loaded during training. These tests
pin the load-bearing, silent-failure-prone pieces WITHOUT building a real
BioReasonModel (which needs ESM3 + a checkpoint + XPU):

  1. Cache lookup in build_prompt_embeds places the cached per-residue features
     into exactly the protein_pad token slots — i.e. identical placement to the
     live-encoder path, just sourced from the cache + live projection.
  2. The cache key + truncation contract matches the precompute script, and a
     missing sequence raises KeyError (never silent zero-fill).
  3. A stale cache (sidecar model/dim mismatch, or missing files) is rejected at
     model init rather than producing wrong embeddings.
  4. _freeze_encoders no-ops cleanly when the encoder was not built.
"""
import json

import pytest

torch = pytest.importorskip("torch")

from torchtune.dev.bioreason.model import BioReasonModel


# ── helpers ──────────────────────────────────────────────────────────────────

PROTEIN_TOK = 9001
GO_TOK = 9002
PAD_TOK = 0
H = 8          # tiny hidden
ESM_DIM = 6    # tiny ESM3 feature dim


def _cache_backed_stub(cache, meta):
    """Bind the unbound BioReasonModel.build_prompt_embeds onto a minimal stub.

    Provides the exact attributes build_prompt_embeds touches: _embed,
    protein_projection (identity-ish), protein_token_id, go_token_id,
    _esm3_cache(+meta), device, dtype, and _get_go_embeds (returns None — no GO
    here, GO is tested by the existing live path).
    """
    class _Stub:
        device = torch.device("cpu")
        dtype = torch.float32
        protein_token_id = PROTEIN_TOK
        go_token_id = GO_TOK
        _esm3_cache = cache
        _esm3_cache_meta = meta
        # methods under test / used
        esm3_cache_key = staticmethod(BioReasonModel.esm3_cache_key)
        build_prompt_embeds = BioReasonModel.build_prompt_embeds

        def _embed(self, ids):
            # deterministic per-id text embedding [.., H]
            return torch.zeros(*ids.shape, H) + ids.unsqueeze(-1).float()

        def _get_go_embeds(self, aspects, B):
            return None  # no GO in these tests

        # protein_projection: ESM_DIM -> H, fixed weights so output is checkable
        protein_projection = torch.nn.Linear(ESM_DIM, H, bias=False)

    s = _Stub()
    torch.nn.init.eye_(s.protein_projection.weight[:min(H, ESM_DIM), :min(H, ESM_DIM)]) \
        if H == ESM_DIM else torch.nn.init.constant_(s.protein_projection.weight, 0.5)
    return s


# ── 1. cache lookup → correct token-slot placement ───────────────────────────

def test_cache_lookup_fills_protein_pad_slots():
    seq = "MKTAYI"            # len 6 -> dataset inserts len+2 = 8 protein_pad toks
    n_prot = len(seq) + 2
    feat = torch.randn(n_prot, ESM_DIM)
    key = BioReasonModel.esm3_cache_key(seq)
    cache = {key: feat}
    meta = {"embedding_dim": ESM_DIM, "max_protein_len": 128,
            "esm3_model_name": "esm3_sm_open_v1", "n_seqs": 1}
    stub = _cache_backed_stub(cache, meta)

    # input_ids: [text, n_prot * protein_pad, text]  (B=1)
    ids = torch.tensor([[5, 6]
                        + [PROTEIN_TOK] * n_prot
                        + [7]], dtype=torch.long)
    out = BioReasonModel.build_prompt_embeds(stub, ids, [seq])

    assert out.shape == (1, ids.shape[1], H)
    # The protein_pad slots must equal projection(feat); non-protein slots untouched.
    expected_prot = stub.protein_projection(feat.to(torch.float32))
    mask = (ids[0] == PROTEIN_TOK)
    torch.testing.assert_close(out[0][mask], expected_prot, rtol=1e-5, atol=1e-5)
    # a text slot stays at the _embed value (no protein overwrite)
    assert out[0, 0, 0].item() == pytest.approx(5.0)


def test_cache_miss_raises_keyerror_not_silent():
    meta = {"embedding_dim": ESM_DIM, "max_protein_len": 128,
            "esm3_model_name": "esm3_sm_open_v1", "n_seqs": 0}
    stub = _cache_backed_stub({}, meta)  # empty cache
    ids = torch.tensor([[PROTEIN_TOK, PROTEIN_TOK]], dtype=torch.long)
    with pytest.raises(KeyError):
        BioReasonModel.build_prompt_embeds(stub, ids, ["MK"])


# ── 2. key/truncation contract matches the precompute script ─────────────────

def test_cache_key_matches_truncated_sequence():
    # The dataset truncates sequence[:max_protein_len]; the cache is keyed on the
    # truncated form. Two sequences sharing a >max_protein_len prefix collide
    # (correct — the model only ever sees the truncated form).
    long_a = "M" * 150 + "AAAA"
    long_b = "M" * 150 + "BBBB"
    mpl = 128
    assert BioReasonModel.esm3_cache_key(long_a[:mpl]) == \
           BioReasonModel.esm3_cache_key(long_b[:mpl])
    # stable + deterministic
    assert BioReasonModel.esm3_cache_key("MKT") == BioReasonModel.esm3_cache_key("MKT")


# ── 3. stale / missing cache rejected at load ────────────────────────────────

def _write_cache(tmp_path, model_name="esm3_sm_open_v1", dim=ESM_DIM):
    cache_path = tmp_path / "esm3_cache.pt"
    torch.save({BioReasonModel.esm3_cache_key("MK"): torch.randn(4, dim)}, cache_path)
    with open(str(cache_path) + ".json", "w") as f:
        json.dump({"embedding_dim": dim, "max_protein_len": 128,
                   "esm3_model_name": model_name, "n_seqs": 1}, f)
    return str(cache_path)


def test_load_cache_rejects_model_mismatch(tmp_path):
    cache_path = _write_cache(tmp_path, model_name="esmc_600m")
    inst = BioReasonModel.__new__(BioReasonModel)
    with pytest.raises(ValueError, match="model mismatch"):
        BioReasonModel._load_esm3_cache(inst, cache_path, "esm3_sm_open_v1")


def test_load_cache_missing_sidecar(tmp_path):
    cache_path = tmp_path / "esm3_cache.pt"
    torch.save({"x": torch.zeros(1)}, cache_path)  # no .json sidecar
    inst = BioReasonModel.__new__(BioReasonModel)
    with pytest.raises(FileNotFoundError, match="sidecar"):
        BioReasonModel._load_esm3_cache(inst, str(cache_path), "esm3_sm_open_v1")


def test_load_cache_ok(tmp_path):
    cache_path = _write_cache(tmp_path)
    inst = BioReasonModel.__new__(BioReasonModel)
    cache, meta = BioReasonModel._load_esm3_cache(inst, cache_path, "esm3_sm_open_v1")
    assert meta["embedding_dim"] == ESM_DIM and len(cache) == 1


# ── 4. _freeze_encoders no-op when encoder not built ─────────────────────────

def test_freeze_encoders_handles_none_protein_encoder():
    inst = BioReasonModel.__new__(BioReasonModel)
    inst.protein_encoder = None
    inst.go_encoder = None
    inst._has_lora = True
    # Must not raise (the cached path leaves protein_encoder=None).
    BioReasonModel._freeze_encoders(inst)


# ── 5. _load_embed_layer casts a fp32 checkpoint embed to self.dtype ──────────
# Regression: bioreason-pro-rl stores model.embed_tokens.weight as FP32. Before the
# fix, _load_embed_layer assigned the raw fp32 tensor to emb.weight.data, so `embeds`
# was fp32 while protein/GO features were bf16 -> `embeds[mask] = flat` crashed with
# "Index put requires the source and destination dtypes match" in build_prompt_embeds.
# (SFT stored bf16, so it silently worked and masked the bug.)

def test_load_embed_layer_casts_fp32_ckpt_to_self_dtype(tmp_path):
    from safetensors.torch import save_file

    class _Cfg:
        vocab_size = 32
        hidden_size = 8

    # Write a checkpoint whose embed_tokens is FP32 (like bioreason-pro-rl).
    save_file(
        {"model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.float32)},
        str(tmp_path / "model.safetensors"),
    )
    inst = BioReasonModel.__new__(BioReasonModel)
    inst.dtype = torch.bfloat16
    inst.device = torch.device("cpu")
    emb = BioReasonModel._load_embed_layer(inst, str(tmp_path), _Cfg())
    # The loaded weight must be cast to self.dtype, not left as the ckpt's fp32.
    assert emb.weight.dtype == torch.bfloat16
    assert emb.num_embeddings == 32 and emb.embedding_dim == 8

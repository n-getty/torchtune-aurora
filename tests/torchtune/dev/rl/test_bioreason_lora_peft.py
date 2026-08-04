# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression tests for the BioReason PEFT-LoRA surface.

These pin the two load-bearing, drift-prone pieces of the LoRA integration
without instantiating the full BioReasonModel (which needs ESM3 + a checkpoint
+ XPU):

  1. ``BioReasonModel._peft_name_to_hf`` — PEFT renames a wrapped linear's weight
     to ``base_model.model.model.layers.N....q_proj.base_layer.weight`` and adds
     ``lora_A``/``lora_B`` adapter params. vLLM's ``load_weights`` expects the
     original HF name ``model.layers.N....q_proj.weight``. A drift here silently
     ships scrambled / wrong-named weights to the rollout engine.

  2. The merge math: after ``merge_adapter()`` the wrapped linear's effective
     weight equals ``base + (alpha/r) * (B @ A)``, and ``unmerge_adapter()``
     restores the pristine base. This is exactly what the server-mode weight
     sync relies on (it brackets the param gather in merge/unmerge so vLLM
     receives the merged W_eff).
"""
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
nn = torch.nn

from torchtune.dev.bioreason.model import BioReasonModel


# ── 1. PEFT → HF name translation ────────────────────────────────────────────

def test_peft_name_strips_prefix_and_base_layer():
    f = BioReasonModel._peft_name_to_hf
    cases = {
        # wrapped target linear (base weight carries W_eff after merge_adapter)
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight":
            "model.layers.0.self_attn.q_proj.weight",
        "base_model.model.model.layers.27.mlp.gate_proj.base_layer.weight":
            "model.layers.27.mlp.gate_proj.weight",
        "base_model.model.model.layers.5.self_attn.o_proj.base_layer.weight":
            "model.layers.5.self_attn.o_proj.weight",
        # non-target param (e.g. layernorm) — only the wrapper prefix is stripped
        "base_model.model.model.layers.0.input_layernorm.weight":
            "model.layers.0.input_layernorm.weight",
        "base_model.model.model.norm.weight": "model.norm.weight",
        "base_model.model.lm_head.weight": "lm_head.weight",
    }
    for peft_name, expected in cases.items():
        assert f(peft_name) == expected, peft_name


def test_peft_name_skips_adapter_params():
    f = BioReasonModel._peft_name_to_hf
    adapter_names = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "base_model.model.model.layers.3.mlp.up_proj.lora_A.default.weight",
        "base_model.model.model.layers.3.mlp.up_proj.lora_B.default.weight",
    ]
    for n in adapter_names:
        assert f(n) is None, f"adapter param must not be shipped to vLLM: {n}"


# ── 2. merge / unmerge math equivalence (tiny HF model) ──────────────────────

def _tiny_lora_model(rank=4, alpha=8):
    """A 2-layer Llama-shaped HF model wrapped with PEFT LoRA on all targets.

    Kept tiny (hidden=32) so this runs on a login node in well under the pytest
    timeout. Llama naming (q/k/v/o_proj, gate/up/down_proj) matches the BioReason
    target_modules list.
    """
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model

    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    base = transformers.LlamaForCausalLM(cfg).to(torch.float32)
    lcfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lcfg), rank, alpha


def _wrapped_linears(peft_model):
    """Yield PEFT LoRALinear modules (those exposing base_layer + lora_A/lora_B)."""
    for _name, mod in peft_model.named_modules():
        if hasattr(mod, "base_layer") and hasattr(mod, "lora_A") and "default" in mod.lora_A:
            yield mod


def test_merge_adapter_equals_base_plus_scaled_ba():
    model, rank, alpha = _tiny_lora_model()
    scaling = alpha / rank

    # PEFT inits lora_B to zeros -> fill with noise so the delta is nonzero.
    for mod in _wrapped_linears(model):
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=0.1)

    expected = {}
    pristine = {}
    for name, mod in [(n, m) for n, m in model.named_modules()
                      if hasattr(m, "base_layer") and hasattr(m, "lora_A")
                      and "default" in getattr(m, "lora_A", {})]:
        W = mod.base_layer.weight.detach().clone()
        A = mod.lora_A["default"].weight.detach()          # [r, in]
        B = mod.lora_B["default"].weight.detach()          # [out, r]
        pristine[name] = W
        expected[name] = W + scaling * (B @ A)

    model.merge_adapter()
    for name, mod in model.named_modules():
        if name in expected:
            torch.testing.assert_close(
                mod.base_layer.weight.detach(), expected[name],
                rtol=1e-5, atol=1e-5,
                msg=f"merged weight != base + (alpha/r)*B@A at {name}",
            )

    model.unmerge_adapter()
    for name, mod in model.named_modules():
        if name in pristine:
            torch.testing.assert_close(
                mod.base_layer.weight.detach(), pristine[name],
                rtol=1e-5, atol=1e-5,
                msg=f"unmerge did not restore pristine base at {name}",
            )


def _Stub(model):
    """Bind the real (unbound) BioReasonModel methods onto a tiny stand-in whose
    `.backbone` is a PEFT model (real BioReasonModel needs ESM3 + ckpt + XPU)."""
    class _S:
        _has_lora = True
        backbone = None
        _peft_name_to_hf = staticmethod(BioReasonModel._peft_name_to_hf)
        lora_delta_map = BioReasonModel.lora_delta_map
        lora_delta_iter = BioReasonModel.lora_delta_iter  # lora_delta_map delegates to this
        vllm_param_iter = BioReasonModel.vllm_param_iter
    s = _S()
    s.backbone = model
    return s


def test_lora_delta_iter_matches_map():
    """The streaming lora_delta_iter (used by the colocate merge to bound per-step
    transient → avoid the banned:1 staircase) must yield exactly the same
    {name: delta} as the eager lora_delta_map."""
    model, _r, _a = _tiny_lora_model()
    for mod in _wrapped_linears(model):
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=0.1)
    stub = _Stub(model)
    eager = stub.lora_delta_map()
    streamed = dict(stub.lora_delta_iter())
    assert set(eager) == set(streamed) and len(streamed) > 0
    for k in eager:
        torch.testing.assert_close(streamed[k], eager[k], rtol=0, atol=0)


def test_weff_via_delta_map_matches_base_plus_scaled_ba():
    """The PRODUCTION server-mode gather forms W_eff = base + lora_delta_map()
    WITHOUT mutating the frozen base. Verify (a) it equals base + (alpha/r)*B@A,
    (b) it yields clean HF names with no adapter leakage, and (c) the base is
    NOT mutated (the bf16 merge/unmerge round-trip drift bug this path avoids).
    """
    model, rank, alpha = _tiny_lora_model()
    scaling = alpha / rank
    for mod in _wrapped_linears(model):
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=0.1)

    stub = _Stub(model)

    # Expected merged weight per clean HF name, and a pristine base snapshot.
    expected, base0 = {}, {}
    for name, mod in model.named_modules():
        if hasattr(mod, "base_layer") and hasattr(mod, "lora_A") \
                and "default" in getattr(mod, "lora_A", {}):
            W = mod.base_layer.weight.detach().clone()
            A = mod.lora_A["default"].weight.detach().clone()
            B = mod.lora_B["default"].weight.detach().clone()
            hf = BioReasonModel._peft_name_to_hf(f"{name}.base_layer.weight")
            expected[hf] = W + scaling * (B @ A)
            base0[hf] = W

    # Mirror the production gather: base (vllm_param_iter) + delta (lora_delta_map).
    delta_map = stub.lora_delta_map()
    weff = {}
    for hf_name, param in stub.vllm_param_iter():
        w = param.detach()
        d = delta_map.get(hf_name)
        if d is not None:
            w = (w.float() + d.float()).to(param.dtype)
        weff[hf_name] = w.clone()

    assert not any(".lora_" in n for n in weff), "adapter param leaked to vLLM"
    assert not any(".lora_" in n for n in delta_map), "delta_map leaked adapter name"
    for hf, exp in expected.items():
        assert hf in weff, f"missing merged target {hf}"
        torch.testing.assert_close(
            weff[hf], exp, rtol=1e-4, atol=1e-4,
            msg=f"W_eff via delta_map != base + (alpha/r)*B@A at {hf}",
        )

    # Drift guard: forming W_eff must NOT mutate the frozen base (the whole reason
    # we use get_delta_weight instead of in-place merge_adapter/unmerge_adapter).
    for name, mod in model.named_modules():
        hf = BioReasonModel._peft_name_to_hf(f"{name}.base_layer.weight") \
            if hasattr(mod, "base_layer") else None
        if hf in base0:
            torch.testing.assert_close(
                mod.base_layer.weight.detach(), base0[hf], rtol=0, atol=0,
                msg=f"frozen base was mutated at {hf} (drift bug)",
            )


def test_get_peft_model_freezes_base_only_adapters_trainable():
    """Backbone-freeze invariant: only lora_A/lora_B require grad after wrapping."""
    model, _r, _a = _tiny_lora_model()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "expected at least the LoRA adapters to be trainable"
    for n in trainable:
        assert ".lora_A." in n or ".lora_B." in n, (
            f"non-adapter param is trainable after get_peft_model: {n}"
        )
    # Spot-check a base weight is frozen.
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any("base_layer.weight" in n for n in frozen)


def test_bf16_base_keeps_adapters_bf16_for_fsdp1_uniform_dtype():
    """FSDP1's flat-param requires UNIFORM dtype across a wrapped module. PEFT
    defaults autocast_adapter_dtype=True (fp32 adapters), which on a bf16 base
    crashes FSDP1 with 'Must flatten tensors with uniform dtype but got
    torch.bfloat16 and torch.float32' (hit on the 2026-06-18 first HW smoke).
    The recipe builds with autocast_adapter_dtype=False — this pins that adapters
    stay bf16 so base + adapters + projectors are a single dtype.
    """
    transformers = pytest.importorskip("transformers")
    from peft import LoraConfig, get_peft_model

    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    base = transformers.LlamaForCausalLM(cfg).to(torch.bfloat16)
    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        init_lora_weights="gaussian", bias="none", task_type="CAUSAL_LM",
    )
    # Mirror the recipe: autocast_adapter_dtype=False (model.py).
    pm = get_peft_model(base, lcfg, autocast_adapter_dtype=False)
    dtypes = {p.dtype for _n, p in pm.named_parameters() if p.requires_grad}
    assert dtypes == {torch.bfloat16}, (
        f"trainable adapters must be uniform bf16 for FSDP1, got {dtypes}"
    )


# ── 3. _embed freeze (regression guard for the H200-sibling-project bug class) ─

def test_freeze_embed_copy_sets_requires_grad_false():
    """self._embed is a standalone completion-token-lookup copy, loaded
    independently from checkpoint safetensors (NOT a reference into
    self.backbone), so get_peft_model's freeze has no visibility into it.

    A sibling HF/PEFT BioReason stack left this exact tensor unfrozen under
    LoRA (only the projections were explicitly frozen; the separate embedding
    copy defaulted to requires_grad=True and trained silently every step,
    ~778M extra params at 32B scale — 5.7x the intended LoRA adapter size).
    _freeze_embed_copy is called unconditionally in __init__ regardless of
    enable_lora; this pins that it actually does the freeze.
    """
    stub = types.SimpleNamespace(_embed=nn.Embedding(5, 4))
    BioReasonModel._freeze_embed_copy(stub)
    assert stub._embed.weight.requires_grad is False


def test_freeze_embed_copy_freezes_even_without_lora():
    """The freeze must not be conditioned on _has_lora — full-FT (enable_lora=
    False) also treats _embed as a fixed convenience copy, never a trained
    parameter, matching model_native.py's frozen-by-construction tok_embeddings
    alias semantics."""
    stub = types.SimpleNamespace(_has_lora=False, _embed=nn.Embedding(5, 4))
    BioReasonModel._freeze_embed_copy(stub)
    assert stub._embed.weight.requires_grad is False

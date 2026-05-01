# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe unit tests for torchtune → PEFT LoRA name translation.

Pins the key invariants of ``torchtune_to_peft_state_dict`` without
instantiating any model or requiring XPU / distributed init.

What we guard:
  - All 7 attention + MLP module types translate without error.
  - torchtune ``output_proj`` maps to HF ``o_proj`` (not ``output_proj``).
  - torchtune ``mlp.w1/w2/w3`` map to ``gate_proj/down_proj/up_proj``.
  - ``lora_a.weight`` → ``lora_A.weight`` (PEFT casing).
  - PEFT key prefix is ``base_model.model.model.layers.{i}.{hf_module}.``.
  - ``adapter_config.json`` carries correct ``r``, ``lora_alpha``,
    ``target_modules``, ``peft_type``, ``task_type``.
  - FSDP/AC prefixes are stripped before translation.
  - Keys that are not LoRA adapter keys raise ``ValueError``.
"""
import torch
import pytest

from torchtune.dev.rl.lora_helpers import (
    torchtune_to_peft_state_dict,
    _translate_lora_key,
    _strip_fsdp_prefixes,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_adapter_sd(modules, num_layers=2, rank=4):
    """Build a synthetic torchtune-style adapter state dict."""
    sd = {}
    for i in range(num_layers):
        for module in modules:
            for ab in ("lora_a", "lora_b"):
                key = f"layers.{i}.{module}.{ab}.weight"
                sd[key] = torch.zeros(1)  # shape doesn't matter for name tests
    return sd


_ATTN_MODULES = ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.output_proj"]
_MLP_MODULES  = ["mlp.w1", "mlp.w2", "mlp.w3"]
_ALL_MODULES  = _ATTN_MODULES + _MLP_MODULES


# ── strip_fsdp_prefixes ──────────────────────────────────────────────────────

def test_strip_fsdp_wrapped_module_prefix():
    key = "_fsdp_wrapped_module.layers.0.attn.q_proj.lora_a.weight"
    assert _strip_fsdp_prefixes(key) == "layers.0.attn.q_proj.lora_a.weight"


def test_strip_checkpoint_wrapped_module_prefix():
    key = "_checkpoint_wrapped_module.layers.3.mlp.w1.lora_b.weight"
    assert _strip_fsdp_prefixes(key) == "layers.3.mlp.w1.lora_b.weight"


def test_strip_both_prefixes():
    key = "_fsdp_wrapped_module._checkpoint_wrapped_module.layers.1.attn.v_proj.lora_a.weight"
    assert _strip_fsdp_prefixes(key) == "layers.1.attn.v_proj.lora_a.weight"


# ── _translate_lora_key ───────────────────────────────────────────────────────

@pytest.mark.parametrize("tune_module,expected_hf_module", [
    ("attn.q_proj",     "self_attn.q_proj"),
    ("attn.k_proj",     "self_attn.k_proj"),
    ("attn.v_proj",     "self_attn.v_proj"),
    ("attn.output_proj","self_attn.o_proj"),   # output_proj → o_proj
    ("mlp.w1",          "mlp.gate_proj"),       # w1 = gate_proj
    ("mlp.w2",          "mlp.down_proj"),       # w2 = down_proj
    ("mlp.w3",          "mlp.up_proj"),         # w3 = up_proj
])
def test_translate_lora_key_module_mapping(tune_module, expected_hf_module):
    key = f"layers.7.{tune_module}.lora_a.weight"
    result = _translate_lora_key(key)
    assert result is not None, f"_translate_lora_key returned None for {key!r}"
    expected = f"base_model.model.model.layers.7.{expected_hf_module}.lora_A.weight"
    assert result == expected, f"Expected {expected!r}, got {result!r}"


def test_translate_lora_key_b_suffix():
    key = "layers.0.attn.q_proj.lora_b.weight"
    result = _translate_lora_key(key)
    assert result is not None
    assert result.endswith(".lora_B.weight"), f"Expected lora_B suffix, got: {result!r}"


def test_translate_lora_key_large_layer_index():
    key = "layers.35.attn.output_proj.lora_a.weight"
    result = _translate_lora_key(key)
    assert result == "base_model.model.model.layers.35.self_attn.o_proj.lora_A.weight"


def test_translate_lora_key_returns_none_for_base_weight():
    assert _translate_lora_key("layers.0.attn.q_proj.weight") is None


def test_translate_lora_key_returns_none_for_embedding():
    assert _translate_lora_key("embed_tokens.weight") is None


def test_translate_lora_key_returns_none_for_unknown_module():
    assert _translate_lora_key("layers.0.attn.unknown_proj.lora_a.weight") is None


def test_translate_lora_key_strips_fsdp_prefix():
    key = "_fsdp_wrapped_module.layers.5.attn.k_proj.lora_a.weight"
    result = _translate_lora_key(key)
    assert result == "base_model.model.model.layers.5.self_attn.k_proj.lora_A.weight"


# ── torchtune_to_peft_state_dict ─────────────────────────────────────────────

def test_roundtrip_attn_modules():
    sd = _make_adapter_sd(_ATTN_MODULES, num_layers=2, rank=4)
    peft_sd, cfg = torchtune_to_peft_state_dict(
        sd,
        model_name="/fake/path",
        rank=4,
        alpha=8.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    assert len(peft_sd) == len(sd), "No keys should be dropped for valid attn modules"
    for k in peft_sd:
        assert k.startswith("base_model.model.model.layers."), f"Unexpected key prefix: {k!r}"
        assert "lora_A." in k or "lora_B." in k, f"PEFT casing missing in: {k!r}"


def test_roundtrip_mlp_modules():
    sd = _make_adapter_sd(_MLP_MODULES, num_layers=2, rank=4)
    peft_sd, cfg = torchtune_to_peft_state_dict(
        sd,
        model_name="/fake/path",
        rank=4,
        alpha=8.0,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )
    assert len(peft_sd) == len(sd)
    # Check gate/up/down names appear
    peft_names = " ".join(peft_sd.keys())
    assert "gate_proj" in peft_names
    assert "up_proj" in peft_names
    assert "down_proj" in peft_names


def test_output_proj_maps_to_o_proj():
    sd = {"layers.0.attn.output_proj.lora_a.weight": torch.zeros(1)}
    peft_sd, _ = torchtune_to_peft_state_dict(
        sd, model_name="m", rank=4, alpha=8.0, target_modules=["o_proj"]
    )
    key = list(peft_sd.keys())[0]
    assert "o_proj" in key, f"Expected o_proj in key, got: {key!r}"
    assert "output_proj" not in key, f"output_proj should not appear in PEFT key: {key!r}"


def test_adapter_config_fields():
    sd = _make_adapter_sd(["attn.q_proj"], num_layers=1, rank=4)
    _, cfg = torchtune_to_peft_state_dict(
        sd,
        model_name="/path/to/model",
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj"],
    )
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["r"] == 16
    assert cfg["lora_alpha"] == 32.0
    assert cfg["target_modules"] == ["k_proj", "q_proj"]  # sorted
    assert cfg["base_model_name_or_path"] == "/path/to/model"
    assert cfg["lora_dropout"] == 0.0
    assert cfg["bias"] == "none"


def test_raises_on_untranslatable_key():
    sd = {"layers.0.attn.q_proj.lora_a.weight": torch.zeros(1),
          "embed_tokens.weight": torch.zeros(1)}  # not a LoRA key
    with pytest.raises(ValueError, match="Could not translate"):
        torchtune_to_peft_state_dict(
            sd, model_name="m", rank=4, alpha=8.0, target_modules=["q_proj"]
        )


def test_full_all_modules_roundtrip():
    """End-to-end: all 7 target modules × 2 layers × lora_a/lora_b."""
    sd = _make_adapter_sd(_ALL_MODULES, num_layers=2, rank=8)
    target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_sd, cfg = torchtune_to_peft_state_dict(
        sd, model_name="m", rank=8, alpha=16.0, target_modules=target
    )
    # 7 modules × 2 layers × 2 (lora_a + lora_b) = 28 keys
    assert len(peft_sd) == 28, f"Expected 28 keys, got {len(peft_sd)}"
    assert cfg["target_modules"] == sorted(target)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe structural test: LoRA adapter parameter completeness.

Uses a tiny synthetic Qwen3-LoRA model (2 layers, 64 embed_dim) to verify
that ``get_adapter_params`` returns exactly the expected adapter tensors and
that base weights are frozen. The same code paths used in lora_helpers and
the LoRA-GRPO recipe are exercised without the memory overhead of the full 4B
model.

What we guard:
  - Every (attn + mlp) module that LoRA is applied to has exactly one
    lora_a and one lora_b parameter.
  - Base model parameters (non-adapter) are all frozen (requires_grad=False).
  - Adapter parameters are all trainable (requires_grad=True).
  - ``adapter_optimizer_params`` returns the same set as ``get_adapter_params``
    values (flat list, no duplicates).
"""
import torch
import pytest

from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.models.qwen3._component_builders import lora_qwen3
from torchtune.dev.rl.lora_helpers import adapter_optimizer_params


RANK = 4
ALPHA = 8.0
NUM_LAYERS = 2


def _build_tiny_lora_model(
    lora_attn_modules,
    apply_lora_to_mlp=False,
):
    """Build a tiny Qwen3-LoRA model on CPU with minimal dimensions."""
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=False,
        vocab_size=64,
        num_layers=NUM_LAYERS,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        intermediate_dim=128,
        max_seq_len=32,
        head_dim=16,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1_000_000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
    )


def test_adapter_params_attn_only():
    attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
    model = _build_tiny_lora_model(lora_attn_modules=attn_modules)
    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)

    # Each module gets lora_a + lora_b → 4 modules × 2 × num_layers = 16 keys
    expected_count = len(attn_modules) * 2 * NUM_LAYERS
    assert len(adapter_params) == expected_count, (
        f"Expected {expected_count} adapter params for attn-only LoRA, "
        f"got {len(adapter_params)}: {list(adapter_params.keys())}"
    )

    # All adapter params should have requires_grad=True
    for name, param in adapter_params.items():
        assert param.requires_grad, f"Adapter param {name!r} not trainable"

    # All non-adapter params should be frozen
    for name, param in model.named_parameters():
        if name not in adapter_params:
            assert not param.requires_grad, f"Base param {name!r} should be frozen"


def test_adapter_params_attn_and_mlp():
    attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
    model = _build_tiny_lora_model(
        lora_attn_modules=attn_modules,
        apply_lora_to_mlp=True,
    )
    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)

    # MLP adds gate_proj, up_proj, down_proj → 3 more modules per layer
    mlp_modules = 3
    expected_count = (len(attn_modules) + mlp_modules) * 2 * NUM_LAYERS
    assert len(adapter_params) == expected_count, (
        f"Expected {expected_count} adapter params (attn+mlp LoRA), "
        f"got {len(adapter_params)}"
    )


def test_adapter_params_lora_a_b_naming():
    model = _build_tiny_lora_model(lora_attn_modules=["q_proj"])
    adapter_params = get_adapter_params(model)

    # All keys must contain "lora" (PEFT naming)
    for name in adapter_params:
        assert "lora" in name.lower(), f"Unexpected non-lora adapter param: {name!r}"

    # Should have both lora_a and lora_b for each layer
    has_a = any("lora_a" in n for n in adapter_params)
    has_b = any("lora_b" in n for n in adapter_params)
    assert has_a, "No lora_a weights found in adapter_params"
    assert has_b, "No lora_b weights found in adapter_params"


def test_adapter_optimizer_params_flat_list():
    model = _build_tiny_lora_model(lora_attn_modules=["q_proj", "v_proj"])
    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)

    opt_params = adapter_optimizer_params(model)

    assert isinstance(opt_params, list), "adapter_optimizer_params must return a list"
    assert len(opt_params) == len(adapter_params), (
        f"optimizer param count ({len(opt_params)}) != adapter_params count ({len(adapter_params)})"
    )
    # Each entry must be an nn.Parameter
    for p in opt_params:
        assert isinstance(p, torch.nn.Parameter), f"Expected Parameter, got {type(p)}"

    # No duplicates (by identity)
    ids = [id(p) for p in opt_params]
    assert len(ids) == len(set(ids)), "Duplicate parameters in adapter_optimizer_params"


def test_no_base_weights_in_adapter_params():
    """Base model weights must not appear in adapter_params."""
    model = _build_tiny_lora_model(
        lora_attn_modules=["q_proj", "k_proj"],
        apply_lora_to_mlp=True,
    )
    adapter_params = get_adapter_params(model)

    for name in adapter_params:
        # PEFT adapter params always contain "lora" or "magnitude"
        assert "lora" in name or "magnitude" in name, (
            f"Non-adapter param leaked into adapter_params: {name!r}"
        )

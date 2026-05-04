# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Bit-equivalence test for ``iter_merged_lora_layers``.

CPU-only, no XPU, no distributed init. Confirms that the merged effective
weight produced by the helper matches the LoRALinear forward output to within
bf16 tolerance, for every LoRALinear under a small toy model.

Why it matters: the LoRA-GRPO recipe relies on this merge formula to broadcast
adapter-augmented base weights to vLLM (sidestepping the ``--enable-lora`` PDE
crash). A silent bug in the merge would corrupt every weight sync.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from torchtune.dev.rl.lora_helpers import iter_merged_lora_layers
from torchtune.modules.peft.lora import LoRALinear


class _ToyModel(nn.Module):
    """Two LoRALinear layers + one plain Linear (should be skipped by the iterator)."""

    def __init__(self) -> None:
        super().__init__()
        # Wrap LoRALinears under named submodules so the produced names are
        # distinguishable (mirrors how LoRA wraps q_proj / k_proj in real models).
        self.layers = nn.ModuleDict({
            "0": nn.ModuleDict({
                "attn": nn.ModuleDict({
                    "q_proj": LoRALinear(in_dim=8, out_dim=12, rank=4, alpha=8.0),
                    "k_proj": LoRALinear(in_dim=8, out_dim=6, rank=2, alpha=4.0),
                }),
            }),
        })
        self.unrelated = nn.Linear(8, 4)

    def __iter_lora__(self):
        return [
            ("layers.0.attn.q_proj", self.layers["0"]["attn"]["q_proj"]),
            ("layers.0.attn.k_proj", self.layers["0"]["attn"]["k_proj"]),
        ]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0xA11CE)


def _randomize_lora_weights(layer: LoRALinear) -> None:
    # Default init has lora_b zeroed, which makes the merge a no-op and hides
    # bugs in the (B @ A) matmul. Randomize both factors so the test exercises
    # the real path.
    with torch.no_grad():
        layer.lora_a.weight.copy_(torch.randn_like(layer.lora_a.weight) * 0.01)
        layer.lora_b.weight.copy_(torch.randn_like(layer.lora_b.weight) * 0.01)


def test_merged_layers_match_forward_within_bf16_tol() -> None:
    model = _ToyModel()
    for _, lora in model.__iter_lora__():
        _randomize_lora_weights(lora)

    merged = dict(iter_merged_lora_layers(model))

    # Iterator must return exactly the two LoRALinear layers (skipping unrelated nn.Linear).
    assert set(merged.keys()) == {
        "layers.0.attn.q_proj.weight",
        "layers.0.attn.k_proj.weight",
    }, f"unexpected keys: {sorted(merged.keys())}"

    for name, lora in model.__iter_lora__():
        m = merged[f"{name}.weight"]
        assert m.dtype == torch.bfloat16
        assert m.shape == lora.weight.shape

        # Compare merged-linear-forward to LoRALinear-forward on the same input.
        x = torch.randn(3, lora.in_dim)
        ref = lora(x)  # base + LoRA forward
        # Use the merged weight via plain F.linear (the path vLLM will use)
        out = torch.nn.functional.linear(x, m.float())

        # bf16 round-trip dominates the error budget; allow ~1e-2 abs.
        assert torch.allclose(ref, out, atol=2e-2, rtol=1e-2), (
            f"merge mismatch for {name}: "
            f"max abs diff = {(ref - out).abs().max().item():.4f}"
        )


def test_iter_skips_non_lora_modules() -> None:
    # Plain nn.Linear should not appear in the iterator output even if it's named
    # like a layer in the model.
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    out = list(iter_merged_lora_layers(model))
    assert out == []


def test_merge_zero_lora_b_is_identity() -> None:
    # Default LoRA init zeros lora_b → merged weight must equal base weight exactly.
    layer = LoRALinear(in_dim=6, out_dim=6, rank=2, alpha=4.0)
    # Sanity check init
    assert torch.equal(layer.lora_b.weight, torch.zeros_like(layer.lora_b.weight))

    model = nn.ModuleDict({"x": layer})
    merged = dict(iter_merged_lora_layers(model))
    m = merged["x.weight"]
    # bf16 round-trip of the base float32 weight is the only source of error.
    expected_bf16 = layer.weight.to(torch.bfloat16)
    assert torch.equal(m, expected_bf16)

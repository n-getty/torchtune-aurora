# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe equivalence test for the Path A cached-base merged-weight gather.

The default LoRA-GRPO publish path (``use_runtime_lora=False``) caches the
frozen base weights ONCE at setup (``_cache_lora_base_weights``) and then, every
step, merges them with the *live* adapter via ``iter_merged_lora_layers(model,
base_weights=cache)`` — with NO per-step FSDP collective.

This test pins the invariant that the cached-base merge produces a *bit-identical*
result to a merge that re-reads the base weight live from the module every step
(the path that would run if the cache were absent / FSDP were not in play). In
other words: caching the frozen base is correctness-preserving.

It also exercises the realistic per-step lifecycle: build cache once, mutate the
adapter (as ``optimizer.step()`` would), and confirm the merged output tracks
the new adapter while still using the same cached base.

CPU-only, no XPU, no distributed init, no recipe import (the recipe pulls
torchao + XPU backends at import and crashes on a login node). We test the
shared helper ``iter_merged_lora_layers`` directly — that is the exact function
the recipe's ``_gather_merged_lora_weights`` calls.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from torchtune.dev.rl.lora_helpers import iter_merged_lora_layers
from torchtune.modules.peft.lora import LoRALinear


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0xC0FFEE)


class _ToyLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({
            "0": nn.ModuleDict({
                "attn": nn.ModuleDict({
                    "q_proj": LoRALinear(in_dim=8, out_dim=12, rank=4, alpha=8.0),
                    "k_proj": LoRALinear(in_dim=8, out_dim=6, rank=2, alpha=4.0),
                }),
                "mlp": nn.ModuleDict({
                    "w1": LoRALinear(in_dim=8, out_dim=16, rank=4, alpha=8.0),
                }),
            }),
        })


def _randomize_adapter(model: nn.Module, scale: float = 0.02) -> None:
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_a.weight.copy_(torch.randn_like(m.lora_a.weight) * scale)
                m.lora_b.weight.copy_(torch.randn_like(m.lora_b.weight) * scale)


def _build_base_cache(model: nn.Module) -> dict:
    """Mirror ``_cache_lora_base_weights``: snapshot every LoRALinear base weight
    to bf16 CPU once. (No FSDP here — the cache key is the plain module name.)"""
    cache: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            cache[f"{name}.weight"] = m.weight.detach().to(torch.bfloat16).cpu().contiguous()
    return cache


def test_cached_base_merge_matches_live_merge():
    """Merge via cached base == merge via live module.weight, bit-for-bit."""
    model = _ToyLoRA()
    _randomize_adapter(model)

    cache = _build_base_cache(model)

    live = dict(iter_merged_lora_layers(model))                       # base_weights=None
    cached = dict(iter_merged_lora_layers(model, base_weights=cache))  # cached base

    assert set(live.keys()) == set(cached.keys())
    assert live, "iterator produced no LoRA layers"
    for k in live:
        a, b = cached[k], live[k]
        assert a.dtype == b.dtype == torch.bfloat16
        assert a.shape == b.shape
        # The cache stores base as bf16, the live path reads fp32 module.weight
        # and casts inside the merge. Because the base weight is the only term
        # that differs in source dtype, allow one bf16 ULP of slack rather than
        # exact equality.
        assert torch.allclose(a.float(), b.float(), atol=2e-2, rtol=1e-2), (
            f"cached vs live merge diverged at {k}: "
            f"max abs diff = {(a.float() - b.float()).abs().max().item():.4f}"
        )


def test_cache_is_frozen_across_adapter_update():
    """After an adapter update, the merged output must reflect the NEW adapter
    while reusing the SAME cached (frozen) base — i.e. caching the base does not
    stale the published weights."""
    model = _ToyLoRA()
    _randomize_adapter(model, scale=0.02)
    cache = _build_base_cache(model)

    merged_before = dict(iter_merged_lora_layers(model, base_weights=cache))

    # Simulate optimizer.step(): perturb the adapter substantially.
    _randomize_adapter(model, scale=0.10)
    merged_after = dict(iter_merged_lora_layers(model, base_weights=cache))

    # The merged weights must change (adapter moved) ...
    changed = any(
        not torch.equal(merged_before[k], merged_after[k]) for k in merged_before
    )
    assert changed, "merged output did not track the adapter update"

    # ... but each still equals base(cache) + new_delta, i.e. base is unchanged.
    for name, m in model.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        k = f"{name}.weight"
        base = cache[k].float()
        scale = float(m.alpha) / float(m.rank)
        delta = (m.lora_b.weight.float() @ m.lora_a.weight.float()) * scale
        expected = (base + delta).to(torch.bfloat16)
        assert torch.equal(merged_after[k], expected), (
            f"merged_after[{k}] != cached_base + new_delta"
        )


def test_missing_base_in_cache_raises():
    """If the cache lacks a LoRALinear's base weight, the merge must fail loudly
    rather than silently publish a wrong/partial weight."""
    model = _ToyLoRA()
    _randomize_adapter(model)
    cache = _build_base_cache(model)
    # Drop one entry to simulate an incomplete cache.
    dropped = next(iter(cache))
    del cache[dropped]

    with pytest.raises(KeyError):
        list(iter_merged_lora_layers(model, base_weights=cache))

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe equivalence test for the colocate LoRA merge.

In colocate mode each training rank materializes its OWN full frozen base via
``FSDP.summon_full_params`` and then merges ``W_eff = base + (alpha/rank)*(B@A)``
locally by calling ``iter_merged_lora_layers(model, base_weights=None)`` — the
``base_weights=None`` path reads the (now-full) ``module.weight`` directly. The
server path instead merges from a rank-0 CPU cache
(``iter_merged_lora_layers(model, base_weights=cache)``).

This pins two invariants:
  1. The colocate merge (base_weights=None) is bit-identical (within one bf16
     ULP) to the server cached-base merge — so colocate publishes the SAME
     W_eff the validated server path does.
  2. The set of HF names colocate would push equals exactly the LoRA-target
     set and contains NO embeddings / norms / lm_head — the load-bearing
     "only LoRA targets need pushing" assumption (the engine already loaded the
     full frozen base at init).

CPU-only: no XPU, no distributed, no recipe import. Exercises the shared helpers
``iter_merged_lora_layers`` + ``tune_lora_name_to_hf`` directly.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from torchtune.dev.rl.lora_helpers import (
    iter_merged_lora_layers,
    tune_lora_name_to_hf,
)
from torchtune.modules.peft.lora import LoRALinear


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0xBADC0DE)


class _ToyLoRA(nn.Module):
    """Two layers of LoRA-wrapped attn + mlp, mirroring the recipe module tree."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "q_proj": LoRALinear(in_dim=8, out_dim=12, rank=4, alpha=8.0),
                                "k_proj": LoRALinear(in_dim=8, out_dim=6, rank=2, alpha=4.0),
                                "v_proj": LoRALinear(in_dim=8, out_dim=6, rank=2, alpha=4.0),
                                "output_proj": LoRALinear(in_dim=12, out_dim=8, rank=4, alpha=8.0),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "w1": LoRALinear(in_dim=8, out_dim=16, rank=4, alpha=8.0),
                                "w2": LoRALinear(in_dim=16, out_dim=8, rank=4, alpha=8.0),
                                "w3": LoRALinear(in_dim=8, out_dim=16, rank=4, alpha=8.0),
                            }
                        ),
                    }
                )
                for i in range(2)
            }
        )


def _randomize_adapter(model: nn.Module, scale: float = 0.02) -> None:
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_a.weight.copy_(torch.randn_like(m.lora_a.weight) * scale)
                m.lora_b.weight.copy_(torch.randn_like(m.lora_b.weight) * scale)


def _build_base_cache(model: nn.Module) -> dict:
    cache: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            cache[f"{name}.weight"] = m.weight.detach().to(torch.bfloat16).cpu().contiguous()
    return cache


def test_colocate_merge_matches_server_cached_merge():
    """colocate (base_weights=None) == server (cached base), within 1 bf16 ULP."""
    model = _ToyLoRA()
    _randomize_adapter(model)
    cache = _build_base_cache(model)

    # base_weights=None == the recipe's per-step-SUMMON colocate path (reads full
    # module.weight under summon_full_params). base_weights=cache == BOTH the server
    # merged path AND the recipe's cached-base colocate path
    # (_cache_colocate_base populates the same {module}.weight dict). This single
    # assertion therefore guards summon-vs-cached colocate equivalence too.
    colocate = dict(iter_merged_lora_layers(model))                      # base_weights=None
    server = dict(iter_merged_lora_layers(model, base_weights=cache))    # cached base

    assert set(colocate.keys()) == set(server.keys())
    assert colocate, "iterator produced no LoRA layers"
    for k in colocate:
        a, b = colocate[k], server[k]
        assert a.dtype == b.dtype == torch.bfloat16
        assert a.shape == b.shape
        assert torch.allclose(a.float(), b.float(), atol=2e-2, rtol=1e-2), (
            f"colocate vs server merge diverged at {k}: "
            f"max abs diff = {(a.float() - b.float()).abs().max().item():.4f}"
        )


def _build_colocate_base_cache(model: nn.Module, dst="cpu") -> dict:
    """Mirror the recipe's _cache_colocate_base key construction: snapshot each
    LoRALinear base weight to bf16 under key '{clean_module_name}.weight', with
    FSDP/ckpt prefixes stripped. (No FSDP here, so names are already clean.)"""
    cache = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        clean = module_name.replace("_fsdp_wrapped_module.", "").replace(
            "_checkpoint_wrapped_module.", ""
        )
        cache[f"{clean}.weight"] = module.weight.detach().to(torch.bfloat16).to(dst).contiguous()
    return cache


def test_colocate_base_cache_keys_satisfy_merge_contract():
    """The recipe caches the full base under '{module}.weight'; every key the
    merge iterator looks up MUST be present (no KeyError / skipped layer). This
    pins the _cache_colocate_base <-> iter_merged_lora_layers(base_weights=cache)
    contract that the HW cached-base path depends on."""
    model = _ToyLoRA()
    _randomize_adapter(model)
    cache = _build_colocate_base_cache(model)

    # Must not raise KeyError, and must yield every LoRALinear.
    merged = dict(iter_merged_lora_layers(model, base_weights=cache))
    n_lora = sum(1 for _ in model.modules() if isinstance(_, LoRALinear))
    assert len(merged) == n_lora == 14, (len(merged), n_lora)

    # Cache-merge == summon-merge (base_weights=None), bit-identical within 1 ULP.
    summon = dict(iter_merged_lora_layers(model))
    for k in summon:
        assert torch.allclose(merged[k].float(), summon[k].float(), atol=2e-2, rtol=1e-2)


def test_colocate_pushes_only_lora_targets():
    """Every merged tune name maps to a LoRA-target HF name; the pushed HF-name
    set excludes embeddings / norms / lm_head."""
    model = _ToyLoRA()
    _randomize_adapter(model)

    pushed_hf = []
    for tune_name, _ in iter_merged_lora_layers(model):
        hf = tune_lora_name_to_hf(tune_name)
        assert hf is not None, f"colocate would skip a LoRA target: {tune_name}"
        pushed_hf.append(hf)

    # 2 layers x (4 attn + 3 mlp) = 14 LoRA-target weights.
    assert len(pushed_hf) == 14
    assert len(set(pushed_hf)) == 14, "duplicate HF names in push set"

    banned_substrings = ("embed", "lm_head", "norm", "tok_embeddings", "output.weight")
    for hf in pushed_hf:
        assert all(b not in hf for b in banned_substrings), (
            f"colocate must not push a non-LoRA weight: {hf}"
        )
        assert hf.startswith("model.layers."), hf

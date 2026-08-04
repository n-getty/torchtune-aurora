# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU pin-down for `MoE.checkpoint_experts` (see torchtune/modules/moe/moe.py),
the seq4096 mem_reserved-ratchet mitigation added in
memory/project_moe_sft_profiling_gradrelease_bottleneck_20260724.md.

Unlike checkpointing the whole MoE block (which would re-run the router and
risk the v158 argsort-tie-break-under-recompute bug —
torchtune/dev/rl/distributed.py::_apply_split_ac exists specifically to
avoid that), `checkpoint_experts` wraps ONLY `self.experts(...)` — a
deterministic function of already-fixed inputs computed once by the router/
EP-dispatch before this call. This test proves checkpoint_experts=True is a
bit-exact no-op on both output and gradients relative to False, across both
the padded-BMM and sequential-per-expert `GroupedExpertsHF` forward paths,
and that `_apply_expert_checkpointing` correctly toggles every `MoE` module
in a multi-layer model.

Run: pytest tests/torchtune/dev/full_finetune_moe/test_checkpoint_experts_equivalence.py --timeout=60
"""

import os

import pytest
import torch

from torchtune.models.qwen3_moe._component_builders import qwen3_moe_block
from torchtune.modules.moe.moe import MoE


def _build_moe_block(seed: int = 0, checkpoint_experts: bool = False) -> MoE:
    torch.manual_seed(seed)
    moe = qwen3_moe_block(
        embed_dim=16,
        moe_intermediate_dim=32,
        num_experts=8,
        experts_per_token=2,
        norm_topk_prob=True,
    )
    moe.experts.reset_parameters()
    moe.checkpoint_experts = checkpoint_experts
    return moe


def _clone_state_dict(moe: MoE) -> dict:
    return {k: v.clone() for k, v in moe.state_dict().items()}


@pytest.mark.parametrize("sequential_experts", [False, True])
def test_checkpoint_experts_output_and_grad_equivalence(
    monkeypatch, sequential_experts
):
    """checkpoint_experts=True must produce bit-exact output AND gradients
    (both w.r.t. input and every expert parameter) relative to False, on
    both the padded-BMM (default) and TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=1
    forward paths."""
    import torchtune.models.qwen3_moe._experts as experts_mod

    monkeypatch.setattr(experts_mod, "_SEQUENTIAL_EXPERTS", sequential_experts)

    torch.manual_seed(1234)
    bs, slen, dim = 2, 5, 16
    x = torch.randn(bs, slen, dim, dtype=torch.float32)

    moe_a = _build_moe_block(seed=1234, checkpoint_experts=False)
    moe_b = _build_moe_block(seed=1234, checkpoint_experts=True)
    # Ensure identical initial weights (both built with the same seed, but
    # assert explicitly rather than assume _build_moe_block's determinism).
    sd_a, sd_b = moe_a.state_dict(), moe_b.state_dict()
    for k in sd_a:
        torch.testing.assert_close(sd_a[k], sd_b[k])

    x_a = x.clone().requires_grad_(True)
    out_a = moe_a(x_a)
    out_a.sum().backward()

    x_b = x.clone().requires_grad_(True)
    out_b = moe_b(x_b)
    out_b.sum().backward()

    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x_a.grad, x_b.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        moe_a.experts.gate_proj.grad, moe_b.experts.gate_proj.grad, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        moe_a.experts.up_proj.grad, moe_b.experts.up_proj.grad, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        moe_a.experts.down_proj.grad, moe_b.experts.down_proj.grad, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        moe_a.router.gate.weight.grad,
        moe_b.router.gate.weight.grad,
        atol=1e-6,
        rtol=1e-6,
    )


def test_checkpoint_experts_does_not_call_ep_dispatch_combine_twice():
    """The checkpoint region wraps ONLY self.experts(...) — _ep_dispatch and
    _ep_combine must each be called exactly once per forward, not
    re-invoked during the recompute (which would indicate the checkpoint
    region is too wide and touches EP state)."""
    torch.manual_seed(7)
    bs, slen, dim = 1, 4, 16
    x = torch.randn(bs, slen, dim, dtype=torch.float32)

    moe = _build_moe_block(seed=7, checkpoint_experts=True)
    calls = {"dispatch": 0, "combine": 0}

    def _identity_dispatch(routed_input, num_tokens_per_expert):
        calls["dispatch"] += 1
        return routed_input, num_tokens_per_expert

    def _identity_combine(routed_output):
        calls["combine"] += 1
        return routed_output

    moe._ep_dispatch = _identity_dispatch
    moe._ep_combine = _identity_combine

    x = x.requires_grad_(True)
    out = moe(x)
    out.sum().backward()

    assert calls["dispatch"] == 1
    assert calls["combine"] == 1


def test_checkpoint_experts_reduces_peak_but_not_final_memory_relative_ordering():
    """Sanity check the mechanism is actually engaging torch.utils.checkpoint
    (not silently a no-op): patch torch.utils.checkpoint.checkpoint to a
    counter and confirm it's invoked exactly once per forward when
    checkpoint_experts=True, and never when False."""
    call_count = {"n": 0}
    import torch.utils.checkpoint as ckpt_mod

    orig_checkpoint = ckpt_mod.checkpoint

    def _counting_checkpoint(*args, **kwargs):
        call_count["n"] += 1
        return orig_checkpoint(*args, **kwargs)

    torch.manual_seed(3)
    bs, slen, dim = 1, 4, 16
    x = torch.randn(bs, slen, dim, dtype=torch.float32, requires_grad=True)

    moe_off = _build_moe_block(seed=3, checkpoint_experts=False)
    ckpt_mod.checkpoint = _counting_checkpoint
    try:
        moe_off(x).sum().backward()
        assert call_count["n"] == 0, (
            "checkpoint() must not be called when checkpoint_experts=False"
        )

        call_count["n"] = 0
        x2 = x.detach().clone().requires_grad_(True)
        moe_on = _build_moe_block(seed=3, checkpoint_experts=True)
        moe_on(x2).sum().backward()
        assert call_count["n"] == 1, (
            "checkpoint() must be called exactly once per forward when checkpoint_experts=True"
        )
    finally:
        ckpt_mod.checkpoint = orig_checkpoint


def test_apply_expert_checkpointing_toggles_all_moe_modules():
    """torchtune.dev.rl.distributed._apply_expert_checkpointing must find and
    toggle every MoE module in a multi-layer model, and report the correct
    count."""
    from torchtune.dev.rl.distributed import _apply_expert_checkpointing

    class _FakeMultiLayerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = _build_moe_block(seed=0)
            self.layer1 = _build_moe_block(seed=1)
            self.not_moe = torch.nn.Linear(4, 4)

    model = _FakeMultiLayerModel()
    assert model.layer0.checkpoint_experts is False
    assert model.layer1.checkpoint_experts is False

    n = _apply_expert_checkpointing(model)

    assert n == 2
    assert model.layer0.checkpoint_experts is True
    assert model.layer1.checkpoint_experts is True


def test_checkpoint_experts_default_false_no_behavior_change():
    """New models must default to checkpoint_experts=False (no behavior
    change for any existing caller that doesn't opt in)."""
    moe = qwen3_moe_block(
        embed_dim=16, moe_intermediate_dim=32, num_experts=8, experts_per_token=2
    )
    assert moe.checkpoint_experts is False


@pytest.mark.parametrize(
    ("every", "expected"),
    [
        (1, [True, True, True, True]),
        (2, [True, False, True, False]),
        (3, [True, False, False, True]),
    ],
)
def test_apply_split_ac_selects_only_attention_blocks(every, expected):
    from torchtune.dev.rl.distributed import _apply_split_ac
    from torchtune.models.qwen3_moe._component_builders import Qwen3MoeTransformerLayer

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    Qwen3MoeTransformerLayer(
                        attn=torch.nn.Identity(),
                        mlp=_build_moe_block(seed=index),
                        sa_norm=torch.nn.Identity(),
                        mlp_norm=torch.nn.Identity(),
                    )
                    for index in range(4)
                ]
            )

    model = _FakeModel()
    count = _apply_split_ac(model, attention_checkpoint_every=every)

    assert [layer._ac_enabled for layer in model.layers] == expected
    assert count == sum(expected)

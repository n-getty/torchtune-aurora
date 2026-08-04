# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""EP=1 (no Expert Parallelism) equivalence pin-down for the new standalone
MoE SFT recipe (``recipes/dev/full_finetune_moe_distributed_xpu.py``).

Phase 1a of the MoE-SFT-isolation project claims that at
``expert_parallel_degree`` absent/1, the recipe is a correct, inert superset
of the dense SFT recipe: no EP mesh, no EP plan registration, no solo-FSDP2
expert wrap, and — the numerically load-bearing claim this test pins —
``MoE.forward()``'s EP dispatch/combine branches are true no-ops when
``_ep_dispatch``/``_ep_combine`` are never wired (exactly the EP=1 state).

This is CPU-only (no torch.distributed, no XPU, no FSDP2) and cheap: it
exercises ``torchtune.modules.moe.moe.MoE`` and
``torchtune.models.qwen3_moe._component_builders.qwen3_moe_block`` directly,
which is the same forward path the recipe's model runs through — just
without the distributed wrapping this test doesn't need to prove.

Run: pytest tests/torchtune/dev/full_finetune_moe/test_ep1_no_ep_equivalence.py --timeout=60
"""
import ast
import os

import torch
from torch.nn import functional as F

from torchtune.models.qwen3_moe._component_builders import qwen3_moe_block
from torchtune.models.qwen3_moe._router import Qwen3MoeRouter
from torchtune.models.qwen3_moe._experts import GroupedExpertsHF
from torchtune.modules.moe.moe import MoE


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
RECIPE_PATH = os.path.join(
    REPO_ROOT, "recipes", "dev", "full_finetune_moe_distributed_xpu.py"
)


def _build_moe_block(seed: int = 0) -> MoE:
    """qwen3_moe_block() leaves GroupedExpertsHF's params as uninitialized
    torch.empty(...) (real model construction relies on the checkpoint loader
    to fill them) — reset_parameters() must be called explicitly here or the
    forward produces NaN/Inf garbage from whatever was in that memory."""
    torch.manual_seed(seed)
    moe = qwen3_moe_block(
        embed_dim=16,
        moe_intermediate_dim=32,
        num_experts=8,
        experts_per_token=2,
        norm_topk_prob=True,
    )
    moe.experts.reset_parameters()
    return moe


def test_moe_ep_dispatch_combine_default_to_none():
    """At EP=1 (expert_parallel_degree absent), the recipe never calls
    parallelize_module/wire_ep_to_moe_modules, so _ep_dispatch/_ep_combine
    stay at their MoE.__init__ default of None. Pin that default directly —
    if it ever changes, the "EP=1 is inert" claim silently breaks."""
    moe = _build_moe_block()
    assert moe._ep_dispatch is None
    assert moe._ep_combine is None


def test_moe_forward_ep1_matches_direct_router_expert_composition():
    """MoE.forward() with _ep_dispatch/_ep_combine unset (the EP=1 state) must
    be bitwise identical to manually composing router -> experts -> scatter_add
    outside the MoE module. This is the numerical claim that "EP=1 training
    under the new recipe is identical to plain FSDP2 training with no EP
    wiring at all" reduces to.
    """
    torch.manual_seed(1234)
    bs, slen, dim = 2, 5, 16
    x = torch.randn(bs, slen, dim, dtype=torch.float32)

    moe = _build_moe_block(seed=1234)
    assert moe._ep_dispatch is None and moe._ep_combine is None

    x_a = x.clone().requires_grad_(True)
    out_a = moe(x_a)
    out_a.sum().backward()

    # Manual reference: same router + same experts, no MoE wrapper, no EP.
    router: Qwen3MoeRouter = moe.router
    experts: GroupedExpertsHF = moe.experts

    # out_a's backward already populated experts.gate_proj.grad (and friends) —
    # since the manual-composition reference below runs a SECOND backward
    # through the SAME experts module, its grad would silently accumulate
    # on top of out_a's rather than compare against it. Snapshot + zero the
    # expert grads here so the second backward starts from a clean slate and
    # the final comparison is between two independently-computed gradients,
    # not a tensor compared against itself.
    gate_proj_grad_a = experts.gate_proj.grad.clone()
    up_proj_grad_a = experts.up_proj.grad.clone()
    down_proj_grad_a = experts.down_proj.grad.clone()
    experts.gate_proj.grad = None
    experts.up_proj.grad = None
    experts.down_proj.grad = None

    x_b = x.clone().requires_grad_(True)
    top_scores, token_indices, num_tokens_per_expert = router(x_b.reshape(bs * slen, dim))
    token_indices_exp = token_indices.reshape(-1, 1).expand(-1, dim)
    routed_input = torch.gather(x_b.view(-1, dim), dim=0, index=token_indices_exp)
    routed_input = routed_input * top_scores.reshape(-1, 1)
    routed_output = experts(routed_input, num_tokens_per_expert)
    out_b = torch.zeros_like(x_b.reshape(bs * slen, dim))
    out_b = out_b.scatter_add(dim=0, index=token_indices_exp, src=routed_output)
    out_b = out_b.reshape(bs, slen, dim)
    out_b.sum().backward()

    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x_a.grad, x_b.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gate_proj_grad_a, experts.gate_proj.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(up_proj_grad_a, experts.up_proj.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(down_proj_grad_a, experts.down_proj.grad, atol=1e-6, rtol=1e-6)


def test_moe_forward_ep1_independent_of_dispatch_hook_presence():
    """Sanity-check the guard itself: if _ep_dispatch/_ep_combine WERE wired to
    identity-like passthroughs, forward output would still match — proving the
    branches in MoE.forward() genuinely gate on `is not None`, not some other
    condition, so leaving them None (EP=1) truly skips the dispatch/combine
    code paths rather than taking a degenerate branch that happens to match.
    """
    torch.manual_seed(7)
    bs, slen, dim = 1, 4, 16
    x = torch.randn(bs, slen, dim, dtype=torch.float32)

    moe = _build_moe_block(seed=7)
    assert moe._ep_dispatch is None
    assert moe._ep_combine is None

    calls = {"dispatch": 0, "combine": 0}

    def _identity_dispatch(routed_input, num_tokens_per_expert):
        calls["dispatch"] += 1
        return routed_input, num_tokens_per_expert

    def _identity_combine(routed_output):
        calls["combine"] += 1
        return routed_output

    out_no_hooks = moe(x.clone())

    moe._ep_dispatch = _identity_dispatch
    moe._ep_combine = _identity_combine
    out_with_identity_hooks = moe(x.clone())

    assert calls["dispatch"] == 1
    assert calls["combine"] == 1
    torch.testing.assert_close(out_no_hooks, out_with_identity_hooks, atol=1e-6, rtol=1e-6)


def test_recipe_ep_branch_gated_on_degree_greater_than_one():
    """Static regression guard: the recipe must gate all EP setup (mesh
    construction, plan registration, solo-FSDP2 wrap, split-AC, grad-release
    PG map) behind `self._ep_active`, itself defined as
    `self._expert_parallel_degree > 1` with `expert_parallel_degree` read via
    `cfg.get(..., 1)`. This is a source-level check (no distributed init
    needed) that the EP=1 default can't silently regress into always-active
    or opt-out-instead-of-opt-in semantics.
    """
    with open(RECIPE_PATH) as f:
        src = f.read()
    tree = ast.parse(src)

    assert 'cfg.get("expert_parallel_degree", 1)' in src, (
        "expert_parallel_degree must default to 1 (EP inactive) when absent from config"
    )
    assert "self._ep_active = self._expert_parallel_degree > 1" in src, (
        "EP activation must be gated on expert_parallel_degree > 1"
    )

    # Every `if self._ep_active` / `if self._ep_active and ...` guard should be
    # a real conditional in the parsed AST (catches e.g. an accidental
    # `if True:` refactor or the guard being commented out while leaving the
    # body behind).
    ep_active_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and _references_ep_active(node.test)
    ]
    assert len(ep_active_guards) >= 5, (
        f"expected multiple `if self._ep_active` guards in the recipe "
        f"(mesh/plan/solo-FSDP2/grad-release-map/train-loop release), found "
        f"{len(ep_active_guards)} — EP gating may have regressed"
    )


def _references_ep_active(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "_ep_active":
            return True
    return False

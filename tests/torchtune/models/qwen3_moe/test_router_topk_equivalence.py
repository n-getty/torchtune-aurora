# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU equivalence tests for the opt-in Qwen3 MoE top-k router path."""

import torch
from torch import nn

import torchtune.models.qwen3_moe._router as router_module


def _canonicalize_routes(
    scores: torch.Tensor, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    score_order = torch.argsort(scores, stable=True)
    sorted_tokens = tokens[score_order]
    sorted_scores = scores[score_order]
    token_order = torch.argsort(sorted_tokens, stable=True)
    return sorted_tokens[token_order], sorted_scores[token_order]


def _run_router(
    gate: torch.Tensor,
    inputs: torch.Tensor,
    *,
    use_topk: bool,
    use_unstable_grouping: bool = False,
    with_grad: bool = False,
):
    prior = router_module._USE_TOPK_ROUTING
    prior_grouping = router_module._USE_UNSTABLE_EXPERT_GROUPING
    router_module._USE_TOPK_ROUTING = use_topk
    router_module._USE_UNSTABLE_EXPERT_GROUPING = use_unstable_grouping
    try:
        router = router_module.Qwen3MoeRouter(
            gate=nn.Linear(inputs.shape[-1], gate.shape[0], bias=False),
            dim=inputs.shape[-1],
            num_experts=gate.shape[0],
            experts_per_token=2,
            norm_topk_prob=True,
        ).to(dtype=inputs.dtype)
        with torch.no_grad():
            router.gate.weight.copy_(gate)
        router_inputs = inputs.detach().clone().requires_grad_(with_grad)
        outputs = router(router_inputs)
        if with_grad:
            outputs[0].square().sum().backward()
            return outputs, router_inputs.grad.detach(), router.gate.weight.grad.detach()
        return outputs
    finally:
        router_module._USE_TOPK_ROUTING = prior
        router_module._USE_UNSTABLE_EXPERT_GROUPING = prior_grouping


def test_topk_router_matches_stable_sort_for_untied_scores():
    torch.manual_seed(41)
    gate = torch.randn(8, 16)
    inputs = torch.randn(13, 16)
    stable = _run_router(gate, inputs, use_topk=False)
    fast = _run_router(gate, inputs, use_topk=True, use_unstable_grouping=True)
    torch.testing.assert_close(fast[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)


def test_topk_router_preserves_bfloat16_near_tie_ownership():
    torch.manual_seed(61)
    gate = torch.randn(8, 32)
    inputs = torch.randn(64, 32, dtype=torch.bfloat16)
    stable = _run_router(gate, inputs, use_topk=False)
    fast = _run_router(gate, inputs, use_topk=True, use_unstable_grouping=True)
    torch.testing.assert_close(fast[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)


def test_topk_router_matches_bfloat16_reference_across_route_shapes():
    for seed, tokens, num_experts, experts_per_token in (
        (71, 17, 8, 2),
        (72, 31, 16, 4),
        (73, 64, 32, 8),
        (74, 127, 128, 8),
    ):
        torch.manual_seed(seed)
        gate = torch.randn(num_experts, 32)
        inputs = torch.randn(tokens, 32, dtype=torch.bfloat16)
        stable = _run_router(gate, inputs, use_topk=False)
        fast = _run_router(
            gate, inputs, use_topk=True, use_unstable_grouping=True
        )
        torch.testing.assert_close(fast[2], stable[2])
        stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
        fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
        torch.testing.assert_close(fast_tokens, stable_tokens)
        torch.testing.assert_close(fast_scores, stable_scores)


def test_topk_router_falls_back_for_exact_ties():
    gate = torch.zeros(4, 4)
    inputs = torch.randn(7, 4)
    stable = _run_router(gate, inputs, use_topk=False)
    fast = _run_router(gate, inputs, use_topk=True, use_unstable_grouping=True)
    torch.testing.assert_close(fast[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)


def test_topk_router_matches_stable_sort_for_mixed_boundary_ties():
    gate = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    inputs = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    stable = _run_router(gate, inputs, use_topk=False)
    fast = _run_router(gate, inputs, use_topk=True, use_unstable_grouping=True)
    torch.testing.assert_close(fast[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)


def test_topk_boundary_repair_sorts_only_tied_rows(monkeypatch):
    scores = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    calls = []
    original_argsort = router_module.torch.argsort

    def recording_argsort(input_tensor, *args, **kwargs):
        calls.append(tuple(input_tensor.shape))
        return original_argsort(input_tensor, *args, **kwargs)

    monkeypatch.setattr(router_module.torch, "argsort", recording_argsort)
    prior = router_module._USE_TOPK_ROUTING
    router_module._USE_TOPK_ROUTING = True
    try:
        selected = router_module._stable_select_experts(scores, 2)
    finally:
        router_module._USE_TOPK_ROUTING = prior

    torch.testing.assert_close(selected, torch.tensor([[0, 1], [0, 1], [0, 1]]))
    assert calls == [(1, 4)]


def test_topk_boundary_repair_skips_sort_when_no_rows_tie(monkeypatch):
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    original_argsort = router_module.torch.argsort

    def fail_argsort(*args, **kwargs):
        raise AssertionError("untied top-k routing should not call argsort")

    monkeypatch.setattr(router_module.torch, "argsort", fail_argsort)
    prior = router_module._USE_TOPK_ROUTING
    router_module._USE_TOPK_ROUTING = True
    try:
        selected = router_module._stable_select_experts(scores, 2)
    finally:
        router_module._USE_TOPK_ROUTING = prior
        monkeypatch.setattr(router_module.torch, "argsort", original_argsort)

    torch.testing.assert_close(selected, torch.tensor([[0, 1], [0, 1]]))


def test_topk_router_preserves_input_and_gate_gradients():
    torch.manual_seed(43)
    gate = torch.randn(8, 16)
    inputs = torch.randn(13, 16)
    stable = _run_router(gate, inputs, use_topk=False, with_grad=True)
    fast = _run_router(
        gate, inputs, use_topk=True, use_unstable_grouping=True, with_grad=True
    )
    torch.testing.assert_close(fast[0][2], stable[0][2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[0][1], stable[0][0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[0][1], fast[0][0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)
    torch.testing.assert_close(fast[1], stable[1])
    torch.testing.assert_close(fast[2], stable[2])


def test_topk_router_fast_grouping_preserves_route_pairing():
    torch.manual_seed(47)
    gate = torch.randn(8, 16)
    inputs = torch.randn(13, 16)
    stable = _run_router(gate, inputs, use_topk=False)
    fast = _run_router(gate, inputs, use_topk=True, use_unstable_grouping=True)

    stable_scores, stable_tokens, stable_counts = stable
    fast_scores, fast_tokens, fast_counts = fast
    torch.testing.assert_close(fast_counts, stable_counts)
    for token, score in zip(fast_tokens.tolist(), fast_scores.tolist()):
        matching = [
            candidate
            for candidate_token, candidate in zip(
                stable_tokens.tolist(), stable_scores.tolist()
            )
            if candidate_token == token
        ]
        assert any(abs(score - candidate) < 1e-6 for candidate in matching)


def test_topk_router_preserves_full_softmax_when_topk_normalization_is_disabled():
    torch.manual_seed(53)
    gate = torch.randn(8, 16)
    inputs = torch.randn(13, 16)
    prior = router_module._USE_TOPK_ROUTING
    router_module._USE_TOPK_ROUTING = True
    try:
        router = router_module.Qwen3MoeRouter(
            gate=nn.Linear(16, 8, bias=False),
            dim=16,
            num_experts=8,
            experts_per_token=2,
            norm_topk_prob=False,
        )
        with torch.no_grad():
            router.gate.weight.copy_(gate)
        actual = router(inputs)

        logits = inputs @ gate.t()
        selected = torch.argsort(logits, dim=1, stable=True, descending=True)[:, :2]
        expected = torch.softmax(logits, dim=-1).gather(1, selected)
        expected_scores = expected.reshape(-1)[
            torch.argsort(selected.reshape(-1), stable=True)
        ]
        expected_tokens = torch.argsort(selected.reshape(-1), stable=True) // 2

        torch.testing.assert_close(actual[0], expected_scores.to(inputs.dtype))
        torch.testing.assert_close(actual[1], expected_tokens)
    finally:
        router_module._USE_TOPK_ROUTING = prior


def test_topk_router_keeps_fp16_cast_tie_semantics():
    torch.manual_seed(59)
    gate = torch.randn(16, 32)
    inputs = torch.randn(257, 32).half()
    stable = _run_router(gate, inputs, use_topk=False)
    fast = _run_router(gate, inputs, use_topk=True, use_unstable_grouping=True)
    torch.testing.assert_close(fast[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)


def test_unstable_grouping_is_independent_of_topk_selection():
    torch.manual_seed(53)
    gate = torch.randn(8, 16)
    inputs = torch.randn(13, 16)
    stable = _run_router(gate, inputs, use_topk=False)
    grouped = _run_router(
        gate, inputs, use_topk=False, use_unstable_grouping=True
    )
    torch.testing.assert_close(grouped[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    grouped_tokens, grouped_scores = _canonicalize_routes(grouped[1], grouped[0])
    torch.testing.assert_close(grouped_tokens, stable_tokens)
    torch.testing.assert_close(grouped_scores, stable_scores)


def test_topk_router_matches_stable_sort_when_selecting_all_experts():
    gate = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    inputs = torch.randn(5, 2)
    stable_router = router_module.Qwen3MoeRouter(
        gate=nn.Linear(2, 3, bias=False),
        dim=2,
        num_experts=3,
        experts_per_token=3,
        norm_topk_prob=False,
    )
    fast_router = router_module.Qwen3MoeRouter(
        gate=nn.Linear(2, 3, bias=False),
        dim=2,
        num_experts=3,
        experts_per_token=3,
        norm_topk_prob=False,
    )
    with torch.no_grad():
        stable_router.gate.weight.copy_(gate)
        fast_router.gate.weight.copy_(gate)
    prior = router_module._USE_TOPK_ROUTING
    try:
        router_module._USE_TOPK_ROUTING = False
        stable = stable_router(inputs)
        router_module._USE_TOPK_ROUTING = True
        fast = fast_router(inputs)
    finally:
        router_module._USE_TOPK_ROUTING = prior
    torch.testing.assert_close(fast[2], stable[2])
    stable_tokens, stable_scores = _canonicalize_routes(stable[1], stable[0])
    fast_tokens, fast_scores = _canonicalize_routes(fast[1], fast[0])
    torch.testing.assert_close(fast_tokens, stable_tokens)
    torch.testing.assert_close(fast_scores, stable_scores)

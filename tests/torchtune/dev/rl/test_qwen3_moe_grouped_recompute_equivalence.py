# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import torchtune.models.qwen3_moe._experts as experts_mod


def _grouped_mm_reference(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        outputs.append(inputs[start:end] @ weights[expert])
        start = end
    return torch.cat(outputs, dim=0)


def _run_policy(
    policy: str,
    state: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    counts: torch.Tensor,
    loss_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    grouped_mm_calls = 0

    def counted_grouped_mm(
        grouped_inputs: torch.Tensor,
        weights: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal grouped_mm_calls
        grouped_mm_calls += 1
        return _grouped_mm_reference(grouped_inputs, weights, offs)

    prior_grouped_mm: Callable = torch._grouped_mm
    prior_grouped = experts_mod._GROUPED_EXPERTS
    prior_sequential = experts_mod._SEQUENTIAL_EXPERTS
    prior_policy = experts_mod._GROUPED_RECOMPUTE_PREACT
    torch._grouped_mm = counted_grouped_mm
    experts_mod._GROUPED_EXPERTS = True
    experts_mod._SEQUENTIAL_EXPERTS = False
    experts_mod._GROUPED_RECOMPUTE_PREACT = policy
    try:
        experts = experts_mod.GroupedExpertsHF(
            dim=inputs.shape[-1],
            hidden_dim=state["up_proj"].shape[1],
            num_experts=counts.numel(),
        )
        experts.load_state_dict(state)
        policy_inputs = inputs.clone().requires_grad_(True)
        output = experts(policy_inputs, counts)
        (output * loss_weights).sum().backward()
        gradients = {
            "inputs": policy_inputs.grad.detach().clone(),
            **{
                name: parameter.grad.detach().clone()
                for name, parameter in experts.named_parameters()
            },
        }
        return output.detach(), gradients, grouped_mm_calls
    finally:
        torch._grouped_mm = prior_grouped_mm
        experts_mod._GROUPED_EXPERTS = prior_grouped
        experts_mod._SEQUENTIAL_EXPERTS = prior_sequential
        experts_mod._GROUPED_RECOMPUTE_PREACT = prior_policy


@pytest.mark.parametrize(
    "counts",
    [
        [3, 0, 5, 1],
        [0, 0, 0, 9],
        [1, 4, 2, 2],
    ],
)
def test_grouped_recompute_matches_save_all_outputs_and_gradients(counts):
    torch.manual_seed(11)
    num_experts, dim, hidden_dim = len(counts), 16, 32
    count_tensor = torch.tensor(counts, dtype=torch.float32)
    inputs = torch.randn(sum(counts), dim)
    loss_weights = torch.randn(sum(counts), dim)
    reference_experts = experts_mod.GroupedExpertsHF(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
    )
    reference_experts.reset_parameters()
    state = {
        name: value.detach().clone()
        for name, value in reference_experts.state_dict().items()
    }

    reference_output, reference_gradients, reference_calls = _run_policy(
        "0", state, inputs, count_tensor, loss_weights
    )
    assert reference_calls == 3

    for policy, expected_calls in (("1", 5), ("up_only", 4)):
        output, gradients, grouped_mm_calls = _run_policy(
            policy, state, inputs, count_tensor, loss_weights
        )
        torch.testing.assert_close(output, reference_output)
        assert grouped_mm_calls == expected_calls
        assert gradients.keys() == reference_gradients.keys()
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, reference_gradients[name])


def test_grouped_recompute_default_is_disabled(monkeypatch):
    monkeypatch.delenv("TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT", raising=False)
    assert experts_mod._GROUPED_RECOMPUTE_PREACT == "0"


def test_grouped_swiglu_inplace_matches_reference_forward_and_backward():
    torch.manual_seed(31)
    inputs = torch.randn(7, 4, requires_grad=True)
    gate_weight = torch.randn(3, 4, 6, requires_grad=True)
    up_weight = torch.randn(3, 4, 6, requires_grad=True)
    offsets = torch.tensor([3, 3, 7], dtype=torch.int32)

    def grouped_mm(values, weights, offs):
        return _grouped_mm_reference(values, weights, offs)

    inputs_ref = inputs.detach().clone().requires_grad_(True)
    gate_ref = gate_weight.detach().clone().requires_grad_(True)
    up_ref = up_weight.detach().clone().requires_grad_(True)
    gate = _grouped_mm_reference(inputs_ref, gate_ref, offsets)
    up = _grouped_mm_reference(inputs_ref, up_ref, offsets)
    expected = torch.nn.functional.silu(gate) * up
    expected.square().sum().backward()
    expected_grads = (inputs_ref.grad, gate_ref.grad, up_ref.grad)

    prior = torch._grouped_mm
    prior_inplace = experts_mod._USE_INPLACE_SWIGLU
    torch._grouped_mm = grouped_mm
    try:
        for inplace in (False, True):
            experts_mod._USE_INPLACE_SWIGLU = inplace
            test_inputs = inputs.detach().clone().requires_grad_(True)
            test_gate = gate_weight.detach().clone().requires_grad_(True)
            test_up = up_weight.detach().clone().requires_grad_(True)
            actual = experts_mod._grouped_swiglu(
                test_inputs,
                test_gate,
                test_up,
                offsets,
                torch.nn.functional.silu,
            )
            actual.square().sum().backward()
            torch.testing.assert_close(actual.detach(), expected.detach())
            for actual_grad, expected_grad in zip(
                (test_inputs.grad, test_gate.grad, test_up.grad), expected_grads
            ):
                torch.testing.assert_close(actual_grad, expected_grad)
    finally:
        torch._grouped_mm = prior
        experts_mod._USE_INPLACE_SWIGLU = prior_inplace


def test_grouped_experts_preserves_integer_count_dtype(monkeypatch):
    prior = experts_mod._GROUPED_EXPERTS
    prior_mm = torch._grouped_mm
    experts_mod._GROUPED_EXPERTS = True
    torch._grouped_mm = lambda values, weights, offs: _grouped_mm_reference(
        values, weights, offs
    )
    try:
        experts = experts_mod.GroupedExpertsHF(dim=4, hidden_dim=6, num_experts=2)
        counts = torch.tensor([2, 3], dtype=torch.int64)
        inputs = torch.randn(5, 4)
        output = experts(inputs, counts)
        assert output.shape == inputs.shape
        assert counts.dtype == torch.int64
    finally:
        experts_mod._GROUPED_EXPERTS = prior
        torch._grouped_mm = prior_mm


def test_grouped_experts_records_opt_in_gemm_timings():
    prior_grouped = experts_mod._GROUPED_EXPERTS
    prior_mm = torch._grouped_mm
    experts_mod._GROUPED_EXPERTS = True
    torch._grouped_mm = lambda values, weights, offs: _grouped_mm_reference(
        values, weights, offs
    )
    try:
        from torchtune.modules.moe.measurement import MoEMeasurementCollector

        experts = experts_mod.GroupedExpertsHF(dim=4, hidden_dim=6, num_experts=2)
        experts._moe_measurement = MoEMeasurementCollector(enabled=True)
        counts = torch.tensor([2, 3], dtype=torch.int64)
        inputs = torch.randn(5, 4)
        experts(inputs, counts)
        assert set(experts._moe_measurement.record.timings_s) == {
            "grouped_gemm_gate",
            "grouped_gemm_up",
            "grouped_gemm_down",
        }
    finally:
        torch._grouped_mm = prior_mm
        experts_mod._GROUPED_EXPERTS = prior_grouped

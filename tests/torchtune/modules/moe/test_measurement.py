# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torchtune.modules.moe.moe as moe_module

from torchtune.modules.moe.measurement import (
    MoEMeasurementCollector,
    aggregate_measurement_files,
    aggregate_rank_records,
    compare_capacity_value_results,
    evaluate_kernel_parity,
    compare_ep_scaling_summaries,
    compare_optimization_summaries,
    export_model_measurements,
    grouped_gemm_statistics,
    padded_bmm_statistics,
    mark_measurement_artifacts_complete,
    snapshot_model_measurements,
    summarize_ep_scaling_artifact,
    summarize_pipeline_timings,
    summarize_step_timings,
    token_statistics,
)

import json
import torch
from torch import nn
from torchtune.modules.moe import GroupedExperts, MoE
from torchtune.modules.moe.measurement_schema import (
    validate_evaluation_manifest,
    validate_manifest,
)
import torchtune.modules.moe._parallelism as ep_module


def test_index_select_with_zero_padding_matches_sentinel_indexing():
    values = torch.randn(5, 3, requires_grad=True)
    indices = torch.tensor([3, -1, 0, 4, -1], dtype=torch.int32)
    packed = moe_module._index_select_with_zero_padding(values, indices)
    expected = torch.cat((values, values.new_zeros(1, 3)))[
        indices.clamp_min(0)
    ]
    expected[indices < 0] = 0
    torch.testing.assert_close(packed, expected)

    packed.sum().backward()
    expected_grad = torch.zeros_like(values)
    expected_grad[3] = 1
    expected_grad[0] = 1
    expected_grad[4] = 1
    torch.testing.assert_close(values.grad, expected_grad)


def test_index_select_with_zero_padding_supports_wide_indices():
    values = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    indices = torch.tensor([4, -1, 1], dtype=torch.int32)
    wide_indices = indices[:, None].expand(-1, values.shape[1])
    packed = moe_module._index_select_with_zero_padding(values, wide_indices)
    expected = values.new_tensor([[12, 13, 14], [0, 0, 0], [3, 4, 5]])
    torch.testing.assert_close(packed, expected)


def test_grouped_moe_index_select_packing_matches_legacy(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "0")

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.tensor([2, 0, 1]),
                torch.tensor([3]),
            )

    class Experts(nn.Module):
        num_experts = 1

        def forward(self, inputs, counts):
            return inputs

    def permute_indices(counts, experts, ranks, alignment):
        return (
            torch.tensor([1, -1, 0, 2], dtype=torch.int32),
            torch.tensor([4]),
            torch.tensor([4], dtype=torch.int32),
        )

    monkeypatch.setattr(
        "torchtune.modules.moe.indices.generate_permute_indices", permute_indices
    )
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], requires_grad=True)
    moe = MoE(experts=Experts(), router=Router())
    moe.use_grouped_mm = True

    monkeypatch.setattr(moe_module, "_USE_INDEX_SELECT_PACKING", False)
    legacy = moe(inputs)
    legacy.sum().backward()
    legacy_grad = inputs.grad.detach().clone()

    inputs.grad = None
    monkeypatch.setattr(moe_module, "_USE_INDEX_SELECT_PACKING", True)
    packed = moe(inputs)
    packed.sum().backward()
    torch.testing.assert_close(packed, legacy)
    torch.testing.assert_close(inputs.grad, legacy_grad)


def test_final_scatter_index_add_matches_legacy_values_and_gradients(monkeypatch):
    indices = torch.tensor([2, 0, 2, 1], dtype=torch.long)
    base = torch.randn(3, 4)
    routed_base = torch.randn(4, 4)
    results = []
    for use_index_add in (False, True):
        monkeypatch.setattr(moe_module, "_USE_INDEX_ADD_FINAL_SCATTER", use_index_add)
        output = base.clone()
        routed = routed_base.clone().requires_grad_(True)
        result = moe_module._accumulate_routed_output(
            output, indices, routed, shared_expert=None
        )
        result.sum().backward()
        results.append((result.detach(), routed.grad.detach()))

    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])


def test_final_scatter_keeps_shared_expert_fallback(monkeypatch):
    indices = torch.tensor([1, 0], dtype=torch.long)
    output = torch.zeros(2, 3)
    routed = torch.ones(2, 3)
    monkeypatch.setattr(moe_module, "_USE_INDEX_ADD_FINAL_SCATTER", True)
    result = moe_module._accumulate_routed_output(
        output, indices, routed, shared_expert=nn.Identity()
    )
    assert result.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_ag_autograd_anchor_toggle_preserves_both_gradient_edges(monkeypatch):
    indices = torch.tensor([1, 4])

    def run(single_row, inplace):
        all_gathered = torch.randn(6, 4, requires_grad=True)
        routed_output = torch.randn(2, 4, requires_grad=True)
        partial_output = torch.zeros(6, 4)
        partial_output.scatter_add_(
            0, indices[:, None].expand_as(routed_output), routed_output
        )
        monkeypatch.setattr(ep_module, "_USE_INPLACE_AG_ANCHOR", inplace)
        monkeypatch.setattr(ep_module, "_USE_SINGLE_ROW_AG_ANCHOR", single_row)
        anchored = ep_module._add_ag_autograd_anchor(partial_output, all_gathered)
        anchored.sum().backward()
        return routed_output.grad.detach(), all_gathered.grad.detach()

    inplace_single = run(True, True)
    inplace_legacy = run(False, True)
    outplace_single = run(True, False)
    torch.testing.assert_close(inplace_single[0], inplace_legacy[0])
    torch.testing.assert_close(inplace_single[1], inplace_legacy[1])
    torch.testing.assert_close(inplace_single[0], outplace_single[0])
    torch.testing.assert_close(inplace_single[1], outplace_single[1])

    empty = torch.randn(0, 4, requires_grad=True)
    partial_output = torch.zeros(6, 4)
    monkeypatch.setattr(ep_module, "_USE_SINGLE_ROW_AG_ANCHOR", True)
    empty_result = ep_module._add_ag_autograd_anchor(partial_output, empty)
    assert empty_result.shape == partial_output.shape


def test_ag_autograd_anchor_uses_scalar_zero_edge(monkeypatch):
    all_gathered = torch.randn(6, 4, requires_grad=True)
    partial_output = torch.zeros(6, 4)
    monkeypatch.setattr(ep_module, "_USE_SINGLE_ROW_AG_ANCHOR", True)
    anchored = ep_module._add_ag_autograd_anchor(partial_output, all_gathered)

    assert anchored.shape == partial_output.shape
    assert anchored.requires_grad
    anchored.sum().backward()
    torch.testing.assert_close(all_gathered.grad, torch.zeros_like(all_gathered))


def test_ag_autograd_anchor_zero_cost_path_preserves_identity(monkeypatch):
    all_gathered = torch.randn(6, 4, requires_grad=True)
    partial_output = torch.randn(6, 4, requires_grad=True)
    monkeypatch.setattr(ep_module, "_USE_ZERO_COST_AG_ANCHOR", True)
    anchored = ep_module._add_ag_autograd_anchor(partial_output, all_gathered)

    torch.testing.assert_close(anchored, partial_output)
    anchored.sum().backward()
    torch.testing.assert_close(partial_output.grad, torch.ones_like(partial_output))
    torch.testing.assert_close(all_gathered.grad, torch.zeros_like(all_gathered))

    empty_gathered = torch.empty(0, 4, requires_grad=True)
    empty_partial = torch.zeros(6, 4)
    empty_anchored = ep_module._add_ag_autograd_anchor(empty_partial, empty_gathered)
    assert empty_anchored.shape == empty_partial.shape


def test_token_statistics_preserves_zero_token_experts_and_ep_ownership():
    stats = token_statistics([4, 0, 2, 0, 1, 1, 0, 0], ep_degree=2)
    assert stats["zero_token_experts"] == 4
    assert stats["local_experts"] == 4
    assert stats["local_token_count"] == 4
    assert stats["max_tokens_per_expert"] == 4


@pytest.mark.parametrize(
    ("total_tokens", "expert_count", "imbalance_factor"),
    [(0, 4, 1.0), (17, 4, 1.0), (101, 8, 3.0)],
)
def test_synthetic_expert_token_counts_are_deterministic_and_conservative(
    total_tokens, expert_count, imbalance_factor
):
    from torchtune.modules.moe.measurement import synthetic_expert_token_counts

    counts = synthetic_expert_token_counts(
        total_tokens, expert_count, imbalance_factor=imbalance_factor
    )
    assert len(counts) == expert_count
    assert sum(counts) == total_tokens
    assert all(count >= 0 for count in counts)
    assert counts == synthetic_expert_token_counts(
        total_tokens, expert_count, imbalance_factor=imbalance_factor
    )


def test_synthetic_expert_token_counts_rejects_invalid_parameters():
    from torchtune.modules.moe.measurement import synthetic_expert_token_counts

    with pytest.raises(ValueError, match="expert_count"):
        synthetic_expert_token_counts(4, 0)
    with pytest.raises(ValueError, match="imbalance_factor"):
        synthetic_expert_token_counts(4, 2, imbalance_factor=0.5)


def test_grouped_gemm_statistics_reports_zero_shapes():
    stats = grouped_gemm_statistics([3, 0, 2], model_dim=8, hidden_dim=16)
    assert stats["active_expert_gemm_count"] == 2
    assert stats["up_projection_shapes"][1] == [0, 8, 16]


def test_grouped_gemm_statistics_reports_alignment_padding():
    stats = grouped_gemm_statistics(
        [4, 16, 0],
        model_dim=8,
        hidden_dim=16,
        routed_counts=[3, 12, 0],
        alignment=16,
    )
    assert stats["routed_tokens"] == 15
    assert stats["compute_tokens"] == 20
    assert stats["padding_tokens"] == 5
    assert stats["alignment"] == 16
    assert stats["padding_fraction"] == 0.25


def test_padded_bmm_statistics_reports_dense_work():
    stats = padded_bmm_statistics(
        [3, 12, 0], model_dim=8, hidden_dim=16
    )
    assert stats["max_count"] == 12
    assert stats["routed_tokens"] == 15
    assert stats["dense_compute_tokens"] == 36
    assert stats["padding_tokens"] == 21
    assert stats["dense_to_routed_ratio"] == 36 / 15


def test_qwen3_padded_bmm_experts_emit_dense_work_measurement(monkeypatch):
    import torchtune.models.qwen3_moe._experts as experts_module

    monkeypatch.setattr(experts_module, "_GROUPED_EXPERTS", False)
    monkeypatch.setattr(experts_module, "_SEQUENTIAL_EXPERTS", False)
    experts = experts_module.GroupedExpertsHF(
        dim=4, hidden_dim=8, num_experts=3
    )
    experts.reset_parameters()
    collector = MoEMeasurementCollector(enabled=True)
    experts._moe_measurement = collector
    counts = torch.tensor([3, 12, 0], dtype=torch.long)
    output = experts(torch.randn(15, 4), counts)
    assert output.shape == (15, 4)
    assert collector.record.grouped_gemm[-1]["stage"] == "padded_bmm"
    assert collector.record.grouped_gemm[-1]["dense_compute_tokens"] == 36
    assert collector.record.timing_counts["padded_bmm"] == 1


def test_qwen3_sequential_experts_record_compute_timing(monkeypatch):
    import torchtune.models.qwen3_moe._experts as experts_module

    monkeypatch.setattr(experts_module, "_GROUPED_EXPERTS", False)
    monkeypatch.setattr(experts_module, "_SEQUENTIAL_EXPERTS", True)
    experts = experts_module.GroupedExpertsHF(dim=4, hidden_dim=8, num_experts=3)
    experts.reset_parameters()
    collector = MoEMeasurementCollector(enabled=True)
    experts._moe_measurement = collector
    counts = torch.tensor([3, 0, 2], dtype=torch.long)
    inputs = torch.randn(5, 4, requires_grad=True)
    output = experts(inputs, counts)
    output.square().mean().backward()

    assert output.shape == (5, 4)
    assert collector.record.timing_counts["sequential_expert_compute"] == 1
    assert collector.record.timing_counts["sequential_expert_gate"] == 2
    assert collector.record.timing_counts["sequential_expert_up"] == 2
    assert collector.record.timing_counts["sequential_expert_down"] == 2
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("recompute_mode", ["0", "up_only"])
def test_qwen3_grouped_experts_record_projection_timings(monkeypatch, recompute_mode):
    import torchtune.models.qwen3_moe._experts as experts_module

    def fake_grouped_mm(inputs, weights, *, offs):
        outputs = []
        start = 0
        for expert, end in enumerate(offs.tolist()):
            outputs.append(inputs[start:end] @ weights[expert])
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm, raising=False)
    monkeypatch.setattr(experts_module, "_GROUPED_EXPERTS", True)
    monkeypatch.setattr(experts_module, "_GROUPED_RECOMPUTE_PREACT", recompute_mode)
    experts = experts_module.GroupedExpertsHF(
        dim=4, hidden_dim=8, num_experts=2
    ).to(torch.bfloat16)
    collector = MoEMeasurementCollector(enabled=True)
    experts._moe_measurement = collector

    output = experts(
        torch.randn(3, 4, dtype=torch.bfloat16),
        torch.tensor([2, 1], dtype=torch.int32),
    )

    assert output.shape == (3, 4)
    assert all(
        collector.record.timing_counts[name] == 1
        for name in (
            "grouped_gemm_gate",
            "grouped_gemm_up",
            "grouped_gemm_down",
        )
    )


def test_collector_records_routed_and_compute_token_counts():
    collector = MoEMeasurementCollector(enabled=True)
    collector.record_gemm(
        [16, 16],
        model_dim=8,
        hidden_dim=16,
        routed_counts=[15, 9],
        alignment=16,
    )
    record = collector.record.grouped_gemm[0]
    assert record["routed_tokens"] == 24
    assert record["compute_tokens"] == 32
    assert record["padding_tokens"] == 8


def test_collector_is_noop_when_disabled():
    collector = MoEMeasurementCollector(enabled=False)
    assert collector._record is None
    with collector.time("router"):
        pass
    collector.record_tokens([1, 0])
    assert collector._record is None
    assert collector.record.as_dict() == {
        "timings_s": {},
        "timing_counts": {},
        "routed_tokens": [],
        "grouped_gemm": [],
        "collectives": [],
        "memory": [],
    }


def test_collector_record_is_lazy_until_explicitly_accessed():
    collector = MoEMeasurementCollector(enabled=True)
    assert collector._record is None
    record = collector.record
    assert collector._record is record


def test_ep_autograd_collective_boundaries_skip_context_when_disabled(monkeypatch):
    import torchtune.modules.moe._parallelism as ep_module

    calls = []

    class Group:
        pass

    group = Group()
    monkeypatch.setattr(ep_module.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(ep_module, "_ep_all_gather", lambda *args, **kwargs: calls.append("ag"))
    monkeypatch.setattr(ep_module, "_ep_reduce_scatter", lambda *args, **kwargs: args[0])

    input_tensor = torch.randn(2, 3, requires_grad=True)
    gathered = ep_module._AllGatherRS.apply(input_tensor, group, None)
    assert calls == ["ag"]
    gathered.sum().backward()

    calls.clear()
    output = ep_module._ReduceScatterAG.apply(input_tensor, group, None)
    torch.testing.assert_close(output, input_tensor)
    output.sum().backward()
    assert calls == ["ag"]


def test_collector_records_router_timing():
    collector = MoEMeasurementCollector(enabled=True)
    with collector.time("router"):
        pass
    assert collector.record.timing_counts == {"router": 1}
    assert collector.record.timings_s["router"] >= 0


def test_grouped_experts_records_projection_timings(monkeypatch):
    def fake_grouped_mm(inputs, weights, *, offs):
        outputs = []
        start = 0
        for expert, end in enumerate(offs.tolist()):
            outputs.append(inputs[start:end] @ weights[expert])
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm, raising=False)
    experts = GroupedExperts(dim=4, hidden_dim=8, num_experts=2).to(torch.bfloat16)
    experts.use_grouped_mm = True
    collector = MoEMeasurementCollector(enabled=True)
    experts._moe_measurement = collector

    output = experts(
        torch.randn(3, 4, dtype=torch.bfloat16),
        torch.tensor([2, 1], dtype=torch.int32),
    )

    assert output.shape == (3, 4)
    assert set(("grouped_gemm_gate", "grouped_gemm_up", "grouped_gemm_down")) <= set(
        collector.record.timing_counts
    )
    assert all(
        collector.record.timing_counts[name] == 1
        for name in ("grouped_gemm_gate", "grouped_gemm_up", "grouped_gemm_down")
    )


def test_grouped_experts_skips_projection_timing_when_disabled(monkeypatch):
    def fake_grouped_mm(inputs, weights, *, offs):
        outputs = []
        start = 0
        for expert, end in enumerate(offs.tolist()):
            outputs.append(inputs[start:end] @ weights[expert])
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm, raising=False)
    experts = GroupedExperts(dim=4, hidden_dim=8, num_experts=2).to(torch.bfloat16)
    experts.use_grouped_mm = True
    collector = MoEMeasurementCollector(enabled=False)
    experts._moe_measurement = collector

    experts(
        torch.randn(3, 4, dtype=torch.bfloat16),
        torch.tensor([2, 1], dtype=torch.int32),
    )

    assert collector.record.timings_s == {}
    assert collector.record.timing_counts == {}


def test_moe_forward_records_router_timing(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "1")

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.arange(inputs.shape[0]),
                torch.ones(2, dtype=torch.long),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 4, 8))

        def forward(self, inputs, counts):
            return inputs

    moe = MoE(experts=Experts(), router=Router())
    moe(torch.ones(1, 2, 4))
    assert moe.measurement.record.timing_counts["router"] == 1
    assert moe.measurement.record.timings_s["router"] >= 0
    assert moe.measurement.record.timing_counts["final_scatter"] == 1
    assert moe.measurement.record.timings_s["final_scatter"] >= 0


def test_moe_forward_skips_timing_boundaries_when_disabled(monkeypatch):
    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.arange(inputs.shape[0]),
                torch.ones(2, dtype=torch.long),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 4, 8))

        def forward(self, inputs, counts):
            return inputs

    moe = MoE(experts=Experts(), router=Router())

    class TimingSentinel:
        def __call__(self, name):
            raise AssertionError(f"disabled timing boundary entered: {name}")

    monkeypatch.setattr(moe.measurement, "time", TimingSentinel())
    output = moe(torch.ones(1, 2, 4))
    assert output.shape == (1, 2, 4)


def test_moe_measurement_reports_hf_expert_hidden_dimension(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "1")

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.arange(inputs.shape[0]),
                torch.tensor([1, 1]),
            )

    class HFExperts(nn.Module):
        dim = 4
        num_experts = 2

        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 8, 4))

        def forward(self, inputs, counts):
            return inputs

    moe = MoE(experts=HFExperts(), router=Router())
    moe(torch.ones(1, 2, 4))
    assert moe.measurement.record.grouped_gemm[0]["model_dim"] == 4
    assert moe.measurement.record.grouped_gemm[0]["hidden_dim"] == 8


def test_moe_forward_keeps_routing_indices_one_dimensional(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "0")
    observed = {}

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.tensor([1, 0, 1]),
                torch.tensor([1, 2]),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 4, 8))

        def forward(self, inputs, counts):
            observed["shape"] = tuple(inputs.shape)
            return inputs

    moe = MoE(experts=Experts(), router=Router())
    output = moe(torch.ones(1, 3, 4))
    assert observed["shape"] == (3, 4)
    assert output.shape == (1, 3, 4)


def test_moe_forward_preserves_weighted_duplicate_routes(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "0")

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_tensor([0.5, 2.0, 3.0]),
                torch.tensor([1, 0, 1]),
                torch.tensor([1, 2]),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 2, 2))

        def forward(self, inputs, counts):
            return inputs

    moe = MoE(experts=Experts(), router=Router())
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    output = moe(inputs)
    expected = torch.tensor([[[2.0, 4.0], [10.5, 14.0], [0.0, 0.0]]])
    torch.testing.assert_close(output, expected)


def test_moe_forward_wide_index_compatibility_mode_is_equivalent(monkeypatch):
    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_tensor([0.5, 2.0, 3.0]),
                torch.tensor([1, 0, 1]),
                torch.tensor([1, 2]),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 2, 2))

        def forward(self, inputs, counts):
            return inputs

    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    moe = MoE(experts=Experts(), router=Router())
    monkeypatch.setattr(moe_module, "_USE_WIDE_ROUTING_INDICES", False)
    compact = moe(inputs)
    monkeypatch.setattr(moe_module, "_USE_WIDE_ROUTING_INDICES", True)
    wide = moe(inputs)
    torch.testing.assert_close(compact, wide)


def test_moe_forward_inplace_final_scatter_preserves_gradients(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "0")

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.tensor([0, 1]),
                torch.tensor([1, 1]),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 2, 2))

        def forward(self, inputs, counts):
            return inputs.square()

    inputs = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], requires_grad=True)
    output = MoE(experts=Experts(), router=Router())(inputs)
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, 2 * inputs.detach())


def test_moe_final_scatter_toggle_matches_reference(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "0")

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_tensor([0.5, 2.0, 3.0]),
                torch.tensor([1, 0, 1]),
                torch.tensor([1, 2]),
            )

    class Experts(nn.Module):
        def forward(self, inputs, counts):
            return inputs.square()

    def run(inplace):
        monkeypatch.setattr(moe_module, "_USE_INPLACE_FINAL_SCATTER", inplace)
        inputs = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], requires_grad=True
        )
        output = MoE(experts=Experts(), router=Router())(inputs)
        output.sum().backward()
        return output.detach(), inputs.grad.detach()

    output, gradient = run(False)
    inplace_output, inplace_gradient = run(True)
    torch.testing.assert_close(inplace_output, output)
    torch.testing.assert_close(inplace_gradient, gradient)


def test_moe_route_weighting_toggle_matches_reference(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_MEASURE", "0")

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = nn.Parameter(torch.tensor([0.5, 2.0]))

        def forward(self, inputs):
            return self.scores, torch.tensor([0, 1]), torch.tensor([1, 1])

    class Experts(nn.Module):
        def forward(self, inputs, counts):
            return inputs.square()

    def run(inplace):
        monkeypatch.setattr(moe_module, "_USE_INPLACE_ROUTE_WEIGHTING", inplace)
        router = Router()
        inputs = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], requires_grad=True)
        output = MoE(experts=Experts(), router=router)(inputs)
        output.sum().backward()
        return output.detach(), inputs.grad.detach(), router.scores.grad.detach()

    output, gradient, router_gradient = run(False)
    inplace_output, inplace_gradient, inplace_router_gradient = run(True)
    torch.testing.assert_close(inplace_output, output)
    torch.testing.assert_close(inplace_gradient, gradient)
    torch.testing.assert_close(inplace_router_gradient, router_gradient)


def test_moe_forward_inplace_route_weighting_preserves_router_gradients():
    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = nn.Parameter(torch.tensor([0.5, 2.0]))

        def forward(self, inputs):
            return (
                self.scores,
                torch.tensor([0, 1]),
                torch.tensor([1, 1]),
            )

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 2, 2))

        def forward(self, inputs, counts):
            return inputs

    router = Router()
    inputs = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], requires_grad=True)
    output = MoE(experts=Experts(), router=router)(inputs)
    output.square().sum().backward()
    assert router.scores.grad is not None
    assert torch.isfinite(router.scores.grad).all()


def test_grouped_eager_path_does_not_compute_compile_shape_scalar(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        "torchtune.modules.moe.indices.generate_permute_indices",
        lambda counts, experts, ranks, alignment: (
            torch.arange(int(counts.sum().item())),
            counts,
            torch.cumsum(counts, dim=0),
        ),
    )

    class Router(nn.Module):
        def forward(self, inputs):
            return (
                inputs.new_ones((inputs.shape[0],)),
                torch.arange(inputs.shape[0]),
                torch.tensor([1, 1]),
            )

    class Experts(nn.Module):
        num_experts = 2

        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.empty(2, 2, 2))

        def forward(self, inputs, counts):
            return inputs

    moe = MoE(experts=Experts(), router=Router())
    moe.use_grouped_mm = True
    output = moe(torch.ones(1, 2, 2))
    assert output.shape == (1, 2, 2)


def test_collector_records_collective_identity():
    collector = MoEMeasurementCollector(enabled=True)
    with collector.collective("dispatch_alltoall", scope="ep", backend="gloo"):
        pass
    assert len(collector.record.collectives) == 1
    event = collector.record.collectives[0]
    assert event["name"] == "dispatch_alltoall"
    assert event["scope"] == "ep"
    assert event["backend"] == "gloo"
    assert event["locality"] == "unknown"
    assert event["duration_s"] >= 0


def test_fused_alltoall_records_one_dispatch_collective_per_direction(monkeypatch):
    collector = MoEMeasurementCollector(enabled=True)
    monkeypatch.setattr(ep_module.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(
        ep_module,
        "_raw_all_to_all_single",
        lambda send, output_splits, input_splits, group: send.clone(),
    )
    routed_input = torch.randn(3, 2, requires_grad=True)
    dispatched = ep_module._fused_all_to_all_routing(
        routed_input,
        torch.tensor([2, 0, 1]),
        torch.tensor([1, 2, 0]),
        [3],
        [3],
        object(),
        collector,
    )
    output = ep_module._fused_all_to_all_combine(
        dispatched,
        torch.tensor([2, 0, 1]),
        torch.tensor([1, 2, 0]),
        [3],
        [3],
        object(),
        collector,
        3,
    )
    output.sum().backward()

    events = collector.record.collectives
    assert [event["name"] for event in events] == [
        "dispatch_alltoall",
        "combine_alltoall",
        "combine_backward_alltoall",
        "dispatch_backward_alltoall",
    ]
    assert all(event["backend"] == "gloo" for event in events)


def test_fused_alltoall_handles_empty_routed_buffers(monkeypatch):
    monkeypatch.setattr(ep_module.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(
        ep_module,
        "_raw_all_to_all_single",
        lambda send, output_splits, input_splits, group: send.clone(),
    )
    collector = MoEMeasurementCollector(enabled=True)
    routed_input = torch.empty(0, 2, requires_grad=True)
    empty = torch.empty(0, dtype=torch.long)
    dispatched = ep_module._fused_all_to_all_routing(
        routed_input, empty, empty, [0], [0], object(), collector
    )
    output = ep_module._fused_all_to_all_combine(
        dispatched, empty, empty, [0], [0], object(), collector, 0
    )
    assert output.shape == (0, 2)
    output.sum().backward()
    assert routed_input.grad is not None
    assert routed_input.grad.shape == routed_input.shape


def test_collector_records_configured_collective_locality(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_COLLECTIVE_LOCALITY", "cross_node")
    collector = MoEMeasurementCollector(enabled=True)
    with collector.collective("dispatch", scope="ep", backend="xccl"):
        pass
    event = collector.record.collectives[0]
    assert event["scope"] == "ep"
    assert event["locality"] == "cross_node"


def test_collector_rejects_invalid_collective_locality(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_MOE_COLLECTIVE_LOCALITY", "invalid")
    collector = MoEMeasurementCollector(enabled=True)
    with pytest.raises(ValueError, match="COLLECTIVE_LOCALITY"):
        collector.collective("dispatch", scope="ep", backend="xccl")


def test_aggregate_rank_records_rejects_invalid_collective_locality():
    with pytest.raises(ValueError, match="invalid collective locality"):
        aggregate_rank_records(
            [
                {
                    "collectives": [
                        {
                            "name": "dispatch",
                            "scope": "ep",
                            "backend": "xccl",
                            "locality": "remote_unknown",
                            "duration_s": 1.0,
                        }
                    ]
                }
            ]
        )


def test_aggregate_rank_records_rejects_malformed_collectives():
    with pytest.raises(ValueError, match="collective records must be lists"):
        aggregate_rank_records([{"collectives": {}}])
    with pytest.raises(ValueError, match="collective events must be mappings"):
        aggregate_rank_records([{"collectives": ["bad"]}])


def test_disabled_timing_does_not_synchronize(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "torchtune.modules.moe.measurement.synchronize_measurement_device",
        lambda: calls.append(True),
    )
    collector = MoEMeasurementCollector(enabled=False)
    with collector.time("disabled"):
        pass
    with collector.collective("disabled", scope="ep", backend="gloo"):
        pass
    assert calls == []


def test_disabled_timing_reuses_noop_context(monkeypatch):
    collector = MoEMeasurementCollector(enabled=False)
    first = collector.time("first")
    second = collector.time("second")
    first_collective = collector.collective("first", scope="ep", backend="gloo")
    second_collective = collector.collective("second", scope="ep", backend="gloo")
    assert first is second
    assert first is first_collective
    assert first_collective is second_collective

    monkeypatch.setenv("TORCHTUNE_MOE_COLLECTIVE_LOCALITY", "invalid")
    with collector.collective("disabled", scope="ep", backend="gloo"):
        pass


def test_collector_records_fallback_collective_names():
    collector = MoEMeasurementCollector(enabled=True)
    for name in (
        "allgather_forward",
        "reduce_scatter_forward",
        "reduce_scatter_backward",
        "allgather_backward",
    ):
        with collector.collective(name, scope="ep", backend="gloo"):
            pass
    assert [event["name"] for event in collector.record.collectives] == [
        "allgather_forward",
        "reduce_scatter_forward",
        "reduce_scatter_backward",
        "allgather_backward",
    ]


def test_export_model_measurements_writes_rank_local_json(tmp_path):
    class Measured(nn.Module):
        def __init__(self):
            super().__init__()
            self.measurement = MoEMeasurementCollector(enabled=True)

    model = nn.Module()
    model.moe = Measured()
    model.moe.measurement.record.add_timing("expert_forward", 1.25)
    destination = tmp_path / "rank3.json"
    export_model_measurements(model, destination, metadata={"rank": 3})
    payload = json.loads(destination.read_text())
    assert payload["metadata"] == {"rank": 3}
    assert payload["records"]["moe"]["timings_s"] == {"expert_forward": 1.25}


def test_snapshot_model_measurements_skips_disabled_collectors(monkeypatch):
    class Measured(nn.Module):
        def __init__(self, enabled):
            super().__init__()
            self.measurement = MoEMeasurementCollector(enabled=enabled)

    model = nn.Module()
    model.enabled = Measured(True)
    model.disabled = Measured(False)
    calls = []
    monkeypatch.setattr(
        MoEMeasurementCollector,
        "snapshot_memory",
        lambda self, phase, device, **kwargs: calls.append(phase),
    )
    snapshot_model_measurements(model, "forward", "cpu", step=7, microbatch=2)
    assert calls == ["forward"]


def test_snapshot_model_measurements_records_steady_state_phase():
    class Measured(nn.Module):
        def __init__(self):
            super().__init__()
            self.measurement = MoEMeasurementCollector(enabled=True)

    model = nn.Module()
    model.moe = Measured()
    calls = []
    model.moe.measurement.snapshot_memory = lambda phase, device, **kwargs: calls.append(
        (phase, device, kwargs)
    )
    snapshot_model_measurements(model, "steady_state", "cpu", step=3)
    assert calls == [("steady_state", "cpu", {"step": 3, "microbatch": None})]


def test_export_model_measurements_skips_disabled_model(tmp_path):
    class Disabled(nn.Module):
        def __init__(self):
            super().__init__()
            self.measurement = MoEMeasurementCollector(enabled=False)

    destination = tmp_path / "missing.json"
    export_model_measurements(Disabled(), destination)
    assert not destination.exists()


def test_mark_measurement_artifacts_complete_seals_artifact(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {"measurement_completion": "pending"},
                "records": {},
            }
        )
    )
    mark_measurement_artifacts_complete([destination])
    payload = json.loads(destination.read_text())
    assert payload["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_canonical_provenance(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "model": "unknown",
                    "checkpoint": "/models/test",
                    "source_revision": "abc123",
                    "uncommitted_change_state": "dirty",
                },
                "records": {},
            }
        )
    )
    with pytest.raises(ValueError, match="model must be non-placeholder"):
        mark_measurement_artifacts_complete(
            [destination], require_provenance=True
        )
    assert "measurement_completion" not in json.loads(destination.read_text())[
        "metadata"
    ]


def test_mark_measurement_artifacts_complete_accepts_canonical_provenance(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "model": "Qwen3-30B-A3B",
                    "checkpoint": "/models/qwen3",
                    "source_revision": "abc123",
                    "uncommitted_change_state": "dirty",
                },
                "records": {},
            }
        )
    )
    mark_measurement_artifacts_complete(
        [destination], require_provenance=True
    )
    assert json.loads(destination.read_text())["metadata"][
        "measurement_completion"
    ] == "passed"


def test_mark_measurement_artifacts_complete_rejects_empty_or_duplicate_set(
    tmp_path,
):
    destination = tmp_path / "rank0.json"
    destination.write_text(json.dumps({"metadata": {}, "records": {}}))
    with pytest.raises(ValueError, match="empty measurement artifact set"):
        mark_measurement_artifacts_complete([])
    with pytest.raises(ValueError, match="duplicate paths"):
        mark_measurement_artifacts_complete([destination, destination])


def test_mark_measurement_artifacts_complete_requires_memory_phases(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {},
                "records": {
                    "moe": {
                        "memory": [{"phase": "forward"}],
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="required memory phases.*steady_state"):
        mark_measurement_artifacts_complete(
            [destination], required_memory_phases=("forward", "steady_state")
        )
    assert "measurement_completion" not in json.loads(destination.read_text())[
        "metadata"
    ]


@pytest.mark.parametrize(
    "override_name",
    [
        "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE",
        "TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS",
        "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA",
    ],
)
def test_mark_measurement_artifacts_complete_rejects_invalid_binary_override(
    tmp_path, override_name
):
    destination = tmp_path / "invalid.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "environment_overrides": {override_name: "2"},
                },
                "records": {},
            }
        )
    )
    with pytest.raises(ValueError, match=override_name):
        mark_measurement_artifacts_complete([destination])
    assert "measurement_completion" not in json.loads(destination.read_text())[
        "metadata"
    ]


def test_mark_measurement_artifacts_complete_applies_final_gate_metadata(tmp_path):
    import json

    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "semantic_completion": "pending",
                    "measurement_completion": "pending",
                },
                "records": {},
            }
        )
    )

    mark_measurement_artifacts_complete(
        [destination], metadata_updates={"semantic_completion": "passed"}
    )

    payload = json.loads(destination.read_text())
    assert payload["metadata"]["semantic_completion"] == "passed"
    assert payload["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_measurement_steps(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps({"metadata": {"global_step": 3}, "records": {}})
    )

    with pytest.raises(ValueError, match="expected at least 4"):
        mark_measurement_artifacts_complete([destination], minimum_global_step=4)

    mark_measurement_artifacts_complete([destination], minimum_global_step=3)
    payload = json.loads(destination.read_text())
    assert payload["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_declared_window(tmp_path):
    destination = tmp_path / "rank0.json"
    base_metadata = {
        "global_step": 3,
        "measurement_window": {
            "warmup_steps": 1,
            "measurement_steps": 2,
            "steady_state_steps": 1,
        },
    }
    payload = {
        "metadata": base_metadata,
        "records": {},
        "step_timings": [
            {
                "step": step,
                "total_step_s": 1.0,
                "timings_s": {"attention": 0.1, "non_expert": 0.1},
            }
            for step in range(1, 4)
        ],
    }
    destination.write_text(json.dumps(payload))
    mark_measurement_artifacts_complete(
        [destination],
        require_step_timing=True,
        require_declared_measurement_window=True,
    )
    assert json.loads(destination.read_text())["metadata"]["measurement_completion"] == "passed"

    payload["step_timings"] = payload["step_timings"][:2]
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fewer post-warmup"):
        mark_measurement_artifacts_complete(
            [destination],
            require_step_timing=True,
            require_declared_measurement_window=True,
        )


def test_mark_measurement_artifacts_complete_rejects_inconsistent_windows(tmp_path):
    paths = [tmp_path / "rank0.json", tmp_path / "rank1.json"]
    for rank, path in enumerate(paths):
        path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "global_step": 2,
                        "measurement_window": {
                            "warmup_steps": rank,
                            "measurement_steps": 1,
                            "steady_state_steps": 0,
                        },
                    },
                    "records": {},
                    "step_timings": [
                        {
                            "step": 1,
                            "total_step_s": 1.0,
                            "timings_s": {"attention": 0.1, "non_expert": 0.1},
                        }
                    ],
                }
            )
        )
    with pytest.raises(ValueError, match="inconsistent measurement_window"):
        mark_measurement_artifacts_complete(
            paths,
            require_step_timing=True,
            require_declared_measurement_window=True,
        )


def test_mark_measurement_artifacts_complete_strict_gate_rejects_empty_records(
    tmp_path,
):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "device_health": "green",
                    "gate_status": "passed",
                    "semantic_completion": "passed",
                    "global_step": 3,
                },
                "records": {},
            }
        )
    )
    with pytest.raises(ValueError, match="no enabled MoE records"):
        mark_measurement_artifacts_complete(
            [destination],
            minimum_global_step=3,
            require_measurement_records=True,
            require_passed_gates=True,
        )


def test_mark_measurement_artifacts_complete_strict_gate_passes_valid_record(
    tmp_path,
):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "device_health": "green",
                    "gate_status": "passed",
                    "semantic_completion": "pending",
                    "global_step": 3,
                },
                "records": {"model.layers.0.moe": {"timings_s": {}}},
            }
        )
    )
    mark_measurement_artifacts_complete(
        [destination],
        metadata_updates={"semantic_completion": "passed"},
        minimum_global_step=3,
        require_measurement_records=True,
        require_passed_gates=True,
    )
    payload = json.loads(destination.read_text())
    assert payload["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_execution_path(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "device_health": "green",
                    "gate_status": "passed",
                    "semantic_completion": "passed",
                    "global_step": 3,
                    "environment_overrides": {},
                },
                "records": {"model.layers.0.moe": {"timings_s": {}}},
            }
        )
    )
    with pytest.raises(ValueError, match="requires a valid expert_execution_path"):
        mark_measurement_artifacts_complete(
            [destination],
            minimum_global_step=3,
            require_measurement_records=True,
            require_passed_gates=True,
            require_execution_path=True,
        )


def test_mark_measurement_artifacts_complete_requires_path_specific_timing(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "global_step": 3,
                    "expert_execution_path": "sequential",
                    "environment_overrides": {
                        "TORCHTUNE_MOE_GROUPED_EXPERTS": "0",
                        "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS": "1",
                    },
                },
                "records": {
                    "model.layers.0.moe": {
                        "timings_s": {"router": 0.1},
                        "collectives": [],
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="sequential_expert_compute"):
        mark_measurement_artifacts_complete(
            [destination],
            minimum_global_step=3,
            require_execution_path=True,
            expected_execution_path="sequential",
            require_moe_metrics=True,
            required_moe_timings=("router",),
            required_collectives=(),
        )


def test_mark_measurement_artifacts_complete_accepts_sequential_defaults(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "global_step": 3,
                    "expert_execution_path": "sequential",
                    "environment_overrides": {
                        "TORCHTUNE_MOE_GROUPED_EXPERTS": "0",
                        "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS": "1",
                    },
                },
                "records": {
                    "model.layers.0.moe": {
                        "timings_s": {
                            "router": 0.1,
                            "expert_forward": 0.2,
                            "final_scatter": 0.1,
                            "sequential_expert_compute": 0.2,
                            "sequential_expert_gate": 0.05,
                            "sequential_expert_up": 0.05,
                            "sequential_expert_down": 0.1,
                        },
                        "collectives": [],
                    }
                },
            }
        )
    )
    mark_measurement_artifacts_complete(
        [destination],
        minimum_global_step=3,
        require_execution_path=True,
        require_moe_metrics=True,
        required_collectives=(),
    )
    assert json.loads(destination.read_text())["metadata"]["measurement_completion"] == (
        "passed"
    )


def test_mark_measurement_artifacts_complete_accepts_alltoall_collectives(tmp_path):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "global_step": 3,
                    "expert_execution_path": "grouped_mm",
                    "environment_overrides": {
                        "TORCHTUNE_MOE_GROUPED_EXPERTS": "1",
                        "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS": "0",
                    },
                },
                "records": {
                    "model.layers.0.moe": {
                        "timings_s": {
                            "router": 0.1,
                            "expert_forward": 0.2,
                            "final_scatter": 0.1,
                            "routing_metadata_materialization": 0.01,
                            "routing_metadata_permutation": 0.02,
                            "grouped_gemm_gate": 0.05,
                            "grouped_gemm_up": 0.05,
                            "grouped_gemm_down": 0.05,
                        },
                        "collectives": [
                            {"name": "routing_metadata_allgather"},
                            {"name": "dispatch_alltoall"},
                            {"name": "combine_alltoall"},
                            {"name": "dispatch_backward_alltoall"},
                            {"name": "combine_backward_alltoall"},
                        ],
                    }
                },
            }
        )
    )
    mark_measurement_artifacts_complete(
        [destination],
        minimum_global_step=3,
        require_execution_path=True,
        expected_execution_path="grouped_mm",
        require_moe_metrics=True,
        required_moe_timings=(
            "router",
            "expert_forward",
            "final_scatter",
            "routing_metadata_materialization",
            "routing_metadata_permutation",
        ),
        required_collectives=(
            "routing_metadata_allgather",
            "dispatch_alltoall",
            "combine_alltoall",
            "dispatch_backward_alltoall",
            "combine_backward_alltoall",
        ),
    )
    payload = json.loads(destination.read_text())
    assert payload["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_step_phases(tmp_path):
    destination = tmp_path / "rank0.json"
    base = {
        "metadata": {"global_step": 3},
        "records": {"model.layers.0.moe": {"timings_s": {}}},
    }
    destination.write_text(
        json.dumps(
            {
                **base,
                "step_timings": [
                    {"step": 3, "total_step_s": 1.0, "timings_s": {"attention": 0.2}}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="non_expert"):
        mark_measurement_artifacts_complete(
            [destination], require_step_timing=True
        )

    destination.write_text(
        json.dumps(
            {
                **base,
                "step_timings": [
                    {
                        "step": 3,
                        "total_step_s": 1.0,
                        "timings_s": {"attention": 0.2, "non_expert": 0.1},
                    }
                ],
            }
        )
    )
    mark_measurement_artifacts_complete(
        [destination], require_step_timing=True
    )
    assert json.loads(destination.read_text())["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_throughput_metrics(tmp_path):
    destination = tmp_path / "rank0.json"
    payload = {
        "metadata": {"global_step": 3},
        "records": {"model.layers.0.moe": {"timings_s": {}}},
        "step_timings": [
            {
                "step": 3,
                "total_step_s": 1.0,
                "timings_s": {"attention": 0.2, "non_expert": 0.1},
            }
        ],
    }
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="local_tokens"):
        mark_measurement_artifacts_complete(
            [destination],
            require_step_timing=True,
            require_throughput_metrics=True,
        )

    payload["step_timings"][0].update(
        {
            "local_tokens": 100,
            "global_tokens": 800,
            "tokens_per_second_per_gpu": 100.0,
            "aggregate_tokens_per_second": 800.0,
        }
    )
    destination.write_text(json.dumps(payload))
    mark_measurement_artifacts_complete(
        [destination],
        require_step_timing=True,
        require_throughput_metrics=True,
    )
    assert json.loads(destination.read_text())["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_rejects_invalid_throughput_metrics(
    tmp_path,
):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {},
                "records": {},
                "step_timings": [
                    {
                        "step": 1,
                        "total_step_s": 1.0,
                        "timings_s": {"attention": 0.2, "non_expert": 0.1},
                        "local_tokens": 1,
                        "global_tokens": 2,
                        "tokens_per_second_per_gpu": float("nan"),
                        "aggregate_tokens_per_second": 2,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="tokens_per_second_per_gpu"):
        mark_measurement_artifacts_complete(
            [destination],
            require_step_timing=True,
            require_throughput_metrics=True,
        )


def test_mark_measurement_artifacts_complete_requires_moe_metrics(tmp_path):
    destination = tmp_path / "rank0.json"
    required_timings = {
        "router": 0.1,
        "expert_forward": 0.2,
        "final_scatter": 0.1,
        "grouped_gemm_gate": 0.1,
        "grouped_gemm_up": 0.1,
        "grouped_gemm_down": 0.1,
    }
    required_collectives = [
        "dispatch_alltoall",
        "combine_alltoall",
        "dispatch_backward_alltoall",
        "combine_backward_alltoall",
    ]
    payload = {
        "metadata": {"global_step": 3},
        "records": {
            "model.layers.0.moe": {
                "timings_s": {"router": 0.1},
                "collectives": [],
            }
        },
    }
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expert_forward"):
        mark_measurement_artifacts_complete(
            [destination], require_moe_metrics=True
        )

    payload["records"]["model.layers.0.moe"]["timings_s"] = required_timings
    payload["records"]["model.layers.0.moe"]["collectives"] = [
        {"name": name} for name in required_collectives
    ]
    destination.write_text(json.dumps(payload))
    mark_measurement_artifacts_complete(
        [destination], require_moe_metrics=True
    )
    assert json.loads(destination.read_text())["metadata"]["measurement_completion"] == "passed"


def test_mark_measurement_artifacts_complete_requires_moe_metrics_per_rank(tmp_path):
    required_timings = {
        "router": 0.1,
        "expert_forward": 0.2,
        "final_scatter": 0.1,
        "grouped_gemm_gate": 0.1,
        "grouped_gemm_up": 0.1,
        "grouped_gemm_down": 0.1,
    }
    required_collectives = [
        "dispatch_alltoall",
        "combine_alltoall",
        "dispatch_backward_alltoall",
        "combine_backward_alltoall",
    ]
    complete_record = {
        "timings_s": required_timings,
        "collectives": [{"name": name} for name in required_collectives],
    }
    rank0 = tmp_path / "rank0.json"
    rank1 = tmp_path / "rank1.json"
    for destination, record in ((rank0, complete_record), (rank1, {"timings_s": {}})):
        destination.write_text(
            json.dumps(
                {
                    "metadata": {},
                    "records": {"model.layers.0.moe": record},
                }
            )
        )

    with pytest.raises(ValueError, match="rank1.json"):
        mark_measurement_artifacts_complete(
            [rank0, rank1], require_moe_metrics=True
        )

def test_mark_measurement_artifacts_complete_rejects_execution_path_mismatch(
    tmp_path,
):
    destination = tmp_path / "rank0.json"
    destination.write_text(
        json.dumps(
            {
                "metadata": {
                    "device_health": "green",
                    "gate_status": "passed",
                    "semantic_completion": "passed",
                    "global_step": 3,
                    "expert_execution_path": "padded_bmm",
                    "environment_overrides": {
                        "TORCHTUNE_MOE_GROUPED_EXPERTS": "1",
                        "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS": "0",
                    },
                },
                "records": {"model.layers.0.moe": {"timings_s": {}}},
            }
        )
    )
    with pytest.raises(ValueError, match="disagrees with environment_overrides"):
        mark_measurement_artifacts_complete(
            [destination],
            minimum_global_step=3,
            require_measurement_records=True,
            require_passed_gates=True,
            require_execution_path=True,
        )


def test_aggregate_measurement_files_preserves_step_timings(tmp_path):
    metadata = {
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "rank0.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {**metadata, "rank": 0},
                "records": {"moe": {}},
                "step_timings": [
                    {
                        "step": 1,
                        "time_per_step_s": 4.0,
                        "total_step_s": 4.0,
                        "timings_s": {"model_fwd_total": 1.5},
                        "timing_counts": {"model_fwd_total": 1},
                    }
                ],
            }
        )
    )
    result = aggregate_measurement_files([path])
    assert result["step_timings"][0]["steps"][0]["timings_s"] == {
        "model_fwd_total": 1.5
    }


def test_summarize_step_timings_discards_shared_warmup_and_reports_fractions():
    result = summarize_step_timings(
        [
            {
                "rank": 0,
                "steps": [
                    {
                        "step": 1,
                        "total_step_s": 10.0,
                        "timings_s": {"model_fwd_total": 4.0},
                    },
                    {
                        "step": 2,
                        "total_step_s": 20.0,
                        "local_tokens": 100,
                        "global_tokens": 200,
                        "tokens_per_second_per_gpu": 5.0,
                        "aggregate_tokens_per_second": 10.0,
                        "timings_s": {
                            "model_fwd_total": 8.0,
                            "backward_total": 6.0,
                        },
                    },
                    {
                        "step": 3,
                        "total_step_s": 30.0,
                        "local_tokens": 120,
                        "global_tokens": 240,
                        "tokens_per_second_per_gpu": 4.0,
                        "aggregate_tokens_per_second": 8.0,
                        "timings_s": {
                            "model_fwd_total": 12.0,
                            "backward_total": 9.0,
                        },
                    },
                ],
            },
            {
                "rank": 1,
                "steps": [
                    {"step": 1, "total_step_s": 10.0, "timings_s": {}},
                    {"step": 2, "total_step_s": 20.0, "timings_s": {}},
                    {"step": 3, "total_step_s": 30.0, "timings_s": {}},
                ],
            },
        ],
        warmup_steps=1,
    )
    assert result["warmup_steps_discarded"] == 1
    assert result["sample_count"] == 4
    assert result["step_start"] == 2
    assert result["step_end"] == 3
    assert result["mean_total_step_s"] == 25.0
    assert result["mean_local_tokens"] == 110.0
    assert result["mean_global_tokens"] == 220.0
    assert result["mean_tokens_per_second_per_gpu"] == 4.5
    assert result["mean_aggregate_tokens_per_second"] == 9.0
    assert result["phase_mean_s"] == {
        "backward_total": 7.5,
        "model_fwd_total": 10.0,
    }
    assert result["phase_fraction_of_total"] == {
        "backward_total": 15.0 / 100.0,
        "model_fwd_total": 20.0 / 100.0,
    }


def test_summarize_step_timings_limits_to_declared_measurement_window():
    result = summarize_step_timings(
        [
            {
                "step": step,
                "total_step_s": float(step),
                "local_tokens": step,
                "global_tokens": step,
                "tokens_per_second_per_gpu": float(step),
                "aggregate_tokens_per_second": float(step),
                "timings_s": {"model_fwd_total": float(step)},
            }
            for step in range(1, 6)
        ],
        warmup_steps=1,
        measurement_steps=2,
    )
    assert result["step_start"] == 2
    assert result["step_end"] == 3
    assert result["measurement_steps_requested"] == 2
    assert result["measurement_steps_used"] == 2
    assert result["sample_count"] == 2
    assert result["mean_total_step_s"] == 2.5


def test_summarize_step_timings_rejects_short_declared_measurement_window():
    with pytest.raises(ValueError, match="fewer post-warmup steps"):
        summarize_step_timings(
            [{"step": 1, "total_step_s": 1.0, "timings_s": {}}],
            warmup_steps=1,
            measurement_steps=1,
        )


def test_summarize_step_timings_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative"):
        summarize_step_timings(
            [{"step": 1, "total_step_s": -1, "timings_s": {}}]
        )


def test_summarize_pipeline_timings_requires_explicit_phases():
    pending = summarize_pipeline_timings(
        {"mean_total_step_s": 10.0, "phase_mean_s": {"attention": 2.0}},
        {"pipeline_parallel_degree": 2},
    )
    assert pending == {
        "pipeline_parallel_degree": 2,
        "bubble_seconds": None,
        "activation_transfer_seconds": None,
        "bubble_fraction": None,
        "activation_transfer_fraction": None,
        "timing_recorded": False,
    }
    measured = summarize_pipeline_timings(
        {
            "mean_total_step_s": 10.0,
            "phase_mean_s": {
                "pipeline_bubble": 1.5,
                "activation_transfer_forward": 0.5,
                "activation_transfer_backward": 0.25,
            },
        },
        {"pp": 2},
    )
    assert measured["bubble_seconds"] == 1.5
    assert measured["activation_transfer_seconds"] == 0.75
    assert measured["bubble_fraction"] == 0.15
    assert measured["activation_transfer_fraction"] == 0.075
    assert measured["timing_recorded"] is True


def test_summarize_ep_scaling_artifact_is_table_ready():
    summary = summarize_ep_scaling_artifact(
        {
            "common_metadata": {
                "ep_degree": 8,
                "environment_overrides": {"TORCHTUNE_EP_ALL2ALL": "1"},
            },
            "records": {
                "layers.0.moe": {
                    "rank_count": 2,
                    "total_tokens": 48,
                    "max_rank_tokens": 28,
                    "min_rank_tokens": 20,
                    "rank_token_imbalance_ratio": 1.4,
                    "grouped_gemm": [
                        {
                            "stage": "local_compute",
                            "compute_tokens": 48,
                            "routed_tokens": 40,
                            "padding_tokens": 8,
                            "zero_token_experts": 1,
                            "active_expert_gemm_count": 3,
                            "max_tokens_per_expert": 20,
                            "counts": [20, 12, 8],
                            "routed_counts": [18, 12, 6],
                            "rank_count": 2,
                        }
                    ],
                    "collectives": [
                        {
                            "name": "dispatch_alltoall",
                            "locality": "node_local",
                            "duration_s": 2.0,
                            "count": 4,
                        },
                        {
                            "name": "routing_metadata_allgather",
                            "locality": "node_local",
                            "duration_s": 0.75,
                            "count": 4,
                        }
                    ],
                    "memory": [
                        {
                            "phase": "forward",
                            "max_allocated_bytes": 100,
                            "max_reserved_bytes": 120,
                            "min_free_bytes": 880,
                        },
                        {
                            "phase": "steady_state",
                            "max_allocated_bytes": 90,
                            "max_reserved_bytes": 110,
                            "min_free_bytes": 890,
                        },
                    ],
                }
            },
            "step_timings": [
                {
                    "step": 1,
                    "total_step_s": 10.0,
                    "timings_s": {"model_fwd_total": 6.0},
                },
                {
                    "step": 2,
                    "total_step_s": 12.0,
                    "local_tokens": 40,
                    "global_tokens": 80,
                    "tokens_per_second_per_gpu": 5.0,
                    "aggregate_tokens_per_second": 10.0,
                    "timings_s": {"model_fwd_total": 7.0},
                },
            ],
        },
        warmup_steps=1,
    )
    assert summary["metadata"]["ep_degree"] == 8
    assert summary["metadata"]["environment_overrides"] == {
        "TORCHTUNE_EP_ALL2ALL": "1"
    }
    assert summary["transport"] == "alltoall"
    assert summary["token_semantics"].startswith("global routed assignments")
    assert summary["routed_tokens"]["layers"][0]["local_routed_tokens"] == 24.0
    local_gemm = summary["grouped_gemm"]["by_stage"]["local_compute"]
    assert local_gemm["padding_fraction"] == 8 / 48
    assert local_gemm["compute_to_routed_ratio"] == 48 / 40
    assert local_gemm["tokens_per_local_expert"] == 36 / 6
    assert local_gemm["expert_imbalance_ratio"] == 18 / (36 / 3)
    assert local_gemm["zero_routed_token_experts"] == 0
    assert summary["communication"]["by_locality"]["node_local"] == {
        "duration_s": 2.75,
        "count": 8,
    }
    assert summary["phase_timings"]["sample_count"] == 1
    assert summary["throughput"] == {
        "tokens_per_second_per_gpu": 5.0,
        "aggregate_tokens_per_second": 10.0,
        "mean_local_tokens": 40.0,
        "mean_global_tokens": 80.0,
    }
    assert summary["peak_memory"] == {
        "max_allocated_bytes": 100,
        "max_reserved_bytes": 120,
        "min_free_bytes": 880,
    }
    assert summary["steady_state_memory"] == {
        "max_allocated_bytes": 90,
        "max_reserved_bytes": 110,
        "min_free_bytes": 890,
    }


def test_compare_ep_scaling_summaries_separates_routing_metadata_time():
    result = compare_ep_scaling_summaries(
        {
            8: {
                "metadata": {"world_size": 8},
                "throughput": {},
                "communication": {
                    "by_locality": {"node_local": {"duration_s": 3.0}},
                    "by_collective": {
                        "routing_metadata_allgather": {"duration_s": 0.75},
                        "dispatch_alltoall": {"duration_s": 1.5},
                        "combine_alltoall": {"duration_s": 0.75},
                    },
                },
            }
        }
    )
    row = result["rows"][0]
    assert row["communication_seconds"] == 3.0
    assert row["routing_metadata_seconds"] == 0.75
    assert row["dispatch_alltoall_seconds"] == 1.5
    assert row["combine_alltoall_seconds"] == 0.75


def test_compare_ep_scaling_summaries_includes_backward_alltoall_time():
    summary = {
        "metadata": {
            "world_size": 1,
            "environment_overrides": {"TORCHTUNE_EP_ALL2ALL": "1"},
        },
        "throughput": {
            "tokens_per_second_per_gpu": 10.0,
            "aggregate_tokens_per_second": 10.0,
        },
        "communication": {
            "by_collective": {
                "dispatch_alltoall": {"duration_s": 1.0},
                "dispatch_backward_alltoall": {"duration_s": 2.0},
                "combine_alltoall": {"duration_s": 3.0},
                "combine_backward_alltoall": {"duration_s": 4.0},
            },
            "by_locality": {"node_local": {"duration_s": 10.0}},
        },
        "phase_timings": {},
        "routed_tokens": {},
        "expert_compute": {},
        "grouped_gemm": {},
        "peak_memory": {},
        "steady_state_memory": {},
    }
    row = compare_ep_scaling_summaries({8: summary})["rows"][0]
    assert row["dispatch_alltoall_seconds"] == 3.0
    assert row["combine_alltoall_seconds"] == 7.0
    assert row["communication_seconds"] == 10.0


def test_compare_ep_scaling_summaries_decomposes_collective_and_routing_time():
    routing_phases = {
        name: {"seconds": seconds, "event_count": 1}
        for name, seconds in {
            "dispatch_pack": 0.1,
            "dispatch_unpack": 0.2,
            "dispatch_backward_pack": 0.3,
            "dispatch_backward_unpack": 0.4,
            "combine_pack": 0.5,
            "combine_unpack": 0.6,
            "combine_backward_pack": 0.7,
            "combine_backward_unpack": 0.8,
        }.items()
    }
    summary = {
        "metadata": {"world_size": 8},
        "throughput": {},
        "communication": {
            "by_locality": {"node_local": {"duration_s": 5.0}},
            "by_collective": {
                "routing_metadata_allgather": {"duration_s": 0.5},
                "dispatch_alltoall": {"duration_s": 1.0},
            },
        },
        "routing_phases": routing_phases,
    }
    row = compare_ep_scaling_summaries({8: summary})["rows"][0]
    assert row["communication_seconds"] == 5.0
    assert row["collective_communication_seconds"] == pytest.approx(1.0)
    assert row["routing_metadata_collective_seconds"] == 0.5
    assert row["routing_dispatch_pack_unpack_seconds"] == pytest.approx(0.3)
    assert row["routing_dispatch_backward_pack_unpack_seconds"] == pytest.approx(0.7)
    assert row["routing_combine_pack_unpack_seconds"] == pytest.approx(1.1)
    assert row["routing_combine_backward_pack_unpack_seconds"] == pytest.approx(1.5)
    assert row["routing_pack_unpack_seconds"] == pytest.approx(3.6)


def test_summarize_ep_scaling_artifact_reports_sequential_expert_timing():
    summary = summarize_ep_scaling_artifact(
        {
            "common_metadata": {"expert_execution_path": "sequential"},
            "records": {
                "layers.0.moe": {
                    "timings_s": {
                        "sequential_expert_compute": 1.25,
                        "sequential_expert_gate": 0.4,
                        "sequential_expert_up": 0.4,
                        "sequential_expert_down": 0.45,
                    },
                    "timing_counts": {
                        "sequential_expert_compute": 4,
                        "sequential_expert_gate": 4,
                        "sequential_expert_up": 4,
                        "sequential_expert_down": 4,
                    },
                }
            },
        }
    )
    assert summary["expert_compute"] == {
        "execution_path": "sequential",
        "timing": {
            "by_timing": {
                "sequential_expert_compute": {"seconds": 1.25, "event_count": 4},
                "sequential_expert_gate": {"seconds": 0.4, "event_count": 4},
                "sequential_expert_up": {"seconds": 0.4, "event_count": 4},
                "sequential_expert_down": {"seconds": 0.45, "event_count": 4},
            },
            "total_seconds": 1.25,
            "event_count": 4,
        },
    }


def test_summarize_ep_scaling_artifact_reports_routing_metadata_timing():
    summary = summarize_ep_scaling_artifact(
        {
            "records": {
                "layers.0.moe": {
                    "timings_s": {
                        "routing_metadata_materialization": 0.25,
                        "routing_metadata_permutation": 0.75,
                    },
                    "timing_counts": {
                        "routing_metadata_materialization": 2,
                        "routing_metadata_permutation": 2,
                    },
                }
            }
        }
    )
    assert summary["routing_metadata"] == {
        "materialization_seconds": 0.25,
        "materialization_event_count": 2,
        "permutation_seconds": 0.75,
        "permutation_event_count": 2,
    }


def test_summarize_ep_scaling_artifact_reports_routing_phase_timing():
    summary = summarize_ep_scaling_artifact(
        {
            "records": {
                "layers.0.moe": {
                    "timings_s": {
                        "dispatch_pack": 0.1,
                        "dispatch_unpack": 0.2,
                        "combine_pack": 0.3,
                        "combine_unpack": 0.4,
                    },
                    "timing_counts": {
                        "dispatch_pack": 2,
                        "dispatch_unpack": 2,
                        "combine_pack": 2,
                        "combine_unpack": 2,
                    },
                }
            }
        }
    )
    assert summary["routing_phases"] == {
        "dispatch_pack": {"seconds": 0.1, "event_count": 2},
        "dispatch_unpack": {"seconds": 0.2, "event_count": 2},
        "combine_pack": {"seconds": 0.3, "event_count": 2},
        "combine_unpack": {"seconds": 0.4, "event_count": 2},
    }


def test_summarize_ep_scaling_artifact_reports_padded_bmm_timing():
    aggregated = aggregate_rank_records(
        [
            {
                "timings_s": {"padded_bmm": 0.75},
                "timing_counts": {"padded_bmm": 2},
                "grouped_gemm": [
                    {
                        "stage": "padded_bmm",
                        "counts": [3, 1],
                        "routed_tokens": 4,
                        "dense_compute_tokens": 6,
                    }
                ],
            },
            {
                "timings_s": {"padded_bmm": 0.25},
                "timing_counts": {"padded_bmm": 1},
                "grouped_gemm": [
                    {
                        "stage": "padded_bmm",
                        "counts": [2, 2],
                        "routed_tokens": 4,
                        "dense_compute_tokens": 4,
                    }
                ],
            },
        ]
    )
    summary = summarize_ep_scaling_artifact(
        {
            "common_metadata": {"expert_execution_path": "padded_bmm"},
            "records": {"layers.0.moe": aggregated},
        }
    )
    assert summary["expert_compute"]["timing"] == {
        "by_timing": {"padded_bmm": {"seconds": 1.0, "event_count": 3}},
        "total_seconds": 1.0,
        "event_count": 3,
    }


def test_summarize_ep_scaling_artifact_preserves_explicit_model_metrics():
    summary = summarize_ep_scaling_artifact(
        {
            "common_metadata": {
                "mfu_percent": 3.52,
                "active_flop_efficiency": 0.18,
                "total_parameters": 30.0,
                "active_parameters_per_token": 3.35,
            },
            "records": {},
        }
    )
    assert summary["model_metrics"] == {
        "mfu_percent": 3.52,
        "active_flop_efficiency": 0.18,
        "total_parameters": 30.0,
        "active_parameters_per_token": 3.35,
    }


def test_summarize_ep_scaling_artifact_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        summarize_ep_scaling_artifact([])


def test_compare_ep_scaling_summaries_reports_topology_aware_efficiency():
    summaries = {
        8: {
            "metadata": {"world_size": 8, "device_health": "green", "measurement_completion": "passed"},
            "throughput": {
                "tokens_per_second_per_gpu": 100.0,
                "aggregate_tokens_per_second": 800.0,
            },
            "communication": {"by_locality": {"node_local": {"duration_s": 2.0}}},
            "grouped_gemm": {"by_stage": {"gate": {"event_count": 2, "compute_tokens": 40, "routed_tokens": 32}}},
            "peak_memory": {"max_reserved_bytes": 100},
        },
        16: {
            "metadata": {"world_size": 16, "device_health": "green", "measurement_completion": "passed"},
            "throughput": {
                "tokens_per_second_per_gpu": 90.0,
                "aggregate_tokens_per_second": 1440.0,
            },
            "communication": {
                "by_locality": {"cross_node": {"duration_s": 12.0}},
                "by_collective": {
                    "dispatch_alltoall": {"duration_s": 5.0},
                    "combine_all_to_all": {"duration_s": 7.0},
                    "routing_metadata_allgather": {"duration_s": 3.0},
                },
            },
            "grouped_gemm": {"by_stage": {"gate": {"event_count": 3, "compute_tokens": 60, "routed_tokens": 48, "padding_tokens": 12, "zero_routed_token_experts": 2, "tokens_per_local_expert": 3.0, "expert_imbalance_ratio": 1.5}}},
            "peak_memory": {"max_reserved_bytes": 200},
        },
    }
    result = compare_ep_scaling_summaries(summaries)
    assert result["baseline_ep_degree"] == 8
    assert result["rows"][0]["scaling_efficiency"] == 1.0
    assert result["rows"][1]["expected_aggregate_tokens_per_second"] == 1600.0
    assert result["rows"][1]["scaling_efficiency"] == 0.9
    assert result["rows"][1]["communication_seconds"] == 12.0
    assert result["rows"][1]["dispatch_alltoall_seconds"] == 5.0
    assert result["rows"][1]["combine_alltoall_seconds"] == 7.0
    assert result["rows"][1]["routing_metadata_seconds"] == 3.0
    assert result["rows"][1]["expert_compute"] == {
        "event_count": 3,
        "compute_tokens": 60,
        "routed_tokens": 48,
        "padding_tokens": 12,
        "zero_token_experts": 2,
        "tokens_per_local_expert": 3.0,
        "max_expert_imbalance_ratio": 1.5,
        "padding_fraction": 0.2,
        "execution_path": None,
        "timing_seconds": None,
        "timing_event_count": 0,
        "timing_by_name": {},
        "by_stage": {
            "gate": {
                "event_count": 3,
                "compute_tokens": 60,
                "routed_tokens": 48,
                "padding_tokens": 12,
                "zero_routed_token_experts": 2,
                "tokens_per_local_expert": 3.0,
                "expert_imbalance_ratio": 1.5,
            }
        },
    }


def test_compare_ep_scaling_summaries_reports_sequential_timing():
    summaries = {
        8: {
            "metadata": {"world_size": 8},
            "throughput": {},
            "expert_compute": {
                "execution_path": "sequential",
                "timing": {
                    "total_seconds": 2.5,
                    "event_count": 8,
                    "by_timing": {"sequential_expert_compute": {"seconds": 2.5}},
                },
            },
        },
        16: {
            "metadata": {"world_size": 16},
            "throughput": {},
            "expert_compute": {
                "execution_path": "sequential",
                "timing": {
                    "total_seconds": 4.0,
                    "event_count": 8,
                    "by_timing": {"sequential_expert_compute": {"seconds": 4.0}},
                },
            },
        },
    }
    rows = compare_ep_scaling_summaries(summaries)["rows"]
    assert rows[1]["expert_compute"]["execution_path"] == "sequential"
    assert rows[1]["expert_compute"]["timing_seconds"] == 4.0
    assert rows[1]["expert_compute"]["timing_event_count"] == 8


def test_compare_ep_scaling_summaries_keeps_missing_efficiency_unclaimed():
    with pytest.raises(ValueError, match="missing baseline"):
        compare_ep_scaling_summaries({16: {}})
    result = compare_ep_scaling_summaries(
        {8: {"metadata": {}, "throughput": {}}}
    )
    assert result["rows"][0]["scaling_efficiency"] is None


def test_compare_ep_scaling_summaries_rejects_incompatible_router_controls():
    base = {
        "metadata": {
            "world_size": 8,
            "optimization_profile": "topk_router_off",
            "environment_overrides": {"TORCHTUNE_MOE_TOPK_ROUTING": "0"},
        },
        "throughput": {},
    }
    with pytest.raises(ValueError, match="TORCHTUNE_MOE_TOPK_ROUTING"):
        compare_ep_scaling_summaries(
            {
                8: base,
                16: {
                    **base,
                    "metadata": {
                        **base["metadata"],
                        "world_size": 16,
                        "optimization_profile": "topk_router_on",
                        "environment_overrides": {
                            "TORCHTUNE_MOE_TOPK_ROUTING": "1"
                        },
                    },
                },
            }
        )


def test_compare_ep_scaling_summaries_rejects_incompatible_alltoall_controls():
    base = {
        "metadata": {
            "world_size": 8,
            "optimization_profile": "rowwise_alltoall_unpermute_on",
            "environment_overrides": {
                "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE": "1",
                "TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS": "1",
            },
        },
        "throughput": {},
    }
    with pytest.raises(ValueError, match="TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE"):
        compare_ep_scaling_summaries(
            {
                8: base,
                16: {
                    **base,
                    "metadata": {
                        **base["metadata"],
                        "world_size": 16,
                        "optimization_profile": "rowwise_alltoall_unpermute_off",
                        "environment_overrides": {
                            **base["metadata"]["environment_overrides"],
                            "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE": "0",
                        },
                    },
                },
            }
        )


def test_compare_ep_scaling_summaries_rejects_conditional_alltoall_drift():
    base = {
        "metadata": {
            "world_size": 8,
            "environment_overrides": {
                "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS": "1",
            },
        },
        "throughput": {},
    }
    with pytest.raises(
        ValueError,
        match="TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS",
    ):
        compare_ep_scaling_summaries(
            {
                8: base,
                16: {
                    **base,
                    "metadata": {
                        **base["metadata"],
                        "world_size": 16,
                        "environment_overrides": {
                            "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS": "0",
                        },
                    },
                },
            }
        )


def test_compare_ep_scaling_summaries_rejects_recorded_optimization_drift():
    base = {
        "metadata": {
            "world_size": 8,
            "environment_overrides": {
                "TORCHTUNE_EP_DEVICE_ROUTING_METADATA": "0",
            },
        },
        "throughput": {},
    }
    with pytest.raises(ValueError, match="TORCHTUNE_EP_DEVICE_ROUTING_METADATA"):
        compare_ep_scaling_summaries(
            {
                8: base,
                16: {
                    **base,
                    "metadata": {
                        **base["metadata"],
                        "world_size": 16,
                        "environment_overrides": {
                            "TORCHTUNE_EP_DEVICE_ROUTING_METADATA": "1",
                        },
                    },
                },
            }
        )


def test_compare_ep_scaling_summaries_rejects_source_revision_drift():
    base = {
        "metadata": {
            "model": "qwen",
            "checkpoint": "ckpt",
            "source_revision": "abc123",
            "uncommitted_change_state": "clean",
            "sequence_length": 4096,
            "batch_size": 1,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw",
            "optimization_profile": "canonical",
            "routing_index_mode": "compact",
            "expert_execution_path": "grouped_mm",
            "world_size": 8,
            "environment_overrides": {},
        },
        "throughput": {},
    }
    with pytest.raises(ValueError, match="source_revision"):
        compare_ep_scaling_summaries(
            {
                8: base,
                16: {
                    **base,
                    "metadata": {
                        **base["metadata"],
                        "world_size": 16,
                        "source_revision": "def456",
                    },
                },
            }
        )


def test_compare_ep_scaling_summaries_strict_mode_rejects_sparse_metadata():
    with pytest.raises(ValueError, match="missing control metadata"):
        compare_ep_scaling_summaries(
            {8: {"metadata": {}, "throughput": {}}},
            require_control_metadata=True,
        )


def test_compare_optimization_summaries_reports_router_delta():
    def summary(topk, throughput):
        return {
            "metadata": {
                "model": "qwen",
                "checkpoint": "ckpt",
                "world_size": 8,
                "sequence_length": 4096,
                "batch_size": 1,
                "optimizer": "adamw",
                "topology": {"nodes": 1, "world_size": 8},
                "optimization_profile": f"topk_router_{'on' if topk else 'off'}",
                "routing_index_mode": "compact",
                "expert_execution_path": "sequential",
                "device_health": "green",
                "gate_status": "passed",
                "semantic_completion": "passed",
                "measurement_completion": "passed",
                "environment_overrides": {
                    "TORCHTUNE_MOE_TOPK_ROUTING": str(topk),
                    "TORCHTUNE_MOE_VECTOR_PACKING": str(topk),
                },
            },
            "throughput": {"tokens_per_second_per_gpu": throughput},
            "phase_timings": {"mean_total_step_s": 10.0},
            "peak_memory": {"max_reserved_bytes": 100},
        }

    result = compare_optimization_summaries(
        summary(0, 100.0), summary(1, 105.0),
        varying_controls=(
            "TORCHTUNE_MOE_TOPK_ROUTING",
            "TORCHTUNE_MOE_VECTOR_PACKING",
            "optimization_profile",
        ),
    )
    assert result["candidate_improves_throughput"] is True
    assert result["delta"]["tokens_per_second_per_gpu_fraction"] == 0.05
    assert result["promotion_status"] == "pending_independent_hardware_repeat"


def test_compare_optimization_summaries_reports_cpu_metadata_delta():
    def summary(enabled, throughput):
        return {
            "metadata": {
                "model": "qwen",
                "checkpoint": "ckpt",
                "world_size": 8,
                "sequence_length": 4096,
                "batch_size": 1,
                "optimizer": "adamw",
                "topology": {"nodes": 1, "world_size": 8},
                "optimization_profile": f"cpu_vector_routing_metadata_{'on' if enabled else 'off'}",
                "routing_index_mode": "compact",
                "expert_execution_path": "sequential",
                "device_health": "green",
                "gate_status": "passed",
                "semantic_completion": "passed",
                "measurement_completion": "passed",
                "environment_overrides": {
                    "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA": str(int(enabled)),
                },
            },
            "throughput": {"tokens_per_second_per_gpu": throughput},
            "phase_timings": {"mean_total_step_s": 10.0},
            "peak_memory": {"max_reserved_bytes": 100},
        }

    result = compare_optimization_summaries(
        summary(False, 100.0),
        summary(True, 102.0),
        varying_controls=(
            "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA",
            "optimization_profile",
        ),
    )
    assert result["candidate_improves_throughput"] is True
    assert result["delta"]["tokens_per_second_per_gpu_fraction"] == 0.02


def test_compare_optimization_summaries_rejects_unlisted_control_delta():
    baseline = {"metadata": {"model": "qwen"}}
    with pytest.raises(ValueError, match="missing metadata"):
        compare_optimization_summaries(baseline, baseline)


def test_compare_optimization_summaries_rejects_topology_as_varying_control():
    summary = {
        "metadata": {
            "model": "qwen",
            "checkpoint": "ckpt",
            "world_size": 8,
            "sequence_length": 4096,
            "batch_size": 1,
            "microbatch_size": 1,
            "optimizer": "adamw",
            "topology": {"nodes": 1, "world_size": 8},
            "optimization_profile": "baseline",
            "routing_index_mode": "compact",
            "expert_execution_path": "sequential",
            "device_health": "green",
            "gate_status": "passed",
            "semantic_completion": "passed",
            "measurement_completion": "passed",
            "environment_overrides": {},
        },
        "throughput": {"tokens_per_second_per_gpu": 100.0},
    }
    with pytest.raises(ValueError, match="cannot vary.*topology"):
        compare_optimization_summaries(
            summary,
            summary,
            varying_controls=("topology",),
        )


def test_compare_optimization_summaries_requires_real_declared_difference():
    summary = {
        "metadata": {
            "model": "qwen",
            "checkpoint": "ckpt",
            "world_size": 8,
            "sequence_length": 4096,
            "batch_size": 1,
            "optimizer": "adamw",
            "topology": {"nodes": 1, "world_size": 8},
            "optimization_profile": "baseline",
            "routing_index_mode": "compact",
            "expert_execution_path": "sequential",
            "device_health": "green",
            "gate_status": "passed",
            "semantic_completion": "passed",
            "measurement_completion": "passed",
            "environment_overrides": {},
        },
        "throughput": {"tokens_per_second_per_gpu": 100.0},
    }
    with pytest.raises(ValueError, match="must differ"):
        compare_optimization_summaries(
            summary,
            summary,
            varying_controls=("optimization_profile",),
        )


def test_compare_capacity_value_results_preserves_capacity_only_label():
    def result(model, active, total):
        return {
            "model": model,
            "total_parameters": total,
            "active_parameters_per_token": active,
            "sequence_length": 4096,
            "topology": {"nodes": 2, "world_size": 16},
            "per_gpu_throughput": 100.0,
            "aggregate_throughput": 1600.0,
            "mfu_percent": 3.0,
            "active_flop_efficiency": 0.2,
            "peak_memory": {"max_reserved_bytes": 100},
            "communication_fraction": 0.1,
            "expert_compute_fraction": 0.4,
            "stability": "passed",
            "fits_allocation": True,
        }

    comparison = compare_capacity_value_results(
        result("Gemma4-26B-A4B", 3.8, 26.0),
        result("Gemma4-31B", 31.0, 31.0),
    )
    assert comparison["comparison_label"] == "capacity_value_only"
    assert comparison["parity_claim_allowed"] is False
    assert comparison["active_compute_matched"] is False
    assert comparison["capacity_advantage"]["moe_model_larger"] is False
    assert comparison["decision_categories"]["best_capacity_value_result"] == (
        "capacity_value_candidate"
    )
    assert comparison["gap_attribution"]["communication"]["status"] == "measured"
    assert comparison["gap_attribution"]["communication"]["evidence"] == (
        "explicit communication_fraction values for both controls"
    )
    assert comparison["gap_attribution"]["communication"]["provenance"] == {
        "MoE": "explicit_override",
        "dense": "explicit_override",
    }
    assert comparison["gap_attribution"]["attention"]["status"] == "pending"


def test_compare_capacity_value_results_reports_deltas_and_pipeline_status():
    def result(model, throughput, mfu, memory, pipeline=None):
        return {
            "model": model,
            "total_parameters": 10.0,
            "active_parameters_per_token": 3.0,
            "sequence_length": 4096,
            "topology": {"nodes": 2, "world_size": 16},
            "per_gpu_throughput": throughput,
            "aggregate_throughput": throughput * 16,
            "mfu_percent": mfu,
            "active_flop_efficiency": 0.2,
            "peak_memory": {"max_reserved_gib": memory},
            "steady_state_memory": {"max_reserved_gib": memory - 1.0},
            "communication_fraction": 0.1,
            "expert_compute_fraction": 0.4,
            "stability": "passed",
            "pipeline": pipeline or {"timing_recorded": False},
        }

    comparison = compare_capacity_value_results(
        result("moe", 120.0, 4.0, 20.0, {"timing_recorded": True, "bubble_fraction": 0.1}),
        result("dense", 100.0, 5.0, 10.0),
    )
    assert comparison["comparison_metrics"]["per_gpu_throughput"] == {
        "moe_minus_dense": 20.0,
        "moe_over_dense": 1.2,
    }
    assert comparison["comparison_metrics"]["mfu_percent"] == {
        "moe_minus_dense": -1.0,
        "moe_over_dense": 0.8,
    }
    assert comparison["comparison_metrics"]["peak_memory"] == {
        "moe_minus_dense": 10.0,
        "moe_over_dense": 2.0,
    }
    assert comparison["comparison_metrics"]["steady_state_memory"] == {
        "moe_minus_dense": 10.0,
        "moe_over_dense": 2.111111111111111,
    }
    assert comparison["pipeline"]["moe"]["bubble_fraction"] == 0.1
    assert comparison["pipeline"]["dense"]["timing_recorded"] is False


def test_compare_capacity_value_results_rejects_missing_metrics():
    with pytest.raises(ValueError, match="MoE capacity/value result missing"):
        compare_capacity_value_results(
            {field: True for field in ()},
            {field: True for field in ()},
        )


def test_compare_capacity_value_results_rejects_mismatched_sequence_length():
    base = {
        "model": "model",
        "total_parameters": 1.0,
        "active_parameters_per_token": 1.0,
        "sequence_length": 4096,
        "topology": {"nodes": 2, "world_size": 16},
        "per_gpu_throughput": 1.0,
        "aggregate_throughput": 16.0,
        "mfu_percent": 1.0,
        "active_flop_efficiency": 0.1,
        "peak_memory": {},
        "communication_fraction": 0.1,
        "expert_compute_fraction": 0.1,
        "stability": "passed",
    }
    base["peak_memory"] = {"max_reserved_gib": 1.0}
    dense = dict(base, sequence_length=2048)
    with pytest.raises(ValueError, match="same sequence_length"):
        compare_capacity_value_results(base, dense)


def test_compare_capacity_value_results_rejects_nonfinite_metric():
    base = {
        "model": "model",
        "total_parameters": 1.0,
        "active_parameters_per_token": 1.0,
        "sequence_length": 4096,
        "topology": {"nodes": 2, "world_size": 16},
        "per_gpu_throughput": float("nan"),
        "aggregate_throughput": 16.0,
        "mfu_percent": 1.0,
        "active_flop_efficiency": 0.1,
        "peak_memory": {"max_reserved_gib": 1.0},
        "communication_fraction": 0.1,
        "expert_compute_fraction": 0.1,
        "stability": "passed",
    }
    with pytest.raises(ValueError, match="per_gpu_throughput"):
        compare_capacity_value_results(base, dict(base))


def test_compare_capacity_value_results_canonical_rejects_source_drift():
    def result(model, revision):
        return {
            "model": model,
            "checkpoint": "/models/checkpoint",
            "source_revision": revision,
            "uncommitted_change_state": "clean",
            "batch_size": 1,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "torch.optim.AdamW",
            "environment_overrides": {},
            "total_parameters": 10.0,
            "active_parameters_per_token": 3.0,
            "sequence_length": 4096,
            "topology": {"nodes": 2, "world_size": 16},
            "per_gpu_throughput": 100.0,
            "aggregate_throughput": 1600.0,
            "mfu_percent": 3.0,
            "active_flop_efficiency": 0.2,
            "peak_memory": {"max_reserved_gib": 10.0},
            "steady_state_memory": {"max_reserved_gib": 9.0},
            "communication_fraction": 0.1,
            "expert_compute_fraction": 0.4,
            "stability": "passed",
        }

    with pytest.raises(ValueError, match="same source_revision"):
        compare_capacity_value_results(
            result("moe", "abc123"),
            result("dense", "def456"),
            require_canonical_metadata=True,
        )


def test_compare_capacity_value_results_canonical_requires_steady_state_memory():
    result = {
        "model": "Gemma4-26B-A4B",
        "checkpoint": "/models/gemma",
        "source_revision": "abc123",
        "uncommitted_change_state": "clean",
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "torch.optim.AdamW",
        "environment_overrides": {},
        "total_parameters": 10.0,
        "active_parameters_per_token": 3.0,
        "sequence_length": 4096,
        "topology": {"nodes": 2, "world_size": 16},
        "per_gpu_throughput": 100.0,
        "aggregate_throughput": 1600.0,
        "mfu_percent": 3.0,
        "active_flop_efficiency": 0.2,
        "peak_memory": {"max_reserved_gib": 10.0},
        "communication_fraction": 0.1,
        "expert_compute_fraction": 0.4,
        "stability": "passed",
    }
    with pytest.raises(ValueError, match="steady_state_memory"):
        compare_capacity_value_results(
            result,
            {**result, "model": "Gemma4-31B"},
            require_canonical_metadata=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "unknown", "model must be non-placeholder"),
        ("stability", "pending", "stability must be 'passed'"),
    ],
)
def test_compare_capacity_value_results_canonical_rejects_unpromotable_metadata(
    field, value, message
):
    result = {
        "model": "Gemma4-26B-A4B",
        "checkpoint": "/models/gemma",
        "source_revision": "abc123",
        "uncommitted_change_state": "clean",
        "total_parameters": 10.0,
        "active_parameters_per_token": 3.0,
        "sequence_length": 4096,
        "topology": {"nodes": 2, "world_size": 16},
        "per_gpu_throughput": 100.0,
        "aggregate_throughput": 1600.0,
        "mfu_percent": 3.0,
        "active_flop_efficiency": 0.2,
        "peak_memory": {"max_reserved_gib": 10.0},
        "steady_state_memory": {"max_reserved_gib": 9.0},
        "communication_fraction": 0.1,
        "expert_compute_fraction": 0.4,
        "stability": "passed",
    }
    result[field] = value
    with pytest.raises(ValueError, match=message):
        compare_capacity_value_results(
            result,
            dict(result),
            require_canonical_metadata=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "total_parameters",
        "active_parameters_per_token",
        "per_gpu_throughput",
        "aggregate_throughput",
        "mfu_percent",
        "active_flop_efficiency",
    ],
)
def test_compare_capacity_value_results_canonical_rejects_zero_metric(field):
    result = {
        "model": "Gemma4-26B-A4B",
        "checkpoint": "/models/gemma",
        "source_revision": "abc123",
        "uncommitted_change_state": "clean",
        "total_parameters": 10.0,
        "active_parameters_per_token": 3.0,
        "sequence_length": 4096,
        "topology": {"nodes": 2, "world_size": 16},
        "per_gpu_throughput": 100.0,
        "aggregate_throughput": 1600.0,
        "mfu_percent": 3.0,
        "active_flop_efficiency": 0.2,
        "peak_memory": {"max_reserved_gib": 10.0},
        "steady_state_memory": {"max_reserved_gib": 9.0},
        "communication_fraction": 0.1,
        "expert_compute_fraction": 0.4,
        "stability": "passed",
    }
    result[field] = 0.0
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        compare_capacity_value_results(
            result,
            dict(result),
            require_canonical_metadata=True,
        )


def test_evaluate_kernel_parity_requires_repeatable_both_metric_thresholds():
    def result(throughput, mfu, repeats=2):
        return {
            "throughput_tokens_per_second_per_gpu": throughput,
            "mfu_percent": mfu,
            "sequence_length": 4096,
            "topology": {"nodes": 1},
            "optimizer": "torch.optim.AdamW",
            "measurement_window": {"warmup_steps": 3, "measurement_steps": 5},
            "status": "validated",
            "independent_repeats": repeats,
        }

    result_value = evaluate_kernel_parity(
        result(830.044, 4.745), result(873.7295, 4.994)
    )
    assert result_value["meets_threshold"] is True
    assert result_value["repeatable"] is True
    assert result_value["promoted"] is True

    not_repeatable = evaluate_kernel_parity(
        result(900.0, 5.0, repeats=1), result(873.7295, 4.994)
    )
    assert not_repeatable["meets_threshold"] is True
    assert not_repeatable["promoted"] is False


def test_evaluate_kernel_parity_rejects_mismatched_controls():
    base = {
        "throughput_tokens_per_second_per_gpu": 1.0,
        "mfu_percent": 1.0,
        "sequence_length": 4096,
        "topology": {},
        "optimizer": "adamw",
        "measurement_window": {"measurement_steps": 1},
        "status": "validated",
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    with pytest.raises(ValueError, match="same optimizer"):
        evaluate_kernel_parity(base, dict(base, optimizer="adafactor"))


def test_evaluate_kernel_parity_canonical_cli_requires_model_and_active_compute():
    base = {
        "throughput_tokens_per_second_per_gpu": 1.0,
        "mfu_percent": 1.0,
        "sequence_length": 4096,
        "topology": {"nodes": 1, "world_size": 8, "participating_tiles": 8},
        "optimizer": "adamw",
        "measurement_window": {"measurement_steps": 1},
        "status": "validated",
        "source_revision": "abc123",
        "checkpoint": "/models/qwen3",
        "uncommitted_change_state": "clean",
        "device_health": "green",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    with pytest.raises(ValueError, match="model must be"):
        evaluate_kernel_parity(
            dict(base, model="not-qwen", active_parameters_per_token=3.35),
            dict(base, model="Qwen3-4B", active_parameters_per_token=4.0),
            require_canonical_metadata=True,
        )
    with pytest.raises(ValueError, match="active_parameters_per_token"):
        evaluate_kernel_parity(
            dict(base, model="Qwen3-30B-A3B"),
            dict(base, model="Qwen3-4B", active_parameters_per_token=4.0),
            require_canonical_metadata=True,
        )


def test_evaluate_kernel_parity_canonical_cli_requires_provenance_and_gates():
    base = {
        "model": "Qwen3-30B-A3B",
        "active_parameters_per_token": 3.35,
        "throughput_tokens_per_second_per_gpu": 1.0,
        "mfu_percent": 1.0,
        "sequence_length": 4096,
        "topology": {"nodes": 1, "world_size": 8, "participating_tiles": 8},
        "optimizer": "adamw",
        "measurement_window": {"measurement_steps": 1},
        "status": "validated",
    }
    with pytest.raises(ValueError, match="source_revision"):
        evaluate_kernel_parity(
            base,
            dict(base, model="Qwen3-4B", active_parameters_per_token=4.0),
            require_canonical_metadata=True,
        )
    complete = dict(
        base,
        source_revision="abc123",
        checkpoint="/models/qwen3",
        uncommitted_change_state="clean",
        device_health="green",
        semantic_completion="passed",
        measurement_completion="passed",
    )
    incomplete_topology = dict(complete, topology={"nodes": 1, "world_size": 8})
    with pytest.raises(ValueError, match="participating_tiles"):
        evaluate_kernel_parity(
            incomplete_topology,
            dict(incomplete_topology, model="Qwen3-4B", active_parameters_per_token=4.0),
            require_canonical_metadata=True,
        )


def test_evaluate_kernel_parity_canonical_cli_rejects_batch_drift():
    base = {
        "model": "Qwen3-30B-A3B",
        "active_parameters_per_token": 3.35,
        "throughput_tokens_per_second_per_gpu": 1.0,
        "mfu_percent": 1.0,
        "sequence_length": 4096,
        "topology": {"nodes": 1, "world_size": 8, "participating_tiles": 8},
        "optimizer": "adamw",
        "measurement_window": {"measurement_steps": 1},
        "status": "validated",
        "source_revision": "abc123",
        "checkpoint": "/models/qwen3",
        "uncommitted_change_state": "clean",
        "device_health": "green",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    with pytest.raises(ValueError, match="same batch_size"):
        evaluate_kernel_parity(
            base,
            dict(base, model="Qwen3-4B", active_parameters_per_token=4.0, batch_size=2),
            require_canonical_metadata=True,
        )


@pytest.mark.parametrize("field", ["source_revision", "uncommitted_change_state"])
def test_evaluate_kernel_parity_canonical_cli_rejects_source_drift(field):
    base = {
        "model": "Qwen3-30B-A3B",
        "active_parameters_per_token": 3.35,
        "throughput_tokens_per_second_per_gpu": 1.0,
        "mfu_percent": 1.0,
        "sequence_length": 4096,
        "topology": {"nodes": 1, "world_size": 8, "participating_tiles": 8},
        "optimizer": "adamw",
        "measurement_window": {"measurement_steps": 1},
        "status": "validated",
        "source_revision": "abc123",
        "checkpoint": "/models/qwen3",
        "uncommitted_change_state": "clean",
        "device_health": "green",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    dense = dict(base, model="Qwen3-4B", active_parameters_per_token=4.0)
    dense[field] = "def456" if field == "source_revision" else "dirty"
    with pytest.raises(ValueError, match=f"same {field}"):
        evaluate_kernel_parity(
            base,
            dense,
            require_canonical_metadata=True,
        )


def test_canonical_measurement_seal_rejects_wrong_router_semantics(tmp_path):
    from torchtune.modules.moe.measurement import mark_measurement_artifacts_complete

    path = tmp_path / "rank0.json"
    path.write_text(
        __import__("json").dumps(
            {
                "metadata": {
                    "model": "Qwen3-30B-A3B",
                    "router_semantics": "sigmoid_argsort_v1",
                    "global_step": 1,
                },
                "records": {"layer": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="probability_topk_v2"):
        mark_measurement_artifacts_complete(
            [path],
            require_moe_metrics=True,
        )


def test_ep_scaling_report_formats_pending_values():
    from experiments.qwen3_moe.report_ep_scaling import format_markdown

    report = {
        "comparison": {
            "rows": [
                {
                    "ep_degree": 16,
                    "world_size": 16.0,
                    "transport": "alltoall",
                    "tokens_per_second_per_gpu": None,
                    "aggregate_tokens_per_second": None,
                    "expected_aggregate_tokens_per_second": None,
                    "scaling_efficiency": None,
                    "communication_seconds": 1.5,
                    "routing_metadata_seconds": None,
                    "phase_timings": {
                        "phase_mean_s": {
                            "router": 0.05,
                            "attention": 0.1,
                            "non_expert": 0.2,
                            "backward_total": 0.3,
                            "manual_grad_release_total": 0.35,
                            "optimizer_step_total": 0.4,
                        },
                        "phase_fraction_of_total": {"expert_forward": 0.25},
                        "mean_total_step_s": 2.5,
                    },
                    "peak_memory": {"max_reserved_bytes": 10},
                    "steady_state_memory": {
                        "max_reserved_bytes": 8,
                        "max_allocated_bytes": 7,
                    },
                    "device_health": "green",
                    "measurement_completion": "passed",
                }
            ]
        }
    }
    rendered = format_markdown(report)
    row = next(line for line in rendered.splitlines() if line.startswith("| 16 |"))
    assert "| 16 | 16.0000 | pending | pending | pending | alltoall |" in row
    assert "| 0.2500 | 0.0500 | 0.1000 | 0.2000 | 0.3000 | 0.3500 | 0.4000 | 2.5000 |" in row
    assert row.endswith("| 1.5000 | 10 | 8 | 7 | green | passed |")


def test_ep_scaling_report_markdown_preserves_pending_interpretation():
    from experiments.qwen3_moe.report_ep_scaling import format_markdown

    rendered = format_markdown(
        {
            "comparison": {"rows": []},
            "interpretation": {
                "status": "pending_larger_token_volume_control",
                "reason": "larger token volume is required before classification",
            },
        }
    )
    assert "pending_larger_token_volume_control" in rendered
    assert "larger token volume is required before classification" in rendered


def test_ep_scaling_report_markdown_table_has_matching_column_counts():
    from experiments.qwen3_moe.report_ep_scaling import format_markdown

    rendered = format_markdown({"comparison": {"rows": []}})
    table_lines = rendered.splitlines()[2:4]
    assert len(table_lines) == 2
    assert table_lines[0].count("|") == table_lines[1].count("|")


def test_ep_scaling_report_markdown_renders_sequential_projection_timings():
    from experiments.qwen3_moe.report_ep_scaling import format_markdown

    rendered = format_markdown(
        {
            "comparison": {
                "rows": [
                    {
                        "ep_degree": 8,
                        "world_size": 8,
                        "expert_compute": {
                            "timing_by_name": {
                                "sequential_expert_gate": {"seconds": 1.1},
                                "sequential_expert_up": {"seconds": 1.2},
                                "sequential_expert_down": {"seconds": 1.3},
                            }
                        },
                    }
                ]
            }
        }
    )
    row = next(line for line in rendered.splitlines() if line.startswith("| 8 |"))
    assert "| pending | pending | pending | 1.1000 | 1.2000 | 1.3000 |" in row


def test_ep_scaling_report_markdown_renders_grouped_projection_timings():
    from experiments.qwen3_moe.report_ep_scaling import format_markdown

    rendered = format_markdown(
        {
            "comparison": {
                "rows": [
                    {
                        "ep_degree": 8,
                        "world_size": 8,
                        "expert_compute": {
                            "timing_by_name": {
                                "grouped_gemm_gate": {"seconds": 2.1},
                                "grouped_gemm_up": {"seconds": 2.2},
                                "grouped_gemm_down": {"seconds": 2.3},
                            }
                        },
                    }
                ]
            }
        }
    )
    row = next(line for line in rendered.splitlines() if line.startswith("| 8 |"))
    assert "| pending | pending | pending | 2.1000 | 2.2000 | 2.3000 |" in row


def test_ep_scaling_report_markdown_renders_decisions_and_gaps():
    from experiments.qwen3_moe.report_ep_scaling import format_markdown

    rendered = format_markdown(
        {
            "comparison": {"rows": []},
            "decision_categories": {"best_ep_scaling_point": "pending_measurement"},
            "gap_attribution": {
                "communication": {"status": "pending", "evidence": "collectives"}
            },
            "interpretation": {},
        }
    )
    assert "## Decision Categories" in rendered
    assert "pending_measurement" in rendered
    assert "## Gap Attribution" in rendered
    assert "| communication | pending | collectives |" in rendered


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("ep_degree", 8, "ep_degree=8"),
        ("world_size", 8, "world_size=8"),
    ],
)
def test_ep_scaling_report_rejects_mislabeled_ep16_artifact(field, value, message):
    from experiments.qwen3_moe.report_ep_scaling import _validate_ep_leg_metadata

    metadata = {
        "source_revision": "abc",
        "model": "Qwen3-30B-A3B",
        "router_semantics": "probability_topk_v2",
        "checkpoint": "/models/qwen3",
        "ep_degree": 16,
        "world_size": 16,
        "topology": {"ep": 16},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "cross_node",
            "TORCHTUNE_EP_ALL2ALL": "1",
        },
    }
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=16,
            expected_locality="cross_node",
        )


def test_ep_scaling_report_rejects_wrong_topology_and_locality():
    from experiments.qwen3_moe.report_ep_scaling import _validate_ep_leg_metadata

    metadata = {
        "source_revision": "abc",
        "model": "Qwen3-30B-A3B",
        "router_semantics": "probability_topk_v2",
        "checkpoint": "/models/qwen3",
        "ep_degree": 8,
        "world_size": 8,
        "topology": {"ep": 16},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "cross_node",
            "TORCHTUNE_EP_ALL2ALL": "1",
        },
    }
    with pytest.raises(ValueError, match="topology reports ep=16"):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=8,
            expected_locality="node_local",
        )
    metadata["topology"]["ep"] = 8
    with pytest.raises(ValueError, match="locality must be 'node_local'"):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=8,
            expected_locality="node_local",
        )


def test_ep_scaling_report_rejects_missing_canonical_transport():
    from experiments.qwen3_moe.report_ep_scaling import _validate_ep_leg_metadata

    metadata = {
        "source_revision": "abc",
        "model": "Qwen3-30B-A3B",
        "router_semantics": "probability_topk_v2",
        "checkpoint": "/models/qwen3",
        "ep_degree": 8,
        "world_size": 8,
        "topology": {"ep": 8},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "node_local"
        },
    }
    with pytest.raises(ValueError, match="transport must be AllToAll"):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=8,
            expected_locality="node_local",
        )


@pytest.mark.parametrize(
    ("model", "router_semantics", "expected"),
    [
        ("Qwen3-30B-A3B", "sigmoid_argsort_v1", "probability_topk_v2"),
        ("Gemma4-26B-A4B", "probability_topk_v2", "sigmoid_argsort_v1"),
    ],
)
def test_ep_scaling_report_rejects_wrong_router_semantics(
    model, router_semantics, expected
):
    from experiments.qwen3_moe.report_ep_scaling import _validate_ep_leg_metadata

    metadata = {
        "source_revision": "abc",
        "model": model,
        "router_semantics": router_semantics,
        "checkpoint": "/models/moe",
        "ep_degree": 8,
        "world_size": 8,
        "topology": {"ep": 8},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "node_local",
            "TORCHTUNE_EP_ALL2ALL": "1",
        },
    }
    with pytest.raises(ValueError, match=expected):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=8,
            expected_locality="node_local",
        )


@pytest.mark.parametrize("field", ["source_revision", "model", "checkpoint"])
def test_ep_scaling_report_rejects_placeholder_provenance(field):
    from experiments.qwen3_moe.report_ep_scaling import _validate_ep_leg_metadata

    metadata = {
        "source_revision": "abc",
        "model": "Qwen3-30B-A3B",
        "router_semantics": "probability_topk_v2",
        "checkpoint": "/models/qwen3",
        "ep_degree": 8,
        "world_size": 8,
        "topology": {"ep": 8},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "node_local",
            "TORCHTUNE_EP_ALL2ALL": "1",
        },
    }
    metadata[field] = "unknown"
    with pytest.raises(ValueError, match=f"non-placeholder {field}"):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=8,
            expected_locality="node_local",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device_health", "yellow", "device_health must be 'green'"),
        ("gate_status", "failed", "gate_status must be 'passed'"),
        ("semantic_completion", "failed", "semantic_completion must be 'passed'"),
        ("measurement_completion", "failed", "measurement_completion must be 'passed'"),
        ("uncommitted_change_state", "unknown", "non-placeholder uncommitted_change_state"),
    ],
)
def test_ep_scaling_report_rejects_unpassed_or_missing_provenance(
    field, value, message
):
    from experiments.qwen3_moe.report_ep_scaling import _validate_ep_leg_metadata

    metadata = {
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "model": "Qwen3-30B-A3B",
        "router_semantics": "probability_topk_v2",
        "checkpoint": "/models/qwen3",
        "ep_degree": 8,
        "world_size": 8,
        "topology": {"ep": 8},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "node_local",
            "TORCHTUNE_EP_ALL2ALL": "1",
        },
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_ep_leg_metadata(
            {"common_metadata": metadata},
            expected_ep=8,
            expected_locality="node_local",
        )


def test_ep_scaling_report_reads_aggregated_measurement_window():
    from experiments.qwen3_moe.report_ep_scaling import _warmup_steps

    assert _warmup_steps(
        {
            "common_metadata": {
                "measurement_window": {
                    "warmup_steps": 4,
                    "measurement_steps": 8,
                    "steady_state_steps": 4,
                }
            }
        }
    ) == 4


@pytest.mark.parametrize(
    "phase_timings, message",
    [
        ({"sample_count": 0}, "no post-warmup step timing records"),
        (
            {
                "sample_count": 1,
                "mean_total_step_s": 1.0,
                "mean_tokens_per_second_per_gpu": None,
                "mean_aggregate_tokens_per_second": 2.0,
            },
            "no valid mean_tokens_per_second_per_gpu",
        ),
    ],
)
def test_ep_scaling_report_rejects_incomplete_timing_summary(
    phase_timings, message
):
    from experiments.qwen3_moe.report_ep_scaling import _validate_report_timing

    with pytest.raises(ValueError, match=message):
        _validate_report_timing({"phase_timings": phase_timings}, expected_ep=16)


def test_ep_scaling_report_rejects_short_declared_measurement_window():
    from experiments.qwen3_moe.report_ep_scaling import _validate_report_timing

    phase_timings = {
        "sample_count": 3,
        "measurement_steps_used": 3,
        "mean_total_step_s": 1.0,
        "mean_tokens_per_second_per_gpu": 2.0,
        "mean_aggregate_tokens_per_second": 4.0,
    }
    with pytest.raises(ValueError, match="recorded 3 measurement steps; expected 8"):
        _validate_report_timing(
            {"phase_timings": phase_timings},
            expected_ep=16,
            expected_measurement_steps=8,
        )


def test_ep_scaling_report_accepts_complete_declared_measurement_window():
    from experiments.qwen3_moe.report_ep_scaling import _validate_report_timing

    phase_timings = {
        "sample_count": 8,
        "measurement_steps_used": 8,
        "mean_total_step_s": 1.0,
        "mean_tokens_per_second_per_gpu": 2.0,
        "mean_aggregate_tokens_per_second": 4.0,
    }
    _validate_report_timing(
        {"phase_timings": phase_timings},
        expected_ep=16,
        expected_measurement_steps=8,
    )


def test_ep_scaling_report_requires_routing_metadata_attribution():
    from experiments.qwen3_moe.report_ep_scaling import (
        _validate_routing_metadata_attribution,
    )

    summary = {
        "communication": {
            "by_collective": {
                "routing_metadata_allgather": {"duration_s": 0.5, "count": 2}
            }
        },
        "routing_metadata": {
            "materialization_seconds": 0.1,
            "materialization_event_count": 2,
            "permutation_seconds": 0.2,
            "permutation_event_count": 2,
        },
    }
    _validate_routing_metadata_attribution(summary, expected_ep=8)
    del summary["routing_metadata"]["permutation_seconds"]
    with pytest.raises(ValueError, match="missing routing metadata permutation timing"):
        _validate_routing_metadata_attribution(summary, expected_ep=8)


def test_ep_scaling_report_requires_backward_alltoall_attribution():
    from experiments.qwen3_moe.report_ep_scaling import _validate_alltoall_attribution

    summary = {
        "communication": {
            "by_collective": {
                "dispatch_alltoall": {"duration_s": 1.0, "count": 1},
                "combine_alltoall": {"duration_s": 1.0, "count": 1},
                "dispatch_backward_alltoall": {"duration_s": 1.0, "count": 1},
                "combine_backward_alltoall": {"duration_s": 1.0, "count": 1},
            },
            "by_locality": {"node_local": {"duration_s": 4.0, "count": 4}},
        }
    }
    _validate_alltoall_attribution(
        summary, expected_ep=8, expected_locality="node_local"
    )
    summary["communication"]["by_locality"] = {
        "cross_node": {"duration_s": 4.0, "count": 4}
    }
    with pytest.raises(ValueError, match="collective locality"):
        _validate_alltoall_attribution(
            summary, expected_ep=8, expected_locality="node_local"
        )
    summary["communication"]["by_locality"] = {
        "node_local": {"duration_s": 4.0, "count": 4}
    }
    del summary["communication"]["by_collective"]["combine_backward_alltoall"]
    with pytest.raises(ValueError, match="combine_backward_alltoall"):
        _validate_alltoall_attribution(
            summary, expected_ep=8, expected_locality="node_local"
        )


def test_ep_scaling_report_requires_routing_phase_attribution():
    from experiments.qwen3_moe.report_ep_scaling import (
        _validate_routing_phase_attribution,
    )

    names = (
        "dispatch_pack",
        "dispatch_unpack",
        "dispatch_backward_pack",
        "dispatch_backward_unpack",
        "combine_pack",
        "combine_unpack",
        "combine_backward_pack",
        "combine_backward_unpack",
    )
    summary = {
        "routing_phases": {
            name: {"seconds": 0.1, "event_count": 2} for name in names
        }
    }
    _validate_routing_phase_attribution(summary, expected_ep=8)
    del summary["routing_phases"]["combine_backward_unpack"]
    with pytest.raises(ValueError, match="combine_backward_unpack"):
        _validate_routing_phase_attribution(summary, expected_ep=8)


def test_ep_scaling_report_decision_and_gap_fields_are_machine_readable():
    from experiments.qwen3_moe.report_ep_scaling import (
        _decision_categories,
        _gap_attribution,
    )

    rows = [
        {
            "ep_degree": 8,
            "independent_repeats": 2,
            "aggregate_tokens_per_second": 800.0,
            "scaling_efficiency": 1.0,
            "model_metrics": {"mfu_percent": 3.5},
            "communication_seconds": 1.0,
            "expert_compute": {"timing_seconds": 2.0},
            "phase_timings": {
                "phase_mean_s": {
                    "attention": 0.5,
                    "optimizer_step_total": 0.2,
                }
            },
        }
    ]
    decisions = _decision_categories(rows)
    assert decisions["highest_repeatable_moe_per_gpu_mfu"] == "ep8_mfu"
    assert decisions["highest_aggregate_throughput"] == "ep8_aggregate_throughput"
    assert decisions["best_ep_scaling_point"] == "ep8_scaling"
    gaps = _gap_attribution(rows)
    assert gaps["communication"]["status"] == "measured"
    assert gaps["communication"]["pack_unpack_status"] == "pending"
    assert gaps["expert_compute"]["status"] == "measured"
    assert gaps["attention"]["status"] == "measured"
    assert gaps["optimizer"]["status"] == "measured"
    assert gaps["pipeline_bubble"]["status"] == "pending"


def test_ep_scaling_report_gap_attribution_decomposes_routing_overhead():
    from experiments.qwen3_moe.report_ep_scaling import _gap_attribution

    gaps = _gap_attribution(
        [
            {
                "collective_communication_seconds": 1.5,
                "routing_metadata_collective_seconds": 0.25,
                "routing_pack_unpack_seconds": 0.75,
            }
        ]
    )
    assert gaps["communication"]["status"] == "measured"
    assert gaps["communication"]["collective_seconds"] == [1.5]
    assert gaps["communication"]["routing_metadata_collective_seconds"] == [0.25]
    assert gaps["communication"]["pack_unpack_seconds"] == [0.75]
    assert gaps["communication"]["pack_unpack_status"] == "measured"


def test_ep_scaling_report_keeps_single_run_winners_pending():
    from experiments.qwen3_moe.report_ep_scaling import _decision_categories

    decisions = _decision_categories(
        [
            {
                "ep_degree": 8,
                "aggregate_tokens_per_second": 800.0,
                "scaling_efficiency": 1.0,
                "model_metrics": {"mfu_percent": 3.5},
            }
        ]
    )
    assert decisions["highest_repeatable_moe_per_gpu_mfu"] == "pending_measurement"
    assert decisions["highest_aggregate_throughput"] == "pending_measurement"
    assert decisions["best_ep_scaling_point"] == "pending_measurement"


def test_ep_scaling_report_repeatability_contract_is_explicit():
    from experiments.qwen3_moe.report_ep_scaling import _decision_categories

    rows = [
        {
            "ep_degree": 8,
            "aggregate_tokens_per_second": 800.0,
            "scaling_efficiency": 1.0,
            "model_metrics": {"mfu_percent": 3.5},
        }
    ]
    decisions = _decision_categories(rows)
    assert all(value == "pending_measurement" for value in decisions.values() if value != "inherited_kernel_parity_reference" and value != "pending_capacity_value_measurement")


def test_ep_measurement_launchers_share_compile_policy():
    from pathlib import Path

    root = Path(__file__).parents[4] / "experiments" / "qwen3_moe"
    launchers = [
        (root / "run_native_ep8_measurement.pbs").read_text(),
        (root / "run_native_ep16_measurement.pbs").read_text(),
    ]
    required = [
        "TORCHTUNE_MOE_COMPILE=${TORCHTUNE_MOE_COMPILE:-false}",
        'if [[ "${TORCHTUNE_MOE_COMPILE}" == "true" ]]',
        "compile=${TORCHTUNE_MOE_COMPILE}",
        "export TORCH_COMPILE_DISABLE=1",
        "export TORCHTUNE_EP_ALL2ALL=${TORCHTUNE_EP_ALL2ALL:-1}",
        "TORCHTUNE_MOE_OPTIMIZATION_PROFILE",
        "TORCHTUNE_MOE_ROUTING_INDEX_MODE=${TORCHTUNE_MOE_ROUTING_INDEX_MODE:-compact}",
        "TORCHTUNE_MOE_INPLACE_ROUTE_WEIGHTING=${TORCHTUNE_MOE_INPLACE_ROUTE_WEIGHTING:-1}",
        "TORCHTUNE_MOE_INPLACE_FINAL_SCATTER=${TORCHTUNE_MOE_INPLACE_FINAL_SCATTER:-1}",
        "TORCHTUNE_MOE_INPLACE_SWIGLU=${TORCHTUNE_MOE_INPLACE_SWIGLU:-1}",
        "TORCHTUNE_MOE_INDEX_SELECT_PACKING=${TORCHTUNE_MOE_INDEX_SELECT_PACKING:-1}",
        "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER=${TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER:-1}",
        '"dispatch_pack",',
        '"combine_backward_unpack",',
        "TORCHTUNE_EP_INPLACE_AG_ANCHOR=${TORCHTUNE_EP_INPLACE_AG_ANCHOR:-1}",
        "TORCHTUNE_EP_SINGLE_ROW_AG_ANCHOR=${TORCHTUNE_EP_SINGLE_ROW_AG_ANCHOR:-1}",
        "TORCHTUNE_EP_ZERO_COST_AG_ANCHOR=${TORCHTUNE_EP_ZERO_COST_AG_ANCHOR:-1}",
        "TORCHTUNE_EP_UNINITIALIZED_COLLECTIVE_BUFFERS=${TORCHTUNE_EP_UNINITIALIZED_COLLECTIVE_BUFFERS:-1}",
        "TORCHTUNE_EP_CPU_METADATA_TRANSFER=${TORCHTUNE_EP_CPU_METADATA_TRANSFER:-1}",
        "TORCHTUNE_EP_DIRECT_CPU_COPY=${TORCHTUNE_EP_DIRECT_CPU_COPY:-1}",
        "TORCHTUNE_EP_INDEX_ADD_COMBINE=${TORCHTUNE_EP_INDEX_ADD_COMBINE:-1}",
        "TORCHTUNE_MOE_BATCH_SIZE=${TORCHTUNE_MOE_BATCH_SIZE:-1}",
        "TORCHTUNE_MOE_MICROBATCH_SIZE=${TORCHTUNE_MOE_MICROBATCH_SIZE:-${TORCHTUNE_MOE_BATCH_SIZE}}",
        "TORCHTUNE_MOE_STEPS=${TORCHTUNE_MOE_STEPS:-12}",
        "batch_size=${TORCHTUNE_MOE_BATCH_SIZE}",
        "pipeline_microbatch_size=${TORCHTUNE_MOE_MICROBATCH_SIZE}",
        "TORCHTUNE_EP_GRAD_RELEASE_STREAMING=${TORCHTUNE_EP_GRAD_RELEASE_STREAMING:-0}",
        "TORCHTUNE_MOE_NATIVE_FSDP_GRAD_REDUCE=${TORCHTUNE_MOE_NATIVE_FSDP_GRAD_REDUCE:-true}",
        "require_step_timing=True",
        "require_throughput_metrics=True",
        "require_moe_metrics=True",
        '"routing_metadata_allgather", "dispatch_alltoall", "combine_alltoall", "dispatch_backward_alltoall", "combine_backward_alltoall"',
        '"routing_metadata_materialization",',
        '"routing_metadata_permutation",',
        "TORCHTUNE_MOE_WARMUP_STEPS=${TORCHTUNE_MOE_WARMUP_STEPS:-4}",
        "TORCHTUNE_MOE_MEASUREMENT_STEPS=${TORCHTUNE_MOE_MEASUREMENT_STEPS:-8}",
        "TORCHTUNE_MOE_STEADY_STATE_STEPS=${TORCHTUNE_MOE_STEADY_STATE_STEPS:-4}",
        "validate_measurement_sweep()",
        "measurement_window_exceeds_steps",
        "batch_not_divisible_by_microbatch",
    ]
    for launcher in launchers:
        for value in required:
            assert value in launcher
        assert "device_or_transport_signature" in launcher
        assert "NotPresent" in launcher
        assert "OFI[/[:space:]]*EPERM" in launcher
        assert "UR_RESULT_ERROR" in launcher
        assert "preflight_log" in launcher
        assert "reject_health_signature \"${preflight_log}\"" in launcher
    assert "expert_parallel_degree=8" in launchers[0]
    assert "data_parallel_replicate_dim=1" in launchers[0]
    assert "data_parallel_shard_dim=8" in launchers[0]
    assert "expert_parallel_degree=16" in launchers[1]
    assert "data_parallel_replicate_dim=1" in launchers[1]
    assert "data_parallel_shard_dim=16" in launchers[1]
    assert "TORCHTUNE_MOE_COLLECTIVE_LOCALITY=${TORCHTUNE_MOE_COLLECTIVE_LOCALITY:-node_local}" in launchers[0]
    assert "TORCHTUNE_MOE_COLLECTIVE_LOCALITY=${TORCHTUNE_MOE_COLLECTIVE_LOCALITY:-cross_node}" in launchers[1]


def test_moe_model_layers_expose_attention_and_non_expert_timing():
    from pathlib import Path

    qwen = (Path(__file__).parents[4] / "torchtune/models/qwen3_moe/_component_builders.py").read_text()
    gemma = (Path(__file__).parents[4] / "torchtune/models/gemma4/_component_builders.py").read_text()
    for source in (qwen, gemma):
        assert 'with _moe_timed("attention")' in source
        assert 'with _moe_timed("non_expert")' in source
    assert 'with _moe_timed("non_expert"):\n            mlp_input = self.mlp_norm(h)' in qwen
    assert 'mlp_out = self.mlp(mlp_input)' in qwen

def test_ep_scaling_manifest_declares_collective_locality_control():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    assert manifest["controls"]["collective_locality_by_topology"] == {
        "ep8": "node_local",
        "ep16": "cross_node",
    }
    assert "TORCHTUNE_MOE_COLLECTIVE_LOCALITY" in manifest["controls"][
        "controlled_overrides"
    ]
    assert manifest["artifacts"]["scaling_summary"] == (
        "torchtune.modules.moe.measurement.summarize_ep_scaling_artifact"
    )


def test_ep_scaling_manifest_declares_combine_and_final_scatter_controls():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    overrides = manifest["controls"]["controlled_overrides"]
    assert "TORCHTUNE_EP_INDEX_ADD_COMBINE" in overrides
    assert "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER" in overrides
    assert "TORCHTUNE_MOE_NATIVE_FSDP_GRAD_REDUCE" in overrides
    assert "TORCHTUNE_EP_GRAD_RELEASE_STREAMING" in overrides
    assert "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS" in overrides
    assert "TORCHTUNE_MOE_TOPK_ROUTING" in overrides
    assert "TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING" in overrides
    assert manifest["controls"]["grad_release_policy_sweep"] == [
        "native_fsdp",
        "streaming_manual",
    ]
    assert "final_scatter" in manifest["required_metrics"]
    assert {
        "expert_execution_path",
        "expert_compute_timing",
        "attention",
        "non_expert",
        "routing_metadata_allgather",
        "routing_metadata_materialization",
        "routing_metadata_permutation",
        "dispatch_pack",
        "dispatch_unpack",
        "dispatch_backward_pack",
        "dispatch_backward_unpack",
        "combine_pack",
        "combine_unpack",
        "combine_backward_pack",
        "combine_backward_unpack",
        "dispatch_backward_alltoall",
        "combine_backward_alltoall",
    }.issubset(manifest["required_metrics"])
    assert manifest["controls"]["checkpoint_experts"] is True


def test_evaluation_manifest_mirrors_ep_scaling_required_metrics():
    import pathlib
    import yaml

    manifest_path = pathlib.Path("experiments/qwen3_moe/evaluation_tracks_manifest.yaml")
    manifest = yaml.safe_load(manifest_path.read_text())
    required = set(manifest["tracks"]["ep_scaling"]["required_metrics"])
    assert {
        "routing_metadata_allgather",
        "routing_metadata_materialization",
        "routing_metadata_permutation",
        "dispatch_pack",
        "dispatch_unpack",
        "dispatch_backward_pack",
        "dispatch_backward_unpack",
        "combine_pack",
        "combine_unpack",
        "combine_backward_pack",
        "combine_backward_unpack",
    }.issubset(required)
    validate_evaluation_manifest(manifest, manifest_path=manifest_path)


def test_manifest_schema_rejects_ep_scaling_without_projection_timing():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    manifest["required_metrics"].remove("expert_compute_timing")
    with pytest.raises(ValueError, match="required_metrics"):
        validate_manifest(manifest)


def test_ep_measurement_launchers_resolve_router_profiles():
    import os
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).parents[4] / "experiments" / "qwen3_moe"
    for filename in (
        "run_native_ep8_measurement.pbs",
        "run_native_ep16_measurement.pbs",
    ):
        launcher = (root / filename).read_text()
        profile_start = launcher.index(
            'if [[ -z "${TORCHTUNE_MOE_OPTIMIZATION_PROFILE+x}" ]]; then'
        )
        profile_end = launcher.index(
            "\nexport TORCHTUNE_MOE_ROUTING_INDEX_MODE", profile_start
        )
        profile_block = launcher[profile_start:profile_end]
        script = "\n".join(
            (
                "export TORCHTUNE_MOE_TOPK_ROUTING=${TORCHTUNE_MOE_TOPK_ROUTING:-0}",
                "export TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING=${TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING:-0}",
                "export TORCHTUNE_EP_INDEX_ADD_COMBINE=${TORCHTUNE_EP_INDEX_ADD_COMBINE:-1}",
                "export TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER=${TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER:-1}",
                "export TORCHTUNE_MOE_VECTOR_PACKING=${TORCHTUNE_MOE_VECTOR_PACKING:-0}",
                "export TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE=${TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE:-1}",
                "export TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS=${TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS:-1}",
                "export TORCHTUNE_MOE_GROUPED_EXPERTS=${TORCHTUNE_MOE_GROUPED_EXPERTS:-1}",
                "export TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=${TORCHTUNE_MOE_SEQUENTIAL_EXPERTS:-0}",
                "if [[ \"${TORCHTUNE_MOE_GROUPED_EXPERTS}\" == \"1\" ]]; then export TORCHTUNE_MOE_EXECUTION_PROFILE_SUFFIX=grouped_mm; elif [[ \"${TORCHTUNE_MOE_SEQUENTIAL_EXPERTS}\" == \"1\" ]]; then export TORCHTUNE_MOE_EXECUTION_PROFILE_SUFFIX=sequential; else export TORCHTUNE_MOE_EXECUTION_PROFILE_SUFFIX=padded_bmm; fi",
                profile_block,
                'printf "%s" "$TORCHTUNE_MOE_OPTIMIZATION_PROFILE"',
            )
        )
        assert profile_block.index("TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER") >= 0
        assert "TORCHTUNE_MOE_EXECUTION_PROFILE_SUFFIX" in profile_block
        for final_scatter, combine, topk, grouping, rowwise, uninitialized in (
            ("0", "0", "0", "0", "0", "0"),
            ("1", "1", "0", "0", "1", "1"),
            ("0", "0", "1", "0", "0", "1"),
            ("1", "1", "1", "0", "1", "0"),
            ("0", "0", "0", "1", "0", "0"),
            ("1", "1", "0", "1", "1", "1"),
            ("0", "0", "1", "1", "0", "1"),
            ("1", "1", "1", "1", "1", "0"),
        ):
            environment = os.environ.copy()
            environment.pop("TORCHTUNE_MOE_OPTIMIZATION_PROFILE", None)
            environment["TORCHTUNE_MOE_TOPK_ROUTING"] = topk
            environment["TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING"] = grouping
            environment["TORCHTUNE_EP_INDEX_ADD_COMBINE"] = combine
            environment["TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER"] = final_scatter
            environment["TORCHTUNE_MOE_VECTOR_PACKING"] = "0"
            environment["TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE"] = rowwise
            environment["TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS"] = uninitialized
            result = subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                universal_newlines=True,
                env=environment,
            )
            assert (
                f"index_add_final_scatter_{'on' if final_scatter == '1' else 'off'}"
                in result.stdout
            )
            assert (
                f"index_add_combine_{'on' if combine == '1' else 'off'}"
                in result.stdout
            )
            assert (
                f"rowwise_alltoall_unpermute_{'on' if rowwise == '1' else 'off'}"
                in result.stdout
            )
            assert (
                f"uninitialized_alltoall_buffers_{'on' if uninitialized == '1' else 'off'}"
                in result.stdout
            )
            assert (
                f"topk_router_{'on' if topk == '1' else 'off'}_"
                f"unstable_grouping_{'on' if grouping == '1' else 'off'}"
                in result.stdout
            )

        environment = os.environ.copy()
        environment["TORCHTUNE_MOE_TOPK_ROUTING"] = "1"
        environment["TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING"] = "1"
        environment["TORCHTUNE_MOE_OPTIMIZATION_PROFILE"] = "caller_selected_profile"
        result = subprocess.run(
            ["bash", "-c", script],
            check=True,
            capture_output=True,
            universal_newlines=True,
            env=environment,
        )
        assert result.stdout == "caller_selected_profile"


def test_aggregate_measurement_files_is_deterministic_and_validates_ranks(tmp_path):
    import json

    def metadata(rank):
        return {
            "rank": rank,
            "source_revision": "abc",
            "uncommitted_change_state": "dirty",
            "world_size": 2,
            "ep_degree": 2,
            "global_step": 4,
            "sequence_length": 4096,
            "model": "qwen3",
            "checkpoint": "/models/qwen3",
            "batch_size": 1,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw",
            "topology": {"ep": 2, "pp": 1, "tp": 1},
            "pipeline_stage": 0,
            "environment_overrides": {"TORCHTUNE_EP_ALL2ALL": "1"},
            "optimization_profile": "test",
            "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
            "device_health": "green",
            "gate_status": "passed",
            "semantic_completion": "passed",
            "measurement_completion": "passed",
        }

    records = {
        "moe_b": {"routed_tokens": [{"total_tokens": 2}]},
        "moe_a": {"routed_tokens": [{"total_tokens": 1}]},
    }
    first = tmp_path / "rank1.json"
    second = tmp_path / "rank0.json"
    first.write_text(json.dumps({"metadata": metadata(1), "records": records}))
    second.write_text(json.dumps({"metadata": metadata(0), "records": records}))
    result = aggregate_measurement_files([first, second])
    assert result["rank_count"] == 2
    assert result["rank_files"] == [str(second), str(first)]
    assert list(result["records"]) == ["moe_a", "moe_b"]
    assert result["records"]["moe_a"]["total_tokens"] == 2
    assert result["common_metadata"]["sequence_length"] == 4096
    assert result["common_metadata"]["microbatch_size"] == 1
    assert list(result["records_by_pipeline_stage"]) == ["0"]
    assert result["collective_locality"] == {}

    with pytest.raises(ValueError, match="rank set is incomplete"):
        aggregate_measurement_files([second], require_complete_rank_set=True)
    complete = aggregate_measurement_files(
        [first, second], require_complete_rank_set=True
    )
    assert complete["rank_count"] == 2

    rank10 = tmp_path / "rank10.json"
    rank10.write_text(json.dumps({"metadata": metadata(10), "records": records}))
    result = aggregate_measurement_files([first, second, rank10])
    assert list(result["metadata_by_rank"]) == ["0", "1", "10"]

    duplicate = tmp_path / "rank0_duplicate.json"
    duplicate.write_text(json.dumps({"metadata": metadata(0), "records": records}))
    with pytest.raises(ValueError, match="duplicate measurement rank"):
        aggregate_measurement_files([second, duplicate])

    mixed = tmp_path / "mixed.json"
    mixed_metadata = metadata(2)
    mixed_metadata["source_revision"] = "different"
    mixed.write_text(json.dumps({"metadata": mixed_metadata, "records": records}))
    with pytest.raises(ValueError, match="invariant metadata"):
        aggregate_measurement_files([second, mixed])

    marked = metadata(1)
    marked["router_semantics"] = "probability_topk_v2"
    first.write_text(json.dumps({"metadata": marked, "records": records}))
    with pytest.raises(ValueError, match="invariant metadata"):
        aggregate_measurement_files([first, second])


def test_aggregate_measurement_files_requires_consistent_measurement_window(tmp_path):
    import json

    def metadata(rank, window=None):
        values = {
            "rank": rank,
            "source_revision": "abc",
            "uncommitted_change_state": "clean",
            "world_size": 2,
            "ep_degree": 2,
            "global_step": 12,
            "sequence_length": 4096,
            "model": "qwen3",
            "checkpoint": "/models/qwen3",
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw",
            "topology": {"ep": 2, "pp": 1, "tp": 1},
            "pipeline_stage": 0,
            "environment_overrides": {"TORCHTUNE_EP_ALL2ALL": "0"},
            "optimization_profile": "test",
            "routing_index_mode": "compact",
            "expert_execution_path": "padded_bmm",
            "device_health": "green",
            "gate_status": "passed",
            "semantic_completion": "passed",
            "measurement_completion": "passed",
        }
        if window is not None:
            values["measurement_window"] = window
        return values

    records = {"moe": {}}
    rank0 = tmp_path / "rank0.json"
    rank1 = tmp_path / "rank1.json"
    window = {"warmup_steps": 4, "measurement_steps": 8, "steady_state_steps": 4}
    rank0.write_text(json.dumps({"metadata": metadata(0, window), "records": records}))
    rank1.write_text(json.dumps({"metadata": metadata(1, window), "records": records}))
    result = aggregate_measurement_files([rank0, rank1])
    assert result["common_metadata"]["sequence_length"] == 4096
    assert result["common_metadata"]["measurement_window"] == window

    mismatched = dict(window)
    mismatched["warmup_steps"] = 3
    rank1.write_text(
        json.dumps({"metadata": metadata(1, mismatched), "records": records})
    )
    with pytest.raises(ValueError, match="measurement window metadata differs"):
        aggregate_measurement_files([rank0, rank1])

    rank1.write_text(json.dumps({"metadata": metadata(1), "records": records}))
    with pytest.raises(ValueError, match="measurement_window is missing"):
        aggregate_measurement_files([rank0, rank1])


def test_aggregate_measurement_files_summarizes_locality_across_ranks(tmp_path):
    import json

    def metadata(rank):
        return {
            "rank": rank,
            "source_revision": "abc",
            "uncommitted_change_state": "clean",
            "world_size": 2,
            "ep_degree": 2,
            "global_step": 4,
            "sequence_length": 4096,
            "model": "qwen3",
            "checkpoint": "/model",
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw",
            "topology": {"ep": 2, "pp": 1, "tp": 1},
            "pipeline_stage": 0,
            "environment_overrides": {
                "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "cross_node"
            },
            "optimization_profile": "test",
            "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
            "device_health": "green",
            "gate_status": "passed",
            "semantic_completion": "passed",
            "measurement_completion": "passed",
        }

    def payload(rank):
        return {
            "metadata": metadata(rank),
            "records": {
                "moe": {
                    "collectives": [
                        {
                            "name": "allgather_forward",
                            "scope": "ep",
                            "backend": "xccl",
                            "locality": "cross_node",
                            "duration_s": 1.0 + rank,
                        }
                    ]
                }
            },
        }

    paths = []
    for rank in (1, 0):
        path = tmp_path / f"rank{rank}.json"
        path.write_text(json.dumps(payload(rank)))
        paths.append(path)

    result = aggregate_measurement_files(paths)
    assert result["collective_locality"] == {
        "cross_node": {"duration_s": 3.0, "count": 2}
    }
    assert result["records"]["moe"]["collective_locality"] == {
        "cross_node": {"duration_s": 3.0, "count": 2}
    }


def test_aggregate_measurement_files_empty_schema():
    result = aggregate_measurement_files([])
    assert result == {
        "rank_count": 0,
        "rank_files": [],
        "metadata_by_rank": {},
        "common_metadata": {},
        "records": {},
        "records_by_pipeline_stage": {},
        "collective_locality": {},
        "step_timings": [],
    }


def test_aggregate_measurement_files_groups_pipeline_stages(tmp_path):
    import json

    base = {
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 2,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 2, "tp": 1},
        "environment_overrides": {},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    files = []
    for rank, stage, name in ((0, 0, "moe.0"), (1, 1, "moe.1")):
        metadata = {**base, "rank": rank, "pipeline_stage": stage}
        path = tmp_path / f"rank{rank}.json"
        path.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "records": {name: {"routed_tokens": [{"total_tokens": 1}]}},
                }
            )
        )
        files.append(path)
    result = aggregate_measurement_files(files)
    assert result["records"] == {}
    assert list(result["records_by_pipeline_stage"]) == ["0", "1"]
    assert list(result["records_by_pipeline_stage"]["0"]) == ["moe.0"]


def test_canonical_aggregation_requires_pipeline_stage_metadata(tmp_path):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "Qwen3-30B-A3B",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {"TORCHTUNE_MOE_GROUPED_EXPERTS": "1"},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "router_semantics": "probability_topk_v2",
        "expert_execution_path": "grouped_mm",
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "rank0.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {"moe": {}}}))
    with pytest.raises(ValueError, match="missing pipeline_stage"):
        aggregate_measurement_files([path], require_router_semantics=True)


def test_aggregate_measurement_files_rejects_ungated_artifacts(tmp_path):
    import json

    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
    }
    path = tmp_path / "ungated.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="device_health"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_pending_gate(tmp_path):
    import json

    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "pending",
        "semantic_completion": "passed",
        "measurement_completion": "pending",
    }
    path = tmp_path / "pending.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="gate_status"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_invalid_profile_metadata(tmp_path):
    import json

    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {"TORCHTUNE_EP_ALL2ALL": "2"},
        "optimization_profile": "",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "invalid_profile.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="optimization_profile"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_invalid_index_add_override(tmp_path):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {
            "TORCHTUNE_EP_INDEX_ADD_COMBINE": "2",
        },
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "invalid_index_add_override.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="TORCHTUNE_EP_INDEX_ADD_COMBINE"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_invalid_final_scatter_override(tmp_path):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {
            "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER": "2",
        },
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "invalid_final_scatter_override.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_invalid_vector_packing_override(tmp_path):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {"TORCHTUNE_MOE_VECTOR_PACKING": "2"},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "invalid_vector_packing_override.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="TORCHTUNE_MOE_VECTOR_PACKING"):
        aggregate_measurement_files([path])


@pytest.mark.parametrize(
    "override_name",
    [
        "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE",
        "TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS",
    ],
)
def test_aggregate_measurement_files_rejects_invalid_alltoall_override(
    tmp_path, override_name
):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {override_name: "2"},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "invalid_alltoall_override.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match=override_name):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_optimizer_attribution_mismatch(tmp_path):
    import json

    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "torch.optim.AdamW",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {
            "TORCHTUNE_EP_ALL2ALL": "0",
            "TORCHTUNE_MOE_OPTIMIZER_COMPONENT": "torchtune.dev.bioreason.optim.DeviceFactoredAdafactor",
        },
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "optimizer_mismatch.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="optimizer metadata"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_invalid_expert_execution_path(tmp_path):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "unknown_kernel",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "invalid_execution_path.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="expert_execution_path"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_missing_expert_execution_path(tmp_path):
    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {},
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "missing_execution_path.json"
    path.write_text(json.dumps({"metadata": metadata, "records": {}}))
    with pytest.raises(ValueError, match="metadata missing.*expert_execution_path"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_locality_attribution_mismatch(tmp_path):
    import json

    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "cross_node"
        },
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    records = {
        "moe": {
            "collectives": [
                {
                    "name": "allgather_forward",
                    "scope": "ep",
                    "backend": "xccl",
                    "locality": "node_local",
                    "duration_s": 1.0,
                }
            ]
        }
    }
    path = tmp_path / "locality_mismatch.json"
    path.write_text(json.dumps({"metadata": metadata, "records": records}))
    with pytest.raises(ValueError, match="locality disagrees"):
        aggregate_measurement_files([path])


def test_aggregate_measurement_files_rejects_malformed_collective_records(tmp_path):
    import json

    metadata = {
        "rank": 0,
        "source_revision": "abc",
        "uncommitted_change_state": "clean",
        "world_size": 1,
        "ep_degree": 1,
        "global_step": 2,
        "sequence_length": 128,
        "model": "qwen",
        "checkpoint": "/model",
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "adamw",
        "topology": {"ep": 1, "pp": 1, "tp": 1},
        "environment_overrides": {
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY": "cross_node"
        },
        "optimization_profile": "test",
        "routing_index_mode": "compact",
        "expert_execution_path": "padded_bmm",
        "pipeline_stage": 0,
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "malformed_collectives.json"
    path.write_text(
        json.dumps({"metadata": metadata, "records": {"moe": {"collectives": {}}}})
    )
    with pytest.raises(ValueError, match="collective records are not a list"):
        aggregate_measurement_files([path])


def test_disabled_token_record_does_not_materialize_tensor_counts(monkeypatch):
    collector = MoEMeasurementCollector(enabled=False)

    class TensorLike:
        def detach(self):
            raise AssertionError("disabled measurement must not inspect counts")

    collector.record_tokens(TensorLike())
    assert collector.record.routed_tokens == []


def test_rank_aggregation_handles_uneven_token_ownership():
    result = aggregate_rank_records(
        [
            {
                "routed_tokens": [{"total_tokens": 8}],
                "timings_s": {"dispatch": 2, "final_scatter": 0.25},
                "timing_counts": {"dispatch": 1, "final_scatter": 1},
                "collectives": [
                    {
                        "name": "dispatch_alltoall",
                        "scope": "ep",
                        "backend": "gloo",
                        "duration_s": 1.5,
                    }
                ],
            },
            {
                "routed_tokens": [{"total_tokens": 4}],
                "timings_s": {"dispatch": 3, "final_scatter": 0.5},
                "timing_counts": {"dispatch": 2, "final_scatter": 1},
                "collectives": [
                    {
                        "name": "dispatch_alltoall",
                        "scope": "ep",
                        "backend": "gloo",
                        "duration_s": 2.5,
                    }
                ],
            },
        ]
    )
    assert result["total_tokens"] == 12
    assert result["rank_token_imbalance_ratio"] == 2
    assert result["timings_s"]["dispatch"] == 5
    assert result["timing_counts"]["dispatch"] == 3
    assert result["timings_s"]["final_scatter"] == 0.75
    assert result["timing_counts"]["final_scatter"] == 2
    assert result["collectives"] == [
        {
            "name": "dispatch_alltoall",
            "scope": "ep",
            "backend": "gloo",
            "locality": "unknown",
            "duration_s": 4.0,
            "count": 2,
        }
    ]
    assert result["collective_locality"] == {
        "unknown": {"duration_s": 4.0, "count": 2}
    }


def test_rank_aggregation_preserves_gemm_and_memory_summaries():
    records = [
        {
            "grouped_gemm": [
                {
                    "model_dim": 8,
                    "hidden_dim": 16,
                    "counts": [3, 0, 2, 1],
                    "total_tokens": 6,
                    "active_expert_gemm_count": 3,
                    "zero_token_experts": 1,
                    "max_tokens_per_expert": 3,
                }
            ],
            "memory": [
                {
                    "phase": "forward",
                    "step": 4,
                    "microbatch": 0,
                    "allocated_bytes": 100,
                    "reserved_bytes": 120,
                    "free_bytes": 880,
                    "total_bytes": 1000,
                }
            ],
        },
        {
            "grouped_gemm": [
                {
                    "model_dim": 8,
                    "hidden_dim": 16,
                    "counts": [0, 4, 1, 0],
                    "total_tokens": 5,
                    "active_expert_gemm_count": 2,
                    "zero_token_experts": 2,
                    "max_tokens_per_expert": 4,
                }
            ],
            "memory": [
                {
                    "phase": "forward",
                    "step": 4,
                    "microbatch": 0,
                    "allocated_bytes": 140,
                    "reserved_bytes": 160,
                    "free_bytes": 840,
                    "total_bytes": 1000,
                }
            ],
        },
    ]
    result = aggregate_rank_records(records)
    assert result["grouped_gemm"] == [
        {
            "event_index": 0,
            "model_dim": 8,
            "hidden_dim": 16,
            "rank_count": 2,
            "total_tokens": 11,
            "active_expert_gemm_count": 5,
            "zero_token_experts": 3,
            "max_tokens_per_expert": 4,
            "counts": [3, 4, 3, 1],
        }
    ]
    assert result["memory"] == [
        {
            "phase": "forward",
            "step": 4,
            "microbatch": 0,
            "rank_count": 2,
            "max_allocated_bytes": 140,
            "max_reserved_bytes": 160,
            "min_free_bytes": 840,
            "max_total_bytes": 1000,
        }
    ]


def test_rank_aggregation_separates_global_and_local_gemm_stages():
    result = aggregate_rank_records(
        [
            {
                "grouped_gemm": [
                    {
                        "model_dim": 8,
                        "hidden_dim": 16,
                        "stage": "global_aligned",
                        "counts": [16, 16, 0, 16],
                        "routed_counts": [10, 12, 0, 8],
                        "total_tokens": 48,
                        "compute_tokens": 48,
                        "routed_tokens": 30,
                        "padding_tokens": 18,
                        "padding_fraction": 0.375,
                    },
                    {
                        "model_dim": 8,
                        "hidden_dim": 16,
                        "stage": "local_compute",
                        "counts": [16, 16],
                        "total_tokens": 32,
                        "compute_tokens": 32,
                    },
                ]
            }
        ]
    )
    assert [item["stage"] for item in result["grouped_gemm"]] == [
        "global_aligned",
        "local_compute",
    ]
    assert result["grouped_gemm"][0]["padding_tokens"] == 18
    assert result["grouped_gemm"][0]["padding_fraction"] == 0.375
    assert result["grouped_gemm"][0]["compute_to_routed_ratio"] == 48 / 30
    assert result["grouped_gemm"][1]["counts"] == [16, 16]


def test_rank_aggregation_preserves_padded_bmm_work():
    result = aggregate_rank_records(
        [
            {
                "grouped_gemm": [
                    {
                        "stage": "padded_bmm",
                        "model_dim": 8,
                        "hidden_dim": 16,
                        "counts": [3, 12, 0],
                        "total_tokens": 15,
                        "max_count": 12,
                        "dense_compute_tokens": 36,
                        "routed_tokens": 15,
                        "padding_tokens": 21,
                    }
                ]
            }
        ]
    )
    stats = result["grouped_gemm"][0]
    assert stats["stage"] == "padded_bmm"
    assert stats["dense_compute_tokens"] == 36
    assert stats["dense_padding_tokens"] == 21
    assert stats["dense_to_routed_ratio"] == 36 / 15


def test_manifest_schema_requires_completion_metadata():
    manifest = {
        "result_class": "kernel_parity",
        "source_revision": "abc",
        "model": "qwen",
        "hardware": {"nodes": 2},
        "topology": {"ep": 16, "pp": 1},
        "sequence_length": 4096,
        "batch_size": 1,
        "optimizer": "adamw",
        "measurement_window": {"warmup_steps": 2, "measurement_steps": 8},
        "environment": {},
        "controls": {
            "promotion_evaluator": "report_kernel_parity.py",
            "matched_sequence_length": True,
            "matched_optimizer_policy": True,
            "matched_measurement_window": True,
        },
        "decision_categories": {
            "highest_repeatable_moe_per_gpu_mfu": "pending",
            "highest_aggregate_throughput": "pending",
            "best_ep_scaling_point": "pending",
            "closest_strict_dense_parity": "pending",
            "best_capacity_value_result": "pending",
        },
        "gap_attribution": {
            "communication": "pending",
            "expert_compute": "pending",
            "attention": "pending",
            "optimizer": "pending",
            "pipeline_bubble": "pending",
        },
        "completion": {
            "semantic_completion": True,
            "device_health": "green",
            "status": "pending",
        },
    }
    validate_manifest(manifest)
    with pytest.raises(ValueError, match="result_class"):
        validate_manifest({**manifest, "result_class": "parity"})


def test_manifest_schema_rejects_incomplete_ep_scaling_controls():
    manifest = {
        "result_class": "ep_scaling",
        "source_revision": "abc",
        "model": "qwen",
        "hardware": {},
        "topology": {},
        "sequence_length": 4096,
        "batch_size": 1,
        "optimizer": "adamw",
        "measurement_window": {"warmup_steps": 2, "measurement_steps": 8},
        "environment": {},
        "decision_categories": {key: "pending" for key in (
            "highest_repeatable_moe_per_gpu_mfu",
            "highest_aggregate_throughput",
            "best_ep_scaling_point",
            "closest_strict_dense_parity",
            "best_capacity_value_result",
        )},
        "gap_attribution": {key: "pending" for key in (
            "communication", "expert_compute", "attention", "optimizer", "pipeline_bubble"
        )},
        "completion": {"semantic_completion": "pending", "device_health": "pending", "status": "pending"},
        "controls": {
            "ep_degrees": [8, 16],
            "local_batch_sweep": [1, 2],
            "grad_release_policy_sweep": ["native_fsdp", "streaming_manual"],
                "gradient_accumulation_is_proxy": False,
                "synthetic_token_volume_diagnostic": {
                    "driver": "experiments/qwen3_moe/benchmark_routed_token_volume.py",
                    "promotion_artifact": False,
                },
                "controlled_overrides": [
                    "TORCHTUNE_EP_INDEX_ADD_COMBINE",
                    "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER",
                    "TORCHTUNE_MOE_NATIVE_FSDP_GRAD_REDUCE",
                        "TORCHTUNE_EP_GRAD_RELEASE_STREAMING",
                        "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS",
                        "TORCHTUNE_MOE_TOPK_ROUTING",
                        "TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING",
                    ],
        },
        "artifacts": {"required_execution_path": "controlled_by_launcher"},
        "required_gates": [],
        "required_metrics": [],
    }
    with pytest.raises(ValueError, match="required_gates"):
        validate_manifest(manifest)


@pytest.mark.parametrize("local_batch_sweep", [[1], [2, 1], [1, 1], [0, 1]])
def test_manifest_schema_rejects_invalid_ep_scaling_batch_sweep(local_batch_sweep):
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    manifest["controls"]["local_batch_sweep"] = local_batch_sweep
    with pytest.raises(ValueError, match="local_batch_sweep"):
        validate_manifest(manifest)


def test_manifest_schema_rejects_ep_scaling_without_final_scatter_metric():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    manifest["required_metrics"].remove("final_scatter")
    with pytest.raises(ValueError, match="required_metrics"):
        validate_manifest(manifest)


def test_manifest_schema_requires_grouped_gemm_projection_metrics():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    for metric in ("grouped_gemm_gate", "grouped_gemm_up", "grouped_gemm_down"):
        reduced = yaml.safe_load(
            pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
        )
        reduced["required_metrics"].remove(metric)
        with pytest.raises(ValueError, match="required_metrics"):
            validate_manifest(reduced)
    validate_manifest(manifest)


def test_ep_scaling_manifest_declares_nonpromotable_token_volume_diagnostic():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    diagnostic = manifest["controls"]["synthetic_token_volume_diagnostic"]
    assert diagnostic["driver"].endswith("benchmark_routed_token_volume.py")
    assert diagnostic["promotion_artifact"] is False
    assert set(diagnostic["output_contract"]) >= {
        "rows",
        "grouped_gemm",
        "volume_scaling",
        "promotion_artifact_false",
    }
    validate_manifest(manifest)


def test_routed_token_volume_diagnostic_rejects_invalid_inputs():
    import importlib.util
    import pathlib

    path = pathlib.Path("experiments/qwen3_moe/benchmark_routed_token_volume.py")
    spec = importlib.util.spec_from_file_location("routed_token_volume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.argparse.ArgumentParser()
    parser.add_argument("--volumes", nargs="+", type=int, default=[4096])
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args([])
    assert args.volumes == [4096]
    with pytest.raises(ValueError, match="non-negative"):
        module.synthetic_expert_token_counts(-1, args.experts)
    with pytest.raises(ValueError, match="at least one"):
        module.synthetic_expert_token_counts(1, 0)


def test_routed_token_volume_diagnostic_declares_execution_path_controls():
    import importlib.util
    import pathlib

    path = pathlib.Path("experiments/qwen3_moe/benchmark_routed_token_volume.py")
    spec = importlib.util.spec_from_file_location("routed_token_volume_paths", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.argparse.ArgumentParser()
    parser.add_argument(
        "--execution-path",
        choices=("auto", "grouped_mm", "expert_loop", "both"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=17)
    assert parser.parse_args([]).execution_path == "auto"
    assert parser.parse_args(["--seed", "23"]).seed == 23
    assert parser.parse_args(["--execution-path", "both"]).execution_path == "both"
    with pytest.raises(SystemExit):
        parser.parse_args(["--execution-path", "invalid"])


def test_routed_token_volume_equivalence_metrics_cover_shape_and_tolerance():
    import importlib.util
    import pathlib

    path = pathlib.Path("experiments/qwen3_moe/benchmark_routed_token_volume.py")
    spec = importlib.util.spec_from_file_location("routed_token_volume_equivalence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = torch.ones(2, 3)
    candidate = reference + 5e-5
    result = module._equivalence_metrics(reference, candidate, dtype=torch.float32)
    assert result["passed"] is True
    assert result["max_absolute_error"] > 0
    bf16_result = module._equivalence_metrics(
        reference.to(torch.bfloat16),
        (reference + 1e-2).to(torch.bfloat16),
        dtype=torch.bfloat16,
    )
    assert bf16_result["atol"] == 5e-2
    assert bf16_result["rtol"] == 5e-2
    mismatch = module._equivalence_metrics(
        reference, torch.ones(1, 3), dtype=torch.float32
    )
    assert mismatch["passed"] is False
    assert mismatch["max_absolute_error"] == float("inf")


def test_routed_token_volume_paired_summary_defines_speedup_direction():
    import importlib.util
    import pathlib

    path = pathlib.Path("experiments/qwen3_moe/benchmark_routed_token_volume.py")
    spec = importlib.util.spec_from_file_location("routed_token_volume_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"total_tokens": 8, "execution_path": "expert_loop", "mean_seconds": 4.0},
        {"total_tokens": 8, "execution_path": "grouped_mm", "mean_seconds": 2.0},
    ]
    grouped = next(row for row in rows if row["execution_path"] == "grouped_mm")
    loop = next(row for row in rows if row["execution_path"] == "expert_loop")
    grouped["speedup_vs_expert_loop"] = loop["mean_seconds"] / grouped["mean_seconds"]
    assert grouped["speedup_vs_expert_loop"] == 2.0


def test_routed_token_volume_summary_reports_volume_scaling():
    import importlib.util
    import pathlib

    path = pathlib.Path("experiments/qwen3_moe/benchmark_routed_token_volume.py")
    spec = importlib.util.spec_from_file_location("routed_token_volume_scaling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarize_volume_scaling(
        [
            {
                "execution_path": "grouped_mm",
                "total_tokens": 100,
                "mean_seconds": 2.0,
                "grouped_gemm": {"mean_tokens_per_expert": 10.0},
            },
            {
                "execution_path": "grouped_mm",
                "total_tokens": 200,
                "mean_seconds": 3.0,
                "grouped_gemm": {"mean_tokens_per_expert": 20.0},
            },
        ]
    )
    result = summary["by_execution_path"]["grouped_mm"]
    assert summary["promotion_artifact"] is False
    assert result["throughput_gain_ratio"] == pytest.approx(4 / 3)
    assert result["seconds_per_token_gain_ratio"] == pytest.approx(0.75)
    assert result["first_tokens_per_expert"] == 10.0
    assert result["last_tokens_per_expert"] == 20.0
    assert result["interpretation"] == "throughput_scales_with_volume"


def test_manifest_schema_rejects_ep_scaling_without_device_health_metric():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    manifest["required_metrics"].remove("device_health")
    with pytest.raises(ValueError, match="required_metrics"):
        validate_manifest(manifest)


def test_manifest_schema_rejects_ep_scaling_without_execution_path():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    del manifest["artifacts"]["required_execution_path"]
    with pytest.raises(ValueError, match="controlled_by_launcher execution path"):
        validate_manifest(manifest)


def test_manifest_schema_rejects_ep_scaling_without_ab_controls():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    manifest["controls"]["controlled_overrides"].remove(
        "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER"
    )
    with pytest.raises(ValueError, match="controlled_overrides"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "missing_control",
    [
        "TORCHTUNE_MOE_TOPK_ROUTING",
        "TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING",
    ],
)
def test_manifest_schema_rejects_ep_scaling_without_router_ab_control(missing_control):
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/ep_scaling_manifest.yaml").read_text()
    )
    manifest["controls"]["controlled_overrides"].remove(missing_control)
    with pytest.raises(ValueError, match="controlled_overrides"):
        validate_manifest(manifest)


def test_capacity_manifest_requires_executable_capacity_only_control():
    import pathlib
    import yaml

    path = pathlib.Path("experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml")
    manifest = yaml.safe_load(path.read_text())
    validate_manifest(manifest)
    manifest["controls"]["dense_control"]["comparison_label"] = "strict_parity"
    with pytest.raises(ValueError, match="capacity_value_only"):
        validate_manifest(manifest)


def test_capacity_manifest_declares_alltoall_contiguity_control():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path(
            "experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml"
        ).read_text()
    )
    assert manifest["environment"][
        "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS"
    ] == "1"


def test_manifest_path_validation_checks_declared_configuration_paths():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path(
            "experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml"
        ).read_text()
    )
    manifest["controls"]["dense_control"]["configuration"] = (
        "recipes/configs/dev/production/missing_capacity_config.yaml"
    )
    with pytest.raises(ValueError, match="configuration=recipes/configs/dev/production/missing_capacity_config.yaml"):
        validate_manifest(manifest)


def test_manifest_path_validation_rejects_missing_executable_path():
    import pathlib
    import yaml

    path = pathlib.Path("experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml")
    manifest = yaml.safe_load(path.read_text())
    manifest["controls"]["moe_launcher"] = "experiments/qwen3_moe/missing.pbs"
    with pytest.raises(ValueError, match="missing repository paths"):
        validate_manifest(manifest)


def test_capacity_manifest_requires_dense_measurement_provenance():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml").read_text()
    )
    for field in (
        "measurement_configuration",
        "measurement_launcher",
        "measurement_artifact",
    ):
        manifest["controls"]["dense_control"].pop(field)
        with pytest.raises(ValueError, match=f"dense_control must record {field}"):
            validate_manifest(manifest)
        manifest = yaml.safe_load(
            pathlib.Path("experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml").read_text()
        )


def test_capacity_dense_log_parser_discards_warmup_and_aggregates(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import parse_dense_metric_log

    path = tmp_path / "dense.log"
    path.write_text(
        "Step 1 | loss:10 time_per_step_s:10 tokens_per_second_per_gpu:10 peak_memory_reserved:20\n"
        "Step 2 | loss:5 time_per_step_s:5 tokens_per_second_per_gpu:20 peak_memory_reserved:30\n"
        "Step 3 | loss:4 time_per_step_s:4 tokens_per_second_per_gpu:30 peak_memory_reserved:25\n"
    )
    result = parse_dense_metric_log(path, warmup_steps=1, world_size=16)
    assert result["sample_count"] == 2
    assert result["per_gpu_throughput"] == 25.0
    assert result["aggregate_throughput"] == 400.0
    assert result["peak_memory_reserved"] == 30.0
    assert result["steady_state_memory_reserved"] == 25.0


def test_capacity_dense_log_parser_rejects_missing_window(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import parse_dense_metric_log

    path = tmp_path / "dense.log"
    path.write_text("Step 1 | loss:1 time_per_step_s:1 tokens_per_second_per_gpu:1\n")
    with pytest.raises(ValueError, match="no post-warmup"):
        parse_dense_metric_log(path, warmup_steps=1, world_size=16)


def test_capacity_dense_result_rejects_missing_peak_memory(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_dense_log

    path = tmp_path / "dense.log"
    path.write_text("Step 1 | loss:1 time_per_step_s:1 tokens_per_second_per_gpu:16\n")
    with pytest.raises(ValueError, match="peak_memory_reserved"):
        result_from_dense_log(
            path,
            model="dense",
            total_parameters=31.0,
            active_parameters_per_token=31.0,
            topology={"nodes": 2, "world_size": 16},
            sequence_length=4096,
            world_size=16,
            warmup_steps=0,
            mfu_percent=1.0,
            active_flop_efficiency=1.0,
            communication_fraction=0.0,
            expert_compute_fraction=0.0,
        )


def test_capacity_dense_artifact_parser_preserves_provenance(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_dense_artifact

    path = tmp_path / "dense.json"
    path.write_text(
        __import__("json").dumps(
            {
                "result_class": "capacity_value_dense_control",
                "model": "Gemma4-31B",
                "checkpoint": "/models/gemma-4-31B",
                "total_parameters": 31.0,
                "active_parameters_per_token": 31.0,
                "topology": {"nodes": 2, "world_size": 16},
                "sequence_length": 4096,
                "batch_size": 1,
                "microbatch_size": 1,
                "gradient_accumulation_steps": 1,
                "optimizer": "torch.optim.AdamW",
                "environment_overrides": {"TORCH_COMPILE_DISABLE": "1"},
                "source_revision": "abc123",
                "uncommitted_change_state": "dirty",
                "measurement_window": {
                    "warmup_steps": 4,
                    "measurement_steps": 2,
                    "retained_step_start": 5,
                    "retained_step_end": 6,
                },
                "records": [
                    {"step": 5, "loss": 1.0, "time": 2.0, "throughput": 10.0, "memory": 20.0},
                    {"step": 6, "loss": 0.9, "time": 2.0, "throughput": 12.0, "memory": 21.0},
                ],
                "device_health": "green",
                "gate_status": "passed",
                "semantic_completion": "passed",
                "measurement_completion": "passed",
            }
        )
    )
    result = result_from_dense_artifact(
        path,
        active_flop_efficiency=0.2,
        mfu_percent=1.0,
        communication_fraction=0.1,
        expert_compute_fraction=0.0,
    )
    assert result["per_gpu_throughput"] == 11.0
    assert result["aggregate_throughput"] == 176.0
    assert result["total_parameters"] == 31.0
    assert result["active_parameters_per_token"] == 31.0
    assert result["steady_state_memory"]["max_reserved_gib"] == 21.0
    assert result["source_revision"] == "abc123"
    assert result["uncommitted_change_state"] == "dirty"


def test_capacity_result_from_ep_scaling_report(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import (
        result_from_ep_scaling_report,
    )

    path = tmp_path / "ep.json"
    path.write_text(
        __import__("json").dumps(
            {
                "result_class": "ep_scaling",
                "ep_summaries": {
                    "16": {
                        "metadata": {
                            "model": "Gemma4-26B-A4B",
                            "router_semantics": "sigmoid_argsort_v1",
                            "checkpoint": "/models/gemma-4-26B-A4B",
                            "source_revision": "abc123",
                            "uncommitted_change_state": "dirty",
                            "world_size": 16,
                            "sequence_length": 4096,
                            "batch_size": 1,
                            "microbatch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "optimizer": "torch.optim.AdamW",
                            "environment_overrides": {},
                            "gate_status": "passed",
                            "device_health": "green",
                            "semantic_completion": "passed",
                            "measurement_completion": "passed",
                        },
                        "model_metrics": {
                            "total_parameters": 26.0,
                            "active_parameters_per_token": 4.0,
                        },
                        "throughput": {
                            "tokens_per_second_per_gpu": 100.0,
                            "aggregate_tokens_per_second": 1600.0,
                        },
                        "peak_memory": {"max_reserved_bytes": 100},
                        "scaling_efficiency": 0.625,
                    }
                },
            }
        )
    )
    result = result_from_ep_scaling_report(
        path,
        ep_degree=16,
        nodes=2,
        active_flop_efficiency=0.2,
        mfu_percent=1.0,
        communication_fraction=0.4,
        expert_compute_fraction=0.3,
    )
    assert result["model"] == "Gemma4-26B-A4B"
    assert result["aggregate_throughput"] == 1600.0
    assert result["topology"] == {"nodes": 2, "world_size": 16, "ep": 16}
    assert result["scaling_efficiency"] == 0.625


def test_capacity_result_from_ep_scaling_report_keeps_missing_efficiency_pending(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_ep_scaling_report

    path = tmp_path / "ep.json"
    path.write_text(
        __import__("json").dumps(
            {
                "result_class": "ep_scaling",
                "ep_summaries": {
                    "16": {
                        "metadata": {
                            "model": "Gemma4-26B-A4B",
                            "router_semantics": "sigmoid_argsort_v1",
                            "checkpoint": "/models/gemma4",
                            "source_revision": "abc123",
                            "world_size": 16,
                            "sequence_length": 4096,
                            "batch_size": 1,
                            "microbatch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "optimizer": "torch.optim.AdamW",
                            "environment_overrides": {},
                            "gate_status": "passed",
                            "device_health": "green",
                            "semantic_completion": "passed",
                            "measurement_completion": "passed",
                        },
                        "model_metrics": {
                            "total_parameters": 26.0,
                            "active_parameters_per_token": 4.0,
                        },
                        "throughput": {
                            "tokens_per_second_per_gpu": 100.0,
                            "aggregate_tokens_per_second": 1600.0,
                        },
                    }
                },
            }
        )
    )
    result = result_from_ep_scaling_report(
        path,
        ep_degree=16,
        nodes=2,
        active_flop_efficiency=0.2,
        mfu_percent=1.0,
        communication_fraction=0.4,
        expert_compute_fraction=0.3,
    )
    assert result["scaling_efficiency"] is None


def test_capacity_result_from_ep_scaling_report_derives_timing_fractions(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_ep_scaling_report

    path = tmp_path / "ep.json"
    path.write_text(
        __import__("json").dumps(
            {
                "result_class": "ep_scaling",
                "ep_summaries": {
                    "16": {
                        "metadata": {
                            "model": "Gemma4-26B-A4B",
                            "router_semantics": "sigmoid_argsort_v1",
                            "checkpoint": "/models/gemma4",
                            "source_revision": "abc123",
                            "world_size": 16,
                            "sequence_length": 4096,
                            "batch_size": 1,
                            "microbatch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "optimizer": "torch.optim.AdamW",
                            "environment_overrides": {},
                            "gate_status": "passed",
                            "device_health": "green",
                            "semantic_completion": "passed",
                            "measurement_completion": "passed",
                        },
                        "model_metrics": {
                            "total_parameters": 26.0,
                            "active_parameters_per_token": 4.0,
                        },
                        "throughput": {
                            "tokens_per_second_per_gpu": 100.0,
                            "aggregate_tokens_per_second": 1600.0,
                        },
                        "phase_timings": {
                            "mean_total_step_s": 10.0,
                            "phase_mean_s": {},
                        },
                        "communication": {
                            "by_locality": {
                                "cross_node": {"duration_s": 2.0}
                            }
                        },
                        "expert_compute": {
                            "timing": {"total_seconds": 3.0}
                        },
                    }
                },
            }
        )
    )
    result = result_from_ep_scaling_report(
        path,
        ep_degree=16,
        nodes=2,
        active_flop_efficiency=0.2,
        mfu_percent=1.0,
        communication_fraction=None,
        expert_compute_fraction=None,
    )
    assert result["communication_fraction"] == 0.2
    assert result["expert_compute_fraction"] == 0.3
    assert result["fraction_provenance"] == {
        "communication": "sealed_phase_timing",
        "expert compute": "sealed_phase_timing",
    }


def test_capacity_result_from_ep_scaling_report_reads_comparison_row(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_ep_scaling_report

    path = tmp_path / "ep.json"
    path.write_text(
        __import__("json").dumps(
            {
                "result_class": "ep_scaling",
                "ep_summaries": {
                    "16": {
                        "metadata": {
                            "model": "Gemma4-26B-A4B",
                            "router_semantics": "sigmoid_argsort_v1",
                            "checkpoint": "/models/gemma4",
                            "source_revision": "abc123",
                            "world_size": 16,
                            "sequence_length": 4096,
                            "batch_size": 1,
                            "microbatch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "optimizer": "torch.optim.AdamW",
                            "environment_overrides": {},
                            "gate_status": "passed",
                            "device_health": "green",
                            "semantic_completion": "passed",
                            "measurement_completion": "passed",
                        },
                        "model_metrics": {
                            "total_parameters": 26.0,
                            "active_parameters_per_token": 4.0,
                        },
                        "throughput": {
                            "tokens_per_second_per_gpu": 100.0,
                            "aggregate_tokens_per_second": 1600.0,
                        },
                    }
                },
                "comparison": {"rows": [{"ep_degree": 16, "scaling_efficiency": 0.75}]},
            }
        )
    )
    result = result_from_ep_scaling_report(
        path,
        ep_degree=16,
        nodes=2,
        active_flop_efficiency=0.2,
        mfu_percent=1.0,
        communication_fraction=0.4,
        expert_compute_fraction=0.3,
    )
    assert result["scaling_efficiency"] == 0.75


def test_capacity_markdown_reports_scaling_efficiency_as_pending_for_dense():
    from experiments.qwen3_moe.report_capacity_value import format_markdown

    report = {
        "rows": {
            "moe": {"scaling_efficiency": 0.625},
            "dense": {},
        },
        "comparison_metrics": {},
        "pipeline": {"moe": {}, "dense": {}},
    }
    markdown = format_markdown(report)
    assert "EP scaling efficiency" in markdown
    assert "0.6250" in markdown
    assert markdown.count("pending") >= 1


def test_capacity_markdown_renders_decisions_and_gaps():
    from experiments.qwen3_moe.report_capacity_value import format_markdown

    markdown = format_markdown(
        {
            "rows": {"moe": {}, "dense": {}},
            "comparison_metrics": {},
            "pipeline": {"moe": {}, "dense": {}},
            "decision_categories": {"best_capacity_value_result": "pending_measurement"},
            "gap_attribution": {
                "attention": {"status": "pending", "evidence": "not timed"}
            },
        }
    )
    assert "## Decision Categories" in markdown
    assert "pending_measurement" in markdown
    assert "| attention | pending | not timed |" in markdown


def test_capacity_markdown_renders_fraction_provenance():
    from experiments.qwen3_moe.report_capacity_value import format_markdown

    markdown = format_markdown(
        {
            "rows": {
                "moe": {
                    "fraction_provenance": {
                        "communication": "sealed_phase_timing",
                        "expert compute": "explicit_override",
                    }
                },
                "dense": {
                    "fraction_provenance": {
                        "communication": "explicit_override",
                        "expert compute": "explicit_override",
                    }
                },
            },
            "comparison_metrics": {},
            "pipeline": {"moe": {}, "dense": {}},
        }
    )
    assert "## Fraction Provenance" in markdown
    assert "| moe | sealed_phase_timing | explicit_override |" in markdown
    assert "| dense | explicit_override | explicit_override |" in markdown


def test_compare_capacity_value_results_reports_measured_fraction_provenance():
    def result(model, provenance):
        return {
            "model": model,
            "total_parameters": 10.0,
            "active_parameters_per_token": 3.0,
            "sequence_length": 4096,
            "topology": {"nodes": 2, "world_size": 16},
            "per_gpu_throughput": 100.0,
            "aggregate_throughput": 1600.0,
            "mfu_percent": 3.0,
            "active_flop_efficiency": 0.2,
            "peak_memory": {"max_reserved_gib": 10.0},
            "communication_fraction": 0.1,
            "expert_compute_fraction": 0.4,
            "fraction_provenance": provenance,
            "stability": "passed",
        }

    comparison = compare_capacity_value_results(
        result(
            "Gemma4-26B-A4B",
            {
                "communication": "sealed_phase_timing",
                "expert compute": "sealed_phase_timing",
            },
        ),
        result(
            "Gemma4-31B",
            {
                "communication": "sealed_phase_timing",
                "expert compute": "sealed_phase_timing",
            },
        ),
    )
    assert comparison["gap_attribution"]["communication"]["evidence"] == (
        "sealed phase timing for both controls"
    )


def test_capacity_result_from_ep_scaling_report_rejects_placeholder_metadata(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import (
        result_from_ep_scaling_report,
    )

    path = tmp_path / "ep.json"
    path.write_text(
        __import__("json").dumps(
            {
                "result_class": "ep_scaling",
                "ep_summaries": {
                    "16": {
                        "metadata": {
                            "model": "unknown",
                            "checkpoint": "/models/gemma",
                            "source_revision": "abc",
                            "world_size": 16,
                            "sequence_length": 4096,
                            "batch_size": 1,
                            "microbatch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "optimizer": "torch.optim.AdamW",
                            "environment_overrides": {},
                            "gate_status": "passed",
                            "device_health": "green",
                            "semantic_completion": "passed",
                            "measurement_completion": "passed",
                        },
                        "model_metrics": {
                            "total_parameters": 26.0,
                            "active_parameters_per_token": 4.0,
                        },
                        "throughput": {
                            "tokens_per_second_per_gpu": 100.0,
                            "aggregate_tokens_per_second": 1600.0,
                        },
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="model must be non-placeholder"):
        result_from_ep_scaling_report(
            path,
            ep_degree=16,
            nodes=2,
            active_flop_efficiency=0.2,
            mfu_percent=1.0,
            communication_fraction=0.4,
            expert_compute_fraction=0.3,
        )


def test_capacity_result_from_ep_artifacts_uses_sealed_summary(monkeypatch, tmp_path):
    from experiments.qwen3_moe import report_capacity_value

    artifact = {
        "common_metadata": {
            "ep_degree": 16,
            "world_size": 16,
            "model": "Gemma4-26B-A4B",
            "router_semantics": "sigmoid_argsort_v1",
            "checkpoint": "/models/gemma-4-26B-A4B",
            "source_revision": "abc123",
            "uncommitted_change_state": "dirty",
            "sequence_length": 4096,
            "batch_size": 1,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "torch.optim.AdamW",
            "environment_overrides": {},
            "gate_status": "passed",
            "device_health": "green",
            "semantic_completion": "passed",
            "measurement_completion": "passed",
            "measurement_window": {"warmup_steps": 4},
        }
    }
    summary = {
        "throughput": {
            "tokens_per_second_per_gpu": 100.0,
            "aggregate_tokens_per_second": 1600.0,
        },
        "peak_memory": {"max_reserved_bytes": 100},
    }
    monkeypatch.setattr(
        report_capacity_value,
        "aggregate_measurement_files",
        lambda paths, require_complete_rank_set, require_router_semantics: artifact,
    )
    monkeypatch.setattr(
        report_capacity_value,
        "summarize_ep_scaling_artifact",
        lambda artifact, warmup_steps: summary,
    )
    result = report_capacity_value.result_from_ep_artifacts(
        [tmp_path / "rank0.json"],
        ep_degree=16,
        nodes=2,
        total_parameters=26.0,
        active_parameters_per_token=4.0,
        active_flop_efficiency=0.2,
        mfu_percent=1.0,
        communication_fraction=0.4,
        expert_compute_fraction=0.3,
    )
    assert result["per_gpu_throughput"] == 100.0
    assert result["model"] == "Gemma4-26B-A4B"


def test_capacity_cli_rejects_multiple_moe_sources():
    import subprocess

    result = subprocess.run(
        [
            "python3",
            "experiments/qwen3_moe/report_capacity_value.py",
            "--moe",
            "moe.json",
            "--moe-ep-scaling-report",
            "ep.json",
            "--output",
            "out.json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


@pytest.mark.parametrize("field", ["model", "checkpoint", "source_revision"])
def test_capacity_dense_artifact_rejects_placeholder_provenance(tmp_path, field):
    from experiments.qwen3_moe.report_capacity_value import result_from_dense_artifact

    artifact = {
        "result_class": "capacity_value_dense_control",
        "model": "Gemma4-31B",
        "checkpoint": "/models/gemma-4-31B",
        "total_parameters": 31.0,
        "active_parameters_per_token": 31.0,
        "topology": {"nodes": 2, "world_size": 16},
        "sequence_length": 4096,
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "torch.optim.AdamW",
        "environment_overrides": {"TORCH_COMPILE_DISABLE": "1"},
        "source_revision": "abc123",
        "uncommitted_change_state": "dirty",
        "measurement_window": {
            "warmup_steps": 0,
            "measurement_steps": 1,
            "retained_step_start": 1,
            "retained_step_end": 1,
        },
        "records": [
            {"step": 1, "loss": 1.0, "time": 2.0, "throughput": 10.0, "memory": 20.0}
        ],
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    artifact[field] = "unknown"
    path = tmp_path / "dense.json"
    path.write_text(__import__("json").dumps(artifact))
    with pytest.raises(ValueError, match=f"dense artifact {field}"):
        result_from_dense_artifact(
            path,
            active_flop_efficiency=0.2,
            mfu_percent=1.0,
            communication_fraction=0.1,
            expert_compute_fraction=0.0,
        )


@pytest.mark.parametrize("missing_field", [
    "total_parameters",
    "active_parameters_per_token",
])
def test_capacity_dense_artifact_requires_parameter_metadata(tmp_path, missing_field):
    from experiments.qwen3_moe.report_capacity_value import result_from_dense_artifact

    artifact = {
        "result_class": "capacity_value_dense_control",
        "model": "Gemma4-31B",
        "checkpoint": "/models/gemma-4-31B",
        "total_parameters": 31.0,
        "active_parameters_per_token": 31.0,
        "topology": {"nodes": 2, "world_size": 16},
        "sequence_length": 4096,
        "source_revision": "abc123",
        "uncommitted_change_state": "dirty",
        "measurement_window": {
            "warmup_steps": 0,
            "measurement_steps": 1,
            "retained_step_start": 1,
            "retained_step_end": 1,
        },
        "records": [
            {"step": 1, "loss": 1.0, "time": 2.0, "throughput": 10.0, "memory": 20.0}
        ],
        "device_health": "green",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    del artifact[missing_field]
    path = tmp_path / "dense.json"
    path.write_text(__import__("json").dumps(artifact))
    with pytest.raises(ValueError, match="required metadata"):
        result_from_dense_artifact(
            path,
            active_flop_efficiency=0.2,
            mfu_percent=1.0,
            communication_fraction=0.1,
            expert_compute_fraction=0.0,
        )


def test_capacity_dense_artifact_requires_optimizer_metadata(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_dense_artifact

    artifact = {
        "model": "Gemma4-31B",
        "checkpoint": "/models/gemma-4-31B",
        "total_parameters": 31.0,
        "active_parameters_per_token": 31.0,
        "topology": {"nodes": 2, "world_size": 16},
        "sequence_length": 4096,
        "batch_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 1,
        "environment_overrides": {},
        "source_revision": "abc123",
        "uncommitted_change_state": "dirty",
        "measurement_window": {
            "warmup_steps": 0,
            "measurement_steps": 1,
            "retained_step_start": 1,
            "retained_step_end": 1,
        },
        "records": [
            {"step": 1, "loss": 1.0, "time": 2.0, "throughput": 10.0, "memory": 20.0}
        ],
        "device_health": "green",
        "gate_status": "passed",
        "semantic_completion": "passed",
        "measurement_completion": "passed",
    }
    path = tmp_path / "dense.json"
    path.write_text(__import__("json").dumps(artifact))
    with pytest.raises(ValueError, match="required metadata.*optimizer"):
        result_from_dense_artifact(
            path,
            active_flop_efficiency=0.2,
            mfu_percent=1.0,
            communication_fraction=0.1,
            expert_compute_fraction=0.0,
        )


def test_capacity_dense_result_requires_explicit_sequence_and_topology(tmp_path):
    from experiments.qwen3_moe.report_capacity_value import result_from_dense_log

    path = tmp_path / "dense.log"
    path.write_text(
        "Step 1 | loss:1 time_per_step_s:1 tokens_per_second_per_gpu:16 "
        "peak_memory_reserved:20\n"
    )
    result = result_from_dense_log(
        path,
        model="dense",
        total_parameters=31.0,
        active_parameters_per_token=31.0,
        topology={"nodes": 2, "world_size": 16},
        sequence_length=4096,
        world_size=16,
        warmup_steps=0,
        mfu_percent=1.0,
        active_flop_efficiency=1.0,
        communication_fraction=0.0,
        expert_compute_fraction=0.0,
    )
    assert result["sequence_length"] == 4096
    assert result["topology"]["nodes"] == 2
    with pytest.raises(ValueError, match="sequence_length"):
        result_from_dense_log(
            path,
            model="dense",
            total_parameters=31.0,
            active_parameters_per_token=31.0,
            topology={"nodes": 2, "world_size": 16},
            sequence_length=0,
            world_size=16,
            warmup_steps=0,
            mfu_percent=1.0,
            active_flop_efficiency=1.0,
            communication_fraction=0.0,
            expert_compute_fraction=0.0,
        )
    with pytest.raises(ValueError, match="topology world_size"):
        result_from_dense_log(
            path,
            model="dense",
            total_parameters=31.0,
            active_parameters_per_token=31.0,
            topology={"nodes": 2, "world_size": 8},
            sequence_length=4096,
            world_size=16,
            warmup_steps=0,
            mfu_percent=1.0,
            active_flop_efficiency=1.0,
            communication_fraction=0.0,
            expert_compute_fraction=0.0,
        )


def test_capacity_manifest_declares_native_gemma4_launcher_contract():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path(
            "experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml"
        ).read_text()
    )
    launcher = pathlib.Path(manifest["controls"]["moe_launcher"]).read_text()
    config = pathlib.Path(manifest["controls"]["moe_config"]).read_text()
    assert "GEMMA4_EP16_COLLECTIVE_PREFLIGHT_PASS" in launcher
    assert "GEMMA4_EP16_PREFLIGHT_FAIL expected_two_nodes" in launcher
    assert "GEMMA4_EP16_PREFLIGHT_FAIL missing=" in launcher
    assert "FW_SITE=$(python3 -c" in launcher
    assert "model-00001-of-00002.safetensors' ]" in launcher
    assert "rm -rf '${LOCAL_MODEL}'; cp -r '${MODEL_PATH}' '${LOCAL_MODEL}'" in launcher
    assert "GEMMA4_EP16_SMOKE_PASS" in launcher
    assert "SMOKE_LOG=\"${PROJDIR}/experiments/qwen3_moe/capacity_gemma4_ep16_${PBS_JOBID}.smoke.log\"" in launcher
    assert "device_or_transport_signature" in launcher
    assert "banned[[:space:]]*:[[:space:]]*1" in launcher
    assert "TORCHTUNE_EP_GRAD_RELEASE_STREAMING=${TORCHTUNE_EP_GRAD_RELEASE_STREAMING:-1}" in launcher
    assert "TORCHTUNE_EP_ALL2ALL=${TORCHTUNE_EP_ALL2ALL:-1}" in launcher
    assert "TORCHTUNE_MOE_ROUTER_SEMANTICS=${TORCHTUNE_MOE_ROUTER_SEMANTICS:-sigmoid_argsort_v1}" in launcher
    assert "TORCHTUNE_MOE_ROUTING_INDEX_MODE=${TORCHTUNE_MOE_ROUTING_INDEX_MODE:-compact}" in launcher
    assert "gemma4_capacity_grouped_mm_${_CPU_METADATA_PROFILE_SUFFIX}" in launcher
    assert "TORCHTUNE_MOE_EXPECTED_EXECUTION_PATH=grouped_mm" in launcher
    assert "mark_measurement_artifacts_complete" in launcher
    assert 'expected_execution_path=os.environ["TORCHTUNE_MOE_EXPECTED_EXECUTION_PATH"]' in launcher
    for timing in ("grouped_gemm_gate", "grouped_gemm_up", "grouped_gemm_down"):
        assert f'    "{timing}",' in launcher
    assert "expert_parallel_degree: 16" in config
    assert "native_ep_sharded_experts: true" in config
    assert "model-00001-of-00002.safetensors" in config
    assert "model-00002-of-00002.safetensors" in config
    dense_launcher = pathlib.Path(
        manifest["controls"]["dense_control"]["measurement_launcher"]
    ).read_text()
    dense_config = pathlib.Path(
        manifest["controls"]["dense_control"]["measurement_configuration"]
    ).read_text()
    assert "GEMMA4_31B_DENSE_COLLECTIVE_PREFLIGHT_PASS" in dense_launcher
    assert "GEMMA4_31B_DENSE_PREFLIGHT_FAIL expected_two_nodes" in dense_launcher
    assert "GEMMA4_31B_DENSE_PREFLIGHT_FAIL missing=" in dense_launcher
    assert "GEMMA4_31B_DENSE_SMOKE_PASS" in dense_launcher
    assert "GEMMA4_31B_DENSE_MEASUREMENT_GATE_PASS" in dense_launcher
    assert "GEMMA4_31B_DENSE_ARTIFACT_WRITTEN" in dense_launcher
    assert "export DENSE_ARTIFACT LOCAL_MODEL" in dense_launcher
    assert '"total_parameters": 31.0' in dense_launcher
    assert '"batch_size": int(os.environ["TORCHTUNE_MOE_BATCH_SIZE"])' in dense_launcher
    assert '"microbatch_size": int(os.environ["TORCHTUNE_MOE_MICROBATCH_SIZE"])' in dense_launcher
    assert '"gradient_accumulation_steps": int(os.environ["TORCHTUNE_MOE_GRADIENT_ACCUMULATION_STEPS"])' in dense_launcher
    assert "dense_control_has_no_pipeline_microbatching" in dense_launcher
    assert "gradient_accumulation_is_not_token_volume" in dense_launcher

    recipe = pathlib.Path(
        "recipes/dev/full_finetune_moe_distributed_xpu.py"
    ).read_text()
    assert '"router_semantics": os.environ.get(' in recipe
    assert '"sigmoid_argsort_v1"' in recipe
    assert '"probability_topk_v2"' in recipe
    assert '"active_parameters_per_token": 31.0' in dense_launcher
    assert "peak_memory_reserved" in dense_launcher
    assert "gemma4_31b" in dense_config
    assert "seq_len: 4096" in dense_config


def test_canonical_moe_measurement_launchers_share_fast_path_defaults():
    from pathlib import Path

    root = Path("experiments/qwen3_moe")
    for filename in (
        "run_native_ep8_measurement.pbs",
        "run_native_ep16_measurement.pbs",
        "run_capacity_gemma4_ep16_measurement.pbs",
    ):
        launcher = (root / filename).read_text()
        assert "TORCHTUNE_MOE_GROUPED_EXPERTS=${TORCHTUNE_MOE_GROUPED_EXPERTS:-1}" in launcher
        assert "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS=${TORCHTUNE_MOE_SEQUENTIAL_EXPERTS:-0}" in launcher
        assert "TORCHTUNE_EP_ALL2ALL=${TORCHTUNE_EP_ALL2ALL:-1}" in launcher
        assert "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA=${TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA:-1}" in launcher
        assert "TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER=${TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER:-1}" in launcher
        assert "TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING=${TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING:-0}" in launcher
        assert "cpu_vector_routing_metadata_on" in launcher
        assert "cpu_vector_routing_metadata_off" in launcher
        assert "packed_routing_metadata_transfer_on" in launcher
        assert "packed_routing_metadata_transfer_off" in launcher
        assert "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS=${TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS:-1}" in launcher


def test_canonical_qwen_ep_measurement_preserves_seq4096_memory_gate():
    from pathlib import Path

    root = Path("experiments/qwen3_moe")
    for filename in ("run_native_ep8_measurement.pbs", "run_native_ep16_measurement.pbs"):
        launcher = (root / filename).read_text()
        assert "checkpoint_experts=true" in launcher
        assert "checkpoint_experts=false" not in launcher


def test_fused_alltoall_flag_is_exported_before_profile_construction():
    from pathlib import Path

    for filename in (
        "run_native_ep8_measurement.pbs",
        "run_native_ep16_measurement.pbs",
    ):
        launcher = (Path("experiments/qwen3_moe") / filename).read_text()
        flag = "export TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING="
        profile = 'export TORCHTUNE_MOE_OPTIMIZATION_PROFILE="'
        assert launcher.index(flag) < launcher.index(profile)
        assert "fused_alltoall_routing_on" in launcher
        assert "fused_alltoall_routing_off" in launcher


def test_cpu_vector_metadata_flag_is_exported_before_profile_construction():
    from pathlib import Path

    for filename in (
        "run_native_ep8_measurement.pbs",
        "run_native_ep16_measurement.pbs",
        "run_capacity_gemma4_ep16_measurement.pbs",
    ):
        launcher = (Path("experiments/qwen3_moe") / filename).read_text()
        flag = "export TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA="
        profile = "export TORCHTUNE_MOE_OPTIMIZATION_PROFILE"
        assert launcher.index(flag) < launcher.index(profile)
        packed_flag = "export TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER="
        assert launcher.index(packed_flag) < launcher.index(profile)


def test_canonical_ep_measurement_launchers_record_worktree_state():
    import pathlib

    root = pathlib.Path("experiments/qwen3_moe")
    for filename in ("run_native_ep8_measurement.pbs", "run_native_ep16_measurement.pbs"):
        launcher = (root / filename).read_text()
        assert "git status --porcelain" in launcher
        assert "TORCHTUNE_MOE_UNCOMMITTED=dirty" in launcher
        assert "TORCHTUNE_MOE_UNCOMMITTED=clean" in launcher


def test_standalone_track_manifests_validate():
    import pathlib
    import yaml

    for filename, result_class in (
        ("kernel_parity_manifest.yaml", "kernel_parity"),
        ("ep_scaling_manifest.yaml", "ep_scaling"),
        ("capacity_value_gemma4_manifest.yaml", "capacity_value"),
    ):
        path = pathlib.Path("experiments/qwen3_moe") / filename
        manifest = yaml.safe_load(path.read_text())
        assert manifest["result_class"] == result_class
        validate_manifest(manifest)


def test_evaluation_manifest_validates_track_graph():
    import pathlib
    import yaml

    path = pathlib.Path("experiments/qwen3_moe/evaluation_tracks_manifest.yaml")
    manifest = yaml.safe_load(path.read_text())
    validate_evaluation_manifest(manifest, manifest_path=path)
    manifest["tracks"]["ep_scaling"]["measurement_artifact"]["comparison_report"] = (
        "experiments/qwen3_moe/missing_report.py"
    )
    with pytest.raises(
        ValueError, match="missing ep_scaling.measurement_artifact.comparison_report"
    ):
        validate_evaluation_manifest(manifest, manifest_path=path)


def test_capacity_manifest_declares_canonical_artifact_inputs():
    import pathlib
    import yaml

    manifest = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/evaluation_tracks_manifest.yaml").read_text()
    )
    inputs = manifest["tracks"]["capacity_value"]["canonical_report_inputs"]
    assert "--moe-ep-artifact" in inputs["moe"]
    assert "--dense-artifact" in inputs["dense"]
    assert "steady_state_memory" in manifest["tracks"]["capacity_value"]["required_metrics"]
    required_metrics = set(
        yaml.safe_load(
            pathlib.Path(
                "experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml"
            ).read_text()
        )["required_metrics"]
    )
    assert {"mfu_percent", "total_parameters"}.issubset(required_metrics)
    assert not {"mfu", "model_size"}.intersection(required_metrics)

    historical = yaml.safe_load(
        (pathlib.Path("experiments/qwen3_moe")
         / "native_ep8_seq4096_baseline_manifest.yaml").read_text()
    )
    assert "result_class" not in historical


def test_manifest_requires_explicit_decision_and_gap_deliverables():
    import pathlib
    import yaml

    required_decisions = {
        "highest_repeatable_moe_per_gpu_mfu",
        "highest_aggregate_throughput",
        "best_ep_scaling_point",
        "closest_strict_dense_parity",
        "best_capacity_value_result",
    }
    required_gaps = {
        "communication",
        "expert_compute",
        "attention",
        "optimizer",
        "pipeline_bubble",
    }
    for filename in (
        "kernel_parity_manifest.yaml",
        "ep_scaling_manifest.yaml",
        "capacity_value_gemma4_manifest.yaml",
    ):
        path = pathlib.Path("experiments/qwen3_moe") / filename
        manifest = yaml.safe_load(path.read_text())
        assert set(manifest["decision_categories"]) == required_decisions
        assert set(manifest["gap_attribution"]) == required_gaps

    evaluation = yaml.safe_load(
        pathlib.Path("experiments/qwen3_moe/evaluation_tracks_manifest.yaml").read_text()
    )
    assert set(evaluation["decision_categories"]) == required_decisions
    assert set(evaluation["gap_categories"]) == required_gaps


def test_manifest_documents_collective_metrics_and_canonical_transport():
    manifest_text = (
        __import__("pathlib").Path(
            "experiments/qwen3_moe/evaluation_tracks_manifest.yaml"
        ).read_text()
    )
    for metric in (
        "expert_forward",
        "allgather_forward",
        "reduce_scatter_forward",
        "reduce_scatter_backward",
        "allgather_backward",
    ):
        assert metric in manifest_text
    assert "collective_metric_policy: canonical_ep_is_alltoall" in manifest_text
    assert "canonical EP8/EP16 measurements default to the validated grouped_mm and AllToAll path" in manifest_text
    kernel_manifest = __import__("pathlib").Path(
        "experiments/qwen3_moe/kernel_parity_manifest.yaml"
    ).read_text()
    assert 'TORCHTUNE_EP_ALL2ALL: "1"' in kernel_manifest
    marker = "- change: avoid_device_routing_metadata_copy"
    marker_start = manifest_text.index(marker)
    marker_end = manifest_text.index("- change:", marker_start + len(marker))
    metadata_entry = manifest_text[marker_start:marker_end]
    assert "ab_override: TORCHTUNE_EP_CPU_METADATA_TRANSFER=0" in metadata_entry
    assert "TORCHTUNE_EP_UNINITIALIZED_COLLECTIVE_BUFFERS=0" not in metadata_entry

    assert "candidate_adapter_identified_pending_capacity_gate" in manifest_text
    assert "qualification: adapter_candidate_only" in manifest_text
    assert "dense_control_does_not_fit_one_node_or_comparison_is_explicitly_capacity_only" in manifest_text
    assert "factored_optimizer_inplace_normalization" in manifest_text
    assert "passed_reference_update_and_state_shape_tests" in manifest_text
    for launcher_name in (
        "run_native_ep8_measurement.pbs",
        "run_native_ep16_measurement.pbs",
    ):
        launcher_text = (
            __import__("pathlib").Path("experiments/qwen3_moe", launcher_name).read_text()
        )
        assert "TORCHTUNE_MOE_OPTIMIZER_COMPONENT" in launcher_text
        assert "optimizer._component_=${TORCHTUNE_MOE_OPTIMIZER_COMPONENT}" in launcher_text
        assert "optimizer.fused=${TORCHTUNE_MOE_OPTIMIZER_FUSED}" in launcher_text
        assert "TORCHTUNE_EP_ZERO_COST_AG_ANCHOR=${TORCHTUNE_EP_ZERO_COST_AG_ANCHOR:-1}" in launcher_text
        assert "zero_cost_ag_anchor" in launcher_text
        assert "TORCHTUNE_MOE_VECTOR_PACKING=${TORCHTUNE_MOE_VECTOR_PACKING:-0}" in launcher_text
        assert "vector_packing_on" in launcher_text
        assert "vector_packing_off" in launcher_text
        assert "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE=${TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE:-1}" in launcher_text
        assert "TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS=${TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS:-1}" in launcher_text
        assert "rowwise_alltoall_unpermute_on" in launcher_text
        assert "rowwise_alltoall_unpermute_off" in launcher_text
        assert "uninitialized_alltoall_buffers_on" in launcher_text
        assert "uninitialized_alltoall_buffers_off" in launcher_text
        assert "TORCHTUNE_MOE_EXPECTED_EXECUTION_PATH" in launcher_text
        assert "expected_execution_path=os.environ[\"TORCHTUNE_MOE_EXPECTED_EXECUTION_PATH\"]" in launcher_text
        assert "FW_SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')" in launcher_text
        assert "/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.3.1" not in launcher_text

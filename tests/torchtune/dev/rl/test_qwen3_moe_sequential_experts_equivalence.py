"""CPU pin-down for TORCHTUNE_MOE_SEQUENTIAL_EXPERTS in GroupedExpertsHF.

Isolates a UR_RESULT_ERROR_OUT_OF_RESOURCES (error 40) crash found 2026-07-21
at Qwen3-30B-A3B EP=8 forward_batch_size=4: the padded-BMM forward allocates
a fresh ``[E, max_count, dim]`` tensor every call (48 layers/step), which
exhausts the XPU Level Zero handle table at this chunk size (same failure
class as the historical fix in torchtune/modules/moe/experts.py::GroupedExperts,
see memory/project_expert_forward_fix.md). The sequential path avoids the
large padded allocation entirely.

This test proves the sequential path matches the padded-BMM path to fp32
tolerance (atol=rtol=1e-5) on CPU — both compute the same per-expert
matmuls, just with different memory layouts (padded-then-sliced vs
sliced-directly). This pins down forward/backward math and routing edge
cases (zero-count experts, all-tokens-on-one-expert); it does NOT establish
bf16/XPU numerical equivalence — torch.bmm and per-expert `@` can use
different kernels and reduction orders on XPU, and small per-layer
differences could compound across 48 MoE layers. Treat HW bf16 parity as
unverified by this test; the HW run itself (job 8681553, both G=4/fbs=4 and
G=8/fbs=4 clean 4/4 steps) is the actual correctness+stability evidence for
production use.

Pure-Python (no torch.distributed, no XPU); runs on a login node in ~1s.
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch


def _reload_experts_module(sequential: bool):
    os.environ["TORCHTUNE_MOE_SEQUENTIAL_EXPERTS"] = "1" if sequential else "0"
    import torchtune.models.qwen3_moe._experts as experts_mod
    importlib.reload(experts_mod)
    # importlib.reload() mutates the module in place and re-executes the
    # class body, so the returned module's _SEQUENTIAL_EXPERTS and
    # GroupedExpertsHF must reflect the value just set — assert this
    # explicitly rather than trusting the reload side effect silently.
    assert experts_mod._SEQUENTIAL_EXPERTS is sequential
    return experts_mod


@pytest.fixture(autouse=True)
def _restore_env():
    prior = os.environ.get("TORCHTUNE_MOE_SEQUENTIAL_EXPERTS")
    yield
    if prior is None:
        os.environ.pop("TORCHTUNE_MOE_SEQUENTIAL_EXPERTS", None)
        _reload_experts_module(sequential=False)
    else:
        os.environ["TORCHTUNE_MOE_SEQUENTIAL_EXPERTS"] = prior
        _reload_experts_module(sequential=(prior == "1"))


@pytest.mark.parametrize(
    "num_experts, dim, hidden_dim, counts",
    [
        (16, 64, 128, [4, 0, 3, 1, 2, 0, 0, 5, 1, 1, 1, 1, 0, 0, 0, 1]),  # EP=8-shape, ragged
        (16, 64, 128, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # uniform
        (16, 64, 128, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16]),  # all-on-one-expert
        (4, 32, 64, [2, 3, 1, 4]),  # small smoke
    ],
)
def test_sequential_matches_padded_bmm_forward(num_experts, dim, hidden_dim, counts):
    torch.manual_seed(0)
    total = sum(counts)
    x = torch.randn(total, dim, dtype=torch.float32)
    ntpe = torch.tensor(counts, dtype=torch.float32)

    padded_mod = _reload_experts_module(sequential=False)
    padded_cls = padded_mod.GroupedExpertsHF
    padded_experts = padded_cls(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
    padded_experts.reset_parameters()
    state = {k: v.clone() for k, v in padded_experts.state_dict().items()}
    out_padded = padded_experts(x.clone(), ntpe.clone())

    seq_mod = _reload_experts_module(sequential=True)
    # reload() mutates the module in place (same object) but re-executes the
    # class body, so the two GroupedExpertsHF class objects must differ even
    # though padded_mod is seq_mod — this is what actually proves both code
    # paths were exercised rather than the same one twice.
    assert padded_mod is seq_mod, "reload must mutate the same module object"
    assert seq_mod.GroupedExpertsHF is not padded_cls, "reload must redefine the class"
    seq_experts = seq_mod.GroupedExpertsHF(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
    )
    seq_experts.load_state_dict(state)
    out_seq = seq_experts(x.clone(), ntpe.clone())

    torch.testing.assert_close(out_padded, out_seq, atol=1e-5, rtol=1e-5)


def test_sequential_matches_padded_bmm_backward():
    torch.manual_seed(1)
    num_experts, dim, hidden_dim = 8, 32, 64
    counts = [2, 0, 3, 1, 4, 0, 1, 1]
    total = sum(counts)
    x_base = torch.randn(total, dim, dtype=torch.float32)
    ntpe = torch.tensor(counts, dtype=torch.float32)

    padded_mod = _reload_experts_module(sequential=False)
    padded_cls = padded_mod.GroupedExpertsHF
    padded_experts = padded_cls(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
    padded_experts.reset_parameters()
    state = {k: v.clone() for k, v in padded_experts.state_dict().items()}

    x_p = x_base.clone().requires_grad_(True)
    out_p = padded_experts(x_p, ntpe.clone())
    out_p.sum().backward()

    seq_mod = _reload_experts_module(sequential=True)
    assert padded_mod is seq_mod, "reload must mutate the same module object"
    assert seq_mod.GroupedExpertsHF is not padded_cls, "reload must redefine the class"
    seq_experts = seq_mod.GroupedExpertsHF(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
    )
    seq_experts.load_state_dict(state)
    x_s = x_base.clone().requires_grad_(True)
    out_s = seq_experts(x_s, ntpe.clone())
    out_s.sum().backward()

    torch.testing.assert_close(out_p, out_s, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_p.grad, x_s.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        padded_experts.gate_proj.grad, seq_experts.gate_proj.grad, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        padded_experts.up_proj.grad, seq_experts.up_proj.grad, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        padded_experts.down_proj.grad, seq_experts.down_proj.grad, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("inplace", [False, True])
def test_sequential_swiglu_inplace_matches_out_of_place(monkeypatch, inplace):
    import torchtune.models.qwen3_moe._experts as experts_mod

    torch.manual_seed(11)
    counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    x_base = torch.randn(int(counts.sum()), 16)
    reference = experts_mod.GroupedExpertsHF(dim=16, hidden_dim=24, num_experts=4)
    reference.reset_parameters()
    candidate = experts_mod.GroupedExpertsHF(dim=16, hidden_dim=24, num_experts=4)
    candidate.load_state_dict(reference.state_dict())
    monkeypatch.setattr(experts_mod, "_SEQUENTIAL_EXPERTS", True)

    x_reference = x_base.clone().requires_grad_(True)
    monkeypatch.setattr(experts_mod, "_USE_INPLACE_SWIGLU", False)
    output_reference = reference(x_reference, counts)
    output_reference.square().mean().backward()

    x_candidate = x_base.clone().requires_grad_(True)
    monkeypatch.setattr(experts_mod, "_USE_INPLACE_SWIGLU", inplace)
    output_candidate = candidate(x_candidate, counts)
    output_candidate.square().mean().backward()

    torch.testing.assert_close(output_candidate, output_reference)
    torch.testing.assert_close(x_candidate.grad, x_reference.grad)
    for candidate_parameter, reference_parameter in zip(
        candidate.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            candidate_parameter.grad, reference_parameter.grad
        )


def test_sequential_zero_total_tokens_matches():
    """Both paths must handle the total==0 edge case identically."""
    num_experts, dim, hidden_dim = 4, 16, 32
    ntpe = torch.zeros(num_experts, dtype=torch.float32)
    x = torch.zeros(0, dim, dtype=torch.float32)

    padded_mod = _reload_experts_module(sequential=False)
    padded_experts = padded_mod.GroupedExpertsHF(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
    )
    padded_experts.reset_parameters()
    state = {k: v.clone() for k, v in padded_experts.state_dict().items()}
    out_padded = padded_experts(x.clone(), ntpe.clone())

    seq_mod = _reload_experts_module(sequential=True)
    seq_experts = seq_mod.GroupedExpertsHF(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
    )
    seq_experts.load_state_dict(state)
    out_seq = seq_experts(x.clone(), ntpe.clone())

    assert out_padded.shape == out_seq.shape == (0, dim)


def test_grouped_experts_matches_sequential_forward_backward(monkeypatch):
    import torchtune.models.qwen3_moe._experts as experts_mod

    def grouped_mm_reference(inputs, weights, offs=None):
        outputs = []
        start = 0
        for expert, end in enumerate(offs.tolist()):
            outputs.append(inputs[start:end] @ weights[expert])
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm_reference)
    monkeypatch.setattr(experts_mod, "_GROUPED_EXPERTS", False)
    monkeypatch.setattr(experts_mod, "_SEQUENTIAL_EXPERTS", True)

    torch.manual_seed(2)
    counts = [3, 0, 5, 1]
    ntpe = torch.tensor(counts, dtype=torch.float32)
    x_base = torch.randn(sum(counts), 32)
    sequential = experts_mod.GroupedExpertsHF(
        dim=32, hidden_dim=64, num_experts=len(counts)
    )
    sequential.reset_parameters()
    grouped = experts_mod.GroupedExpertsHF(
        dim=32, hidden_dim=64, num_experts=len(counts)
    )
    grouped.load_state_dict(sequential.state_dict())

    x_sequential = x_base.clone().requires_grad_(True)
    output_sequential = sequential(x_sequential, ntpe)
    output_sequential.sum().backward()

    monkeypatch.setattr(experts_mod, "_GROUPED_EXPERTS", True)
    x_grouped = x_base.clone().requires_grad_(True)
    output_grouped = grouped(x_grouped, ntpe)
    output_grouped.sum().backward()

    torch.testing.assert_close(output_grouped, output_sequential)
    torch.testing.assert_close(x_grouped.grad, x_sequential.grad)
    for grouped_param, sequential_param in zip(
        grouped.parameters(), sequential.parameters()
    ):
        torch.testing.assert_close(grouped_param.grad, sequential_param.grad)

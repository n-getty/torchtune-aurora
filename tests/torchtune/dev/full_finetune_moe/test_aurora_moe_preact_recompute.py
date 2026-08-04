# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


OVERLAY = Path("/lus/flare/projects/ModCon/ngetty/aurora_moe_dropin_overlay")


@pytest.fixture
def kernel(monkeypatch):
    monkeypatch.syspath_prepend(str(OVERLAY))
    module_name = "aurora_moe._kernels.expert_major_segmented_sonic"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)

    monkeypatch.setattr(module, "_check_inputs", lambda *args: None)
    monkeypatch.setattr(module, "_check_payload_inputs", lambda *args: None)
    monkeypatch.setattr(module, "_record", lambda *args: nullcontext())
    monkeypatch.setattr(module, "make_expert_major_layout", _identity_layout(module))
    monkeypatch.setattr(module, "_pack", lambda values, *args: values)
    monkeypatch.setattr(module, "_unpack", lambda values, *args: values)
    monkeypatch.setattr(module, "_pack_pair", lambda left, right, *args: (left, right))
    monkeypatch.setattr(
        module,
        "_unpack_scaled",
        lambda values, scores, *args: values * scores.unsqueeze(-1),
    )
    monkeypatch.setattr(
        module,
        "_pack_payload",
        lambda payload, *args: (
            payload[:, :-1].contiguous(),
            payload[:, -1].contiguous(),
        ),
    )
    monkeypatch.setattr(
        module,
        "_split_expert_major_payload",
        lambda payload: (payload[:, :-1].contiguous(), payload[:, -1].contiguous()),
    )
    monkeypatch.setattr(
        module,
        "_unpack_payload",
        lambda tokens, scores, *args: torch.cat((tokens, scores[:, None]), dim=-1),
    )
    monkeypatch.setattr(
        module,
        "_fuse_expert_major_payload_grad",
        lambda tokens, scores: torch.cat((tokens, scores[:, None]), dim=-1),
    )
    monkeypatch.setattr(
        module,
        "expert_major_to_payload_zero_score_parallel",
        lambda tokens, *args: torch.cat(
            (tokens, torch.zeros_like(tokens[:, :1])), dim=-1
        ),
    )
    monkeypatch.setattr(
        module,
        "expert_major_to_payload_zero_score_row_parallel",
        lambda tokens, *args: torch.cat(
            (tokens, torch.zeros_like(tokens[:, :1])), dim=-1
        ),
    )
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_GEMM", "torch")
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_DW", "torch")
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_REORDER", "serial")
    monkeypatch.setenv("AURORA_MOE_SEGMENTED_POINTWISE", "torch")
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_DOWN_BACKWARD", "reordered")
    return module


def _identity_layout(module):
    def make_layout(segments, include_row_schedule=False):
        counts = segments.counts.sum(dim=0, dtype=torch.int64)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        return module.ExpertMajorLayout(
            offsets,
            offsets.to(torch.int32),
            segments.source_expert_offsets[:, :-1],
            offsets.new_empty(0),
        )

    return make_layout


def _run_case(kernel, monkeypatch, *, payload_input, activation, packed, suppress):
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE", str(int(packed)))
    monkeypatch.setenv("AURORA_MOE_IGNORE_ROUTER_GRAD", str(int(suppress)))
    counts = torch.tensor([[2, 3]], dtype=torch.int32)
    segments = kernel.make_peer_expert_segments(counts)
    rows, model_dim, hidden_dim = 5, 3, 4
    torch.manual_seed(0)
    base_tokens = torch.randn(rows, model_dim)
    base_scores = torch.randn(rows)
    base_up = torch.randn(2, model_dim, hidden_dim)
    base_gate = torch.randn(2, model_dim, hidden_dim)
    base_down = torch.randn(2, hidden_dim, model_dim)
    results = []

    for recompute in (False, True):
        monkeypatch.setenv(
            "AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT", str(int(recompute))
        )
        up = base_up.clone().requires_grad_()
        gate = base_gate.clone().requires_grad_() if activation == "swiglu" else None
        down = base_down.clone().requires_grad_()
        saved_shapes = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
            lambda tensor: tensor,
        ):
            if payload_input:
                payload = torch.cat(
                    (base_tokens, base_scores[:, None]), dim=-1
                ).requires_grad_()
                output = kernel.expert_major_segmented_local_moe_payload(
                    payload,
                    segments,
                    up,
                    gate,
                    down,
                    activation=activation,
                    expert_rows=(2, 3),
                    already_expert_major=True,
                )
                output.square().sum().backward()
                input_grads = (payload.grad.clone(),)
            else:
                tokens = base_tokens.clone().requires_grad_()
                scores = base_scores.clone().requires_grad_()
                output = kernel.expert_major_segmented_local_moe(
                    tokens,
                    scores,
                    segments,
                    up,
                    gate,
                    down,
                    activation=activation,
                    expert_rows=(2, 3),
                )
                output.square().sum().backward()
                input_grads = (tokens.grad.clone(), scores.grad.clone())
        results.append(
            (
                output.detach(),
                input_grads,
                up.grad.clone(),
                gate.grad.clone() if gate is not None else None,
                down.grad.clone(),
                saved_shapes,
            )
        )
    return results


@pytest.mark.parametrize(
    ("payload_input", "activation", "packed", "suppress"),
    [
        (False, "swiglu", False, False),
        (True, "swiglu", False, False),
        (True, "swiglu", False, True),
        (True, "swiglu", True, False),
        (False, "squared-relu", False, False),
    ],
)
def test_preact_recompute_output_gradients_and_saved_tensors(
    kernel, monkeypatch, payload_input, activation, packed, suppress
):
    saved, recomputed = _run_case(
        kernel,
        monkeypatch,
        payload_input=payload_input,
        activation=activation,
        packed=packed,
        suppress=suppress,
    )
    for saved_value, recomputed_value in zip(saved[:5], recomputed[:5]):
        if saved_value is None:
            assert recomputed_value is None
        elif isinstance(saved_value, tuple):
            for left, right in zip(saved_value, recomputed_value):
                torch.testing.assert_close(left, right)
        else:
            torch.testing.assert_close(saved_value, recomputed_value)

    preact_shape = (5, 8 if packed else 4)
    saved_preact_count = 1 if packed or activation == "squared-relu" else 2
    assert (
        saved[5].count(preact_shape)
        == recomputed[5].count(preact_shape) + saved_preact_count
    )
    if payload_input and suppress:
        assert torch.count_nonzero(recomputed[1][0][:, -1]) == 0


def test_preact_recompute_env_defaults_off_and_rejects_invalid(kernel, monkeypatch):
    monkeypatch.delenv("AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT", raising=False)
    assert kernel._preact_policy_requested() == "save"
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT", "up_only")
    assert kernel._preact_policy_requested() == "up_only"
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT", "yes")
    with pytest.raises(ValueError, match="must be '0', '1', or 'up_only'"):
        kernel._preact_policy_requested()


def test_up_only_recompute_retains_gate_and_matches_gradients(kernel, monkeypatch):
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_PACKED_UP_GATE", "0")
    monkeypatch.setenv("AURORA_MOE_IGNORE_ROUTER_GRAD", "0")
    saved, _ = _run_case(
        kernel,
        monkeypatch,
        payload_input=True,
        activation="swiglu",
        packed=False,
        suppress=False,
    )
    monkeypatch.setenv("AURORA_MOE_EXPERT_MAJOR_RECOMPUTE_PREACT", "up_only")
    counts = torch.tensor([[2, 3]], dtype=torch.int32)
    segments = kernel.make_peer_expert_segments(counts)
    torch.manual_seed(0)
    payload = torch.cat((torch.randn(5, 3), torch.randn(5, 1)), dim=-1).requires_grad_()
    up = torch.randn(2, 3, 4, requires_grad=True)
    gate = torch.randn(2, 3, 4, requires_grad=True)
    down = torch.randn(2, 4, 3, requires_grad=True)
    saved_shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
        lambda tensor: tensor,
    ):
        output = kernel.expert_major_segmented_local_moe_payload(
            payload,
            segments,
            up,
            gate,
            down,
            expert_rows=(2, 3),
            already_expert_major=True,
        )
        output.square().sum().backward()
    for saved_value, partial_value in zip(
        saved[:5],
        (output.detach(), (payload.grad,), up.grad, gate.grad, down.grad),
    ):
        if isinstance(saved_value, tuple):
            torch.testing.assert_close(saved_value[0], partial_value[0])
        else:
            torch.testing.assert_close(saved_value, partial_value)
    assert saved_shapes.count((5, 4)) == saved[5].count((5, 4)) - 1

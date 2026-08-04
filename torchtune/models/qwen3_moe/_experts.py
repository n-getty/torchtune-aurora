# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

# Opt-in: replace the padded-BMM forward with a sequential per-expert forward
# (no new [E, max_count, dim] allocation per call). Candidate fix for a
# UR_RESULT_ERROR_OUT_OF_RESOURCES (error 40) crash isolated 2026-07-21 at
# EP=8 forward_batch_size=4 (48 layers x fresh x_padded alloc/call exhausts
# the L0 handle table — same failure class as the historical
# torchtune/modules/moe/experts.py::GroupedExperts fix, see
# memory/project_expert_forward_fix.md). Default OFF: the padded-BMM path is
# a measured 6.3x speedup over sequential (project_bmm_expert_speedup.md) and
# is the validated production default at fbs<=2.
_SEQUENTIAL_EXPERTS = os.environ.get("TORCHTUNE_MOE_SEQUENTIAL_EXPERTS", "0") == "1"
_GROUPED_EXPERTS = os.environ.get("TORCHTUNE_MOE_GROUPED_EXPERTS", "0") == "1"
_GROUPED_RECOMPUTE_PREACT = os.environ.get(
    "TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT", "0"
)
_VECTOR_PACKING = os.environ.get("TORCHTUNE_MOE_VECTOR_PACKING", "0") == "1"
_USE_INPLACE_SWIGLU = (
    os.environ.get("TORCHTUNE_MOE_INPLACE_SWIGLU", "1") == "1"
)
if _GROUPED_RECOMPUTE_PREACT not in {"0", "1", "up_only"}:
    raise ValueError(
        "TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT must be 0, 1, or up_only, "
        f"got {_GROUPED_RECOMPUTE_PREACT!r}"
    )


def _grouped_swiglu(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    offsets: torch.Tensor,
    activation: Callable,
    measurement=None,
) -> torch.Tensor:
    if measurement is not None and measurement.enabled:
        with measurement.time("grouped_gemm_gate"):
            gate = torch._grouped_mm(x, gate_proj, offs=offsets)
        with measurement.time("grouped_gemm_up"):
            up = torch._grouped_mm(x, up_proj, offs=offsets)
    else:
        gate = torch._grouped_mm(x, gate_proj, offs=offsets)
        up = torch._grouped_mm(x, up_proj, offs=offsets)
    activated_gate = activation(gate)
    if _USE_INPLACE_SWIGLU:
        activated_gate.mul_(up)
    else:
        activated_gate = activated_gate * up
    return activated_gate


def _grouped_swiglu_up_only(
    x: torch.Tensor,
    gate: torch.Tensor,
    up_proj: torch.Tensor,
    offsets: torch.Tensor,
    activation: Callable,
    measurement=None,
) -> torch.Tensor:
    if measurement is not None and measurement.enabled:
        with measurement.time("grouped_gemm_up"):
            up = torch._grouped_mm(x, up_proj, offs=offsets)
    else:
        up = torch._grouped_mm(x, up_proj, offs=offsets)
    activated_gate = activation(gate)
    if _USE_INPLACE_SWIGLU:
        activated_gate.mul_(up)
    else:
        activated_gate = activated_gate * up
    return activated_gate


def _expert_padded_row_indices(
    counts: torch.Tensor, max_count: int, total: int
) -> torch.Tensor:
    """Return padded-BMM rows corresponding to sorted routed tokens."""
    if counts.dtype != torch.long:
        counts = counts.to(dtype=torch.long)
    starts = torch.cumsum(counts, dim=0)
    starts.sub_(counts)
    expert_offsets = torch.arange(
        counts.numel(), device=counts.device, dtype=counts.dtype
    )
    expert_offsets.mul_(max_count)
    expert_offsets.sub_(starts)
    row_indices = torch.repeat_interleave(expert_offsets, counts)
    row_indices.add_(torch.arange(
        total, device=counts.device, dtype=counts.dtype
    ))
    return row_indices


def _pack_expert_tokens_vectorized(
    x: torch.Tensor, counts: torch.Tensor, max_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack sorted expert tokens with one indexed copy instead of E slices."""
    row_indices = _expert_padded_row_indices(counts, max_count, x.shape[0])
    packed = x.new_zeros(counts.numel() * max_count, x.shape[1])
    packed.index_copy_(0, row_indices, x)
    return packed.view(counts.numel(), max_count, x.shape[1]), row_indices


class GroupedExpertsHF(nn.Module):
    """Grouped experts with HF-native weight layout [E, out_features, in_features].

    Identical computation to GroupedExperts but stores weights in HuggingFace
    convention (nn.Linear's [out, in] per expert). This eliminates all transpose
    overhead at checkpoint load, save, and weight sync — the storage format
    matches HF/vLLM directly.

    The forward path uses .mT (a free view, no memory copy) to transpose
    weights for matmul.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension (expert intermediate size).
        num_experts (int): Number of experts. Default is 1.
        activation (Callable): Activation function. Default is F.silu.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        num_experts: int = 1,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        # HF layout: [E, out_features, in_features]
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.act_fn = activation

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

    @torch.inference_mode(mode=False)
    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(total_tokens, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``

        Returns:
            torch.Tensor: tensor with shape ``(total_tokens, dim)``
        """
        num_local_experts = num_tokens_per_expert.shape[0]
        total = x.shape[0]

        if total == 0:
            x_zero = (
                x.reshape(0, self.dim) if x.numel() == 0 else x.new_empty(0, self.dim)
            )
            anchor = (
                self.gate_proj.sum() + self.down_proj.sum() + self.up_proj.sum()
            ) * 0.0
            x_anchor = (x.sum(dim=0, keepdim=False) * 0.0) if x.requires_grad else None
            if x_anchor is not None:
                anchor = anchor + x_anchor.sum()
            return x_zero + anchor

        counts = (
            num_tokens_per_expert
            if not torch.is_floating_point(num_tokens_per_expert)
            else num_tokens_per_expert.round().to(torch.int64)
        )

        if _GROUPED_EXPERTS:
            offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
            gate_proj = self.gate_proj.transpose(-1, -2)
            up_proj = self.up_proj.transpose(-1, -2)
            measurement = getattr(self, "_moe_measurement", None)
            if _GROUPED_RECOMPUTE_PREACT == "1" and torch.is_grad_enabled():
                hidden = torch.utils.checkpoint.checkpoint(
                    _grouped_swiglu,
                    x,
                    gate_proj,
                    up_proj,
                    offsets,
                    self.act_fn,
                    measurement,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif _GROUPED_RECOMPUTE_PREACT == "up_only" and torch.is_grad_enabled():
                if measurement is not None and measurement.enabled:
                    with measurement.time("grouped_gemm_gate"):
                        gate = torch._grouped_mm(x, gate_proj, offs=offsets)
                else:
                    gate = torch._grouped_mm(x, gate_proj, offs=offsets)
                hidden = torch.utils.checkpoint.checkpoint(
                    _grouped_swiglu_up_only,
                    x,
                    gate,
                    up_proj,
                    offsets,
                    self.act_fn,
                    measurement,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = _grouped_swiglu(
                    x, gate_proj, up_proj, offsets, self.act_fn, measurement
                )
            if measurement is not None and measurement.enabled:
                with measurement.time("grouped_gemm_down"):
                    return torch._grouped_mm(
                        hidden, self.down_proj.transpose(-1, -2), offs=offsets
                    )
            else:
                return torch._grouped_mm(
                    hidden, self.down_proj.transpose(-1, -2), offs=offsets
                )

        if _SEQUENTIAL_EXPERTS:
            # Sequential per-expert forward: O(count x hidden) temporaries,
            # no new [E, max_count, dim] allocation per call. See module-level
            # comment on _SEQUENTIAL_EXPERTS for why this exists.
            measurement = getattr(self, "_moe_measurement", None)
            out = torch.empty(total, self.dim, dtype=x.dtype, device=x.device)
            timing = (
                measurement.time("sequential_expert_compute")
                if measurement is not None and measurement.enabled
                else nullcontext()
            )
            with timing:
                count_list = counts.tolist()
                offset = 0
                for e in range(num_local_experts):
                    c = int(count_list[e])
                    if c == 0:
                        continue
                    x_e = x[offset : offset + c]
                    gate_timing = (
                        measurement.time("sequential_expert_gate")
                        if measurement is not None and measurement.enabled
                        else nullcontext()
                    )
                    with gate_timing:
                        g = x_e @ self.gate_proj[e].mT
                    h = self.act_fn(g)
                    up_timing = (
                        measurement.time("sequential_expert_up")
                        if measurement is not None and measurement.enabled
                        else nullcontext()
                    )
                    with up_timing:
                        up = x_e @ self.up_proj[e].mT
                    if _USE_INPLACE_SWIGLU:
                        h.mul_(up)
                    else:
                        h = h * up
                    down_timing = (
                        measurement.time("sequential_expert_down")
                        if measurement is not None and measurement.enabled
                        else nullcontext()
                    )
                    with down_timing:
                        out[offset : offset + c] = h @ self.down_proj[e].mT
                    offset += c
            return out

        measurement = getattr(self, "_moe_measurement", None)
        count_list = None
        if not _VECTOR_PACKING or (
            measurement is not None and measurement.enabled
        ) or os.environ.get("TORCHTUNE_MOE_BMM_DEBUG") == "1":
            count_list = counts.tolist()
        if _VECTOR_PACKING:
            max_count = (
                int(counts.max().item()) if count_list is None else max(count_list)
            )
        else:
            max_count = int(max(count_list))

        if measurement is not None and measurement.enabled:
            measurement.record_padded_bmm(
                count_list,
                model_dim=self.dim,
                hidden_dim=self.gate_proj.shape[-2],
            )

        if os.environ.get("TORCHTUNE_MOE_BMM_DEBUG") == "1":
            import torch.distributed as _bmm_dist

            _r = _bmm_dist.get_rank() if _bmm_dist.is_initialized() else 0
            if _r == 0:
                print(
                    f"[bmm_debug] E={num_local_experts} total={total} "
                    f"max_count={max_count} counts={count_list} "
                    f"mem_alloc={torch.xpu.memory_allocated() / 1e9:.2f}GB "
                    f"mem_reserved={torch.xpu.memory_reserved() / 1e9:.2f}GB",
                    flush=True,
                )

        # Scatter: pack sorted tokens into [E, max_count, dim] padded tensor.
        # Zero-padded positions contribute zero to bmm output and zero gradient.
        if _VECTOR_PACKING:
            x_padded, row_indices = _pack_expert_tokens_vectorized(
                x, counts, max_count
            )
        else:
            x_padded = x.new_zeros(num_local_experts, max_count, self.dim)
            offset = 0
            for e in range(num_local_experts):
                c = int(count_list[e])
                if c > 0:
                    x_padded[e, :c] = x[offset : offset + c]
                    offset += c

        # 3 batched matmuls replace E×3 sequential matmuls (128→1 kernel launch)
        measurement = getattr(self, "_moe_measurement", None)
        timing = (
            measurement.time("padded_bmm")
            if measurement is not None and measurement.enabled
            else nullcontext()
        )
        with timing:
            gate_out = torch.bmm(x_padded, self.gate_proj.mT)
            h = self.act_fn(gate_out)
            h = h * torch.bmm(x_padded, self.up_proj.mT)
            out_padded = torch.bmm(h, self.down_proj.mT)

        # Gather: extract results back to flat tensor.
        if _VECTOR_PACKING:
            out = out_padded.reshape(-1, self.dim).index_select(0, row_indices)
        else:
            out = torch.empty(total, self.dim, dtype=x.dtype, device=x.device)
            offset = 0
            for e in range(num_local_experts):
                c = int(count_list[e])
                if c > 0:
                    out[offset : offset + c] = out_padded[e, :c]
                    offset += c
        return out

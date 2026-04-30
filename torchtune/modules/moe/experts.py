# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
from typing import Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torchtune.modules.peft import AdapterModule

from .utils import should_use_grouped_mm

# v111 diag: count expert forward calls to log token load on first call only.
_EXPERT_FWD_CALL_COUNT: int = 0


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        activation (Callable): Activation function to use. Default is F.silu.
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
        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.act_fn = activation
        self.use_grouped_mm = should_use_grouped_mm()

    def reset_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        if self.up_proj is not None:
            nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

    def _forward_no_grouped_mm(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        # Sequential per-expert forward: process each expert's token slice independently.
        # Tokens in x are pre-sorted by expert (post-AllToAll dispatch), so we can
        # slice directly without scatter/gather, avoiding large padded intermediate tensors.
        # This approach uses O(count * hidden) temporary memory per expert rather than
        # O(E * max_T * dim) for the padded-BMM approach, preventing Level Zero UR handle
        # exhaustion on XPU during activation recompute in EP backward.
        #
        # NOTE: this function is called within a torch.utils.checkpoint context
        # (use_reentrant=False). With use_reentrant=False, PyTorch does NOT compare
        # saved-tensor shapes between forward and recompute — so per-expert intermediates
        # with variable shapes [count_e, hidden_dim] (due to AllToAll routing non-
        # determinism) are safe. With use_reentrant=True they cause SIGSEGV.
        #
        # When EP is active, self.gate_proj/up_proj/down_proj are local slices
        # (shape [num_local_experts, ...]) — E matches their dim 0.
        global _EXPERT_FWD_CALL_COUNT
        _EXPERT_FWD_CALL_COUNT += 1
        _call_n = _EXPERT_FWD_CALL_COUNT
        E = num_tokens_per_expert.shape[0]
        total = x.shape[0]
        # v111 diag: log expert token load on first expert forward call per run.
        # Print to stderr with flush so it bypasses SSH pipe buffering.
        if _call_n == 1:
            try:
                _rank = dist.get_rank() if dist.is_initialized() else -1
            except Exception:
                _rank = -1
            print(
                f"[v111-EXPERT-TOKENS] rank={_rank} call#={_call_n} "
                f"total_tokens={total} num_experts={E} "
                f"counts={num_tokens_per_expert.round().to(torch.int64).tolist()[:8]}...",
                flush=True,
            )
        if total == 0:
            # v161: keep an autograd link back to x AND to all expert weights so
            # _AllGatherRS.backward fires on this rank. Returning a bare
            # new_empty pinches off the graph → empty-dispatch rank silently
            # exits backward early → asymmetric #238 deadlock (v158-v160 root
            # cause). Use a no-op gather+sum+broadcast: 0 tokens means 0 cost
            # but the grad-fn chain stays alive.
            try:
                _r = dist.get_rank() if dist.is_initialized() else -1
            except Exception:
                _r = -1
            print(f"[v161-EXPERT-EMPTY] rank={_r} call#={_call_n} total=0", flush=True)
            x_zero = x.reshape(0, self.dim) if x.numel() == 0 else x.new_empty(0, self.dim)
            # Anchor: 0.0 * (gate_proj.sum() + down_proj.sum() + up_proj.sum()).
            # Adding a 0-d tensor of value 0 broadcast to (0, dim) is a no-op on
            # values but registers a grad-fn dependency on the expert weights.
            anchor = (self.gate_proj.sum() + self.down_proj.sum() + self.up_proj.sum()) * 0.0
            x_anchor = (x.sum(dim=0, keepdim=False) * 0.0) if x.requires_grad else None
            if x_anchor is not None:
                anchor = anchor + x_anchor.sum()
            # Broadcast scalar anchor onto (0, dim) — empty add is well-defined.
            return x_zero + anchor

        # Round before int conversion to guard against float32 rounding (e.g. 44.9999 → 45).
        counts = num_tokens_per_expert.round().to(torch.int64)
        out = torch.empty(total, self.dim, dtype=x.dtype, device=x.device)
        offset = 0
        for e in range(E):
            count = int(counts[e].item())
            if count == 0:
                continue
            x_e = x[offset : offset + count]  # [count, dim] — slice, no new alloc
            g = x_e @ self.gate_proj[e]  # [count, hidden_dim]
            h = self.act_fn(g)
            if self.up_proj is not None:
                h = h * (x_e @ self.up_proj[e])  # [count, hidden_dim]
            out[offset : offset + count] = h @ self.down_proj[e]  # [count, dim]
            offset += count
        return out

    # TODO: force no inference mode as a hack to get around
    # "Cannot set version_counter for inference tensor"
    @torch.inference_mode(mode=False)
    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``
                enumerating the number of tokens each expert receives

        Returns:
            torch.Tensor: tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
        """
        if not self.use_grouped_mm:
            return self._forward_no_grouped_mm(x, num_tokens_per_expert)

        # grouped mm implementation
        if num_tokens_per_expert is not None:
            # https://github.com/pytorch/pytorch/pull/150374
            # NOTE: torch._gouped_mm requires bf16 dtypes
            #       and shapes to be multiple of 16
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        w1, w2, w3 = (
            self.gate_proj,
            self.down_proj,
            self.up_proj,
        )
        assert (
            x.dtype == w1.dtype == w2.dtype == w3.dtype == torch.bfloat16
        ), "torch._grouped_mm only supports bf16 dtypes"
        h = self.act_fn(torch._grouped_mm(x, w1, offs=offsets))
        h = h * torch._grouped_mm(x, w3, offs=offsets)
        out = torch._grouped_mm(h, w2, offs=offsets)
        return out


class LoRAGroupedExperts(nn.Module, AdapterModule):
    """This class implements the grouped experts layer used in Mixture of Experts with additional LoRA
    adapter parameters.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability before LoRA layer. Default: 0.0
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        activation (Callable): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        num_experts: int = 1,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.act_fn = activation

        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_gate_a = nn.Parameter(torch.empty(num_experts, dim, rank))
        self.lora_gate_b = nn.Parameter(torch.empty(num_experts, rank, hidden_dim))
        self.lora_down_a = nn.Parameter(torch.empty(num_experts, hidden_dim, rank))
        self.lora_down_b = nn.Parameter(torch.empty(num_experts, rank, dim))
        self.lora_up_a = nn.Parameter(torch.empty(num_experts, dim, rank))
        self.lora_up_b = nn.Parameter(torch.empty(num_experts, rank, hidden_dim))
        self.merged = False
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        if self.up_proj is not None:
            nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.lora_gate_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_gate_b)
        nn.init.kaiming_uniform_(self.lora_down_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down_b)
        if self.lora_up_a is not None:
            nn.init.kaiming_uniform_(self.lora_up_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up_b)

    def adapter_params(self) -> list[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.

        For LoRA this means lora_gate, lora_up, lora_down a and b weights.
        """
        # NOTE: this function has to be updated if the names of the lora parameters
        # in this module change.
        adapter_params = [
            "lora_gate_a",
            "lora_gate_b",
            "lora_down_a",
            "lora_down_b",
            "lora_up_a",
            "lora_up_b",
        ]
        return adapter_params

    def _lora_tc_layer_forward(
        self,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_a_weight: torch.Tensor,
        lora_b_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass a single linear layer with lora adapter layers for Token Choice routing.

        Args:
            x (torch.Tensor): Input tensor with shape ``(tokens_per_expert, in_dim)``.
            base_weight (torch.Tensor): weight of the base linear projection, shape ``(in_dim, out_dim)``.
            lora_a_weight (torch.Tensor): weight of the lora adapter A layer, shape ``(in_dim, rank)``.
            lora_b_weight (torch.Tensor): weight of the lora adapter B layer, shape ``(rank, out_dim)``.

        Returns:
            torch.Tensor: Output tensor with shape ``(tokens_per_expert, out_dim)``.
        """
        out = torch.matmul(x, base_weight)
        if self.disabled:
            return out
        lora_out = torch.matmul(self.dropout(x), lora_a_weight)
        lora_out = (self.alpha / self.rank) * torch.matmul(lora_out, lora_b_weight)
        return out + lora_out

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``
                enumerating the number of tokens each expert receives

        Returns:
            torch.Tensor: tuple of input tensors each with shape ``(num_experts, tokens_per_expert, dim)`` for Token Choice(TC)
                or a single tensor with shape (num_experts, tokens_per_expert, dim) for Expert Choice(EC).
        """
        # a tuple of tensors indexed by experts
        # each with shape (tokens_per_expert(varying), dim)
        x = torch.split(
            x,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        out_experts_splits = []
        for expert_idx, x_expert in enumerate(x):
            gate_proj, down_proj = (
                self.gate_proj[expert_idx],
                self.down_proj[expert_idx],
            )
            lora_gate_a, lora_gate_b, lora_down_a, lora_down_b = (
                self.lora_gate_a[expert_idx],
                self.lora_gate_b[expert_idx],
                self.lora_down_a[expert_idx],
                self.lora_down_b[expert_idx],
            )
            h = self.act_fn(
                self._lora_tc_layer_forward(
                    x_expert, gate_proj, lora_gate_a, lora_gate_b
                )
            )

            if self.up_proj is not None:
                up_proj = self.up_proj[expert_idx]
                lora_up_a, lora_up_b = (
                    self.lora_up_a[expert_idx],
                    self.lora_up_b[expert_idx],
                )
                h = h * self._lora_tc_layer_forward(
                    x_expert, up_proj, lora_up_a, lora_up_b
                )

            h = self._lora_tc_layer_forward(h, down_proj, lora_down_a, lora_down_b)

            # h shape (tokens_per_expert(varying), hidden_dim)
            out_experts_splits.append(h)
        out = torch.cat(out_experts_splits, dim=0)

        return out

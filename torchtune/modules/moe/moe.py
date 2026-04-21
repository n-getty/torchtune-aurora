# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import nn

from .utils import should_use_grouped_mm


class TokenChoiceTopKRouter(nn.Module):
    """This class implements Token Choice routing. In Token Choice top K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        experts_per_token: int,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid is performed in float32 to avoid loss explosion
        scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)

        # Deterministic top-k: use argsort(stable=True) + slice instead of topk.
        # torch.topk lacks stable=True on XPU → non-deterministic tie-breaking →
        # AC recompute produces different expert assignments → backward matmul shape
        # mismatch. argsort(stable=True) breaks ties by original expert index order.
        # top scores shape (bs*slen, top_k)
        sorted_indices = torch.argsort(scores, dim=1, stable=True, descending=True)
        selected_experts_indices = sorted_indices[:, : self.experts_per_token]
        top_scores = torch.gather(scores, 1, selected_experts_indices)
        self.selected_experts_indices = selected_experts_indices
        # top_scores /= top_scores.sum(dim=-1, keep_dim=True).to(x.dtype)

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        # Use bincount (int64) instead of histc (float32) to avoid float rounding errors.
        # histc float32 counts can be e.g. 44.9999 or 45.0001 for a true count of 45, causing
        # inconsistent truncation in _permute vs _forward_no_grouped_mm → shape mismatch in backward.
        num_tokens_per_expert = torch.bincount(
            selected_experts_indices.view(-1),
            minlength=self.num_experts,
        )  # int64 — exact integer counts, safe for allgather/alltoall split computation
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.experts_per_token
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert


class MoE(nn.Module):
    """This class implements the moe layer which is Mixture of Experts. Mixture of Experts
    typically consists of a set of expert networks, alongside with a router, which directs input tokens
    to the appropriate experts. See more details in https://arxiv.org/pdf/2407.06204.

    Args:
        experts (nn.Module): experts module.
        router (nn.Module): router module.
        shared_expert (Optional[nn.Module]): shared expert module. Default is None.
    """

    def __init__(
        self,
        *,
        experts: nn.Module,
        router: nn.Module,
        shared_expert: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert
        self.use_grouped_mm = should_use_grouped_mm()
        # EP dispatch/combine callables — set by setup code after parallelize_module.
        # If set, MoE.forward() calls these directly around self.experts(),
        # replacing the broken hook approach (FSDP2 fully_shard drops EP hooks).
        self._ep_dispatch: Optional[Callable] = None
        self._ep_combine: Optional[Callable] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        # top_scores and selected_indices shape (bs*slen*experts_per_token,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim))

        # shape (bs*slen*experts_per_token, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*experts_per_token, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )
        routed_input = routed_input * top_scores.reshape(-1, 1)

        if self.use_grouped_mm:
            # NOTE: In order to use torch._grouped_mm, we need to make sure
            # the number of tokens each expert gets is a multiple of 16.
            # The following kernel helps achieve this via padding, without
            # incurring synchronization between device and host.
            from torchtune.modules.moe.indices import generate_permute_indices

            ALIGN_SIZE_M = 16  # noqa

            with torch.no_grad():
                (
                    permuted_indices,
                    num_tokens_per_expert,
                    _,
                ) = generate_permute_indices(
                    num_tokens_per_expert,
                    self.experts.num_experts,
                    1,
                    ALIGN_SIZE_M,
                )
            token_indices = torch.vstack(
                (token_indices, token_indices.new_zeros((dim)))
            )
            token_indices = token_indices[permuted_indices, :]
            routed_input = torch.vstack((routed_input, routed_input.new_zeros((dim))))
            routed_input = routed_input[permuted_indices, :]

        # EP dispatch: route tokens to expert-owning ranks via All-to-All.
        # _ep_dispatch is set by setup code (not hooks — FSDP2 fully_shard drops
        # hooks registered on GroupedExperts by parallelize_module).
        if self._ep_dispatch is not None:
            routed_input, num_tokens_per_expert = self._ep_dispatch(
                routed_input, num_tokens_per_expert
            )

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # EP combine: reverse All-to-All to return outputs to originating ranks.
        if self._ep_combine is not None:
            routed_output = self._ep_combine(routed_output)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x).reshape(bs * slen, dim)
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))

        if self.use_grouped_mm:
            num_tokens = num_tokens_per_expert.sum().item()
            if torch.compiler.is_compiling():
                # Hints to compile dynamic shapes to pass through slice shape checks.
                torch._check_is_size(num_tokens)
                torch._check(num_tokens <= token_indices.size(0))
                torch._check(num_tokens <= routed_output.size(0))
            out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        else:
            out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class Gemma4MoeRouter(nn.Module):
    """
    Router for Gemma 4 26B-A4B MoE layers.

    Gemma 4's router differs from a simple linear gate:
    - Applies RMSNorm (no learnable scale — matches HF ``with_scale=False``) to input
    - Element-wise scales the normed input by a learned ``scale`` vector
    - Projects to expert logits via a linear layer
    - Scales logits per-expert via a learned ``per_expert_scale`` vector
    - Selects top-k experts via sigmoid + topk

    Returns the same ``(top_scores, token_indices, num_tokens_per_expert)`` tuple as
    ``TokenChoiceTopKRouter`` so it is a drop-in replacement inside ``MoE.forward()``.

    Args:
        hidden_dim (int): Model hidden dimension (input to router).
        num_experts (int): Total number of experts.
        top_k (int): Number of experts each token is routed to.
        norm_eps (float): Epsilon for RMSNorm. Default: 1e-6
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_eps = norm_eps
        # No learnable norm scale — HF router uses with_scale=False (pure RMSNorm).
        # This avoids 30 "missing key" errors when loading HF checkpoints with strict=True.
        self.proj = nn.Linear(hidden_dim, num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Flattened token tensor with shape ``(bs*slen, hidden_dim)``.

        Returns:
            top_scores (torch.Tensor): Routing weights for selected expert assignments,
                shape ``(bs*slen*top_k,)``, ordered by expert (expert-sorted).
            token_indices (torch.Tensor): Token indices (into the bs*slen axis) for each
                expert assignment, shape ``(bs*slen*top_k,)``, sorted by expert index.
            num_tokens_per_expert (torch.Tensor): Number of tokens routed to each expert,
                shape ``(num_experts,)``.
        """
        # Step 1: Normalize input (no learnable scale — matches HF with_scale=False), then scale
        normed = F.rms_norm(x.to(torch.float32), (x.shape[-1],), eps=self.norm_eps).to(x.dtype)
        normed = normed * self.scale                       # element-wise learned scale

        # Step 2: Project to expert logits and scale per expert
        logits = self.proj(normed)                         # [T, num_experts]
        logits = logits * self.per_expert_scale            # broadcast over T

        # Step 3: Sigmoid routing (not softmax — matches HF Gemma4 implementation)
        scores = torch.sigmoid(logits.to(torch.float32)).to(x.dtype)  # [T, num_experts]

        # Step 4: Select top-k experts per token.
        # Use argsort(stable=True) + slice instead of topk for deterministic XPU behavior.
        # torch.topk lacks stable=True on XPU → non-deterministic tie-breaking →
        # AC recompute produces different expert assignments → AllToAll size mismatch →
        # backward matmul shape crash. argsort(stable=True) breaks ties consistently.
        sorted_indices = torch.argsort(scores, dim=1, stable=True, descending=True)
        selected_experts = sorted_indices[:, : self.top_k]
        top_scores = torch.gather(scores, 1, selected_experts)
        # top_scores: [T, top_k], selected_experts: [T, top_k]

        # Step 5: Count tokens per expert using bincount (int64, exact) instead of
        # histc (float32) to avoid float rounding (e.g. 44.9999 → 45 truncates wrong).
        num_tokens_per_expert = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.num_experts,
        )  # int64 — exact integer counts

        # Step 6: Sort by expert index (group tokens going to same expert together)
        # selected_experts.view(-1) has shape [T*top_k] — sort by expert value
        token_indices_sorted = torch.argsort(selected_experts.reshape(-1), stable=True)
        # Reorder top_scores to match expert-sorted order
        top_scores = top_scores.reshape(-1)[token_indices_sorted]
        # Convert from position-in-flattened-topk to position-in-token-sequence
        token_indices_sorted = token_indices_sorted // self.top_k

        return top_scores, token_indices_sorted, num_tokens_per_expert

import os

import torch
from torch import nn


_USE_TOPK_ROUTING = os.environ.get("TORCHTUNE_MOE_TOPK_ROUTING", "0") == "1"
_USE_UNSTABLE_EXPERT_GROUPING = (
    os.environ.get("TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING", "0") == "1"
)


def _stable_select_experts(
    scores: torch.Tensor, experts_per_token: int
) -> torch.Tensor:
    """Select top experts with a fast tie-free path and stable fallback."""
    if not _USE_TOPK_ROUTING:
        return torch.argsort(scores, dim=1, stable=True, descending=True)[
            :, :experts_per_token
        ]
    if experts_per_token >= scores.shape[1]:
        return torch.argsort(scores, dim=1, stable=True, descending=True)[
            :, :experts_per_token
        ]

    candidate_scores, candidate_indices = torch.topk(
        scores,
        k=min(experts_per_token + 1, scores.shape[1]),
        dim=1,
        largest=True,
        sorted=True,
    )
    selected = candidate_indices[:, :experts_per_token].clone()
    boundary_ties = (
        candidate_scores.shape[1] > experts_per_token
        and candidate_scores[:, experts_per_token - 1]
        == candidate_scores[:, experts_per_token]
    )
    tied_rows = torch.nonzero(boundary_ties, as_tuple=True)[0]
    if tied_rows.numel() > 0:
        tied_scores = scores.index_select(0, tied_rows)
        stable_tied = torch.argsort(
            tied_scores, dim=1, stable=True, descending=True
        )[:, :experts_per_token]
        selected.index_copy_(0, tied_rows, stable_tied)
    return selected


class Qwen3MoeRouter(nn.Module):
    """Softmax-based top-k router for Qwen3 MoE.

    Unlike TokenChoiceTopKRouter (sigmoid), this uses softmax routing with
    optional top-k probability renormalization (norm_topk_prob).

    Args:
        gate: Linear projection to expert logits, typically nn.Linear(dim, num_experts, bias=False).
        dim: Input embedding dimension.
        num_experts: Total number of experts.
        experts_per_token: Number of experts each token is routed to (top-k).
        norm_topk_prob: If True, renormalize selected expert weights to sum to 1.
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        experts_per_token: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.norm_topk_prob = norm_topk_prob

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            top_scores: Expert-sorted routing weights with shape ``(bs*slen*experts_per_token,)``.
            token_indices: Expert-sorted token indices with shape ``(bs*slen*experts_per_token,)``.
            num_tokens_per_expert: Token count per expert with shape ``(num_experts,)``.
        """
        logits = self.gate(x).to(torch.float32)

        scores = torch.softmax(logits, dim=-1).to(x.dtype)
        if _USE_TOPK_ROUTING:
            # Select from the same cast probability tensor as the reference
            # path. Selecting from float32 logits changes ownership for BF16
            # near-ties even though the model's routed weights are BF16.
            selected_experts = _stable_select_experts(scores, self.experts_per_token)
        else:
            selected_experts = torch.argsort(
                scores, dim=1, stable=True, descending=True
            )[:, : self.experts_per_token]
        top_scores = torch.gather(scores, 1, selected_experts)

        if self.norm_topk_prob:
            denom = top_scores.sum(dim=-1, keepdim=True)
            top_scores = top_scores / denom.clamp(min=1e-8)

        self.selected_experts_indices = selected_experts
        self.selected_expert_scores = top_scores

        # Count tokens per expert using bincount (int64, exact — avoids float32 rounding
        # errors from histc that cause shape mismatches in backward).
        selected_experts_flat = selected_experts.reshape(-1)
        num_tokens_per_expert = torch.bincount(
            selected_experts_flat,
            minlength=self.num_experts,
        )

        # Sort tokens by expert index for grouped expert forward
        token_indices_experts_sorted = torch.argsort(
            selected_experts_flat, stable=not _USE_UNSTABLE_EXPERT_GROUPING
        )
        # .clone() forces a fresh, fully-owned dense allocation. Without it, this
        # fancy-indexed tensor's underlying storage/metadata (from the XPU
        # advanced-indexing kernel) causes `RuntimeError: NULL pointer argument
        # in memory copy operation` when activation offloading's saved_tensors_hooks
        # tries to pin-copy it to CPU (torchtune/training/_activation_offloading.py's
        # pack_tensor) — HW-caught on Qwen3-30B-A3B EP=8 seq4096 with
        # enable_activation_offloading=true. A plain .reshape(-1)[idx] should
        # already produce a dense tensor in PyTorch, so this is an XPU-specific
        # backend quirk, not expected behavior — but .clone() is a values-only
        # no-op and cheap relative to the crash it avoids.
        top_scores = top_scores.reshape(-1)[token_indices_experts_sorted].clone()
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.experts_per_token
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

import torch
from torch import nn


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
        # [T, num_experts]
        scores = self.gate(x)

        # Softmax in float32 for numerical stability
        scores = torch.softmax(scores.to(torch.float32), dim=-1).to(x.dtype)

        # Deterministic top-k via argsort(stable=True) — avoids XPU non-deterministic
        # tie-breaking in torch.topk which causes AC recompute mismatches.
        sorted_indices = torch.argsort(scores, dim=1, stable=True, descending=True)
        selected_experts = sorted_indices[:, : self.experts_per_token]
        top_scores = torch.gather(scores, 1, selected_experts)

        # Renormalize selected expert probabilities to sum to 1
        if self.norm_topk_prob:
            denom = top_scores.sum(dim=-1, keepdim=True)
            top_scores = top_scores / denom.clamp(min=1e-8)

        # Count tokens per expert using bincount (int64, exact — avoids float32 rounding
        # errors from histc that cause shape mismatches in backward).
        num_tokens_per_expert = torch.bincount(
            selected_experts.reshape(-1),
            minlength=self.num_experts,
        )

        # Sort tokens by expert index for grouped expert forward
        token_indices_experts_sorted = torch.argsort(
            selected_experts.reshape(-1), stable=True
        )
        top_scores = top_scores.reshape(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.experts_per_token
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

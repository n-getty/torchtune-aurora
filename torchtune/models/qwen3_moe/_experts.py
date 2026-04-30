import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


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
        E = num_tokens_per_expert.shape[0]
        total = x.shape[0]

        if total == 0:
            x_zero = x.reshape(0, self.dim) if x.numel() == 0 else x.new_empty(0, self.dim)
            anchor = (self.gate_proj.sum() + self.down_proj.sum() + self.up_proj.sum()) * 0.0
            x_anchor = (x.sum(dim=0, keepdim=False) * 0.0) if x.requires_grad else None
            if x_anchor is not None:
                anchor = anchor + x_anchor.sum()
            return x_zero + anchor

        counts = num_tokens_per_expert.round().to(torch.int64)
        count_list = counts.tolist()
        max_count = int(max(count_list))

        # Scatter: pack sorted tokens into [E, max_count, dim] padded tensor.
        # Zero-padded positions contribute zero to bmm output and zero gradient.
        x_padded = x.new_zeros(E, max_count, self.dim)
        offset = 0
        for e in range(E):
            c = int(count_list[e])
            if c > 0:
                x_padded[e, :c] = x[offset : offset + c]
                offset += c

        # 3 batched matmuls replace E×3 sequential matmuls (128→1 kernel launch)
        gate_out = torch.bmm(x_padded, self.gate_proj.mT)
        h = self.act_fn(gate_out)
        h = h * torch.bmm(x_padded, self.up_proj.mT)
        out_padded = torch.bmm(h, self.down_proj.mT)

        # Gather: extract results back to flat tensor
        out = torch.empty(total, self.dim, dtype=x.dtype, device=x.device)
        offset = 0
        for e in range(E):
            c = int(count_list[e])
            if c > 0:
                out[offset : offset + c] = out_padded[e, :c]
                offset += c
        return out

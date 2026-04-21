"""Benchmark padded-bmm approach vs loop for XPU MoE experts forward.

GroupedExperts receives:
  x: [total_tokens, dim]  (pre-sorted by expert)
  num_tokens_per_expert: [num_experts]

Padded-bmm packs each expert's tokens into x_padded [E, max_T, D],
does 3 bmms, then unpacks. Key question: is memory overhead acceptable?
"""
import sys, time
import torch
import torch.nn.functional as F

device = torch.device("xpu:0")
hidden_dim, inter_dim, num_experts, top_k = 2816, 704, 128, 8

gate = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
up   = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
down = torch.randn(num_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16)


def forward_loop(x, num_tokens_per_expert):
    x_splits = torch.split(x, num_tokens_per_expert.tolist(), dim=0)
    outs = []
    for i, xe in enumerate(x_splits):
        if xe.shape[0] == 0:
            continue
        h = F.gelu(xe @ gate[i], approximate="tanh") * (xe @ up[i])
        outs.append(h @ down[i])
    return torch.cat(outs, dim=0) if outs else x.new_empty(0, hidden_dim)


def forward_bmm_padded(x, num_tokens_per_expert):
    """Pack tokens into [E, max_T, D], do 3 bmms, unpack."""
    max_t = int(num_tokens_per_expert.max().item())
    if max_t == 0:
        return x.new_empty(0, hidden_dim)
    E = num_experts

    # Build padded input [E, max_T, D]
    x_padded = x.new_zeros(E, max_t, hidden_dim)
    offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(num_tokens_per_expert, dim=0)
    for e in range(E):
        n = int(num_tokens_per_expert[e].item())
        if n > 0:
            x_padded[e, :n] = x[offsets[e]:offsets[e] + n]

    # 3 bmms: [E, max_T, D] @ [E, D, inter] = [E, max_T, inter]
    g = x_padded @ gate  # [E, max_T, inter]
    u = x_padded @ up    # [E, max_T, inter]
    h = F.gelu(g, approximate="tanh") * u
    o = h @ down         # [E, max_T, D]

    # Unpack: gather non-padded rows back into [total_tokens, D]
    total = int(x.shape[0])
    out = x.new_empty(total, hidden_dim)
    for e in range(E):
        n = int(num_tokens_per_expert[e].item())
        if n > 0:
            out[offsets[e]:offsets[e] + n] = o[e, :n]
    return out


def forward_bmm_scatter(x, num_tokens_per_expert):
    """Use scatter/gather instead of Python loop for packing/unpacking."""
    max_t = int(num_tokens_per_expert.max().item())
    if max_t == 0:
        return x.new_empty(0, hidden_dim)

    E = num_experts
    total = x.shape[0]

    # Build position-within-expert index for each token row
    # expert_for_token: which expert each row of x belongs to
    # We know tokens are sorted by expert, so:
    expert_ids = torch.repeat_interleave(
        torch.arange(E, device=device),
        num_tokens_per_expert.to(torch.int64)
    )  # [total]
    pos_within_expert = (torch.arange(total, device=device) -
                         torch.cumsum(num_tokens_per_expert, 0)[expert_ids] +
                         num_tokens_per_expert[expert_ids]).to(torch.int64)
    # Simpler: cumsum gives start of each expert's block
    offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(num_tokens_per_expert, 0)
    pos_within_expert = torch.arange(total, device=device) - offsets[expert_ids]

    # Scatter x into [E, max_T, D]
    x_padded = x.new_zeros(E, max_t, hidden_dim)
    x_padded[expert_ids, pos_within_expert] = x  # scatter by (expert, position)

    # 3 bmms
    g = x_padded @ gate
    u = x_padded @ up
    h = F.gelu(g, approximate="tanh") * u
    o = h @ down  # [E, max_T, D]

    # Gather back
    out = o[expert_ids, pos_within_expert]  # [total, D]
    return out


def benchmark(fn, x, counts, name, n=10):
    # warmup
    fn(x, counts)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x, counts)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {name}: {ms:.2f}ms")


print("=== MoE Expert Forward Benchmarks ===")
for num_tokens in [1, 4, 16, 64, 256, 512]:
    total = num_tokens * top_k
    ti = torch.randint(0, num_experts, (num_tokens * top_k,), device=device, dtype=torch.int64)
    ti, _ = torch.sort(ti)  # tokens sorted by expert
    counts = torch.bincount(ti, minlength=num_experts).to(torch.int64)
    x = torch.randn(total, hidden_dim, device=device, dtype=torch.bfloat16)
    max_t = int(counts.max().item())
    print(f"\nT={num_tokens} (total={total}, max_per_expert={max_t}):")
    benchmark(forward_loop, x, counts, "loop")
    benchmark(forward_bmm_padded, x, counts, "bmm_padded (python pack)")
    benchmark(forward_bmm_scatter, x, counts, "bmm_scatter (index ops)")

print("\nDone.")

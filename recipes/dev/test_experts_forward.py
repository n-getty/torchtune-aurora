"""Test that the new bmm-based _forward_no_grouped_mm gives same results as old loop."""
import sys
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune")
import torch
import torch.nn.functional as F
from torchtune.modules.moe.experts import GroupedExperts

device = torch.device("xpu:0")
torch.manual_seed(42)

hidden_dim, inter_dim, num_experts, top_k = 2816, 704, 128, 8

experts = GroupedExperts(
    dim=hidden_dim,
    hidden_dim=inter_dim,
    num_experts=num_experts,
    activation=lambda x: F.gelu(x, approximate="tanh"),
).to(device=device, dtype=torch.bfloat16)

# Old loop reference implementation
def forward_loop_ref(x, num_tokens_per_expert):
    x_splits = torch.split(x, num_tokens_per_expert.tolist(), dim=0)
    outs = []
    for i, xe in enumerate(x_splits):
        if xe.shape[0] == 0:
            continue
        w1, w2, w3 = experts.gate_proj[i], experts.down_proj[i], experts.up_proj[i]
        h = F.gelu(xe @ w1, approximate="tanh") * (xe @ w3)
        outs.append(h @ w2)
    return torch.cat(outs, dim=0)

print("Correctness tests:")
for num_tokens in [1, 4, 16, 64]:
    total = num_tokens * top_k
    ti = torch.randint(0, num_experts, (total,), device=device, dtype=torch.int64)
    ti, _ = torch.sort(ti)
    counts = torch.bincount(ti, minlength=num_experts).to(torch.int64)
    x = torch.randn(total, hidden_dim, device=device, dtype=torch.bfloat16)

    ref = forward_loop_ref(x, counts)
    new = experts._forward_no_grouped_mm(x, counts)

    max_diff = (ref - new).abs().max().item()
    print(f"  T={num_tokens:4d}: max_diff={max_diff:.6f}  {'PASS' if max_diff < 0.01 else 'FAIL'}")

import time

print("\nPerformance (new bmm vs old loop):")
for num_tokens in [1, 4, 16, 64, 256, 512]:
    total = num_tokens * top_k
    ti = torch.randint(0, num_experts, (total,), device=device, dtype=torch.int64)
    ti, _ = torch.sort(ti)
    counts = torch.bincount(ti, minlength=num_experts).to(torch.int64)
    x = torch.randn(total, hidden_dim, device=device, dtype=torch.bfloat16)

    # New
    experts._forward_no_grouped_mm(x, counts)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        experts._forward_no_grouped_mm(x, counts)
    torch.xpu.synchronize()
    ms_new = (time.perf_counter() - t0) / 10 * 1000

    # Old loop
    forward_loop_ref(x, counts)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        forward_loop_ref(x, counts)
    torch.xpu.synchronize()
    ms_old = (time.perf_counter() - t0) / 10 * 1000

    print(f"  T={num_tokens:4d}: new={ms_new:.2f}ms  old={ms_old:.2f}ms  speedup={ms_old/ms_new:.1f}x")

print("Done.")

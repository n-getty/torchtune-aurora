"""Benchmark Triton fused_moe vs loop baseline for Gemma4 26B-A4B dimensions."""
import sys, time, os
sys.path.insert(0, "/flare/AuroraGPT/ngetty/torchtitan_xpu/torchtitan/torchtitan/models/moe")
import torch
import torch.nn.functional as F
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
from triton_fused_moe_xpu import fused_moe

device = torch.device("xpu:0")
hidden_dim, inter_dim, num_experts, top_k = 2816, 704, 128, 8

w1 = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
w2 = torch.randn(num_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16)
w3 = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)

# --- Triton ---
print("=== Triton fused_moe ===")
for num_tokens in [1, 4, 16, 64, 512]:
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    tw = torch.rand(num_tokens, top_k, device=device, dtype=torch.bfloat16)
    ti = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    # warmup
    _ = fused_moe(hidden, w1, w2, w3, tw, ti)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        fused_moe(hidden, w1, w2, w3, tw, ti)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  T={num_tokens:4d}: {ms:.1f}ms")

# --- Loop baseline ---
print("=== Loop baseline ===")
for num_tokens in [1, 4, 16]:
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    ti = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    counts = torch.bincount(ti.flatten().to(torch.int64), minlength=num_experts)
    x = hidden.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    x_splits = torch.split(x, counts.tolist(), dim=0)
    t0 = time.perf_counter()
    outs = []
    for i, xe in enumerate(x_splits):
        if xe.shape[0] == 0:
            continue
        h = F.gelu(xe @ w1[i], approximate="tanh") * (xe @ w3[i])
        outs.append(h @ w2[i])
    if outs:
        torch.cat(outs)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  T={num_tokens:4d}: {ms:.1f}ms")

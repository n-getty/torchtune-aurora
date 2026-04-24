"""Benchmark IPEX GatedMLPMOE vs loop baseline vs Triton for Gemma4 26B-A4B dimensions."""
import sys, time, os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
sys.path.insert(0, "/flare/AuroraGPT/ngetty/torchtitan_xpu/torchtitan/torchtitan/models/moe")
import torch
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.llm.modules import GatedMLPMOE

device = torch.device("xpu:0")
hidden_dim, inter_dim, num_experts, top_k = 2816, 704, 128, 8

# IPEX uses w13 [E, 2*inter, hidden] (fused gate+up) and w2 [E, hidden, inter]
# But check if it's [E, inter, hidden] for w2...
# From vLLM source: w13=[E, 2*inter, hidden], w2=[E, hidden, inter]
w13 = torch.randn(num_experts, 2 * inter_dim, hidden_dim, device=device, dtype=torch.bfloat16)
w2  = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)

# Also prep loop weights
w1_loop = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
w3_loop = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
w2_loop = torch.randn(num_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16)

# --- IPEX GatedMLPMOE ---
print("=== IPEX GatedMLPMOE ===")
moe_module = GatedMLPMOE(w13, w2, use_prepack=True)

for num_tokens in [1, 4, 16, 64, 512]:
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    # Fake router logits: shape [num_tokens, num_experts]
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

    # Warmup
    try:
        out = moe_module(hidden, use_grouped_topk=False, top_k=top_k,
                         router_logits=router_logits, renormalize=False)
        torch.xpu.synchronize()
    except Exception as e:
        print(f"  T={num_tokens:4d}: ERROR warmup: {e}")
        break

    t0 = time.perf_counter()
    for _ in range(10):
        out = moe_module(hidden, use_grouped_topk=False, top_k=top_k,
                         router_logits=router_logits, renormalize=False)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"  T={num_tokens:4d}: {ms:.2f}ms  out_shape={out.shape}")

# --- IPEX with prepack=False for comparison ---
print("\n=== IPEX GatedMLPMOE (no prepack) ===")
moe_module_np = GatedMLPMOE(w13, w2, use_prepack=False)

for num_tokens in [1, 4, 16, 64]:
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)

    try:
        out = moe_module_np(hidden, use_grouped_topk=False, top_k=top_k,
                            router_logits=router_logits, renormalize=False)
        torch.xpu.synchronize()
    except Exception as e:
        print(f"  T={num_tokens:4d}: ERROR: {e}")
        break

    t0 = time.perf_counter()
    for _ in range(10):
        out = moe_module_np(hidden, use_grouped_topk=False, top_k=top_k,
                            router_logits=router_logits, renormalize=False)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"  T={num_tokens:4d}: {ms:.2f}ms")

# --- Loop baseline (reference) ---
print("\n=== Loop baseline ===")
for num_tokens in [1, 4, 16, 64, 512]:
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    ti = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64)
    counts = torch.bincount(ti.flatten(), minlength=num_experts)
    x = hidden.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    x_splits = torch.split(x, counts.tolist(), dim=0)

    # Warmup
    outs = []
    for i, xe in enumerate(x_splits):
        if xe.shape[0] == 0:
            continue
        h = F.gelu(xe @ w1_loop[i], approximate="tanh") * (xe @ w3_loop[i])
        outs.append(h @ w2_loop[i])
    if outs:
        torch.cat(outs)
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        outs = []
        for i, xe in enumerate(x_splits):
            if xe.shape[0] == 0:
                continue
            h = F.gelu(xe @ w1_loop[i], approximate="tanh") * (xe @ w3_loop[i])
            outs.append(h @ w2_loop[i])
        if outs:
            torch.cat(outs)
    torch.xpu.synchronize()
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  T={num_tokens:4d}: {ms:.2f}ms")

# --- Triton for comparison ---
print("\n=== Triton fused_moe ===")
try:
    from triton_fused_moe_xpu import fused_moe
    w1_t = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)
    w2_t = torch.randn(num_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16)
    w3_t = torch.randn(num_experts, hidden_dim, inter_dim, device=device, dtype=torch.bfloat16)

    for num_tokens in [1, 4, 16, 64, 512]:
        hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
        tw = torch.rand(num_tokens, top_k, device=device, dtype=torch.bfloat16)
        ti_t = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)

        _ = fused_moe(hidden, w1_t, w2_t, w3_t, tw, ti_t)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            fused_moe(hidden, w1_t, w2_t, w3_t, tw, ti_t)
        torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"  T={num_tokens:4d}: {ms:.2f}ms")
except Exception as e:
    print(f"  Triton error: {e}")

print("\nDone.")

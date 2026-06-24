"""
Micro-benchmark: PyTorch SDPA vs IPEX varlen_attention IPC handle / memory traffic.

Goal: confirm whether varlen_attention reduces per-step transient allocations
at BioReason-realistic shapes (Qwen3-4B GQA, prompt+gen up to 1024+P tokens).

We do NOT have direct visibility into L0 IPC handle registration counts, but
we can use these proxies:
  - torch.xpu.memory_stats()['num_alloc_retries'], ['allocation.all.allocated']
    (number of allocator alloc events per attention call)
  - delta of reserved memory per call
  - peak allocated during each call
  - wall time

Shapes match Qwen3-4B at G=8, fbs=8, max_gen=1024, prompt~512: total tokens
per chunk ~1536 * 8 = 12288 packed (varlen) or [8, 32, 1536, 128] padded (SDPA).

Run on a held compute node:
    module load frameworks
    ZE_AFFINITY_MASK=0 python3 /tmp/bench_attention_ipc.py
"""
import os
import time
import torch

# ---- Config ----
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16

# BioReason-realistic shapes
BATCH = 8           # G=8 fbs=8
SEQ_LEN = 1536      # 512 prompt + 1024 gen
NUM_LAYERS = 36     # Qwen3-4B layer count
NUM_ITERS = 10      # iterations per backend (warmup excluded)

device = torch.device("xpu:0")
torch.manual_seed(0)


def bench(name: str, fn):
    """Run fn() NUM_ITERS times after 1 warmup; report mem and timing deltas."""
    # Warmup
    fn()
    torch.xpu.synchronize()

    torch.xpu.empty_cache()  # baseline
    torch.xpu.reset_peak_memory_stats()

    pre_alloc = torch.xpu.memory_allocated()
    pre_resv = torch.xpu.memory_reserved()
    pre_stats = torch.xpu.memory_stats()
    pre_alloc_count = pre_stats.get("allocation.all.allocated", 0)

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        out = fn()
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    post_stats = torch.xpu.memory_stats()
    post_alloc_count = post_stats.get("allocation.all.allocated", 0)
    peak_alloc = torch.xpu.max_memory_allocated()
    post_resv = torch.xpu.memory_reserved()

    print(f"\n=== {name} ===")
    print(f"  per-iter wall: {(t1-t0)/NUM_ITERS*1000:.1f} ms")
    print(f"  alloc events over {NUM_ITERS} iters: {post_alloc_count - pre_alloc_count} "
          f"({(post_alloc_count - pre_alloc_count) / NUM_ITERS:.1f} per iter)")
    print(f"  peak alloc during run: {peak_alloc / 1024**2:.1f} MiB "
          f"(delta vs pre: {(peak_alloc - pre_alloc) / 1024**2:.1f} MiB)")
    print(f"  reserved delta: {(post_resv - pre_resv) / 1024**2:.1f} MiB")

    del out
    return {
        "per_iter_ms": (t1 - t0) / NUM_ITERS * 1000,
        "alloc_per_iter": (post_alloc_count - pre_alloc_count) / NUM_ITERS,
        "peak_mib": peak_alloc / 1024**2,
        "resv_delta_mib": (post_resv - pre_resv) / 1024**2,
    }


# =============================================================
# Setup tensors (allocated once, reused across both backends)
# =============================================================
print(f"Device: {device}, dtype: {DTYPE}")
print(f"Shape: B={BATCH} S={SEQ_LEN} Hq={NUM_Q_HEADS} Hkv={NUM_KV_HEADS} D={HEAD_DIM}")
print(f"Per-chunk q_tensor: {BATCH * SEQ_LEN * NUM_Q_HEADS * HEAD_DIM * 2 / 1024**2:.1f} MiB")

# SDPA-style: [B, H, S, D]
q_sdpa = torch.randn(BATCH, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=device)
k_sdpa = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=device)
v_sdpa = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=device)

# Varlen: [total_tokens, H, D] packed
total_tokens = BATCH * SEQ_LEN
q_varlen = torch.randn(total_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
k_varlen = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
v_varlen = torch.randn(total_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
out_varlen_persistent = torch.empty_like(q_varlen)
seqlen_q = torch.arange(0, BATCH * SEQ_LEN + 1, SEQ_LEN, dtype=torch.int32, device=device)
seqlen_k = seqlen_q.clone()
softmax_scale = 1.0 / (HEAD_DIM ** 0.5)


# =============================================================
# Backend 1: PyTorch SDPA (current 2-node config: optimized flash/mem-eff)
# =============================================================
def sdpa_optimized():
    # GQA expansion
    k = k_sdpa.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    v = v_sdpa.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k, v, is_causal=True
    )


# =============================================================
# Backend 2: PyTorch SDPA, math-only (current 1-node baseline)
# =============================================================
def sdpa_math_only():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    k = k_sdpa.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    v = v_sdpa.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k, v, is_causal=True
    )
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    return out


# =============================================================
# Backend 3: IPEX varlen_attention with NEW output buffer per call
# =============================================================
from intel_extension_for_pytorch.llm.functional import varlen_attention


alibi_slopes = torch.zeros(NUM_Q_HEADS, dtype=torch.float32, device=device)


def varlen_fresh_buffer():
    out = torch.empty_like(q_varlen)
    varlen_attention(
        q_varlen, k_varlen, v_varlen, out,
        seqlen_q, seqlen_k,
        alibi_slopes,
        SEQ_LEN, SEQ_LEN,
        0.0, softmax_scale,
        False, True, False, None,
    )
    return out


# =============================================================
# Backend 4: IPEX varlen_attention with PERSISTENT output buffer (the win)
# =============================================================
def varlen_persistent_buffer():
    varlen_attention(
        q_varlen, k_varlen, v_varlen, out_varlen_persistent,
        seqlen_q, seqlen_k,
        alibi_slopes,
        SEQ_LEN, SEQ_LEN,
        0.0, softmax_scale,
        False, True, False, None,
    )
    return out_varlen_persistent


# =============================================================
# Run all
# =============================================================
results = {}
results["sdpa_optimized"] = bench("PyTorch SDPA (flash/mem-eff, current 2-node)", sdpa_optimized)
results["sdpa_math_only"] = bench("PyTorch SDPA (math-only, current 1-node)", sdpa_math_only)
results["varlen_fresh_buf"] = bench("IPEX varlen (fresh output buffer per call)", varlen_fresh_buffer)
results["varlen_persistent"] = bench("IPEX varlen (persistent output buffer)", varlen_persistent_buffer)

# =============================================================
# Multi-layer simulation: 36 attention calls per fwd pass (Qwen3-4B layer count)
# =============================================================
print("\n" + "=" * 70)
print(f"Simulating {NUM_LAYERS}-layer fwd (Qwen3-4B), single chunk")
print("=" * 70)


def multilayer(call):
    for _ in range(NUM_LAYERS):
        out = call()
    return out


for name, call in [
    ("sdpa_optimized", sdpa_optimized),
    ("sdpa_math_only", sdpa_math_only),
    ("varlen_fresh", varlen_fresh_buffer),
    ("varlen_persistent", varlen_persistent_buffer),
]:
    bench(f"{NUM_LAYERS}-layer {name}", lambda c=call: multilayer(c))

print("\n" + "=" * 70)
print("Summary (single attention call):")
print("=" * 70)
print(f"{'Backend':<35} {'ms/iter':>8} {'allocs/iter':>12} {'peak MiB':>10}")
for name, r in results.items():
    print(f"{name:<35} {r['per_iter_ms']:>8.1f} {r['alloc_per_iter']:>12.1f} {r['peak_mib']:>10.1f}")

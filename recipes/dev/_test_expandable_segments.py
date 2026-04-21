"""Test whether expandable_segments actually changes XPU allocator behavior.

Run twice:
  PYTORCH_ALLOC_CONF= python3 _test_expandable_segments.py
  PYTORCH_ALLOC_CONF=expandable_segments:True python3 _test_expandable_segments.py

Single-process (no CCL) — safe to run with expandable_segments since the
USM pointer issue only affects oneCCL collectives.
"""
import os
import torch

device = torch.device("xpu:0")
torch.xpu.set_device(0)

alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "unset")
print(f"PYTORCH_ALLOC_CONF: {alloc_conf}")
print(f"torch version: {torch.__version__}")

def get_stats():
    s = torch.xpu.memory_stats(0)
    return {
        "allocated_MB": s.get("allocated_bytes.all.current", 0) / 1e6,
        "reserved_MB": s.get("reserved_bytes.all.current", 0) / 1e6,
        "num_alloc_retries": s.get("num_alloc_retries", 0),
        "num_ooms": s.get("num_ooms", 0),
    }

def fmt(stats):
    return f"alloc={stats['allocated_MB']:.1f}MB resv={stats['reserved_MB']:.1f}MB retries={stats['num_alloc_retries']} ooms={stats['num_ooms']}"

print(f"\n=== Phase 1: Allocate-free-reallocate cycle (fragmentation test) ===")
tensors = []
for i in range(20):
    tensors.append(torch.randn(1024, 1024, device=device))  # ~4MB each
print(f"After 20 x 4MB allocs: {fmt(get_stats())}")

# Free every other tensor (creates holes)
for i in range(0, 20, 2):
    tensors[i] = None
torch.xpu.synchronize()
print(f"After freeing evens:   {fmt(get_stats())}")

# Try to allocate a large contiguous block (should fail or cause fragmentation without expandable)
try:
    big = torch.randn(10240, 1024, device=device)  # ~40MB
    print(f"40MB contiguous alloc: {fmt(get_stats())} — SUCCESS")
    del big
except RuntimeError as e:
    print(f"40MB contiguous alloc: FAILED — {e}")

print(f"\n=== Phase 2: Growing allocation pattern ===")
torch.xpu.empty_cache()
torch.xpu.reset_peak_memory_stats()
del tensors

sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # MB
for size_mb in sizes:
    n = size_mb * 1024 * 1024 // 4  # float32 elements
    t = torch.randn(n, device=device)
    s = get_stats()
    gap = s["reserved_MB"] - s["allocated_MB"]
    print(f"  {size_mb:4d}MB alloc: {fmt(s)}, gap={gap:.1f}MB")
    del t

print(f"\n=== Phase 3: Repeated alloc/free stress ===")
torch.xpu.empty_cache()
torch.xpu.reset_peak_memory_stats()

for iteration in range(5):
    tensors = [torch.randn(2048, 2048, device=device) for _ in range(10)]  # 10 x 16MB
    del tensors
    torch.xpu.synchronize()

s = get_stats()
print(f"After 5 rounds of 10x16MB: {fmt(s)}")
print(f"Peak allocated: {torch.xpu.max_memory_allocated(0)/1e6:.1f}MB")
print(f"Peak reserved:  {torch.xpu.max_memory_reserved(0)/1e6:.1f}MB")
print(f"\nDone.")

"""
Minimal test: arena allocator + FSDP2 unshard pattern.

Tests whether the usm_arena_alloc.so allocator is compatible with FSDP2
unshard (allgather) + reshard (free) cycles. Reproduces the hang observed
in 32B GRPO policy forward.

Usage:
    # Without arena (baseline):
    torchrun --nproc_per_node=4 recipes/dev/test_arena_fsdp.py

    # With arena allocator:
    XPU_USM_ALLOC_SO=recipes/dev/usm_arena_alloc.so \
        torchrun --nproc_per_node=4 recipes/dev/test_arena_fsdp.py

    # With arena debug:
    XPU_USM_ALLOC_SO=recipes/dev/usm_arena_alloc.so USM_ALLOC_DEBUG=1 \
        torchrun --nproc_per_node=4 recipes/dev/test_arena_fsdp.py 2>arena_debug.log
"""
import os
import sys
import time

# --- Arena allocator registration (must be before any XPU init) ---
_usm_so = os.environ.get("XPU_USM_ALLOC_SO")
if _usm_so:
    import torch
    from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
    _alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
    change_current_allocator(_alloc)
    torch.xpu.memory_allocated = lambda device=None: 0
    torch.xpu.memory_reserved = lambda device=None: 0
    torch.xpu.reset_peak_memory_stats = lambda device=None: None
    torch.xpu.empty_cache = lambda: None
    print(f"[rank ?] Arena allocator registered: {_usm_so}", flush=True)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

def main():
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(device)

    print(f"[rank {rank}] device={device}, world_size={world_size}, arena={_usm_so or 'OFF'}", flush=True)

    # Build a model similar to one layer of Qwen 32B:
    # hidden=5120, intermediate=27648, 64 layers → we test 1 layer
    hidden = 5120
    intermediate = 27648

    class FakeTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden // 8, bias=False)  # GQA
            self.v_proj = nn.Linear(hidden, hidden // 8, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)

        def forward(self, x):
            h = self.norm1(x)
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)
            h = self.o_proj(q)  # simplified
            x = x + h
            h = self.norm2(x)
            h = self.gate_proj(h) * self.up_proj(h)
            h = self.down_proj(h)
            return x + h

    class FakeModel(nn.Module):
        def __init__(self, n_layers=8):
            super().__init__()
            self.embed = nn.Embedding(32000, hidden)
            self.layers = nn.ModuleList([FakeTransformerLayer() for _ in range(n_layers)])
            self.head = nn.Linear(hidden, 32000, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    n_layers = 8  # Enough to stress-test the alloc pattern
    model = FakeModel(n_layers=n_layers).to(dtype=torch.bfloat16, device=device)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if rank == 0:
        print(f"Model: {n_layers} layers, {param_bytes / 1e9:.2f} GB, {sum(1 for _ in model.parameters())} params", flush=True)

    # Apply FSDP2 — top-level only (matching production config)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)

    if rank == 0:
        print(f"FSDP2 applied (reshard_after_forward=True)", flush=True)

    # Warmup
    dummy_input = torch.randint(0, 32000, (2, 128), device=device)
    if rank == 0:
        print("Starting warmup forward...", flush=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy_input)
    torch.xpu.synchronize(device)
    t_warmup = time.perf_counter() - t0
    if rank == 0:
        print(f"Warmup forward: {t_warmup:.2f}s", flush=True)

    # Timed forward passes (this is the pattern that hangs with arena allocator)
    for i in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy_input)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        if rank == 0:
            print(f"Forward {i}: {dt:.3f}s", flush=True)

    # Test backward too (optimizer step involves more alloc/free)
    if rank == 0:
        print("Starting forward+backward...", flush=True)

    model.train()
    for i in range(3):
        t0 = time.perf_counter()
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        if rank == 0:
            print(f"Fwd+bwd {i}: {dt:.3f}s", flush=True)

    if rank == 0:
        print("ALL TESTS PASSED", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

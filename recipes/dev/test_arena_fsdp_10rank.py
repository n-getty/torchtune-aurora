"""
Test arena allocator + FSDP2 at 10-rank scale matching 32B GRPO layout.

Key difference from test_arena_fsdp.py: 10 ranks (matching production),
64 layers (matching Qwen 32B), per-layer FSDP sharding with prefetch.

If this works but real recipe hangs, the issue is recipe-specific.
If this hangs too, the issue is allocator + FSDP at scale.

Usage (10 tiles, matching production layout):
    ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7,8,9 \
        torchrun --nproc_per_node=10 recipes/dev/test_arena_fsdp_10rank.py [--arena]
"""
import os
import sys
import time
import argparse

use_arena = "--arena" in sys.argv or os.environ.get("XPU_USM_ALLOC_SO")

if use_arena:
    _usm_so = os.environ.get("XPU_USM_ALLOC_SO",
                              "recipes/dev/usm_arena_alloc.so")
    import torch
    from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
    _alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
    change_current_allocator(_alloc)
    torch.xpu.memory_allocated = lambda device=None: 0
    torch.xpu.memory_reserved = lambda device=None: 0
    torch.xpu.reset_peak_memory_stats = lambda device=None: None
    torch.xpu.empty_cache = lambda: None

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from copy import deepcopy

def main():
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(device)

    # Qwen 32B dimensions
    hidden = 5120
    intermediate = 27648
    n_heads = 40
    n_kv_heads = 8
    head_dim = 128
    n_layers = 64
    vocab = 152064

    class TransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.norm1 = nn.RMSNorm(hidden)
            self.norm2 = nn.RMSNorm(hidden)

        def forward(self, x):
            h = self.norm1(x)
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)
            h = self.o_proj(q)
            x = x + h
            h = self.norm2(x)
            h = self.gate_proj(h) * self.up_proj(h)
            h = self.down_proj(h)
            return x + h

    class FakeQwen32B(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([TransformerLayer() for _ in range(n_layers)])
            self.norm = nn.RMSNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for i, layer in enumerate(self.layers):
                t0 = time.perf_counter()
                x = layer(x)
                if rank == 0 and i % 16 == 0:
                    dt = time.perf_counter() - t0
                    print(f"  layer {i}: {dt*1000:.1f}ms", flush=True)
            x = self.norm(x)
            return self.head(x)

    if rank == 0:
        print(f"Building model: {n_layers} layers, hidden={hidden}, "
              f"world_size={world_size}, arena={'ON' if use_arena else 'OFF'}", flush=True)

    model = FakeQwen32B().to(dtype=torch.bfloat16, device=device)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_params = sum(1 for _ in model.parameters())
    if rank == 0:
        print(f"Model: {param_bytes / 1e9:.2f} GB, {n_params} params", flush=True)

    # Per-layer FSDP sharding (matching shard_model pattern)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fsdp_kwargs = {"reshard_after_forward": True, "mp_policy": mp_policy}

    n_sharded = 0
    for name, mod in reversed(list(model.named_modules())):
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            fully_shard(mod, **fsdp_kwargs)
            n_sharded += 1

    # Root shard with prefetch (reshard_after_forward=None)
    root_kwargs = deepcopy(fsdp_kwargs)
    try:
        root_kwargs["reshard_after_forward"] = None
        fully_shard(model, **root_kwargs)
    except (TypeError, ValueError):
        root_kwargs["reshard_after_forward"] = False
        fully_shard(model, **root_kwargs)

    if rank == 0:
        print(f"FSDP2 applied: {n_sharded} layers sharded, root with prefetch", flush=True)

    # Forward pass test
    dummy_input = torch.randint(0, vocab, (2, 128), device=device)

    if rank == 0:
        print("=== Warmup forward ===", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy_input)
    torch.xpu.synchronize(device)
    if rank == 0:
        print(f"Warmup: {time.perf_counter() - t0:.2f}s", flush=True)

    if rank == 0:
        print("=== Timed forwards ===", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy_input)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        if rank == 0:
            print(f"Forward {i}: {dt:.3f}s", flush=True)

    if rank == 0:
        print("ALL TESTS PASSED", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

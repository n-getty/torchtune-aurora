"""
Stress test: arena allocator + FSDP2 at 32B-like memory pressure.

Scales up model to fill ~50+ GB per tile (matching 32B GRPO training state).
The original hang occurred AFTER vLLM generation completed — during the
policy forward pass, which means the arena had already been used for
model init + generation and was partially fragmented.

Usage:
    # Without arena (baseline):
    ZE_AFFINITY_MASK=0,1,2,3 torchrun --nproc_per_node=4 recipes/dev/test_arena_fsdp_stress.py

    # With arena:
    XPU_USM_ALLOC_SO=recipes/dev/usm_arena_alloc.so \
        ZE_AFFINITY_MASK=0,1,2,3 torchrun --nproc_per_node=4 recipes/dev/test_arena_fsdp_stress.py
"""
import os
import sys
import time

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
    print(f"[rank ?] Arena allocator registered", flush=True)

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

    # Match 32B model dimensions: hidden=5120, intermediate=27648, 64 layers
    # With 4-way FSDP: sharded params = ~15 GB/rank, optimizer states ~45 GB/rank
    # That's too much for 4 tiles. Use 10 tiles or reduce layers.
    #
    # With 4 tiles: use 16 layers (1/4 of 64) → ~15 GB total, ~3.8 GB/rank sharded
    # Full unshard would temporarily need 15 GB — fits in 64 GB tile.
    hidden = 5120
    intermediate = 27648
    n_layers = 16
    vocab = 32000

    class FakeTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden // 8, bias=False)
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
            h = self.o_proj(q)
            x = x + h
            h = self.norm2(x)
            h = self.gate_proj(h) * self.up_proj(h)
            h = self.down_proj(h)
            return x + h

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([FakeTransformerLayer() for _ in range(n_layers)])
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for i, layer in enumerate(self.layers):
                t0 = time.perf_counter()
                x = layer(x)
                if rank == 0 and self._log_layers:
                    dt = time.perf_counter() - t0
                    print(f"  layer {i}: {dt*1000:.1f}ms", flush=True)
            return self.head(x)

    model = FakeModel().to(dtype=torch.bfloat16, device=device)
    model._log_layers = False

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if rank == 0:
        print(f"Model: {n_layers} layers, {param_bytes / 1e9:.2f} GB, "
              f"{sum(1 for _ in model.parameters())} params, "
              f"arena={_usm_so or 'OFF'}", flush=True)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)

    if rank == 0:
        print(f"FSDP2 applied, world_size={world_size}", flush=True)

    # Phase 1: Warmup + baseline forward
    dummy_input = torch.randint(0, vocab, (2, 128), device=device)
    if rank == 0:
        print("=== Phase 1: Forward passes ===", flush=True)

    for i in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy_input)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        if rank == 0:
            print(f"Forward {i}: {dt:.3f}s", flush=True)

    # Phase 2: Simulate memory fragmentation — allocate then free various sizes
    # This mimics what happens during vLLM generation on the same tile
    if rank == 0:
        print("=== Phase 2: Fragmenting memory ===", flush=True)
    tensors = []
    for size_mb in [64, 128, 256, 512, 1024, 64, 128, 256, 512]:
        t = torch.empty(size_mb * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)
        tensors.append(t)
    # Free in non-sequential order to create gaps
    del tensors[1], tensors[3], tensors[5]
    del tensors  # free the rest
    if rank == 0:
        print("Memory fragmented", flush=True)

    # Phase 3: Forward after fragmentation — this is where the hang occurs
    if rank == 0:
        print("=== Phase 3: Forward after fragmentation ===", flush=True)

    for i in range(3):
        if rank == 0 and i == 0:
            model._log_layers = True
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy_input)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        if rank == 0:
            model._log_layers = False
            print(f"Forward post-frag {i}: {dt:.3f}s", flush=True)

    # Phase 4: Forward+backward after fragmentation
    if rank == 0:
        print("=== Phase 4: Fwd+Bwd after fragmentation ===", flush=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(3):
        t0 = time.perf_counter()
        opt.zero_grad()
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        opt.step()
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        if rank == 0:
            print(f"Train step {i}: {dt:.3f}s", flush=True)

    if rank == 0:
        print("ALL TESTS PASSED", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

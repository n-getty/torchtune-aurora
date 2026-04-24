"""
Test arena allocator under high memory pressure (simulating 32B training state).

Creates a model + holds large dummy tensors to push device utilization to ~85%,
then does FSDP forwards. If the arena hangs when grow_arena calls
sycl::malloc_device under pressure, this will reproduce it.

Usage:
    XPU_USM_ALLOC_SO=recipes/dev/usm_arena_alloc.so \
        torchrun --nproc_per_node=10 recipes/dev/test_arena_pressure.py
"""
import os
import sys
import time
import threading

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

    hidden = 5120
    intermediate = 27648
    n_layers = 32  # ~31.86 GB total, ~3.2 GB/rank sharded
    vocab = 32000

    class TransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, 5120, bias=False)
            self.k_proj = nn.Linear(hidden, 1024, bias=False)
            self.v_proj = nn.Linear(hidden, 1024, bias=False)
            self.o_proj = nn.Linear(5120, hidden, bias=False)
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

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([TransformerLayer() for _ in range(n_layers)])
            self.norm = nn.RMSNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.head(x)

    if rank == 0:
        print(f"Building model: {n_layers} layers, world_size={world_size}, "
              f"arena={_usm_so or 'OFF'}", flush=True)

    model = FakeModel().to(dtype=torch.bfloat16, device=device)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if rank == 0:
        print(f"Model: {param_bytes / 1e9:.2f} GB", flush=True)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for name, mod in reversed(list(model.named_modules())):
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            fully_shard(mod, reshard_after_forward=True, mp_policy=mp_policy)
    try:
        fully_shard(model, reshard_after_forward=None, mp_policy=mp_policy)
    except (TypeError, ValueError):
        fully_shard(model, reshard_after_forward=False, mp_policy=mp_policy)

    if rank == 0:
        print("FSDP2 applied", flush=True)

    # Optimizer creates 2× param state (Adam momentum + variance)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    dummy_input = torch.randint(0, vocab, (2, 128), device=device)

    # Warmup train step to populate optimizer states
    model.train()
    out = model(dummy_input)
    out.sum().backward()
    opt.step()
    opt.zero_grad()
    torch.xpu.synchronize(device)
    if rank == 0:
        print("Optimizer warmed up", flush=True)

    # Now allocate dummy tensors to simulate 32B memory pressure.
    # After model+optimizer: ~13 GB sharded on each rank.
    # Real 32B: ~40 GB/rank (model=6 + opt=12 + grad=6 + ref=6 + misc=10).
    # Fill to ~50 GB to leave only ~14 GB free (simulating ref model + activations).
    pressure_tensors = []
    pressure_gb = 32  # allocate 32 GB of dummy data
    chunk_gb = 1
    if rank == 0:
        print(f"Allocating {pressure_gb} GB of pressure tensors...", flush=True)
    try:
        for i in range(pressure_gb // chunk_gb):
            t = torch.empty(chunk_gb * 1024**3 // 2, dtype=torch.bfloat16, device=device)
            t.fill_(0.0)  # force physical allocation
            pressure_tensors.append(t)
            if rank == 0 and (i + 1) % 8 == 0:
                print(f"  Allocated {(i+1) * chunk_gb} GB", flush=True)
    except RuntimeError as e:
        if rank == 0:
            print(f"  Stopped at {len(pressure_tensors) * chunk_gb} GB: {e}", flush=True)

    allocated_gb = len(pressure_tensors) * chunk_gb
    if rank == 0:
        print(f"Memory pressure: {allocated_gb} GB extra allocated", flush=True)

    # Watchdog
    _last_progress = [time.perf_counter()]
    _phase = ["pre-fwd"]
    def watchdog():
        while True:
            time.sleep(15)
            elapsed = time.perf_counter() - _last_progress[0]
            if elapsed > 15:
                print(f"\n[WATCHDOG rank {rank}] Possible hang: {elapsed:.0f}s in '{_phase[0]}'",
                      flush=True)
                if elapsed > 60:
                    import faulthandler
                    faulthandler.dump_traceback(all_threads=True)
                    break
    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()

    # Now try FSDP forward under pressure
    for trial in range(3):
        _phase[0] = f"forward_{trial}"
        _last_progress[0] = time.perf_counter()
        t0 = time.perf_counter()
        if rank == 0:
            print(f"Forward {trial} start...", flush=True)
        with torch.no_grad():
            out = model(dummy_input)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        _last_progress[0] = time.perf_counter()
        if rank == 0:
            print(f"Forward {trial}: {dt:.3f}s", flush=True)

    # Free some pressure tensors, simulate ref model CPU offload pattern
    if rank == 0:
        print("Freeing pressure tensors (simulating ref cpu offload)...", flush=True)
    del pressure_tensors
    torch.xpu.synchronize(device)

    # Train step after freeing
    for trial in range(2):
        _phase[0] = f"train_post_free_{trial}"
        _last_progress[0] = time.perf_counter()
        t0 = time.perf_counter()
        if rank == 0:
            print(f"Train {trial} start...", flush=True)
        opt.zero_grad()
        out = model(dummy_input)
        out.sum().backward()
        opt.step()
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        _last_progress[0] = time.perf_counter()
        if rank == 0:
            print(f"Train {trial}: {dt:.3f}s", flush=True)

    if rank == 0:
        print("ALL TESTS PASSED", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

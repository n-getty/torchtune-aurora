"""Quick test: compare FSDP2 forward pass time with 1D vs 2D mesh.

Uses a simple Transformer model (not full 32B) to isolate mesh overhead.
Run with: torchrun --standalone --nproc_per_node=10 recipes/dev/test_fsdp2_mesh_simple.py

Expected: if 2D mesh itself causes slowdown, we'll see it even with a small model.
If it's only with large models, the issue is in AllGather size/scheduling.
"""
import os, sys, time, types, gc, datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# Init
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")
dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=5))
rank = dist.get_rank()
world_size = dist.get_world_size()

# Force math-only SDPA
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

if rank == 0:
    print(f"World size: {world_size}")
    print(f"Testing FSDP2 forward pass: 1D mesh (None) vs 2D mesh (1x{world_size})")
    print()


class BigLinearStack(nn.Module):
    """Stack of large linear layers to simulate FSDP AllGather workload."""
    def __init__(self, hidden=8192, layers=16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_forward(label, dp_mesh, hidden=8192, layers=16, warmup=3, trials=5):
    """Time forward passes with the given mesh."""
    model = BigLinearStack(hidden=hidden, layers=layers).to(dtype=torch.bfloat16, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Shard each layer with FSDP2
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward=True)
    fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)

    # Input
    x = torch.randn(4, 512, hidden, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
        torch.xpu.synchronize()
    dist.barrier()

    # Timed trials
    times = []
    for i in range(trials):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.xpu.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        if rank == 0:
            print(f"  [{label}] trial {i}: {t*1000:.1f}ms")

    avg = sum(times) / len(times)
    if rank == 0:
        print(f"  [{label}] avg: {avg*1000:.1f}ms  (min={min(times)*1000:.1f}, max={max(times)*1000:.1f})")

    del model
    gc.collect()
    torch.xpu.synchronize()
    dist.barrier()
    return avg


# ============================================================
# Test 1: 1D mesh (flat FSDP, mesh=None)
# ============================================================
if rank == 0:
    print("=== Test 1: 1D mesh (mesh=None, flat FSDP) ===")
t1 = test_forward("1D", dp_mesh=None)

# ============================================================
# Test 2: 2D mesh with dp_replicate=1 (same sharding, 2D structure)
# ============================================================
if rank == 0:
    print(f"\n=== Test 2: 2D mesh (1x{world_size}, dp_replicate=1) ===")
mesh_2d = init_device_mesh("xpu", (1, world_size), mesh_dim_names=("dp_replicate", "dp_shard"))
t2 = test_forward("2D(1xN)", dp_mesh=mesh_2d)

# ============================================================
# Test 3: 2D mesh with dp_replicate=2 (actual HSDP, if even ranks)
# ============================================================
t3 = None
if world_size >= 4 and world_size % 2 == 0:
    shard_size = world_size // 2
    if rank == 0:
        print(f"\n=== Test 3: 2D mesh (2x{shard_size}, actual HSDP) ===")
    mesh_hsdp = init_device_mesh("xpu", (2, shard_size), mesh_dim_names=("dp_replicate", "dp_shard"))
    t3 = test_forward(f"2D(2x{shard_size})", dp_mesh=mesh_hsdp)

# ============================================================
# Results
# ============================================================
if rank == 0:
    print(f"\n{'='*50}")
    print(f"RESULTS (avg forward pass time):")
    print(f"  1D mesh (None):        {t1*1000:.1f}ms")
    print(f"  2D mesh (1x{world_size}):      {t2*1000:.1f}ms  ({t2/t1:.2f}x vs 1D)")
    if t3 is not None:
        print(f"  2D HSDP (2x{world_size//2}):      {t3*1000:.1f}ms  ({t3/t1:.2f}x vs 1D)")
    print(f"{'='*50}")

dist.destroy_process_group()

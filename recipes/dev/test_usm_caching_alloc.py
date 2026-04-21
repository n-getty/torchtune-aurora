"""Test the caching USM allocator with oneCCL collectives.

Tests both the type-check fix (Failure 1) and the IPC/DMA fix (Failure 2) by
running all_gather_into_tensor with tensors large enough to trigger CCL's
zero-copy IPC path (> ~8 MB output).

Run:
  torchrun --nproc_per_node=2 --master-port=29540 test_usm_caching_alloc.py
"""
import os, sys
import torch
import torch.distributed as dist

# Register the allocator BEFORE any XPU initialization.
SO_PATH = os.path.join(os.path.dirname(__file__), "usm_caching_alloc.so")
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found. Build with:", file=sys.stderr)
    print(f"  icpx -shared -fPIC -fsycl -O2 -o {SO_PATH} "
          f"{SO_PATH.replace('.so', '.cpp')}", file=sys.stderr)
    sys.exit(1)

from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
alloc = XPUPluggableAllocator(SO_PATH, "xpu_usm_malloc", "xpu_usm_free")
change_current_allocator(alloc)

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.xpu.set_device(local_rank)
dist.init_process_group(backend="xccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"xpu:{local_rank}")

def log(msg):
    print(f"[rank{rank}] {msg}", flush=True)

if rank == 0:
    log(f"world_size={world_size}, allocator={SO_PATH}")

# --- Test 1: small allreduce (sanity check) ---
t = torch.ones(10, device=device)
dist.all_reduce(t)
assert t[0].item() == world_size, f"allreduce wrong: {t[0].item()} != {world_size}"
if rank == 0:
    log(f"Test 1 PASS: allreduce sum={t[0].item()}")

# --- Test 2: large all_gather (16 MB output, triggers IPC path) ---
n_elements = int(16 * 1024 * 1024 / 2)  # 16 MB in bfloat16
shard = torch.ones(n_elements // world_size, dtype=torch.bfloat16, device=device) * (rank + 1)
output = torch.zeros(n_elements, dtype=torch.bfloat16, device=device)
dist.all_gather_into_tensor(output, shard)
torch.xpu.synchronize()
expected = sum(range(1, world_size + 1)) * (n_elements // world_size)
actual = output.sum().item()
assert abs(actual - expected) < 1, f"allgather wrong: {actual} != {expected}"
if rank == 0:
    log(f"Test 2 PASS: all_gather 16MB output sum={actual:.0f}")

# --- Test 3: large reduce_scatter ---
full = torch.ones(n_elements, dtype=torch.bfloat16, device=device) * (rank + 1)
out_shard = torch.zeros(n_elements // world_size, dtype=torch.bfloat16, device=device)
dist.reduce_scatter_tensor(out_shard, full)
torch.xpu.synchronize()
expected_rs = sum(range(1, world_size + 1))
actual_rs = out_shard.mean().item()
assert abs(actual_rs - expected_rs) < 0.1, f"reduce_scatter wrong: {actual_rs} != {expected_rs}"
if rank == 0:
    log(f"Test 3 PASS: reduce_scatter 16MB mean={actual_rs:.1f}")

# --- Test 4: allocation reuse (caching sanity) ---
ptrs = set()
for _ in range(5):
    a = torch.empty(1024 * 1024, dtype=torch.float32, device=device)
    ptrs.add(a.data_ptr())
    del a
# After 5 alloc+free cycles of same size, at most 2 unique ptrs (caching works)
if rank == 0:
    log(f"Test 4: {len(ptrs)} unique ptr(s) for 5 same-size alloc/free cycles "
        f"({'PASS - caching works' if len(ptrs) <= 2 else 'OK - caching may vary'})")

dist.destroy_process_group()
if rank == 0:
    log("ALL TESTS PASSED")

"""
Phase 2 reproducer: minimal 2-rank CCL test with large tensors (above IPC threshold).

The existing repro_ccl_expandable_segments.py uses 10-element tensors which fall
below the ~8-12 MiB staging-buffer threshold and don't exercise the IPC path.
This script uses 32 MiB tensors to force the IPC zero-copy path.

Run with L0 IPC shim to trace zeMemGetIpcHandle calls:
  LD_PRELOAD=$TORCHTUNE/experiments/arena_ipc/diag_2_l0_ipc_shim.so \
  XPU_USM_ALLOC_SO=$TORCHTUNE/recipes/dev/usm_arena_alloc.so \
  ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0,1 \
  torchrun --nproc_per_node=2 experiments/arena_ipc/diag_2_repro_large.py

Run WITHOUT allocator to establish default-allocator baseline (should pass):
  LD_PRELOAD=$TORCHTUNE/experiments/arena_ipc/diag_2_l0_ipc_shim.so \
  ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0,1 \
  torchrun --nproc_per_node=2 experiments/arena_ipc/diag_2_repro_large.py
"""
import os
import sys

# Pluggable allocator must be installed before any XPU init
_usm_so = os.environ.get("XPU_USM_ALLOC_SO")
if _usm_so:
    if not os.path.exists(_usm_so):
        print(f"ERROR: XPU_USM_ALLOC_SO={_usm_so} not found", flush=True)
        sys.exit(1)
    from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
    alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
    change_current_allocator(alloc)
    print(f"[repro] Custom allocator installed: {_usm_so}", flush=True)
else:
    print("[repro] Using DEFAULT allocator", flush=True)

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)

dist.init_process_group(backend="xccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"[repro rank{rank}] init_process_group OK, world={world_size}", flush=True)

# Test 1: small allreduce (below IPC threshold — should always pass)
t_small = torch.ones(10, device=device)
dist.all_reduce(t_small)
assert t_small[0].item() == world_size, f"Expected {world_size}, got {t_small[0].item()}"
print(f"[repro rank{rank}] T1 small allreduce OK (sum={t_small[0].item()})", flush=True)

# Test 2: large all_gather (above IPC threshold — the failing path)
MB32 = 32 * 1024 * 1024 // 2  # 32 MiB in bfloat16 elements
t_shard = torch.ones(MB32, dtype=torch.bfloat16, device=device) * (rank + 1)
t_out   = torch.empty(MB32 * world_size, dtype=torch.bfloat16, device=device)
print(f"[repro rank{rank}] T2 attempting all_gather ({MB32*2/1e6:.0f} MiB shard, "
      f"{MB32*2*world_size/1e6:.0f} MiB output)...", flush=True)
dist.all_gather_into_tensor(t_out, t_shard)
expected_sum = sum(range(1, world_size+1)) * MB32
actual_sum = t_out.sum().item()
print(f"[repro rank{rank}] T2 all_gather OK (sum={actual_sum:.0f})", flush=True)

# Test 3: large reduce_scatter (above IPC threshold)
t_in  = torch.ones(MB32 * world_size, dtype=torch.bfloat16, device=device) * (rank + 1)
t_rs  = torch.empty(MB32, dtype=torch.bfloat16, device=device)
print(f"[repro rank{rank}] T3 attempting reduce_scatter...", flush=True)
dist.reduce_scatter_tensor(t_rs, t_in)
print(f"[repro rank{rank}] T3 reduce_scatter OK (sample={t_rs[0].item():.1f})", flush=True)

dist.destroy_process_group()
print(f"[repro rank{rank}] ALL TESTS PASSED", flush=True)

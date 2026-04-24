"""Test: Can a second ProcessGroupXCCL coexist with the training PG?

If this works, we can use the existing vllm_client.py init_communicator()
code path to create a cross-process XCCL group with vLLM — no launcher
restructuring needed.

2 ranks via torchrun. Each rank:
  1. init_process_group(backend="xccl") — simulates training PG
  2. Create a SECOND ProcessGroupXCCL via constructor — simulates weight sync PG
  3. Test both PGs work (allreduce on PG1, broadcast on PG2)
"""
import os
import sys
import time

import types as _types
import importlib.util as _imp_util
if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)


def log(msg):
    print(f"[rank {rank}] {msg}", flush=True)


# Step 1: Init training PG (normal init_process_group)
log("Step 1: init_process_group(backend='xccl')")
dist.init_process_group(backend="xccl", device_id=device)
t = torch.ones(10, device=device)
dist.all_reduce(t)
log(f"  Training PG works: allreduce sum={t[0].item()}")

# Step 2: Create a SECOND ProcessGroupXCCL via constructor
log("Step 2: Creating second ProcessGroupXCCL via constructor...")
try:
    store = dist.TCPStore(
        host_name="127.0.0.1",
        port=51216,
        world_size=2,
        is_master=(rank == 0),
    )
    prefixed_store = c10d.PrefixStore("wsync_test", store)
    xccl_options = c10d.ProcessGroupXCCL.Options()
    second_pg = c10d.ProcessGroupXCCL(
        store=prefixed_store,
        rank=rank,
        size=2,
        options=xccl_options,
    )
    log("  Second PG created successfully (no SIGABRT)")
except Exception as e:
    log(f"  FAILED to create second PG: {e}")
    dist.destroy_process_group()
    sys.exit(1)

# Step 3: Test second PG — broadcast
log("Step 3: broadcast on second PG...")
try:
    t2 = torch.ones(1000, device=device) * (rank + 1)
    second_pg.broadcast(t2, root=0).wait()
    torch.xpu.synchronize(device)
    log(f"  Broadcast on second PG: val={t2[0].item()} (expected 1.0)")
    assert abs(t2[0].item() - 1.0) < 1e-3, f"Broadcast data mismatch: {t2[0].item()}"
    log("  PASS: broadcast works")
except Exception as e:
    log(f"  FAIL: broadcast on second PG failed: {e}")
    dist.destroy_process_group()
    sys.exit(1)

# Step 4: Verify first PG still works
log("Step 4: allreduce on training PG (coexistence)...")
t3 = torch.ones(1000, device=device) * rank
dist.all_reduce(t3)
expected = float(sum(range(dist.get_world_size())))
log(f"  Training PG allreduce: sum={t3[0].item()}, expected={expected}")
assert abs(t3[0].item() - expected) < 1e-1

# Step 5: Larger broadcast (100MB)
log("Step 5: 100MB broadcast on second PG...")
SIZE = 100 * 1024 * 1024 // 2  # bf16
if rank == 0:
    t4 = torch.randn(SIZE, device=device, dtype=torch.bfloat16)
else:
    t4 = torch.zeros(SIZE, device=device, dtype=torch.bfloat16)

t0 = time.perf_counter()
second_pg.broadcast(t4, root=0).wait()
torch.xpu.synchronize(device)
dt = time.perf_counter() - t0
log(f"  100MB broadcast: {dt:.3f}s ({100/dt:.1f} MB/s)")

# Cleanup
del second_pg
dist.destroy_process_group()
log("ALL TESTS PASSED — second ProcessGroupXCCL coexists with training PG")

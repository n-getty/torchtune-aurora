"""Test: Can two XCCL sub-groups coexist on Aurora XPU?

3 ranks under torchrun --nproc_per_node=3:
  Ranks 0,1 = "training" (allreduce on sub-group)
  Rank 2    = "vLLM"     (receives broadcast only)

Tests whether dist.new_group() works with XCCL backend, and whether
broadcast on the global PG coexists with allreduce on a sub-group.

If this works, XCCL broadcast can replace SHM-based weight sync entirely.
"""
import os
import sys
import time

# Pre-register torchtune to bypass __init__.py (avoids expandable_segments)
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
import torch.distributed.distributed_c10d as _dc10d

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "3"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)
is_training = rank < 2
role = "TRAIN" if is_training else "VLLM"

results = []


def log(msg):
    print(f"[rank {rank}/{role}] {msg}", flush=True)


def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    log(f"  {name}: {status} {detail}")


# ============================================================
# Step 0: Global PG init
# ============================================================
log(f"Initializing global XCCL PG (world_size={world_size}, device={device})")
dist.init_process_group(backend="xccl", device_id=device)
log(f"Global PG ready")

# Verify global PG works
t = torch.ones(10, device=device)
dist.all_reduce(t)
expected = float(world_size)
record("global_allreduce", abs(t[0].item() - expected) < 1e-3,
       f"sum={t[0].item()}, expected={expected}")

# ============================================================
# Step 1: Create training sub-group via new_group
# ============================================================
log("Creating training sub-group [0, 1] via dist.new_group()...")

# Clear bound_device_id — required for new_group to work
# (learned from grpo recipe line 864-874)
_default_pg = _dc10d._get_default_group()
_orig_bound_device_id = _default_pg.bound_device_id
_default_pg.bound_device_id = None

try:
    training_pg = dist.new_group([0, 1])
    record("new_group_create", True)
except Exception as e:
    record("new_group_create", False, str(e))
    log("FATAL: new_group failed — XCCL sub-groups not supported")
    dist.destroy_process_group()
    sys.exit(1)
finally:
    _default_pg.bound_device_id = _orig_bound_device_id

# ============================================================
# Test A: Training allreduce on sub-group
# ============================================================
log("Test A: allreduce on training sub-group...")
if is_training:
    t_a = torch.ones(100, device=device) * (rank + 1)
    dist.all_reduce(t_a, group=training_pg)
    expected_a = 3.0  # 1 + 2
    record("test_a_training_allreduce", abs(t_a[0].item() - expected_a) < 1e-3,
           f"sum={t_a[0].item()}, expected={expected_a}")
else:
    record("test_a_training_allreduce", True, "skipped (vLLM rank)")
dist.barrier()

# ============================================================
# Test B: Broadcast on global PG (rank 0 → all)
# ============================================================
log("Test B: broadcast 100MB tensor on global PG...")
SIZE_100MB = 100 * 1024 * 1024 // 4  # 100MB of float32
if rank == 0:
    t_b = torch.arange(SIZE_100MB, device=device, dtype=torch.float32)
else:
    t_b = torch.zeros(SIZE_100MB, device=device, dtype=torch.float32)

t0 = time.perf_counter()
dist.broadcast(t_b, src=0)
torch.xpu.synchronize(device)
dt = time.perf_counter() - t0

# Verify data integrity on all ranks
check_val = t_b[12345].item()
expected_val = 12345.0
record("test_b_broadcast_100mb", abs(check_val - expected_val) < 1e-1,
       f"val@12345={check_val}, expected={expected_val}, time={dt:.3f}s, "
       f"bw={100 / dt:.1f} MB/s")
del t_b
dist.barrier()

# ============================================================
# Test C: Training allreduce still works after broadcast
# ============================================================
log("Test C: training allreduce after broadcast (coexistence)...")
if is_training:
    t_c = torch.ones(100, device=device) * (rank + 10)
    dist.all_reduce(t_c, group=training_pg)
    expected_c = 21.0  # 10 + 11
    record("test_c_post_broadcast_allreduce", abs(t_c[0].item() - expected_c) < 1e-3,
           f"sum={t_c[0].item()}, expected={expected_c}")
else:
    record("test_c_post_broadcast_allreduce", True, "skipped (vLLM rank)")
dist.barrier()

# ============================================================
# Test D: 1GB broadcast — bandwidth measurement
# ============================================================
log("Test D: 1GB broadcast bandwidth (3 iterations, bf16)...")
SIZE_1GB = 1024 * 1024 * 1024 // 2  # 1GB of bf16
times_d = []
for i in range(3):
    if rank == 0:
        t_d = torch.randn(SIZE_1GB, device=device, dtype=torch.bfloat16)
    else:
        t_d = torch.zeros(SIZE_1GB, device=device, dtype=torch.bfloat16)

    torch.xpu.synchronize(device)
    dist.barrier()
    t0 = time.perf_counter()
    dist.broadcast(t_d, src=0)
    torch.xpu.synchronize(device)
    dt = time.perf_counter() - t0
    times_d.append(dt)
    del t_d

median_d = sorted(times_d)[1]
bw_gbps = 1.0 / median_d
record("test_d_1gb_broadcast", True,
       f"times={[f'{t:.3f}s' for t in times_d]}, median={median_d:.3f}s, "
       f"bw={bw_gbps:.2f} GB/s")
dist.barrier()

# ============================================================
# Test E: Interleaved allreduce + broadcast (5 iterations)
# ============================================================
log("Test E: 5 iterations of (training allreduce → global broadcast)...")
SIZE_10MB = 10 * 1024 * 1024 // 2  # 10MB bf16
try:
    for i in range(5):
        # Training allreduce on sub-group
        if is_training:
            t_e1 = torch.ones(SIZE_10MB, device=device, dtype=torch.bfloat16) * (rank + 1)
            dist.all_reduce(t_e1, group=training_pg)
            del t_e1

        dist.barrier()

        # Global broadcast
        if rank == 0:
            t_e2 = torch.randn(SIZE_10MB, device=device, dtype=torch.bfloat16)
        else:
            t_e2 = torch.zeros(SIZE_10MB, device=device, dtype=torch.bfloat16)
        dist.broadcast(t_e2, src=0)
        del t_e2

        dist.barrier()

    record("test_e_interleaved_stress", True, "5 iterations OK")
except Exception as e:
    record("test_e_interleaved_stress", False, str(e))

# ============================================================
# Summary
# ============================================================
dist.barrier()
if rank == 0:
    print("\n" + "=" * 60)
    print("XCCL BROADCAST TEST RESULTS")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED — XCCL broadcast can replace SHM weight sync")
    else:
        print("SOME TESTS FAILED — see details above")
    print("=" * 60)

dist.destroy_process_group()
log("Done")

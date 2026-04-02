"""Test AllGather performance with HSDP sub-communicator on multi-node.

Measures raw AllGather time to isolate CCL overhead from model forward.
Tests both the shard sub-group (should be intra-node) and full world group.

Run: mpiexec -n 20 -ppn 10 --hostfile $PBS_NODEFILE ... wrapper.sh test_allgather_multinode.py
"""
import os, sys, time, types, datetime
import torch
import torch.distributed as dist

# Pre-register torchtune
if "torchtune" not in sys.modules:
    sys.modules["torchtune"] = types.ModuleType("torchtune")
    sys.modules["torchtune"].__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "torchtune")]

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")
dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=5))
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

if rank == 0:
    print(f"World size: {world_size}")
    print(f"CCL_ALLREDUCE={os.environ.get('CCL_ALLREDUCE', '<default>')}")
    print(f"CCL_REDUCE_SCATTER={os.environ.get('CCL_REDUCE_SCATTER', '<default>')}")
    print(f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")

# Create HSDP mesh (2 nodes × 10 ranks)
from torch.distributed.device_mesh import init_device_mesh
if world_size == 20:
    mesh = init_device_mesh("xpu", (2, 10), mesh_dim_names=("dp_replicate", "dp_shard"))
    shard_pg = mesh.get_group("dp_shard")
    replicate_pg = mesh.get_group("dp_replicate")
    if rank == 0:
        print(f"HSDP mesh: 2x10")
        print(f"Shard group size: {shard_pg.size()}")
        print(f"Replicate group size: {replicate_pg.size()}")
elif world_size == 10:
    mesh = None
    shard_pg = None
    replicate_pg = None
    if rank == 0:
        print(f"Single-node: no mesh, using world group")
else:
    if rank == 0:
        print(f"Unsupported world_size={world_size}, need 10 or 20")
    dist.destroy_process_group()
    sys.exit(1)

# Test different data sizes
# 32B model: each transformer layer has ~1 GiB params
# With FSDP-10, each shard is ~100 MiB per layer
# AllGather unshard: 10 × 100 MiB = 1 GiB per layer, 64 layers = 64 GiB total
sizes_mb = [10, 100, 500, 1000]  # AllGather data sizes in MiB

for size_mb in sizes_mb:
    # Create sharded tensor (what each rank holds)
    shard_size = size_mb * 1024 * 1024 // 2  # BF16 = 2 bytes
    shard = torch.randn(shard_size, dtype=torch.bfloat16, device=device)

    if shard_pg is not None:
        group = shard_pg
        group_size = shard_pg.size()
    else:
        group = dist.group.WORLD
        group_size = world_size

    # Pre-allocate output
    full_size = shard_size * group_size
    output = torch.empty(full_size, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        dist.all_gather_into_tensor(output, shard, group=group)
    torch.xpu.synchronize()
    dist.barrier()

    # Timed trials
    times = []
    for _ in range(5):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_gather_into_tensor(output, shard, group=group)
        torch.xpu.synchronize()
        t = time.perf_counter() - t0
        times.append(t)

    avg = sum(times) / len(times)
    total_gathered = size_mb * group_size  # MiB
    bw = total_gathered / avg / 1024  # GiB/s
    if rank == 0:
        print(f"AllGather {size_mb}MiB x {group_size} = {total_gathered}MiB: "
              f"avg={avg*1000:.1f}ms min={min(times)*1000:.1f}ms "
              f"bw={bw:.1f} GiB/s")

    del shard, output
    torch.xpu.synchronize()
    dist.barrier()

dist.destroy_process_group()

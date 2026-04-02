"""Compare ReduceScatter bandwidth: world group vs shard sub-group in multi-node.
Usage: mpiexec --pmi=pmix -n 20 -ppn 10 bash wrapper.sh recipes/dev/test_rs_multinode.py
"""
import os, time, torch, torch.distributed as dist, datetime
from torch.distributed.device_mesh import init_device_mesh

if os.environ.get('CCL_ATL_TRANSPORT') == 'mpi':
    from mpi4py import MPI; MPI.COMM_WORLD.Barrier()

local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.xpu.set_device(local_rank)
dist.init_process_group('xccl', timeout=datetime.timedelta(minutes=2))
rank = dist.get_rank()
ws = dist.get_world_size()
lws = int(os.environ.get('LOCAL_WORLD_SIZE', str(ws)))
nn = ws // lws
device = torch.device(f'xpu:{local_rank}')

if rank == 0:
    print(f"World={ws}, Local={lws}, Nodes={nn}")

# Create HSDP mesh and sub-groups
mesh = init_device_mesh('xpu', (nn, lws), mesh_dim_names=('rep', 'shard'))
shard_pg = mesh.get_group('shard')

for size_mb in [100, 500, 1000]:
    elems = size_mb * 1024 * 1024 // 2
    shard_elems = elems // lws

    # ReduceScatter on SHARD sub-group (10 intra-node ranks)
    full_s = torch.randn(elems, dtype=torch.bfloat16, device=device)
    out_s = torch.empty(shard_elems, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        dist.reduce_scatter_tensor(out_s, full_s, group=shard_pg)
    torch.xpu.synchronize(); dist.barrier()
    shard_times = []
    for _ in range(5):
        torch.xpu.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out_s, full_s, group=shard_pg)
        torch.xpu.synchronize()
        shard_times.append(time.perf_counter() - t0)
    del full_s, out_s

    # ReduceScatter on WORLD group (20 ranks)
    full_w = torch.randn(elems, dtype=torch.bfloat16, device=device)
    out_w = torch.empty(elems // ws, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        dist.reduce_scatter_tensor(out_w, full_w)
    torch.xpu.synchronize(); dist.barrier()
    world_times = []
    for _ in range(5):
        torch.xpu.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out_w, full_w)
        torch.xpu.synchronize()
        world_times.append(time.perf_counter() - t0)
    del full_w, out_w

    if rank == 0:
        s_avg = sum(shard_times) / len(shard_times)
        w_avg = sum(world_times) / len(world_times)
        s_bw = size_mb / s_avg / 1024
        w_bw = size_mb / w_avg / 1024
        print(f"{size_mb:>4}MiB: Shard RS(10)={s_avg*1000:.1f}ms ({s_bw:.1f} GiB/s)  "
              f"World RS(20)={w_avg*1000:.1f}ms ({w_bw:.1f} GiB/s)  "
              f"(single-node baseline: 138 GiB/s)")

dist.destroy_process_group()

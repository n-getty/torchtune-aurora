"""Quick benchmark: ReduceScatter vs AllGather on single node.
Usage: torchrun --standalone --nproc_per_node=10 recipes/dev/test_rs_vs_ag.py
"""
import os, time, torch, torch.distributed as dist, datetime

torch.xpu.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=2))
rank = dist.get_rank()
ws = dist.get_world_size()
device = torch.device(f"xpu:{rank}")

for size_mb in [100, 500, 1000]:
    elems = size_mb * 1024 * 1024 // 2  # BF16
    shard_elems = elems // ws

    # AllGather
    shard = torch.randn(shard_elems, dtype=torch.bfloat16, device=device)
    output = torch.empty(elems, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        dist.all_gather_into_tensor(output, shard)
    torch.xpu.synchronize(); dist.barrier()
    ag_times = []
    for _ in range(5):
        torch.xpu.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        dist.all_gather_into_tensor(output, shard)
        torch.xpu.synchronize()
        ag_times.append(time.perf_counter() - t0)
    del shard, output

    # ReduceScatter
    full = torch.randn(elems, dtype=torch.bfloat16, device=device)
    out_rs = torch.empty(shard_elems, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        dist.reduce_scatter_tensor(out_rs, full)
    torch.xpu.synchronize(); dist.barrier()
    rs_times = []
    for _ in range(5):
        torch.xpu.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out_rs, full)
        torch.xpu.synchronize()
        rs_times.append(time.perf_counter() - t0)
    del full, out_rs

    if rank == 0:
        ag_avg = sum(ag_times) / len(ag_times)
        rs_avg = sum(rs_times) / len(rs_times)
        ag_bw = size_mb / ag_avg / 1024
        rs_bw = size_mb / rs_avg / 1024
        print(f"{size_mb:>4}MiB: AG={ag_avg*1000:.1f}ms ({ag_bw:.1f} GiB/s)  "
              f"RS={rs_avg*1000:.1f}ms ({rs_bw:.1f} GiB/s)  "
              f"ratio={rs_avg/ag_avg:.1f}x")

dist.destroy_process_group()

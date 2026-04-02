"""Test AllGather bandwidth under different ZE_AFFINITY_MASK modes.

Isolates why torchrun gets 111 GiB/s but mpiexec gets 2.4 GiB/s.

Environment variables:
  AFFINITY_MODE: "single" — ZE_AFFINITY_MASK=$LOCAL_RANK (1 tile visible per rank)
                 "training" — ZE_AFFINITY_MASK=0,...,N-1 (all training tiles visible)
                 "none" — unset mask (all 12 tiles visible)
                 Default: "single"
"""
import os
import time
import datetime
import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    mode = os.environ.get("AFFINITY_MODE", "single")

    # Set affinity BEFORE any XPU operations
    if mode == "single":
        os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
        device_idx = 0  # Only 1 tile visible
    elif mode == "training":
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE",
                               os.environ.get("WORLD_SIZE", "10")))
        os.environ["ZE_AFFINITY_MASK"] = ",".join(str(i) for i in range(local_world_size))
        device_idx = local_rank  # N tiles visible, use LOCAL_RANK-th
    elif mode == "none":
        os.environ.pop("ZE_AFFINITY_MASK", None)
        device_idx = local_rank
    else:
        device_idx = local_rank

    torch.xpu.set_device(device_idx)
    device = torch.device(f"xpu:{device_idx}")

    # MPI transport pre-init
    if os.environ.get("CCL_ATL_TRANSPORT") == "mpi":
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()
        except ImportError:
            pass

    dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        mask = os.environ.get("ZE_AFFINITY_MASK", "<unset>")
        print(f"=== AllGather Affinity Test ===")
        print(f"AFFINITY_MODE:    {mode}")
        print(f"ZE_AFFINITY_MASK: {mask}")
        print(f"device_idx:       {device_idx}")
        print(f"World size:       {world_size}")
        print(f"CCL_ATL_TRANSPORT: {os.environ.get('CCL_ATL_TRANSPORT', '<default>')}")
        print(f"CCL_PROCESS_LAUNCHER: {os.environ.get('CCL_PROCESS_LAUNCHER', '<default>')}")
        print(f"CCL_WORKER_COUNT: {os.environ.get('CCL_WORKER_COUNT', '<default>')}")
        print(f"CCL_OP_SYNC:      {os.environ.get('CCL_OP_SYNC', '<default>')}")
        print()

    sizes_mb = [10, 100, 500, 1000]
    for size_mb in sizes_mb:
        shard_elems = size_mb * 1024 * 1024 // 2  # BF16
        shard = torch.randn(shard_elems, dtype=torch.bfloat16, device=device)
        output = torch.empty(shard_elems * world_size, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            dist.all_gather_into_tensor(output, shard)
        torch.xpu.synchronize()
        dist.barrier()

        # Timed
        times = []
        for _ in range(5):
            torch.xpu.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_gather_into_tensor(output, shard)
            torch.xpu.synchronize()
            t = time.perf_counter() - t0
            times.append(t)

        avg = sum(times) / len(times)
        total_mb = size_mb * world_size
        bw = total_mb / avg / 1024  # GiB/s
        if rank == 0:
            print(f"  AllGather {size_mb:>4}MiB x {world_size} = {total_mb:>5}MiB: "
                  f"avg={avg*1000:.1f}ms  bw={bw:.1f} GiB/s  "
                  f"times={[f'{t*1000:.1f}' for t in times]}")

        del shard, output
        torch.xpu.synchronize()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

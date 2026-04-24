"""Test: torch.distributed with XCCL — use ZE_AFFINITY_MASK instead of device_id."""
import os, torch

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# With ZE_AFFINITY_MASK set per-rank, each rank sees only 1 tile as xpu:0
device = torch.device("xpu:0")
torch.xpu.set_device(0)
print(f"Rank {local_rank}: device={device}, AFFINITY={os.environ.get('ZE_AFFINITY_MASK','unset')}")

torch.distributed.init_process_group(backend="xccl")
print(f"Rank {local_rank}: PG initialized, world_size={torch.distributed.get_world_size()}")

# Simple allreduce test
t = torch.ones(10, device=device)
torch.distributed.all_reduce(t)
print(f"Rank {local_rank}: allreduce OK, sum={t[0].item()}")

torch.distributed.destroy_process_group()
print(f"Rank {local_rank}: Done")

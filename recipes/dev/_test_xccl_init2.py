"""Minimal test: torch.distributed.init_process_group with XCCL — NO torchao."""
import os, sys, torch

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)
print(f"Rank {local_rank}: device={device}")

torch.distributed.init_process_group(
    backend="xccl",
    device_id=device,
)
print(f"Rank {local_rank}: PG initialized, world_size={torch.distributed.get_world_size()}")

# Simple allreduce test
t = torch.ones(10, device=device)
torch.distributed.all_reduce(t)
print(f"Rank {local_rank}: allreduce OK, sum={t[0].item()}")

torch.distributed.destroy_process_group()
print(f"Rank {local_rank}: Done")

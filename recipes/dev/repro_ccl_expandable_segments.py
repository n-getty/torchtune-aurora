"""Minimal reproducer: expandable_segments breaks oneCCL allreduce on XPU.

PASS:  torchrun --nproc_per_node=2 repro_ccl_expandable_segments.py
FAIL:  PYTORCH_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 repro_ccl_expandable_segments.py

Note: in production, ZE_AFFINITY_MASK=$LOCAL_RANK is set per-rank by the PBS
launcher so each rank sees only its own tile. For this two-rank reproducer you
can set ZE_AFFINITY_MASK=0,1 before invoking torchrun, or omit it (both ranks
will enumerate all tiles but will still bind to xpu:0 / xpu:1 respectively).
"""
import os, torch

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)

torch.distributed.init_process_group(backend="xccl")

t = torch.ones(10, device=device)
torch.distributed.all_reduce(t)  # fails with expandable_segments:True
print(f"Rank {local_rank}: allreduce OK, sum={t[0].item()}")

torch.distributed.destroy_process_group()

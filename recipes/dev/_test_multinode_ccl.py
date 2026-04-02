#!/usr/bin/env python3
"""Minimal multi-node CCL test on Aurora XPU.
Tests init_process_group + broadcast + allreduce across nodes.
"""
import os
import time
import socket
import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
master_addr = os.environ.get("MASTER_ADDR", "localhost")
master_port = os.environ.get("MASTER_PORT", "29500")

print(f"[rank {rank}] node={socket.gethostname()} local_rank={local_rank} "
      f"world_size={world_size} master={master_addr}:{master_port} "
      f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")

# Detect device
if hasattr(torch, "xpu") and torch.xpu.is_available():
    use_affinity = bool(os.environ.get("ZE_AFFINITY_MASK", ""))
    device_idx = 0 if use_affinity else local_rank
    device = torch.device(f"xpu:{device_idx}")
    torch.xpu.set_device(device_idx)
    backend = "xccl"
    print(f"[rank {rank}] Using xpu:{device_idx} (affinity_mask={'set' if use_affinity else 'unset'})")
else:
    device = torch.device("cpu")
    backend = "gloo"
    print(f"[rank {rank}] Using CPU")

# Init process group
t0 = time.perf_counter()
init_method = f"tcp://{master_addr}:{master_port}"
if not use_affinity:
    dist.init_process_group(
        backend=backend, init_method=init_method,
        world_size=world_size, rank=rank,
        device_id=device,
    )
else:
    dist.init_process_group(
        backend=backend, init_method=init_method,
        world_size=world_size, rank=rank,
    )
print(f"[rank {rank}] init_process_group took {time.perf_counter() - t0:.2f}s")

# Test 1: barrier
t0 = time.perf_counter()
dist.barrier()
print(f"[rank {rank}] barrier took {time.perf_counter() - t0:.4f}s")

# Test 2: broadcast (rank 0 sends to all)
tensor = torch.tensor([rank * 1.0, rank * 2.0, rank * 3.0], device=device)
t0 = time.perf_counter()
dist.broadcast(tensor, src=0)
print(f"[rank {rank}] broadcast took {time.perf_counter() - t0:.4f}s, tensor={tensor.tolist()}")

# Test 3: allreduce
tensor = torch.ones(1024, device=device) * (rank + 1)
t0 = time.perf_counter()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected = sum(range(1, world_size + 1))
actual = tensor[0].item()
print(f"[rank {rank}] allreduce took {time.perf_counter() - t0:.4f}s, "
      f"expected={expected}, got={actual}, match={abs(actual - expected) < 0.01}")

# Test 4: larger broadcast (simulate vLLM generation result)
if rank == 0:
    big_tensor = torch.randn(4, 1024, device=device)
else:
    big_tensor = torch.zeros(4, 1024, device=device)
t0 = time.perf_counter()
dist.broadcast(big_tensor, src=0)
dt = time.perf_counter() - t0
print(f"[rank {rank}] big broadcast (4x1024) took {dt:.4f}s")

dist.destroy_process_group()
print(f"[rank {rank}] PASSED all tests")

"""Test if XCCL init_process_group works for world_size=1 inside torchrun.

Run with:
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
        recipes/dev/_test_xccl_single_rank.py
"""
import os
import sys
import time
import socket
import tempfile

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"

import torch

print(f"[rank {rank}] Starting XCCL single-rank test")

if rank == 0:
    # Save torchrun env vars
    saved_env = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_RUN_ID"):
        saved_env[key] = os.environ.pop(key, None)

    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Use a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    os.environ["MASTER_PORT"] = str(port)

    print(f"[rank 0] Attempting XCCL init_process_group(world_size=1, port={port})...")
    t0 = time.time()

    try:
        torch.distributed.init_process_group(
            backend="xccl",
            world_size=1,
            rank=0,
        )
        elapsed = time.time() - t0
        print(f"[rank 0] XCCL PG initialized in {elapsed:.1f}s!")

        # Test a simple operation
        device = torch.device("xpu:0")
        tensor = torch.ones(4, device=device)
        torch.distributed.all_reduce(tensor)
        print(f"[rank 0] allreduce works: {tensor.tolist()}")

        torch.distributed.destroy_process_group()
        print("[rank 0] PG destroyed")
    except Exception as e:
        print(f"[rank 0] XCCL init failed: {e}")
        import traceback
        traceback.print_exc()

    # Restore env vars
    for key, val in saved_env.items():
        if val is not None:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]

    # Signal rank 1
    with open("/tmp/torchtune/xccl_test_done", "w") as f:
        f.write("done")

else:
    print(f"[rank {rank}] Waiting for rank 0...")
    while not os.path.exists("/tmp/torchtune/xccl_test_done"):
        time.sleep(0.5)

# Now init the real training PG
print(f"[rank {rank}] Initializing real training XCCL PG...")
torch.distributed.init_process_group(backend="xccl")
print(f"[rank {rank}] Training PG OK: world_size={torch.distributed.get_world_size()}")

# Cleanup
if rank == 0:
    try:
        os.unlink("/tmp/torchtune/xccl_test_done")
    except OSError:
        pass

torch.distributed.destroy_process_group()
print(f"[rank {rank}] SUCCESS")

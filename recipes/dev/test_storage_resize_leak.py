#!/usr/bin/env python3
"""
Test: does tensor.untyped_storage().resize_() leak UR handles on XPU?

FSDP uses storage.resize_(0) to "free" and storage.resize_(size) to "alloc"
tensor backing storage without destroying the tensor object. This is how
FSDP reshards (free unsharded params) and unshards (alloc space for full params).

If this leaks on XPU, it would explain the FSDP-specific crash.

Usage:
  # Single device (no distributed):
  ZE_AFFINITY_MASK=0 python3 test_storage_resize_leak.py

  # With distributed (2 tiles):
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_storage_resize_leak.py --distributed

  # With allgather between resize cycles:
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_storage_resize_leak.py --distributed --with-allgather

  # Full FSDP simulation (allgather into resizable storage):
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_storage_resize_leak.py --distributed --fsdp-sim
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def test_resize_only(device, max_iters, tensors, rank):
    """Test storage.resize_(0) / resize_(size) cycles."""
    if rank == 0:
        print(f"\n--- Test: Storage Resize Only ({len(tensors)} tensors/cycle) ---")

    t0 = time.time()
    for i in range(max_iters):
        for t in tensors:
            size = t.numel() * t.itemsize
            storage = t.untyped_storage()
            # Free
            storage.resize_(0)
            # Alloc
            storage.resize_(size)

        if rank == 0 and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            total = (i + 1) * len(tensors)
            mem = torch.xpu.memory_allocated(device) / 1e9
            print(f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | resize_cycles={total} | mem={mem:.2f} GiB")

    if rank == 0:
        total = max_iters * len(tensors)
        print(f"  PASSED: {total} resize cycles without crash")


def test_resize_with_allgather(device, max_iters, num_layers, rank, world_size):
    """Test FSDP pattern: allgather into buffer, use, then resize_(0) to free."""
    if rank == 0:
        print(f"\n--- Test: Allgather + Storage Resize ({num_layers} layers/iter) ---")

    shard_size = 2048 * 2048
    full_size = shard_size * world_size

    # Pre-allocate persistent tensors like FSDP does
    shards = [torch.randn(shard_size, device=device, dtype=torch.bfloat16) for _ in range(num_layers)]
    # Unsharded buffers (start with 0 storage, like FSDP's resharded state)
    unsharded = [torch.empty(full_size, device=device, dtype=torch.bfloat16) for _ in range(num_layers)]
    for u in unsharded:
        u.untyped_storage().resize_(0)

    t0 = time.time()
    for i in range(max_iters):
        for layer_idx in range(num_layers):
            # Unshard: alloc storage + allgather
            storage = unsharded[layer_idx].untyped_storage()
            storage.resize_(full_size * 2)  # bf16 = 2 bytes

            output_chunks = list(unsharded[layer_idx].chunk(world_size))
            with torch.no_grad():
                dist.all_gather(output_chunks, shards[layer_idx])

            # Use the unsharded param
            x = torch.randn(32, full_size, device=device, dtype=torch.bfloat16)
            y = x @ unsharded[layer_idx].unsqueeze(-1)
            del x, y

            # Reshard: free storage
            storage.resize_(0)

        torch.xpu.synchronize(device)

        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            total = (i + 1) * num_layers
            mem = torch.xpu.memory_allocated(device) / 1e9
            mem_res = torch.xpu.memory_reserved(device) / 1e9
            print(
                f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | "
                f"cycles={total} | mem={mem:.2f}/{mem_res:.2f} GiB"
            )

    if rank == 0:
        total = max_iters * num_layers
        print(f"  PASSED: {total} allgather+resize cycles without crash")


def test_fsdp_sim(device, max_iters, num_layers, rank, world_size):
    """Full FSDP simulation: unshard → forward → reshard, with model compute."""
    if rank == 0:
        print(f"\n--- Test: Full FSDP Simulation ({num_layers} layers/iter) ---")
        print(f"  Pattern: alloc → allgather → compute → free, repeated per layer")

    hidden = 2048
    shard_size = hidden * hidden // world_size  # one linear layer, sharded
    full_size = hidden * hidden

    # Persistent shards (like FSDP's sharded params)
    shards = [torch.randn(shard_size, device=device, dtype=torch.bfloat16) for _ in range(num_layers)]
    # Persistent unsharded buffers
    unsharded = [torch.empty(full_size, device=device, dtype=torch.bfloat16) for _ in range(num_layers)]
    for u in unsharded:
        u.untyped_storage().resize_(0)

    ag_stream = torch.xpu.Stream(device)

    t0 = time.time()
    for i in range(max_iters):
        # Simulate 3 no_grad forward passes (like RL pattern)
        for fwd_pass in range(3):
            x = torch.randn(4, 512, hidden, device=device, dtype=torch.bfloat16)

            for layer_idx in range(num_layers):
                # Unshard
                storage = unsharded[layer_idx].untyped_storage()
                storage.resize_(full_size * 2)

                with torch.xpu.stream(ag_stream):
                    output_chunks = list(unsharded[layer_idx].chunk(world_size))
                    dist.all_gather(output_chunks, shards[layer_idx])
                    ag_event = ag_stream.record_event()

                torch.xpu.current_stream(device).wait_event(ag_event)

                # Compute (simulate linear layer)
                with torch.no_grad():
                    weight = unsharded[layer_idx].view(hidden, hidden)
                    x = x @ weight.T

                # Reshard
                reshard_event = torch.xpu.Event()
                reshard_event.record()
                storage.resize_(0)

                del ag_event, reshard_event

            del x

        # Fourth forward (grad-enabled, no backward for simplicity)
        with torch.no_grad():
            x = torch.randn(4, 512, hidden, device=device, dtype=torch.bfloat16)
            for layer_idx in range(num_layers):
                storage = unsharded[layer_idx].untyped_storage()
                storage.resize_(full_size * 2)
                output_chunks = list(unsharded[layer_idx].chunk(world_size))
                dist.all_gather(output_chunks, shards[layer_idx])
                weight = unsharded[layer_idx].view(hidden, hidden)
                x = x @ weight.T
                reshard_event = torch.xpu.Event()
                reshard_event.record()
                storage.resize_(0)
                del reshard_event
            del x

        torch.xpu.synchronize(device)

        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            total_allgathers = (i + 1) * num_layers * 4  # 3 no_grad + 1 grad
            mem = torch.xpu.memory_allocated(device) / 1e9
            mem_res = torch.xpu.memory_reserved(device) / 1e9
            print(
                f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | "
                f"allgathers={total_allgathers} | mem={mem:.2f}/{mem_res:.2f} GiB"
            )

    if rank == 0:
        total = max_iters * num_layers * 4
        print(f"  PASSED: {total} total allgather+resize cycles without crash")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--num-layers", type=int, default=12,
                        help="Simulated layers (default: 12, like repro script)")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--with-allgather", action="store_true")
    parser.add_argument("--fsdp-sim", action="store_true",
                        help="Full FSDP simulation with RL pattern")
    args = parser.parse_args()

    if args.distributed or args.with_allgather or args.fsdp_sim:
        dist.init_process_group(backend="xccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    if rank == 0:
        print(f"=== Storage Resize UR Handle Leak Test ===")
        print(f"Device: {torch.xpu.get_device_name(local_rank)}")

    # Test 1: resize only
    tensors = [torch.randn(2048 * 2048, device=device, dtype=torch.bfloat16)
               for _ in range(args.num_layers)]
    test_resize_only(device, args.max_iters, tensors, rank)
    del tensors

    # Test 2: allgather + resize (if distributed)
    if args.with_allgather or args.fsdp_sim:
        test_resize_with_allgather(device, args.max_iters, args.num_layers, rank, world_size)

    # Test 3: full FSDP sim
    if args.fsdp_sim:
        test_fsdp_sim(device, args.max_iters, args.num_layers, rank, world_size)

    if rank == 0:
        print(f"\nAll tests PASSED.")

    if args.distributed or args.with_allgather or args.fsdp_sim:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test: does torch.Event / stream.record_event() leak UR handles on XPU?

FSDP calls record_event() on every allgather for stream synchronization.
If events leak UR handles, this would explain the crash pattern:
- More allgathers (FSDP no_grad) → more events → faster exhaustion
- Simple fwd/bwd has fewer allgathers → fewer events → slower exhaustion

Tests:
  1. Raw events only (no allgather)
  2. Allgather + events (FSDP-like pattern)
  3. Multiple streams + events (full FSDP pattern)

Usage:
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_event_leak.py [--test events|streams|fsdp_like]
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def test_events_only(device, max_iters, events_per_iter, rank):
    """Create and record events without allgather."""
    if rank == 0:
        print(f"\n--- Test: Events Only ({events_per_iter}/iter) ---")

    t0 = time.time()
    for i in range(max_iters):
        for _ in range(events_per_iter):
            event = torch.xpu.Event()
            event.record()
            torch.xpu.current_stream(device).wait_event(event)
            del event

        if rank == 0 and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            total = (i + 1) * events_per_iter
            mem = torch.xpu.memory_allocated(device) / 1e9
            print(f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | events={total} | mem={mem:.2f} GiB")

    if rank == 0:
        total = max_iters * events_per_iter
        print(f"  PASSED: {total} events created/destroyed without crash")


def test_streams_and_events(device, max_iters, ops_per_iter, rank):
    """Create events across multiple streams (like FSDP's allgather streams)."""
    if rank == 0:
        print(f"\n--- Test: Multi-Stream Events ({ops_per_iter}/iter) ---")

    stream1 = torch.xpu.Stream(device)
    stream2 = torch.xpu.Stream(device)

    t0 = time.time()
    for i in range(max_iters):
        for _ in range(ops_per_iter):
            # Simulate FSDP's stream switching pattern
            with torch.xpu.stream(stream1):
                t = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
                event1 = stream1.record_event()

            stream2.wait_event(event1)
            with torch.xpu.stream(stream2):
                t2 = t + 1
                event2 = stream2.record_event()

            torch.xpu.current_stream(device).wait_event(event2)
            del t, t2, event1, event2

        if rank == 0 and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            total = (i + 1) * ops_per_iter
            mem = torch.xpu.memory_allocated(device) / 1e9
            print(f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | ops={total} | mem={mem:.2f} GiB")

    if rank == 0:
        total = max_iters * ops_per_iter
        print(f"  PASSED: {total} multi-stream ops without crash")


def test_fsdp_like(device, max_iters, layers_per_iter, rank, world_size):
    """Simulate FSDP's allgather+event pattern as closely as possible."""
    if rank == 0:
        print(f"\n--- Test: FSDP-like Allgather+Events ({layers_per_iter} layers/iter) ---")

    ag_copy_in_stream = torch.xpu.Stream(device)
    ag_stream = torch.xpu.Stream(device)
    shard_size = 2048 * 2048  # ~16 MiB per "layer" in bf16

    t0 = time.time()
    for i in range(max_iters):
        for _ in range(layers_per_iter):
            # Simulate FSDP's foreach_all_gather
            with torch.xpu.stream(ag_copy_in_stream):
                shard = torch.randn(shard_size, device=device, dtype=torch.bfloat16)
                ag_output = torch.empty(shard_size * world_size, device=device, dtype=torch.bfloat16)
                # Copy shard into correct position
                ag_output[rank * shard_size:(rank + 1) * shard_size].copy_(shard)

            ag_stream.wait_stream(ag_copy_in_stream)
            with torch.xpu.stream(ag_stream):
                # Allgather
                output_tensors = list(ag_output.chunk(world_size))
                dist.all_gather(output_tensors, shard)
                ag_event = ag_stream.record_event()

            # Wait for allgather (like FSDP's wait_for_unshard)
            torch.xpu.current_stream(device).wait_event(ag_event)

            # Simulate using the gathered param
            with torch.no_grad():
                full_size = shard_size * world_size
                x = torch.randn(32, full_size, device=device, dtype=torch.bfloat16)
                y = x @ ag_output.unsqueeze(-1)

            # Simulate reshard (free unsharded)
            copy_out_event = torch.xpu.Event()
            copy_out_event.record()

            del shard, ag_output, x, y, ag_event, copy_out_event

        torch.xpu.synchronize(device)

        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            total = (i + 1) * layers_per_iter
            mem = torch.xpu.memory_allocated(device) / 1e9
            mem_res = torch.xpu.memory_reserved(device) / 1e9
            print(
                f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | "
                f"allgathers={total} | mem={mem:.2f}/{mem_res:.2f} GiB"
            )

    if rank == 0:
        total = max_iters * layers_per_iter
        print(f"  PASSED: {total} FSDP-like ops without crash")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["events", "streams", "fsdp_like", "all"],
                        default="all")
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--ops-per-iter", type=int, default=20,
                        help="Events/ops per iteration")
    args = parser.parse_args()

    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    if rank == 0:
        print(f"=== Event/Stream UR Handle Leak Test ===")
        print(f"World size: {world_size}, device: {torch.xpu.get_device_name(local_rank)}")

    if args.test in ("events", "all"):
        test_events_only(device, args.max_iters, args.ops_per_iter, rank)

    if args.test in ("streams", "all"):
        test_streams_and_events(device, args.max_iters, args.ops_per_iter, rank)

    if args.test in ("fsdp_like", "all"):
        test_fsdp_like(device, args.max_iters, args.ops_per_iter, rank, world_size)

    if rank == 0:
        print(f"\nAll tests PASSED — events and streams do not leak UR handles in isolation.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

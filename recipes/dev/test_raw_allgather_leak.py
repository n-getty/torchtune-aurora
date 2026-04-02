#!/usr/bin/env python3
"""
Test: does raw torch.distributed.all_gather leak UR handles on XPU?

If this crashes with UR_RESULT_ERROR_OUT_OF_RESOURCES, the leak is in the
Level Zero / XCCL collective implementation (driver-level).

If this runs 500+ iterations without issue, the leak is in FSDP's parameter
lifecycle management under no_grad (software-level, fixable in PyTorch).

Usage:
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_raw_allgather_leak.py

  # With more tensors per iteration (stress test):
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_raw_allgather_leak.py --tensors-per-iter 20

  # Simulating FSDP allgather pattern (allocate + gather + free):
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_raw_allgather_leak.py --fsdp-pattern
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--tensors-per-iter", type=int, default=10,
                        help="Number of allgather ops per iteration (FSDP does ~1 per layer)")
    parser.add_argument("--tensor-size", type=int, default=2048 * 2048,
                        help="Elements per tensor (default: ~16 MiB in bf16)")
    parser.add_argument("--fsdp-pattern", action="store_true",
                        help="Simulate FSDP: allgather shards into full tensor, use, then free")
    parser.add_argument("--with-grad", action="store_true",
                        help="Run allgathers WITH grad enabled (control test)")
    args = parser.parse_args()

    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    if rank == 0:
        print(f"=== Raw Allgather UR Handle Leak Test ===")
        print(f"World size: {world_size}, device: {torch.xpu.get_device_name(local_rank)}")
        print(f"Tensors/iter: {args.tensors_per_iter}, size: {args.tensor_size} elements")
        print(f"FSDP pattern: {args.fsdp_pattern}, with_grad: {args.with_grad}")
        print(f"Max iterations: {args.max_iters}")
        print()

    t0 = time.time()

    for i in range(args.max_iters):
        for t in range(args.tensors_per_iter):
            if args.fsdp_pattern:
                # Simulate FSDP allgather pattern:
                # Each rank has a shard, allgather to reconstruct full param
                shard_size = args.tensor_size // world_size
                shard = torch.randn(shard_size, device=device, dtype=torch.bfloat16)

                if args.with_grad:
                    # Grad-enabled: like FSDP's training forward
                    shard.requires_grad_(True)
                    output_tensors = [
                        torch.empty(shard_size, device=device, dtype=torch.bfloat16)
                        for _ in range(world_size)
                    ]
                    dist.all_gather(output_tensors, shard)
                    full_param = torch.cat(output_tensors)
                    # Simulate using the param
                    full_size = shard_size * world_size
                    x = torch.randn(32, full_size, device=device, dtype=torch.bfloat16)
                    y = x @ full_param.unsqueeze(-1)
                    del y, x, full_param, output_tensors, shard
                else:
                    # No-grad: like FSDP's no_grad forward (the bug trigger)
                    with torch.no_grad():
                        output_tensors = [
                            torch.empty(shard_size, device=device, dtype=torch.bfloat16)
                            for _ in range(world_size)
                        ]
                        dist.all_gather(output_tensors, shard)
                        full_param = torch.cat(output_tensors)
                        full_size = shard_size * world_size
                        x = torch.randn(32, full_size, device=device, dtype=torch.bfloat16)
                        y = x @ full_param.unsqueeze(-1)
                        del y, x, full_param, output_tensors, shard
            else:
                # Simple allgather: just gather tensors
                tensor = torch.randn(args.tensor_size, device=device, dtype=torch.bfloat16)
                output_tensors = [
                    torch.empty(args.tensor_size, device=device, dtype=torch.bfloat16)
                    for _ in range(world_size)
                ]

                if args.with_grad:
                    tensor.requires_grad_(True)
                    dist.all_gather(output_tensors, tensor)
                else:
                    with torch.no_grad():
                        dist.all_gather(output_tensors, tensor)

                del output_tensors, tensor

        # Sync and report
        torch.xpu.synchronize(device)

        if rank == 0 and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            mem = torch.xpu.memory_allocated(device) / 1e9
            mem_res = torch.xpu.memory_reserved(device) / 1e9
            total_ops = (i + 1) * args.tensors_per_iter
            print(
                f"  iter {i+1:4d}/{args.max_iters} | "
                f"{elapsed:.1f}s | "
                f"total_allgathers={total_ops} | "
                f"mem={mem:.2f}/{mem_res:.2f} GiB"
            )

    if rank == 0:
        elapsed = time.time() - t0
        total_ops = args.max_iters * args.tensors_per_iter
        print(f"\nCompleted {args.max_iters} iterations ({total_ops} allgather ops) "
              f"in {elapsed:.1f}s without crash.")
        if args.with_grad:
            print("RESULT: With-grad allgather is STABLE.")
        else:
            print("RESULT: No-grad allgather is STABLE → leak is in FSDP, not driver.")
            print("        (If this had crashed, leak would be in driver/XCCL.)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

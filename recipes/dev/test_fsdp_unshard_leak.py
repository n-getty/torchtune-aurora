#!/usr/bin/env python3
"""
Bisect the FSDP UR handle leak by testing FSDP internals incrementally.

Tests (from simplest to most complex):
  1. fully_shard model, call unshard/reshard in loop (no forward)
  2. fully_shard model, forward under no_grad (the actual bug pattern)
  3. fully_shard model, forward WITH grad (control — should be stable)
  4. Manual FSDP-style: use foreach_all_gather + copy_out + DTensor wrapping
  5. Test _unsafe_preserve_version_counter in isolation

Usage:
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_fsdp_unshard_leak.py --test unshard_reshard

  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_fsdp_unshard_leak.py --test no_grad_forward

  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_fsdp_unshard_leak.py --test all
"""

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def create_model(num_layers=12, hidden=1024, num_heads=8, dtype=torch.bfloat16):
    """Same model as repro script."""
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.RMSNorm(hidden, dtype=dtype)
            self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dtype=dtype)
            self.ln2 = nn.RMSNorm(hidden, dtype=dtype)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden * 4, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden * 4, hidden, dtype=dtype),
            )

        def forward(self, x):
            h = self.ln(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h
            x = x + self.ffn(self.ln2(x))
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32000, hidden, dtype=dtype)
            self.layers = nn.ModuleList([Block() for _ in range(num_layers)])
            self.norm = nn.RMSNorm(hidden, dtype=dtype)
            self.head = nn.Linear(hidden, 32000, dtype=dtype)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(self.norm(x))

    return Model()


def shard_model(model):
    """Apply fully_shard like torchtune does."""
    from torch.distributed._composable.fsdp import fully_shard
    for layer in reversed(model.layers):
        fully_shard(layer)
    fully_shard(model)
    return model


def report(rank, device, i, max_iters, t0, label=""):
    if rank == 0 and (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        mem = torch.xpu.memory_allocated(device) / 1e9
        mem_res = torch.xpu.memory_reserved(device) / 1e9
        print(f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | mem={mem:.2f}/{mem_res:.2f} GiB {label}")


def test_unshard_reshard(device, rank, max_iters):
    """Test 1: Just call FSDP's unshard/reshard without forward."""
    if rank == 0:
        print(f"\n--- Test: unshard/reshard cycle (no forward) ---")

    model = create_model().to(device)
    model = shard_model(model)

    t0 = time.time()
    for i in range(max_iters):
        # Manually trigger unshard on each param group
        for module in model.modules():
            if hasattr(module, '_fsdp_param_group') and module._fsdp_param_group is not None:
                pg = module._fsdp_param_group
                pg.unshard()
                pg.wait_for_unshard()
                pg.reshard()

        torch.xpu.synchronize(device)
        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} unshard/reshard cycles")
    del model


def test_no_grad_forward(device, rank, max_iters, num_fwd=3):
    """Test 2: The actual bug pattern — multiple no_grad forwards through FSDP model."""
    if rank == 0:
        print(f"\n--- Test: {num_fwd}x no_grad forward per iter (THE BUG PATTERN) ---")

    model = create_model().to(device)
    model = shard_model(model)
    model.eval()

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        for _ in range(num_fwd):
            with torch.no_grad():
                out = model(input_ids)
                del out

        torch.xpu.synchronize(device)
        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} iterations ({max_iters * num_fwd} no_grad forwards)")
    del model


def test_grad_forward(device, rank, max_iters):
    """Test 3: Control — grad-enabled forward + backward (should be stable)."""
    if rank == 0:
        print(f"\n--- Test: grad forward + backward (CONTROL) ---")

    model = create_model().to(device)
    model = shard_model(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del out, loss

        torch.xpu.synchronize(device)
        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} grad forward+backward iterations")
    del model, optimizer


def test_no_grad_then_grad(device, rank, max_iters):
    """Test 4: RL pattern — 2x no_grad forward then 1x grad forward+backward."""
    if rank == 0:
        print(f"\n--- Test: 2x no_grad + 1x grad forward+backward (RL PATTERN) ---")

    model = create_model().to(device)
    model = shard_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        # 2 no_grad forwards
        model.eval()
        for _ in range(2):
            with torch.no_grad():
                out = model(input_ids)
                del out

        # 1 grad forward + backward
        model.train()
        out = model(input_ids)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del out, loss

        torch.xpu.synchronize(device)
        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} RL-pattern iterations")
    del model, optimizer


def test_no_grad_only(device, rank, max_iters):
    """Test 5: ONLY no_grad forwards, no backward ever — isolates no_grad path."""
    if rank == 0:
        print(f"\n--- Test: no_grad forward ONLY (no backward ever) ---")

    model = create_model().to(device)
    model = shard_model(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        with torch.no_grad():
            out = model(input_ids)
            del out

        torch.xpu.synchronize(device)
        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} no_grad-only forwards")
    del model


def test_version_counter(device, rank, max_iters):
    """Test 6: _unsafe_preserve_version_counter in isolation."""
    if rank == 0:
        print(f"\n--- Test: _unsafe_preserve_version_counter ---")

    tensors = [torch.randn(2048, 2048, device=device, dtype=torch.bfloat16) for _ in range(12)]

    t0 = time.time()
    for i in range(max_iters):
        for t in tensors:
            with torch.autograd._unsafe_preserve_version_counter(t):
                t.mul_(0.99)

        if rank == 0 and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            mem = torch.xpu.memory_allocated(device) / 1e9
            print(f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | mem={mem:.2f} GiB")

    if rank == 0:
        print(f"  PASSED: {max_iters} version_counter cycles")
    del tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["unshard_reshard", "no_grad_forward", "grad_forward",
                                 "no_grad_then_grad", "no_grad_only", "version_counter", "all"])
    parser.add_argument("--max-iters", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    if rank == 0:
        print(f"=== FSDP Unshard/Reshard Leak Bisection ===")
        print(f"Device: {torch.xpu.get_device_name(local_rank)}")
        print(f"Max iterations: {args.max_iters}")

    tests = {
        "version_counter": test_version_counter,
        "unshard_reshard": test_unshard_reshard,
        "no_grad_only": test_no_grad_only,
        "no_grad_forward": test_no_grad_forward,
        "grad_forward": test_grad_forward,
        "no_grad_then_grad": test_no_grad_then_grad,
    }

    if args.test == "all":
        # Run in order from simplest to most complex
        for name in ["version_counter", "unshard_reshard", "grad_forward",
                      "no_grad_only", "no_grad_forward", "no_grad_then_grad"]:
            try:
                tests[name](device, rank, args.max_iters)
            except Exception as e:
                if rank == 0:
                    print(f"  FAILED: {name} — {e}")
                break
            gc.collect()
            torch.xpu.empty_cache()
    else:
        tests[args.test](device, rank, args.max_iters)

    if rank == 0:
        print(f"\nDone.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

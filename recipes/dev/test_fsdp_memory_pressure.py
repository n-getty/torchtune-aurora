#!/usr/bin/env python3
"""
Bisect: is the UR leak triggered by memory pressure or by specific tensor ops?

The repro crashes at ~70 iters with 1.3/4.6 GiB mem.
Our bisection test passes 500+ iters at 0.36/0.86 GiB mem.

This test increases memory usage step by step to find the threshold.

Usage:
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    test_fsdp_memory_pressure.py [--test mem_alloc|logprob_ops|full_rl]
"""

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def create_model(num_layers=12, hidden=1024, num_heads=8, dtype=torch.bfloat16):
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


def shard_model(model, per_layer=False):
    from torch.distributed._composable.fsdp import fully_shard
    if per_layer:
        for layer in reversed(model.layers):
            fully_shard(layer)
    fully_shard(model)
    return model


def report(rank, device, i, max_iters, t0):
    if rank == 0 and (i + 1) % 5 == 0:
        elapsed = time.time() - t0
        mem = torch.xpu.memory_allocated(device) / 1e9
        mem_res = torch.xpu.memory_reserved(device) / 1e9
        print(f"  iter {i+1:4d}/{max_iters} | {elapsed:.1f}s | mem={mem:.2f}/{mem_res:.2f} GiB")


def test_mem_alloc(device, rank, max_iters):
    """Just allocate large tensors alongside FSDP no_grad forwards."""
    if rank == 0:
        print(f"\n--- Test: no_grad forward + large tensor allocation ---")

    model = create_model().to(device)
    model = shard_model(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        # Allocate ~3 GiB of extra tensors to match repro's memory footprint
        extra_tensors = [
            torch.randn(4, 512, 32000, device=device, dtype=torch.bfloat16)  # ~128 MiB each
            for _ in range(24)  # ~3 GiB total
        ]

        with torch.no_grad():
            out = model(input_ids)
            del out

        del extra_tensors
        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} iterations")


def test_logprob_ops(device, rank, max_iters):
    """No_grad forward + log_softmax + gather (like RL logprob computation)."""
    if rank == 0:
        print(f"\n--- Test: no_grad forward + logprob ops ---")

    model = create_model().to(device)
    model = shard_model(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        with torch.no_grad():
            # Forward 1: generation simulation
            logits = model(input_ids)
            probs = F.softmax(logits, dim=-1)
            gen_tokens = torch.argmax(logits, dim=-1)
            del logits, probs

            # Forward 2: logprob computation
            logits2 = model(input_ids)
            lp = F.log_softmax(logits2, dim=-1)
            logprobs = torch.gather(lp, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            del logits2, lp

            del gen_tokens, logprobs

        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} iterations")


def test_full_rl(device, rank, max_iters):
    """Full RL pattern matching the repro script exactly."""
    if rank == 0:
        print(f"\n--- Test: FULL RL pattern (matches repro_xpu_resource_leak.py) ---")

    model = create_model().to(device)
    model = shard_model(model)

    ref = create_model().to(device)
    ref = shard_model(ref)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    model.train()

    for i in range(max_iters):
        # Step 1: simulate generation
        with torch.no_grad():
            gen_logits = model(input_ids)
            gen_probs = F.softmax(gen_logits, dim=-1)
            gen_tokens = torch.argmax(gen_logits, dim=-1)
            del gen_logits, gen_probs

        # Step 2: policy logprobs
        with torch.no_grad():
            chunk_logits = model(input_ids)
            chunk_lp = F.log_softmax(chunk_logits, dim=-1)
            policy_logprobs = torch.gather(chunk_lp, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            del chunk_logits, chunk_lp

        # Step 3: ref logprobs
        with torch.no_grad():
            ref_logits = ref(input_ids)
            ref_lp = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_lp, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            del ref_logits, ref_lp

        # Step 4: loss + backward
        kl = policy_logprobs - ref_logprobs
        rewards = torch.randn(4, device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        logits = model(input_ids)
        lp = F.log_softmax(logits, dim=-1)
        curr_logprobs = torch.gather(lp, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        del logits, lp

        ratio = torch.exp(curr_logprobs - policy_logprobs.detach())
        pg_loss = -(advantages.unsqueeze(-1) * ratio).mean()
        kl_loss = kl.mean() * 0.01
        loss = pg_loss + kl_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del gen_tokens, policy_logprobs, ref_logprobs, kl, curr_logprobs, loss

        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} iterations")


def test_two_models_no_grad(device, rank, max_iters):
    """Two FSDP models, both under no_grad — isolate multi-model effect."""
    if rank == 0:
        print(f"\n--- Test: Two FSDP models, no_grad forward ---")

    model1 = create_model().to(device)
    model1 = shard_model(model1)
    model1.eval()
    for p in model1.parameters():
        p.requires_grad = False

    model2 = create_model().to(device)
    model2 = shard_model(model2)
    model2.eval()
    for p in model2.parameters():
        p.requires_grad = False

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        with torch.no_grad():
            out1 = model1(input_ids)
            del out1
            out2 = model2(input_ids)
            del out2

        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} iterations")


def test_no_grad_with_intermediates(device, rank, max_iters):
    """No_grad forward keeping intermediate tensors alive (like RL pattern)."""
    if rank == 0:
        print(f"\n--- Test: no_grad forward, keeping intermediate tensors alive ---")

    model = create_model().to(device)
    model = shard_model(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    input_ids = torch.randint(0, 32000, (4, 512), device=device)

    t0 = time.time()
    for i in range(max_iters):
        # Keep intermediate tensors alive across multiple forwards
        with torch.no_grad():
            logits1 = model(input_ids)  # ~256 MiB (4×512×32000 bf16)
            tokens = torch.argmax(logits1, dim=-1)  # small
            # Don't delete logits1 yet — keep alive during second forward
            logits2 = model(input_ids)
            lp = torch.gather(F.log_softmax(logits2, dim=-1), -1, tokens.unsqueeze(-1))
            # NOW delete everything
            del logits1, logits2, tokens, lp

        report(rank, device, i, max_iters, t0)

    if rank == 0:
        print(f"  PASSED: {max_iters} iterations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all",
                        choices=["mem_alloc", "logprob_ops", "full_rl",
                                 "two_models", "intermediates", "all"])
    parser.add_argument("--max-iters", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    if rank == 0:
        print(f"=== FSDP Memory Pressure Leak Bisection ===")
        print(f"Device: {torch.xpu.get_device_name(local_rank)}")

    tests = {
        "mem_alloc": test_mem_alloc,
        "logprob_ops": test_logprob_ops,
        "two_models": test_two_models_no_grad,
        "intermediates": test_no_grad_with_intermediates,
        "full_rl": test_full_rl,
    }

    if args.test == "all":
        for name in ["two_models", "intermediates", "logprob_ops", "full_rl"]:
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

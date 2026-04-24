#!/usr/bin/env python3
"""
Minimal reproduction: UR_RESULT_ERROR_OUT_OF_RESOURCES with FSDP + empty_cache() on XPU.

ROOT CAUSE: torch.xpu.empty_cache() + FSDP storage.resize_() leaks UR handles in
Level Zero. Each zeMemAllocDevice/zeMemFree cycle (triggered when empty_cache() forces
the caching allocator to release blocks, then FSDP re-acquires them) leaks a UR handle.
After ~70 iterations the handle pool is exhausted.

WORKAROUND: Remove empty_cache() calls. The caching allocator reuses blocks from its
free pool without touching Level Zero, preventing the leak. Use --no-empty-cache-in-chunks
to demonstrate the fix (200+ iterations stable).

Simulates a GRPO/PPO reinforcement learning training pattern:
  1. Generate sequences (forward-only through model, creating varied-length tensors)
  2. Compute log-probabilities with policy model (chunked forward, no grad)
     -> empty_cache() between chunks (THIS TRIGGERS THE BUG with FSDP)
  3. Compute log-probabilities with reference model (chunked forward, no grad)
     -> empty_cache() between chunks (THIS TRIGGERS THE BUG with FSDP)
  4. Compute loss and backward through policy model

Environment:
  - Hardware: Intel Data Center GPU Max 1550 (Aurora HPC, 12 tiles/node, 64 GiB/tile)
  - Driver: I915_25.2.29_PSB_250224.35 (Level Zero 1.24.0)
  - PyTorch: 2.10.0a0+git449b176 (Aurora frameworks 2025.3.1)
  - OS: SLES 15 SP4 (kernel 5.14.21-150400.24.55-default)

Usage:
  # TRIGGERS THE BUG — FSDP2 + RL + empty_cache() in chunked forwards (~70 iters):
  python3 -m torch.distributed.run --nproc_per_node=2 \
    repro_xpu_resource_leak.py --fsdp --layers 12 --hidden 1024 --heads 8

  # WORKAROUND — skip empty_cache() in chunks (200+ iterations stable):
  python3 -m torch.distributed.run --nproc_per_node=2 \
    repro_xpu_resource_leak.py --fsdp --layers 12 --hidden 1024 --heads 8 \
    --no-empty-cache-in-chunks

  # TRIGGERS THE BUG — FSDP1 (~145 iters, leaks at half the rate of FSDP2):
  python3 -m torch.distributed.run --nproc_per_node=2 \
    repro_xpu_resource_leak.py --fsdp1 --layers 12 --hidden 1024 --heads 8

  # DOES NOT TRIGGER — no FSDP (500+ iters stable even with empty_cache):
  ZE_AFFINITY_MASK=0 python3 repro_xpu_resource_leak.py --layers 12 --hidden 1024 --heads 8

  # DOES NOT TRIGGER — simple fwd/bwd (no RL pattern, 500+ iters stable):
  python3 -m torch.distributed.run --nproc_per_node=2 \
    repro_xpu_resource_leak.py --fsdp --simple --layers 12 --hidden 1024 --heads 8

Crash signature:
  RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)
  or:
  Segmentation fault from GPU at 0xff000004XXXXXXXX, ctx_id: 1 (CCS)
    type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.

See docs/bugs/intel_xpu_resource_leak_bug_report.md for full analysis.
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(num_layers, hidden_dim, num_heads, vocab_size=32000, dtype=torch.bfloat16):
    """Create a minimal transformer decoder (no dependencies beyond PyTorch)."""

    class TransformerBlock(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super().__init__()
            self.ln1 = nn.RMSNorm(hidden_dim, dtype=dtype)
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True, dtype=dtype,
            )
            self.ln2 = nn.RMSNorm(hidden_dim, dtype=dtype)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype),
            )

        def forward(self, x, mask=None):
            h = self.ln1(x)
            h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + h
            x = x + self.ffn(self.ln2(x))
            return x

    class MiniDecoder(nn.Module):
        def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim, dtype=dtype)
            self.layers = nn.ModuleList(
                [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
            )
            self.norm = nn.RMSNorm(hidden_dim, dtype=dtype)
            self.head = nn.Linear(hidden_dim, vocab_size, dtype=dtype)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            seq_len = x.shape[1]
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            for layer in self.layers:
                x = layer(x, mask=mask)
            x = self.norm(x)
            return self.head(x)

    return MiniDecoder(vocab_size, hidden_dim, num_layers, num_heads)


def setup_fsdp(model, version=2):
    """Wrap model with FSDP."""
    if version == 2:
        from torch.distributed._composable.fsdp import fully_shard
        fully_shard(model)
        return model
    else:
        # FSDP1 (classic FullyShardedDataParallel)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            use_orig_params=True,
            limit_all_gathers=True,
        )
        return model


def main():
    parser = argparse.ArgumentParser(description="XPU resource leak reproduction")
    parser.add_argument("--layers", type=int, default=36,
                        help="Transformer layers (default: 36)")
    parser.add_argument("--hidden", type=int, default=2048,
                        help="Hidden dimension (default: 2048)")
    parser.add_argument("--heads", type=int, default=16,
                        help="Attention heads (default: 16)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length (default: 512)")
    parser.add_argument("--num-seqs", type=int, default=4,
                        help="Sequences per iteration (default: 4)")
    parser.add_argument("--fwd-chunk", type=int, default=0,
                        help="If >0, chunk forward passes into this many seqs at a time")
    parser.add_argument("--max-iters", type=int, default=200,
                        help="Maximum iterations (default: 200)")
    parser.add_argument("--fsdp", action="store_true",
                        help="Use FSDP2 (requires torchrun)")
    parser.add_argument("--fsdp1", action="store_true",
                        help="Use FSDP1 instead of FSDP2 (requires torchrun)")
    parser.add_argument("--no-ref-model", action="store_true",
                        help="Skip reference model (test single-model pattern)")
    parser.add_argument("--empty-cache", action="store_true",
                        help="Call empty_cache + gc.collect each iter (THIS TRIGGERS THE BUG)")
    parser.add_argument("--no-empty-cache-in-chunks", action="store_true",
                        help="Skip empty_cache in chunked forward loops (workaround)")
    parser.add_argument("--simple", action="store_true",
                        help="Simple fwd/bwd loop only (no RL pattern)")
    args = parser.parse_args()

    # --- Device setup ---
    use_distributed = args.fsdp or args.fsdp1
    if use_distributed:
        torch.distributed.init_process_group(backend="xccl")
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("xpu:0")
        torch.xpu.set_device(0)

    fsdp_version = 1 if args.fsdp1 else 2

    # --- Models ---
    policy = create_model(args.layers, args.hidden, args.heads).to(device)
    param_count = sum(p.numel() for p in policy.parameters())

    ref = None
    if not args.no_ref_model:
        ref = create_model(args.layers, args.hidden, args.heads).to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False

    if use_distributed:
        policy = setup_fsdp(policy, version=fsdp_version)
        if ref is not None:
            ref = setup_fsdp(ref, version=fsdp_version)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    fsdp_label = f"FSDP{fsdp_version}" if use_distributed else "None"
    if rank == 0:
        print(f"=== XPU Resource Leak Reproduction ===")
        print(f"Model: {args.layers} layers, hidden={args.hidden}, heads={args.heads}")
        print(f"Parameters: {param_count:,} ({param_count * 2 / 1e9:.1f} GiB in BF16)")
        print(f"Sequences/iter: {args.num_seqs} x {args.seq_len} tokens")
        print(f"Ref model: {ref is not None}, FSDP: {fsdp_label}, simple: {args.simple}")
        print(f"Device: {torch.xpu.get_device_name(local_rank)}")
        print(f"Driver: I915 {open('/sys/module/i915/version').read().strip()}" if os.path.exists('/sys/module/i915/version') else "")
        print(f"PyTorch: {torch.__version__}")
        print(f"Level Zero: {os.popen('rpm -q level-zero 2>/dev/null').read().strip()}")
        print(f"Max iterations: {args.max_iters}")
        mem = torch.xpu.memory_allocated(device) / 1e9
        print(f"Initial memory: {mem:.1f} GiB")
        print()

    policy.train()
    t0 = time.time()
    fwd_chunk = args.fwd_chunk if args.fwd_chunk > 0 else args.num_seqs

    for i in range(args.max_iters):
        input_ids = torch.randint(0, 32000, (args.num_seqs, args.seq_len), device=device)
        labels = torch.randint(0, 32000, (args.num_seqs, args.seq_len), device=device)

        if args.simple:
            # --- Simple forward/backward (does NOT crash) ---
            logits = policy(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            # --- RL-style pattern (crashes) ---

            # Step 1: Simulate generation — forward-only, create intermediate tensors
            with torch.no_grad():
                gen_logits = policy(input_ids)
                # Simulate sampling: create token-level tensors that will be used later
                gen_probs = F.softmax(gen_logits, dim=-1)
                gen_tokens = torch.argmax(gen_logits, dim=-1)
                del gen_logits, gen_probs

            # Step 2: Policy forward — compute logprobs (chunked, like GRPO recipe)
            policy_logprobs_chunks = []
            for cs in range(0, args.num_seqs, fwd_chunk):
                ce = min(cs + fwd_chunk, args.num_seqs)
                with torch.no_grad():
                    chunk_logits = policy(input_ids[cs:ce])
                    chunk_lp = F.log_softmax(chunk_logits, dim=-1)
                    # Gather logprobs for specific tokens
                    chunk_lp = torch.gather(
                        chunk_lp, -1, gen_tokens[cs:ce].unsqueeze(-1)
                    ).squeeze(-1)
                    policy_logprobs_chunks.append(chunk_lp)
                    del chunk_logits
                if not args.no_empty_cache_in_chunks:
                    torch.xpu.empty_cache()  # THIS TRIGGERS THE BUG with FSDP
            policy_logprobs = torch.cat(policy_logprobs_chunks, dim=0)
            del policy_logprobs_chunks

            # Step 3: Ref model forward — compute ref logprobs (chunked)
            if ref is not None:
                ref_logprobs_chunks = []
                for cs in range(0, args.num_seqs, fwd_chunk):
                    ce = min(cs + fwd_chunk, args.num_seqs)
                    with torch.no_grad():
                        chunk_logits = ref(input_ids[cs:ce])
                        chunk_lp = F.log_softmax(chunk_logits, dim=-1)
                        chunk_lp = torch.gather(
                            chunk_lp, -1, gen_tokens[cs:ce].unsqueeze(-1)
                        ).squeeze(-1)
                        ref_logprobs_chunks.append(chunk_lp)
                        del chunk_logits
                    if not args.no_empty_cache_in_chunks:
                        torch.xpu.empty_cache()  # THIS TRIGGERS THE BUG with FSDP
                ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
                del ref_logprobs_chunks
            else:
                ref_logprobs = policy_logprobs.detach()

            # Step 4: Compute loss and backward (with grad)
            # Simulate GRPO/PPO loss: policy gradient + KL penalty
            kl = policy_logprobs - ref_logprobs
            rewards = torch.randn(args.num_seqs, device=device)  # fake rewards
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Re-forward through policy WITH grad for backward
            logits = policy(input_ids)
            lp = F.log_softmax(logits, dim=-1)
            curr_logprobs = torch.gather(lp, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
            del logits, lp

            # Policy gradient loss
            ratio = torch.exp(curr_logprobs - policy_logprobs.detach())
            pg_loss = -(advantages.unsqueeze(-1) * ratio).mean()
            kl_loss = kl.mean() * 0.01
            loss = pg_loss + kl_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            del gen_tokens, policy_logprobs, ref_logprobs, kl, curr_logprobs

        if args.empty_cache:
            gc.collect()
            torch.xpu.empty_cache()

        if rank == 0 and (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            mem = torch.xpu.memory_allocated(device) / 1e9
            mem_reserved = torch.xpu.memory_reserved(device) / 1e9
            print(
                f"  iter {i+1:4d}/{args.max_iters} | "
                f"{elapsed:.1f}s | "
                f"{(i+1)/elapsed:.1f} it/s | "
                f"loss={loss.item():.4f} | "
                f"mem={mem:.1f}/{mem_reserved:.1f} GiB"
            )

    if rank == 0:
        elapsed = time.time() - t0
        print(f"\nCompleted {args.max_iters} iterations in {elapsed:.1f}s without crash.")
        print("BUG NOT REPRODUCED with this configuration.")

    if use_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

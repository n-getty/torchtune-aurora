#!/usr/bin/env python3
"""
Minimal reproducer: oneCCL Level-Zero IPC-handle cache growth under FSDP2 AllGather.

!!! KNOWN-INSUFFICIENT AS AN ISOLATION (HW-tested 2026-07-06, do NOT file on this alone) !!!
  At small scale this script CANNOT separate the CCL-internal IPC-handle leak from ordinary
  torch caching-allocator behavior. The VA churn that fills the IPC-handle cache is the SAME
  churn that grows torch `reserved` to cover the largest seqlen seen, so the `mem_get_info`
  drop it measures is dominated by (attributable to) torch reserved, not CCL metadata. Result
  on node x4313c2s7b0n0: churn leg dropped 24.7 GiB total but drop/step -> ~0 after step 10
  (plateau) and reserved grew 24->54 GiB in lockstep; --stable-va control was totally flat.
  The documented ~10 MB/step CCL-internal signature (docs/bugs/ccl_ipc_handle_cache.md) was
  seen in the FULL 32B GRPO recipe at steady state over ~28 steps where reserved is flat.
  To make this filable: run at real 32B scale with a FIXED shape for many steps (reserved flat,
  watch slow ext_free creep), or instrument oneCCL directly (UR_L0 logging / handle count)
  instead of inferring from mem_get_info. Kept as a starting point, not a proof.


ROOT CAUSE (see docs/bugs/ccl_ipc_handle_cache.md):
  oneCCL caches L0 IPC handles for the GPU virtual addresses that participate in
  intra-node collectives, capped by CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD.
    * High threshold (65536): handles are effectively never evicted; each fresh VA
      that FSDP2's AllGather output buffers churn through accumulates O(KB)-O(MB) of
      L0/CCL bookkeeping. Device-global free memory (mem_get_info) drops steadily
      even though PyTorch's own allocator (memory_stats) returns to baseline every
      step. Eventually the device OOMs.
    * Default threshold (~1000): LRU eviction kicks in mid-run; a sibling rank can
      still hold an evicted handle, and the next collective DMAs against a
      no-longer-mapped VA -> GPU page fault "banned:1".

  This is INVISIBLE to torch.xpu.memory_stats() because the leak is inside CCL, not
  the PyTorch caching allocator. The diagnostic signature is:
      mem_get_info FREE drops by ~C/step   while   memory_reserved stays flat.

  This script isolates the effect to *pure FSDP2 + XCCL collectives* — no GRPO, no
  generation, no weight sync, no vLLM. If external free still falls at ~constant
  rate/step here, the source is FSDP2 AllGather VA churn + CCL IPC-handle caching.

WHAT TO LOOK FOR:
  - THRESHOLD=65536 (default in this script): "ext_free_drop/step" trends to a
    stable positive value (device-global leak); reserved is flat after step ~2.
  - THRESHOLD=1000 (pass --low-threshold, i.e. leave CCL default): a banned:1 PDE
    page fault around the tens-of-steps range under enough VA churn.
  - Compare against --stable-va (reuse one fixed AllGather shape so no fresh VAs
    churn): external free should stay ~flat -> confirms VA churn is the driver.

Environment (validated 2026-07-06):
  - Intel Data Center GPU Max 1550 (Aurora, 12 tiles/node, 64 GiB/tile)
  - frameworks/2025.3.1 (PyTorch 2.10.0a0+xpu, oneCCL 2021.17, Level Zero 1.24.0,
    I915_25.2.29)

Usage (single node; run on >= as many tiles as --nproc_per_node so intra-node
XeLink IPC path is exercised — this bug is intra-node, NOT cross-node):

  # Accumulation signature (high threshold) — watch ext_free_drop/step:
  CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 \
  python3 -m torch.distributed.run --standalone --nproc_per_node=10 \
    recipes/dev/repro_ccl_ipc_handle_cache.py --layers 24 --hidden 4096 --heads 32 --steps 40

  # Control: reuse a single fixed VA shape (no churn) — ext_free should stay flat:
  CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 \
  python3 -m torch.distributed.run --standalone --nproc_per_node=10 \
    recipes/dev/repro_ccl_ipc_handle_cache.py --layers 24 --hidden 4096 --heads 32 --steps 40 --stable-va

  # Eviction / banned:1 signature (leave CCL threshold at its ~1000 default):
  python3 -m torch.distributed.run --standalone --nproc_per_node=10 \
    recipes/dev/repro_ccl_ipc_handle_cache.py --layers 24 --hidden 4096 --heads 32 --steps 60

NOTE: NO torch.xpu.empty_cache() anywhere in this loop — that deliberately
separates this bug from the empty_cache()+FSDP UR-handle leak
(docs/bugs/intel_xpu_resource_leak_bug_report.md, repro_xpu_resource_leak.py).
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(num_layers, hidden_dim, num_heads, vocab_size=32000, dtype=torch.bfloat16):
    class Block(nn.Module):
        def __init__(self, h, nh):
            super().__init__()
            self.ln1 = nn.RMSNorm(h, dtype=dtype)
            self.attn = nn.MultiheadAttention(h, nh, batch_first=True, dtype=dtype)
            self.ln2 = nn.RMSNorm(h, dtype=dtype)
            self.ffn = nn.Sequential(
                nn.Linear(h, h * 4, dtype=dtype),
                nn.SiLU(),
                nn.Linear(h * 4, h, dtype=dtype),
            )

        def forward(self, x, mask=None):
            h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
            x = x + h
            return x + self.ffn(self.ln2(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim, dtype=dtype)
            self.layers = nn.ModuleList([Block(hidden_dim, num_heads) for _ in range(num_layers)])
            self.norm = nn.RMSNorm(hidden_dim, dtype=dtype)
            self.head = nn.Linear(hidden_dim, vocab_size, dtype=dtype)

        def forward(self, ids):
            x = self.embed(ids)
            s = x.shape[1]
            mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
            for layer in self.layers:
                x = layer(x, mask=mask)
            return self.head(self.norm(x))

    return Net()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stable-va", action="store_true",
                   help="Reuse a single fixed (num_seqs, seq_len) shape every step so "
                        "AllGather output VAs do not churn (control: external free stays flat).")
    p.add_argument("--wrap", choices=["layer", "top"], default="layer",
                   help="FSDP2 wrapping. 'layer' = per-decoder-layer units (many distinct AllGather "
                        "buffers -> maximal VA churn, the config that exhibits the 32B/EP IPC-handle "
                        "growth). 'top' = single top-level unit (one fixed-size param AllGather; "
                        "production XPU default, but one VA -> unlikely to churn). Default: layer.")
    args = p.parse_args()

    torch.distributed.init_process_group(backend="xccl")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    from torch.distributed._composable.fsdp import fully_shard

    model = create_model(args.layers, args.hidden, args.heads).to(device)
    # Wrapping choice matters for THIS bug: the CCL IPC-handle cache is keyed by the
    # virtual address of each collective's buffer. Per-layer units produce many
    # distinct AllGather buffers (the churn that fills the handle cache — matching the
    # 32B FSDP2 / EP production configs where this was first seen). A single top-level
    # unit gives one fixed-size param AllGather VA, which does not churn.
    if args.wrap == "layer":
        for layer in model.layers:
            fully_shard(layer)
    fully_shard(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    thr = os.environ.get("CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD", "(unset=CCL default ~1000)")
    if rank == 0:
        nparam = sum(p.numel() for p in model.parameters())
        print("=== oneCCL IPC-handle cache reproducer (pure FSDP2 + XCCL) ===")
        print(f"world={world}  layers={args.layers} hidden={args.hidden} heads={args.heads}")
        print(f"params={nparam:,} ({nparam*2/1e9:.1f} GiB bf16, sharded /{world})")
        print(f"CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD={thr}")
        print(f"fsdp_wrap={args.wrap}  stable_va={args.stable_va}  (churn VAs unless set)")
        print(f"PyTorch={torch.__version__}")
        print("NO empty_cache() is called anywhere in this loop.\n")

    def free_gib():
        free, total = torch.xpu.mem_get_info(device)
        return free / 1e9, total / 1e9

    torch.distributed.barrier()
    base_free = None
    prev_free = None
    t0 = time.time()

    for step in range(args.steps):
        if args.stable_va:
            ns, sl = args.num_seqs, args.seq_len
        else:
            # Vary the shape each step so FSDP2 AllGather output buffers land on
            # fresh virtual addresses -> new CCL IPC handles cached each step.
            ns = args.num_seqs
            sl = args.seq_len + (step % 8) * 64

        ids = torch.randint(0, 32000, (ns, sl), device=device)
        labels = torch.randint(0, 32000, (ns, sl), device=device)

        logits = model(ids)                                   # FSDP2 AllGather (fwd)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()                                       # FSDP2 ReduceScatter (bwd)
        opt.step()
        opt.zero_grad(set_to_none=True)

        torch.xpu.synchronize()
        torch.distributed.barrier()

        if rank == 0:
            f, tot = free_gib()
            resv = torch.xpu.memory_reserved(device) / 1e9
            alloc = torch.xpu.memory_allocated(device) / 1e9
            if base_free is None:
                base_free, prev_free = f, f
            drop_step = prev_free - f
            drop_total = base_free - f
            prev_free = f
            print(f"  step {step:3d} | ext_free={f:6.2f}/{tot:.0f} GiB "
                  f"| drop/step={drop_step:+.3f} tot_drop={drop_total:+.3f} "
                  f"| torch alloc/resv={alloc:.2f}/{resv:.2f} GiB "
                  f"| shape=({ns},{sl})", flush=True)

    if rank == 0:
        dt = time.time() - t0
        f, tot = free_gib()
        print(f"\nDONE {args.steps} steps in {dt:.1f}s. "
              f"Total device-global free drop = {base_free - f:+.3f} GiB "
              f"(torch reserved is flat -> the drop is CCL-internal IPC-handle metadata).")
        print("If drop/step is ~constant & positive with churn but ~0 with --stable-va, "
              "the CCL IPC-handle cache is the leak.")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

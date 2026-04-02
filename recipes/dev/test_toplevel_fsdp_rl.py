#!/usr/bin/env python3
"""Test: full RL pattern with TOP-LEVEL ONLY fully_shard (matching repro script)."""
import os, gc, time, torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F

def create_model(num_layers=12, hidden=1024, num_heads=8, dtype=torch.bfloat16):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.RMSNorm(hidden, dtype=dtype)
            self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dtype=dtype)
            self.ln2 = nn.RMSNorm(hidden, dtype=dtype)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, hidden * 4, dtype=dtype), nn.SiLU(),
                nn.Linear(hidden * 4, hidden, dtype=dtype))
        def forward(self, x):
            h = self.ln(x); h, _ = self.attn(h, h, h, need_weights=False); x = x + h
            x = x + self.ffn(self.ln2(x)); return x
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32000, hidden, dtype=dtype)
            self.layers = nn.ModuleList([Block() for _ in range(num_layers)])
            self.norm = nn.RMSNorm(hidden, dtype=dtype)
            self.head = nn.Linear(hidden, 32000, dtype=dtype)
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers: x = layer(x)
            return self.head(self.norm(x))
    return Model()

dist.init_process_group(backend="xccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")

from torch.distributed._composable.fsdp import fully_shard

# TOP-LEVEL ONLY
policy = create_model().to(device); fully_shard(policy)
ref = create_model().to(device); ref.eval()
for p in ref.parameters(): p.requires_grad = False
fully_shard(ref)

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
input_ids = torch.randint(0, 32000, (4, 512), device=device)
policy.train()

if rank == 0:
    mem = torch.xpu.memory_allocated(device)/1e9
    mem_res = torch.xpu.memory_reserved(device)/1e9
    print(f"=== Full RL, TOP-LEVEL ONLY fully_shard ===")
    print(f"Initial mem: {mem:.2f}/{mem_res:.2f} GiB")

t0 = time.time()
for i in range(200):
    with torch.no_grad():
        gl = policy(input_ids); gt = torch.argmax(gl, dim=-1); del gl
    with torch.no_grad():
        cl = policy(input_ids); clp = F.log_softmax(cl, dim=-1)
        plp = torch.gather(clp, -1, gt.unsqueeze(-1)).squeeze(-1); del cl, clp
    with torch.no_grad():
        rl = ref(input_ids); rlp = F.log_softmax(rl, dim=-1)
        reflp = torch.gather(rlp, -1, gt.unsqueeze(-1)).squeeze(-1); del rl, rlp
    kl = plp - reflp; rw = torch.randn(4, device=device)
    adv = (rw - rw.mean()) / (rw.std() + 1e-8)
    lo = policy(input_ids); lp = F.log_softmax(lo, dim=-1)
    clp2 = torch.gather(lp, -1, gt.unsqueeze(-1)).squeeze(-1); del lo, lp
    ratio = torch.exp(clp2 - plp.detach())
    loss = -(adv.unsqueeze(-1) * ratio).mean() + kl.mean() * 0.01
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    del gt, plp, reflp, kl, clp2, loss

    if rank == 0 and (i+1) % 5 == 0:
        elapsed = time.time() - t0
        mem = torch.xpu.memory_allocated(device)/1e9
        mem_res = torch.xpu.memory_reserved(device)/1e9
        print(f"  iter {i+1:4d}/200 | {elapsed:.1f}s | mem={mem:.2f}/{mem_res:.2f} GiB")

if rank == 0: print(f"PASSED: 200 iterations")
dist.destroy_process_group()

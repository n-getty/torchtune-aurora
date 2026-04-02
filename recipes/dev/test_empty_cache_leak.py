#!/usr/bin/env python3
"""Test: does torch.xpu.empty_cache() CAUSE the UR handle leak?"""
import os, time, torch, torch.distributed as dist, torch.nn.functional as F
import sys; sys.path.insert(0, "recipes/dev")
from repro_xpu_resource_leak import create_model, setup_fsdp

dist.init_process_group(backend="xccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")

policy = create_model(12, 1024, 8).to(device)
policy = setup_fsdp(policy, version=2)
ref = create_model(12, 1024, 8).to(device)
ref.eval()
for p in ref.parameters(): p.requires_grad = False
ref = setup_fsdp(ref, version=2)

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
input_ids = torch.randint(0, 32000, (4, 512), device=device)
policy.train()

if rank == 0: print(f"=== RL loop WITH empty_cache() calls ===")

t0 = time.time()
for i in range(200):
    with torch.no_grad():
        gl = policy(input_ids); gt = torch.argmax(gl, dim=-1); del gl
    torch.xpu.empty_cache()  # <-- THE SUSPECT

    with torch.no_grad():
        cl = policy(input_ids); clp = F.log_softmax(cl, dim=-1)
        plp = torch.gather(clp, -1, gt.unsqueeze(-1)).squeeze(-1); del cl, clp
    torch.xpu.empty_cache()  # <-- THE SUSPECT

    with torch.no_grad():
        rl = ref(input_ids); rlp = F.log_softmax(rl, dim=-1)
        reflp = torch.gather(rlp, -1, gt.unsqueeze(-1)).squeeze(-1); del rl, rlp
    torch.xpu.empty_cache()  # <-- THE SUSPECT

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

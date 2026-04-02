#!/usr/bin/env python3
"""Control: empty_cache() WITHOUT FSDP — should be stable."""
import os, time, torch, torch.nn.functional as F
import sys; sys.path.insert(0, "recipes/dev")
from repro_xpu_resource_leak import create_model

torch.xpu.set_device(0)
device = torch.device("xpu:0")

policy = create_model(12, 1024, 8).to(device)
input_ids = torch.randint(0, 32000, (4, 512), device=device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
policy.train()

print(f"=== RL loop + empty_cache(), NO FSDP ===")
t0 = time.time()
for i in range(200):
    with torch.no_grad():
        gl = policy(input_ids); gt = torch.argmax(gl, dim=-1); del gl
    torch.xpu.empty_cache()
    with torch.no_grad():
        cl = policy(input_ids); clp = F.log_softmax(cl, dim=-1)
        plp = torch.gather(clp, -1, gt.unsqueeze(-1)).squeeze(-1); del cl, clp
    torch.xpu.empty_cache()
    logits = policy(input_ids); lp = F.log_softmax(logits, dim=-1)
    clp2 = torch.gather(lp, -1, gt.unsqueeze(-1)).squeeze(-1); del logits, lp
    ratio = torch.exp(clp2 - plp.detach())
    loss = -ratio.mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    del gt, plp, clp2, loss
    if (i+1) % 10 == 0:
        elapsed = time.time() - t0
        mem = torch.xpu.memory_allocated(device)/1e9
        mem_res = torch.xpu.memory_reserved(device)/1e9
        print(f"  iter {i+1:4d}/200 | {elapsed:.1f}s | mem={mem:.2f}/{mem_res:.2f} GiB")
print(f"PASSED: 200 iterations")

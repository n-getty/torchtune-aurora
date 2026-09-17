"""Repro variant: wrap GroupedExpertsHF with the SAME trivial 1-rank solo
FSDP2 fully_shard() the real recipe uses (reshard_after_forward=False,
matching expert_cpu_offload=False), to test whether FSDP2's own per-call
unshard allocation (not the padded-BMM computation itself) is the actual
L0 handle-table exhaustion trigger.
"""
import sys
import torch
import torch.distributed as dist
import time
import os

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune")
from torchtune.models.qwen3_moe._experts import GroupedExpertsHF
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29511")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

dist.init_process_group(backend="gloo")

device = "xpu"
E = 16
dim = 2048
hidden_dim = 768
NUM_LAYERS = 48
TOTAL_TOKENS = 3072

torch.manual_seed(0)

solo_pg = dist.new_group([0])
solo_mesh = DeviceMesh.from_group(solo_pg, device)

experts_per_layer = []
for _ in range(NUM_LAYERS):
    m = GroupedExpertsHF(dim=dim, hidden_dim=hidden_dim, num_experts=E)
    m = m.to(device=device, dtype=torch.bfloat16)
    m.reset_parameters()
    fully_shard(m, mesh=solo_mesh, reshard_after_forward=False)
    experts_per_layer.append(m)

def make_counts(total, e, gen):
    cuts = torch.sort(torch.randint(0, total, (e - 1,), generator=gen)).values
    cuts = torch.cat([torch.tensor([0]), cuts, torch.tensor([total])])
    counts = (cuts[1:] - cuts[:-1]).to(torch.float32)
    return counts

gen = torch.Generator().manual_seed(0)

print(f"Starting {NUM_LAYERS}-layer forward loop, FSDP2-wrapped (solo 1-rank, reshard_after_forward=False)", flush=True)
t0 = time.time()
for step in range(5):
    for i, m in enumerate(experts_per_layer):
        counts = make_counts(TOTAL_TOKENS, E, gen).to(device)
        x = torch.randn(TOTAL_TOKENS, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        out = m(x, counts)
        loss = out.sum()
        loss.backward()
        if i == 0:
            print(f"step {step} layer {i} ok, mem_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB reserved={torch.xpu.memory_reserved()/1e9:.2f}GB", flush=True)
    print(f"=== step {step} complete, t={time.time()-t0:.1f}s ===", flush=True)
print("ALL CLEAN")

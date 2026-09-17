"""Repro variant: simulate the L0 handle-table pressure from the REST of the
FSDP2-sharded 30B model (params/optimizer state/other-layer activations) by
pre-allocating many small-to-medium tensors before running the MoE loop,
to test whether proximity to a handle-table/memory ceiling (not the MoE
compute itself) is what triggers the crash.
"""
import sys
import torch
import time

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune")
from torchtune.models.qwen3_moe._experts import GroupedExpertsHF

device = "xpu"
E = 16
dim = 2048
hidden_dim = 768
NUM_LAYERS = 48
TOTAL_TOKENS = 3072

torch.manual_seed(0)

# Simulate ~45 GiB of "other model state" resident on this tile (leaving only
# ~19 GiB of the 64 GiB tile for MoE activations, mimicking real FSDP2 +
# optimizer-state occupancy) via many DISTINCT allocations (to stress the L0
# handle table specifically, not just raw bytes).
pressure_tensors = []
target_gb = 45
per_tensor_mb = 50
n_tensors = int(target_gb * 1024 / per_tensor_mb)
elems = int(per_tensor_mb * 1024 * 1024 / 2)  # bf16 = 2 bytes
print(f"Allocating {n_tensors} distinct {per_tensor_mb}MB tensors (~{target_gb}GB, simulating FSDP2/optim pressure)...", flush=True)
for i in range(n_tensors):
    pressure_tensors.append(torch.empty(elems, dtype=torch.bfloat16, device=device))
print(f"Pressure alloc done. mem_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB reserved={torch.xpu.memory_reserved()/1e9:.2f}GB", flush=True)

experts_per_layer = []
for _ in range(NUM_LAYERS):
    m = GroupedExpertsHF(dim=dim, hidden_dim=hidden_dim, num_experts=E).to(device=device, dtype=torch.bfloat16)
    m.reset_parameters()
    experts_per_layer.append(m)

def make_counts(total, e, gen):
    cuts = torch.sort(torch.randint(0, total, (e - 1,), generator=gen)).values
    cuts = torch.cat([torch.tensor([0]), cuts, torch.tensor([total])])
    counts = (cuts[1:] - cuts[:-1]).to(torch.float32)
    return counts

gen = torch.Generator().manual_seed(0)

print(f"Starting {NUM_LAYERS}-layer forward loop under memory pressure", flush=True)
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

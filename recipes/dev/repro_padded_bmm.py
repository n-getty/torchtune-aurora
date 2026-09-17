"""Minimal repro of the padded-BMM UR:40 crash isolated to GroupedExpertsHF.forward().

Real Qwen3-30B-A3B shapes: 128 experts / EP=8 -> E=16 local experts,
dim=2048, hidden_dim(moe_intermediate)=768, experts_per_token=8, 48 layers.
At batch_size=2, seq_len=1536: total tokens routed to top_k=8 experts,
locally on this rank after EP dispatch ~= (bs*slen*top_k)/ep_degree (roughly,
modulo real routing imbalance) = (2*1536*8)/8 = 3072 tokens/rank on average.

This script calls forward() in a loop across "layers" (48) to mimic the
real per-step call count, without any FSDP/EP/distributed/dataset overhead,
to see if the L0 handle exhaustion reproduces on a single XPU device.
"""
import sys
import torch
import time

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune")
from torchtune.models.qwen3_moe._experts import GroupedExpertsHF

device = "xpu"
E = 16            # local experts per rank at EP=8
dim = 2048
hidden_dim = 768
NUM_LAYERS = 48
TOTAL_TOKENS = 3072   # rough average tokens/rank at bs=2/seq1536/topk=8/EP=8

torch.manual_seed(0)

experts_per_layer = []
for _ in range(NUM_LAYERS):
    m = GroupedExpertsHF(dim=dim, hidden_dim=hidden_dim, num_experts=E).to(device=device, dtype=torch.bfloat16)
    m.reset_parameters()
    experts_per_layer.append(m)

# Simulate uneven-but-plausible per-expert token counts (real routing isn't uniform)
def make_counts(total, e, gen):
    # random split
    cuts = torch.sort(torch.randint(0, total, (e - 1,), generator=gen)).values
    cuts = torch.cat([torch.tensor([0]), cuts, torch.tensor([total])])
    counts = (cuts[1:] - cuts[:-1]).to(torch.float32)
    return counts

gen = torch.Generator().manual_seed(0)

print(f"Starting {NUM_LAYERS}-layer forward loop, E={E}, dim={dim}, hidden={hidden_dim}, total={TOTAL_TOKENS}", flush=True)
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

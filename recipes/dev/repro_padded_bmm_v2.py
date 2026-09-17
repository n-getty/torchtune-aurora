"""Corrected repro: matches the REAL recipe's forward/backward pattern.

Root cause found via HW debug instrumentation (2026-07-25): the real per-step
forward pass runs ALL 48 MoE layers' padded-BMM forward BEFORE any backward()
is called (single .backward() at the end of the full model forward) --
MoE layers are specifically EXCLUDED from activation checkpointing (v158
correctness fix in torchtune/dev/rl/distributed.py::_apply_split_ac), so
every layer's [E, max_count, dim] padded activations + 3 bmm outputs stay
resident simultaneously across all 48 layers until the single backward call.

The earlier (WRONG) repro called .backward() after EACH layer, which frees
that layer's activation graph immediately -- never letting memory accumulate
across layers, hence it ran "clean" and failed to reproduce the crash.

Real observed shapes (captured via TORCHTUNE_MOE_BMM_DEBUG=1 HW run):
  total tokens/call: 19456-37071 (mean ~27479) -- NOT ~3072 as originally
    estimated (that was a wrong derivation of post-EP-dispatch average).
  max_count/call: 3314-17632 (mean ~8041) -- severe per-expert imbalance,
    padding overhead ratio (max_count*E/total) averages ~4.6x.
  mem_reserved climbs monotonically: 11.92GB (layer ~1) -> 66.79GB (layer 34),
    crashing between layer 34 and 48 on this 64GB tile.
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

torch.manual_seed(0)

experts_per_layer = []
for _ in range(NUM_LAYERS):
    m = GroupedExpertsHF(dim=dim, hidden_dim=hidden_dim, num_experts=E).to(device=device, dtype=torch.bfloat16)
    m.reset_parameters()
    experts_per_layer.append(m)

# Real-observed-shape generator: mimic actual imbalance (a couple of "hot"
# experts get outsized share, rest get remainder) rather than a smooth random
# cut -- matches the observed pattern (one expert often >30-50% of total).
def make_realistic_counts(total, e, gen):
    hot_share = torch.rand(1, generator=gen).item() * 0.4 + 0.15  # 15-55% to one hot expert
    hot_idx = torch.randint(0, e, (1,), generator=gen).item()
    remaining = total - int(total * hot_share)
    rest_cuts = torch.sort(torch.randint(0, remaining, (e - 2,), generator=gen)).values
    rest_cuts = torch.cat([torch.tensor([0]), rest_cuts, torch.tensor([remaining])])
    rest_counts = (rest_cuts[1:] - rest_cuts[:-1]).tolist()
    counts = rest_counts[:hot_idx] + [total - remaining] + rest_counts[hot_idx:]
    return torch.tensor(counts[:e], dtype=torch.float32)

gen = torch.Generator().manual_seed(0)

print(f"Starting SINGLE forward pass over {NUM_LAYERS} layers (no backward until the end, matching real recipe)", flush=True)
t0 = time.time()
outputs = []
for i, m in enumerate(experts_per_layer):
    total_i = int(torch.randint(19000, 37500, (1,), generator=gen).item())
    counts = make_realistic_counts(total_i, E, gen).to(device)
    x = torch.randn(total_i, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    out = m(x, counts)
    outputs.append(out)
    print(f"layer {i}: total={total_i} max_count={int(counts.max().item())} "
          f"mem_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB "
          f"mem_reserved={torch.xpu.memory_reserved()/1e9:.2f}GB", flush=True)

print(f"Forward done at t={time.time()-t0:.1f}s, running single backward over all {NUM_LAYERS} outputs...", flush=True)
loss = sum(o.sum() for o in outputs)
loss.backward()
print(f"Backward done, mem_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB mem_reserved={torch.xpu.memory_reserved()/1e9:.2f}GB", flush=True)
print("ALL CLEAN")

"""CORRECTED fix-concept validation.

v3 (repro_padded_bmm_v3_chunked.py) chunked along the LAYER dimension --
backward every N of 48 layers. That is NOT a valid mechanism for a real
sequential transformer (layer N's input depends on layer N-1's output; the
loss depends on ALL 48 layers' outputs, so you cannot do a correct partial
backward through only the first N layers while independently continuing
forward through the rest). v3's synthetic "layers" were independent
random-data calls with no real forward dependency, so layer-chunking
"worked" there but does NOT model what a real fix would do.

GRPO's real `forward_batch_size` chunking works along the BATCH/SEQUENCE
dimension: run the FULL model (all layers) on a SMALLER slice of the
microbatch, backward + accumulate, then move to the next slice. This is
mathematically equivalent to one full-batch backward (loss is additive
across sequences) and, critically, proportionally shrinks EVERY layer's
per-call token count and thus the padded-BMM activation footprint
SIMULTANEOUSLY across all 48 layers -- which is the actual mechanism that
would fix this crash.

This script simulates that correctly: for K token-sub-chunks, run ALL 48
layers on that sub-chunk's (smaller) token count, backward, accumulate
grad, then move to the next sub-chunk. Real per-layer token counts are
divided by K (approximating what a real batch/microbatch chunk would see),
using the SAME real-observed hot-expert-imbalance shape generator as v2.
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
NUM_CHUNKS = 4   # e.g. batch_size=2 split into 4 sub-chunks (grad accumulation)

torch.manual_seed(0)

experts_per_layer = []
for _ in range(NUM_LAYERS):
    m = GroupedExpertsHF(dim=dim, hidden_dim=hidden_dim, num_experts=E).to(device=device, dtype=torch.bfloat16)
    m.reset_parameters()
    experts_per_layer.append(m)

def make_realistic_counts(total, e, gen):
    hot_share = torch.rand(1, generator=gen).item() * 0.4 + 0.15
    hot_idx = torch.randint(0, e, (1,), generator=gen).item()
    remaining = total - int(total * hot_share)
    rest_cuts = torch.sort(torch.randint(0, max(remaining, 1), (e - 2,), generator=gen)).values
    rest_cuts = torch.cat([torch.tensor([0]), rest_cuts, torch.tensor([remaining])])
    rest_counts = (rest_cuts[1:] - rest_cuts[:-1]).tolist()
    counts = rest_counts[:hot_idx] + [total - remaining] + rest_counts[hot_idx:]
    return torch.tensor(counts[:e], dtype=torch.float32)

gen = torch.Generator().manual_seed(0)

print(f"Starting BATCH/TOKEN-chunked forward+backward: {NUM_CHUNKS} sub-chunks, "
      f"EACH running the FULL {NUM_LAYERS}-layer model on 1/{NUM_CHUNKS} of the "
      f"real per-layer token count, backward+accumulate per sub-chunk "
      f"(mirrors GRPO's forward_batch_size mechanism correctly)", flush=True)
t0 = time.time()
peak_reserved = 0.0

# Real observed full-batch per-layer totals (19K-37K range); each sub-chunk
# gets roughly 1/NUM_CHUNKS of that.
full_totals = [int(torch.randint(19000, 37500, (1,), generator=gen).item()) for _ in range(NUM_LAYERS)]

for chunk_idx in range(NUM_CHUNKS):
    outputs = []
    for layer_idx, m in enumerate(experts_per_layer):
        total_i = max(full_totals[layer_idx] // NUM_CHUNKS, E)  # this sub-chunk's token count for this layer
        counts = make_realistic_counts(total_i, E, gen).to(device)
        x = torch.randn(total_i, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        out = m(x, counts)
        outputs.append(out)
    loss = sum(o.sum() for o in outputs) / NUM_CHUNKS  # mean-style accumulation
    loss.backward()
    reserved = torch.xpu.memory_reserved() / 1e9
    peak_reserved = max(peak_reserved, reserved)
    print(f"sub-chunk {chunk_idx+1}/{NUM_CHUNKS} done, "
          f"mem_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB mem_reserved={reserved:.2f}GB "
          f"t={time.time()-t0:.1f}s", flush=True)
    del outputs, loss

print(f"ALL CLEAN. peak_reserved={peak_reserved:.2f}GB (vs layer-unchunked crash at ~68GB)", flush=True)

"""Fix-concept validation: chunk the 48-layer forward into groups of N with
a .backward() after each chunk, mirroring GRPO's forward_batch_size
chunking mechanism. If this stays under the memory ceiling where the
un-chunked v2 repro crashed (at layer 29, ~66GB), it validates chunking
as the fix direction for the padded-BMM SFT crash BEFORE committing to a
full recipe-level implementation.
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
CHUNK_SIZE = 16   # layers per backward -- analogous to forward_batch_size chunking

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
    rest_cuts = torch.sort(torch.randint(0, remaining, (e - 2,), generator=gen)).values
    rest_cuts = torch.cat([torch.tensor([0]), rest_cuts, torch.tensor([remaining])])
    rest_counts = (rest_cuts[1:] - rest_cuts[:-1]).tolist()
    counts = rest_counts[:hot_idx] + [total - remaining] + rest_counts[hot_idx:]
    return torch.tensor(counts[:e], dtype=torch.float32)

gen = torch.Generator().manual_seed(0)

print(f"Starting CHUNKED forward+backward over {NUM_LAYERS} layers, chunk_size={CHUNK_SIZE} "
      f"(backward every {CHUNK_SIZE} layers, mimicking forward_batch_size chunking)", flush=True)
t0 = time.time()
peak_reserved = 0.0
for chunk_start in range(0, NUM_LAYERS, CHUNK_SIZE):
    chunk_layers = experts_per_layer[chunk_start:chunk_start + CHUNK_SIZE]
    outputs = []
    for j, m in enumerate(chunk_layers):
        i = chunk_start + j
        total_i = int(torch.randint(19000, 37500, (1,), generator=gen).item())
        counts = make_realistic_counts(total_i, E, gen).to(device)
        x = torch.randn(total_i, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        out = m(x, counts)
        outputs.append(out)
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    reserved = torch.xpu.memory_reserved() / 1e9
    peak_reserved = max(peak_reserved, reserved)
    print(f"chunk [{chunk_start}:{chunk_start+len(chunk_layers)}] done, "
          f"mem_alloc={torch.xpu.memory_allocated()/1e9:.2f}GB mem_reserved={reserved:.2f}GB "
          f"t={time.time()-t0:.1f}s", flush=True)
    del outputs, loss

print(f"ALL CLEAN. peak_reserved={peak_reserved:.2f}GB (vs un-chunked crash at ~68GB / tile ceiling 64GB)", flush=True)

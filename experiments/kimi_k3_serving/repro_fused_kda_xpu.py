#!/usr/bin/env python3
"""Minimal XPU repro for the fused KDA decode kernel's state corruption.

The kernel measures +9.8% but produces wrong text on real weights: it answers
the first question correctly, then degenerates (391 / 391 / 391 instead of
391 / 414 / 437). That is a recurrent state error that compounds.

16 CPU equivalence tests pass under TRITON_INTERPRET=1, including a two-step
carry test written for exactly this failure -- so the defect is in the XPU
CODEGEN, not the algebra. Interpret mode runs the kernel body as Python ops
and never exercises compiled SPIR-V, memory coalescing, static_range
unrolling, or in-place aliasing under real parallelism.

This script runs the SAME comparison on a real XPU device, so the bug can be
localised without a 3-node hold and a 12-minute model load.

Run on one node with one tile:
    ZE_AFFINITY_MASK=0 python repro_fused_kda_xpu.py
    ZE_AFFINITY_MASK=0 python repro_fused_kda_xpu.py --steps 8 --heads 3

Prints per-step max abs error for the output AND the recurrent state. The
signature to look for: step 0 clean, error growing with step index.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/flare/ModCon/ngetty/vllm-xpu-src")

CONV_W = 4
EPS = 1e-5


def eager_step(q_proj, k_proj, v_proj, g, beta, conv_w3, conv_b3, conv_s3,
               state, idx):
    """Reference: the production eager path, transcribed from kda.py."""
    conv_out = []
    for x, (w, b, cs) in zip((q_proj, k_proj, v_proj),
                             zip(conv_w3, conv_b3, conv_s3)):
        xx = x.unsqueeze(-1)
        state_len = cs.shape[-1]
        st = cs[idx]
        history = torch.cat((st, xx[:, :, 0, None]), dim=-1)
        value = (history.float() * w.float()).sum(-1) + b.float()
        cs[idx] = history[:, :, -state_len:]
        conv_out.append(F.silu(value).to(x.dtype))
    q, k, v = conv_out

    n, h, d = g.shape
    q = q.view(n, h, d).float()
    k = k.view(n, h, d).float()
    v = v.view(n, h, d).float()
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)

    rs = state[idx].float() * torch.exp(g.float())[:, :, None, :]
    delta = v - torch.einsum("nhvk,nhk->nhv", rs, k)
    rs = rs + beta.float()[:, :, None, None] * (delta[:, :, :, None]
                                                * k[:, :, None, :])
    out = torch.einsum("nhvk,nhk->nhv", rs, q * (d ** -0.5))
    state[idx] = rs.to(state.dtype)
    return out


def make(n_seq, heads, dim, blocks, dev, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype).to(dev)  # noqa: E731
    return dict(
        q_proj=r(n_seq, heads * dim), k_proj=r(n_seq, heads * dim),
        v_proj=r(n_seq, heads * dim),
        # decay gate is a log value, always <= 0 (K3 clamps to [-5, 0])
        g=(-torch.rand(n_seq, heads, dim, generator=g, dtype=dtype) * 5).to(dev),
        beta=torch.rand(n_seq, heads, generator=g, dtype=dtype).to(dev),
        conv_w3=tuple(r(heads * dim, CONV_W) for _ in range(3)),
        conv_b3=tuple(r(heads * dim) for _ in range(3)),
        conv_s3=tuple(r(blocks, heads * dim, CONV_W - 1) for _ in range(3)),
        state=r(blocks, heads, dim, dim),
        idx=torch.arange(n_seq, dtype=torch.int32, device=dev),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--heads", type=int, default=3, help="local heads (TP=32 -> 3)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("FAIL: no XPU visible. Run on a compute node with ZE_AFFINITY_MASK set.")
        return 2
    dev = "xpu:0"
    dtype = getattr(torch, args.dtype)
    print(f"device={torch.xpu.get_device_name(0)} dtype={args.dtype} "
          f"seqs={args.seqs} heads={args.heads} dim={args.dim}")

    from vllm.model_executor.layers.kda_fused_decode_xpu import fused_kda_decode

    base = make(args.seqs, args.heads, args.dim, args.blocks, dev, dtype, 0)

    def clone(t):
        return {k: (v.clone() if torch.is_tensor(v)
                    else tuple(x.clone() for x in v)) for k, v in t.items()}

    ref, fus = clone(base), clone(base)
    worst = 0.0
    print(f"\n{'step':>4} {'out_maxerr':>12} {'state_maxerr':>14}  verdict")
    for step in range(args.steps):
        nxt = make(args.seqs, args.heads, args.dim, args.blocks, dev, dtype,
                   100 + step)
        for key in ("q_proj", "k_proj", "v_proj", "g", "beta"):
            ref[key] = nxt[key].clone()
            fus[key] = nxt[key].clone()

        r_out = eager_step(ref["q_proj"], ref["k_proj"], ref["v_proj"],
                           ref["g"], ref["beta"], ref["conv_w3"],
                           ref["conv_b3"], ref["conv_s3"], ref["state"],
                           ref["idx"])
        f_out = fused_kda_decode(
            fus["q_proj"], fus["k_proj"], fus["v_proj"], fus["g"], fus["beta"],
            fus["g"],  # g2 unused when apply_gated_norm=False
            fus["conv_w3"], fus["conv_b3"], fus["conv_s3"], fus["state"],
            fus["idx"], None, EPS, apply_gated_norm=False,
        )
        torch.xpu.synchronize()

        oe = (f_out.float() - r_out.float()).abs().max().item()
        se = (fus["state"].float() - ref["state"].float()).abs().max().item()
        worst = max(worst, oe, se)
        flag = "ok" if max(oe, se) < 1e-3 else "MISMATCH"
        print(f"{step:>4} {oe:12.3e} {se:14.3e}  {flag}")

    print()
    if worst < 1e-3:
        print(f"PASS: max error {worst:.3e} across {args.steps} steps.")
        print("  The kernel is correct at this shape. Widen the search:")
        print("  try --dtype bfloat16, --seqs 4, or non-contiguous idx.")
        return 0
    print(f"REPRO CONFIRMED: max error {worst:.3e}")
    print("  If step 0 is clean and error grows, it is the state writeback")
    print("  (in-place tl.store of the [D,D] tile racing its own load).")
    print("  If step 0 already differs, it is the conv window or the reduction.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe: does the XPU fused-MoE kernel give the same result at M=1 as at M=N?

WHY THIS EXISTS
---------------
Kimi-K3 serves a correct FIRST token (prefill, num_tokens=5) then degenerates
from token 2 (decode, num_tokens=1), while a Kimi-Linear-48B surrogate on the
same shared XPU decode path stays clean. So the defect is K3-only or
decode-shape specific.

A static pass (2026-08-06) cleared the router, the attention-residual path, and
SiTU against the HF reference at both num_tokens=5 and 1, and refuted both the
"frozen layer-0 input / slot_mapping" lead and a proposed latent-MoE double
all-reduce. What survives is the fused MoE GEMM itself:

    torch.ops._xpu_C.cutlass_grouped_gemm_interface   (libgrouped_gemm_xe_2.so)

That is a compiled binary, so no amount of source reading can settle it.
Meanwhile XPUExperts.workspace_shapes (xpu_moe.py:152-166) applies NO M-rounding
of any kind and no assertion -- whatever M the caller passes goes straight to
the kernel. If the kernel has an M-tile assumption (16 or 32 is typical for a
PVC GEMM) that misbehaves at M=1, it would produce finite-but-wrong expert
output, i.e. smooth-but-wrong logits -- exactly the observed signature.

This probe needs NO model, NO checkpoint and NO server: one XPU tile and
synthetic weights. It is far cheaper than the reduced-K3 harness and tests the
leading suspect directly, so run it first.

WHAT IT ASSERTS
---------------
Feed the SAME per-token inputs through the kernel two ways:
  (a) one batched call with M=N
  (b) N separate calls with M=1
Row i of (a) must equal the single row of call i in (b), to reduction-order
noise. The MoE is per-token independent -- each token is routed and combined on
its own -- so any disagreement is a batch-size-dependent kernel bug.

INTERPRETING THE RESULT
-----------------------
  FAIL  -> the bug is found, in minutes, with no capacity load. The M=1 row is
           wrong; report the tile geometry and file against the kernel.
  PASS  -> the fused MoE is M-shape-clean. That does NOT clear K3: the defect
           may still be K3-only elsewhere, or in the TP=32/EP-on topology,
           neither of which this single-tile probe exercises. Proceed to the
           reduced-K3 harness (TP=8, one node, free debug queue).

A PASS is a null result, not an all-clear. Do not report it as "MoE is fine".

USAGE
-----
    python3 probe_m1_equivalence.py                 # bf16 (unquantized) path
    python3 probe_m1_equivalence.py --mxfp4         # MXFP4 weight-only path
    python3 probe_m1_equivalence.py --topk 16 --experts 8 --hidden 512

Must run on a compute node with an XPU tile; it exits non-zero on divergence so
it can gate a launcher.
"""

import argparse
import sys

import torch


def build_mxfp4_weights(num_experts, inter_size, hidden_size, device, generator):
    """Synthetic but VALID MXFP4 weights.

    weight_packed is uint8 where every byte is a pair of FP4 nibbles (any byte
    is valid). weight_scale is E8M0 uint8; 127 == 2^0, and staying inside
    [120, 135) keeps scales near 1.0 so the reference and kernel results are
    numerically comparable rather than dominated by extreme exponents.
    Shapes mirror mxfp4_storage_shapes().
    """
    w13 = torch.randint(
        0, 256, (num_experts, 2 * inter_size, hidden_size // 2),
        dtype=torch.uint8, device=device, generator=generator,
    )
    w2 = torch.randint(
        0, 256, (num_experts, hidden_size, inter_size // 2),
        dtype=torch.uint8, device=device, generator=generator,
    )
    w13_s = torch.randint(
        120, 135, (num_experts, 2 * inter_size, hidden_size // 32),
        dtype=torch.uint8, device=device, generator=generator,
    )
    w2_s = torch.randint(
        120, 135, (num_experts, hidden_size, inter_size // 32),
        dtype=torch.uint8, device=device, generator=generator,
    )
    return w13, w13_s, w2, w2_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8, help="N for the batched call")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--inter", type=int, default=256)
    parser.add_argument("--mxfp4", action="store_true", help="use MXFP4 weights")
    parser.add_argument("--situ", action="store_true", help="use K3's SiTU activation")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("FATAL: no XPU available. Run this on a compute node.", file=sys.stderr)
        return 2

    try:
        from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe
    except ImportError as error:
        print(f"FATAL: cannot import xpu_fused_moe: {error}", file=sys.stderr)
        print("Activate the venv that has vllm_xpu_kernels installed.", file=sys.stderr)
        return 2

    device = torch.device("xpu")
    # topk must not exceed the number of experts we actually built.
    topk = min(args.topk, args.experts)
    if topk != args.topk:
        print(f"note: clamping topk {args.topk} -> {topk} (only {args.experts} experts)")

    generator = torch.Generator(device=device).manual_seed(0)
    hidden = torch.randn(
        args.tokens, args.hidden, dtype=torch.bfloat16,
        device=device, generator=generator,
    )
    weights = torch.rand(
        args.tokens, topk, dtype=torch.float32, device=device, generator=generator
    )
    weights = weights / weights.sum(dim=-1, keepdim=True)
    ids = torch.stack([
        torch.randperm(args.experts, device=device, generator=generator)[:topk]
        for _ in range(args.tokens)
    ]).to(torch.int64)

    if args.mxfp4:
        w13, w13_s, w2, w2_s = build_mxfp4_weights(
            args.experts, args.inter, args.hidden, device, generator
        )
    else:
        w13 = torch.randn(
            args.experts, 2 * args.inter, args.hidden, dtype=torch.bfloat16,
            device=device, generator=generator,
        ) * 0.05
        w2 = torch.randn(
            args.experts, args.hidden, args.inter, dtype=torch.bfloat16,
            device=device, generator=generator,
        ) * 0.05
        w13_s = w2_s = None

    common = dict(
        w13=w13.contiguous(), w13_scales=w13_s, w13_bias=None,
        w2=w2.contiguous(), w2_scales=w2_s, w2_bias=None,
        n_experts_per_token=topk,
        activation="situ" if args.situ else "silu",
        num_experts=args.experts,
        is_mxfp4=bool(args.mxfp4),
    )
    if args.situ:
        common.update(situ_beta=4.0, situ_linear_beta=25.0)  # K3's real values

    # (a) one batched call, M = tokens
    batched = torch.empty_like(hidden)
    xpu_fused_moe(
        hidden_states=hidden, topk_weights=weights, topk_ids=ids,
        output=batched, **common,
    )
    torch.xpu.synchronize()

    # (b) N separate calls, M = 1 each
    singles = torch.empty_like(hidden)
    for i in range(args.tokens):
        row = torch.empty_like(hidden[i : i + 1])
        xpu_fused_moe(
            hidden_states=hidden[i : i + 1].contiguous(),
            topk_weights=weights[i : i + 1].contiguous(),
            topk_ids=ids[i : i + 1].contiguous(),
            output=row, **common,
        )
        singles[i] = row[0]
    torch.xpu.synchronize()

    mode = "MXFP4" if args.mxfp4 else "bf16"
    act = "situ" if args.situ else "silu"
    print(f"\n=== M=1 vs M={args.tokens} equivalence [{mode}/{act}] ===")
    print(f"experts={args.experts} topk={topk} hidden={args.hidden} inter={args.inter}")

    a, b = batched.float(), singles.float()
    if not torch.isfinite(b).all():
        print("  !! M=1 output contains non-finite values")
    per_row = (a - b).abs().amax(dim=-1)
    worst = int(per_row.argmax())
    print(f"  max abs diff overall : {per_row.max().item():.6e}")
    print(f"  worst row            : {worst} (diff {per_row[worst].item():.6e})")
    print(f"  per-row diffs        : {[f'{v:.2e}' for v in per_row.tolist()]}")

    ok = torch.allclose(a, b, atol=args.atol, rtol=args.rtol)
    if ok:
        print(f"\n  PASS - fused MoE is M-shape-clean at these dims.")
        print("  NULL RESULT, not an all-clear: this single-tile probe does not")
        print("  exercise TP=32/EP-on or the rest of K3. Go to the reduced-K3")
        print("  harness (TP=8, one node, free debug queue).")
        return 0

    print(f"\n  FAIL - M=1 diverges from M={args.tokens} beyond tolerance.")
    print("  This is the bug: the MoE is per-token independent, so batch size")
    print("  must not change a row. Capture dims + tile geometry and file")
    print("  against cutlass_grouped_gemm_interface (libgrouped_gemm_xe_2.so).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

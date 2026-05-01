"""Numerical parity: FusedAdamWBf16 vs AdamWBf16.

Both classes share the same recipe (FP32 math, BF16 CPU state, decoupled WD,
per-param bias correction). FusedAdamWBf16 batches the math via foreach ops,
so element-wise results must match the per-param loop bit-for-bit (no parallel
reductions inside the loop — every op is per-element).

CPU-only; runs in the regression suite.
"""

import torch

from torchtune.dev.bioreason.optim import AdamWBf16, FusedAdamWBf16


def _make_param(shape, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(shape, generator=g, dtype=torch.bfloat16))
    p.grad = torch.randn(shape, generator=g, dtype=torch.bfloat16) * 0.01
    return p


def _set_grad(p, seed):
    g = torch.Generator().manual_seed(seed)
    p.grad = torch.randn_like(p, dtype=torch.bfloat16) * 0.01


def test_fused_adamw_bf16_matches_sequential_three_steps():
    shapes = [(64,), (16, 32), (8, 4, 4), (128,)]

    seq_params = [_make_param(s, seed=100 + i) for i, s in enumerate(shapes)]
    fused_params = [
        torch.nn.Parameter(p.detach().clone()) for p in seq_params
    ]
    for sp, fp in zip(seq_params, fused_params):
        fp.grad = sp.grad.detach().clone()

    seq_opt = AdamWBf16(
        seq_params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )
    fused_opt = FusedAdamWBf16(
        fused_params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )

    for step_idx in range(3):
        seq_opt.step()
        fused_opt.step()

        for sp, fp in zip(seq_params, fused_params):
            assert torch.equal(sp.data, fp.data), (
                f"step {step_idx}: param mismatch shape={tuple(sp.shape)}, "
                f"max abs diff = {(sp.data.float() - fp.data.float()).abs().max().item()}"
            )

        for sp, fp in zip(seq_params, fused_params):
            seed = 200 + step_idx * 100 + id(sp) % 1000
            _set_grad(sp, seed)
            fp.grad = sp.grad.detach().clone()

    for sp, fp in zip(seq_params, fused_params):
        sp_state = seq_opt.state[sp]
        fp_state = fused_opt.state[fp]
        assert torch.equal(sp_state["exp_avg"], fp_state["exp_avg"]), (
            "exp_avg mismatch after 3 steps"
        )
        assert torch.equal(sp_state["exp_avg_sq"], fp_state["exp_avg_sq"]), (
            "exp_avg_sq mismatch after 3 steps"
        )
        assert sp_state["step"] == fp_state["step"]


def test_fused_adamw_bf16_handles_no_weight_decay():
    p_seq = _make_param((32,), seed=42)
    p_fused = torch.nn.Parameter(p_seq.detach().clone())
    p_fused.grad = p_seq.grad.detach().clone()

    seq_opt = AdamWBf16([p_seq], lr=1e-3, weight_decay=0.0)
    fused_opt = FusedAdamWBf16([p_fused], lr=1e-3, weight_decay=0.0)

    seq_opt.step()
    fused_opt.step()

    assert torch.equal(p_seq.data, p_fused.data)


def test_fused_adamw_bf16_skips_params_without_grad():
    p_with_grad = _make_param((16,), seed=7)
    p_no_grad = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    p_no_grad.grad = None

    fused_opt = FusedAdamWBf16(
        [p_with_grad, p_no_grad], lr=1e-3, weight_decay=0.01,
    )

    snapshot = p_no_grad.data.clone()
    fused_opt.step()
    assert torch.equal(p_no_grad.data, snapshot), "no-grad param should be untouched"

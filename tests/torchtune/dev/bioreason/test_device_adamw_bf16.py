import torch

from torchtune.dev.bioreason.optim import (
    AdamWBf16,
    DeviceFactoredAdafactor,
    DeviceAdamWBf16,
    DeviceAdamWInt8,
)


def test_factored_adafactor_uses_factored_state_and_updates():
    parameter = torch.nn.Parameter(torch.ones(8, 4, dtype=torch.float32))
    optimizer = DeviceFactoredAdafactor([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["exp_avg_sq_row"].shape == (8,)
    assert state["exp_avg_sq_col"].shape == (4,)
    assert "exp_avg_sq" not in state
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, torch.ones_like(parameter))


def test_factored_adafactor_handles_nonuniform_factors_without_full_state():
    parameter = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float32))
    gradient = torch.randn(6, 4, dtype=torch.float32)
    optimizer = DeviceFactoredAdafactor([parameter], lr=1e-3, beta2=0.5)
    parameter.grad = gradient
    optimizer.step()

    state = optimizer.state[parameter]
    assert torch.isfinite(parameter).all()
    assert torch.isfinite(state["exp_avg_sq_row"]).all()
    assert torch.isfinite(state["exp_avg_sq_col"]).all()
    assert (
        state["exp_avg_sq_row"].numel() + state["exp_avg_sq_col"].numel()
        < parameter.numel()
    )


def test_factored_adafactor_matches_reference_update():
    parameter = torch.nn.Parameter(torch.randn(5, 3, dtype=torch.float32))
    initial = parameter.detach().clone()
    gradient = torch.randn_like(parameter)
    beta2 = 0.5
    learning_rate = 1e-3
    optimizer = DeviceFactoredAdafactor(
        [parameter], lr=learning_rate, beta2=beta2, clip_threshold=0.0
    )
    parameter.grad = gradient.clone()
    optimizer.step()

    row = gradient.square().mean(dim=1) * (1.0 - beta2)
    col = gradient.square().mean(dim=0) * (1.0 - beta2)
    second_moment = row[:, None] * col[None, :]
    second_moment.div_(second_moment.mean().clamp_min(1e-30))
    expected = initial - learning_rate * gradient / second_moment.sqrt().clamp_min(1e-3)
    torch.testing.assert_close(parameter, expected)


def test_factored_adafactor_matches_reference_across_steps():
    parameter = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float32))
    reference = parameter.detach().clone()
    optimizer = DeviceFactoredAdafactor(
        [parameter], lr=2e-3, beta2=0.8, clip_threshold=0.0
    )
    row = torch.zeros(4)
    col = torch.zeros(3)
    for step_idx in range(3):
        gradient = torch.randn(4, 3, generator=torch.Generator().manual_seed(50 + step_idx))
        parameter.grad = gradient.clone()
        optimizer.step()

        row.mul_(0.8).add_(gradient.square().mean(dim=1), alpha=0.2)
        col.mul_(0.8).add_(gradient.square().mean(dim=0), alpha=0.2)
        second_moment = row[:, None] * col[None, :]
        second_moment.div_(second_moment.mean().clamp_min(1e-30))
        reference.add_(
            (gradient / second_moment.sqrt().clamp_min(1e-3)), alpha=-2e-3
        )

    torch.testing.assert_close(parameter, reference)


def test_factored_adafactor_preserves_epsilon_clamp_order():
    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
    gradient = torch.full((2, 2), 1e-4)
    optimizer = DeviceFactoredAdafactor(
        [parameter], lr=1e-3, beta2=0.5, eps=(1e-30, 1e-3), clip_threshold=0.0
    )
    parameter.grad = gradient
    optimizer.step()

    row = gradient.square().mean(dim=1) * 0.5
    col = gradient.square().mean(dim=0) * 0.5
    denominator = row[:, None] * col[None, :]
    denominator.div_(denominator.mean().clamp_min(1e-30))
    expected = -1e-3 * gradient / denominator.sqrt().clamp_min(1e-3)
    torch.testing.assert_close(parameter, expected)


def test_device_adamw_bf16_matches_cpu_bf16_state_optimizer():
    cpu_parameter = torch.nn.Parameter(torch.randn(32, dtype=torch.bfloat16))
    device_parameter = torch.nn.Parameter(cpu_parameter.detach().clone())
    cpu_optimizer = AdamWBf16([cpu_parameter], lr=3e-4, weight_decay=0.01)
    device_optimizer = DeviceAdamWBf16(
        [device_parameter], lr=3e-4, weight_decay=0.01
    )

    for step_idx in range(3):
        generator = torch.Generator().manual_seed(1234 + step_idx)
        gradient = torch.randn(32, generator=generator, dtype=torch.bfloat16)
        cpu_parameter.grad = gradient.clone()
        device_parameter.grad = gradient.clone()
        cpu_optimizer.step()
        device_optimizer.step()

    assert torch.allclose(cpu_parameter, device_parameter, atol=0.02, rtol=0.02)
    assert device_optimizer.state[device_parameter]["exp_avg"].dtype == torch.bfloat16
    assert device_optimizer.state[device_parameter]["exp_avg"].device.type == "cpu"


def test_device_adamw_int8_moments_are_finite_and_compressed():
    parameter = torch.nn.Parameter(torch.randn(32, dtype=torch.bfloat16))
    optimizer = DeviceAdamWInt8([parameter], lr=3e-4, weight_decay=0.01)

    for step_idx in range(3):
        generator = torch.Generator().manual_seed(5678 + step_idx)
        parameter.grad = torch.randn(
            32, generator=generator, dtype=torch.bfloat16
        )
        optimizer.step()

    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == torch.int8
    assert state["exp_avg_sq"].dtype == torch.int8
    assert torch.isfinite(parameter).all()
    assert torch.isfinite(state["exp_avg_scale"])
    assert torch.isfinite(state["exp_avg_sq_scale"])

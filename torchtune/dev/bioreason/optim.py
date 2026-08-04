import torch


class DeviceFactoredAdafactor(torch.optim.Optimizer):
    """Device-local factored second-moment optimizer for memory-limited runs.

    Two-dimensional parameters keep one row and one column accumulator instead
    of a full second-moment tensor. Vectors and tensors with rank other than
    two use a single scalar accumulator. This is intentionally an experimental
    state-light optimizer and does not claim AdamW numerical equivalence.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta2=0.999,
        eps=(1e-30, 1e-3),
        weight_decay=0.0,
        clip_threshold=1.0,
        fused=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _local_tensor(value):
        return value._local_tensor if hasattr(value, "_local_tensor") else value

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta2 = group["beta2"]
            epsilon1, epsilon2 = group["eps"]
            learning_rate = group["lr"]
            weight_decay = group["weight_decay"]
            clip_threshold = group["clip_threshold"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter_local = self._local_tensor(parameter)
                gradient_local = self._local_tensor(parameter.grad)
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    if parameter_local.ndim == 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            parameter_local.shape[0],
                            device=parameter_local.device,
                            dtype=torch.float32,
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            parameter_local.shape[1],
                            device=parameter_local.device,
                            dtype=torch.float32,
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros(
                            (), device=parameter_local.device, dtype=torch.float32
                        )

                state["step"] += 1
                gradient = gradient_local.float()
                if parameter_local.ndim == 2:
                    second_moment_row = state["exp_avg_sq_row"]
                    second_moment_col = state["exp_avg_sq_col"]
                    second_moment_row.mul_(beta2).add_(
                        gradient.square().mean(dim=1), alpha=1.0 - beta2
                    )
                    second_moment_col.mul_(beta2).add_(
                        gradient.square().mean(dim=0), alpha=1.0 - beta2
                    )
                    normalized_gradient = gradient
                    factored_second_moment = (
                        second_moment_row[:, None] * second_moment_col[None, :]
                    )
                    factored_second_moment.div_(
                        (
                            second_moment_row.mean() * second_moment_col.mean()
                        ).clamp_min(epsilon1)
                    )
                    normalized_gradient.div_(
                        factored_second_moment.sqrt().clamp_min(epsilon2)
                    )
                else:
                    second_moment = state["exp_avg_sq"]
                    second_moment.mul_(beta2).add_(gradient.square().mean(), alpha=1.0 - beta2)
                    normalized_gradient = gradient / second_moment.sqrt().clamp_min(epsilon2)

                rms = normalized_gradient.square().mean().sqrt().clamp_min(epsilon1)
                if clip_threshold > 0.0:
                    normalized_gradient.div_(torch.maximum(rms / clip_threshold, torch.ones_like(rms)))
                if weight_decay != 0.0:
                    normalized_gradient.add_(parameter_local, alpha=weight_decay)
                parameter_local.add_(normalized_gradient.to(parameter_local.dtype), alpha=-learning_rate)

        return loss


class AdamWBf16(torch.optim.AdamW):
    """AdamW with BF16 momentum buffers and a fully CPU-side update.

    Each `step()` walks parameters one at a time. For each param:
      1. Move grad (DTensor local BF16) to CPU as FP32.
      2. Update CPU BF16 moments in place (Adam math in FP32, cast back to BF16).
      3. Compute the param delta on CPU as FP32, cast to param dtype,
         move to GPU and add to the param's local tensor.

    Peak extra GPU memory per step is bounded by a single param's delta
    (largest sharded tensor on the model, typically <1 GiB for 30B/24-rank EP).
    No optimizer state ever lives on GPU. Trades ~PCIe bandwidth for memory.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                # Unwrap DTensor → local plain tensor on the param's device.
                grad_local = grad._local_tensor if hasattr(grad, "_local_tensor") else grad
                # Pull to CPU FP32 for the math.
                grad_cpu = grad_local.detach().to(device="cpu", dtype=torch.float32, copy=True)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad_cpu, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(grad_cpu, dtype=torch.bfloat16)

                state["step"] += 1
                step_n = state["step"]

                exp_avg_bf16 = state["exp_avg"]
                exp_avg_sq_bf16 = state["exp_avg_sq"]

                # Promote moments to FP32 for the update, cast back when storing.
                exp_avg = exp_avg_bf16.to(torch.float32)
                exp_avg_sq = exp_avg_sq_bf16.to(torch.float32)

                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1 ** step_n
                bias_c2 = 1.0 - beta2 ** step_n
                step_size = lr / bias_c1
                denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                update_cpu = (exp_avg / denom).mul_(step_size)

                # AdamW decoupled weight decay: applied directly to params.
                if weight_decay != 0.0:
                    # delta += lr * wd * p   (then we subtract delta from p)
                    # Pull a CPU FP32 copy of the local param shard for the wd term.
                    p_local_cpu = (
                        p._local_tensor if hasattr(p, "_local_tensor") else p
                    ).detach().to(device="cpu", dtype=torch.float32, copy=True)
                    update_cpu.add_(p_local_cpu, alpha=lr * weight_decay)
                    del p_local_cpu

                # Store moments back as BF16 (in place where shape matches).
                state["exp_avg"] = exp_avg.to(torch.bfloat16)
                state["exp_avg_sq"] = exp_avg_sq.to(torch.bfloat16)
                del exp_avg, exp_avg_sq, exp_avg_bf16, exp_avg_sq_bf16

                # Move the delta to GPU (param dtype) and apply: p -= update.
                p_local = p._local_tensor if hasattr(p, "_local_tensor") else p
                update_gpu = update_cpu.to(device=p_local.device, dtype=p_local.dtype)
                p_local.sub_(update_gpu)
                del update_cpu, update_gpu, grad_cpu, grad_local

        return loss


class DeviceAdamWBf16(torch.optim.AdamW):
    """AdamW with BF16 moment state kept on each parameter device."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                parameter_local = (
                    parameter._local_tensor
                    if hasattr(parameter, "_local_tensor")
                    else parameter
                )
                grad_local = (
                    parameter.grad._local_tensor
                    if hasattr(parameter.grad, "_local_tensor")
                    else parameter.grad
                )
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter_local, dtype=torch.bfloat16
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter_local, dtype=torch.bfloat16
                    )

                state["step"] += 1
                step_n = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_local, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_local, grad_local, value=1.0 - beta2
                )

                bias_c1 = 1.0 - beta1**step_n
                bias_c2 = 1.0 - beta2**step_n
                update = exp_avg.float().div_(bias_c1)
                update.div_(exp_avg_sq.float().div_(bias_c2).sqrt().add_(eps))
                if weight_decay != 0.0:
                    update.add_(parameter_local.float(), alpha=weight_decay)
                parameter_local.sub_(update, alpha=lr)

        return loss


class DeviceAdamWInt8(torch.optim.AdamW):
    """AdamW with dynamically scaled int8 moment state on the parameter device."""

    @staticmethod
    def _quantize(moment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = moment.abs().amax().clamp_min(1.0e-12) / 127.0
        quantized = torch.round(moment / scale).clamp(-127, 127).to(torch.int8)
        return quantized, scale

    @staticmethod
    def _dequantize(
        quantized: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return quantized.float() * scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                parameter_local = (
                    parameter._local_tensor
                    if hasattr(parameter, "_local_tensor")
                    else parameter
                )
                grad_local = (
                    parameter.grad._local_tensor
                    if hasattr(parameter.grad, "_local_tensor")
                    else parameter.grad
                )
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter_local, dtype=torch.int8
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter_local, dtype=torch.int8
                    )
                    state["exp_avg_scale"] = torch.ones(
                        (), device=parameter_local.device, dtype=torch.float32
                    )
                    state["exp_avg_sq_scale"] = torch.ones(
                        (), device=parameter_local.device, dtype=torch.float32
                    )

                state["step"] += 1
                step_n = state["step"]
                exp_avg = self._dequantize(
                    state["exp_avg"], state["exp_avg_scale"]
                )
                exp_avg_sq = self._dequantize(
                    state["exp_avg_sq"], state["exp_avg_sq_scale"]
                )
                exp_avg.mul_(beta1).add_(grad_local, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_local, grad_local, value=1.0 - beta2
                )
                (
                    state["exp_avg"],
                    state["exp_avg_scale"],
                ) = self._quantize(exp_avg)
                (
                    state["exp_avg_sq"],
                    state["exp_avg_sq_scale"],
                ) = self._quantize(exp_avg_sq)

                bias_c1 = 1.0 - beta1**step_n
                bias_c2 = 1.0 - beta2**step_n
                update = exp_avg.div_(bias_c1)
                update.div_(exp_avg_sq.div_(bias_c2).sqrt().add_(eps))
                if weight_decay != 0.0:
                    update.add_(parameter_local, alpha=weight_decay)
                parameter_local.sub_(update, alpha=lr)

        return loss


class FusedAdamWBf16(torch.optim.AdamW):
    """Foreach AdamW with BF16 CPU moments — same memory profile as AdamWBf16,
    same numerical recipe (FP32 math, BF16 storage, decoupled weight decay),
    but the per-param Python loop is replaced with `torch._foreach_*` ops over
    a batched list. State stays on CPU; PCIe round-trips per param remain, but
    the CPU FP32 math is the dominant cost in the AdamWBf16 loop and `_foreach_*`
    pipelines that work over the whole param list at once.

    Why: validated 2026-05-01 on Qwen3-30B-A3B EP=8 v10f. AdamWBf16's per-param
    sequential Python loop measured 85s/step (66% of post-iter2 wall). Moving
    state on-device is not an option — 15 GiB of BF16 moments doesn't fit the
    13 GiB headroom on PVC tile under the v10f envelope.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            params = []
            grads_cpu_fp32 = []
            exp_avg_fp32 = []
            exp_avg_sq_fp32 = []
            step_sizes = []
            denom_corrections = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                grad_local = grad._local_tensor if hasattr(grad, "_local_tensor") else grad
                grad_cpu = grad_local.detach().to(device="cpu", dtype=torch.float32, copy=True)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad_cpu, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(grad_cpu, dtype=torch.bfloat16)

                state["step"] += 1
                step_n = state["step"]

                params.append(p)
                grads_cpu_fp32.append(grad_cpu)
                exp_avg_fp32.append(state["exp_avg"].to(torch.float32))
                exp_avg_sq_fp32.append(state["exp_avg_sq"].to(torch.float32))

                bias_c1 = 1.0 - beta1 ** step_n
                bias_c2 = 1.0 - beta2 ** step_n
                step_sizes.append(lr / bias_c1)
                denom_corrections.append(bias_c2 ** 0.5)

            if not params:
                continue

            # Batched moment update (CPU FP32):
            #   exp_avg     = beta1 * exp_avg     + (1-beta1) * grad
            #   exp_avg_sq  = beta2 * exp_avg_sq  + (1-beta2) * grad * grad
            torch._foreach_mul_(exp_avg_fp32, beta1)
            torch._foreach_add_(exp_avg_fp32, grads_cpu_fp32, alpha=1.0 - beta1)

            torch._foreach_mul_(exp_avg_sq_fp32, beta2)
            torch._foreach_addcmul_(
                exp_avg_sq_fp32, grads_cpu_fp32, grads_cpu_fp32,
                value=1.0 - beta2,
            )

            # denom = sqrt(exp_avg_sq) / bias_c2_sqrt + eps   (per param)
            denoms = torch._foreach_sqrt(exp_avg_sq_fp32)
            torch._foreach_div_(denoms, denom_corrections)
            torch._foreach_add_(denoms, eps)

            # update = (exp_avg / denom) * step_size   (per param)
            updates_cpu = torch._foreach_div(exp_avg_fp32, denoms)
            torch._foreach_mul_(updates_cpu, step_sizes)

            # AdamW decoupled weight decay: update += lr * wd * p_local_cpu_fp32.
            # Mirror AdamWBf16's behavior — applied via the update tensor (so we
            # only need the H2D copy once per param). Skip the per-param H2D
            # of params: pull all in one foreach copy.
            if weight_decay != 0.0:
                p_local_list = [
                    (p._local_tensor if hasattr(p, "_local_tensor") else p)
                    for p in params
                ]
                p_cpu_fp32 = [
                    pl.detach().to(device="cpu", dtype=torch.float32, copy=True)
                    for pl in p_local_list
                ]
                torch._foreach_add_(updates_cpu, p_cpu_fp32, alpha=lr * weight_decay)
                del p_cpu_fp32, p_local_list

            # Move all updates to GPU at once (foreach .to is a per-tensor copy
            # under the hood; the win is the surrounding Python overhead, not
            # the PCIe budget itself).
            updates_gpu = []
            for upd, p in zip(updates_cpu, params):
                p_local = p._local_tensor if hasattr(p, "_local_tensor") else p
                updates_gpu.append(
                    upd.to(device=p_local.device, dtype=p_local.dtype)
                )

            # p_local -= update_gpu  (foreach over local tensors)
            p_local_targets = [
                (p._local_tensor if hasattr(p, "_local_tensor") else p)
                for p in params
            ]
            torch._foreach_sub_(p_local_targets, updates_gpu)

            # Store moments back as BF16 (cast happens per tensor; the loop is
            # short — len(params) — and unavoidable since state[p] is a dict
            # keyed by param object).
            for p, ea, eas in zip(params, exp_avg_fp32, exp_avg_sq_fp32):
                self.state[p]["exp_avg"] = ea.to(torch.bfloat16)
                self.state[p]["exp_avg_sq"] = eas.to(torch.bfloat16)

            del exp_avg_fp32, exp_avg_sq_fp32, denoms, updates_cpu, updates_gpu
            del grads_cpu_fp32, params, p_local_targets

        return loss

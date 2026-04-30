import torch


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

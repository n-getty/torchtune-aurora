import torch


class AdamWBf16(torch.optim.AdamW):
    """AdamW with bf16 momentum buffers, CPU-offloaded between steps.

    Optimizer moments (exp_avg + exp_avg_sq) are cast to bf16 and moved to CPU
    after each step, freeing ~16 GiB of GPU memory between steps. They are moved
    back to GPU at the start of the next step. This restores the same GPU memory
    profile at each step start regardless of step index.
    """

    def step(self, closure=None):
        # Restore moments to GPU before computing update
        self._moments_to_gpu()
        loss = super().step(closure)
        # Cast to bf16 and offload to CPU to free GPU memory between steps
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in state:
                        state[k] = state[k].to(torch.bfloat16).cpu()
        return loss

    def _moments_to_gpu(self):
        if not self.state:
            return
        # Use the device of the first parameter as target
        for group in self.param_groups:
            for p in group["params"]:
                device = p.device
                break
            else:
                continue
            break
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in state and state[k].device != device:
                        state[k] = state[k].to(device)

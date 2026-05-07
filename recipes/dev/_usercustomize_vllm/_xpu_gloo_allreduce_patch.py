"""Plan-B fallback patch: route vLLM XPU TP all_reduce / all_gather / reduce_scatter
through a gloo subgroup (CPU bounce) instead of the XCCL device_group.

Use ONLY if env-only fixes (CCL_ZE_IPC_EXCHANGE=sockets, CCL_ALLREDUCE=ring) do NOT
unblock the vLLM TP>1 prefill plen>32 hang.

Activation: set TORCHTUNE_VLLM_XPU_GLOO_TP=1 in the environment. The patch is
applied lazily once vllm.distributed.device_communicators.xpu_communicator is
imported. cpu_group is already a gloo group inside vLLM, so we just route the
collective through it and bounce the device tensor through CPU. Bandwidth-bound
(8 ranks × hidden bf16 → ~32 KB per token-step on TP boundary) but should be
faster than 'never returns'.

Equivalence: D2H bf16 -> dist.all_reduce(group=cpu_group) -> H2D. Bit-exact.
"""
import os
import sys
import builtins
import logging

_log = logging.getLogger("xpu_gloo_allreduce_patch")
_PATCHED = False
_ENABLED = os.environ.get("TORCHTUNE_VLLM_XPU_GLOO_TP", "0") == "1"


def _patch_xpu_communicator(mod):
    global _PATCHED
    if _PATCHED:
        return
    import torch
    import torch.distributed as dist

    Cls = getattr(mod, "XpuCommunicator", None)
    if Cls is None:
        return

    _orig_all_reduce = Cls.all_reduce
    _orig_all_gather = getattr(Cls, "all_gather", None)
    _orig_reduce_scatter = Cls.reduce_scatter

    def _gloo_all_reduce(self, input_):
        # CPU bounce through self.cpu_group (gloo, already created by vLLM).
        # Preserves dtype, shape, layout. No autograd inside vLLM.
        cpu = input_.detach().to("cpu", non_blocking=False)
        dist.all_reduce(cpu, group=self.cpu_group)
        out = cpu.to(input_.device, non_blocking=False)
        if out.data_ptr() != input_.data_ptr():
            input_.copy_(out)
        return input_

    def _gloo_reduce_scatter(self, input_, dim=-1):
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()
        input_tensor = input_.movedim(0, dim).contiguous()
        assert input_tensor.shape[0] % world_size == 0
        chunk = input_tensor.shape[0] // world_size
        out_shape = (chunk,) + input_tensor.shape[1:]
        cpu_in = input_tensor.detach().to("cpu", non_blocking=False)
        cpu_out = torch.empty(out_shape, dtype=cpu_in.dtype, device="cpu")
        dist.reduce_scatter_tensor(cpu_out, cpu_in, group=self.cpu_group)
        out = cpu_out.to(input_.device, non_blocking=False)
        return out.movedim(0, dim).contiguous()

    Cls.all_reduce = _gloo_all_reduce
    Cls.reduce_scatter = _gloo_reduce_scatter
    _PATCHED = True
    print(
        f"[xpu_gloo_allreduce_patch] PID={os.getpid()} "
        "XpuCommunicator.all_reduce/reduce_scatter routed via cpu_group (gloo).",
        flush=True,
    )


def install():
    """Hook builtins.__import__ to patch XpuCommunicator on first import.

    Returns True if hook installed (or already installed), False if disabled.
    """
    if not _ENABLED:
        return False
    if getattr(install, "_done", False):
        return True

    _orig = builtins.__import__
    _in = [False]

    def _hook(name, *a, **kw):
        mod = _orig(name, *a, **kw)
        if _in[0]:
            return mod
        _in[0] = True
        try:
            xc = sys.modules.get(
                "vllm.distributed.device_communicators.xpu_communicator"
            )
            if xc is not None:
                _patch_xpu_communicator(xc)
                if _PATCHED:
                    builtins.__import__ = _orig
        finally:
            _in[0] = False
        return mod

    builtins.__import__ = _hook
    install._done = True
    print(
        f"[xpu_gloo_allreduce_patch] install() armed PID={os.getpid()}",
        flush=True,
    )
    return True


# Auto-install on module import (matches usercustomize style)
install()

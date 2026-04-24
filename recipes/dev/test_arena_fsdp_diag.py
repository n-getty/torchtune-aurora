"""
Diagnostic test: arena allocator + FSDP2 with per-layer timing hooks.
Also launches strace on a background thread when detecting a hang.

Usage:
    # With arena:
    XPU_USM_ALLOC_SO=recipes/dev/usm_arena_alloc.so \
        torchrun --nproc_per_node=10 recipes/dev/test_arena_fsdp_diag.py

    # Without arena (baseline):
    torchrun --nproc_per_node=10 recipes/dev/test_arena_fsdp_diag.py
"""
import os
import sys
import time
import signal
import subprocess
import threading

_usm_so = os.environ.get("XPU_USM_ALLOC_SO")
if _usm_so:
    import torch
    from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
    _alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
    change_current_allocator(_alloc)
    torch.xpu.memory_allocated = lambda device=None: 0
    torch.xpu.memory_reserved = lambda device=None: 0
    torch.xpu.reset_peak_memory_stats = lambda device=None: None
    torch.xpu.empty_cache = lambda: None

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from copy import deepcopy

HANG_TIMEOUT = 30  # seconds to wait before declaring a hang

def _strace_self(duration=10):
    """Attach strace for `duration` seconds and print summary of syscalls."""
    pid = os.getpid()
    try:
        result = subprocess.run(
            ["strace", "-c", "-p", str(pid), "-e", "trace=all"],
            timeout=duration, capture_output=True, text=True,
        )
        print(f"[strace summary]\n{result.stderr}", flush=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[strace failed: {e}]", flush=True)


def main():
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(device)

    # Qwen 32B dimensions
    hidden = 5120
    intermediate = 27648
    n_layers = 32
    vocab = 32000

    class TransformerLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            self.q_proj = nn.Linear(hidden, 5120, bias=False)
            self.k_proj = nn.Linear(hidden, 1024, bias=False)
            self.v_proj = nn.Linear(hidden, 1024, bias=False)
            self.o_proj = nn.Linear(5120, hidden, bias=False)
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.norm1 = nn.RMSNorm(hidden)
            self.norm2 = nn.RMSNorm(hidden)
            self._fwd_count = 0

        def forward(self, x):
            self._fwd_count += 1
            h = self.norm1(x)
            q = self.q_proj(h)
            k = self.k_proj(h)
            v = self.v_proj(h)
            h = self.o_proj(q)
            x = x + h
            h = self.norm2(x)
            h = self.gate_proj(h) * self.up_proj(h)
            h = self.down_proj(h)
            return x + h

    class FakeQwen32B(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([TransformerLayer(i) for i in range(n_layers)])
            self.norm = nn.RMSNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.head(x)

    if rank == 0:
        print(f"Building model: {n_layers} layers, hidden={hidden}, "
              f"world_size={world_size}, arena={_usm_so or 'OFF'}", flush=True)

    model = FakeQwen32B().to(dtype=torch.bfloat16, device=device)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if rank == 0:
        print(f"Model: {param_bytes / 1e9:.2f} GB, {sum(1 for _ in model.parameters())} params", flush=True)

    # Per-layer FSDP sharding
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fsdp_kwargs = {"reshard_after_forward": True, "mp_policy": mp_policy}

    for name, mod in reversed(list(model.named_modules())):
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            fully_shard(mod, **fsdp_kwargs)

    root_kwargs = deepcopy(fsdp_kwargs)
    try:
        root_kwargs["reshard_after_forward"] = None
        fully_shard(model, **root_kwargs)
    except (TypeError, ValueError):
        root_kwargs["reshard_after_forward"] = False
        fully_shard(model, **root_kwargs)

    if rank == 0:
        print(f"FSDP2 applied", flush=True)

    # Install per-layer forward hooks for timing (rank 0 only)
    layer_times = {}
    if rank == 0:
        def make_pre_hook(idx):
            def hook(mod, args):
                layer_times[idx] = time.perf_counter()
            return hook

        def make_post_hook(idx):
            def hook(mod, args, output):
                t0 = layer_times.get(idx, time.perf_counter())
                dt = time.perf_counter() - t0
                if dt > 1.0:  # only log slow layers
                    print(f"  SLOW layer {idx}: {dt:.3f}s", flush=True)
            return hook

        for i, layer in enumerate(model.layers):
            layer.register_forward_pre_hook(make_pre_hook(i))
            layer.register_forward_hook(make_post_hook(i))

    # Watchdog: detect hangs
    _last_progress = [time.perf_counter()]
    _phase = ["init"]

    def watchdog():
        while True:
            time.sleep(HANG_TIMEOUT)
            elapsed = time.perf_counter() - _last_progress[0]
            if elapsed > HANG_TIMEOUT:
                print(f"\n[WATCHDOG] HANG DETECTED after {elapsed:.0f}s in phase '{_phase[0]}'",
                      flush=True)
                # Try to get stack trace
                import traceback
                import faulthandler
                faulthandler.dump_traceback(all_threads=True)
                break

    if rank == 0:
        wd = threading.Thread(target=watchdog, daemon=True)
        wd.start()

    dummy_input = torch.randint(0, vocab, (2, 128), device=device)

    # Forward passes
    for trial in range(3):
        _phase[0] = f"forward_{trial}"
        _last_progress[0] = time.perf_counter()
        t0 = time.perf_counter()
        if rank == 0:
            print(f"Forward {trial} start...", flush=True)
        with torch.no_grad():
            out = model(dummy_input)
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        _last_progress[0] = time.perf_counter()
        if rank == 0:
            print(f"Forward {trial}: {dt:.3f}s", flush=True)

    # Also test with optimizer (Adam creates extra buffers)
    if rank == 0:
        print("Setting up optimizer...", flush=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    for trial in range(2):
        _phase[0] = f"train_{trial}"
        _last_progress[0] = time.perf_counter()
        t0 = time.perf_counter()
        if rank == 0:
            print(f"Train {trial} start...", flush=True)
        opt.zero_grad()
        out = model(dummy_input)
        loss = out.sum()
        loss.backward()
        opt.step()
        torch.xpu.synchronize(device)
        dt = time.perf_counter() - t0
        _last_progress[0] = time.perf_counter()
        if rank == 0:
            print(f"Train {trial}: {dt:.3f}s", flush=True)

    if rank == 0:
        print("ALL TESTS PASSED", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

"""
Patches for Aurora frameworks environment, applied to all Python processes.

1. Transformers version check: huggingface-hub 1.7.x vs <1.0 requirement.
2. vLLM registry subprocess: segfaults on XPU, run in-process instead.
   MUST be patched BEFORE vllm.config is imported (vLLM 0.15.0 changed the
   import chain so the registry is reached during config.__init__).
3. XPU memory: fix vLLM XPUWorker.determine_available_memory on Aurora.
"""
import importlib
import importlib.util
import os
import sys
import types

_pid = os.getpid()

# --- Patch 1: transformers version check ---
# Pre-register stub modules to prevent the version check from firing
# during transformers import. The check rejects huggingface-hub 1.7.x.
_m = types.ModuleType("transformers.utils.versions")
_m.require_version = lambda *a, **kw: None
_m.require_version_core = lambda *a, **kw: None
_m._compare_versions = lambda *a, **kw: None
sys.modules["transformers.utils.versions"] = _m

_dvc = types.ModuleType("transformers.dependency_versions_check")
_dvc.dep_version_check = lambda *a, **kw: None
sys.modules["transformers.dependency_versions_check"] = _dvc

# --- Patch 2: vLLM registry subprocess ---
# vLLM's model registry uses _run_in_subprocess to probe architectures.
# On XPU, the forked subprocess segfaults in Level Zero. Patch to fall
# back to in-process execution. Must happen BEFORE vllm.config is imported.
try:
    import vllm.model_executor.models.registry as _reg
    if hasattr(_reg, "_run_in_subprocess"):
        _orig_run = _reg._run_in_subprocess

        def _safe_run_in_subprocess(fn):
            try:
                return _orig_run(fn)
            except Exception:
                return fn()

        _reg._run_in_subprocess = _safe_run_in_subprocess
except Exception:
    pass

# --- Patch 3: XPU memory ---
# Applied lazily via __import__ hook since vllm.v1.worker.xpu_worker is
# imported much later (during engine init, not during config import).
import builtins
_original_import = builtins.__import__
_xpu_worker_patched = False
_in_hook = False


def _xpu_memory_import_hook(name, *args, **kwargs):
    global _xpu_worker_patched, _in_hook

    mod = _original_import(name, *args, **kwargs)

    if _in_hook or _xpu_worker_patched:
        return mod
    _in_hook = True

    try:
        _xpu_mod = sys.modules.get("vllm.v1.worker.xpu_worker")
        if _xpu_mod is not None and hasattr(_xpu_mod, "XPUWorker"):
            _xpu_worker_patched = True
            _apply_xpu_memory_patch(_xpu_mod.XPUWorker)
            builtins.__import__ = _original_import
    finally:
        _in_hook = False

    return mod


def _apply_xpu_memory_patch(XPUWorker):
    """Patch XPUWorker.determine_available_memory for Aurora L0 context overhead."""
    import torch
    import logging
    _logger = logging.getLogger("usercustomize_xpu")

    @torch.inference_mode()
    def _patched_determine_available_memory(self) -> int:
        """KV cache budget based on actual free memory after profiling.

        On Aurora, L0 driver contexts consume ~52 GiB. The default impl
        computes budget as total(64G) * util - peak, which over-allocates.
        We use actual free memory from mem_get_info() after profiling.
        """
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()

        free_before, total = torch.xpu.mem_get_info()
        _logger.info(
            "XPU before profile: total=%.1f GiB, free=%.1f GiB, "
            "pytorch=%.1f GiB",
            total / 1024**3, free_before / 1024**3,
            torch.xpu.memory_allocated() / 1024**3,
        )

        self.model_runner.profile_run()
        torch.xpu.empty_cache()

        free_after, _ = torch.xpu.mem_get_info()
        pytorch_current = torch.xpu.memory_allocated()

        available = int(free_after * self.cache_config.gpu_memory_utilization)

        _logger.info(
            "XPU memory (patched): free_after=%.1f GiB, "
            "pytorch=%.1f GiB, util=%.0f%%, KV budget=%.1f GiB",
            free_after / 1024**3,
            pytorch_current / 1024**3,
            self.cache_config.gpu_memory_utilization * 100,
            available / 1024**3,
        )
        return available

    XPUWorker.determine_available_memory = _patched_determine_available_memory
    print(
        f"[usercustomize] PID={_pid} "
        "XPUWorker.determine_available_memory patched",
        flush=True,
    )


builtins.__import__ = _xpu_memory_import_hook

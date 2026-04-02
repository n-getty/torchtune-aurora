"""
Patches for Aurora frameworks environment, applied to all Python processes.

1. Transformers version check: huggingface-hub 1.7.x vs <1.0 requirement.
2. vLLM registry subprocess: segfaults on XPU, run in-process instead.
3. XPU memory: fix vLLM XPUWorker.determine_available_memory on Aurora.
"""
import importlib
import importlib.util
import os
import pathlib
import sys
import types

_pid = os.getpid()

# --- Patch 1: transformers version check ---
for p in sys.path:
    _vp = pathlib.Path(p) / "transformers" / "utils" / "versions.py"
    if _vp.exists():
        _stub = types.ModuleType("transformers.utils")
        _stub.__path__ = []
        sys.modules["transformers.utils"] = _stub

        _spec = importlib.util.spec_from_file_location(
            "transformers.utils.versions", str(_vp)
        )
        _vm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vm)
        _vm._compare_versions = lambda *a, **kw: None
        sys.modules["transformers.utils.versions"] = _vm

        del sys.modules["transformers.utils"]
        break

# --- Patches 2 & 3: vLLM registry + XPU memory ---
import builtins
_original_import = builtins.__import__
_registry_patched = False
_xpu_worker_patched = False
_in_hook = False  # re-entrancy guard


def _combined_patching_import(name, *args, **kwargs):
    global _registry_patched, _xpu_worker_patched, _in_hook

    mod = _original_import(name, *args, **kwargs)

    # Avoid re-entrancy (importing inside the hook triggers the hook again)
    if _in_hook:
        return mod
    _in_hook = True

    try:
        # Patch 2: vLLM registry subprocess
        if not _registry_patched:
            _reg = sys.modules.get("vllm.model_executor.models.registry")
            if _reg is not None and hasattr(_reg, "_run_in_subprocess"):
                _orig_run = _reg._run_in_subprocess

                def _safe_run_in_subprocess(fn):
                    try:
                        return _orig_run(fn)
                    except RuntimeError:
                        return fn()

                _reg._run_in_subprocess = _safe_run_in_subprocess
                _registry_patched = True

        # Patch 3: XPU memory — check if xpu_worker is now in sys.modules
        if not _xpu_worker_patched:
            _xpu_mod = sys.modules.get("vllm.v1.worker.xpu_worker")
            if _xpu_mod is not None and hasattr(_xpu_mod, "XPUWorker"):
                _xpu_worker_patched = True  # Set BEFORE patching to prevent re-entry
                _apply_xpu_memory_patch(_xpu_mod.XPUWorker)

        # Once both patches are applied, restore original import
        if _registry_patched and _xpu_worker_patched:
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

        # Use actual free memory, scaled by utilization for headroom
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


builtins.__import__ = _combined_patching_import

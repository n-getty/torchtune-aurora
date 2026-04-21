"""
Combined usercustomize for Aurora vLLM + Gemma4 overlay.

This replaces the standalone _usercustomize_vllm/usercustomize.py when
serving Gemma4 models. It includes all existing Aurora patches PLUS
Gemma4 config and model registration.

Auto-loaded via PYTHONPATH (usercustomize.py convention).
"""
import builtins
import importlib
import importlib.util
import os
import pathlib
import sys
import types

_pid = os.getpid()

# ============================================================
# Patch 1: transformers version check (from _usercustomize_vllm)
# ============================================================
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

# ============================================================
# Patches 2-4: vLLM registry + XPU memory + Gemma4 (lazy hooks)
# ============================================================
_original_import = builtins.__import__
_registry_patched = False
_xpu_worker_patched = False
_gemma4_autoconfig_patched = False
_gemma4_config_patched = False
_gemma4_model_patched = False
_gemma4_tokenizer_patched = False
_in_hook = False


def _combined_patching_import(name, *args, **kwargs):
    global _registry_patched, _xpu_worker_patched, _in_hook
    global _gemma4_autoconfig_patched, _gemma4_config_patched, _gemma4_model_patched
    global _gemma4_tokenizer_patched

    mod = _original_import(name, *args, **kwargs)

    if _in_hook:
        return mod
    _in_hook = True

    try:
        # Patch 0: Register Gemma4 with transformers.AutoConfig (lazy).
        # Must fire AFTER ipex initializes XPU (so platform detection works),
        # but BEFORE vLLM calls AutoConfig.from_pretrained("gemma4").
        # Triggered when transformers.models.auto is imported by vLLM.
        # NOT done eagerly at usercustomize time — that would run before ipex,
        # causing vllm.platforms.resolve_current_platform() to cache
        # UnspecifiedPlatform (no XPU yet), breaking the vLLM server startup.
        if not _gemma4_autoconfig_patched:
            _auto_mod = sys.modules.get("transformers.models.auto.configuration_auto")
            if _auto_mod is not None and hasattr(_auto_mod, "AutoConfig"):
                try:
                    from vllm_gemma4_config import Gemma4TextConfig
                    _auto_mod.AutoConfig.register("gemma4", Gemma4TextConfig)
                    _gemma4_autoconfig_patched = True
                    print(
                        f"[gemma4_overlay] PID={_pid} Registered Gemma4TextConfig "
                        "with AutoConfig (lazy, post-ipex)",
                        flush=True,
                    )
                except Exception as _e:
                    print(
                        f"[gemma4_overlay] PID={_pid} AutoConfig.register failed: {_e}",
                        flush=True,
                    )

        # Patch 1b: Fix Gemma4 tokenizer extra_special_tokens format.
        # Gemma4 tokenizer_config.json has extra_special_tokens as a list,
        # but transformers 4.57.6 _set_model_specific_special_tokens()
        # calls special_tokens.keys() expecting a dict → AttributeError.
        # Fix: skip the method body when special_tokens isn't a dict.
        if not _gemma4_tokenizer_patched:
            _tok_mod = sys.modules.get("transformers.tokenization_utils_base")
            if _tok_mod is not None and hasattr(_tok_mod, "PreTrainedTokenizerBase"):
                _cls = _tok_mod.PreTrainedTokenizerBase
                _orig_set = getattr(_cls, "_set_model_specific_special_tokens", None)
                if _orig_set is not None:
                    def _safe_set_model_specific_special_tokens(self, special_tokens):
                        if not isinstance(special_tokens, dict):
                            return  # Gemma4: list format, no attribute names to add
                        return _orig_set(self, special_tokens)
                    _cls._set_model_specific_special_tokens = (
                        _safe_set_model_specific_special_tokens
                    )
                    _gemma4_tokenizer_patched = True
                    print(
                        f"[gemma4_overlay] PID={_pid} Patched "
                        "_set_model_specific_special_tokens for list format",
                        flush=True,
                    )

        # Patch 2: vLLM registry subprocess (from _usercustomize_vllm)
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

        # Patch 3: XPU memory (from _usercustomize_vllm)
        if not _xpu_worker_patched:
            _xpu_mod = sys.modules.get("vllm.v1.worker.xpu_worker")
            if _xpu_mod is not None and hasattr(_xpu_mod, "XPUWorker"):
                _xpu_worker_patched = True
                _apply_xpu_memory_patch(_xpu_mod.XPUWorker)

        # Patch 4a: Gemma4 config registration
        if not _gemma4_config_patched:
            config_mod = sys.modules.get("vllm.transformers_utils.config")
            if config_mod is not None and hasattr(config_mod, "_CONFIG_REGISTRY"):
                from vllm_gemma4_config import Gemma4TextConfig
                config_mod._CONFIG_REGISTRY["gemma4"] = Gemma4TextConfig
                _gemma4_config_patched = True
                print(
                    f"[gemma4_overlay] PID={_pid} "
                    "Registered Gemma4TextConfig in _CONFIG_REGISTRY",
                    flush=True,
                )

        # Patch 4b: Gemma4 model registration (separate — registry may load later)
        if not _gemma4_model_patched:
            registry_mod = sys.modules.get("vllm.model_executor.models.registry")
            if registry_mod is not None and hasattr(registry_mod, "ModelRegistry"):
                registry_mod.ModelRegistry.register_model(
                    "Gemma4ForCausalLM",
                    "gemma4:Gemma4ForCausalLM",
                )
                registry_mod.ModelRegistry.register_model(
                    "Gemma4ForConditionalGeneration",
                    "gemma4:Gemma4ForCausalLM",
                )
                _gemma4_model_patched = True
                print(
                    f"[gemma4_overlay] PID={_pid} "
                    "Registered Gemma4ForCausalLM in ModelRegistry",
                    flush=True,
                )

        # Restore original import when all patches applied
        all_done = (_registry_patched and _xpu_worker_patched
                    and _gemma4_autoconfig_patched and _gemma4_tokenizer_patched
                    and _gemma4_config_patched and _gemma4_model_patched)
        if all_done:
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


builtins.__import__ = _combined_patching_import

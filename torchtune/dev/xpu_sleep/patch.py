"""Monkey-patches vLLM to enable sleep/wake on XPU via XPUMemAllocator.

Call patch_vllm_for_xpu_sleep() BEFORE creating the vLLM LLM instance.
This patches:
  1. XPUPlatform.is_sleep_mode_available() -> True
  2. XPUWorker.sleep() / wake_up() -> delegate to XPUMemAllocator
  3. XPUWorker._maybe_get_memory_pool_context() -> use XPUMemAllocator
  4. XPUWorker.initialize_from_config() -> register KV caches after creation
  5. Worker.load_model() -> register weights after loading
"""

import logging
from contextlib import nullcontext

import torch

from torchtune.dev.xpu_sleep.allocator import XPUMemAllocator

logger = logging.getLogger(__name__)

_patched = False


def patch_vllm_for_xpu_sleep() -> None:
    """Apply all monkey-patches to enable XPU sleep mode in vLLM."""
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Patch XPUPlatform.is_sleep_mode_available
    from vllm.platforms.xpu import XPUPlatform
    XPUPlatform.is_sleep_mode_available = lambda self: True
    logger.info("Patched XPUPlatform.is_sleep_mode_available -> True")

    # 2. Patch XPUWorker methods
    from vllm.v1.worker.xpu_worker import XPUWorker

    def xpu_sleep(self, level: int = 1) -> None:
        """XPU sleep: back up tensors to CPU and release GPU storage."""
        allocator = XPUMemAllocator.get_instance()

        free_before = torch.xpu.mem_get_info()[0]

        # For level 2, save named_buffers separately (same as CUDA worker)
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buf.cpu().clone()
                for name, buf in model.named_buffers()
            }

        # Offload weights at level 1, nothing at level 2 (discard and reload)
        offload = ("weights",) if level == 1 else ()
        allocator.sleep(offload_tags=offload)

        free_after = torch.xpu.mem_get_info()[0]
        freed = free_after - free_before
        logger.info(
            "XPU sleep(level=%d) freed %.2f GiB, %.2f GiB still used",
            level, freed / 2**30, (torch.xpu.mem_get_info()[1] - free_after) / 2**30,
        )

    def xpu_wake_up(self, tags: list[str] | None = None) -> None:
        """XPU wake_up: restore tensors from CPU backups."""
        allocator = XPUMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore named_buffers saved during level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buf in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buf.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        # Reset FP8 scales if needed
        if (
            (tags is None or "kv_cache" in tags)
            and self.cache_config.cache_dtype.startswith("fp8")
            and hasattr(self.model_runner, "init_fp8_kv_scales")
        ):
            self.model_runner.init_fp8_kv_scales()

    def xpu_maybe_get_memory_pool_context(self, tag: str):
        """Return a context manager for tagged memory pool tracking."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = XPUMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()

    # Save original initialize_from_config to wrap it
    _orig_initialize_from_config = XPUWorker.initialize_from_config

    def xpu_initialize_from_config(self, kv_cache_config) -> None:
        """Wrap KV cache init to register caches with XPUMemAllocator."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = XPUMemAllocator.get_instance()
            with allocator.use_memory_pool(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
            # Register the KV cache tensors after creation
            if hasattr(self.model_runner, 'kv_caches'):
                allocator.register_kv_caches(self.model_runner.kv_caches)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    XPUWorker.sleep = xpu_sleep
    XPUWorker.wake_up = xpu_wake_up
    XPUWorker._maybe_get_memory_pool_context = xpu_maybe_get_memory_pool_context
    XPUWorker.initialize_from_config = xpu_initialize_from_config
    logger.info("Patched XPUWorker sleep/wake_up/initialize_from_config")

    # 3. Patch the parent Worker.load_model to register weights after loading
    from vllm.v1.worker.gpu_worker import Worker

    _orig_load_model = Worker.load_model

    def load_model_with_registration(self) -> None:
        """Load model, then register weights with XPUMemAllocator if on XPU."""
        _orig_load_model(self)

        if (
            hasattr(self, 'device_config')
            and self.device_config.device_type == "xpu"
            and self.vllm_config.model_config.enable_sleep_mode
        ):
            allocator = XPUMemAllocator.get_instance()
            allocator.register_model_weights(self.model_runner.model)

    Worker.load_model = load_model_with_registration
    logger.info("Patched Worker.load_model for XPU weight registration")

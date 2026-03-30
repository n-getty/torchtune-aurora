"""XPU sleep/wake for vLLM — enables colocated inference + FSDP training."""

from torchtune.dev.xpu_sleep.allocator import XPUMemAllocator
from torchtune.dev.xpu_sleep.patch import patch_vllm_for_xpu_sleep

__all__ = ["XPUMemAllocator", "patch_vllm_for_xpu_sleep"]

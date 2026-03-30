"""XPU memory allocator for vLLM sleep/wake on Intel GPUs.

Provides the same sleep()/wake_up() semantics as CuMemAllocator but operates
at the Python tensor level instead of using CUDA virtual memory APIs.

sleep(): Backs up GPU tensors to CPU, then releases GPU storage via
    storage().resize_(0) so the caching allocator can reuse the blocks.
wake_up(): Restores GPU tensors from CPU backups, then frees CPU copies.

This avoids two XPU-specific issues:
    1. CuMemAllocator requires CUDA driver APIs (cuMemCreate/cuMemMap) that
       don't exist on XPU.
    2. torch.xpu.empty_cache() leaks UR handles in Level Zero when combined
       with FSDP storage.resize_() — so we never call it.
"""

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorRecord:
    """Tracks a registered GPU tensor for sleep/wake management."""
    name: str
    tensor: torch.Tensor
    original_storage_size: int
    original_shape: tuple = ()
    device: torch.device | None = None
    cpu_backup: torch.Tensor | None = None


class XPUMemAllocator:
    """Singleton allocator managing sleep/wake for XPU tensors.

    Tensors are registered under string tags (e.g. "weights", "kv_cache").
    sleep() offloads all registered tensors to CPU and releases GPU storage.
    wake_up() restores them from CPU backups.
    """

    _instance: "XPUMemAllocator | None" = None

    @classmethod
    def get_instance(cls) -> "XPUMemAllocator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # tag -> list of TensorRecords
        self._pools: dict[str, list[TensorRecord]] = {}
        self._current_tag: str = "default"
        self._is_sleeping: bool = False

    def register_tensor(self, tag: str, name: str, tensor: torch.Tensor) -> None:
        """Register a GPU tensor under the given tag for sleep/wake management."""
        if tag not in self._pools:
            self._pools[tag] = []
        record = TensorRecord(
            name=name,
            tensor=tensor,
            original_storage_size=tensor.untyped_storage().size(),
            original_shape=tuple(tensor.shape),
            device=tensor.device,
        )
        self._pools[tag].append(record)

    def register_model_weights(self, model: torch.nn.Module) -> None:
        """Register all model parameters and buffers under the 'weights' tag."""
        count = 0
        total_bytes = 0
        for name, param in model.named_parameters():
            self.register_tensor("weights", name, param.data)
            total_bytes += param.data.nelement() * param.data.element_size()
            count += 1
        for name, buf in model.named_buffers():
            self.register_tensor("weights", f"buffer.{name}", buf)
            total_bytes += buf.nelement() * buf.element_size()
            count += 1
        logger.info(
            "Registered %d weight tensors (%.2f GiB) for sleep/wake",
            count, total_bytes / 2**30,
        )

    def register_kv_caches(self, kv_caches: list[torch.Tensor]) -> None:
        """Register KV cache tensors under the 'kv_cache' tag."""
        total_bytes = 0
        for i, kv in enumerate(kv_caches):
            self.register_tensor("kv_cache", f"kv_cache_{i}", kv)
            total_bytes += kv.nelement() * kv.element_size()
        logger.info(
            "Registered %d KV cache tensors (%.2f GiB) for sleep/wake",
            len(kv_caches), total_bytes / 2**30,
        )

    def sleep(self, offload_tags: tuple[str, ...] | None = None) -> None:
        """Release GPU memory by backing up tensors to CPU.

        Args:
            offload_tags: Tags whose tensors should be backed up to CPU before
                releasing GPU storage. If None or empty, GPU storage is released
                without CPU backup (level 2 behavior — weights are discarded
                and must be reloaded from disk on wake_up).
        """
        if self._is_sleeping:
            logger.warning("XPUMemAllocator is already sleeping.")
            return

        offload_tags = offload_tags or ()
        freed_bytes = 0

        for tag, records in self._pools.items():
            for rec in records:
                if rec.tensor.untyped_storage().size() == 0:
                    continue  # Already released
                nbytes = rec.tensor.nelement() * rec.tensor.element_size()

                # Backup to CPU if this tag should be offloaded
                if tag in offload_tags:
                    rec.cpu_backup = rec.tensor.data.cpu()

                # Save metadata before releasing
                rec.original_storage_size = rec.tensor.untyped_storage().size()
                rec.original_shape = tuple(rec.tensor.shape)
                rec.device = rec.tensor.device

                # Release GPU storage — resize_(0) returns blocks to caching
                # allocator's free list, making them available for reuse
                rec.tensor.untyped_storage().resize_(0)
                freed_bytes += nbytes

        gc.collect()
        # NOTE: Do NOT call torch.xpu.empty_cache() — leaks UR handles with FSDP
        self._is_sleeping = True
        logger.info("XPU sleep freed %.2f GiB GPU memory", freed_bytes / 2**30)

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Restore GPU tensors from CPU backups.

        Args:
            tags: Which tags to restore. None means restore all.
        """
        if not self._is_sleeping:
            logger.warning("XPUMemAllocator is not sleeping.")
            return

        tags_to_wake = set(tags) if tags else set(self._pools.keys())
        restored_bytes = 0

        for tag in tags_to_wake:
            records = self._pools.get(tag, [])
            for rec in records:
                if rec.cpu_backup is not None:
                    # Restore storage size, then copy from CPU backup
                    rec.tensor.untyped_storage().resize_(rec.original_storage_size)
                    rec.tensor.copy_(rec.cpu_backup)
                    restored_bytes += (
                        rec.tensor.nelement() * rec.tensor.element_size()
                    )
                    rec.cpu_backup = None
                elif rec.original_storage_size > 0 and rec.tensor.untyped_storage().size() == 0:
                    # No backup — reallocate zeroed storage (for kv_cache at level 2)
                    rec.tensor.untyped_storage().resize_(rec.original_storage_size)
                    rec.tensor.zero_()
                    restored_bytes += (
                        rec.tensor.nelement() * rec.tensor.element_size()
                    )

        # Check if fully awake
        all_restored = all(
            rec.cpu_backup is None and rec.tensor.untyped_storage().size() > 0
            for records in self._pools.values()
            for rec in records
        )
        if all_restored:
            self._is_sleeping = False

        logger.info("XPU wake_up restored %.2f GiB GPU memory (tags=%s)",
                     restored_bytes / 2**30, tags)

    @contextmanager
    def use_memory_pool(self, tag: str):
        """Context manager to set the current tag for tensor tracking.

        On CUDA, CuMemAllocator uses MemPool to intercept allocations.
        On XPU, this is a no-op context that just sets the current tag.
        Actual tensor registration happens explicitly after model/cache loading.
        """
        prev_tag = self._current_tag
        self._current_tag = tag
        try:
            yield
        finally:
            self._current_tag = prev_tag

    def get_current_usage(self) -> int:
        """Return total GPU bytes tracked across all pools."""
        total = 0
        for records in self._pools.values():
            for rec in records:
                if rec.tensor.untyped_storage().size() > 0:
                    total += rec.tensor.nelement() * rec.tensor.element_size()
        return total

    def clear(self) -> None:
        """Clear all tracked tensors."""
        self._pools.clear()
        self._is_sleeping = False

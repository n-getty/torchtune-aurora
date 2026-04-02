# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Device-agnostic utilities for XPU (Intel Max Series GPU) and multi-backend
distributed training. These wrappers replace hardcoded ``torch.cuda.*`` calls
in recipes so the same recipe can run on CUDA, XPU, or other accelerators.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from torchtune.training._distributed import VALID_BACKENDS_FOR_MEMORY_STATS
log = logging.getLogger(__name__)

# Device types that support empty_cache()
_CACHEABLE_DEVICE_TYPES = frozenset(VALID_BACKENDS_FOR_MEMORY_STATS)


def _get_device_ns(device_type: str):
    """Return the torch device namespace (e.g. ``torch.cuda``, ``torch.xpu``)."""
    return getattr(torch, device_type)


def device_empty_cache(device: torch.device) -> None:
    """Clear the device memory cache. No-op for CPU and MPS.

    Replaces scattered ``torch.cuda.empty_cache()`` calls with a
    device-agnostic alternative.

    Args:
        device: The torch device whose cache should be cleared.
    """
    if device.type in _CACHEABLE_DEVICE_TYPES:
        _get_device_ns(device.type).empty_cache()


def device_record_memory_history(
    device: torch.device, enabled: Optional[bool] = True
) -> None:
    """Start or stop recording memory allocation history.

    On CUDA this delegates to ``torch.cuda.memory._record_memory_history``.
    On XPU it attempts the equivalent API if available. On unsupported
    devices it is a no-op.

    Args:
        device: The torch device.
        enabled: ``True`` to start recording, ``None``/``False`` to stop.
    """
    if device.type == "cuda":
        torch.cuda.memory._record_memory_history(enabled=enabled)
    elif device.type == "xpu":
        ns = _get_device_ns("xpu")
        if hasattr(ns, "memory") and hasattr(ns.memory, "_record_memory_history"):
            ns.memory._record_memory_history(enabled=enabled)
        else:
            log.debug(
                "torch.xpu.memory._record_memory_history not available; skipping"
            )


def supports_memory_stats(device: torch.device) -> bool:
    """Return whether the device supports peak memory tracking.

    Args:
        device: The torch device to check.

    Returns:
        ``True`` for cuda, xpu, npu, hpu; ``False`` otherwise.
    """
    return device.type in VALID_BACKENDS_FOR_MEMORY_STATS


def get_xpu_distributed_backend(
    device_type: str, offload_ops_to_cpu: bool = False
) -> str:
    """Return the correct distributed backend string for the given device.

    For XPU on Aurora this ensures ``"xccl"`` is returned (the oneCCL
    backend registered by IPEX). For other device types it delegates to
    ``torch.distributed.Backend.default_device_backend_map``.

    Args:
        device_type: One of ``"cuda"``, ``"xpu"``, ``"cpu"``, etc.
        offload_ops_to_cpu: If ``True``, returns a composite backend
            string (e.g. ``"xpu:xccl,cpu:gloo"``).

    Returns:
        Backend string suitable for ``init_process_group()``.
    """
    backend_map = {
        "cuda": "nccl",
        "xpu": "xccl",
        "npu": "hccl",
        "hpu": "hccl",
        "cpu": "gloo",
        "mps": "gloo",
    }

    # Prefer the runtime map if populated (IPEX registers the real backend there)
    if hasattr(dist.Backend, "default_device_backend_map"):
        runtime_backend = dist.Backend.default_device_backend_map.get(device_type)
        if runtime_backend is not None:
            backend = runtime_backend
        else:
            backend = backend_map.get(device_type, "gloo")
    else:
        backend = backend_map.get(device_type, "gloo")

    if offload_ops_to_cpu and device_type not in ("cpu", "mps"):
        return f"{device_type}:{backend},cpu:gloo"
    return backend


def init_xpu_process_group(backend: str, device_index: int = 0, **kwargs) -> None:
    """Initialize the distributed process group, with XPU-safe defaults.

    Two modes depending on ``ZE_AFFINITY_MASK``:

    1. **ZE_AFFINITY_MASK set** (multi-node standard): each rank sees only its
       tile as ``xpu:0``.  Do NOT pass ``device_id`` — it causes DataLoader
       worker deadlocks and is unnecessary with ring algorithms.
    2. **ZE_AFFINITY_MASK unset** (single-node): all 12 tiles visible.  Pass
       ``device_id=xpu:{device_index}`` so CCL's topology-aware path selects
       the right tile for ``ReduceOp.AVG``.

    Args:
        backend: Distributed backend string (e.g. ``"xccl"``, ``"nccl"``).
        device_index: XPU device index for this rank (default 0).
        **kwargs: Forwarded to ``torch.distributed.init_process_group``.
    """
    use_affinity_mask = bool(os.environ.get("ZE_AFFINITY_MASK", ""))
    # Multi-node detection: if WORLD_SIZE > LOCAL_WORLD_SIZE, we're multi-node.
    # Do NOT pass device_id on multi-node — it causes XCCL to eagerly init GPU
    # contexts that break DataLoader workers and sub-communicator creation.
    # (PRISM reference: do not pass device_id on XPU multi-node.)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    is_multi_node = world_size > local_world_size
    if (backend in ("xccl", "ccl") or "xccl" in backend) and "device_id" not in kwargs:
        if not use_affinity_mask and not is_multi_node:
            kwargs["device_id"] = torch.device(f"xpu:{device_index}")

    import datetime
    if "timeout" not in kwargs:
        kwargs["timeout"] = datetime.timedelta(minutes=10)
    dist.init_process_group(backend=backend, **kwargs)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""empty_cache() + FSDP leaks UR handles in Level Zero on XPU
(see docs/bugs/intel_xpu_resource_leak_bug_report.md). The override in
torchtune.dev.rl.distributed.device_empty_cache must be a no-op for XPU
devices, regardless of whether the surrounding code path forgot the guard.

Regressions in this function silently re-introduce ~70-iter crashes.
"""
from unittest.mock import patch

import torch

from torchtune.dev.rl.distributed import device_empty_cache


def test_xpu_device_empty_cache_does_not_call_torch_xpu_empty_cache():
    with patch("torch.xpu.empty_cache") as xpu_empty:
        device_empty_cache(torch.device("xpu", 0))
    assert xpu_empty.call_count == 0, (
        "device_empty_cache must NEVER call torch.xpu.empty_cache — "
        "FSDP + empty_cache leaks UR handles on Aurora XPU."
    )


def test_cpu_device_is_a_noop_too():
    # Sanity: original delegates to torchtune.training.device_empty_cache,
    # which is a no-op for CPU. We just make sure it doesn't blow up.
    device_empty_cache(torch.device("cpu"))


def test_cuda_path_delegates_to_original():
    # The XPU override should still allow CUDA-equivalent behavior on CUDA
    # builds. On a CPU-only test box torch.cuda.empty_cache exists but is a
    # no-op; we just ensure the call dispatches without raising.
    with patch(
        "torchtune.dev.rl.distributed._orig_device_empty_cache"
    ) as orig:
        device_empty_cache(torch.device("cuda", 0))
    orig.assert_called_once()

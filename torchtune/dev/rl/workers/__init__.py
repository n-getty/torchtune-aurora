# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CUDA/Ray async worker stack — NOT MAINTAINED for Aurora/XPU.
# All modules under this package hardcode `device_type="cuda"`, NCCL backend,
# and `torch.cuda.empty_cache()`. Importing them on Aurora may pull CUDA
# initialization through the Ray actor wiring. To prevent accidental imports
# from the dense-GRPO XPU code paths, this package is gated behind an opt-in.
#
# Set TORCHTUNE_ENABLE_RAY=1 to use the Ray async recipe.
import os as _os

if not _os.environ.get("TORCHTUNE_ENABLE_RAY"):
    raise ImportError(
        "torchtune.dev.rl.workers is gated behind TORCHTUNE_ENABLE_RAY=1 "
        "because its modules are CUDA/NCCL-only and not supported on Aurora/XPU. "
        "Set TORCHTUNE_ENABLE_RAY=1 to opt in (only for non-Aurora environments)."
    )

from .datacollectors import SyncLLMCollector
from .metric_logger import MetricLoggerWorker
from .parameter_servers import VLLMParameterServer
from .postprocessing import PostProcessingWorker
from .trainers import TrainingWorker
from .weight_updaters import VLLMHFWeightUpdateReceiver

__all__ = [
    "SyncLLMCollector",
    "MetricLoggerWorker",
    "VLLMParameterServer",
    "PostProcessingWorker",
    "TrainingWorker",
    "VLLMHFWeightUpdateReceiver",
]

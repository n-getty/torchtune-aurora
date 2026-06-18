# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test for the LoRA-GRPO vllm_mode gate.

The standalone LoRA-GRPO recipe accepts ``vllm_mode`` in {"server", "colocate"}
and rejects everything else. The check lives in
``torchtune.dev.rl.lora_helpers.validate_vllm_mode`` so it is unit-testable on a
login node without importing the recipe (which pulls torchao + XPU backends).
"""
from __future__ import annotations

import pytest

from torchtune.dev.rl.lora_helpers import validate_vllm_mode


@pytest.mark.parametrize("mode", ["server", "colocate"])
def test_supported_modes_pass(mode):
    # Should not raise.
    validate_vllm_mode(mode)


@pytest.mark.parametrize(
    "mode", ["dedicated_rank", "colocate_sleep", "colocate_ray", "", "SERVER", "http"]
)
def test_unsupported_modes_raise(mode):
    with pytest.raises(ValueError):
        validate_vllm_mode(mode)


def test_error_message_names_the_mode():
    with pytest.raises(ValueError, match="dedicated_rank"):
        validate_vllm_mode("dedicated_rank")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU drift-guard for vllm_backend._lora_engine_kwargs.

``_lora_engine_kwargs(cfg)`` is spread into the vLLM ``LLM(...)`` kwargs at
every engine-init site (TP=1, TP>1, dedicated, colocate). If it silently
returns ``{}`` when LoRA was requested, vLLM boots without adapter support and
the LoRA-GRPO publish path fails downstream with a confusing error. It is a
pure cfg→dict function, so it is exactly the kind of thing a fast CPU test
should pin.

The module-level ``import`` of ``torchtune.dev.rl.vllm_backend`` is safe on a
login node: all vLLM/XPU imports inside that module are deferred into function
bodies, so importing the module does not require a GPU.
"""
from __future__ import annotations

from torchtune.dev.rl.vllm_backend import _lora_engine_kwargs


def test_no_vllm_block_returns_empty():
    # A recipe with no `vllm:` config block must be unaffected.
    assert _lora_engine_kwargs({}) == {}


def test_enable_lora_false_returns_empty():
    assert _lora_engine_kwargs({"vllm": {"enable_lora": False}}) == {}


def test_enable_lora_absent_returns_empty():
    # `vllm:` present but no enable_lora key → default False → empty.
    assert _lora_engine_kwargs({"vllm": {"max_loras": 4}}) == {}


def test_enable_lora_true_uses_explicit_values():
    out = _lora_engine_kwargs(
        {"vllm": {"enable_lora": True, "max_lora_rank": 32, "max_loras": 4}}
    )
    assert out == {"enable_lora": True, "max_lora_rank": 32, "max_loras": 4}


def test_enable_lora_true_uses_defaults():
    out = _lora_engine_kwargs({"vllm": {"enable_lora": True}})
    assert out == {"enable_lora": True, "max_lora_rank": 16, "max_loras": 2}


def test_rank_and_count_coerced_to_int():
    # YAML can surface these as strings; the engine needs ints.
    out = _lora_engine_kwargs(
        {"vllm": {"enable_lora": True, "max_lora_rank": "8", "max_loras": "3"}}
    )
    assert out["max_lora_rank"] == 8 and isinstance(out["max_lora_rank"], int)
    assert out["max_loras"] == 3 and isinstance(out["max_loras"], int)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

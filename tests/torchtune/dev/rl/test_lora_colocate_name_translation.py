# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test for the shared tune->HF LoRA name translation.

Both the server merged publish path and the colocate sync path route base-weight
param names through ``tune_lora_name_to_hf`` so they cannot drift. This pins:
  - every known LoRA module path maps to the expected HF name,
  - FSDP / activation-checkpointing wrapper prefixes are stripped,
  - unknown / non-matching names return None (caller skips + logs, not crash).
"""
from __future__ import annotations

import pytest

from torchtune.dev.rl.lora_helpers import _TUNE_MODULE_TO_HF, tune_lora_name_to_hf


@pytest.mark.parametrize("module_path,hf_module", list(_TUNE_MODULE_TO_HF.items()))
def test_every_known_module_maps(module_path, hf_module):
    tune_name = f"layers.7.{module_path}.weight"
    assert tune_lora_name_to_hf(tune_name) == f"model.layers.7.{hf_module}.weight"


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "_fsdp_wrapped_module.",
        "_checkpoint_wrapped_module.",
        "_fsdp_wrapped_module._checkpoint_wrapped_module.",
        "model.",
    ],
)
def test_wrapper_prefixes_stripped(prefix):
    tune_name = f"{prefix}layers.0.attn.q_proj.weight"
    assert tune_lora_name_to_hf(tune_name) == "model.layers.0.self_attn.q_proj.weight"


@pytest.mark.parametrize(
    "bad_name",
    [
        "tok_embeddings.weight",          # not a layer param
        "norm.weight",                    # final norm
        "output.weight",                  # lm_head
        "layers.0.attn.unknown.weight",   # unknown module path
        "layers.0.attn.q_proj.bias",      # not a .weight
        "garbage",
    ],
)
def test_non_lora_names_return_none(bad_name):
    assert tune_lora_name_to_hf(bad_name) is None

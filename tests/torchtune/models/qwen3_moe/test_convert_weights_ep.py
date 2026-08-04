# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtune.models.qwen3_moe._convert_weights import qwen3_moe_hf_to_tune


def test_hf_to_tune_filters_interleaved_ep_experts(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTUNE_MOE_CHECKPOINT_EP_DEGREE", "2")
    monkeypatch.setenv("TORCHTUNE_MOE_CHECKPOINT_EP_RANK", "1")
    state_dict = {}
    for expert in range(4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            state_dict[
                f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            ] = torch.full((1, 1), float(expert))

    converted = qwen3_moe_hf_to_tune(state_dict, num_experts=4)

    for projection in ("gate_proj", "up_proj", "down_proj"):
        torch.testing.assert_close(
            converted[f"layers.0.mlp.experts.{projection}"].flatten(),
            torch.tensor([1.0, 3.0]),
        )


def test_hf_to_tune_filters_pipeline_stage_state(monkeypatch) -> None:
    monkeypatch.setenv("TORCHTUNE_MOE_CHECKPOINT_PIPELINE_DEGREE", "2")
    monkeypatch.setenv("TORCHTUNE_MOE_CHECKPOINT_PIPELINE_STAGE", "1")
    monkeypatch.setenv("TORCHTUNE_MOE_CHECKPOINT_PIPELINE_SPLIT_LAYER", "1")
    state_dict = {
        "model.embed_tokens.weight": torch.ones(2, 2),
        "model.layers.0.input_layernorm.weight": torch.ones(2),
        "model.layers.1.input_layernorm.weight": torch.ones(2),
        "model.norm.weight": torch.ones(2),
        "lm_head.weight": torch.ones(2, 2),
    }

    converted = qwen3_moe_hf_to_tune(state_dict, num_experts=4)

    assert set(converted) == {
        "layers.1.sa_norm.scale",
        "norm.scale",
        "output.weight",
    }

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
from torch import nn

from torchtune.models.qwen3_moe._pipeline import build_qwen3_moe_pipeline_stage
from torchtune.modules.transformer import TransformerDecoder


class _ResidualLinear(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden + self.proj(hidden)

    def caches_are_enabled(self) -> bool:
        return False


def _tiny_decoder() -> TransformerDecoder:
    return TransformerDecoder(
        tok_embeddings=nn.Embedding(17, 8),
        layers=[_ResidualLinear(8) for _ in range(4)],
        max_seq_len=16,
        num_heads=2,
        head_dim=4,
        norm=nn.LayerNorm(8),
        output=nn.Linear(8, 17, bias=False),
    )


def test_pipeline_stages_match_decoder_forward_and_gradients() -> None:
    reference = _tiny_decoder()
    pipelined = copy.deepcopy(reference)
    stage0 = build_qwen3_moe_pipeline_stage(pipelined, stage_index=0, split_layer=2)
    stage1 = build_qwen3_moe_pipeline_stage(pipelined, stage_index=1, split_layer=2)
    tokens = torch.randint(0, 17, (2, 5))

    reference_output = reference(tokens)
    pipeline_output = stage1(stage0(tokens))
    torch.testing.assert_close(pipeline_output, reference_output)

    reference_output.sum().backward()
    pipeline_output.sum().backward()
    pipeline_grads = dict(stage0.named_parameters()) | dict(stage1.named_parameters())
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(pipeline_grads[name].grad, parameter.grad)


def test_stage_one_retains_global_layer_checkpoint_keys() -> None:
    stage1 = build_qwen3_moe_pipeline_stage(
        _tiny_decoder(), stage_index=1, split_layer=2
    )
    keys = stage1.state_dict()
    assert "layers.2.proj.weight" in keys
    assert "layers.3.proj.weight" in keys
    assert "layers.0.proj.weight" not in keys


def test_stage_one_can_return_hidden_states_for_linear_loss() -> None:
    stage1 = build_qwen3_moe_pipeline_stage(
        _tiny_decoder(), stage_index=1, split_layer=2
    )
    hidden = torch.randn(2, 5, 8)

    stage1.skip_output_layer = True

    assert stage1(hidden).shape == hidden.shape

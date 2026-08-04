# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
import os

import torch
from torch import nn

from torchtune.modules.transformer import TransformerDecoder


class Qwen3MoePipelineStage(nn.Module):
    def __init__(
        self,
        *,
        layers: dict[str, nn.Module],
        hidden_dim: int,
        tok_embeddings: nn.Module | None = None,
        norm: nn.Module | None = None,
        output: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        self.tok_embeddings = tok_embeddings
        self.norm = norm
        self.output = output
        self.hidden_dim = hidden_dim
        self.skip_output_layer = False

    @property
    def is_first(self) -> bool:
        return self.tok_embeddings is not None

    @property
    def is_last(self) -> bool:
        return self.norm is not None and self.output is not None

    def forward(self, stage_input: torch.Tensor) -> torch.Tensor:
        hidden = self.tok_embeddings(stage_input) if self.is_first else stage_input
        trace_pipeline = os.environ.get("TORCHTUNE_MOE_PIPELINE_TRACE", "0")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        trace_rank = trace_pipeline == "all" or (
            trace_pipeline == "1" and rank in (0, 8)
        )
        for layer_index, layer in self.layers.items():
            if trace_rank:
                print(
                    f"PP_LAYER_BEGIN rank={rank} "
                    f"layer={layer_index} alloc={torch.xpu.memory_allocated()} "
                    f"reserved={torch.xpu.memory_reserved()}",
                    flush=True,
                )
            hidden = layer(hidden)
            if trace_rank:
                print(
                    f"PP_LAYER_END rank={rank} "
                    f"layer={layer_index} alloc={torch.xpu.memory_allocated()} "
                    f"reserved={torch.xpu.memory_reserved()}",
                    flush=True,
                )
        if self.is_last:
            hidden = self.norm(hidden)
            if not self.skip_output_layer:
                hidden = self.output(hidden).float()
        return hidden


def build_qwen3_moe_pipeline_stage(
    model: TransformerDecoder,
    *,
    stage_index: int,
    split_layer: int,
) -> Qwen3MoePipelineStage:
    if stage_index not in (0, 1):
        raise ValueError("stage_index must be 0 or 1")
    if not 0 < split_layer < len(model.layers):
        raise ValueError(
            f"split_layer must be between 1 and {len(model.layers) - 1}, "
            f"got {split_layer}"
        )
    layer_indices: Iterable[int]
    if stage_index == 0:
        layer_indices = range(split_layer)
    else:
        layer_indices = range(split_layer, len(model.layers))
    layers = {str(index): model.layers[index] for index in layer_indices}
    return Qwen3MoePipelineStage(
        layers=layers,
        hidden_dim=model.tok_embeddings.embedding_dim,
        tok_embeddings=model.tok_embeddings if stage_index == 0 else None,
        norm=model.norm if stage_index == 1 else None,
        output=model.output if stage_index == 1 else None,
    )

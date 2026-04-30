# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchtune.models.qwen3_moe._experts import GroupedExpertsHF
from torchtune.modules.moe._parallelism import ExpertParallel


def qwen3_moe_ep_plan(model: nn.Module) -> dict[str, ParallelStyle]:
    """Build the Expert Parallelism plan for Qwen3 MoE (e.g. 30B-A3B).

    Maps each MoE layer's ``experts`` (``GroupedExpertsHF``) to ``ExpertParallel``
    for AllGather + ReduceScatter token dispatch. Iterates the live module graph
    so AC-wrapped paths (``layers.{i}._checkpoint_wrapped_module.mlp.experts``)
    are matched alongside un-wrapped paths (``layers.{i}.mlp.experts``).

    Note vs ``gemma4_ep_plan``:
      * Gemma4: experts at ``layers.{i}.moe_block.experts`` (additive MoE alongside
        a dense MLP), uses ``GroupedExperts`` (loop or grouped_mm).
      * Qwen3 MoE: experts at ``layers.{i}.mlp.experts`` (MoE replaces the dense
        MLP entirely), uses ``GroupedExpertsHF`` (HF-native [E, out, in] layout).

    Args:
        model: Instantiated Qwen3 MoE model (``TransformerDecoder`` whose layers
            are ``Qwen3MoeTransformerLayer`` with ``mlp`` set to a ``MoE`` block).

    Returns:
        Dictionary mapping module paths to ``ParallelStyle`` instances, suitable
        for ``parallelize_module(model, ep_mesh, plan)``.
    """
    plan: dict[str, ParallelStyle] = {}
    for name, module in model.named_modules():
        if name.endswith(".experts") and isinstance(module, GroupedExpertsHF):
            plan[name] = ExpertParallel()
    return plan

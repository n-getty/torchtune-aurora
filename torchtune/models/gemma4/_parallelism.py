# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torch.distributed.tensor.parallel.style import ParallelStyle

from torchtune.modules.moe._parallelism import ExpertParallel, NoParallel


def gemma4_ep_plan(model: nn.Module) -> dict[str, ParallelStyle]:
    """Build the Expert Parallelism plan for Gemma4 26B-A4B.

    Maps each MoE layer's ``experts`` (``GroupedExperts``) to ``ExpertParallel``
    for All-to-All token dispatch, and wraps the router gate with ``NoParallel``
    to ensure it is registered as a DTensor on the EP mesh (required for gradient
    norm clipping and fused optimizer compatibility).

    Usage::

        from torch.distributed.tensor.parallel import parallelize_module
        from torchtune.models.gemma4._parallelism import gemma4_ep_plan

        ep_plan = gemma4_ep_plan(model)
        parallelize_module(model, ep_mesh, ep_plan)

    Args:
        model: Instantiated Gemma4 model (``TransformerDecoder`` with
            ``Gemma4TransformerLayer`` layers). MoE layers must have a
            ``moe_block`` attribute (``MoE`` instance).

    Returns:
        Dictionary mapping module paths to ``ParallelStyle`` instances,
        suitable for passing to ``parallelize_module``.
    """
    from torchtune.modules.moe.experts import GroupedExperts

    plan: dict[str, ParallelStyle] = {}
    # Build plan by walking the actual named_modules to handle the case where
    # activation checkpointing has added a _checkpoint_wrapped_module layer.
    # Without AC: layers.{i}.moe_block.experts
    # With AC:    layers.{i}._checkpoint_wrapped_module.moe_block.experts
    for name, module in model.named_modules():
        if name.endswith(".experts") and isinstance(module, GroupedExperts):
            # Route expert computation via All-to-All: each EP rank processes
            # its local experts (128 / ep_degree = 32 for EP=4)
            plan[name] = ExpertParallel()
    return plan

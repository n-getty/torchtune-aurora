# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Optional

import torch
from torch import nn, Tensor
from torch.distributed.tensor import DTensor
from torch.nn.utils.clip_grad import _no_grad, _tensor_or_tensors
from torch.utils._foreach_utils import _device_has_foreach_support, _has_foreach_support
from torchtune.utils._logging import deprecated


@deprecated(msg="Please use `scale_grads_` instead.")
def scale_grads(model: nn.Module, scaler: torch.Tensor) -> None:
    """
    Utility to scale the gradients of a model.
    This is useful for gradient accumulation where we want to normalize
    the gradients by the total number of tokens seen.

    Inputs:
        model (nn.Module): model whose gradients should be scaled
        scaler (torch.Tensor): scaling factor to apply to the gradients

    Outputs:
        None (grad fields are modified in place)
    """
    device = None
    for p in model.parameters():
        # First ensure scaler is on the same device as the model
        if not device:
            device = p.device
            scaler = scaler.to(device)
        if p.grad is not None:
            p.grad *= scaler


@_no_grad
def scale_grads_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    r"""Scale gradients of iterable parameters.

    This function is equivalent to :func:`torch.mul_` applied to each parameter.
    Gradients are modified in-place, multiplying by specified scaler.

    Args:
        parameters (_tensor_or_tensors): an iterable of Tensors or a
            single Tensor that will have gradients scaled
        scaler (torch.Tensor): multiplier to scale gradients
        foreach (Optional[bool]): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    _scale_grad_(parameters, scaler, foreach)


@_no_grad
def scale_grads_for_native_ep_(
    parameters: _tensor_or_tensors,
    native_expert_parameter_ids: set[int],
    distributed_scaler: torch.Tensor,
    native_expert_scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    distributed_parameters = [
        parameter
        for parameter in parameters
        if id(parameter) not in native_expert_parameter_ids
    ]
    native_expert_parameters = [
        parameter
        for parameter in parameters
        if id(parameter) in native_expert_parameter_ids
    ]
    if not distributed_parameters or not native_expert_parameters:
        raise RuntimeError(
            "native EP gradient scaling requires distributed and expert parameters"
        )
    _scale_grad_(distributed_parameters, distributed_scaler, foreach)
    _scale_grad_(native_expert_parameters, native_expert_scaler, foreach)


def _group_tensors_by_device(
    tensors: list[torch.Tensor],
) -> dict[tuple, list[Tensor]]:
    # Key on (device, mesh id) rather than just device: DTensors sharing a
    # physical device but living on DIFFERENT DTensor meshes (e.g. Expert
    # Parallelism's solo 1-rank FSDP2 mesh for expert params vs the root
    # dp_shard mesh for everything else) cannot be passed to torch._foreach_mul_
    # together -- "Could not run pointwise computation across different mesh".
    # Grouping by mesh id keeps foreach batching within a single mesh while
    # remaining a no-op (single group per device) for the non-DTensor/
    # single-mesh case every other recipe uses.
    ret = defaultdict(list)
    for tensor in tensors:
        mesh = tensor.device_mesh if isinstance(tensor, DTensor) else None
        ret[(tensor.device, id(mesh))].append(tensor)

    return ret


@_no_grad
def _scale_grad_(
    parameters: _tensor_or_tensors,
    scaler: torch.Tensor,
    foreach: Optional[bool] = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    grouped_grads = _group_tensors_by_device(grads)

    for (device, _mesh_id), device_grads in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, scaler.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            scaler_device = scaler.to(device)
            for g in device_grads:
                if isinstance(g, DTensor):
                    g[:] = g * scaler_device
                else:
                    g.mul_(scaler_device)

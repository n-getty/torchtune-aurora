# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchtune.training._grad_scaler import scale_grads_for_native_ep_


def test_scale_grads_for_native_ep_uses_distinct_normalization() -> None:
    distributed_parameter = torch.nn.Parameter(torch.ones(2))
    native_expert_parameter = torch.nn.Parameter(torch.ones(2))
    distributed_parameter.grad = torch.full((2,), 3.0)
    native_expert_parameter.grad = torch.full((2,), 3.0)

    scale_grads_for_native_ep_(
        [distributed_parameter, native_expert_parameter],
        {id(native_expert_parameter)},
        torch.tensor(0.5),
        torch.tensor(0.125),
        foreach=False,
    )

    torch.testing.assert_close(distributed_parameter.grad, torch.full((2,), 1.5))
    torch.testing.assert_close(native_expert_parameter.grad, torch.full((2,), 0.375))


@pytest.mark.parametrize("native_ids", [set(), {-1}])
def test_scale_grads_for_native_ep_requires_both_groups(native_ids: set[int]) -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.ones(1)
    if native_ids:
        native_ids = {id(parameter)}

    with pytest.raises(RuntimeError, match="distributed and expert parameters"):
        scale_grads_for_native_ep_(
            [parameter],
            native_ids,
            torch.tensor(1.0),
            torch.tensor(1.0),
            foreach=False,
        )


def test_scale_grads_for_native_ep_rejects_stale_parameter_ids() -> None:
    distributed_parameter = torch.nn.Parameter(torch.ones(1))
    native_expert_parameter = torch.nn.Parameter(torch.ones(1))
    replacement_expert_parameter = torch.nn.Parameter(torch.ones(1))

    with pytest.raises(RuntimeError, match="distributed and expert parameters"):
        scale_grads_for_native_ep_(
            [distributed_parameter, replacement_expert_parameter],
            {id(native_expert_parameter)},
            torch.tensor(1.0),
            torch.tensor(1.0),
            foreach=False,
        )

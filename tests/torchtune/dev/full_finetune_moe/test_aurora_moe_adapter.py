# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtune.modules.moe.moe import _aurora_moe_expert_ids


def test_aurora_moe_expert_ids_map_interleaved_to_contiguous_ownership():
    ep_degree = 8
    local_experts = 16
    global_experts = ep_degree * local_experts
    selected_experts = torch.arange(global_experts).reshape(ep_degree, local_experts)

    remapped = _aurora_moe_expert_ids(
        selected_experts, ep_degree, local_experts
    )

    for ep_rank in range(ep_degree):
        interleaved_ids = torch.arange(ep_rank, global_experts, ep_degree)
        expected_ids = torch.arange(
            ep_rank * local_experts, (ep_rank + 1) * local_experts
        )
        torch.testing.assert_close(
            _aurora_moe_expert_ids(interleaved_ids, ep_degree, local_experts),
            expected_ids,
        )
    torch.testing.assert_close(
        remapped.flatten().sort().values, torch.arange(global_experts)
    )

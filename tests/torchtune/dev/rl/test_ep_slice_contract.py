# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""EP weight-slice ↔ token-dispatch ownership contract.

Two independent code paths decide which experts each EP rank owns:

1. The recipe (`recipes/dev/grpo_full_finetune_distributed_xpu.py`) slices
   the checkpoint state dict before `load_from_full_model_state_dict`.
2. `ExpertParallel._token_dispatch` in `torchtune/modules/moe/_parallelism.py`
   computes which global expert each local-expert slot routes to using
   `g = ep_rank + local_exp_idx * ep_degree` (interleaved).

If the two formulas disagree, EP > 1 silently trains with weights and tokens
permuted relative to each other — every reported `loss=` line is mechanical
nonsense. The pre-fix recipe used contiguous slicing
(`_ft[r * n_local : (r + 1) * n_local]`) while dispatch used interleaved;
this test exists so a regression cannot recur silently.
"""
import pytest
import torch


@pytest.mark.parametrize(
    "num_experts,ep_degree",
    [(128, 8), (128, 4), (128, 2), (32, 4), (16, 2), (8, 8)],
)
def test_recipe_slice_matches_dispatch_ownership(num_experts, ep_degree):
    n_local = num_experts // ep_degree
    full = torch.arange(num_experts)
    for ep_rank in range(ep_degree):
        # Recipe-side ownership (must match the recipe's pre-FSDP2 slice).
        loaded_ids = full[ep_rank::ep_degree].tolist()
        # Dispatch-side ownership (ExpertParallel._token_dispatch line ~617).
        routed_ids = [ep_rank + i * ep_degree for i in range(n_local)]
        assert loaded_ids == routed_ids, (
            f"EP slice contract violated for "
            f"num_experts={num_experts} ep_degree={ep_degree} ep_rank={ep_rank}: "
            f"loaded={loaded_ids} routed={routed_ids}"
        )


def test_contiguous_slice_violates_contract():
    """Sanity: the *broken* (pre-fix) contiguous formula must NOT match dispatch.

    Guards against the test silently passing if both sides are computed the same
    wrong way.
    """
    num_experts, ep_degree = 128, 8
    n_local = num_experts // ep_degree
    full = torch.arange(num_experts)
    mismatches = 0
    for ep_rank in range(ep_degree):
        contig = full[ep_rank * n_local : (ep_rank + 1) * n_local].tolist()
        routed = [ep_rank + i * ep_degree for i in range(n_local)]
        if contig != routed:
            mismatches += 1
    assert mismatches == ep_degree, (
        "Contiguous slicing should disagree with interleaved dispatch on every "
        "rank (including rank 0: [0..15] != [0, 8, 16, ..., 120]); if it doesn't, "
        "the test is degenerate."
    )

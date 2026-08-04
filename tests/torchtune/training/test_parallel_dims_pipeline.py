# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtune.training._distributed import ParallelDims


def test_pipeline_dimension_infers_node_local_ep_shard() -> None:
    dims = ParallelDims(
        dp_replicate=1,
        dp_shard=-1,
        tp=1,
        cp=1,
        world_size=16,
        ep=8,
        pp=2,
    )

    assert dims.dp_shard == 8
    assert dims.ep_enabled
    assert dims.pp_enabled
    assert dims.non_data_parallel_size == 2


def test_pipeline_dimension_participates_in_world_size_validation() -> None:
    with pytest.raises(AssertionError, match="WORLD_SIZE"):
        ParallelDims(
            dp_replicate=1,
            dp_shard=8,
            tp=1,
            cp=1,
            world_size=8,
            ep=8,
            pp=2,
        )

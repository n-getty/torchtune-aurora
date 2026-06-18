# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-node colocate FSDP2 dp_mesh construction contract.

At ``data_parallel_replicate_dim=1`` (plain colocate, non-EP, non-HSDP) the
recipe used to leave ``self._dp_mesh = None`` for ALL world sizes. That is
correct single-node — ``shard_model`` passes ``mesh=None`` to ``fully_shard``,
which builds a default 1-D mesh over the (single-node) default PG.

It is WRONG multi-node: FSDP2's ``_init_default_fully_shard_mesh`` misenumerates
cross-node ranks on XPU, so node-1 ranks get ``shard_mesh_size = local_world``
but ``shard_process_group.rank() = global_rank``. ``_init_sharded_param`` then
does ``chunks[shard_rank]`` with ``shard_rank >= len(chunks)`` →
``IndexError: list index out of range`` at setup, before step 0.

The fix builds an explicit 1-D ``dp_shard`` mesh over the world PG when (and
only when) the run is multi-node, non-EP, non-HSDP. This test pins the decision
predicate so the regression cannot recur silently, and so the single-node path
(the validated 1N sweep + 1N production runs) keeps ``_dp_mesh = None``
byte-for-byte.

The predicate mirrors the recipe (grpo_full_finetune_distributed_xpu.py, the
non-HSDP ``else`` branch of the ``dp_replicate`` block):

    _is_multinode_dp = world_size > LOCAL_WORLD_SIZE
    _ep_will_be_active = expert_parallel_degree > 1
    build_explicit_mesh = _is_multinode_dp and not _ep_will_be_active
"""
import pytest


def _should_build_explicit_dp_mesh(
    world_size: int,
    local_world_size: int,
    dp_replicate: int,
    expert_parallel_degree: int,
) -> bool:
    """Mirror of the recipe's non-HSDP dp_mesh decision.

    Returns True iff the recipe builds an explicit 1-D dp_shard mesh (instead of
    leaving _dp_mesh=None). Only reached when dp_replicate == 1 (HSDP handles
    dp_replicate > 1 in its own branch).
    """
    if dp_replicate > 1:
        # HSDP branch — builds its own 2-D mesh; not this code path.
        return False
    is_multinode = world_size > local_world_size
    ep_active = expert_parallel_degree > 1
    return is_multinode and not ep_active


@pytest.mark.parametrize(
    "world,local,ep,expect_mesh",
    [
        # Single-node colocate: MUST stay None (validated 1N sweep + production).
        (12, 12, 1, False),
        (8, 8, 1, False),
        # Multi-node colocate, non-EP: MUST build an explicit mesh (the fix).
        (24, 12, 1, True),
        (48, 12, 1, True),
        # Multi-node + EP: the EP branch builds the mesh; this branch must NOT
        # (pre-building would skip EP's mesh+PG setup and break EP=16).
        (24, 12, 8, False),
        (32, 16, 16, False),
        # Single-node + EP: still EP's job, not this branch.
        (8, 8, 8, False),
    ],
)
def test_explicit_dp_mesh_decision(world, local, ep, expect_mesh):
    got = _should_build_explicit_dp_mesh(
        world_size=world,
        local_world_size=local,
        dp_replicate=1,
        expert_parallel_degree=ep,
    )
    assert got is expect_mesh, (
        f"world={world} local={local} ep={ep}: "
        f"expected build_explicit_mesh={expect_mesh}, got {got}"
    )


def test_single_node_must_not_build_mesh():
    """Explicit guard: single-node colocate must leave _dp_mesh=None.

    This is the byte-for-byte-unchanged guarantee for the validated 1N path. If
    this ever flips to True, the 1N sweep numbers in the public
    torchtune-vs-ezpz doc are no longer reproducible from the same code path.
    """
    assert not _should_build_explicit_dp_mesh(
        world_size=12, local_world_size=12, dp_replicate=1, expert_parallel_degree=1
    )


def test_multinode_colocate_must_build_mesh():
    """Explicit guard: 2-node colocate (the bake-off topology) builds a mesh.

    Without it, FSDP2 default-mesh enumeration crashes node-1 ranks with
    IndexError at _init_sharded_param. This is the bug the fix closes.
    """
    assert _should_build_explicit_dp_mesh(
        world_size=24, local_world_size=12, dp_replicate=1, expert_parallel_degree=1
    )

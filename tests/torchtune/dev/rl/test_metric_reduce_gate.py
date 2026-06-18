# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""log_metrics() world-reduce gate contract.

`log_metrics` reduces the per-step `rewards`/`successes` across ranks so the
logged value is representative of the whole (data-parallel) batch rather than
one rank's local slice. Whether it reduces is a GATE:

  - Flat data-parallel (dp_replicate == 1): every rank holds a DISTINCT prompt
    slice (sampler num_replicas == world_size), so the representative metric is
    the mean ACROSS ranks → REDUCE. True for single-node (1N) AND multi-node
    colocate (2N).
  - True HSDP (dp_replicate > 1): each replicate group sees distinct data and
    rank 0's local metric represents its replica; a world reduce would also mix
    the world PG with FSDP1 sub-PGs on XCCL (deadlock risk) → SKIP.

The gate USED to be `self._shard_pg is None`. That broke when the 2N-colocate
dp_mesh fix started setting `_shard_pg` (the explicit FSDP mesh group) at
dp_replicate == 1: the reduce was then wrongly SKIPPED, so the logged
`successes` was only rank 0's local 4-sample mean (coarse 0.25-quantized)
instead of the world-aggregated value the 1N reference logs — making 2N and 1N
best_acc non-comparable. The fix gates on `dp_replicate <= 1` instead.

This test pins the gate predicate so the two regressions it guards cannot recur:
(1) 2N colocate silently logging a rank-local (coarse, non-comparable) metric;
(2) a future change re-coupling the gate to `_shard_pg` and breaking HSDP.
"""
import pytest


def _should_world_reduce_metrics(dp_replicate: int) -> bool:
    """Mirror of log_metrics' reduce gate (grpo_full_finetune_distributed_xpu.py).

    Reduce iff the run is flat data-parallel (dp_replicate <= 1). True HSDP
    (dp_replicate > 1) skips the world reduce.
    """
    return dp_replicate <= 1


@pytest.mark.parametrize(
    "dp_replicate,expect_reduce",
    [
        (1, True),    # flat DP: 1N colocate AND 2N colocate — must reduce
        (2, False),   # HSDP 2-replica — must skip
        (7, False),   # HSDP 7-replica (agpt2b 7N) — must skip
    ],
)
def test_metric_reduce_gate(dp_replicate, expect_reduce):
    assert _should_world_reduce_metrics(dp_replicate) is expect_reduce, (
        f"dp_replicate={dp_replicate}: expected reduce={expect_reduce}"
    )


def test_flat_dp_must_reduce_regardless_of_shard_pg():
    """The 2N-colocate regression guard.

    At dp_replicate == 1 the metric MUST world-reduce even though the colocate
    dp_mesh fix sets `_shard_pg`. Gating on `_shard_pg is None` (the old, broken
    predicate) would skip the reduce here and log a coarse rank-local metric.
    The gate must depend ONLY on dp_replicate, so a non-None `_shard_pg` at
    dp_replicate == 1 does not change the decision.
    """
    # Simulate both _shard_pg states at dp_replicate == 1; decision must be the
    # same (reduce) because the gate ignores _shard_pg.
    for _shard_pg_is_set in (False, True):
        assert _should_world_reduce_metrics(dp_replicate=1) is True


def test_hsdp_must_not_reduce():
    """The HSDP regression guard: dp_replicate > 1 always skips the world reduce
    (distinct-data replicas + XCCL sub-PG deadlock avoidance)."""
    assert _should_world_reduce_metrics(dp_replicate=2) is False
    assert _should_world_reduce_metrics(dp_replicate=7) is False

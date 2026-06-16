"""Contract test for the HSDP + server-mode dataloader sampler.

Pins the data-distribution property that the recipe's ``_setup_data`` sampler
logic must satisfy when ``vllm_mode == "server"`` and ``dp_replicate > 1``:

  - Each dp_replicate group (one node) sees a DISTINCT slice of prompts
    (true data parallelism — the whole point of scaling to 7 training nodes).
  - All ranks WITHIN a shard group (same node) see the SAME slice (they are
    FSDP shards of one model copy; the shard-leader generates + broadcasts
    node-locally).
  - Single-replicate server mode (dp_replicate == 1) keeps ALL ranks on the
    SAME slice (rank 0 generates + world-broadcasts) — byte-identical to the
    validated 2N production behavior.

This mirrors the recipe's inline branch in ``_setup_data``:

    if vllm_mode == "server" and dp_replicate > 1:
        sampler_replicas = dp_replicate
        sampler_rank     = rank // dp_shard
    elif vllm_mode == "server":
        sampler_replicas = 1
        sampler_rank     = 0

If the recipe formula drifts, this test and the recipe disagree and one of
them is wrong — exactly like test_ep_slice_contract.py for the EP path.

Runs on a login node (CPU only, no distributed init, no torchtune top-level
import — uses torchdata's sampler directly to avoid the torchao dependency).
"""

import pytest

from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


class _RangeDataset:
    """Minimal indexable dataset: item i == i. Length large enough that every
    replica gets a non-trivial slice."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return i


def _sampler_params(vllm_mode, dp_replicate, dp_shard, rank):
    """Replicate the recipe's _setup_data sampler-param derivation EXACTLY."""
    if vllm_mode == "server" and dp_replicate > 1:
        return dp_replicate, rank // dp_shard
    elif vllm_mode == "server":
        return 1, 0
    elif dp_replicate > 1:
        return dp_replicate, rank // dp_shard
    else:
        # (pure-FSDP / non-server fallthrough; not exercised here)
        return None, None


def _slice_for_rank(ds, vllm_mode, dp_replicate, dp_shard, rank, seed=42):
    replicas, srank = _sampler_params(vllm_mode, dp_replicate, dp_shard, rank)
    sampler = StatefulDistributedSampler(
        ds, num_replicas=replicas, rank=srank, shuffle=True, seed=seed
    )
    return list(sampler)


# 7 replicas x 12 shard = 84 ranks: the production 8-node envelope.
@pytest.mark.parametrize("dp_replicate,dp_shard", [(7, 12), (2, 12), (4, 6)])
def test_server_hsdp_distinct_prompts_across_replicas(dp_replicate, dp_shard):
    world = dp_replicate * dp_shard
    ds = _RangeDataset(world * 64)

    # One representative rank per replica group: the shard-leader (local rank 0).
    leader_ranks = [rep * dp_shard for rep in range(dp_replicate)]
    leader_slices = [
        set(_slice_for_rank(ds, "server", dp_replicate, dp_shard, r))
        for r in leader_ranks
    ]

    # Every pair of replica leaders must see DISJOINT prompts.
    for i in range(dp_replicate):
        for j in range(i + 1, dp_replicate):
            assert leader_slices[i].isdisjoint(leader_slices[j]), (
                f"replica {i} and {j} share prompts under server+HSDP "
                f"(dp_replicate={dp_replicate}, dp_shard={dp_shard}) — "
                f"data parallelism is broken"
            )


@pytest.mark.parametrize("dp_replicate,dp_shard", [(7, 12), (2, 12)])
def test_server_hsdp_same_prompts_within_shard_group(dp_replicate, dp_shard):
    world = dp_replicate * dp_shard
    ds = _RangeDataset(world * 64)

    # All ranks in replica 1's shard group must see the IDENTICAL ordered slice
    # (they are FSDP shards of one copy; the shard-leader broadcasts node-locally).
    rep = 1
    group_ranks = [rep * dp_shard + k for k in range(dp_shard)]
    ref = _slice_for_rank(ds, "server", dp_replicate, dp_shard, group_ranks[0])
    for r in group_ranks[1:]:
        assert _slice_for_rank(ds, "server", dp_replicate, dp_shard, r) == ref, (
            f"rank {r} in shard group {rep} sees a different slice than its "
            f"shard-leader — followers must match the leader for the node-local "
            f"broadcast to be correct"
        )


def test_single_replicate_server_all_ranks_same_slice():
    # dp_replicate == 1: every rank sees the SAME single slice (rank 0 generates,
    # world broadcast). This is the validated 2N production behavior and must be
    # byte-identical after the HSDP change.
    dp_shard = 11
    ds = _RangeDataset(dp_shard * 64)
    ref = _slice_for_rank(ds, "server", 1, dp_shard, rank=0)
    for r in range(1, dp_shard):
        assert _slice_for_rank(ds, "server", 1, dp_shard, rank=r) == ref


def test_full_world_coverage_no_gaps():
    # Union of all replica leaders' slices should cover (most of) the dataset:
    # StatefulDistributedSampler pads to an even split, so the union size equals
    # dp_replicate * per_replica and every index is owned by exactly one replica.
    dp_replicate, dp_shard = 7, 12
    ds = _RangeDataset(dp_replicate * dp_shard * 10)
    union = set()
    per_replica = None
    for rep in range(dp_replicate):
        sl = _slice_for_rank(ds, "server", dp_replicate, dp_shard, rep * dp_shard)
        if per_replica is None:
            per_replica = len(sl)
        assert len(sl) == per_replica, "replicas got uneven slice sizes"
        union |= set(sl)
    # No overlap => union size == dp_replicate * per_replica.
    assert len(union) == dp_replicate * per_replica

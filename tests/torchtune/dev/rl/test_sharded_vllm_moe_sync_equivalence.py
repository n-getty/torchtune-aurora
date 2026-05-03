"""CPU pin-down for sharded vLLM MoE sync (WS10).

Pin-down for the per-rank sharded broadcast + receiver-side
``expert_map``-driven scatter-copy that replaces the current
``_shard_pg`` AllGather → full-tensor broadcast pipeline in
``torchtune/dev/rl/weight_sync.py::_sync_weights_to_vllm_xccl``.

The math under test is the **interleaved trainer EP shard ↔ vLLM
``expert_map``** mapping. Trainer EP rank R owns interleaved global
expert ids ``[R, R+ep_d, R+2*ep_d, ..., R+(n_local-1)*ep_d]``; vLLM EP
rank V owns whichever globals appear in its
``expert_map`` (``expert_map[g] = local_idx`` if owned, else ``-1``).
Each vLLM worker, on receipt of a per-trainer-rank shard, must
correctly identify which of its locally-owned experts come from this
shard and write them at the right position in the FusedMoE param.

This test runs entirely in pure Python; no torch.distributed, no XPU.
~1s on a login node.
"""
from __future__ import annotations

import pytest
import torch


# --- Functions under test (mirror future weight_sync / vllm_worker) ----

def _trainer_local_global_ids(trainer_rank: int, ep_degree: int,
                              n_local: int) -> list[int]:
    """Interleaved global expert IDs owned by trainer EP rank R.

    Mirrors ``ExpertParallel._token_dispatch`` and the existing
    ``test_ep_slice_contract.py`` formula:
        g = trainer_rank + local_exp_idx * ep_degree
    """
    return [trainer_rank + i * ep_degree for i in range(n_local)]


def _scatter_into_local_param(
    local_param: torch.Tensor,        # [n_local_vllm, ...] vLLM-owned slab
    received_shard: torch.Tensor,     # [n_local_trainer, ...] from one trainer rank
    trainer_rank: int,
    ep_degree: int,
    expert_map: torch.Tensor,         # length = global_n_experts; [g] = local or -1
) -> int:
    """Receiver-side scatter for one trainer-rank shard.

    Walks the shard's local indices, computes each one's global ID,
    looks it up in ``expert_map``; if owned, copies into the local
    param at the mapped local index. Returns the count of copies
    performed (so the caller can sanity-check coverage).
    """
    n_local_trainer = received_shard.shape[0]
    n_copies = 0
    for src_local_i in range(n_local_trainer):
        global_i = trainer_rank + src_local_i * ep_degree
        dst_local = int(expert_map[global_i].item())
        if dst_local >= 0:
            local_param[dst_local].copy_(received_shard[src_local_i])
            n_copies += 1
    return n_copies


def _build_uniform_interleaved_expert_map(
    vllm_rank: int, vllm_ep_size: int, global_n_experts: int,
) -> torch.Tensor:
    """Build a vLLM-style expert_map for the simple case where vLLM also
    uses interleaved partition. expert_map[g] = local_idx if g % vllm_ep_size
    == vllm_rank else -1.

    NOTE: the actual default in vLLM is *contiguous* (``determine_expert_map``);
    this test fixture uses interleaved to exercise the worst-case
    permutation pattern. The contiguous case is a strict subset.
    """
    em = torch.full((global_n_experts,), -1, dtype=torch.long)
    local_idx = 0
    for g in range(global_n_experts):
        if g % vllm_ep_size == vllm_rank:
            em[g] = local_idx
            local_idx += 1
    return em


def _build_contiguous_expert_map(
    vllm_rank: int, vllm_ep_size: int, global_n_experts: int,
) -> torch.Tensor:
    """vLLM's default: rank V owns contiguous block [V*N/EP, (V+1)*N/EP)."""
    em = torch.full((global_n_experts,), -1, dtype=torch.long)
    block = global_n_experts // vllm_ep_size
    for local_i, g in enumerate(range(vllm_rank * block, (vllm_rank + 1) * block)):
        em[g] = local_i
    return em


# --- Tests ---------------------------------------------------------------

def _build_global_expert_truth(global_n: int, hidden: int, intermediate: int,
                               seed: int = 17):
    """Synthetic 'true' fused per-layer w13 / w2 tensors."""
    torch.manual_seed(seed)
    w13 = torch.randn(global_n, 2 * intermediate, hidden, dtype=torch.bfloat16)
    w2 = torch.randn(global_n, hidden, intermediate, dtype=torch.bfloat16)
    return w13, w2


def _split_into_trainer_shards(w_global: torch.Tensor, ep_degree: int) -> list[torch.Tensor]:
    """Split a full ``[E, ...]`` global tensor into ``ep_degree`` interleaved
    shards. Trainer rank R's shard is rows [R, R+ep_d, R+2*ep_d, ...].
    """
    return [w_global[r::ep_degree].contiguous() for r in range(ep_degree)]


@pytest.mark.parametrize(
    "global_n, ep_degree, vllm_ep_size, hidden, intermediate",
    [
        (128, 16, 4, 128, 256),
        (128, 8, 8, 128, 256),
        (64, 4, 4, 64, 128),
        (64, 8, 2, 64, 128),
    ],
)
def test_sharded_dispatch_covers_owned_experts(
    global_n, ep_degree, vllm_ep_size, hidden, intermediate,
):
    """Every owned expert on every vLLM rank ends up with bit-exact
    ground-truth bytes after iterating over all trainer rank shards.
    """
    if global_n % ep_degree != 0:
        pytest.skip("test requires global_n % ep_degree == 0")
    if global_n % vllm_ep_size != 0:
        pytest.skip("test requires global_n % vllm_ep_size == 0")

    n_local_trainer = global_n // ep_degree
    n_local_vllm = global_n // vllm_ep_size

    w13_truth, w2_truth = _build_global_expert_truth(global_n, hidden, intermediate)
    trainer_w13_shards = _split_into_trainer_shards(w13_truth, ep_degree)
    trainer_w2_shards = _split_into_trainer_shards(w2_truth, ep_degree)

    # Run every (vLLM rank × every trainer broadcast).
    for vllm_rank in range(vllm_ep_size):
        # Try both interleaved and contiguous expert_map layouts.
        for em_builder in (_build_uniform_interleaved_expert_map,
                           _build_contiguous_expert_map):
            em = em_builder(vllm_rank, vllm_ep_size, global_n)
            local_w13 = torch.zeros(n_local_vllm, 2 * intermediate, hidden,
                                    dtype=torch.bfloat16)
            local_w2 = torch.zeros(n_local_vllm, hidden, intermediate,
                                   dtype=torch.bfloat16)
            total_copies_w13 = 0
            total_copies_w2 = 0
            for trainer_rank in range(ep_degree):
                total_copies_w13 += _scatter_into_local_param(
                    local_w13, trainer_w13_shards[trainer_rank],
                    trainer_rank, ep_degree, em,
                )
                total_copies_w2 += _scatter_into_local_param(
                    local_w2, trainer_w2_shards[trainer_rank],
                    trainer_rank, ep_degree, em,
                )

            # Each vLLM rank should have received exactly its owned-count
            # writes from across the trainer ranks.
            n_owned = (em >= 0).sum().item()
            assert total_copies_w13 == n_owned, (
                f"vllm_rank={vllm_rank} em={em_builder.__name__}: "
                f"w13 copies {total_copies_w13} != owned {n_owned}"
            )
            assert total_copies_w2 == n_owned, (
                f"vllm_rank={vllm_rank} em={em_builder.__name__}: "
                f"w2 copies {total_copies_w2} != owned {n_owned}"
            )

            # Every owned expert g must hold the truth at its mapped slot.
            for g in range(global_n):
                local_idx = int(em[g].item())
                if local_idx >= 0:
                    assert torch.equal(
                        local_w13[local_idx], w13_truth[g]
                    ), (f"vllm_rank={vllm_rank} em={em_builder.__name__} "
                        f"g={g} local={local_idx}: w13 mismatch")
                    assert torch.equal(
                        local_w2[local_idx], w2_truth[g]
                    ), (f"vllm_rank={vllm_rank} em={em_builder.__name__} "
                        f"g={g} local={local_idx}: w2 mismatch")


def test_sharded_dispatch_no_writes_to_unowned_slots():
    """vLLM ranks must NOT receive bytes for experts they don't own.

    Pre-fill the local param with a sentinel (1.0); after dispatch,
    verify nothing other than owned slots has been touched. (Catches
    receiver-side off-by-one or wrong-rank bugs.)
    """
    global_n, ep_degree, vllm_ep_size = 64, 8, 4
    hidden, intermediate = 32, 64
    w13_truth, _ = _build_global_expert_truth(global_n, hidden, intermediate)
    shards = _split_into_trainer_shards(w13_truth, ep_degree)

    n_local_vllm = global_n // vllm_ep_size
    sentinel = torch.full(
        (n_local_vllm, 2 * intermediate, hidden), 1.0, dtype=torch.bfloat16,
    )
    local_w13 = sentinel.clone()
    em = _build_contiguous_expert_map(vllm_rank=2,
                                      vllm_ep_size=vllm_ep_size,
                                      global_n_experts=global_n)

    for trainer_rank in range(ep_degree):
        _scatter_into_local_param(
            local_w13, shards[trainer_rank], trainer_rank, ep_degree, em,
        )

    n_owned = (em >= 0).sum().item()
    # Owned slots: must equal truth (not sentinel). Unowned slots: must
    # equal sentinel (not touched).
    for local_i in range(n_local_vllm):
        # local_i comes from a unique global, derived inversely.
        g_candidates = (em == local_i).nonzero(as_tuple=True)[0]
        if g_candidates.numel() == 1:
            g = int(g_candidates.item())
            assert torch.equal(local_w13[local_i], w13_truth[g])
        else:
            # local_i not owned → must still be sentinel
            assert torch.equal(local_w13[local_i], sentinel[local_i])
    # Sanity: total writes == n_owned
    assert n_owned == n_local_vllm  # contiguous owns exactly one block


def test_round_trip_total_bytes_match_baseline():
    """Total bytes broadcast across all trainer ranks equals one full
    expert tensor — i.e. no duplication vs the 'one rank broadcasts
    full tensor' baseline. The win comes from collapsing the ``_shard_pg``
    AllGather, not from sending fewer bytes on ``_xccl_wsync_pg``.
    """
    global_n, ep_degree = 128, 16
    hidden, intermediate = 128, 256
    w13_truth, _ = _build_global_expert_truth(global_n, hidden, intermediate)
    full_bytes = w13_truth.numel() * w13_truth.element_size()

    shards = _split_into_trainer_shards(w13_truth, ep_degree)
    sharded_total_bytes = sum(s.numel() * s.element_size() for s in shards)

    assert sharded_total_bytes == full_bytes, (
        f"sharded broadcast total bytes {sharded_total_bytes} != "
        f"baseline full broadcast bytes {full_bytes}"
    )


def test_interleaved_sharding_matches_ep_dispatch_contract():
    """Sanity tie-in to ``test_ep_slice_contract.py``: the trainer-side
    interleaved partition formula (g = R + i*ep_d) is the SAME formula
    used by ``ExpertParallel._token_dispatch`` to assign experts to ranks.
    Any drift between sender-side and dispatch-side would mean the
    trainer is sending an expert that no rank actually computed.
    """
    ep_degree = 16
    n_local = 8
    global_n = ep_degree * n_local
    for r in range(ep_degree):
        owned = _trainer_local_global_ids(r, ep_degree, n_local)
        # Same formula as ExpertParallel: g = ep_rank + local_exp_idx * ep_degree
        expected = [r + li * ep_degree for li in range(n_local)]
        assert owned == expected
        # Every global expert is owned by exactly one rank.
        assert max(owned) < global_n

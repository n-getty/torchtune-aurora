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


@pytest.mark.parametrize(
    "global_n, ep_degree, hidden, intermediate",
    [(128, 16, 64, 128), (128, 8, 64, 128), (64, 4, 32, 64)],
)
def test_tp_only_vllm_assembles_full_global_tensor(
    global_n, ep_degree, hidden, intermediate,
):
    """TP-only vLLM (no --enable-expert-parallel): every worker holds
    all experts and ``FusedMoE.expert_map`` is None. The receiver must
    use identity perm — assembled global tensor equals truth, no
    permutation. Pin-down for the bug that wedged holds 8467299 and
    8467388 (receiver raised on every layer because expert_map=None).
    """
    if global_n % ep_degree != 0:
        pytest.skip("test requires global_n % ep_degree == 0")
    n_local_trainer = global_n // ep_degree

    w13_truth, w2_truth = _build_global_expert_truth(global_n, hidden, intermediate)
    trainer_w13_shards = _split_into_trainer_shards(w13_truth, ep_degree)
    trainer_w2_shards = _split_into_trainer_shards(w2_truth, ep_degree)

    # Mirror the receiver's TP-only branch: assemble global from per-rank
    # shards via g = R + i*ep_degree, identity perm (em=None path).
    w13_global = torch.empty_like(w13_truth)
    w2_global = torch.empty_like(w2_truth)
    n_filled_w13 = n_filled_w2 = 0
    for trainer_rank in range(ep_degree):
        shard_w13 = trainer_w13_shards[trainer_rank]
        shard_w2 = trainer_w2_shards[trainer_rank]
        for src_local_i in range(shard_w13.shape[0]):
            g = trainer_rank + src_local_i * ep_degree
            assert g < global_n
            w13_global[g].copy_(shard_w13[src_local_i])
            n_filled_w13 += 1
        for src_local_i in range(shard_w2.shape[0]):
            g = trainer_rank + src_local_i * ep_degree
            w2_global[g].copy_(shard_w2[src_local_i])
            n_filled_w2 += 1

    # Coverage: every global slot filled exactly once.
    assert n_filled_w13 == global_n
    assert n_filled_w2 == global_n
    # Identity perm: assembled == truth, bit-exact.
    assert torch.equal(w13_global, w13_truth)
    assert torch.equal(w2_global, w2_truth)


def test_tp_only_global_n_derivation_from_shard():
    """Receiver TP-only branch derives global_n as ``shard.shape[0] *
    ep_degree`` (since expert_map is None and can't supply it). Pin
    that arithmetic so a future change to the manifest doesn't silently
    break the derivation.
    """
    global_n, ep_degree = 128, 16
    n_local_trainer = global_n // ep_degree
    shard = torch.empty(n_local_trainer, 4, 4, dtype=torch.bfloat16)
    derived_global_n = int(shard.shape[0]) * ep_degree
    assert derived_global_n == global_n


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


# --- Sender-side manifest pin-down ----------------------------------------
#
# These tests pin the contract the WS10 sender must satisfy so the receiver
# (`_load_fused_moe_experts_sharded` + the dispatcher in
# `vllm_weight_sync_worker.py:receive_weights_xccl_streaming`) sees what it
# expects. The receiver routes any manifest entry tagged with both
# `trainer_ep_rank` and `ep_degree` into `sharded_pending`, applies once
# `len(w13)==ep_degree` and `len(w2)==ep_degree`, and falls back to the
# legacy non-sharded path when those tags are absent.


def _sender_build_layer_entries(
    layer_idx: int,
    trainer_rank: int,
    ep_degree: int,
    w13_local: torch.Tensor,
    w2_local: torch.Tensor,
) -> list[dict]:
    """Build the per-(layer, kind) manifest entries one trainer rank emits
    when WS10 sender wire is engaged.

    Receiver expects:
      - name format `model.layers.{L}.mlp.experts.{w13|w2}_weight`
      - shape == this rank's *local* shard shape (`[n_local, ...]`), NOT
        global `[E, ...]`
      - both `trainer_ep_rank` and `ep_degree` set on the entry; without
        BOTH, receiver falls into the legacy non-sharded fused path.
      - n_local matches `shape[0]` (receiver reads this off the entry but
        falls back to `shape[0]` if missing).
    """
    return [
        {
            "name": f"model.layers.{layer_idx}.mlp.experts.w13_weight",
            "shape": list(w13_local.shape),
            "numel": w13_local.numel(),
            "trainer_ep_rank": trainer_rank,
            "ep_degree": ep_degree,
            "n_local": w13_local.shape[0],
        },
        {
            "name": f"model.layers.{layer_idx}.mlp.experts.w2_weight",
            "shape": list(w2_local.shape),
            "numel": w2_local.numel(),
            "trainer_ep_rank": trainer_rank,
            "ep_degree": ep_degree,
            "n_local": w2_local.shape[0],
        },
    ]


def test_sender_manifest_dispatches_to_sharded_path():
    """Manifest produced by sender must trigger the receiver's WS10 branch
    (lines 940-949 of vllm_weight_sync_worker.py): both `trainer_ep_rank`
    and `ep_degree` present.
    """
    ep_degree = 8
    n_local = 4
    hidden, intermediate = 32, 64
    w13_local = torch.randn(n_local, 2 * intermediate, hidden, dtype=torch.bfloat16)
    w2_local = torch.randn(n_local, hidden, intermediate, dtype=torch.bfloat16)
    entries = _sender_build_layer_entries(
        layer_idx=0, trainer_rank=3, ep_degree=ep_degree,
        w13_local=w13_local, w2_local=w2_local,
    )
    for e in entries:
        assert e.get("trainer_ep_rank") is not None
        assert e.get("ep_degree") is not None
        # Without BOTH tags, the receiver falls into the non-sharded
        # _load_fused_moe_experts path.


def test_sender_manifest_total_bytes_match_baseline():
    """Across all ep_degree trainer ranks, total bytes in the manifest
    equal the bytes one rank would have broadcast in the legacy
    full-expert-tensor path. WS10 wins by collapsing _shard_pg, not by
    sending fewer bytes on _xccl_wsync_pg.
    """
    global_n, ep_degree = 128, 16
    n_local = global_n // ep_degree
    hidden, intermediate = 128, 256

    w13_truth, w2_truth = _build_global_expert_truth(global_n, hidden, intermediate)
    legacy_w13_bytes = w13_truth.numel() * w13_truth.element_size()
    legacy_w2_bytes = w2_truth.numel() * w2_truth.element_size()

    sharded_w13_bytes = 0
    sharded_w2_bytes = 0
    for r in range(ep_degree):
        w13_local = w13_truth[r::ep_degree].contiguous()
        w2_local = w2_truth[r::ep_degree].contiguous()
        entries = _sender_build_layer_entries(
            layer_idx=0, trainer_rank=r, ep_degree=ep_degree,
            w13_local=w13_local, w2_local=w2_local,
        )
        for e in entries:
            elem_bytes = w13_truth.element_size()  # bf16
            if e["name"].endswith("w13_weight"):
                sharded_w13_bytes += e["numel"] * elem_bytes
            else:
                sharded_w2_bytes += e["numel"] * elem_bytes

    assert sharded_w13_bytes == legacy_w13_bytes
    assert sharded_w2_bytes == legacy_w2_bytes


def test_sender_manifest_round_trip_through_receiver_dispatcher():
    """End-to-end: sender emits ep_degree*2 entries per layer; receiver
    dispatcher (modeled inline) accumulates; once all 2*ep_degree shards
    arrive, _scatter_into_local_param reproduces the truth at every owned
    slot. This is the closest CPU-only test to the live WS10 wire — the
    only thing missing is the actual broadcast.
    """
    global_n, ep_degree, vllm_ep_size = 128, 16, 4
    hidden, intermediate = 64, 128
    n_local_trainer = global_n // ep_degree

    w13_truth, w2_truth = _build_global_expert_truth(global_n, hidden, intermediate)
    trainer_w13 = _split_into_trainer_shards(w13_truth, ep_degree)
    trainer_w2 = _split_into_trainer_shards(w2_truth, ep_degree)

    # Build the full stream of manifest entries (one layer, ep_degree senders).
    stream: list[tuple[dict, torch.Tensor]] = []
    for r in range(ep_degree):
        entries = _sender_build_layer_entries(
            layer_idx=0, trainer_rank=r, ep_degree=ep_degree,
            w13_local=trainer_w13[r], w2_local=trainer_w2[r],
        )
        # entries[0] is w13, entries[1] is w2 — pair with payload tensors.
        stream.append((entries[0], trainer_w13[r]))
        stream.append((entries[1], trainer_w2[r]))

    # Inline copy of the receiver dispatcher logic
    # (vllm_weight_sync_worker.py:933-955).
    sharded_pending: dict = {}
    fused_pending: dict = {}
    import re as _re
    fused_re = _re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
    )
    for entry, payload in stream:
        m = fused_re.match(entry["name"])
        assert m is not None
        layer_idx = int(m.group(1))
        kind = m.group(2)
        trainer_ep_rank = entry.get("trainer_ep_rank")
        ep_d_entry = entry.get("ep_degree")
        if trainer_ep_rank is not None and ep_d_entry is not None:
            lyr = sharded_pending.setdefault(layer_idx, {
                "w13": {}, "w2": {},
                "ep_degree": int(ep_d_entry),
                "n_local_trainer": int(entry.get("n_local", entry["shape"][0])),
            })
            lyr[kind][int(trainer_ep_rank)] = payload.clone()
        else:
            fused_pending.setdefault(layer_idx, {})[kind] = payload.clone()

    # Receiver should NEVER fall into the non-sharded path with WS10 tags.
    assert not fused_pending, (
        "WS10 entries leaked into legacy fused_pending — sender manifest "
        "is missing trainer_ep_rank or ep_degree on at least one entry."
    )

    # All ep_degree shards present for both kinds — would now apply.
    assert len(sharded_pending) == 1
    lyr = sharded_pending[0]
    assert len(lyr["w13"]) == ep_degree
    assert len(lyr["w2"]) == ep_degree
    assert lyr["n_local_trainer"] == n_local_trainer

    # Run the apply step (mirrors _load_fused_moe_experts_sharded staging
    # via _scatter_into_local_param, then verify each vLLM rank's slab).
    for vllm_rank in range(vllm_ep_size):
        em = _build_contiguous_expert_map(vllm_rank, vllm_ep_size, global_n)
        n_local_vllm = global_n // vllm_ep_size
        local_w13 = torch.zeros(
            n_local_vllm, 2 * intermediate, hidden, dtype=torch.bfloat16,
        )
        local_w2 = torch.zeros(
            n_local_vllm, hidden, intermediate, dtype=torch.bfloat16,
        )
        for r in range(ep_degree):
            _scatter_into_local_param(
                local_w13, lyr["w13"][r], r, ep_degree, em,
            )
            _scatter_into_local_param(
                local_w2, lyr["w2"][r], r, ep_degree, em,
            )
        for g in range(global_n):
            local_idx = int(em[g].item())
            if local_idx >= 0:
                assert torch.equal(local_w13[local_idx], w13_truth[g])
                assert torch.equal(local_w2[local_idx], w2_truth[g])


def _canonical_manifest_sort_key(entry: dict) -> tuple:
    """Deterministic ordering for cross-rank manifest unification.

    Both sender (every trainer rank's local manifest, after
    `all_gather_object`) and receiver (the ordered batch payload it
    consumes) must agree on this key. If they drift, the broadcast
    loop hangs because rank R sources batch i while vLLM expects
    rank R' to source it.

    Sort key: (layer_idx, kind_rank, trainer_ep_rank). kind_rank
    pins w13 before w2 (matches the streaming fuse order in the
    sender's _ep_stream_buf).
    """
    import re as _re
    fused_re = _re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight"
    )
    m = fused_re.match(entry["name"])
    if m:
        layer_idx = int(m.group(1))
        kind_rank = 0 if m.group(2) == "w13" else 1
    else:
        # Non-expert entries sort last (they go through the legacy
        # rank-0-broadcasts-everything path).
        layer_idx, kind_rank = 10**9, 9
    R = int(entry.get("trainer_ep_rank", -1))
    return (layer_idx, kind_rank, R)


def test_manifest_ordering_is_deterministic_across_ranks():
    """When the sender on every trainer rank `all_gather_object`s its
    local manifest and rank-0 concatenates, the resulting global
    manifest must be byte-identical regardless of which trainer rank
    happened to emit which entry first.

    This test simulates ep_degree=4 ranks each producing the same
    set of (layer, kind) entries in random local orders, then verifies
    that the canonical sort produces the same global sequence
    regardless of input permutation.
    """
    import random
    ep_degree = 4
    n_local = 8
    n_layers = 6
    hidden, intermediate = 16, 32

    def _build_rank_manifest(R: int, perm_seed: int) -> list[dict]:
        entries = []
        for L in range(n_layers):
            w13_local = torch.randn(
                n_local, 2 * intermediate, hidden, dtype=torch.bfloat16,
            )
            w2_local = torch.randn(
                n_local, hidden, intermediate, dtype=torch.bfloat16,
            )
            entries.extend(_sender_build_layer_entries(
                layer_idx=L, trainer_rank=R, ep_degree=ep_degree,
                w13_local=w13_local, w2_local=w2_local,
            ))
        rng = random.Random(perm_seed)
        rng.shuffle(entries)
        return entries

    # Two simulated runs — each rank's local order is randomized
    # differently. After canonical sort, the global manifest must
    # match exactly between runs.
    def _global_sorted(seed_base: int) -> list[tuple]:
        all_entries = []
        for R in range(ep_degree):
            all_entries.extend(_build_rank_manifest(R, seed_base + R))
        all_entries.sort(key=_canonical_manifest_sort_key)
        # Strip payload-dependent fields (numel, shape) — only the
        # ordering keys matter for the contract.
        return [(e["name"], int(e["trainer_ep_rank"])) for e in all_entries]

    seq_a = _global_sorted(seed_base=100)
    seq_b = _global_sorted(seed_base=200)
    assert seq_a == seq_b, "Canonical sort is not order-stable"

    # Coverage: every (L, kind, R) triple appears exactly once.
    expected = [
        (f"model.layers.{L}.mlp.experts.{k}_weight", R)
        for L in range(n_layers) for k in ("w13", "w2") for R in range(ep_degree)
    ]
    # The canonical sort: (layer, w13<w2, R-asc). Verify that's what we got.
    assert seq_a == expected, (
        f"Canonical order mismatch.\nGot:      {seq_a[:6]}\n"
        f"Expected: {expected[:6]}"
    )


def test_manifest_ordering_groups_by_layer_then_kind_then_rank():
    """The sort key prescribes (layer asc, w13 before w2, rank asc).
    This pin-down catches drift from anyone who 'fixes' the sort
    later (e.g. switches to (rank, layer, kind) for cache locality
    on the sender), which would silently desync the receiver.
    """
    entries = [
        {"name": "model.layers.5.mlp.experts.w2_weight", "trainer_ep_rank": 0},
        {"name": "model.layers.0.mlp.experts.w13_weight", "trainer_ep_rank": 3},
        {"name": "model.layers.0.mlp.experts.w13_weight", "trainer_ep_rank": 1},
        {"name": "model.layers.0.mlp.experts.w2_weight", "trainer_ep_rank": 0},
        {"name": "model.layers.5.mlp.experts.w13_weight", "trainer_ep_rank": 2},
    ]
    entries.sort(key=_canonical_manifest_sort_key)
    got = [(e["name"].split(".")[2], e["name"].split(".")[-1].rsplit("_", 1)[0],
            e["trainer_ep_rank"]) for e in entries]
    expected = [
        ("0", "w13", 1),
        ("0", "w13", 3),
        ("0", "w2", 0),
        ("5", "w13", 2),
        ("5", "w2", 0),
    ]
    assert got == expected


def test_sender_manifest_off_path_when_ep_degree_one():
    """When ep_degree==1 there is no sharding — the sender MUST emit
    legacy untagged entries so the receiver routes through the original
    _load_fused_moe_experts path. Otherwise WS10 mode would silently
    engage on dense models or non-EP MoE.
    """
    # If we ever build entries for ep_degree=1, both should be untagged.
    # Convention: WS10 is gated by `_ep_deg > 1` in the sender; this test
    # encodes that contract.
    n_local = 4
    hidden, intermediate = 32, 64
    w13_local = torch.randn(n_local, 2 * intermediate, hidden, dtype=torch.bfloat16)
    w2_local = torch.randn(n_local, hidden, intermediate, dtype=torch.bfloat16)

    # Legacy entry (no tags). Mirrors the existing pre-WS10 code path.
    legacy = [
        {
            "name": "model.layers.0.mlp.experts.w13_weight",
            "shape": list(w13_local.shape),
            "numel": w13_local.numel(),
        },
        {
            "name": "model.layers.0.mlp.experts.w2_weight",
            "shape": list(w2_local.shape),
            "numel": w2_local.numel(),
        },
    ]
    for e in legacy:
        assert "trainer_ep_rank" not in e
        assert "ep_degree" not in e


# ---------------------------------------------------------------------------
# Layer 3 pin-downs: real helpers from torchtune.dev.rl.weight_sync
# ---------------------------------------------------------------------------
#
# The two helpers below run pure-CPU and contain the contract that the WS10
# sender path must obey: per-rank staging + cross-rank manifest unification.
# Once Layer 4 (deferred broadcast loop) lands these become the load-bearing
# tests for the recipe-side wire-up.

from torchtune.dev.rl.weight_sync import (
    _ws10_build_local_payload,
    _ws10_unify_manifests,
    _ws10_sort_key,
    _WS10_FUSED_RE,
)


def _make_local_experts(layer_ids, n_local, hidden, intermediate):
    """Build a {hf_name: bf16 cpu tensor} dict mirroring a rank's local
    fused expert shards across `layer_ids`."""
    out = {}
    for L in layer_ids:
        out[f"model.layers.{L}.mlp.experts.w13_weight"] = torch.randn(
            n_local, 2 * intermediate, hidden, dtype=torch.bfloat16,
        )
        out[f"model.layers.{L}.mlp.experts.w2_weight"] = torch.randn(
            n_local, hidden, intermediate, dtype=torch.bfloat16,
        )
    return out


def test_build_local_payload_tags_every_entry():
    """Every produced manifest entry must carry the three WS10 tags
    (trainer_ep_rank, ep_degree, n_local). Untagged entries would route
    through the legacy receiver path and silently corrupt sharded sync."""
    locals_dict = _make_local_experts(layer_ids=[0, 1, 2], n_local=4,
                                      hidden=16, intermediate=32)
    cpu_batches, meta = _ws10_build_local_payload(
        locals_dict, ep_rank=2, ep_degree=8, batch_max_numel=10**7,
    )
    assert len(meta) == 6  # 3 layers × {w13, w2}
    for e in meta:
        assert e["trainer_ep_rank"] == 2
        assert e["ep_degree"] == 8
        assert e["n_local"] == 4
        # Internal _tensor field must NOT leak into the wire manifest.
        assert "_tensor" not in e
    assert len(cpu_batches) >= 1


def test_build_local_payload_canonical_order():
    """Manifest order must follow (layer asc, w13 before w2). The CPU
    batches must concatenate in the same order — receiver consumes them
    in this order, so any drift hangs the broadcast loop."""
    locals_dict = _make_local_experts(layer_ids=[5, 1, 3], n_local=2,
                                      hidden=8, intermediate=16)
    _, meta = _ws10_build_local_payload(
        locals_dict, ep_rank=0, ep_degree=4, batch_max_numel=10**7,
    )
    names = [e["name"] for e in meta]
    expected = [
        "model.layers.1.mlp.experts.w13_weight",
        "model.layers.1.mlp.experts.w2_weight",
        "model.layers.3.mlp.experts.w13_weight",
        "model.layers.3.mlp.experts.w2_weight",
        "model.layers.5.mlp.experts.w13_weight",
        "model.layers.5.mlp.experts.w2_weight",
    ]
    assert names == expected


def test_build_local_payload_byte_count_matches_sum_of_shards():
    """Across all CPU batches we must transmit exactly the sum of the
    local shard byte-counts. Drift means we either drop or duplicate
    bytes — the receiver would silently corrupt the FusedMoE param."""
    locals_dict = _make_local_experts(layer_ids=[0, 1, 2, 3], n_local=4,
                                      hidden=16, intermediate=32)
    expected_numel = sum(t.numel() for t in locals_dict.values())
    expected_bytes = sum(
        t.numel() * t.element_size() for t in locals_dict.values()
    )
    # Force at least 2 batches so the splitting logic is exercised.
    cpu_batches, _ = _ws10_build_local_payload(
        locals_dict, ep_rank=1, ep_degree=4,
        batch_max_numel=expected_numel // 3,
    )
    assert len(cpu_batches) >= 2
    actual_numel = sum(b.numel() for b in cpu_batches)
    actual_bytes = sum(b.numel() * b.element_size() for b in cpu_batches)
    assert actual_numel == expected_numel
    assert actual_bytes == expected_bytes
    # bf16 1-D contiguous concat is the wire format.
    for b in cpu_batches:
        assert b.dtype == torch.bfloat16
        assert b.dim() == 1
        assert b.is_contiguous()


def test_build_local_payload_rejects_non_fused_names():
    """Non-fused names (norms, attention) belong to the existing path
    — passing them here is a programmer error and must raise loudly,
    not silently mis-tag them with a trainer_ep_rank."""
    bogus = {"model.layers.0.input_layernorm.weight": torch.randn(8)}
    with pytest.raises(ValueError, match="non-fused name"):
        _ws10_build_local_payload(bogus, ep_rank=0, ep_degree=4,
                                  batch_max_numel=10**7)


def test_build_local_payload_rejects_ep_degree_one():
    """ep_degree=1 has no shards to combine — the helper would produce
    a degenerate manifest that vLLM's sharded receiver still tries to
    accumulate. Better to fail at the sender."""
    locals_dict = _make_local_experts(layer_ids=[0], n_local=8,
                                      hidden=8, intermediate=16)
    with pytest.raises(ValueError, match="ep_degree > 1"):
        _ws10_build_local_payload(locals_dict, ep_rank=0, ep_degree=1,
                                  batch_max_numel=10**7)


def test_build_local_payload_rejects_out_of_range_rank():
    locals_dict = _make_local_experts(layer_ids=[0], n_local=4,
                                      hidden=8, intermediate=16)
    with pytest.raises(ValueError, match="out of range"):
        _ws10_build_local_payload(locals_dict, ep_rank=4, ep_degree=4,
                                  batch_max_numel=10**7)
    with pytest.raises(ValueError, match="out of range"):
        _ws10_build_local_payload(locals_dict, ep_rank=-1, ep_degree=4,
                                  batch_max_numel=10**7)


def test_unify_manifests_groups_by_layer_then_kind_then_rank():
    """The unified manifest is what vLLM's deferred broadcast loop walks.
    Order MUST be (layer, kind, rank). Receiver-side
    `_load_fused_moe_experts_sharded` waits for all `ep_degree` shards
    of a given (layer, kind) before applying — out-of-order doesn't
    break correctness, but lockstep across trainer ranks does."""
    ep_d = 4
    per_rank = []
    layer_ids = [0, 1]
    for R in range(ep_d):
        meta = []
        for L in layer_ids:
            for kind in ("w13", "w2"):
                meta.append({
                    "name": f"model.layers.{L}.mlp.experts.{kind}_weight",
                    "shape": [4, 8, 16],
                    "numel": 4 * 8 * 16,
                    "trainer_ep_rank": R,
                    "ep_degree": ep_d,
                    "n_local": 4,
                })
        per_rank.append(meta)
    flat = _ws10_unify_manifests(per_rank)
    expected_n = ep_d * len(layer_ids) * 2
    assert len(flat) == expected_n
    # First ep_d entries are layer 0, w13, ranks 0..ep_d-1.
    for i in range(ep_d):
        assert flat[i]["name"] == "model.layers.0.mlp.experts.w13_weight"
        assert flat[i]["trainer_ep_rank"] == i
    # Next ep_d are layer 0, w2.
    for i in range(ep_d):
        e = flat[ep_d + i]
        assert e["name"] == "model.layers.0.mlp.experts.w2_weight"
        assert e["trainer_ep_rank"] == i


def test_unify_manifests_rejects_inconsistent_ep_degree():
    """If trainer ranks disagree on ep_degree, unification raises rather
    than picking a winner — receiver would process the manifest assuming
    a uniform ep_degree, producing partial expert coverage."""
    per_rank = [
        [{"name": "model.layers.0.mlp.experts.w13_weight",
          "shape": [4, 8, 16], "numel": 512,
          "trainer_ep_rank": 0, "ep_degree": 4, "n_local": 4}],
        [{"name": "model.layers.0.mlp.experts.w13_weight",
          "shape": [4, 8, 16], "numel": 512,
          "trainer_ep_rank": 1, "ep_degree": 8, "n_local": 4}],
    ]
    with pytest.raises(ValueError, match="heterogeneous ep_degree"):
        _ws10_unify_manifests(per_rank)


def test_unify_manifests_rejects_wrong_count():
    """We expect exactly `ep_degree` per-rank manifests at unify time.
    A short/long list means a trainer rank failed to participate in the
    `all_gather_object` and the receiver would hang waiting for the
    missing rank's shards."""
    per_rank = [
        [{"name": "model.layers.0.mlp.experts.w13_weight",
          "shape": [4, 8, 16], "numel": 512,
          "trainer_ep_rank": 0, "ep_degree": 4, "n_local": 4}],
        [{"name": "model.layers.0.mlp.experts.w13_weight",
          "shape": [4, 8, 16], "numel": 512,
          "trainer_ep_rank": 1, "ep_degree": 4, "n_local": 4}],
    ]
    with pytest.raises(ValueError, match="ep_degree=4"):
        _ws10_unify_manifests(per_rank)


def test_unify_manifests_handles_empty_input():
    """Empty input is a degenerate but legal case (no MoE params on any
    rank). Returning [] keeps the receiver dispatcher's iteration
    well-defined."""
    assert _ws10_unify_manifests([]) == []


def test_unify_then_sort_key_is_idempotent():
    """Receivers / loggers may re-sort the unified manifest using
    `_ws10_sort_key`. Re-sorting must be a no-op — otherwise broadcast
    order on the trainer side could diverge from receiver iteration on
    the vLLM side."""
    per_rank = []
    for R in range(4):
        per_rank.append([
            {"name": "model.layers.7.mlp.experts.w2_weight",
             "shape": [2, 8, 16], "numel": 256,
             "trainer_ep_rank": R, "ep_degree": 4, "n_local": 2},
            {"name": "model.layers.7.mlp.experts.w13_weight",
             "shape": [2, 16, 16], "numel": 512,
             "trainer_ep_rank": R, "ep_degree": 4, "n_local": 2},
        ])
    flat = _ws10_unify_manifests(per_rank)
    again = sorted(flat, key=_ws10_sort_key)
    assert flat == again


def test_round_trip_local_payload_to_unified_manifest_to_receiver():
    """End-to-end Layer 3 → receiver dispatcher contract.

    Each rank builds its local payload, manifests are unified, the
    flat manifest is fed back through the existing receiver helpers
    (_trainer_local_global_ids + _scatter_into_local_param) and we
    verify every (vLLM-owned global expert × {w13, w2}) slot is hit
    exactly once across the full broadcast sequence.
    """
    ep_d = 4
    n_global = 16
    n_local_per_rank = n_global // ep_d
    hidden, intermediate = 8, 16
    layer_ids = [0, 1]

    per_rank_meta = []
    per_rank_payload = []  # cpu_batches per rank (kept for byte-count assert)
    per_rank_local_tensors = {R: {} for R in range(ep_d)}
    for R in range(ep_d):
        locals_dict = _make_local_experts(
            layer_ids, n_local_per_rank, hidden, intermediate,
        )
        per_rank_local_tensors[R] = locals_dict
        cpu_batches, meta = _ws10_build_local_payload(
            locals_dict, ep_rank=R, ep_degree=ep_d, batch_max_numel=10**7,
        )
        per_rank_meta.append(meta)
        per_rank_payload.append(cpu_batches)

    flat = _ws10_unify_manifests(per_rank_meta)
    # ep_d ranks × len(layer_ids) layers × 2 kinds.
    assert len(flat) == ep_d * len(layer_ids) * 2

    # Simulate a vLLM rank that owns globals [0, 4, 8, 12] (ep_d=4 EP
    # vLLM as well; expert_map[g] = local_idx for owned, -1 otherwise).
    vllm_owned_globals = [0, 4, 8, 12]
    expert_map = torch.full((n_global,), -1, dtype=torch.long)
    for li, g in enumerate(vllm_owned_globals):
        expert_map[g] = li

    # Local vLLM param slabs — one per (layer, kind).
    vllm_w13 = {
        L: torch.zeros(len(vllm_owned_globals), 2 * intermediate, hidden,
                       dtype=torch.bfloat16) for L in layer_ids
    }
    vllm_w2 = {
        L: torch.zeros(len(vllm_owned_globals), hidden, intermediate,
                       dtype=torch.bfloat16) for L in layer_ids
    }

    # Walk the unified manifest in canonical order and apply each
    # received shard to the right vLLM slab.
    n_writes_per_layer_kind = {(L, k): 0 for L in layer_ids
                               for k in ("w13", "w2")}
    for entry in flat:
        m = _WS10_FUSED_RE.match(entry["name"])
        assert m is not None
        L = int(m.group(1))
        kind = m.group(2)
        R = entry["trainer_ep_rank"]
        # Receiver consumes the original local tensor (the broadcast
        # round trip is bit-exact on CPU here — we inspect the source
        # directly rather than re-flattening from cpu_batches).
        local_t = per_rank_local_tensors[R][entry["name"]]
        slab = vllm_w13[L] if kind == "w13" else vllm_w2[L]
        n_writes_per_layer_kind[(L, kind)] += _scatter_into_local_param(
            slab, local_t, trainer_rank=R, ep_degree=ep_d,
            expert_map=expert_map,
        )

    # Each vLLM rank owns exactly len(vllm_owned_globals) experts per
    # (layer, kind). After processing all ep_d trainer-rank shards we
    # must have exactly that many writes — no gaps, no duplicates.
    for (L, kind), n_w in n_writes_per_layer_kind.items():
        assert n_w == len(vllm_owned_globals), (
            f"layer {L} {kind}: {n_w} writes, want {len(vllm_owned_globals)}")


# ---------------------------------------------------------------------------
# WS10 Commit B: sender vs receiver batch-boundary alignment.
#
# Each trainer EP rank R sends N_R batches on its own per-R cross PG. The
# receiver iterates the unified manifest and per-batch picks the PG by the
# first entry's ``trainer_ep_rank``. For the broadcast to succeed, the
# receiver's batch boundaries MUST match the sender's per-rank greedy pack.
# Run #6 (FAIL_BCAST_WIRE) hit a gloo
#   "preamble.length <= nbytes  1064400384 vs 622329856"
# because the receiver allowed batches to span trainer_ep_rank transitions
# and the resulting size mismatched what the corresponding rank actually
# broadcast on its PG. The fix breaks the receiver's batch at every R
# transition (and at the expert -> non-expert tag-disappears transition).


def _receiver_batch_sizes_with_fix(unified_meta, batch_max_numel):
    """Emulate receiver's WS10-aware greedy batch packing (post-fix).

    Mirrors the loop in ``vllm_weight_sync_worker.py`` post-fix: each batch
    is restricted to a single ``trainer_ep_rank`` value (with ``None`` —
    the absent-tag case used by rank-0 non-expert tail — counted as its
    own R run). Returns the per-batch numel list.
    """
    sizes = []
    n = len(unified_meta)
    i = 0
    while i < n:
        cn = 0
        R_run = unified_meta[i].get("trainer_ep_rank")
        while i < n:
            pn = unified_meta[i]["numel"]
            R_now = unified_meta[i].get("trainer_ep_rank")
            if cn > 0 and R_now != R_run:
                break
            if cn > 0 and cn + pn > batch_max_numel:
                break
            cn += pn
            i += 1
        sizes.append(cn)
    return sizes


def _sender_per_rank_batch_sizes(per_rank_meta, batch_max_numel,
                                 non_expert_meta_for_rank0=None):
    """Emulate sender's per-rank greedy pack. Each rank packs its own
    expert entries. Rank 0 additionally appends a separate run of
    non-expert entries (ridden to the receiver on R=0's PG)."""
    out = []
    for R, meta in enumerate(per_rank_meta):
        cur, cn = [], 0
        batches = []
        for e in meta:
            n = e["numel"]
            if cn > 0 and cn + n > batch_max_numel:
                batches.append(cn)
                cur, cn = [], 0
            cur.append(e)
            cn += n
        if cn > 0:
            batches.append(cn)
        # Rank 0 non-expert tail packed as a separate run (sender code path).
        # Sender re-sorts non-experts to match the receiver's unified-manifest
        # order (_ws10_sort_key_by_rank => lexicographic by name for non-experts).
        # Run #7 fix: previously sender used insertion (sharded_sd.items())
        # order, which diverged from the receiver's sort and caused a gloo
        # preamble.length size mismatch (1064400384 vs 622329856).
        if R == 0 and non_expert_meta_for_rank0:
            from torchtune.dev.rl.weight_sync import _ws10_sort_key_by_rank
            ne_sorted = sorted(non_expert_meta_for_rank0,
                               key=_ws10_sort_key_by_rank)
            cur, cn = [], 0
            for e in ne_sorted:
                n = e["numel"]
                if cn > 0 and cn + n > batch_max_numel:
                    batches.append(cn)
                    cur, cn = [], 0
                cur.append(e)
                cn += n
            if cn > 0:
                batches.append(cn)
        out.append(batches)
    return out


@pytest.mark.parametrize(
    "ep_degree, n_layers, w13_numel_per_expert, w2_numel_per_expert, "
    "n_non_expert, batch_max_numel",
    [
        # 30B-A3B EP=16 production envelope: large enough that one rank's
        # 96 entries (48 layers × 2 kinds) span multiple 512M batches.
        (16, 48, 25_165_824, 12_582_912, 435, 512 * 1024 * 1024),
        # Smaller MoE: a single rank fits in one batch but boundary still matters.
        (8, 24, 8_388_608, 4_194_304, 200, 64 * 1024 * 1024),
        (4, 12, 4_194_304, 2_097_152, 60, 32 * 1024 * 1024),
    ],
)
def test_receiver_batch_alignment_with_sender_per_rank_pack(
    ep_degree, n_layers, w13_numel_per_expert, w2_numel_per_expert,
    n_non_expert, batch_max_numel,
):
    """Receiver greedy pack (with fix) must yield the same batch size
    sequence as concatenating each trainer rank's per-rank batch sequence
    in unified-manifest order. Gloo broadcast requires exact size match
    on both ends; misalignment killed run #6.
    """
    per_rank_meta = []
    for R in range(ep_degree):
        meta = []
        for li in range(n_layers):
            meta.append({
                "name": f"model.layers.{li}.mlp.experts.w13_weight",
                "numel": int(w13_numel_per_expert // ep_degree),
                "trainer_ep_rank": R,
                "ep_degree": ep_degree,
            })
            meta.append({
                "name": f"model.layers.{li}.mlp.experts.w2_weight",
                "numel": int(w2_numel_per_expert // ep_degree),
                "trainer_ep_rank": R,
                "ep_degree": ep_degree,
            })
        per_rank_meta.append(meta)

    non_expert_meta = [
        {"name": f"model.layers.{li}.input_layernorm.weight", "numel": 4096}
        for li in range(n_non_expert)
    ]

    # Build unified manifest the way the recipe does: rank-0 slot also
    # carries non-expert tail (no trainer_ep_rank tag).
    per_rank_for_unify = list(per_rank_meta)
    per_rank_for_unify[0] = list(per_rank_for_unify[0]) + non_expert_meta
    flat = [e for m in per_rank_for_unify for e in m]
    # Re-use the canonical by-rank ordering from the recipe.
    from torchtune.dev.rl.weight_sync import _ws10_sort_key_by_rank
    flat.sort(key=_ws10_sort_key_by_rank)

    sender_batches = _sender_per_rank_batch_sizes(
        per_rank_meta, batch_max_numel,
        non_expert_meta_for_rank0=non_expert_meta,
    )
    sender_flat = [b for run in sender_batches for b in run]
    recv_flat = _receiver_batch_sizes_with_fix(flat, batch_max_numel)

    assert recv_flat == sender_flat, (
        f"receiver batches {recv_flat} != sender batches (flattened) "
        f"{sender_flat}; mismatch will cause gloo preamble.length error"
    )
    assert sum(recv_flat) == sum(sender_flat)
    # Sanity: receiver batches each have exactly one R (or None for non-expert).
    i = 0
    for n_in_batch in recv_flat:
        ents = []
        size = 0
        while size < n_in_batch:
            ents.append(flat[i])
            size += flat[i]["numel"]
            i += 1
        Rs_in_batch = {e.get("trainer_ep_rank") for e in ents}
        assert len(Rs_in_batch) == 1, (
            f"batch contains entries from multiple ranks {Rs_in_batch}; "
            "sender broadcasts on a single PG per batch — must not span ranks"
        )


def test_non_expert_tail_sort_matches_receiver_unified_order():
    """Regression for run #7 (1064400384 vs 622329856 gloo preamble.length).

    The sender used to pack rank-0's non-expert tail in
    ``sharded_sd.items()`` walk order — i.e. module registration order
    (``embed_tokens``, then ``layers.0.input_layernorm``, ...,
    ``layers.47.post_attention_layernorm``, ``norm``, ``lm_head``). The
    receiver iterates the unified manifest sorted by
    ``_ws10_sort_key_by_rank``, which places non-experts at
    ``(R=0, layer=10**9, kind=9, name)`` — i.e. lexicographic by name
    (``embed_tokens``, ``layers.0.input_layernorm``,
    ``layers.0.post_attention_layernorm``, ``layers.1.input_layernorm``,
    ..., ``lm_head``, ``norm``). Same total bytes, but the greedy-split
    boundaries land in different places so a single batch's announced
    nbytes diverges between sender and receiver — gloo terminates with
    preamble.length > nbytes.

    This test uses a realistic Qwen3-30B-A3B style non-expert tail and
    asserts the FIXED behavior (sender re-sorts by name) yields aligned
    batches; the un-sorted variant must NOT match (otherwise the test
    can't catch a regression).
    """
    from torchtune.dev.rl.weight_sync import _ws10_sort_key_by_rank
    ep_degree = 16
    n_layers = 48
    batch_max_numel = 512 * 1024 * 1024
    # Realistic non-expert tail: per-layer norms in walk order, then
    # final norm + lm_head + embed_tokens. Walk order does NOT match
    # lexicographic order (e.g. layers.10.* sorts before layers.2.*).
    walk_order_non_expert = []
    walk_order_non_expert.append(
        {"name": "model.embed_tokens.weight", "numel": 151_643_904})
    for li in range(n_layers):
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                   "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                   "mlp.gate.weight"):
            walk_order_non_expert.append({
                "name": f"model.layers.{li}.{nm}.weight"
                        if not nm.endswith(".weight") else
                        f"model.layers.{li}.{nm}",
                "numel": 4096 * 5120 if "proj" in nm else 4096,
            })
    walk_order_non_expert.append({"name": "model.norm.weight", "numel": 4096})
    walk_order_non_expert.append(
        {"name": "lm_head.weight", "numel": 151_643_904})

    # Build a synthetic per-rank meta (experts only) at production sizes.
    per_rank_meta = []
    w13 = 25_165_824 // ep_degree
    w2 = 12_582_912 // ep_degree
    for R in range(ep_degree):
        meta = []
        for li in range(n_layers):
            meta.append({
                "name": f"model.layers.{li}.mlp.experts.w13_weight",
                "numel": w13, "trainer_ep_rank": R, "ep_degree": ep_degree,
            })
            meta.append({
                "name": f"model.layers.{li}.mlp.experts.w2_weight",
                "numel": w2, "trainer_ep_rank": R, "ep_degree": ep_degree,
            })
        per_rank_meta.append(meta)

    per_rank_for_unify = list(per_rank_meta)
    per_rank_for_unify[0] = list(per_rank_for_unify[0]) + walk_order_non_expert
    flat = [e for m in per_rank_for_unify for e in m]
    flat.sort(key=_ws10_sort_key_by_rank)

    # FIXED sender: re-sorts non-experts to match receiver's unified order.
    sender_fixed = _sender_per_rank_batch_sizes(
        per_rank_meta, batch_max_numel,
        non_expert_meta_for_rank0=walk_order_non_expert,
    )
    sender_fixed_flat = [b for run in sender_fixed for b in run]
    recv_flat = _receiver_batch_sizes_with_fix(flat, batch_max_numel)
    assert recv_flat == sender_fixed_flat, (
        "Post-fix sender (sorted non-expert tail) must match receiver. "
        f"recv={recv_flat} sender={sender_fixed_flat}"
    )

    # OLD sender: walk-order packing without re-sort. Must produce a
    # different per-batch breakdown — otherwise the test can't catch the
    # regression we just fixed.
    def _sender_walk_order(per_rank_meta, walk_ne):
        out = []
        for R, meta in enumerate(per_rank_meta):
            cur, cn = [], 0
            batches = []
            for e in meta:
                n = e["numel"]
                if cn > 0 and cn + n > batch_max_numel:
                    batches.append(cn); cur, cn = [], 0
                cur.append(e); cn += n
            if cn > 0:
                batches.append(cn)
            if R == 0:
                cur, cn = [], 0
                for e in walk_ne:
                    n = e["numel"]
                    if cn > 0 and cn + n > batch_max_numel:
                        batches.append(cn); cur, cn = [], 0
                    cur.append(e); cn += n
                if cn > 0:
                    batches.append(cn)
            out.append(batches)
        return out

    sender_old = _sender_walk_order(per_rank_meta, walk_order_non_expert)
    sender_old_flat = [b for run in sender_old for b in run]
    assert sum(sender_old_flat) == sum(sender_fixed_flat), (
        "total bytes must match — only the boundary positions differ"
    )
    assert sender_old_flat != recv_flat, (
        "Sanity: walk-order pack must DIVERGE from receiver's sorted "
        "iteration; otherwise this test cannot detect the run #7 bug."
    )

"""WS10 Commit B sender pin-down: manifest ordering + per-rank payload.

Pure-CPU regression for the staging path implemented in
``torchtune/dev/rl/weight_sync.py`` Commit B:

- ``_ws10_build_local_payload`` produces correctly tagged per-rank
  manifests for fused expert tensors (already covered by
  ``test_sharded_vllm_moe_sync_equivalence.py`` — re-asserted here).
- ``_ws10_sort_key_by_rank`` groups entries by trainer EP rank so the
  receiver iterates one PG at a time. Within a rank, fused entries
  come before non-expert entries; layers are in ascending order; w13
  precedes w2.
- ``_ws10_unify_manifests(..., sort_key=_ws10_sort_key_by_rank)``
  produces an ordering where rank 0's entries come first, then rank
  1's, etc., and rank 0's optional non-expert tail trails its
  expert entries.
- A simulated round-trip walks the unified manifest, accumulates
  per-(R, kind) shards, then reconstructs the global expert tensor
  by stacking shards in the canonical interleaved order
  (g = R + i * ep_degree). The reconstruction must equal an oracle
  built directly from the stacked shards.

No torch.distributed, no XPU. ~1s.
"""
from __future__ import annotations

import torch

from torchtune.dev.rl.weight_sync import (
    _ws10_build_local_payload,
    _ws10_sort_key_by_rank,
    _ws10_unify_manifests,
    _WS10_FUSED_RE,
)


# --------------------------------------------------------------------
# Fixture: build per-rank synthetic local fused expert shards.
# 4 layers, ep_degree=4, 2 experts per rank → 8 global experts.
# --------------------------------------------------------------------

NUM_LAYERS = 4
EP_DEGREE = 4
N_LOCAL = 2
HIDDEN = 8
INTER_PER_TP = 6  # per-tp slice; w13 has 2*INTER_PER_TP rows


def _layer_global_w13(layer: int) -> torch.Tensor:
    """Oracle global w13 for a layer: shape [n_total_experts, 2*I, H]."""
    n_total = EP_DEGREE * N_LOCAL
    t = torch.arange(
        n_total * 2 * INTER_PER_TP * HIDDEN, dtype=torch.float32,
    ).reshape(n_total, 2 * INTER_PER_TP, HIDDEN) + layer * 100_000.0
    return t.to(torch.bfloat16)


def _layer_global_w2(layer: int) -> torch.Tensor:
    n_total = EP_DEGREE * N_LOCAL
    t = torch.arange(
        n_total * HIDDEN * INTER_PER_TP, dtype=torch.float32,
    ).reshape(n_total, HIDDEN, INTER_PER_TP) + layer * 1_000_000.0
    return t.to(torch.bfloat16)


def _local_shard(global_t: torch.Tensor, R: int) -> torch.Tensor:
    """Trainer R owns global expert ids [R, R+ep_d, R+2*ep_d, ...]."""
    ids = [R + i * EP_DEGREE for i in range(N_LOCAL)]
    return global_t[ids].clone().contiguous()


def _build_per_rank_payload(R: int):
    """Build the dict ``{hf_name: local_shard}`` for trainer rank R."""
    local_experts = {}
    for layer in range(NUM_LAYERS):
        gw13 = _layer_global_w13(layer)
        gw2 = _layer_global_w2(layer)
        local_experts[
            f"model.layers.{layer}.mlp.experts.w13_weight"
        ] = _local_shard(gw13, R)
        local_experts[
            f"model.layers.{layer}.mlp.experts.w2_weight"
        ] = _local_shard(gw2, R)
    return local_experts


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------

def test_per_rank_payload_carries_correct_tags():
    for R in range(EP_DEGREE):
        local_experts = _build_per_rank_payload(R)
        cpu_batches, meta = _ws10_build_local_payload(
            local_experts, ep_rank=R, ep_degree=EP_DEGREE,
            batch_max_numel=1 << 20,
        )
        # Every entry tagged with R, ep_degree, n_local
        for e in meta:
            assert e["trainer_ep_rank"] == R
            assert e["ep_degree"] == EP_DEGREE
            assert e["n_local"] == N_LOCAL
        # Both kinds for every layer
        kinds_per_layer: dict[int, set[str]] = {}
        for e in meta:
            m = _WS10_FUSED_RE.match(e["name"])
            assert m, f"non-fused name leaked into payload: {e['name']}"
            li = int(m.group(1))
            kinds_per_layer.setdefault(li, set()).add(m.group(2))
        for li in range(NUM_LAYERS):
            assert kinds_per_layer[li] == {"w13", "w2"}
        # cpu_batches non-empty
        assert sum(b.numel() for b in cpu_batches) > 0


def test_unify_groups_by_rank_then_layer_then_kind():
    per_rank_metas = []
    for R in range(EP_DEGREE):
        local_experts = _build_per_rank_payload(R)
        _, meta = _ws10_build_local_payload(
            local_experts, ep_rank=R, ep_degree=EP_DEGREE,
            batch_max_numel=1 << 20,
        )
        per_rank_metas.append(meta)
    # Add a synthetic non-expert tail (no trainer_ep_rank tag) — this is
    # what rank 0 appends in the WS10 sender path. The production code
    # merges these into rank 0's manifest before unify (the unify helper
    # validates len(per_rank_metas) == ep_degree).
    non_expert_meta = [
        {"name": "model.embed_tokens.weight", "shape": [10, 8], "numel": 80},
        {"name": "model.norm.weight", "shape": [8], "numel": 8},
    ]
    per_rank_metas[0] = per_rank_metas[0] + non_expert_meta
    unified = _ws10_unify_manifests(
        per_rank_metas, sort_key=_ws10_sort_key_by_rank,
    )

    # Walk the unified list and check rank monotonicity
    cur_R = -1
    for e in unified:
        # Non-expert entries default to R=0 in the by-rank key
        R = int(e.get("trainer_ep_rank", 0))
        assert R >= cur_R, f"rank order broken at {e['name']}: R={R} < {cur_R}"
        cur_R = R

    # All R=0 fused entries must come before any R=1 entries
    r0_fused = [e for e in unified if e.get("trainer_ep_rank") == 0]
    r1_any = [e for e in unified if e.get("trainer_ep_rank") == 1]
    if r1_any:
        last_r0_fused_idx = unified.index(r0_fused[-1])
        first_r1_idx = unified.index(r1_any[0])
        assert last_r0_fused_idx < first_r1_idx

    # Within rank 0: fused entries (w13/w2) must come BEFORE non-expert
    # entries (the non-expert tail trails the rank's experts).
    r0_block = [
        e for e in unified
        if int(e.get("trainer_ep_rank", 0)) == 0
    ]
    saw_non_expert = False
    for e in r0_block:
        if _WS10_FUSED_RE.match(e["name"]):
            assert not saw_non_expert, (
                f"fused entry {e['name']} appeared after a non-expert entry "
                f"in rank 0's block")
        else:
            saw_non_expert = True
    # And we DID see the non-expert tail in rank 0's block
    assert saw_non_expert, "non-expert tail did not land in rank 0 block"

    # Within each rank's expert block: layers ascending, w13 before w2
    for R in range(EP_DEGREE):
        block = [
            e for e in unified
            if e.get("trainer_ep_rank") == R
        ]
        # Filter to fused only
        block = [e for e in block if _WS10_FUSED_RE.match(e["name"])]
        cur_layer = -1
        cur_kind_rank = -1
        for e in block:
            m = _WS10_FUSED_RE.match(e["name"])
            li = int(m.group(1))
            kr = 0 if m.group(2) == "w13" else 1
            if li == cur_layer:
                assert kr >= cur_kind_rank, (
                    f"R={R} layer {li}: w2 appeared before w13"
                )
            else:
                assert li > cur_layer, (
                    f"R={R} layer order broken: {li} after {cur_layer}"
                )
            cur_layer = li
            cur_kind_rank = kr


def test_round_trip_reconstructs_oracle_global_tensor():
    """Simulated receiver: walk the unified manifest, accumulate
    per-(R, kind) shards by layer, then assemble the global tensor in
    canonical interleaved order. Must equal the oracle global tensor.
    """
    per_rank_payloads = {}
    per_rank_metas = []
    for R in range(EP_DEGREE):
        local_experts = _build_per_rank_payload(R)
        _, meta = _ws10_build_local_payload(
            local_experts, ep_rank=R, ep_degree=EP_DEGREE,
            batch_max_numel=1 << 20,
        )
        per_rank_payloads[R] = local_experts
        per_rank_metas.append(meta)
    unified = _ws10_unify_manifests(
        per_rank_metas, sort_key=_ws10_sort_key_by_rank,
    )

    # Receiver state: layer -> kind -> {R: shard}
    accum: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
    for e in unified:
        m = _WS10_FUSED_RE.match(e["name"])
        if not m:
            continue
        li = int(m.group(1))
        kind = m.group(2)
        R = int(e["trainer_ep_rank"])
        # Receiver pulls this shard from the trainer rank R's payload.
        shard = per_rank_payloads[R][e["name"]]
        accum.setdefault(li, {}).setdefault(kind, {})[R] = shard

    # Assemble global tensor per layer; compare to oracle.
    for layer in range(NUM_LAYERS):
        for kind, oracle_fn in (("w13", _layer_global_w13),
                                ("w2", _layer_global_w2)):
            assert accum[layer][kind].keys() == set(range(EP_DEGREE))
            sample = next(iter(accum[layer][kind].values()))
            n_total = EP_DEGREE * N_LOCAL
            global_t = torch.empty(
                n_total, *sample.shape[1:], dtype=sample.dtype,
            )
            for R in range(EP_DEGREE):
                shard = accum[layer][kind][R]
                for i in range(N_LOCAL):
                    g = R + i * EP_DEGREE
                    global_t[g].copy_(shard[i])
            oracle = oracle_fn(layer)
            assert torch.equal(global_t, oracle), (
                f"layer {layer} {kind} reconstruction mismatch"
            )


def test_unify_default_sort_unchanged():
    """Off-path safety: calling _ws10_unify_manifests with no sort_key
    must produce the SAME ordering as before Commit B (sort by
    layer/kind/R/name)."""
    per_rank_metas = []
    for R in range(EP_DEGREE):
        local_experts = _build_per_rank_payload(R)
        _, meta = _ws10_build_local_payload(
            local_experts, ep_rank=R, ep_degree=EP_DEGREE,
            batch_max_numel=1 << 20,
        )
        per_rank_metas.append(meta)
    unified_default = _ws10_unify_manifests(per_rank_metas)
    # Default order: layer ascending, w13 before w2, then R.
    cur_layer = -1
    cur_kind = -1
    cur_R = -1
    for e in unified_default:
        m = _WS10_FUSED_RE.match(e["name"])
        assert m
        li = int(m.group(1))
        kr = 0 if m.group(2) == "w13" else 1
        R = int(e["trainer_ep_rank"])
        if li > cur_layer:
            cur_layer = li
            cur_kind = -1
            cur_R = -1
        elif li == cur_layer:
            if kr > cur_kind:
                cur_kind = kr
                cur_R = -1
            elif kr == cur_kind:
                assert R >= cur_R
                cur_R = R
            else:
                raise AssertionError(
                    f"kind order broken at layer {li}: kr={kr} < {cur_kind}")
        else:
            raise AssertionError(
                f"layer order broken: {li} after {cur_layer}")

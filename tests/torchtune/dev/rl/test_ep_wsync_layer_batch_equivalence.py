"""Equivalence test for EP wsync per-layer batched AllGather (WS6).

Pin-down test for the slice/unshuffle math behind
``TORCHTUNE_EP_WSYNC_LAYER_BATCH=1`` in
``torchtune/dev/rl/weight_sync.py::_sync_weights_to_vllm_xccl``.

The default EP MoE wsync path issues one ``all_gather`` per expert tensor
(48 layers * 3 projections = 144 collectives per round on
``_shard_pg``). The opt-in path concatenates the 3 expert projections of
each layer into one ``all_gather_into_tensor`` (48 collectives per
round). The full per-projection tensors fed into the existing
``_ep_stream_buf`` fuse path must be bit-exact equal to the per-projection
result, otherwise the fused w13/w2 tensors broadcast to vLLM are
silently corrupted.

This test does the math in pure Python (no ``torch.distributed``) so it
runs on a login node in ~1s.
"""
from __future__ import annotations

import pytest
import torch


def _reference_path(local_shards_per_rank):
    """Per-projection AG: 3 separate all_gathers per layer.

    Mirrors ``torchtune/dev/rl/weight_sync.py`` lines 1731-1741.
    """
    ep_d = len(local_shards_per_rank)
    results: dict = {}
    for layer in local_shards_per_rank[0]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            parts = [local_shards_per_rank[r][layer][proj] for r in range(ep_d)]
            stk = torch.stack(parts, dim=0)
            full = stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()
            results.setdefault(layer, {})[proj] = full
    return results


def _batched_path(local_shards_per_rank):
    """Layer-batched AG: 1 all_gather_into_tensor over concatenated 3-proj local.

    Simulates the planned WS6 hot path. Per rank, builds
    ``local_cat = cat(gate.flatten(), up.flatten(), down.flatten())``,
    'all-gathers' (in-test: torch.cat across ranks), slices each rank's
    chunk back into 3 per-projection sub-tensors, then stacks + unshuffles
    each projection identically to the reference path.
    """
    ep_d = len(local_shards_per_rank)
    results: dict = {}
    for layer in local_shards_per_rank[0]:
        local_cats = []
        sizes = None
        shapes = None
        for r in range(ep_d):
            shards = local_shards_per_rank[r][layer]
            g, u, d = shards["gate_proj"], shards["up_proj"], shards["down_proj"]
            local_cats.append(torch.cat([g.flatten(), u.flatten(), d.flatten()]))
            sizes_r = (g.numel(), u.numel(), d.numel())
            shapes_r = (g.shape, u.shape, d.shape)
            if sizes is None:
                sizes = sizes_r
                shapes = shapes_r
            else:
                # Uniform-shape requirement across ranks for a layer.
                assert sizes == sizes_r
                assert shapes == shapes_r

        cat_size = local_cats[0].numel()
        # Simulated all_gather_into_tensor:
        # out_cat layout [rank0_cat | rank1_cat | ... | rank_{W-1}_cat]
        out_cat = torch.cat(local_cats, dim=0)
        n_g, n_u, n_d = sizes
        g_shape, u_shape, d_shape = shapes

        gate_parts, up_parts, down_parts = [], [], []
        for r in range(ep_d):
            base = r * cat_size
            g_flat = out_cat[base : base + n_g]
            u_flat = out_cat[base + n_g : base + n_g + n_u]
            d_flat = out_cat[base + n_g + n_u : base + n_g + n_u + n_d]
            gate_parts.append(g_flat.reshape(g_shape))
            up_parts.append(u_flat.reshape(u_shape))
            down_parts.append(d_flat.reshape(d_shape))

        for proj_name, parts in (
            ("gate_proj", gate_parts),
            ("up_proj", up_parts),
            ("down_proj", down_parts),
        ):
            stk = torch.stack(parts, dim=0)
            full = stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()
            results.setdefault(layer, {})[proj_name] = full
    return results


def _build_local_shards(ep_d: int, n_local: int, intermediate: int, hidden: int,
                        n_layers: int, seed: int = 42):
    """Synthetic per-rank shards mimicking Qwen3-30B-A3B layout shape."""
    torch.manual_seed(seed)
    shards: list = []
    for r in range(ep_d):
        rank_dict: dict = {}
        for L in range(n_layers):
            rank_dict[L] = {
                "gate_proj": torch.randn(n_local, intermediate, hidden,
                                          dtype=torch.bfloat16),
                "up_proj":   torch.randn(n_local, intermediate, hidden,
                                          dtype=torch.bfloat16),
                "down_proj": torch.randn(n_local, hidden, intermediate,
                                          dtype=torch.bfloat16),
            }
        shards.append(rank_dict)
    return shards


@pytest.mark.parametrize(
    "ep_d, n_local, intermediate, hidden, n_layers",
    [
        (4, 4, 8, 16, 2),
        (8, 2, 12, 24, 3),
        (16, 8, 16, 32, 2),
    ],
)
def test_layer_batched_path_is_bit_exact(ep_d, n_local, intermediate, hidden,
                                         n_layers):
    """The WS6 layer-batched AG must produce bit-exact full tensors per projection."""
    shards = _build_local_shards(ep_d, n_local, intermediate, hidden, n_layers)
    ref = _reference_path(shards)
    new = _batched_path(shards)

    for L in range(n_layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            assert ref[L][p].shape == new[L][p].shape, (
                f"Shape mismatch L={L} proj={p}: ref={ref[L][p].shape}, "
                f"new={new[L][p].shape}"
            )
            assert torch.equal(ref[L][p], new[L][p]), (
                f"Bit-exact mismatch L={L} proj={p}"
            )


def test_full_tensor_total_experts_matches_ep_d_times_n_local():
    """Sanity: post-unshuffle, the leading dim must be ep_d * n_local."""
    ep_d, n_local = 4, 4
    shards = _build_local_shards(ep_d, n_local, 8, 16, n_layers=1)
    ref = _reference_path(shards)
    new = _batched_path(shards)
    assert ref[0]["gate_proj"].shape[0] == ep_d * n_local
    assert new[0]["gate_proj"].shape[0] == ep_d * n_local


def test_unshuffle_preserves_interleaved_expert_order():
    """The expert-i row of the full tensor for layer L must come from rank
    (i % ep_d), local-expert (i // ep_d) - the interleaved EP split documented
    at weight_sync.py line 1722.

    This is the single most important contract: the receiver indexes experts
    by global ID, so any permutation here silently scrambles MoE routing.
    """
    ep_d, n_local = 4, 3
    shards = _build_local_shards(ep_d, n_local, 5, 7, n_layers=1)
    ref = _reference_path(shards)
    new = _batched_path(shards)

    total = ep_d * n_local
    for global_i in range(total):
        rank = global_i % ep_d
        local_i = global_i // ep_d
        expected = shards[rank][0]["gate_proj"][local_i]
        assert torch.equal(ref[0]["gate_proj"][global_i], expected), (
            f"reference unshuffle wrong at global_i={global_i}"
        )
        assert torch.equal(new[0]["gate_proj"][global_i], expected), (
            f"layer-batched unshuffle wrong at global_i={global_i}"
        )

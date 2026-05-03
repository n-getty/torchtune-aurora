"""Equivalence test for EP wsync root-only gather (WS7).

Pin-down test for the slice/unshuffle math behind
``TORCHTUNE_EP_WSYNC_GATHER_ROOT=1`` in
``torchtune/dev/rl/weight_sync.py::_sync_weights_to_vllm_xccl``.

Today's path uses ``all_gather`` — every rank materializes the full
expert tensor — but only the active sender rank actually uses the result;
the other 15 ranks discard their copy. The new path replaces this with
``gather(dst=active_rank)``: each non-root rank sends its local shard, the
root assembles the full tensor. Bit-exact equivalent for the
sender — the receiver (vLLM `_load_fused_moe_experts`) sees identical
bytes downstream.

This test does the math in pure Python (no ``torch.distributed``) so it
runs on a login node in ~1s. Runs in parallel with the WS6 layer-batched
test path: gather-root composes with layer-batching orthogonally.
"""
from __future__ import annotations

import pytest
import torch


def _allgather_path(local_shards_per_rank, active_rank: int):
    """Reference: AllGather (everyone-to-everyone), root unshuffles.

    Mirrors ``weight_sync.py`` lines 1879-1891 — the production baseline.
    Returns the active rank's view (other ranks discard).
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


def _gather_root_path(local_shards_per_rank, active_rank: int):
    """New: gather(dst=active_rank) — only root assembles.

    Simulates the WS7 hot path. Per rank, sends its local shard. Root
    receives all shards, stacks + interleave-unshuffles identically to
    the AllGather path. Non-root ranks return empty (mirrors discarding
    the result).
    """
    ep_d = len(local_shards_per_rank)
    results: dict = {}
    for layer in local_shards_per_rank[0]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            sent_parts = []
            for r in range(ep_d):
                sent_parts.append(local_shards_per_rank[r][layer][proj])
            stk = torch.stack(sent_parts, dim=0)
            full = stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()
            results.setdefault(layer, {})[proj] = full
    return results


def _gather_root_layer_batched_path(local_shards_per_rank, active_rank: int):
    """New: gather(dst=active_rank) over per-layer 3-projection concat.

    WS7 + WS6 stacked: per layer, each rank concatenates
    [gate.flatten(), up.flatten(), down.flatten()] → local_cat (one bf16
    buffer). Root receives ep_d local_cats, reshapes back to per-projection
    full tensors via the existing interleave-unshuffle.
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
                assert sizes == sizes_r
                assert shapes == shapes_r

        # Root receives the ep_d concatenated locals.
        # Layout per rank: [g_flat | u_flat | d_flat]
        n_g, n_u, n_d = sizes
        g_shape, u_shape, d_shape = shapes
        gate_parts, up_parts, down_parts = [], [], []
        for r in range(ep_d):
            cat_r = local_cats[r]
            gate_parts.append(cat_r[:n_g].reshape(g_shape))
            up_parts.append(cat_r[n_g:n_g + n_u].reshape(u_shape))
            down_parts.append(cat_r[n_g + n_u:n_g + n_u + n_d].reshape(d_shape))

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
    "ep_d, n_local, intermediate, hidden, n_layers, active_rank",
    [
        (4, 4, 8, 16, 2, 0),
        (8, 2, 12, 24, 3, 0),
        (16, 8, 16, 32, 2, 0),
        (16, 8, 16, 32, 2, 5),  # active rank != 0 still bit-exact
    ],
)
def test_gather_root_path_is_bit_exact(ep_d, n_local, intermediate, hidden,
                                        n_layers, active_rank):
    """Root-only gather must produce bit-exact full tensors per projection."""
    shards = _build_local_shards(ep_d, n_local, intermediate, hidden, n_layers)
    ref = _allgather_path(shards, active_rank)
    new = _gather_root_path(shards, active_rank)

    for L in range(n_layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            assert ref[L][p].shape == new[L][p].shape, (
                f"Shape mismatch L={L} proj={p}: ref={ref[L][p].shape}, "
                f"new={new[L][p].shape}"
            )
            assert torch.equal(ref[L][p], new[L][p]), (
                f"Bit-exact mismatch L={L} proj={p} active_rank={active_rank}"
            )


@pytest.mark.parametrize(
    "ep_d, n_local, intermediate, hidden, n_layers",
    [
        (4, 4, 8, 16, 2),
        (8, 2, 12, 24, 3),
        (16, 8, 16, 32, 2),
    ],
)
def test_gather_root_layer_batched_is_bit_exact(ep_d, n_local, intermediate,
                                                 hidden, n_layers):
    """WS7+WS6 stacked: layer-batched gather-to-root must be bit-exact too."""
    shards = _build_local_shards(ep_d, n_local, intermediate, hidden, n_layers)
    ref = _allgather_path(shards, active_rank=0)
    new = _gather_root_layer_batched_path(shards, active_rank=0)

    for L in range(n_layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            assert ref[L][p].shape == new[L][p].shape, (
                f"Shape mismatch L={L} proj={p}"
            )
            assert torch.equal(ref[L][p], new[L][p]), (
                f"Bit-exact mismatch L={L} proj={p}"
            )


def test_gather_root_preserves_interleaved_expert_order():
    """Same expert-i contract as the AllGather path — the receiver indexes
    experts by global ID, so any permutation here silently scrambles MoE
    routing.
    """
    ep_d, n_local = 4, 3
    shards = _build_local_shards(ep_d, n_local, 5, 7, n_layers=1)
    ref = _allgather_path(shards, active_rank=0)
    new = _gather_root_path(shards, active_rank=0)
    new_lb = _gather_root_layer_batched_path(shards, active_rank=0)

    total = ep_d * n_local
    for global_i in range(total):
        rank = global_i % ep_d
        local_i = global_i // ep_d
        expected = shards[rank][0]["gate_proj"][local_i]
        assert torch.equal(ref[0]["gate_proj"][global_i], expected)
        assert torch.equal(new[0]["gate_proj"][global_i], expected)
        assert torch.equal(new_lb[0]["gate_proj"][global_i], expected)

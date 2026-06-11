"""Equivalence test for EP wsync into-tensor AllGather conversion.

Pin-down test for the bit-exact slice/unshuffle math behind the
list-style ``torch.distributed.all_gather`` -> ``all_gather_into_tensor``
conversion in
``torchtune/dev/rl/weight_sync.py::_sync_weights_to_vllm_xccl``
(the EP MoE per-projection path).

Motivation: ProcessGroupXCCL on Aurora's XPU stack allocates a hidden
flat ``newLikeFlat()`` temp inside list-style ``all_gather``. That temp
is leaked to Level Zero whenever ``torch.xpu.empty_cache()`` runs in the
same training loop (upstream: Aurora #143 / torch-xpu-ops #3744). The
``all_gather_into_tensor`` API does not allocate a hidden temp -- the
caller owns the flat output buffer -- so converting all list-style call
sites in this path eliminates the leak and removes a latent footgun
that would block any future ``empty_cache()`` reintroduction.

This test verifies the conversion is bit-exact for:
  * bf16 baseline path (default EP MoE per-projection AG)
  * fp8-wire path (TORCHTUNE_EP_WSYNC_FP8_WIRE=1 shards+scales)

It runs in pure Python (no torch.distributed) on a login node in ~1s.
"""
from __future__ import annotations

import pytest
import torch


def _list_path_bf16(local_per_rank: list[torch.Tensor]) -> torch.Tensor:
    """Reference: old list-style all_gather + stack(dim=0) + unshuffle.

    Mirrors the pre-patch path in weight_sync.py (the ``else`` branch
    at the bottom of the EP per-projection block).
    """
    ep_d = len(local_per_rank)
    parts = [t.contiguous() for t in local_per_rank]
    stk = torch.stack(parts, dim=0)  # [ep_d, n_local, ...]
    return stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()


def _into_tensor_path_bf16(local_per_rank: list[torch.Tensor]) -> torch.Tensor:
    """New: all_gather_into_tensor into pre-allocated flat buffer.

    Simulated AG: torch.cat(list, dim=0) gives the same layout that
    all_gather_into_tensor would produce (rank-major).
    """
    ep_d = len(local_per_rank)
    local_c = local_per_rank[0].contiguous()
    n_local = local_c.shape[0]
    out_cat = torch.empty(
        ep_d * n_local, *local_c.shape[1:],
        dtype=local_c.dtype, device=local_c.device,
    )
    # Simulated AG: each rank's local fills its slice.
    for r in range(ep_d):
        out_cat[r * n_local : (r + 1) * n_local].copy_(local_per_rank[r])

    stk = out_cat.view(ep_d, n_local, *out_cat.shape[1:])
    return stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()


def _list_path_fp8(
    local_fp8_per_rank: list[torch.Tensor],
    local_sc_per_rank: list[torch.Tensor],
) -> torch.Tensor:
    """Reference fp8-wire: list AGs for fp8 + scales, per-rank dequant, stack, unshuffle."""
    ep_d = len(local_fp8_per_rank)
    fp8_parts = [t.contiguous() for t in local_fp8_per_rank]
    sc_parts = [t.contiguous() for t in local_sc_per_rank]
    bf16_parts = []
    for r in range(ep_d):
        bf16_parts.append(
            (fp8_parts[r].to(torch.float32) * sc_parts[r]).to(torch.bfloat16)
        )
    stk = torch.stack(bf16_parts, dim=0)
    return stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()


def _into_tensor_path_fp8(
    local_fp8_per_rank: list[torch.Tensor],
    local_sc_per_rank: list[torch.Tensor],
) -> torch.Tensor:
    """New fp8-wire: into-tensor AGs for fp8 + scales, view as [ep_d, n_local, ...], dequant, unshuffle."""
    ep_d = len(local_fp8_per_rank)
    local_fp8 = local_fp8_per_rank[0].contiguous()
    local_sc = local_sc_per_rank[0].contiguous()

    n_fp8 = local_fp8.shape[0]
    n_sc = local_sc.shape[0]

    out_fp8 = torch.empty(
        ep_d * n_fp8, *local_fp8.shape[1:],
        dtype=local_fp8.dtype, device=local_fp8.device,
    )
    out_sc = torch.empty(
        ep_d * n_sc, *local_sc.shape[1:],
        dtype=local_sc.dtype, device=local_sc.device,
    )
    for r in range(ep_d):
        out_fp8[r * n_fp8 : (r + 1) * n_fp8].copy_(local_fp8_per_rank[r])
        out_sc[r * n_sc : (r + 1) * n_sc].copy_(local_sc_per_rank[r])

    fp8_view = out_fp8.view(ep_d, n_fp8, *out_fp8.shape[1:])
    sc_view = out_sc.view(ep_d, n_sc, *out_sc.shape[1:])

    bf16_parts = []
    for r in range(ep_d):
        bf16_parts.append(
            (fp8_view[r].to(torch.float32) * sc_view[r]).to(torch.bfloat16)
        )
    stk = torch.stack(bf16_parts, dim=0)
    return stk.transpose(0, 1).reshape(-1, *stk.shape[2:]).contiguous()


def _build_shards(ep_d: int, n_local: int, intermediate: int, hidden: int,
                  seed: int = 42) -> list[torch.Tensor]:
    """Synthetic per-rank shards mimicking Qwen3-30B-A3B layout for one projection."""
    torch.manual_seed(seed)
    return [
        torch.randn(n_local, intermediate, hidden, dtype=torch.bfloat16)
        for _ in range(ep_d)
    ]


@pytest.mark.parametrize(
    "ep_d, n_local, intermediate, hidden",
    [
        (4, 4, 8, 16),
        (8, 2, 12, 24),
        (16, 8, 16, 32),
    ],
)
def test_bf16_into_tensor_path_is_bit_exact(ep_d, n_local, intermediate, hidden):
    """The into_tensor AG conversion must produce a bit-exact full tensor vs the list path."""
    shards = _build_shards(ep_d, n_local, intermediate, hidden)
    ref = _list_path_bf16(shards)
    new = _into_tensor_path_bf16(shards)

    assert ref.shape == new.shape, f"shape mismatch ref={ref.shape} new={new.shape}"
    assert torch.equal(ref, new), "bf16 into-tensor unshuffle is not bit-exact"


def test_bf16_full_tensor_total_experts_matches_ep_d_times_n_local():
    """Post-unshuffle, leading dim must be ep_d * n_local."""
    ep_d, n_local = 4, 4
    shards = _build_shards(ep_d, n_local, 8, 16)
    ref = _list_path_bf16(shards)
    new = _into_tensor_path_bf16(shards)
    assert ref.shape[0] == ep_d * n_local
    assert new.shape[0] == ep_d * n_local


def test_bf16_unshuffle_preserves_interleaved_expert_order():
    """Global expert i must come from rank (i % ep_d), local-expert (i // ep_d).

    This is the EP slice contract -- the receiver indexes experts by global ID,
    so any permutation here silently scrambles MoE routing.
    """
    ep_d, n_local = 4, 3
    shards = _build_shards(ep_d, n_local, 5, 7)
    ref = _list_path_bf16(shards)
    new = _into_tensor_path_bf16(shards)

    total = ep_d * n_local
    for global_i in range(total):
        rank = global_i % ep_d
        local_i = global_i // ep_d
        expected = shards[rank][local_i]
        assert torch.equal(ref[global_i], expected), (
            f"reference unshuffle wrong at global_i={global_i}"
        )
        assert torch.equal(new[global_i], expected), (
            f"into-tensor unshuffle wrong at global_i={global_i}"
        )


def _build_fp8_shards(ep_d: int, n_local: int, intermediate: int, hidden: int,
                      seed: int = 13):
    """Per-rank fp8 shards + per-row scales mimicking the WS8 fp8-wire local cast."""
    torch.manual_seed(seed)
    fp8_per_rank = []
    sc_per_rank = []
    for _ in range(ep_d):
        # Synthetic local bf16 expert shard
        local_bf = torch.randn(n_local, intermediate, hidden, dtype=torch.bfloat16)
        local_f32 = local_bf.to(torch.float32)
        # Per-output-row scale on last dim (matches WS8 path).
        row_amax = local_f32.abs().amax(dim=-1, keepdim=True)
        scale = (row_amax / 448.0).clamp(min=1e-12)
        fp8 = ((local_f32 / scale).clamp(-448.0, 448.0)
               .to(torch.float8_e4m3fn).contiguous())
        fp8_per_rank.append(fp8)
        sc_per_rank.append(scale.to(torch.float32).contiguous())
    return fp8_per_rank, sc_per_rank


@pytest.mark.parametrize(
    "ep_d, n_local, intermediate, hidden",
    [
        (4, 4, 8, 16),
        (8, 2, 12, 24),
        (16, 4, 16, 32),
    ],
)
def test_fp8_wire_into_tensor_path_is_bit_exact(ep_d, n_local, intermediate, hidden):
    """The fp8-wire into_tensor conversion must produce a bit-exact final tensor vs list path."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn not available on this build")
    fp8_shards, sc_shards = _build_fp8_shards(ep_d, n_local, intermediate, hidden)
    ref = _list_path_fp8(fp8_shards, sc_shards)
    new = _into_tensor_path_fp8(fp8_shards, sc_shards)

    assert ref.shape == new.shape, f"shape mismatch ref={ref.shape} new={new.shape}"
    assert torch.equal(ref, new), "fp8-wire into-tensor unshuffle is not bit-exact"


def test_into_tensor_layout_matches_torch_stack_dim0():
    """Sanity: out_cat.view(ep_d, n_local, ...) must equal torch.stack(list, dim=0).

    This is the core layout assumption the patch relies on -- both APIs lay
    rank-major contiguous blocks of ``n_local`` rows. If a future torch version
    diverges, the unshuffle math breaks silently.
    """
    ep_d, n_local = 8, 5
    shards = _build_shards(ep_d, n_local, 6, 9)
    stacked = torch.stack([t.contiguous() for t in shards], dim=0)

    n = shards[0].shape[0]
    out_cat = torch.empty(
        ep_d * n, *shards[0].shape[1:],
        dtype=shards[0].dtype,
    )
    for r in range(ep_d):
        out_cat[r * n : (r + 1) * n].copy_(shards[r])
    viewed = out_cat.view(ep_d, n, *out_cat.shape[1:])

    assert torch.equal(stacked, viewed), (
        "out_cat.view(ep_d, n_local, ...) diverged from torch.stack(list, 0); "
        "the AG conversion's unshuffle math assumes these are byte-identical"
    )

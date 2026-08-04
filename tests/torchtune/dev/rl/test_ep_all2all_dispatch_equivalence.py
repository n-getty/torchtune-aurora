# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""EP all_to_all dispatch ≡ AllGather+ReduceScatter dispatch (CPU/gloo).

Track A discipline gate. `TORCHTUNE_EP_ALL2ALL=1` replaces the AllGather+
ReduceScatter EP token dispatch/combine with a true all_to_all_single dispatch
(`ExpertParallel._token_dispatch_all2all` / `_token_combine_all2all`). This test
proves the two paths are numerically equivalent — SAME expert outputs AND SAME
gradients w.r.t. both the routed input and the expert weights — across several
(num_experts, ep_degree) configs, run over gloo on CPU (no XPU needed).

If this regresses, EP > 1 under TORCHTUNE_EP_ALL2ALL silently trains on permuted
tokens/grads. Mirrors the intent of test_ep_slice_contract.py but exercises the
actual dispatch/combine math end-to-end through GroupedExperts.

Run: pytest tests/torchtune/dev/rl/test_ep_all2all_dispatch_equivalence.py --timeout=120
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtune.modules.moe._parallelism import (
    _build_all_to_all_metadata,
    _build_all_to_all_metadata_vectorized,
)


@pytest.mark.parametrize("ep_degree,num_experts", [(2, 8), (4, 16), (8, 128)])
def test_vectorized_metadata_matches_reference(ep_degree, num_experts):
    generator = torch.Generator().manual_seed(7)
    all_ntpe = torch.randint(
        0, 12, (ep_degree, num_experts), generator=generator
    )
    for ep_rank in range(ep_degree):
        reference = _build_all_to_all_metadata(all_ntpe, ep_rank)
        vectorized = _build_all_to_all_metadata_vectorized(all_ntpe, ep_rank)
        torch.testing.assert_close(vectorized[0], reference[0])
        torch.testing.assert_close(vectorized[1], reference[1])
        assert vectorized[2] == reference[2]
        assert vectorized[3] == reference[3]
        torch.testing.assert_close(
            vectorized[4], torch.tensor(reference[4], dtype=torch.long)
        )
        torch.testing.assert_close(
            torch.sort(vectorized[0]).values,
            torch.arange(vectorized[0].numel(), dtype=torch.long),
        )
        torch.testing.assert_close(
            torch.sort(vectorized[1]).values,
            torch.arange(vectorized[1].numel(), dtype=torch.long),
        )


@pytest.mark.parametrize("ep_degree,num_experts", [(2, 4), (4, 8), (8, 16)])
def test_vectorized_metadata_matches_empty_and_uneven_counts(ep_degree, num_experts):
    cases = [
        torch.zeros(ep_degree, num_experts, dtype=torch.long),
        torch.tensor(
            [
                [
                    (row + column * 3) % (column + 2)
                    for column in range(num_experts)
                ]
                for row in range(ep_degree)
            ],
            dtype=torch.long,
        ),
    ]
    for all_ntpe in cases:
        for ep_rank in range(ep_degree):
            reference = _build_all_to_all_metadata(all_ntpe, ep_rank)
            vectorized = _build_all_to_all_metadata_vectorized(all_ntpe, ep_rank)
            torch.testing.assert_close(vectorized[0], reference[0])
            torch.testing.assert_close(vectorized[1], reference[1])
            assert vectorized[2] == reference[2]
            assert vectorized[3] == reference[3]
            torch.testing.assert_close(
                vectorized[4], torch.tensor(reference[4], dtype=torch.long)
            )


def test_cpu_vectorized_metadata_matches_reference_contract():
    from torchtune.modules.moe import _parallelism as ep_mod

    counts = torch.tensor(
        [[3, 0, 1, 0, 2, 1, 0, 4], [0, 2, 0, 3, 1, 0, 2, 0]],
        dtype=torch.long,
    )
    original = ep_mod._CPU_VECTOR_ROUTING_METADATA
    try:
        ep_mod._CPU_VECTOR_ROUTING_METADATA = True
        for ep_rank in range(counts.shape[0]):
            reference = _build_all_to_all_metadata(counts, ep_rank)
            vectorized = _build_all_to_all_metadata_vectorized(counts, ep_rank)
            torch.testing.assert_close(vectorized[0], reference[0])
            torch.testing.assert_close(vectorized[1], reference[1])
            assert vectorized[2] == reference[2]
            assert vectorized[3] == reference[3]
            torch.testing.assert_close(
                vectorized[4], torch.tensor(reference[4], dtype=torch.long)
            )
    finally:
        ep_mod._CPU_VECTOR_ROUTING_METADATA = original


def test_metadata_control_flags_have_distinct_cpu_fallbacks():
    from torchtune.modules.moe import _parallelism as ep_mod

    counts = torch.tensor([[2, 0, 1, 3], [0, 4, 2, 0]], dtype=torch.long)
    original_device = ep_mod._DEVICE_ROUTING_METADATA
    original_cpu_vector = ep_mod._CPU_VECTOR_ROUTING_METADATA
    try:
        ep_mod._DEVICE_ROUTING_METADATA = False
        ep_mod._CPU_VECTOR_ROUTING_METADATA = False
        reference = _build_all_to_all_metadata(counts, ep_rank=1)
        vectorized = _build_all_to_all_metadata_vectorized(counts, ep_rank=1)
        assert reference[2] == vectorized[2]
        assert reference[3] == vectorized[3]
    finally:
        ep_mod._DEVICE_ROUTING_METADATA = original_device
        ep_mod._CPU_VECTOR_ROUTING_METADATA = original_cpu_vector


@pytest.mark.parametrize("ep_degree,num_experts,ep_rank", [(2, 8, 0), (4, 16, 3)])
def test_expert_parallel_topology_index_cache_is_reused_and_separated(
    ep_degree, num_experts, ep_rank
):
    from torchtune.modules.moe._parallelism import (
        ExpertParallel,
        _build_ag_dispatch_metadata_vectorized,
    )

    parallel = ExpertParallel()
    first = parallel._topology_indices(
        ep_degree, num_experts, ep_rank, torch.device("cpu")
    )
    second = parallel._topology_indices(
        ep_degree, num_experts, ep_rank, torch.device("cpu")
    )
    assert first[0] is second[0]
    assert first[1] is second[1]

    other_rank = (ep_rank + 1) % ep_degree
    other = parallel._topology_indices(
        ep_degree, num_experts, other_rank, torch.device("cpu")
    )
    assert other[1] is not first[1]
    assert not torch.equal(other[1], first[1])

    counts = torch.randint(0, 12, (ep_degree, num_experts), dtype=torch.long)
    cached = _build_all_to_all_metadata_vectorized(
        counts, ep_rank, first
    )
    uncached = _build_all_to_all_metadata_vectorized(counts, ep_rank)
    for cached_value, uncached_value in zip(cached[:2], uncached[:2]):
        torch.testing.assert_close(cached_value, uncached_value)
    assert cached[2:4] == uncached[2:4]
    torch.testing.assert_close(cached[4], uncached[4])

    ag_cached = _build_ag_dispatch_metadata_vectorized(
        counts, ep_rank, s_local=17, owned_experts=first[1]
    )
    ag_uncached = _build_ag_dispatch_metadata_vectorized(
        counts, ep_rank, s_local=17
    )
    torch.testing.assert_close(ag_cached[0], ag_uncached[0])
    torch.testing.assert_close(ag_cached[1], ag_uncached[1])


def test_packed_routing_metadata_transfer_matches_separate_transfers():
    from torchtune.modules.moe._parallelism import (
        _materialize_all_to_all_permutations,
    )

    send_perm = torch.tensor([4, 0, 3, 1], dtype=torch.long)
    expert_perm = torch.tensor([2, 1, 0], dtype=torch.long)
    local_ntpe = torch.tensor([2, 1], dtype=torch.long)
    separate = _materialize_all_to_all_permutations(
        send_perm, expert_perm, local_ntpe, torch.device("cpu"), packed_transfer=False
    )
    packed = _materialize_all_to_all_permutations(
        send_perm, expert_perm, local_ntpe, torch.device("cpu"), packed_transfer=True
    )
    for separate_tensor, packed_tensor in zip(separate, packed):
        torch.testing.assert_close(separate_tensor, packed_tensor)


@pytest.mark.parametrize(
    "send_perm,expert_perm,local_ntpe",
    [
        ([], [], []),
        ([5, 0, 4, 2], [1, 0], [3, 0, 1]),
        ([2, 1], [], [0, 2, 0, 1]),
    ],
)
def test_packed_routing_metadata_handles_empty_and_uneven_inputs(
    send_perm, expert_perm, local_ntpe
):
    from torchtune.modules.moe._parallelism import (
        _materialize_all_to_all_permutations,
    )

    inputs = tuple(
        torch.tensor(values, dtype=torch.long)
        for values in (send_perm, expert_perm, local_ntpe)
    )
    separate = _materialize_all_to_all_permutations(
        *inputs, torch.device("cpu"), packed_transfer=False
    )
    packed = _materialize_all_to_all_permutations(
        *inputs, torch.device("cpu"), packed_transfer=True
    )
    for separate_tensor, packed_tensor, input_tensor in zip(
        separate, packed, inputs
    ):
        torch.testing.assert_close(packed_tensor, separate_tensor)
        assert packed_tensor.dtype == input_tensor.dtype
        assert packed_tensor.device == torch.device("cpu")


def test_packed_routing_metadata_preserves_slices_on_device_path():
    from torchtune.modules.moe._parallelism import (
        _materialize_all_to_all_permutations,
    )

    inputs = (
        torch.tensor([4, 0, 3, 1], dtype=torch.long),
        torch.tensor([2, 1, 0], dtype=torch.long),
        torch.tensor([2, 1], dtype=torch.long),
    )
    packed = _materialize_all_to_all_permutations(
        *inputs, torch.device("meta"), packed_transfer=True
    )
    for packed_tensor, input_tensor in zip(packed, inputs):
        assert packed_tensor.shape == input_tensor.shape
        assert packed_tensor.dtype == input_tensor.dtype
        assert packed_tensor.device == torch.device("meta")


@pytest.mark.parametrize("ep_rank", [0, 1, 3])
def test_cpu_metadata_builder_handles_zero_and_uneven_owned_experts(ep_rank):
    from torchtune.modules.moe._parallelism import _build_all_to_all_metadata

    counts = torch.tensor(
        [
            [3, 0, 1, 0, 2, 1, 0, 4],
            [0, 2, 0, 3, 1, 0, 2, 0],
            [1, 1, 0, 0, 0, 4, 1, 2],
            [0, 0, 2, 1, 3, 0, 0, 1],
        ],
        dtype=torch.long,
    )
    send_perm, expert_perm, input_splits, output_splits, local_counts = (
        _build_all_to_all_metadata(counts, ep_rank)
    )
    assert send_perm.numel() == int(counts[ep_rank].sum())
    assert sum(input_splits) == send_perm.numel()
    assert sum(output_splits) == expert_perm.numel()
    assert local_counts[0] >= 0 and local_counts[1] >= 0

    local_experts = [ep_rank, ep_rank + 4]
    expected_expert_perm = []
    source_offsets = []
    for source in range(4):
        source_offsets.append(sum(output_splits[:source]))
    for expert in local_experts:
        for source in range(4):
            prior = sum(int(counts[source, previous]) for previous in local_experts if previous < expert)
            start = source_offsets[source] + prior
            expected_expert_perm.extend(range(start, start + int(counts[source, expert])))
    assert expert_perm.tolist() == expected_expert_perm


@pytest.mark.parametrize("ep_degree,num_experts", [(2, 4), (4, 8), (8, 16)])
def test_cpu_metadata_builder_handles_empty_routing(ep_degree, num_experts):
    from torchtune.modules.moe._parallelism import _build_all_to_all_metadata

    counts = torch.zeros(ep_degree, num_experts, dtype=torch.long)
    for ep_rank in range(ep_degree):
        send_perm, expert_perm, input_splits, output_splits, local_counts = (
            _build_all_to_all_metadata(counts, ep_rank)
        )
        assert send_perm.numel() == 0
        assert expert_perm.numel() == 0
        assert input_splits == [0] * ep_degree
        assert output_splits == [0] * ep_degree
        assert local_counts == [0] * (num_experts // ep_degree)


@pytest.mark.parametrize("ep_degree,num_experts", [(2, 4), (4, 8), (8, 16)])
def test_ag_dispatch_metadata_vectorized_matches_reference(ep_degree, num_experts):
    from torchtune.modules.moe._parallelism import (
        _build_ag_dispatch_metadata_vectorized,
        _build_all_to_all_metadata,
    )

    generator = torch.Generator().manual_seed(19)
    cases = [
        torch.randint(0, 12, (ep_degree, num_experts), generator=generator),
        torch.zeros(ep_degree, num_experts, dtype=torch.long),
    ]
    cases.append(
        torch.tensor(
            [
                [(row + column) % 4 for column in range(num_experts)]
                for row in range(ep_degree)
            ],
            dtype=torch.long,
        )
    )
    s_local = 37
    for all_ntpe in cases:
        for ep_rank in range(ep_degree):
            reference = _build_all_to_all_metadata(all_ntpe, ep_rank)
            gather_idx, local_ntpe = _build_ag_dispatch_metadata_vectorized(
                all_ntpe, ep_rank, s_local
            )
            expected = []
            for local_expert in range(num_experts // ep_degree):
                expert = ep_rank + local_expert * ep_degree
                for source in range(ep_degree):
                    start = source * s_local + int(
                        all_ntpe[source, :expert].sum().item()
                    )
                    count = int(all_ntpe[source, expert].item())
                    expected.extend(range(start, start + count))
            assert gather_idx.tolist() == expected
            assert local_ntpe.tolist() == reference[4]


def test_ag_dispatch_metadata_vectorized_preserves_int32_input():
    from torchtune.modules.moe._parallelism import (
        _build_ag_dispatch_metadata_vectorized,
    )

    counts = torch.tensor(
        [[2, 0, 1, 3], [0, 4, 2, 0]], dtype=torch.int32
    )
    original = counts.clone()
    gather_idx, local_ntpe = _build_ag_dispatch_metadata_vectorized(
        counts, ep_rank=1, s_local=6
    )

    assert counts.dtype == torch.int32
    torch.testing.assert_close(counts, original)
    assert gather_idx.dtype == torch.long
    assert local_ntpe.dtype == torch.long


@pytest.mark.parametrize(
    "starts,begins,counts,expected",
    [
        ([10, 20, 30], [0, 2, 2], [2, 0, 3], [10, 11, 30, 31, 32]),
        ([0, 4], [0, 4], [0, 0], []),
        ([7, 8, 15], [0, 0, 4], [1, 3, 2], [7, 9, 10, 11, 15, 16]),
    ],
)
def test_expand_grouped_ranges_handles_zero_and_uneven_groups(
    starts, begins, counts, expected
):
    from torchtune.modules.moe._parallelism import _expand_grouped_ranges

    result = _expand_grouped_ranges(
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(begins, dtype=torch.long),
        torch.tensor(counts, dtype=torch.long),
    )
    assert result.tolist() == expected


def test_expert_output_accumulation_a_b_matches_values_and_gradients():
    import torchtune.modules.moe._parallelism as ep_mod

    gather_idx = torch.tensor([3, 0, 3, 5, 1], dtype=torch.long)
    routed_base = torch.randn(5, 4)
    expected_base = torch.randn(7, 4)
    outputs = []
    try:
        for use_index_add in (False, True):
            ep_mod._USE_INDEX_ADD_COMBINE = use_index_add
            routed = routed_base.clone().requires_grad_(True)
            partial = expected_base.clone()
            result = ep_mod._accumulate_expert_outputs(
                partial, gather_idx, routed
            )
            loss = result.square().sum()
            loss.backward()
            outputs.append((result.detach(), routed.grad.detach()))
    finally:
        ep_mod._USE_INDEX_ADD_COMBINE = True

    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("direct_cpu_transfer", [False, True])
def test_routing_metadata_materialization_a_b_matches(direct_cpu_transfer):
    import torchtune.modules.moe._parallelism as ep_mod

    metadata = torch.tensor([3, 0, 7, 2, 5, 1], dtype=torch.long)
    materialized = ep_mod._materialize_routing_metadata(
        metadata,
        torch.device("cpu"),
        direct_cpu_transfer=direct_cpu_transfer,
    )
    torch.testing.assert_close(materialized, metadata)
    if direct_cpu_transfer:
        assert materialized.data_ptr() == metadata.data_ptr()
    else:
        assert materialized.data_ptr() != metadata.data_ptr()


@pytest.mark.parametrize("direct_cpu_copy", [False, True])
def test_cpu_collective_output_copy_a_b_matches(direct_cpu_copy):
    import torchtune.modules.moe._parallelism as ep_mod

    source = torch.randn(5, 3)
    destination = torch.empty_like(source)
    result = ep_mod._copy_cpu_collective_output(
        destination, source, direct_cpu_copy=direct_cpu_copy
    )
    torch.testing.assert_close(result, source)
    assert result.data_ptr() == destination.data_ptr()


@pytest.mark.parametrize("make_non_contiguous", [False, True])
def test_raw_all_to_all_single_only_contiguous_copies_when_needed(monkeypatch, make_non_contiguous):
    import torchtune.modules.moe._parallelism as ep_mod

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    send = base[:, ::2] if make_non_contiguous else base[:, :2].contiguous()
    observed = []
    original_flag = ep_mod._CONDITIONAL_ALLTOALL_CONTIGUOUS

    def fake_all_to_all_single(output, input, **kwargs):
        observed.append(input)
        output.copy_(input)

    monkeypatch.setattr(ep_mod.dist, "all_to_all_single", fake_all_to_all_single)
    try:
        ep_mod._CONDITIONAL_ALLTOALL_CONTIGUOUS = True
        result = ep_mod._raw_all_to_all_single(send, [3], [3], object())
    finally:
        ep_mod._CONDITIONAL_ALLTOALL_CONTIGUOUS = original_flag

    torch.testing.assert_close(result, send)
    assert observed[0].is_contiguous()
    if make_non_contiguous:
        assert observed[0].data_ptr() != send.data_ptr()
    else:
        assert observed[0].data_ptr() == send.data_ptr()


def _run_one_rank(rank, world, num_experts, dim, hidden, tokens_per_rank, seed, ret):
    """One EP rank: build identical inputs, run BOTH dispatch paths, compare."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29591"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group("gloo", rank=rank, world_size=world)

    try:
        import torchtune.modules.moe._parallelism as ep_mod
        from torchtune.modules.moe.experts import GroupedExperts
        from torch.distributed.device_mesh import init_device_mesh

        # gloo EP group for the NTPE all_gather bounce used by both paths.
        ep_mod._GLOO_EP_PG = dist.new_group(backend="gloo")
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))
        ep_degree = world
        num_local = num_experts // ep_degree

        # Deterministic per-rank routed_input (expert-sorted) + ntpe.
        # CRITICAL: S (= sum ntpe) must be IDENTICAL across ranks — the AG+RS
        # reference path AllGathers (s_local, dim) into (ep_degree*s_local, dim)
        # and aborts on unequal s_local. In real GRPO S = bs*slen*top_k is equal
        # across EP ranks; only the per-expert DISTRIBUTION varies. We enforce
        # exactly-S via a multinomial draw of S tokens over experts.
        g = torch.Generator().manual_seed(seed + rank)
        S = tokens_per_rank
        probs = torch.rand(num_experts, generator=g) + 0.05
        probs = probs / probs.sum()
        assign = torch.multinomial(probs, S, replacement=True, generator=g)
        ntpe = torch.bincount(assign, minlength=num_experts).to(torch.long)
        assert int(ntpe.sum().item()) == S
        base = torch.randn(S, dim, generator=g, dtype=torch.float32)

        # Shared expert weights: FULL set on every rank, then slice this rank's
        # OWNED experts (interleaved) — exactly as the recipe does.
        gw = torch.randn(num_experts, dim, hidden, generator=torch.Generator().manual_seed(seed))
        uw = torch.randn(num_experts, dim, hidden, generator=torch.Generator().manual_seed(seed + 1))
        dw = torch.randn(num_experts, hidden, dim, generator=torch.Generator().manual_seed(seed + 2))
        owned = list(range(rank, num_experts, ep_degree))

        def build_experts():
            e = GroupedExperts(dim=dim, hidden_dim=hidden, num_experts=num_local)
            with torch.no_grad():
                e.gate_proj.copy_(gw[owned])
                e.up_proj.copy_(uw[owned])
                e.down_proj.copy_(dw[owned])
            return e

        def run_path(use_a2a, use_fused=False):
            os.environ["TORCHTUNE_EP_ALL2ALL"] = "1" if use_a2a else "0"
            # reload flag
            ep_mod._EP_ALL2ALL = use_a2a
            ep_mod._USE_FUSED_ALLTOALL_ROUTING = use_fused
            experts = build_experts()
            epstyle = ep_mod.ExpertParallel()
            ri = base.clone().requires_grad_(True)
            nt = ntpe.clone()
            disp, local_ntpe = epstyle._token_dispatch(
                experts, ri, nt, device_mesh=mesh
            )
            out_experts = experts(disp, local_ntpe)
            combined = epstyle._token_combine(
                experts, out_experts, device_mesh=mesh
            )
            loss = combined.float().pow(2).sum()
            loss.backward()
            return (
                combined.detach().clone(),
                ri.grad.detach().clone(),
                experts.gate_proj.grad.detach().clone(),
                experts.down_proj.grad.detach().clone(),
            )

        try:
            out_ag, gin_ag, gg_ag, gd_ag = run_path(False)
            out_a2, gin_a2, gg_a2, gd_a2 = run_path(True)
            out_fused, gin_fused, gg_fused, gd_fused = run_path(True, True)
        finally:
            ep_mod._USE_FUSED_ALLTOALL_ROUTING = False

        def maxerr(a, b):
            return (a - b).abs().max().item() if a.numel() else 0.0

        res = {
            "rank": rank,
            "out": maxerr(out_ag, out_a2),
            "grad_in": maxerr(gin_ag, gin_a2),
            "grad_gate": maxerr(gg_ag, gg_a2),
            "grad_down": maxerr(gd_ag, gd_a2),
            "fused_out": maxerr(out_a2, out_fused),
            "fused_grad_in": maxerr(gin_a2, gin_fused),
            "fused_grad_gate": maxerr(gg_a2, gg_fused),
            "fused_grad_down": maxerr(gd_a2, gd_fused),
            "S": S,
        }
        ret[rank] = res
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "num_experts,ep_degree",
    [(8, 2), (8, 4), (16, 4), (32, 4)],
)
def test_all2all_matches_aggrs(num_experts, ep_degree):
    dim, hidden, tokens_per_rank, seed = 16, 32, 40, 1234
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(
        _run_one_rank,
        args=(ep_degree, num_experts, dim, hidden, tokens_per_rank, seed, ret),
        nprocs=ep_degree,
        join=True,
    )
    assert len(ret) == ep_degree, f"only {len(ret)}/{ep_degree} ranks reported"
    for rank in range(ep_degree):
        r = ret[rank]
        # bf16-free (fp32 CPU) so tolerance is tight — this is exact math, only
        # float reassociation from different reduction order differs.
        assert r["out"] < 1e-4, f"output mismatch rank{rank}: {r}"
        assert r["grad_in"] < 1e-4, f"grad_in mismatch rank{rank}: {r}"
        assert r["grad_gate"] < 1e-4, f"grad_gate mismatch rank{rank}: {r}"
        assert r["grad_down"] < 1e-4, f"grad_down mismatch rank{rank}: {r}"
        assert r["fused_out"] < 1e-4, f"fused output mismatch rank{rank}: {r}"
        assert r["fused_grad_in"] < 1e-4, f"fused grad_in mismatch rank{rank}: {r}"
        assert r["fused_grad_gate"] < 1e-4, f"fused grad_gate mismatch rank{rank}: {r}"
        assert r["fused_grad_down"] < 1e-4, f"fused grad_down mismatch rank{rank}: {r}"

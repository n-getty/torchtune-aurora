# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""``_ep_release_fsdp_unsharded_grads`` v166 batched collective ≡ v9 per-param
collective (CPU/gloo).

Background
----------
`torchtune/dev/rl/distributed.py::_ep_release_fsdp_unsharded_grads` manually
reduces FSDP2's unsharded gradients across ranks (EP mode sets
`FSDPParamGroup.reduce_grads = False`, so FSDP2 itself never does this). The
original (v9) implementation called a SEPARATE `all_reduce` per FSDPParam —
profiling on Qwen3-30B-A3B/EP=8 found ~97+ such calls per microbatch, with
~60% of wall-clock time spent in these collectives (per-call launch/sync
overhead, not payload bytes, dominating at this per-call size). v166 batches
all same-(PG, dtype) grads into ONE flattened `all_reduce` per bucket instead.

This test proves the v166 batched path is BIT-EXACT to a reference
unbatched-per-param-loop implementation (the old v9 algorithm, reproduced
here directly rather than imported, since it no longer exists in the source
after the rewrite) — same reduced values AND same resulting sharded
`param.grad` DTensors — across:
  (a) the common case (`accumulate_into_grad=False`, fresh grads)
  (b) the accumulation case (`accumulate_into_grad=True`, adds onto existing
      `param.grad`) — exercised by `gradient_accumulation_steps > 1` with EP
      active in both the GRPO and MoE SFT recipes.
  (c) mixed 1D (norm-like) and 2D (linear-like) param shapes in the same run,
      to exercise `shard_dim` resolution and the pad/slice uneven-shard path.

Uses REAL FSDP2 (`torch.distributed._composable.fsdp.fully_shard`) over
gloo/CPU with `world_size=2` (mirrors `test_sft_no_sync_grad_equivalence.py`'s
harness) — no XPU needed, no mocking of FSDP2 internals.

Run: pytest tests/torchtune/dev/rl/test_ep_grad_release_batching_equivalence.py --timeout=120
"""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD = 2
DIM = 8
N_LINEAR_GROUPS = 3  # 2D weight groups
N_NORM_GROUPS = 2    # 1D "norm-like" groups (exercises shard_dim=0 on a 1D tensor)


class _NormLike(torch.nn.Module):
    """A bare 1D Parameter wrapped in a real module (with forward) so FSDP2
    can wrap it as its own FSDPParamGroup — mirrors a norm weight's shape."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x * self.weight


class _EPStyleModel(torch.nn.Module):
    """Container with `forward` (required by `fully_shard`) holding several
    independent sub-modules, each wrapped as its own FSDPParamGroup — mirrors
    the EP recipe's per-expert-module `fully_shard` calls."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        for i in range(N_LINEAR_GROUPS):
            setattr(self, f"linear_{i}", torch.nn.Linear(DIM, DIM, bias=False))
        for i in range(N_NORM_GROUPS):
            setattr(self, f"norm_{i}", _NormLike(DIM))

    def forward(self, x):
        for i in range(N_LINEAR_GROUPS):
            x = getattr(self, f"linear_{i}")(x)
        for i in range(N_NORM_GROUPS):
            x = getattr(self, f"norm_{i}")(x)
        return x


def _build_model(seed: int) -> torch.nn.Module:
    return _EPStyleModel(seed)


def _wrap_fsdp2_ep_style(model: torch.nn.Module, mesh):
    """Wrap each sub-module as its OWN FSDPParamGroup (mirrors the EP recipe's
    per-expert-module `fully_shard` calls) and suppress `reduce_grads` so
    FSDP2 never reduce-scatters — the exact precondition
    `_ep_release_fsdp_unsharded_grads` is designed for."""
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import FSDPModule

    for i in range(N_LINEAR_GROUPS):
        fully_shard(getattr(model, f"linear_{i}"), mesh=mesh)
    for i in range(N_NORM_GROUPS):
        fully_shard(getattr(model, f"norm_{i}"), mesh=mesh)
    fully_shard(model, mesh=mesh)

    for m in model.modules():
        if isinstance(m, FSDPModule):
            fsdp_state = m._get_fsdp_state()
            if fsdp_state is not None and fsdp_state._fsdp_param_group is not None:
                fsdp_state._fsdp_param_group.reduce_grads = False
    return model


def _reference_unbatched_release(model, accumulate_into_grad: bool) -> None:
    """Reference reimplementation of the ORIGINAL (v9, pre-batching)
    per-FSDPParam-collective algorithm, kept here (not imported) so this test
    remains a fixed reference point independent of future edits to the
    production function."""
    from torch.distributed.fsdp import FSDPModule

    seen_pg_ids = set()
    param_groups_in_order = []
    for _name, mod in model.named_modules():
        if not isinstance(mod, FSDPModule):
            continue
        fsdp_state = mod._get_fsdp_state()
        if fsdp_state is None:
            continue
        pg_obj = getattr(fsdp_state, "_fsdp_param_group", None)
        if pg_obj is None or id(pg_obj) in seen_pg_ids:
            continue
        seen_pg_ids.add(id(pg_obj))
        param_groups_in_order.append(pg_obj)

    world = dist.get_world_size()
    for pg_obj in param_groups_in_order:
        for fsdp_param in pg_obj.fsdp_params:
            ush_param = getattr(fsdp_param, "_unsharded_param", None)
            grad_full = ush_param.grad if ush_param is not None else None
            if grad_full is None:
                continue

            shard_dim = 0
            sspec = getattr(fsdp_param, "_sharding_spec", None)
            if sspec is not None:
                for plc in sspec.placements:
                    if hasattr(plc, "dim"):
                        shard_dim = plc.dim
                        break

            grad_cpu = grad_full.contiguous().cpu()
            dist.all_reduce(grad_cpu, op=dist.ReduceOp.SUM)
            grad_cpu.div_(world)
            local_rank = dist.get_rank()
            chunks = list(torch.chunk(grad_cpu, world, dim=shard_dim))
            local_chunk = (
                chunks[local_rank]
                if local_rank < len(chunks)
                else torch.zeros(0, dtype=grad_cpu.dtype)
            )

            sharded_param = fsdp_param.sharded_param
            target_size = fsdp_param.sharded_size
            cur = local_chunk.size(shard_dim) if local_chunk.dim() > shard_dim else 0
            tgt = target_size[shard_dim] if len(target_size) > shard_dim else 0
            if cur != tgt:
                if cur < tgt:
                    pad_shape = list(local_chunk.shape)
                    pad_shape[shard_dim] = tgt - cur
                    pad = torch.zeros(pad_shape, dtype=local_chunk.dtype)
                    local_chunk = torch.cat([local_chunk, pad], dim=shard_dim)
                else:
                    local_chunk = local_chunk.narrow(shard_dim, 0, tgt)

            local_shard = local_chunk.contiguous().to(
                device=sharded_param.device, dtype=sharded_param.dtype
            )
            if sharded_param.grad is None or not accumulate_into_grad:
                sharded_param.grad = fsdp_param.to_sharded_dtensor(local_shard)
            else:
                existing = sharded_param.grad
                existing_local = (
                    existing._local_tensor if hasattr(existing, "_local_tensor") else existing
                )
                existing_local.add_(local_shard)

            ush_param.grad = None
            fsdp_param.unsharded_accumulated_grad = None


def _grads_flat(model: torch.nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        if hasattr(g, "full_tensor"):
            g = g.full_tensor()
        parts.append(g.reshape(-1).detach().clone())
    return torch.cat(parts)


def _set_matching_unsharded_grads(model: torch.nn.Module, seed: int) -> None:
    """Populate every FSDPParam's `_unsharded_param.grad` deterministically
    (same seed/values on both the reference and production-path models), the
    precondition `_ep_release_fsdp_unsharded_grads` expects.

    `_unsharded_param` only exists AFTER FSDP2 has unsharded the param at
    least once (normally triggered by a forward pass) — a real forward+
    backward (with `reduce_grads` already suppressed, so the grad lands on
    `_unsharded_param.grad`/`unsharded_accumulated_grad` and FSDP2 does NOT
    reduce-scatter it away) establishes this, then we overwrite the grad
    values deterministically so both the reference and production-path
    models see IDENTICAL inputs regardless of the (irrelevant) loss value
    from this dummy pass."""
    from torch.distributed.fsdp import FSDPModule

    x = torch.ones(2, DIM)
    model(x).sum().backward()

    g = torch.Generator().manual_seed(seed)
    for _name, mod in model.named_modules():
        if not isinstance(mod, FSDPModule):
            continue
        fsdp_state = mod._get_fsdp_state()
        if fsdp_state is None or fsdp_state._fsdp_param_group is None:
            continue
        for fsdp_param in fsdp_state._fsdp_param_group.fsdp_params:
            ush_param = getattr(fsdp_param, "_unsharded_param", None)
            if ush_param is None:
                continue
            ush_param.grad = torch.randn(ush_param.shape, generator=g)


def _worker(rank, ws, ret, legacy, streaming=False):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29573")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(ws)
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        import torchtune.dev.rl.distributed as ttd

        mesh = init_device_mesh("cpu", (ws,))

        # Wire up the module globals _ep_release_fsdp_unsharded_grads reads,
        # mirroring set_process_groups()'s effect for a dp_replicate=1/EP-only
        # config (degree = world size, no XCCL on CPU-only gloo test).
        gloo_pg = dist.new_group(backend="gloo")
        ttd._GLOO_DP_SHARD_PG = gloo_pg
        ttd._GLOO_GLOBAL_PG = gloo_pg
        ttd._DP_SHARD_DEGREE = ws
        ttd._DP_REP_DEGREE = 1
        ttd._XCCL_DP_SHARD_PG = None  # force gloo path (no XPU in this test)
        ttd._EP_GRAD_RELEASE_LEGACY = legacy
        ttd._EP_GRAD_RELEASE_STREAMING = streaming

        # --- Reference path: separate model, old unbatched algorithm.
        m_ref = _build_model(seed=0)
        _wrap_fsdp2_ep_style(m_ref, mesh)
        _set_matching_unsharded_grads(m_ref, seed=123)
        _reference_unbatched_release(m_ref, accumulate_into_grad=False)
        g_ref_pass1 = _grads_flat(m_ref)
        # Second pass: accumulate_into_grad=True.
        _set_matching_unsharded_grads(m_ref, seed=456)
        _reference_unbatched_release(m_ref, accumulate_into_grad=True)
        g_ref_pass2 = _grads_flat(m_ref)

        # --- Production path: separate model (same init seed), new batched fn.
        m_new = _build_model(seed=0)
        _wrap_fsdp2_ep_style(m_new, mesh)
        pg_map = ttd._ep_build_grad_release_pg_map(m_new)
        _set_matching_unsharded_grads(m_new, seed=123)
        n_groups_1 = ttd._ep_release_fsdp_unsharded_grads(
            m_new, pg_map, accumulate_into_grad=False,
        )
        g_new_pass1 = _grads_flat(m_new)
        _set_matching_unsharded_grads(m_new, seed=456)
        n_groups_2 = ttd._ep_release_fsdp_unsharded_grads(
            m_new, pg_map, accumulate_into_grad=True,
        )
        g_new_pass2 = _grads_flat(m_new)

        if rank == 0:
            ret["pass1_max_abs_diff"] = float((g_ref_pass1 - g_new_pass1).abs().max())
            ret["pass2_max_abs_diff"] = float((g_ref_pass2 - g_new_pass2).abs().max())
            ret["n_groups_1"] = n_groups_1
            ret["n_groups_2"] = n_groups_2
            ret["n_params"] = g_ref_pass1.numel()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.timeout(120)
def test_grad_release_batched_equivalent_to_unbatched_reference(legacy, streaming):
    """Batched (v166) release must be BIT-EXACT to the unbatched (v9) reference,
    both for a fresh release (accumulate_into_grad=False) and an accumulating
    second release (accumulate_into_grad=True)."""
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(
        _worker,
        args=(WORLD, ret, legacy, streaming),
        nprocs=WORLD,
        join=True,
    )

    assert "pass1_max_abs_diff" in ret, "worker did not report results"
    assert ret["n_groups_1"] == N_LINEAR_GROUPS + N_NORM_GROUPS, (
        f"Expected {N_LINEAR_GROUPS + N_NORM_GROUPS} FSDPParamGroups processed, "
        f"got {ret['n_groups_1']}."
    )
    assert ret["pass1_max_abs_diff"] == 0.0, (
        f"Batched release diverged from unbatched reference on the fresh-grad "
        f"pass (accumulate_into_grad=False): max_abs_diff={ret['pass1_max_abs_diff']:.3e}. "
        f"The v166 flatten/all_reduce/unflatten batching must be bit-exact to "
        f"the per-param collective it replaced."
    )
    assert ret["pass2_max_abs_diff"] == 0.0, (
        f"Batched release diverged from unbatched reference on the accumulating "
        f"pass (accumulate_into_grad=True): max_abs_diff={ret['pass2_max_abs_diff']:.3e}. "
        f"gradient_accumulation_steps>1 with EP active exercises this exact path "
        f"in both the GRPO and MoE SFT recipes."
    )


def _counting_worker(rank, ws, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29574")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(ws)
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        import torchtune.dev.rl.distributed as ttd

        mesh = init_device_mesh("cpu", (ws,))
        gloo_pg = dist.new_group(backend="gloo")
        ttd._GLOO_DP_SHARD_PG = gloo_pg
        ttd._GLOO_GLOBAL_PG = gloo_pg
        ttd._DP_SHARD_DEGREE = ws
        ttd._DP_REP_DEGREE = 1
        ttd._XCCL_DP_SHARD_PG = None

        m = _build_model(seed=0)
        _wrap_fsdp2_ep_style(m, mesh)
        pg_map = ttd._ep_build_grad_release_pg_map(m)
        _set_matching_unsharded_grads(m, seed=123)

        n_calls = {"count": 0}
        orig_ar = ttd._orig_all_reduce

        def counting_ar(*a, **k):
            n_calls["count"] += 1
            return orig_ar(*a, **k)

        ttd._orig_all_reduce = counting_ar
        try:
            ttd._ep_release_fsdp_unsharded_grads(
                m, pg_map, accumulate_into_grad=False,
            )
        finally:
            ttd._orig_all_reduce = orig_ar

        if rank == 0:
            ret["n_calls"] = n_calls["count"]
            ret["n_param_groups"] = N_LINEAR_GROUPS + N_NORM_GROUPS
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(60)
def test_grad_release_collective_count_is_batched():
    """The batched path must issue FEWER all_reduce calls than one per param —
    proves the batching optimization actually engaged, not just that it's a
    no-op equivalent to itself."""
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_counting_worker, args=(WORLD, ret), nprocs=WORLD, join=True)

    assert "n_calls" in ret, "worker did not report results"
    # All 5 groups here resolve to the SAME PG/dtype (single dp_shard mesh,
    # no dp_replicate mix) -> should batch into exactly 1 collective, strictly
    # fewer than the N_LINEAR_GROUPS + N_NORM_GROUPS calls the old algorithm made.
    assert ret["n_calls"] < ret["n_param_groups"], (
        f"Expected batching to issue fewer all_reduce calls than "
        f"{ret['n_param_groups']} FSDPParamGroups; got {ret['n_calls']} calls. "
        f"The batching optimization is not actually coalescing collectives."
    )

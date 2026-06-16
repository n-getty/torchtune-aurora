"""Unit test for the FSDP1 HSDP inter-node all_reduce gloo reroute.

Pins the gating contract of `_xpu_all_reduce_inter_node_gloo` (distributed.py):
the dense FSDP1 HYBRID_SHARD inter-node grad reduction
(`dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)`) must be
CPU-bounced over the gloo replicate PG to avoid the Aurora XCCL/RDMA CXI MR leak
that crashed the 84-rank run at step 11 (job 8544731) — but ONLY:

  - when the flag is set (dense FSDP1 HSDP path; NOT EP/FSDP2, NOT flat 2N), AND
  - for the call on the replicate-dim group (world size == _DP_REP_DEGREE).

Every other all_reduce (CPU tensors, non-replicate groups, flag off) must fall
through to the original XCCL all_reduce untouched. A regression here either
re-introduces the leak (reroute not firing) or breaks EP/intra-node collectives
(reroute firing too broadly).

CPU-only, no distributed init: we monkeypatch the module's `_orig_all_reduce`
and PG globals with fakes and assert which path each call takes.
"""

import types

import pytest
import torch

import torchtune.dev.rl.distributed as D


class _FakeWork:
    pass


@pytest.fixture
def patched(monkeypatch):
    """Install fakes for _orig_all_reduce and the gloo replicate PG, and record
    which group each original-all_reduce call targeted."""
    calls = {"orig": []}

    _REP_PG = object()    # sentinel for the gloo replicate PG
    _XCCL_REP = object()  # sentinel for the XCCL replicate group (the FSDP arg)
    _OTHER = object()     # some other (e.g. shard / world) group

    def fake_orig_all_reduce(tensor, op=None, group=None, async_op=False):
        calls["orig"].append({"group": group, "device": tensor.device.type})
        return _FakeWork() if async_op else None

    def fake_get_world_size(group):
        # Only the XCCL replicate group has the replicate degree.
        if group is _XCCL_REP:
            return D._DP_REP_DEGREE
        return 999

    monkeypatch.setattr(D, "_orig_all_reduce", fake_orig_all_reduce)
    monkeypatch.setattr(D, "_GLOO_DP_REP_PG", _REP_PG)
    monkeypatch.setattr(D, "_DP_REP_DEGREE", 7)
    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)

    return types.SimpleNamespace(
        calls=calls, REP_PG=_REP_PG, XCCL_REP=_XCCL_REP, OTHER=_OTHER
    )


class _XpuLikeTensor:
    """Minimal stand-in for an XPU grad tensor: reports device.type='xpu' but
    backs onto a real CPU tensor so the D2H/H2D bounce in the patch is exercised
    without real XPU hardware. The patch does:
        tensor_cpu = tensor.contiguous().to("cpu")   # D2H  → real cpu tensor
        _orig_all_reduce(tensor_cpu, ..., group=gloo)
        tensor.copy_(tensor_cpu.to(tensor.device))   # H2D  → back onto self
    We make `.device` a real torch.device('xpu') (constructs fine on a CPU box),
    `.to('cpu')` return the backing cpu tensor, and intercept the H2D `.to(xpu)`
    on the RESULT by returning a plain cpu tensor `copy_` can consume."""
    def __init__(self, x):
        self._x = x
        self.device = torch.device("xpu")
    def contiguous(self):
        return self
    def to(self, *a, **k):
        # D2H: return the backing cpu tensor wrapped so its own .to(xpu) is a no-op.
        return _CpuBounce(self._x)
    def copy_(self, other):
        self._x.copy_(other._x if isinstance(other, _CpuBounce) else other)


class _CpuBounce:
    """The 'cpu' tensor inside the patch: real enough for _orig_all_reduce (which
    is faked in the test) and whose .to(device) is a no-op returning itself."""
    def __init__(self, x):
        self._x = x
        self.device = torch.device("cpu")
    def contiguous(self):
        return self
    def to(self, *a, **k):
        return self


def test_reroutes_inter_node_replicate_call_to_gloo(patched, monkeypatch):
    monkeypatch.setattr(D, "_FSDP1_HSDP_INTER_NODE_GLOO", True)
    shim = _XpuLikeTensor(torch.ones(4))
    D._xpu_all_reduce_inter_node_gloo(shim, group=patched.XCCL_REP)
    # The original all_reduce must have been called ONCE, on the GLOO replicate PG,
    # with a CPU tensor (the D2H bounce) — not on the XCCL replicate group.
    assert len(patched.calls["orig"]) == 1
    assert patched.calls["orig"][0]["group"] is patched.REP_PG
    assert patched.calls["orig"][0]["device"] == "cpu"


def test_passthrough_when_flag_off(patched, monkeypatch):
    monkeypatch.setattr(D, "_FSDP1_HSDP_INTER_NODE_GLOO", False)
    t = torch.ones(4)
    D._xpu_all_reduce_inter_node_gloo(t, group=patched.XCCL_REP)
    # Flag off → must pass straight through to original on the SAME group given.
    assert len(patched.calls["orig"]) == 1
    assert patched.calls["orig"][0]["group"] is patched.XCCL_REP


def test_passthrough_for_non_replicate_group(patched, monkeypatch):
    monkeypatch.setattr(D, "_FSDP1_HSDP_INTER_NODE_GLOO", True)
    shim = _XpuLikeTensor(torch.ones(4))
    # _OTHER has world size 999 != _DP_REP_DEGREE → must NOT reroute.
    D._xpu_all_reduce_inter_node_gloo(shim, group=patched.OTHER)
    assert len(patched.calls["orig"]) == 1
    assert patched.calls["orig"][0]["group"] is patched.OTHER


def test_passthrough_for_cpu_tensor(patched, monkeypatch):
    monkeypatch.setattr(D, "_FSDP1_HSDP_INTER_NODE_GLOO", True)
    t = torch.ones(4)  # real cpu tensor → device.type == 'cpu' → no reroute
    D._xpu_all_reduce_inter_node_gloo(t, group=patched.XCCL_REP)
    assert len(patched.calls["orig"]) == 1
    assert patched.calls["orig"][0]["group"] is patched.XCCL_REP


def test_enable_sets_flag_and_installs_patch(monkeypatch):
    # enable_fsdp1_hsdp_inter_node_gloo flips the flag and swaps dist.all_reduce.
    monkeypatch.setattr(D, "_FSDP1_HSDP_INTER_NODE_GLOO", False)
    import torch.distributed as _d
    orig = _d.all_reduce
    try:
        D.enable_fsdp1_hsdp_inter_node_gloo()
        assert D._FSDP1_HSDP_INTER_NODE_GLOO is True
        assert _d.all_reduce is D._xpu_all_reduce_inter_node_gloo
    finally:
        _d.all_reduce = orig
        D._FSDP1_HSDP_INTER_NODE_GLOO = False

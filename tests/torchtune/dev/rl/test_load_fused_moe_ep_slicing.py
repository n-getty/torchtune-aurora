"""Pin-down test for `_load_fused_moe_experts` EP-aware slicing.

Failure mode (hold 8465103, attempt 4): receiver assumed TP-only sharding
(`ep_start = tp_rank * e_local`) but vLLM was running 8-way EP × 1-way TP
internally (folded from --tensor-parallel-size=4 + DP=2). The global
expert tensor `[128, 2*intermediate, hidden]` mismatched the local param
`[16, ...]` → `RuntimeError: size 128 vs 16` on every vLLM worker.

The fix reads `expert_map`, `tp_rank`, `tp_size` directly off the
FusedMoE module instead of guessing from `dist.get_rank()`. This test
simulates the dispatch in pure Python with a fake FusedMoE-like object,
no vLLM and no torch.distributed.
"""
from __future__ import annotations

import torch


class _FakeMoE:
    """Mimic the few attributes the receiver reads off vLLM's FusedMoE."""

    def __init__(self, expert_map, tp_rank, tp_size):
        self.expert_map = expert_map
        self.tp_rank = tp_rank
        self.tp_size = tp_size


class _FakeModel:
    def __init__(self, layers):
        self._layers = layers
        self._named = {}

    def add_param(self, name, tensor):
        self._named[name] = tensor

    def named_parameters(self):
        return list(self._named.items())

    def get_submodule(self, name):
        return self._layers[name]


class _FakeRunner:
    def __init__(self, model):
        self.model = model


class _FakeWorker:
    """Wraps the real method onto a fake worker for unit testing."""

    def __init__(self, model):
        self.model_runner = _FakeRunner(model)

    # Bind the real implementation so the test exercises production code.
    from torchtune.dev.vllm_weight_sync_worker import (  # noqa: E402
        WeightSyncFromFileExtension,
    )
    _load_fused_moe_experts = WeightSyncFromFileExtension._load_fused_moe_experts


def _build(global_e: int, local_e: int, inter: int, hidden: int,
           tp_size: int, tp_rank: int, ep_rank: int, transposed: bool):
    """Build a (worker, fused_data) pair for a single MoE layer."""
    inter_per_tp = inter // tp_size
    # GLOBAL fused tensors as the trainer would emit them.
    w13_global = torch.arange(
        global_e * 2 * inter * hidden, dtype=torch.float32
    ).reshape(global_e, 2 * inter, hidden)
    w2_global = torch.arange(
        global_e * hidden * inter, dtype=torch.float32
    ).reshape(global_e, hidden, inter)

    # Build expert_map (vLLM's contiguous EP partition).
    em = torch.full((global_e,), -1, dtype=torch.int32)
    start = ep_rank * local_e
    em[start:start + local_e] = torch.arange(local_e, dtype=torch.int32)

    fake_moe = _FakeMoE(em, tp_rank=tp_rank, tp_size=tp_size)

    if transposed:
        # IPEX-style: param has hidden along dim 1 instead of intermediate.
        w13_param = torch.zeros(local_e, hidden, 2 * inter_per_tp)
        w2_param = torch.zeros(local_e, inter_per_tp, hidden)
    else:
        w13_param = torch.zeros(local_e, 2 * inter_per_tp, hidden)
        w2_param = torch.zeros(local_e, hidden, inter_per_tp)

    model = _FakeModel({"model.layers.0.mlp.experts": fake_moe})
    model.add_param("model.layers.0.mlp.experts.w13_weight", w13_param)
    model.add_param("model.layers.0.mlp.experts.w2_weight", w2_param)

    worker = _FakeWorker(model)
    fused_data = {0: {"w13": w13_global, "w2": w2_global}}
    return worker, fused_data, w13_global, w2_global, w13_param, w2_param


def test_ep_only_slices_correctly():
    """8-way EP, no TP: each worker gets its EP slab of the expert dim."""
    for ep_rank in [0, 3, 7]:
        worker, fused, w13g, w2g, w13p, w2p = _build(
            global_e=128, local_e=16, inter=768, hidden=2048,
            tp_size=1, tp_rank=0, ep_rank=ep_rank, transposed=False,
        )
        worker._load_fused_moe_experts(fused)

        start = ep_rank * 16
        assert torch.equal(w13p, w13g[start:start + 16])
        assert torch.equal(w2p, w2g[start:start + 16])


def test_ep_plus_tp_slices_both_dims():
    """4-way EP × 2-way TP: shard expert dim AND intermediate dim."""
    global_e, local_e = 32, 8
    inter, hidden = 64, 16
    for ep_rank in [0, 1, 3]:
        for tp_rank in [0, 1]:
            worker, fused, w13g, w2g, w13p, w2p = _build(
                global_e=global_e, local_e=local_e, inter=inter, hidden=hidden,
                tp_size=2, tp_rank=tp_rank, ep_rank=ep_rank, transposed=False,
            )
            worker._load_fused_moe_experts(fused)

            ep_start = ep_rank * local_e
            inter_per_tp = inter // 2
            tp_lo, tp_hi = tp_rank * inter_per_tp, (tp_rank + 1) * inter_per_tp

            expected_gate = w13g[ep_start:ep_start + local_e, tp_lo:tp_hi, :]
            expected_up = w13g[
                ep_start:ep_start + local_e,
                inter + tp_lo:inter + tp_hi,
                :,
            ]
            expected_w13 = torch.cat([expected_gate, expected_up], dim=1)
            expected_w2 = w2g[ep_start:ep_start + local_e, :, tp_lo:tp_hi]

            assert torch.equal(w13p, expected_w13), (
                f"w13 mismatch at ep_rank={ep_rank} tp_rank={tp_rank}"
            )
            assert torch.equal(w2p, expected_w2), (
                f"w2 mismatch at ep_rank={ep_rank} tp_rank={tp_rank}"
            )


def test_ep_with_ipex_transpose():
    """Transposed-param layout (IPEX): receiver must still produce
    bit-exact data after transpose."""
    worker, fused, w13g, w2g, w13p, w2p = _build(
        global_e=16, local_e=4, inter=32, hidden=8,
        tp_size=1, tp_rank=0, ep_rank=2, transposed=True,
    )
    worker._load_fused_moe_experts(fused)

    expected_w13 = w13g[8:12].transpose(1, 2).contiguous()
    expected_w2 = w2g[8:12].transpose(1, 2).contiguous()
    assert torch.equal(w13p, expected_w13)
    assert torch.equal(w2p, expected_w2)


def test_missing_expert_map_raises_not_silently_misshards():
    """If FusedMoE has no expert_map, refuse rather than silently
    feeding the wrong slab."""
    import pytest

    global_e, local_e = 16, 4
    inter, hidden = 32, 8
    w13_global = torch.arange(global_e * 2 * inter * hidden, dtype=torch.float32).reshape(
        global_e, 2 * inter, hidden
    )
    w2_global = torch.arange(global_e * hidden * inter, dtype=torch.float32).reshape(
        global_e, hidden, inter
    )
    fake_moe = _FakeMoE(expert_map=None, tp_rank=0, tp_size=1)
    model = _FakeModel({"model.layers.0.mlp.experts": fake_moe})
    model.add_param(
        "model.layers.0.mlp.experts.w13_weight",
        torch.zeros(local_e, 2 * inter, hidden),
    )
    model.add_param(
        "model.layers.0.mlp.experts.w2_weight",
        torch.zeros(local_e, hidden, inter),
    )
    worker = _FakeWorker(model)

    with pytest.raises(RuntimeError, match="no expert_map"):
        worker._load_fused_moe_experts({0: {"w13": w13_global, "w2": w2_global}})

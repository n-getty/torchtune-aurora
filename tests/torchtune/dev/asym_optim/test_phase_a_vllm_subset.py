# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU-only smoke for the Phase A vllm-rank-subset plumbing in
# torchtune/dev/rl/vllm_backend.py. Verifies:
#   - len(vllm_ranks) % tp_size assertion fires/passes correctly
#   - _init_vllm_early early-returns on spare ranks
#   - dp/tp grid is computed from vllm_ranks (not the full world)

import os
import types
from unittest import mock

import pytest


@pytest.fixture
def fake_self():
    obj = types.SimpleNamespace()
    obj._init_vllm_tp_called_with = None
    obj._init_vllm_tp1_called_with = None
    obj._device = types.SimpleNamespace(type="xpu")

    def _record_tp(cfg, rank, ws, lr, tp, *args, **kwargs):
        obj._init_vllm_tp_called_with = (rank, ws, tp)

    def _record_tp1(cfg, rank, ws, lr, *args, **kwargs):
        obj._init_vllm_tp1_called_with = (rank, ws)

    obj._init_vllm_tp = _record_tp
    obj._init_vllm_tp1 = _record_tp1
    return obj


def _patch_env(rank, world_size):
    return mock.patch.dict(
        os.environ,
        {"RANK": str(rank), "WORLD_SIZE": str(world_size), "LOCAL_RANK": "0"},
    )


def test_init_vllm_early_skips_spare_ranks(fake_self):
    """Rank 8 (spare) on a 12-rank world with vllm_ranks=[0..7] must NOT
    invoke _init_vllm_tp / _init_vllm_tp1 at all."""
    from torchtune.dev.rl import vllm_backend

    fake_self._vllm_ranks = list(range(8))
    cfg = {"vllm_mode": "colocate", "vllm_tensor_parallel_size": 8,
           "base_model_path": "/tmp/x",
           "batch_size": 1, "grpo_samples": 1}

    class _Cfg(dict):
        def get(self, k, d=None):
            return super().get(k, d)

    with _patch_env(rank=8, world_size=12):
        vllm_backend._init_vllm_early(fake_self, _Cfg(cfg))

    assert fake_self._init_vllm_tp_called_with is None
    assert fake_self._init_vllm_tp1_called_with is None


def test_init_vllm_tp_assertion_accepts_subset_divisor():
    """len(vllm_ranks)=8 with tp_size=8 must NOT raise (8 % 8 == 0)."""
    from torchtune.dev.rl import vllm_backend

    fake = types.SimpleNamespace()
    fake._vllm_ranks = list(range(8))
    fake._device = types.SimpleNamespace(type="xpu")

    class _LLM:
        def __init__(self, **kw):
            self.kw = kw
            self.llm_engine = types.SimpleNamespace()

    # We only need the function to get past the divisor assertion; mock the
    # heavy init steps (file barriers, dist init, LLM construction) so the
    # CPU smoke completes without touching XPU/dist.
    with _patch_env(rank=0, world_size=12), \
         mock.patch("torch.distributed.init_process_group"), \
         mock.patch("torch.distributed.is_initialized", return_value=True), \
         mock.patch("torch.distributed.destroy_process_group"), \
         mock.patch("torch.distributed.get_world_size", return_value=8), \
         mock.patch("os.makedirs"), \
         mock.patch("builtins.open", mock.mock_open()), \
         mock.patch("os.path.exists", return_value=True):
        try:
            vllm_backend._init_vllm_tp(
                fake, cfg={}, rank=0, world_size=12, local_rank=0,
                tp_size=8, model_path="/tmp/m", gpu_mem=0.5,
                max_model_len=128, max_num_seqs=1,
                vllm_mode="colocate", LLM=_LLM,
            )
        except AssertionError as e:
            pytest.fail(f"divisor assertion incorrectly fired: {e}")
        except Exception:
            # Anything past the divisor assertion (e.g. vllm_ps import) is
            # acceptable for this test.
            pass

    # dp/tp grid must reflect the SUBSET, not the world.
    assert getattr(fake, "_vllm_dp_size", None) == 1
    assert getattr(fake, "_vllm_dp_rank", None) == 0
    assert getattr(fake, "_vllm_tp_rank", None) == 0


def test_init_vllm_tp_assertion_rejects_mismatch():
    """len(vllm_ranks)=7 with tp_size=8 must raise (7 % 8 != 0)."""
    from torchtune.dev.rl import vllm_backend

    fake = types.SimpleNamespace()
    fake._vllm_ranks = list(range(7))
    fake._device = types.SimpleNamespace(type="xpu")

    class _LLM:
        pass

    with _patch_env(rank=0, world_size=12):
        with pytest.raises(AssertionError, match="must be divisible by"):
            vllm_backend._init_vllm_tp(
                fake, cfg={}, rank=0, world_size=12, local_rank=0,
                tp_size=8, model_path="/tmp/m", gpu_mem=0.5,
                max_model_len=128, max_num_seqs=1,
                vllm_mode="colocate", LLM=_LLM,
            )


def test_init_vllm_tp_rejects_spare_rank_invocation():
    """If a spare rank somehow reaches _init_vllm_tp, the safety assertion fires."""
    from torchtune.dev.rl import vllm_backend

    fake = types.SimpleNamespace()
    fake._vllm_ranks = [0, 1, 2, 3, 4, 5, 6, 7]
    fake._device = types.SimpleNamespace(type="xpu")

    with _patch_env(rank=10, world_size=12):
        with pytest.raises(AssertionError, match="spare rank 10"):
            vllm_backend._init_vllm_tp(
                fake, cfg={}, rank=10, world_size=12, local_rank=2,
                tp_size=8, model_path="/tmp/m", gpu_mem=0.5,
                max_model_len=128, max_num_seqs=1,
                vllm_mode="colocate", LLM=object,
            )

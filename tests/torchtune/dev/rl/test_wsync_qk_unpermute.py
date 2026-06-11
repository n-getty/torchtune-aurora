# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test for the Llama-family wsync Q/K un-permute.

Bug it pins down (root-caused 2026-06-11): the bake-off launcher uses
LLAMA3 model_type for AuroraGPT-2B. The checkpointer applies
``torchtune.models.convert_weights.hf_to_tune``, which permutes Q/K so
torchtune's attention forward can consume them in its native layout.
Qwen2/3/Gemma checkpointers do NOT permute — only Llama-family does.

At weight sync time, vLLM expects HF-format unpermuted Q/K. Before the
fix, ``_sync_colocated_weights`` sent ``param.full_tensor()`` straight to
vLLM's ``load_weights``, so vLLM's Q/K projections were scrambled after
the first sync and the model emitted pretraining-corpus continuations
instead of conditioning on the prompt. This was the silent cause of the
AuroraGPT-2B GRPO bake-off accuracy gap (RESULTS.md 2026-06-11): rewards
flat ~0.20 over 50 steps vs ezpz climbing 0.08→0.65 on identical prompts.
"""
import unittest

import torch

from torchtune.dev.rl.weight_sync import _qk_unpermute_for_vllm
from torchtune.models.convert_weights import hf_to_tune


class TestQKUnpermuteRoundTrip(unittest.TestCase):
    """``_qk_unpermute_for_vllm`` must exactly invert ``hf_to_tune``'s permute."""

    def _round_trip(self, n_heads: int, n_kv: int, head_dim: int, dim: int):
        hf_q = torch.randn(n_heads * head_dim, dim)
        hf_k = torch.randn(n_kv * head_dim, dim)
        sd = {
            "model.layers.0.self_attn.q_proj.weight": hf_q.clone(),
            "model.layers.0.self_attn.k_proj.weight": hf_k.clone(),
        }
        tune = hf_to_tune(sd, num_heads=n_heads, num_kv_heads=n_kv, dim=dim, head_dim=head_dim)
        tune_q = tune["layers.0.attn.q_proj.weight"]
        tune_k = tune["layers.0.attn.k_proj.weight"]
        # Sanity: the permute is non-trivial (would catch a future hf_to_tune no-op refactor).
        self.assertFalse(
            torch.allclose(hf_q, tune_q),
            "hf_to_tune did not change Q weights — this test would falsely pass.",
        )
        unp_q = _qk_unpermute_for_vllm(tune_q, n_heads, head_dim)
        unp_k = _qk_unpermute_for_vllm(tune_k, n_kv, head_dim)
        torch.testing.assert_close(hf_q, unp_q, atol=0.0, rtol=0.0)
        torch.testing.assert_close(hf_k, unp_k, atol=0.0, rtol=0.0)

    def test_agpt_2b_dims(self):
        # AuroraGPT-2B: num_heads=16, num_kv_heads=4, hidden=2048, head_dim=128
        self._round_trip(n_heads=16, n_kv=4, head_dim=128, dim=2048)

    def test_mha_no_gqa(self):
        self._round_trip(n_heads=32, n_kv=32, head_dim=128, dim=4096)

    def test_gqa_8kv(self):
        self._round_trip(n_heads=32, n_kv=8, head_dim=128, dim=4096)


class TestPermutingModelTypesDetection(unittest.TestCase):
    """The Llama-family detection should match the checkpointer's behavior."""

    def test_llama_family_detected(self):
        # Reach into the module to exercise _needs_qk_unpermute with a stub.
        from torchtune.dev.rl.weight_sync import _needs_qk_unpermute
        from torchtune.training.checkpointing._utils import ModelType

        class _StubCkpt:
            def __init__(self, mt):
                self._model_type = mt

        class _StubSelf:
            def __init__(self, ckpt):
                self._checkpointer = ckpt

        for mt_name in ("LLAMA2", "LLAMA3", "LLAMA3_2", "LLAMA3_VISION"):
            mt = getattr(ModelType, mt_name, None)
            if mt is None:
                continue
            ckpt = _StubCkpt(mt)
            self.assertTrue(
                _needs_qk_unpermute(_StubSelf(ckpt)),
                f"_needs_qk_unpermute must be True for {mt_name}",
            )

    def test_non_llama_not_detected(self):
        from torchtune.dev.rl.weight_sync import _needs_qk_unpermute
        from torchtune.training.checkpointing._utils import ModelType

        class _StubCkpt:
            def __init__(self, mt):
                self._model_type = mt

        class _StubSelf:
            def __init__(self, ckpt):
                self._checkpointer = ckpt

        for mt_name in ("QWEN2", "QWEN3", "QWEN3_MOE", "GEMMA2", "GEMMA4"):
            mt = getattr(ModelType, mt_name, None)
            if mt is None:
                continue
            self.assertFalse(
                _needs_qk_unpermute(_StubSelf(_StubCkpt(mt))),
                f"_needs_qk_unpermute must be False for {mt_name}",
            )

    def test_no_checkpointer_returns_false(self):
        from torchtune.dev.rl.weight_sync import _needs_qk_unpermute

        class _StubSelf:
            _checkpointer = None

        self.assertFalse(_needs_qk_unpermute(_StubSelf()))


if __name__ == "__main__":
    unittest.main()

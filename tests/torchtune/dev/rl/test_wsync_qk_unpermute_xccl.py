# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test: the XCCL wsync path must apply the Llama-family
Q/K un-permute (same fix as the colocate path, landed earlier in
test_wsync_qk_unpermute.py).

Bug pinned (found 2026-06-12 during AGPT-2B GSM8K 2N server-mode run):
``_sync_weights_to_vllm_xccl`` cast ``param.full_tensor()`` straight to bf16
and broadcast to vLLM without inverting torchtune's Q/K permute. Symptom on
AGPT-2B GSM8K: vLLM emitted pretrain-corpus continuations, BATCH_REWARD
flat ~0.1, kl_loss(policy||ref) → 1e12 within 10 steps (ref Q/K still
correct on training side, but vLLM forward used scrambled Q/K).

This test does NOT exercise the live FSDP / XCCL machinery (CPU-only,
no GPUs, no torch.distributed init). Instead it pins that the source has
exactly one call to ``_maybe_unpermute_qk`` inside the per-param loop of
``_sync_weights_to_vllm_xccl`` and that the call happens BEFORE the bf16
cast that produces the tensor handed to vLLM. The source-text check is
intentional and brittle on purpose: any future refactor that drops the
unpermute call will trip this test.

The full numerical round-trip (Q/K → tune permute → unpermute identity) is
already covered by tests/torchtune/dev/rl/test_wsync_qk_unpermute.py; we
don't repeat it here.
"""
import inspect
import re
import unittest

import torchtune.dev.rl.weight_sync as ws


def _xccl_sync_source() -> str:
    """Combined source of the xccl wsync path.

    The gather logic was extracted from ``_sync_weights_to_vllm_xccl`` into
    two per-FSDP-mode helpers (``_xccl_gather_fsdp1`` /
    ``_xccl_gather_and_stage_fsdp2``) on 2026-06-17. The dispatcher now just
    builds the rename closure and calls the right helper, so the Q/K
    un-permute code lives in the helpers. These source-scan guards therefore
    inspect all three functions together — any of them dropping the unpermute
    still trips the test, which is the whole point.
    """
    parts = [inspect.getsource(ws._sync_weights_to_vllm_xccl)]
    for _name in ("_xccl_gather_fsdp1", "_xccl_gather_and_stage_fsdp2"):
        _fn = getattr(ws, _name, None)
        if _fn is not None:
            parts.append(inspect.getsource(_fn))
    return "\n".join(parts)


class TestXcclSyncCallsUnpermute(unittest.TestCase):
    def test_unperm_predicate_captured_before_loop(self):
        """The xccl sync must compute the unpermute predicate outside the
        per-param loop (cheap) and apply it inside."""
        src = _xccl_sync_source()
        # Captured once outside the loop (any of these forms is fine).
        self.assertRegex(
            src,
            r"_unperm_needed_x\s*=\s*_needs_qk_unpermute\(self\)",
            "xccl sync must capture _unperm_needed_x = _needs_qk_unpermute(self) "
            "outside the per-param loop",
        )

    def test_unpermute_called_inside_loop(self):
        """`_maybe_unpermute_qk` must be invoked on the full tensor before the
        bf16 cast that ships the weight to vLLM."""
        src = _xccl_sync_source()
        # Match the precise call site we added.
        match = re.search(
            r"if\s+_unperm_needed_x:\s*\n\s*param\s*=\s*_maybe_unpermute_qk\(\s*self,\s*hf_name,\s*param\s*\)",
            src,
        )
        self.assertIsNotNone(
            match,
            "Expected `if _unperm_needed_x: param = _maybe_unpermute_qk(self, "
            "hf_name, param)` inside _sync_weights_to_vllm_xccl. The colocate "
            "path has had this call since 2026-06-11; the xccl path was added "
            "2026-06-12 for the server-mode AGPT-2B GSM8K run.",
        )

    def test_unpermute_immediately_before_gpu_tensor_cast(self):
        """The unpermute MUST happen between `hf_name = _xccl_accept_and_rename(...)`
        and the bf16 cast that produces `gpu_tensor`. Otherwise the staged
        copy that flows to vLLM would have wrong Q/K. This pins the exact
        ordering in the non-MoE path."""
        src = _xccl_sync_source()
        # The block we care about is the canonical 4-line sequence:
        #   hf_name = _xccl_accept_and_rename(param_name)
        #   ...
        #   if _unperm_needed_x: param = _maybe_unpermute_qk(self, hf_name, param)
        #   ...
        #   gpu_tensor = param.to(torch.bfloat16).contiguous()
        # Find the index of _xccl_accept_and_rename, then assert that BOTH the
        # unpermute and the gpu_tensor cast come AFTER it, in that order, with
        # no intervening `gpu_tensor =` rebind.
        idx_rename = src.find("hf_name = _xccl_accept_and_rename(param_name)")
        self.assertGreater(
            idx_rename, -1,
            "expected `hf_name = _xccl_accept_and_rename(param_name)` in source",
        )
        idx_unperm = src.find(
            "_maybe_unpermute_qk(self, hf_name, param)", idx_rename
        )
        idx_gpu_tensor = src.find(
            "gpu_tensor = param.to(torch.bfloat16).contiguous()", idx_rename
        )
        self.assertGreater(
            idx_unperm, -1,
            "no `_maybe_unpermute_qk(self, hf_name, param)` call found after "
            "the hf_name assignment",
        )
        self.assertGreater(
            idx_gpu_tensor, -1,
            "no `gpu_tensor = param.to(torch.bfloat16).contiguous()` found",
        )
        self.assertLess(
            idx_unperm, idx_gpu_tensor,
            "_maybe_unpermute_qk must precede the gpu_tensor bf16 cast — "
            "otherwise gpu_tensor (and therefore vLLM) sees permuted Q/K",
        )


if __name__ == "__main__":
    unittest.main()

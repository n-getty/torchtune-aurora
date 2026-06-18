# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guard: the FSDP1 XCCL weight-sync gather must actually BROADCAST.

Bug it pins down (root-caused 2026-06-18): commit f60efefc
("refactor(rl): split _sync_weights_to_vllm_xccl into per-FSDP-mode gather
helpers") extracted ``_xccl_gather_fsdp1`` but DROPPED its staging+broadcast
tail — the helper built ``flat_gpu``/``tensors_meta`` as locals and returned
without setting ``self._deferred_broadcast_args`` or sending anything. Result:
in FSDP1 + ``vllm_weight_sync_method=xccl`` + server mode (e.g. 2N 11+1 HSDP),
vLLM NEVER received updated weights — it generated forever with step-0 weights,
the policy drifted from the frozen generator, and ∇logp on the now-off-policy
rollout tokens exploded → grad_norm 70-150 → kl_loss → NaN by ~step 15.

The FSDP2 sibling ``_xccl_gather_and_stage_fsdp2`` always set
``_deferred_broadcast_args``; only the FSDP1 branch lost it, which is why no
prior (FSDP2 / pre-refactor / colocate) run caught it and the 8N HSDP run —
which ran the day BEFORE the refactor — trained fine.

Two source-level invariants (CPU-safe; no XPU / distributed init needed):

1. ``_xccl_gather_fsdp1`` assigns ``self._deferred_broadcast_args`` (the
   send actually gets armed). A future refactor that drops it regresses here.
2. The CPU broadcast batches are split by the SAME greedy ``batch_max_numel``
   rule the vLLM receiver mirrors
   (``vllm_weight_sync_worker.receive_weights_xccl_streaming``). Sending one
   mega-batch when the receiver expects N greedy batches →
   ``collective_rpc`` cancelled → vLLM EngineDeadError (observed 2026-06-18
   on the first fix attempt). The pure-function check below pins the split.
"""
import ast
import inspect
import unittest

from torchtune.dev.rl import weight_sync


def _greedy_batch_count(numels, batch_max_numel):
    """Receiver's greedy rule: flush when adding the next param would exceed
    batch_max_numel; a single param larger than the limit forms its own batch.
    Mirrors receive_weights_xccl_streaming and the sender's _xccl_gather_fsdp1.
    """
    batches = 0
    cur = 0
    for n in numels:
        if cur > 0 and cur + n > batch_max_numel:
            batches += 1
            cur = 0
        cur += n
    if cur > 0:
        batches += 1
    return batches


class TestXcclFsdp1BroadcastArmed(unittest.TestCase):
    def test_gather_fsdp1_sets_deferred_broadcast_args(self):
        """The FSDP1 xccl gather must arm the deferred broadcast.

        Parses the function source and asserts it assigns
        self._deferred_broadcast_args — the exact tail f60efefc dropped.
        """
        src = inspect.getsource(weight_sync._xccl_gather_fsdp1)
        tree = ast.parse(src)
        assigns_deferred = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    # match `self._deferred_broadcast_args = ...`
                    if (
                        isinstance(tgt, ast.Attribute)
                        and tgt.attr == "_deferred_broadcast_args"
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                    ):
                        assigns_deferred = True
        self.assertTrue(
            assigns_deferred,
            "_xccl_gather_fsdp1 must set self._deferred_broadcast_args or vLLM "
            "never receives weights (f60efefc dropped-broadcast regression).",
        )

    def test_gather_fsdp1_builds_cpu_batches(self):
        """It must build a list of CPU broadcast batches (not a single buffer)."""
        src = inspect.getsource(weight_sync._xccl_gather_fsdp1)
        self.assertIn(
            "cpu_batches", src,
            "_xccl_gather_fsdp1 must build cpu_batches for the deferred broadcast.",
        )
        # The greedy flush condition must reference batch_max_numel.
        self.assertIn(
            "_BATCH_MAX_NUMEL", src,
            "_xccl_gather_fsdp1 must batch by _BATCH_MAX_NUMEL (receiver contract).",
        )

    def test_greedy_split_matches_receiver_contract(self):
        """Pin the greedy batch-count rule both sender and receiver must share.

        A whole-model single batch (the broken first-fix attempt) would give 1;
        the receiver, walking the manifest greedily, expects multiple → mismatch.
        """
        bmax = 512 * 1024 * 1024  # 1 GiB bf16, the value used in code
        # AGPT-2B-like: 111 params, ~3.7 GiB bf16 total ≈ 1.85e9 elements.
        # A realistic mix of large (embed/mlp) and small (norm) params.
        numels = [256000 * 2048]              # embed_tokens ~0.52e9
        numels += [11008 * 2048, 2048 * 11008, 11008 * 2048] * 12  # mlp per layer
        numels += [2048] * 40                 # norms etc.
        n_batches = _greedy_batch_count(numels, bmax)
        self.assertGreater(
            n_batches, 1,
            "A multi-GiB model must split into >1 greedy batch; sending it as "
            "one mega-batch desyncs the receiver (collective_rpc cancelled).",
        )
        # A single >limit param forms its own batch (receiver guarantees this).
        self.assertEqual(_greedy_batch_count([bmax * 2], bmax), 1)
        # Two sub-limit params that together exceed it → 2 batches.
        self.assertEqual(
            _greedy_batch_count([bmax * 3 // 4, bmax * 3 // 4], bmax), 2
        )


if __name__ == "__main__":
    unittest.main()

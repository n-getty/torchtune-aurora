# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pins when ``expand()`` is actually free, and when it silently materializes.

HW gate #3 (job 8830196) burned a node-hour discovering that
``k.unsqueeze(1).expand(B, G, ...).flatten(0, 1)`` -- the obvious way to present one
prefix KV to ``G`` rollout rows -- returns a **contiguous copy**. Folding a broadcast
(stride-0) dimension into a neighbouring one always materializes, because the result is
not expressible as a single stride pattern. This is a PyTorch invariant, not an XPU
kernel limitation, so it is checkable on CPU in milliseconds.

The same cancellation sits unnoticed in ``torchtune/dev/rl/ref_prefix_share.py``:
``expand_cache_batch_`` expands the cache to a stride-0 view and its docstring claims the
prefix "costs one row of KV", then notes that the following ``cache.update()``
concatenates and materializes it. The saving is negated inside the same function.

What makes this worth a test rather than a comment: the failure is **silent and
performance-only**. Nothing errors, numerics are identical, and memory quietly grows by
``G``x -- at BioReason-32B production shape, 4.3 GiB becomes ~34 GiB on a 64 GiB tile.
The only way to catch it is to assert on strides and storage identity.

The positive cases matter just as much as the negative ones: ``[B*G, ...] -> [B, G, ...]``
IS a free view when rows are group-contiguous (GRPO's layout guarantees this), which is
what makes the "fold G into the head dim" design viable at all.
"""
import pytest
import torch

B, G, N_KV, N_Q, P, R, D = 4, 8, 2, 8, 16, 5, 4


class TestExpandThatMaterializes:
    def test_expand_itself_is_free(self):
        """Baseline: the expand alone really is a stride-0 view."""
        k = torch.randn(B, N_KV, P, D)
        e = k.unsqueeze(1).expand(B, G, N_KV, P, D)
        assert 0 in e.stride(), f"expected a stride-0 dim, got stride={e.stride()}"
        assert e.data_ptr() == k.data_ptr()

    @pytest.mark.parametrize("fold", ["flatten", "reshape"])
    def test_folding_the_broadcast_dim_into_batch_materializes(self, fold):
        """THE trap: the fold silently copies, and nothing signals it."""
        k = torch.randn(B, N_KV, P, D)
        e = k.unsqueeze(1).expand(B, G, N_KV, P, D)
        out = e.flatten(0, 1) if fold == "flatten" else e.reshape(B * G, N_KV, P, D)

        assert 0 not in out.stride(), (
            f"stride-0 unexpectedly survived {fold}; if PyTorch gained this ability the "
            "prefix-dedup design can be simplified -- re-read "
            "memory/project_bioreason_stride0_kv_impossible_fold_g_into_heads_20260916.md"
        )
        assert out.data_ptr() != k.data_ptr(), (
            f"{fold} appears to share storage; see above -- this would be good news"
        )
        assert out.is_contiguous()

    def test_view_refuses_rather_than_copying(self):
        """`view` at least fails loudly; `reshape`/`flatten` do not. Know the difference."""
        k = torch.randn(B, N_KV, P, D)
        e = k.unsqueeze(1).expand(B, G, N_KV, P, D)
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            e.view(B * G, N_KV, P, D)

    def test_materialized_fold_costs_g_times_the_memory(self):
        """Quantify the silent cost, so the number in the memory entry is reproducible."""
        k = torch.randn(B, N_KV, P, D)
        folded = k.unsqueeze(1).expand(B, G, N_KV, P, D).flatten(0, 1)
        assert folded.numel() == k.numel() * G


class TestFoldsThatAreFree:
    def test_group_contiguous_rows_split_into_a_view(self):
        """[B*G, ...] -> [B, G, ...] is free, which the dedup design depends on.

        GRPO lays prompt b's G continuations at rows [b*G, (b+1)*G), so this split is a
        plain view. If rows were group-STRIDED it would not be, and every rollout would
        be paired with the wrong prompt.
        """
        q = torch.randn(B * G, N_Q, R, D)
        v = q.view(B, G, N_Q, R, D)
        assert v.data_ptr() == q.data_ptr()
        for b in range(B):
            for g in range(G):
                assert torch.equal(v[b, g], q[b * G + g])

    def test_head_fold_ordering_must_be_head_major(self):
        """GQA maps query head i to KV head i // (n_q_total // n_kv).

        For folded head index ``h*G + g`` that yields ``h``, correctly routing group g of
        prompt b to prompt b's KV head h. The naive ``view`` gives ``g*n_q + h``, which
        routes by GROUP -- finite losses, no crash, silently wrong training. Pin both the
        right answer and the fact that the wrong one differs.
        """
        q = torch.arange(B * G * N_Q, dtype=torch.float32).reshape(B * G, N_Q, 1, 1)
        head_major = (
            q.view(B, G, N_Q, 1, 1).permute(0, 2, 1, 3, 4).reshape(B, N_Q * G, 1, 1)
        )
        ratio = (N_Q * G) // N_KV
        for b in range(B):
            for h in range(N_Q):
                for g in range(G):
                    idx = h * G + g
                    assert head_major[b, idx, 0, 0].item() == q[b * G + g, h, 0, 0].item()
                    # the KV head this query head will read
                    assert idx // ratio == (h * G + g) // ratio

        naive = q.view(B, G * N_Q, 1, 1)
        assert not torch.equal(naive, head_major), (
            "naive view and head-major permute agree; the fixture is degenerate and "
            "would not catch a mis-routed fold"
        )

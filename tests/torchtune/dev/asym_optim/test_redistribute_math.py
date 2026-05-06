# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU-only tests for the 12->4 / 4->12 redistribute helpers used by
# AsymAdamWXPU. These run on a login node — no XPU, no dist init.

import pytest
import torch

from torchtune.dev.asym_optim.redistribute import (
    build_a2a_splits,
    compute_overlap_matrix,
    cpu_simulate_round_trip,
)


@pytest.mark.parametrize(
    "n_src,src_shard_size,n_dst",
    [
        (12, 256, 4),
        (12, 257, 4),  # not divisible by n_dst — exercise pad path
        (8, 33, 4),
        (4, 100, 4),   # n_src == n_dst, identity-ish
        (12, 1, 4),    # tiny shards
    ],
)
def test_round_trip_identity(n_src, src_shard_size, n_dst):
    torch.manual_seed(0)
    src_shards = [
        torch.arange(i * src_shard_size, (i + 1) * src_shard_size, dtype=torch.float32)
        for i in range(n_src)
    ]
    out = cpu_simulate_round_trip(src_shards, n_dst)
    for orig, got in zip(src_shards, out):
        assert torch.allclose(orig, got), f"round-trip mismatch: {orig} vs {got}"


def test_overlap_matrix_row_and_col_sums():
    n_src, src_shard_size, n_dst = 12, 32, 4
    overlap, dst_split_size = compute_overlap_matrix(n_src, src_shard_size, n_dst)
    # Every trainer shard must be fully accounted for in the gather (rows sum to src_shard_size).
    for i in range(n_src):
        assert sum(overlap[i]) == src_shard_size
    # Every spare shard receives exactly dst_split_size elements (cols may include padding).
    total_received = sum(sum(overlap[i][j] for i in range(n_src)) for j in range(n_dst))
    assert total_received == n_src * src_shard_size
    assert dst_split_size * n_dst >= n_src * src_shard_size


def test_a2a_splits_consistency_gather_direction():
    # Trainer i's input_split[j] must equal spare j's output_split[i] (mirror property).
    n_src, src_shard_size, n_dst = 12, 33, 4
    train_ranks = list(range(n_src))
    spare_ranks = list(range(n_src, n_src + n_dst))
    pg_ranks = train_ranks + spare_ranks

    overlap, _ = compute_overlap_matrix(n_src, src_shard_size, n_dst)
    for i, tr in enumerate(train_ranks):
        in_t, _ = build_a2a_splits(
            overlap, pg_ranks, train_ranks, spare_ranks, tr, "gather"
        )
        for j, sp in enumerate(spare_ranks):
            _, out_s = build_a2a_splits(
                overlap, pg_ranks, train_ranks, spare_ranks, sp, "gather"
            )
            # in_t at the spare-j position must match out_s at the trainer-i position.
            assert in_t[pg_ranks.index(sp)] == out_s[pg_ranks.index(tr)] == overlap[i][j]


def test_a2a_splits_zero_for_idle_pairs():
    # In gather direction, spare->spare and trainer->trainer entries are zero.
    n_src, src_shard_size, n_dst = 4, 16, 2
    train_ranks = [0, 1, 2, 3]
    spare_ranks = [4, 5]
    pg_ranks = train_ranks + spare_ranks
    overlap, _ = compute_overlap_matrix(n_src, src_shard_size, n_dst)
    for sp in spare_ranks:
        in_s, _ = build_a2a_splits(
            overlap, pg_ranks, train_ranks, spare_ranks, sp, "gather"
        )
        for sp_other in spare_ranks:
            assert in_s[pg_ranks.index(sp_other)] == 0
    for tr in train_ranks:
        _, out_t = build_a2a_splits(
            overlap, pg_ranks, train_ranks, spare_ranks, tr, "gather"
        )
        for tr_other in train_ranks:
            assert out_t[pg_ranks.index(tr_other)] == 0


def test_invalid_direction_raises():
    overlap, _ = compute_overlap_matrix(4, 16, 2)
    with pytest.raises(ValueError):
        build_a2a_splits(
            overlap, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3], [4, 5], 0, "BOGUS",
        )


def test_compute_overlap_matrix_validates_inputs():
    with pytest.raises(ValueError):
        compute_overlap_matrix(0, 10, 4)
    with pytest.raises(ValueError):
        compute_overlap_matrix(12, 10, 0)


def test_adamw_step_matches_dense_baseline():
    """Single AdamW step on the asym-optim pipeline (CPU-simulated) must match
    a vanilla 12-shard AdamW step within 1e-6 (FP32 throughout).

    This is the behavioral equivalent: collect all 12 trainer shards into a
    global tensor, run one AdamW step, then re-shard back.
    """
    torch.manual_seed(0)
    n_src, src_shard_size, n_dst = 12, 64, 4
    total = n_src * src_shard_size

    global_param = torch.randn(total, dtype=torch.float32)
    global_grad = torch.randn(total, dtype=torch.float32)
    src_shards = [
        global_param[i * src_shard_size : (i + 1) * src_shard_size].clone()
        for i in range(n_src)
    ]
    grad_shards = [
        global_grad[i * src_shard_size : (i + 1) * src_shard_size].clone()
        for i in range(n_src)
    ]

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    # ---- baseline: vanilla AdamW on the global tensor ----
    p = global_param.clone()
    g = global_grad.clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    p_baseline = p - lr * m_hat / (v_hat.sqrt() + eps)

    # ---- asym pipeline ----
    # 1. gather grads 12 -> 4
    overlap, dst_split_size = compute_overlap_matrix(n_src, src_shard_size, n_dst)
    grad_dst = [torch.zeros(dst_split_size, dtype=torch.float32) for _ in range(n_dst)]
    for j in range(n_dst):
        offset = 0
        for i in range(n_src):
            o = overlap[i][j]
            if o == 0:
                continue
            dst_lo = j * dst_split_size
            src_lo = i * src_shard_size
            a = max(0, dst_lo + offset - src_lo)
            grad_dst[j][offset : offset + o] = grad_shards[i][a : a + o]
            offset += o
    # Same for params (seed master).
    param_dst = [torch.zeros(dst_split_size, dtype=torch.float32) for _ in range(n_dst)]
    for j in range(n_dst):
        offset = 0
        for i in range(n_src):
            o = overlap[i][j]
            if o == 0:
                continue
            dst_lo = j * dst_split_size
            src_lo = i * src_shard_size
            a = max(0, dst_lo + offset - src_lo)
            param_dst[j][offset : offset + o] = src_shards[i][a : a + o]
            offset += o

    # 2. AdamW math per spare shard
    new_param_dst = []
    for j in range(n_dst):
        master = param_dst[j].clone()
        m4 = torch.zeros_like(master)
        v4 = torch.zeros_like(master)
        g32 = grad_dst[j]
        m4.mul_(beta1).add_(g32, alpha=1.0 - beta1)
        v4.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
        m_hat = m4 / (1.0 - beta1)
        v_hat = v4 / (1.0 - beta2)
        master.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
        new_param_dst.append(master)

    # 3. scatter 4 -> 12
    out_shards = [torch.zeros(src_shard_size, dtype=torch.float32) for _ in range(n_src)]
    for i in range(n_src):
        offset = 0
        src_lo = i * src_shard_size
        for j in range(n_dst):
            o = overlap[i][j]
            if o == 0:
                continue
            dst_lo = j * dst_split_size
            a = max(0, src_lo + offset - dst_lo)
            out_shards[i][offset : offset + o] = new_param_dst[j][a : a + o]
            offset += o
    asym_global = torch.cat(out_shards)

    assert torch.allclose(asym_global, p_baseline, atol=1e-6), (
        f"asym AdamW diverges from baseline: max diff = "
        f"{(asym_global - p_baseline).abs().max().item():.2e}"
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pins the FLOP accounting that decides the SHAPE of prefix dedup.

Three numbers drive the design, and getting any of them wrong sends the work in the
wrong direction:

1. **Attention is only ~8% of the layer's work** at BioReason-32B production shape. The
   other ~92% is QKV/O projections and MLP.
2. **Attention-only dedup is therefore a 1.00x no-op** -- not "a smaller win." Sharing
   prefix *keys* does not shrink the *queries*: all ``B*G`` rows still carry ``P+R``
   query positions, so neither the score work nor any linear layer shrinks. There is no
   partial-credit version of this lever.
3. **Two-stream dedup is ~3.17x** (prefix once over ``B`` rows, responses over ``B*G``),
   which independently reproduces the 3.15x token-count ratio in
   ``torchtune/dev/rl/group_chunking.py`` -- a cross-check that the token-count proxy is
   measuring the right thing.

(2) is the load-bearing claim: it means the stride-0 prefix-KV question (HW gate #3,
``experiments/bioreason/probe_split_sdpa_stride0_kv.py``) is not about how much dedup is
worth, but whether it exists at all. A future reader tempted by "just split the
attention, it's simpler" should fail this test instead of spending a run finding out.

These are FLOP counts, not time predictions. The backward on this workload is measured
*not* to be FLOP-bound (memory/project_bioreason_lensort_removes_padding_but_slows_bwd_
20260915.md: 25.6% of padded FLOPs removed for +0%), so treat 3.17x as an upper bound.
The 1.00x, by contrast, is exact in any model: nothing is removed.
"""
import pytest

# Qwen3-32B
LAYERS, HIDDEN, N_Q, N_KV, HEAD_DIM, INTERMEDIATE = 64, 5120, 64, 8, 128, 25600
# BioReason 2N production rollout shape
PREFIX, RESP, GROUP, BATCH = 4096, 1152, 8, 4
SEQ = PREFIX + RESP


def _layer_flops(q_len: int, kv_len: int, rows: int) -> tuple[float, float]:
    """Return ``(linear_flops, attention_flops)`` for one layer.

    Linear work scales with query positions only; attention score/AV work scales with
    ``q_len * kv_len`` (halved for causal).
    """
    qkv = 2 * q_len * HIDDEN * (N_Q * HEAD_DIM + 2 * N_KV * HEAD_DIM)
    o_proj = 2 * q_len * (N_Q * HEAD_DIM) * HIDDEN
    mlp = 2 * q_len * HIDDEN * INTERMEDIATE * 3
    linear = (qkv + o_proj + mlp) * rows
    attn = 2 * 2 * q_len * kv_len * N_Q * HEAD_DIM * 0.5 * rows
    return linear, attn


@pytest.fixture(scope="module")
def totals():
    lin_b, attn_b = _layer_flops(SEQ, SEQ, BATCH * GROUP)
    lin_p, attn_p = _layer_flops(PREFIX, PREFIX, BATCH)
    lin_r, attn_r = _layer_flops(RESP, SEQ, BATCH * GROUP)
    return {
        "baseline": lin_b + attn_b,
        "baseline_linear": lin_b,
        "baseline_attn": attn_b,
        "two_stream": lin_p + attn_p + lin_r + attn_r,
    }


def test_linear_layers_dominate(totals):
    """If attention were most of the work, an attention-only split might suffice."""
    share = totals["baseline_linear"] / totals["baseline"]
    assert share > 0.85, (
        f"linear layers are only {share:.1%} of per-layer FLOPs; the premise that "
        "dedup must remove MLP/QKV work (not just attention) no longer holds"
    )


def test_attention_only_dedup_is_a_no_op(totals):
    """THE load-bearing claim: sharing prefix KEYS removes zero FLOPs.

    An attention-only split leaves every row carrying P+R query positions, so the
    linear layers are untouched AND the score work is untouched. Asserted exactly,
    because the result is exact -- nothing is removed.
    """
    lin_ao, attn_ao = _layer_flops(SEQ, SEQ, BATCH * GROUP)
    assert lin_ao + attn_ao == totals["baseline"], (
        "attention-only dedup appears to remove FLOPs; it cannot. Sharing prefix keys "
        "does not shrink the queries, so no linear layer and no score computation "
        "changes. If this fails, the FLOP model changed -- re-derive before trusting it."
    )


def test_two_stream_dedup_is_worth_about_three_x(totals):
    speedup = totals["baseline"] / totals["two_stream"]
    assert 2.9 < speedup < 3.4, (
        f"two-stream dedup FLOP speedup is {speedup:.2f}x, outside the expected ~3.2x. "
        "Either the model shape constants drifted or the accounting changed."
    )


def test_flop_model_agrees_with_token_count_proxy(totals):
    """Cross-check: the token-count ratio used elsewhere should land in the same place.

    group_chunking.py quotes 3.15x at fbs=8 from pure token counting. Two independent
    routes agreeing is what licenses using the cheap proxy for the fbs table.
    """
    flop_speedup = totals["baseline"] / totals["two_stream"]
    token_baseline = BATCH * GROUP * SEQ
    token_two_stream = BATCH * PREFIX + BATCH * GROUP * RESP
    token_speedup = token_baseline / token_two_stream
    assert abs(flop_speedup - token_speedup) / token_speedup < 0.05, (
        f"FLOP model says {flop_speedup:.2f}x but token counting says "
        f"{token_speedup:.2f}x; the proxy and the model disagree, so at least one of "
        "them is not measuring what the fbs payoff table claims"
    )

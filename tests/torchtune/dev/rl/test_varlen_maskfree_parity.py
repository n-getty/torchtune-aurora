# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU parity test for TORCHTUNE_MASKFREE_CAUSAL mask-free causal forward.

Proves three invariants:

1. SDPA_PARITY — for right-padded-only sequences (no prompt padding), PyTorch SDPA
   with `is_causal=True, attn_mask=None` is numerically equivalent (within fp32
   tolerance) to SDPA with `is_causal=False, attn_mask=explicit_causal_mask`. This
   underpins the correctness of the mask-free forward path.

2. MID_SEQ_PADDING_DIVERGES — when a sequence has PAD tokens in the middle (from
   variable-length prompt batching), the two SDPA calls DO diverge. This proves the
   runtime guard in generate_trajectory() is load-bearing — without it the maskfree
   path silently produces wrong gradients.

3. PROMPT_PAD_GUARD_LOGIC — unit test for the exact detection expression used in the
   recipe:
       has_prompt_pad = (query_responses[:, :context_length] == pad_id).any().item()
   Verifies True on padded prompts and False on clean prompts.

4. TINY_MODEL_PARITY — end-to-end forward parity on a tiny (randomly initialized)
   Qwen3-like transformer: 2 layers, embed=64, 4 heads, head_dim=16.
   Uses `torchtune.models.qwen3.qwen3` (the component builder) at micro scale.
   Keeps peak memory <50 MB; runs in <0.5 s on CPU.

No XPU, no distributed init, no checkpoint required.  Runs in <5 s on CPU.
"""
import unittest

import torch
import torch.nn.functional as F

from torchtune.generation import (
    get_causal_mask_from_padding_mask,
    get_position_ids_from_padding_mask,
)

PAD_ID = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explicit_causal_mask(seq_len: int, device="cpu") -> torch.Tensor:
    """Lower-triangular bool mask [seq_len, seq_len]."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _sdpa_causal_implicit(q, k, v):
    """SDPA with is_causal=True (mask=None path)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)


def _sdpa_causal_explicit(q, k, v, seq_len):
    """SDPA with explicit lower-triangular causal mask."""
    mask = _explicit_causal_mask(seq_len, device=q.device)
    # SDPA expects additive mask [B, H, S, S] or broadcastable bool
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[None, None, :, :], is_causal=False
    )


def _make_qkv(batch, heads, seq, head_dim, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)
    return q, k, v


def _make_qr_batch(
    batch_size, context_length, response_length,
    prompt_pad_counts, response_pad_counts, vocab_size, seed=0,
):
    """
    Build [B, P+R] integer token tensor.
    prompt_pad_counts[i]: trailing PAD tokens in the prompt portion.
    response_pad_counts[i]: trailing PAD tokens in the response portion.
    Non-PAD tokens are random in [1, vocab_size-1].
    """
    torch.manual_seed(seed)
    total = context_length + response_length
    tokens = torch.randint(1, vocab_size, (batch_size, total))
    for i in range(batch_size):
        if prompt_pad_counts[i] > 0:
            tokens[i, context_length - prompt_pad_counts[i]: context_length] = PAD_ID
        if response_pad_counts[i] > 0:
            tokens[i, total - response_pad_counts[i]:] = PAD_ID
    return tokens


# ---------------------------------------------------------------------------
# 1. SDPA parity: is_causal=True vs explicit mask
# ---------------------------------------------------------------------------

class TestSDPAParity(unittest.TestCase):
    """
    is_causal=True and an explicit lower-triangular causal mask must produce
    numerically equivalent outputs (within float32 rounding).
    This is a PyTorch invariant that the maskfree path relies on.
    """

    def _check_parity(self, B, H, S, D, seed):
        q, k, v = _make_qkv(B, H, S, D, seed=seed)
        out_implicit = _sdpa_causal_implicit(q, k, v)
        out_explicit = _sdpa_causal_explicit(q, k, v, S)
        max_diff = (out_implicit - out_explicit).abs().max().item()
        self.assertLess(
            max_diff, 1e-5,
            f"SDPA causal implicit vs explicit mismatch: max_diff={max_diff:.2e} "
            f"(B={B} H={H} S={S} D={D} seed={seed})"
        )

    def test_small_sequence(self):
        self._check_parity(2, 4, 16, 8, seed=1)

    def test_medium_sequence(self):
        self._check_parity(3, 8, 64, 16, seed=2)

    def test_single_batch(self):
        self._check_parity(1, 2, 32, 8, seed=3)


# ---------------------------------------------------------------------------
# 2. Mid-sequence padding causes divergence
# ---------------------------------------------------------------------------

class TestMidSeqPaddingDivergence(unittest.TestCase):
    """
    With mid-sequence padding (prompt holes), `is_causal=True` attends through
    PAD positions while the explicit mask (from get_causal_mask_from_padding_mask)
    zeros them out — outputs must differ.
    Proves the runtime guard in generate_trajectory() is load-bearing.
    """

    def test_mid_seq_padding_diverges(self):
        """
        Construct a sequence where token at position P is PAD (mid-sequence).
        is_causal=True will include that PAD's K/V in attention output for
        tokens after P; the explicit padding mask excludes it.
        Outputs at positions after P must differ.
        """
        B, H, S, D = 1, 2, 8, 8
        torch.manual_seed(99)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        # Build explicit mask that blacks out position 3 (a "PAD" position)
        # Simulate get_causal_mask_from_padding_mask: causal but position 3 is masked out.
        causal = _explicit_causal_mask(S)
        # Mark position 3 as PAD: no token at position 3 can be attended to
        causal[:, 3] = False

        out_implicit = _sdpa_causal_implicit(q, k, v)
        out_explicit = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal[None, None, :, :], is_causal=False
        )

        # Positions after the PAD (3) must see different attention — attn to pos 3 differs
        diff = (out_implicit[:, :, 4:, :] - out_explicit[:, :, 4:, :]).abs()
        max_diff = diff.max().item()
        self.assertGreater(
            max_diff, 1e-4,
            f"Expected divergence after mid-sequence PAD masking, but max_diff={max_diff:.2e}. "
            "The runtime guard may be unnecessary, or the attention heads collapsed."
        )


# ---------------------------------------------------------------------------
# 3. Prompt-padding guard logic
# ---------------------------------------------------------------------------

class TestPromptPadGuard(unittest.TestCase):
    """
    Unit test for the exact detection expression used in generate_trajectory():
        has_prompt_pad = (query_responses[:, :context_length] == PAD_ID).any().item()
    """

    def test_clean_prompts_no_guard(self):
        context_length = 16
        response_length = 8
        qr = _make_qr_batch(
            2, context_length, response_length,
            prompt_pad_counts=[0, 0],
            response_pad_counts=[3, 1],
            vocab_size=1000,
            seed=5,
        )
        has_pad = (qr[:, :context_length] == PAD_ID).any().item()
        self.assertFalse(has_pad, "Guard must NOT fire when prompts have no padding")

    def test_padded_prompt_triggers_guard(self):
        context_length = 16
        response_length = 8
        qr = _make_qr_batch(
            2, context_length, response_length,
            prompt_pad_counts=[4, 0],   # seq 0 has 4 prompt-pad tokens
            response_pad_counts=[0, 0],
            vocab_size=1000,
            seed=6,
        )
        has_pad = (qr[:, :context_length] == PAD_ID).any().item()
        self.assertTrue(has_pad, "Guard must fire when a prompt contains PAD tokens")

    def test_response_pad_does_not_trigger_guard(self):
        """PAD tokens only in the response portion must NOT trigger the guard."""
        context_length = 16
        response_length = 8
        qr = _make_qr_batch(
            2, context_length, response_length,
            prompt_pad_counts=[0, 0],
            response_pad_counts=[5, 0],
            vocab_size=1000,
            seed=7,
        )
        has_pad = (qr[:, :context_length] == PAD_ID).any().item()
        self.assertFalse(has_pad, "Guard must NOT fire for response-only padding")


# ---------------------------------------------------------------------------
# 4. Tiny model end-to-end parity
# ---------------------------------------------------------------------------

class TestTinyModelParity(unittest.TestCase):
    """
    End-to-end parity on a micro Qwen3-like model (2 layers, embed=64).
    Compares mask=get_causal_mask vs mask=None for right-padded batches.
    Peak memory <50 MB; runs in <0.5 s on CPU.
    """

    @classmethod
    def setUpClass(cls):
        from torchtune.models.qwen3._component_builders import qwen3
        torch.manual_seed(42)
        cls.model = qwen3(
            vocab_size=512,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            embed_dim=64,
            intermediate_dim=128,
            max_seq_len=128,
            head_dim=16,
            attn_dropout=0.0,
            norm_eps=1e-6,
            rope_base=10000.0,
            q_proj_bias=False,
            k_proj_bias=False,
            v_proj_bias=False,
            q_norm=True,
            k_norm=True,
        )
        cls.model.eval()
        for p in cls.model.parameters():
            p.requires_grad_(False)
        cls.vocab_size = 512
        cls.context_length = 12
        cls.response_length = 8

    def _parity(self, qr):
        pad_mask = qr != PAD_ID
        mask = get_causal_mask_from_padding_mask(pad_mask)
        pos_ids = get_position_ids_from_padding_mask(pad_mask)
        with torch.no_grad():
            logits_masked = self.model(qr, input_pos=pos_ids, mask=mask)
            logits_free = self.model(qr, input_pos=pos_ids, mask=None)
        cl = self.context_length
        rl = self.response_length
        valid = (qr[:, cl:] != PAD_ID)  # [B, R]
        # logits[:, cl-1] predicts response token 0, …, logits[:, cl-1+rl-1] predicts token rl-1.
        # Slice to exactly [B, R, V] so valid's [B, R] shape matches the first two dims.
        lp_m = torch.log_softmax(logits_masked[:, cl - 1: cl - 1 + rl], dim=-1)[valid]
        lp_f = torch.log_softmax(logits_free[:, cl - 1: cl - 1 + rl], dim=-1)[valid]
        return lp_m, lp_f

    def test_uniform_prompts_parity(self):
        """No prompt padding + right-padded responses → bit-exact at valid positions."""
        qr = _make_qr_batch(
            2, self.context_length, self.response_length,
            prompt_pad_counts=[0, 0],
            response_pad_counts=[2, 0],
            vocab_size=self.vocab_size,
            seed=10,
        )
        lp_m, lp_f = self._parity(qr)
        self.assertTrue(
            torch.equal(lp_m, lp_f),
            f"Uniform-prompt parity failed: max_diff={(lp_m - lp_f).abs().max().item():.2e}"
        )

    def test_response_only_padding_parity(self):
        """All prompts full + multiple response pad lengths → bit-exact at valid positions."""
        qr = _make_qr_batch(
            3, self.context_length, self.response_length,
            prompt_pad_counts=[0, 0, 0],
            response_pad_counts=[5, 2, 0],
            vocab_size=self.vocab_size,
            seed=11,
        )
        lp_m, lp_f = self._parity(qr)
        self.assertTrue(
            torch.equal(lp_m, lp_f),
            f"Response-only padding parity failed: max_diff={(lp_m - lp_f).abs().max().item():.2e}"
        )

    def test_mid_seq_padding_diverges(self):
        """Variable-length prompts create prompt-side padding → outputs diverge."""
        qr = _make_qr_batch(
            2, self.context_length, self.response_length,
            prompt_pad_counts=[4, 0],
            response_pad_counts=[0, 0],
            vocab_size=self.vocab_size,
            seed=12,
        )
        lp_m, lp_f = self._parity(qr)
        max_diff = (lp_m - lp_f).abs().max().item()
        self.assertGreater(
            max_diff, 1e-4,
            f"Expected divergence with prompt padding, but max_diff={max_diff:.2e}"
        )


# ---------------------------------------------------------------------------
# 5. is_causal 3-way AND routing in MultiHeadAttention.forward
# ---------------------------------------------------------------------------

class TestIsCausalRouting(unittest.TestCase):
    """
    Tests the 3-way AND condition in MultiHeadAttention.forward (attention.py:~298):
        is_causal = kv_cache is None AND mask is None AND self.is_causal

    Strategy: instantiate a tiny MHA, replace its _attention_call with a spy that
    captures the is_causal kwarg, then assert the captured value.
    """

    @classmethod
    def setUpClass(cls):
        from torch import nn
        from torchtune.modules import MultiHeadAttention
        E, H, KVH, HD = 8, 2, 2, 4

        def _make_mha(is_causal=True):
            return MultiHeadAttention(
                embed_dim=E,
                num_heads=H,
                num_kv_heads=KVH,
                head_dim=HD,
                q_proj=nn.Linear(E, H * HD, bias=False),
                k_proj=nn.Linear(E, KVH * HD, bias=False),
                v_proj=nn.Linear(E, KVH * HD, bias=False),
                output_proj=nn.Linear(H * HD, E, bias=False),
                is_causal=is_causal,
            )

        cls._make_mha = staticmethod(_make_mha)
        cls.E = E

    def _spy_call(self, mha):
        """Replace mha._attention_call with a spy; return (mha, captured list)."""
        captured = []

        def _spy(q, k, v, mask, dropout_p, is_causal):
            captured.append(is_causal)
            return torch.zeros_like(q)

        mha._attention_call = _spy
        return mha, captured

    def _forward(self, mha, mask=None):
        B, S = 1, 4
        x = torch.randn(B, S, self.E)
        with torch.no_grad():
            mha(x, x, mask=mask)  # self-attention: y=x (required when no kv_cache)

    def test_all_conditions_true(self):
        """kv_cache=None, mask=None, is_causal=True → is_causal passed as True."""
        mha, captured = self._spy_call(self._make_mha(is_causal=True))
        self._forward(mha, mask=None)
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0], "Expected is_causal=True when all conditions met")

    def test_explicit_mask_suppresses(self):
        """mask != None → is_causal passed as False regardless of self.is_causal."""
        mha, captured = self._spy_call(self._make_mha(is_causal=True))
        B, S = 1, 4
        explicit_mask = torch.tril(torch.ones(B, S, S, dtype=torch.bool))
        self._forward(mha, mask=explicit_mask)
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0], "Explicit mask should suppress is_causal=True")

    def test_kv_cache_suppresses(self):
        """kv_cache is not None → is_causal passed as False regardless of self.is_causal."""
        mha = self._make_mha(is_causal=True)
        # Inject a non-None kv_cache without calling setup_cache() so cache_enabled=False
        mha.kv_cache = object()
        mha, captured = self._spy_call(mha)
        self._forward(mha, mask=None)
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0], "Non-None kv_cache should suppress is_causal=True")

    def test_module_not_causal(self):
        """self.is_causal=False → is_causal passed as False even with mask=None, kv_cache=None."""
        mha, captured = self._spy_call(self._make_mha(is_causal=False))
        self._forward(mha, mask=None)
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0], "self.is_causal=False should yield is_causal=False")


if __name__ == "__main__":
    unittest.main()

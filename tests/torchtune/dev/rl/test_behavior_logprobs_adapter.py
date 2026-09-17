"""vLLM behavior-policy logprobs must equal the trainer's pi_old, or refuse.

CONTEXT. Async gen/train overlap nets 1.282x instead of ~1.60x because turning it on ADDS
a ~95s rollout policy forward: GRPOLoss needs a real pi_old and the recipe's only source
is a second no-grad forward over the whole [B*G, P+C] batch. vLLM's sampler already
computed exactly those numbers. This adapter converts them; these tests are its
correctness certificate.

Three failure modes are asserted, in order of how invisible they would be:

1. THE SHIFT. In the logits path position t predicts token t+1, so the recipe slices
   [:, ctx-1:-1]. vLLM's token_logprobs[j] is the logprob OF comp[j] -- already gathered,
   no shift. Applying the logits path's off-by-one would still produce a correctly-shaped
   tensor of plausible magnitudes and would bias every IS ratio. A negative control here
   asserts the shifted variant genuinely FAILS the equivalence, so the passing test is
   not passing vacuously on a symmetric fixture.

2. THE MODE. vLLM defaults to raw_logprobs (unscaled). The trainer computes
   log_softmax(logits / T). At T=0.8 that is a systematic mismatch, worst IS-ratio error
   4.09x. The guard must reject both the wrong mode AND an unknown one -- "unknown" is
   far more likely to be the wrong default than the right override.

3. THE MIXED BATCH. Rows that fail must send the WHOLE batch to the policy forward, never
   be patched in place. vLLM logprobs and a recomputed forward differ by recompute noise
   (ratios up to 1.0739 on a healthy run), so a per-row mixture carries a systematic
   difference correlated with which rows failed -- a confound nothing downstream detects.

Login-node tests: no XPU, no vLLM, no allocation.
"""

import math

import pytest
import torch

from torchtune.dev.rl.behavior_logprobs import (
    PAD_FILL,
    PROCESSED_LOGPROBS_MODE,
    build_behavior_logprobs,
    require_processed_mode,
    rows_needing_fallback,
)
from torchtune.rlhf.sequence_processing import batched_logits_to_logprobs


# ---------------------------------------------------------------- mode guard

def test_processed_mode_accepted():
    require_processed_mode(PROCESSED_LOGPROBS_MODE)  # must not raise


@pytest.mark.parametrize("mode", ["raw_logprobs", None, "", "processed", "RAW"])
def test_non_processed_mode_rejected(mode):
    """The default mode is the wrong one, so anything unconfirmed must fail closed."""
    with pytest.raises(ValueError, match="processed_logprobs"):
        require_processed_mode(mode)


def test_mode_error_names_the_consequence():
    """An error that only says 'wrong mode' invites someone to override the check."""
    with pytest.raises(ValueError) as ei:
        require_processed_mode("raw_logprobs")
    msg = str(ei.value)
    assert "temperature" in msg
    assert "systematic" in msg


# ------------------------------------------------- equivalence to the trainer

def _trainer_logprobs(logits, tokens, temperature):
    return batched_logits_to_logprobs(logits, tokens, temperature)


def test_matches_trainer_formula_exactly():
    """A processed-mode sampler logprob IS log_softmax(logits/T) gathered at the token.

    Simulates the server: build logits, compute what a processed-mode vLLM would report
    per sampled token, ship it through the adapter, and compare against the trainer's own
    function on the same logits.
    """
    torch.manual_seed(0)
    b, c, v, temp = 4, 7, 23, 0.8
    logits = torch.randn(b, c, v, dtype=torch.float32)
    tokens = torch.randint(0, v, (b, c))

    trainer = _trainer_logprobs(logits, tokens, temp)

    # What a processed-mode server returns: the gathered value, one per generated token.
    wire = [[float(trainer[i, j]) for j in range(c)] for i in range(b)]
    comps = [[int(tokens[i, j]) for j in range(c)] for i in range(b)]

    got = build_behavior_logprobs(wire, comps, num_seqs=b, max_generated_tokens=c)

    assert got.shape == (b, c)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, trainer, rtol=1e-6, atol=1e-6)


def test_shifted_variant_is_actually_wrong():
    """Negative control for the no-shift claim.

    If the fixture were symmetric enough that a shifted alignment also passed, the
    equivalence test above would be vacuous. Assert the off-by-one genuinely diverges.
    """
    torch.manual_seed(1)
    b, c, v, temp = 3, 9, 31, 0.8
    logits = torch.randn(b, c, v, dtype=torch.float32)
    tokens = torch.randint(0, v, (b, c))
    trainer = _trainer_logprobs(logits, tokens, temp)

    # The bug: annotate token j with the logprob belonging to token j-1.
    shifted = [[float(trainer[i, max(j - 1, 0)]) for j in range(c)] for i in range(b)]
    comps = [[int(tokens[i, j]) for j in range(c)] for i in range(b)]
    got = build_behavior_logprobs(shifted, comps, num_seqs=b, max_generated_tokens=c)

    assert not torch.allclose(got, trainer, rtol=1e-3, atol=1e-3), (
        "a shifted alignment matched the trainer -- this fixture cannot detect the "
        "off-by-one, so the equivalence test proves nothing"
    )


# --------------------------------------------------------------- padding/shape

def test_short_rows_are_padded_not_truncated_away():
    wire = [[-0.1, -0.2], [-0.3], [-0.4, -0.5, -0.6]]
    comps = [[5, 6], [7], [8, 9, 10]]
    got = build_behavior_logprobs(wire, comps, num_seqs=3, max_generated_tokens=5)
    assert got.shape == (3, 5)
    torch.testing.assert_close(got[0, :2], torch.tensor([-0.1, -0.2]))
    assert torch.all(got[0, 2:] == PAD_FILL)
    torch.testing.assert_close(got[1, :1], torch.tensor([-0.3]))
    assert torch.all(got[1, 1:] == PAD_FILL)


def test_pad_fill_is_not_a_plausible_logprob():
    """A real logprob is <= 0; the sentinel must be visibly impossible.

    The recipe overwrites padded positions anyway, so this is about a dumped intermediate
    never being misread as data.
    """
    assert PAD_FILL > 0


def test_empty_completion_row_is_all_pad():
    got = build_behavior_logprobs([[]], [[]], num_seqs=1, max_generated_tokens=4)
    assert torch.all(got == PAD_FILL)


def test_overlong_row_truncates_at_the_cap():
    """vLLM may return more tokens than the cap; the recipe truncates, so must we."""
    wire = [[-0.1, -0.2, -0.3, -0.4]]
    comps = [[1, 2, 3, 4]]
    got = build_behavior_logprobs(wire, comps, num_seqs=1, max_generated_tokens=2)
    assert got.shape == (1, 2)
    torch.testing.assert_close(got[0], torch.tensor([-0.1, -0.2]))


# ------------------------------------------------------------------ fallback

def test_none_row_needs_fallback():
    assert rows_needing_fallback([None, [-0.1]], [[1], [2]], 4) == [0]


def test_length_mismatch_needs_fallback():
    """A misaligned pi_old corrupts every ratio in the row and is invisible downstream."""
    assert rows_needing_fallback([[-0.1, -0.2]], [[1, 2, 3]], 8) == [0]


def test_nonfinite_needs_fallback():
    assert rows_needing_fallback([[-0.1, float("-inf")]], [[1, 2]], 8) == [0]
    assert rows_needing_fallback([[-0.1, float("nan")]], [[1, 2]], 8) == [0]


def test_clean_rows_need_no_fallback():
    assert rows_needing_fallback([[-0.1, -0.2], [-0.3]], [[1, 2], [3]], 8) == []


def test_builder_refuses_a_bad_row_rather_than_patching_it():
    """Refusal is the point: a mixed batch is a confound, not a degraded-but-usable input."""
    with pytest.raises(ValueError, match="WHOLE batch"):
        build_behavior_logprobs([None, [-0.1]], [[1], [2]], num_seqs=2,
                                max_generated_tokens=4)


def test_builder_refuses_a_short_batch():
    with pytest.raises(ValueError, match="expected 4 completions"):
        build_behavior_logprobs([[-0.1]], [[1]], num_seqs=4, max_generated_tokens=4)


def test_fallback_check_uses_the_recipe_truncation():
    """A row longer than the cap is fine as long as tokens and logprobs agree.

    Flagging it would send an entire healthy batch down the slow path whenever any
    sequence ran past max_generated_tokens, which is common.
    """
    assert rows_needing_fallback([[-0.1] * 6], [[1] * 6], 3) == []


# ------------------------------------------------------------------ plumbing

def test_client_supports_return_logprobs():
    """The adapter is useless if the transport cannot ask for the values."""
    import inspect

    from torchtune.dev.rl.vllm_client import VLLMClient

    sig = inspect.signature(VLLMClient.generate_from_embeds)
    assert "return_logprobs" in sig.parameters


def test_engine_sets_processed_mode_at_every_llm_site():
    """A single un-patched engine site would silently serve raw logprobs.

    Counted dynamically rather than pinned to a number so a NEW engine site fails here
    instead of shipping the wrong mode.
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[4]
        / "torchtune" / "dev" / "rl" / "vllm_backend.py"
    ).read_text(errors="replace")
    assert "_logprobs_engine_kwargs" in src
    assert src.count("_logprobs_engine_kwargs") >= 2, (
        "the helper exists but is applied at no engine site"
    )
    assert PROCESSED_LOGPROBS_MODE in src

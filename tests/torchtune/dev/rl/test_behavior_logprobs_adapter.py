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


# ------------------------------------------------- site-3 hazard: the broadcast
#
# Only the shard leader issues the vLLM HTTP, so only the leader can build pi_old.
# If it is not broadcast, followers see None and run the policy forward while the leader
# skips it -- divergent collective participation, i.e. a HANG, not an error message.
# These tests simulate a multi-rank broadcast on CPU so that hazard cannot reach HW.


def _fake_broadcast(ranks_payloads, leader_idx):
    """Simulate an in-place broadcast: every rank ends up with the leader's buffer."""
    src = ranks_payloads[leader_idx].clone()
    return [src.clone() for _ in ranks_payloads]


def _run_ranks(leader_tensor, world, num_seqs, ctok, leader_idx=0):
    """Drive broadcast_behavior_logprobs on `world` simulated ranks; return their results.

    A real collective requires every rank to enter before any leaves, so this runs two
    passes: first capture what each rank puts on the wire, then resolve the broadcast and
    let each rank parse the delivered buffer.
    """
    from torchtune.dev.rl.behavior_logprobs import broadcast_behavior_logprobs

    bufs = [(r == leader_idx, leader_tensor if r == leader_idx else None)
            for r in range(world)]

    def _call(is_leader, tensor, fn):
        return broadcast_behavior_logprobs(
            tensor,
            num_seqs=num_seqs,
            max_generated_tokens=ctok,
            is_leader=is_leader,
            broadcast_fn=fn,
        )

    # Pass 1 -- capture each rank's outgoing payload (identity "broadcast").
    sent = []
    for is_leader, t in bufs:
        captured = []
        _call(is_leader, t, lambda p: (captured.append(p), p)[1])
        sent.append(captured[0].clone())

    # Pass 2 -- deliver the leader's buffer to everyone, then parse on each rank.
    delivered = _fake_broadcast(sent, leader_idx)
    return [
        _call(is_leader, t, lambda _p, _r=r: delivered[_r])
        for r, (is_leader, t) in enumerate(bufs)
    ]


def test_followers_receive_the_leaders_logprobs():
    """The site-3 hazard: a follower left holding None takes a branch the leader skips."""
    torch.manual_seed(2)
    num_seqs, ctok, world = 6, 5, 4
    leader = torch.randn(num_seqs, ctok) - 2.0  # plausible logprobs, all < 0

    results = _run_ranks(leader, world, num_seqs, ctok)

    assert len(results) == world
    for r, got in enumerate(results):
        assert got is not None, (
            f"rank {r} got None while the leader has values -- it would run the policy "
            f"forward while the leader skips it: divergent collectives, i.e. a hang"
        )
        torch.testing.assert_close(got, leader)


def test_fallback_decision_is_unanimous():
    """If the leader falls back, EVERY rank must fall back -- not just the leader."""
    num_seqs, ctok, world = 6, 5, 4
    results = _run_ranks(None, world, num_seqs, ctok)
    assert all(g is None for g in results), (
        "a rank kept vLLM logprobs while the leader fell back to the policy forward"
    )


def test_a_dropped_broadcast_is_caught_by_these_tests():
    """Negative control: the no-broadcast implementation must FAIL the test above.

    Without this, `test_followers_receive_the_leaders_logprobs` could pass against a
    harness that never actually moved data between ranks.
    """
    num_seqs, ctok, world = 6, 5, 4
    leader = torch.randn(num_seqs, ctok) - 2.0
    # The bug: each rank keeps its OWN buffer (no data movement).
    from torchtune.dev.rl.behavior_logprobs import broadcast_behavior_logprobs

    got = [
        broadcast_behavior_logprobs(
            leader if r == 0 else None,
            num_seqs=num_seqs,
            max_generated_tokens=ctok,
            is_leader=(r == 0),
            broadcast_fn=lambda p: p,  # no-op "broadcast"
        )
        for r in range(world)
    ]
    assert got[0] is not None
    assert all(g is None for g in got[1:]), (
        "a no-op broadcast still delivered values to followers -- the passing test "
        "above proves nothing about the broadcast"
    )


def test_leader_shape_mismatch_refused():
    """A follower pre-allocates the declared shape; a mismatch is a silent misparse."""
    from torchtune.dev.rl.behavior_logprobs import broadcast_behavior_logprobs

    with pytest.raises(ValueError, match="silent"):
        broadcast_behavior_logprobs(
            torch.zeros(3, 5),
            num_seqs=6,
            max_generated_tokens=5,
            is_leader=True,
            broadcast_fn=lambda p: p,
        )


def test_have_flag_cannot_collide_with_a_real_logprob():
    """Row 0 is a sentinel row, never data -- payload is num_seqs+1 rows wide."""
    from torchtune.dev.rl.behavior_logprobs import broadcast_behavior_logprobs

    seen = {}

    def cap(p):
        seen["shape"] = tuple(p.shape)
        return p

    broadcast_behavior_logprobs(
        torch.zeros(6, 5),
        num_seqs=6,
        max_generated_tokens=5,
        is_leader=True,
        broadcast_fn=cap,
    )
    assert seen["shape"] == (7, 5), (
        "the have-flag must occupy its own row; packing it into a data row would make "
        "a real logprob of exactly 1.0 indistinguishable from the flag"
    )


# ------------------------------------------------- width: cap vs actual (HW crash)

def test_fit_narrows_cap_width_to_actual_response_width():
    """The builder pads to the CAP; the trainer's tensors are at the ACTUAL width.

    HW provenance: job 8833972. `build_behavior_logprobs` returns
    [num_seqs, max_generated_tokens]; the recipe's `responses` is the batch's longest
    generation, which equals the cap only when some row ran to the limit. Steps 0 and 1
    both had such a row (3072) and ran clean; step 2's longest was 1743 and the run died
    in `logprobs.masked_fill_(response_padding_masks, 1.0)`:

        RuntimeError: The expanded size of the tensor (3072) must match the existing
        size (1743) at non-singleton dimension 1.
    """
    from torchtune.dev.rl.behavior_logprobs import fit_behavior_logprobs_width

    blp = build_behavior_logprobs([[-0.1, -0.2]], [[7, 8]], 1, 3072)
    assert blp.shape == (1, 3072)
    fitted = fit_behavior_logprobs_width(blp, 1743)
    assert fitted.shape == (1, 1743)
    # the real values survive the narrowing -- this must not silently zero pi_old
    assert math.isclose(fitted[0, 0].item(), -0.1, rel_tol=1e-6)
    assert math.isclose(fitted[0, 1].item(), -0.2, rel_tol=1e-6)


def test_fit_is_identity_when_widths_already_agree():
    """Steps 0-1 of the crashing job took this branch; it must stay a no-op."""
    from torchtune.dev.rl.behavior_logprobs import fit_behavior_logprobs_width

    blp = build_behavior_logprobs([[-0.1, -0.2]], [[7, 8]], 1, 4)
    assert fit_behavior_logprobs_width(blp, 4) is blp


def test_fit_refuses_to_widen_past_the_cap():
    """Padding up would fabricate pi_old for positions holding real tokens.

    A caller asking for more than the cap has a broken max_generated_tokens; quietly
    zero-filling would push an unreadable inconsistency into the loss instead of
    failing where the disagreement actually is.
    """
    from torchtune.dev.rl.behavior_logprobs import fit_behavior_logprobs_width

    blp = build_behavior_logprobs([[-0.1, -0.2]], [[7, 8]], 1, 4)
    with pytest.raises(ValueError, match="exceeds the behavior-logprob width"):
        fit_behavior_logprobs_width(blp, 5)


def test_narrowed_tensor_survives_the_masked_fill_that_crashed():
    """Reproduce the exact failing operation, pre- and post-fix.

    The negative half matters more than the positive half: without it this test would
    pass against a no-op implementation of the fitter.
    """
    from torchtune.dev.rl.behavior_logprobs import fit_behavior_logprobs_width

    cap, actual, n = 3072, 1743, 4
    blp = build_behavior_logprobs([[-0.1] * 10] * n, [[7] * 10] * n, n, cap)
    pad = torch.zeros(n, actual, dtype=torch.bool)
    pad[:, 900:] = True

    with pytest.raises(RuntimeError):  # what production actually hit
        blp.clone().masked_fill_(pad, 1.0)

    fit_behavior_logprobs_width(blp, actual).clone().masked_fill_(pad, 1.0)


def test_recipe_fits_width_on_the_substitution_path():
    """AST guard: the substitution branch must not assign _behavior_logprobs raw.

    The bug survived the audit because TORCHTUNE_BLP_AUDIT forces the recompute branch
    -- disabling the shortcut in order to measure it also disabled its plumbing. So the
    audited steps could never reach this line, and no CPU test covered it either. Pin
    the call site, not just the helper.
    """
    import ast
    import pathlib

    src = pathlib.Path("recipes/dev/grpo_bioreason_distributed_xpu.py").read_text()
    tree = ast.parse(src)

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "logprobs" for t in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "_behavior_logprobs"
    ]
    assert not assigns, (
        "the substitution path assigns _behavior_logprobs to logprobs without fitting "
        "its width; that is the job-8833972 crash. Wrap it in "
        "fit_behavior_logprobs_width(..., responses.shape[1])"
    )
    assert "fit_behavior_logprobs_width" in src, (
        "recipe no longer fits the behavior-logprob width anywhere"
    )

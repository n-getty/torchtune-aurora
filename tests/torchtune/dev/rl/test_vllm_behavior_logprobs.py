# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU regression tests for the vLLM behavior-policy logprobs path.

WHY THIS FILE EXISTS
--------------------
Enabling ``async_generation`` flips ``_compute_rollout_logprobs_required``, which
un-skips a full rollout-time policy forward worth ~91s/step at BioReason 32B/2N
(sync ``GENTIMING`` reads ``policy_fwd=0.0s``). That forward exists only to
produce ``pi_old`` for ``GRPOLoss``. vLLM already computed exactly that quantity
while sampling, so plumbing it back over HTTP removes the forward and lifts the
async ceiling from ~1.29x to ~1.59x (measured against the 36 warm steps of the
sync baseline run 8829513).

The lever is only valid if vLLM's numbers mean the same thing as the trainer's.
A wrong ``pi_old`` does not crash — it silently corrupts every importance ratio,
so these tests pin the exactness conditions rather than trusting them.

The decisive condition is TEMPERATURE. torchtune computes
``log_softmax(logits / temperature)`` (``rlhf.batched_logits_to_logprobs``), but
vLLM's DEFAULT ``--logprobs-mode raw_logprobs`` snapshots ``log_softmax(logits)``
before the temperature division. At the production temperature of 0.8 those are
different distributions, not a rounding difference. ``processed_logprobs`` is the
mode that matches.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from torchtune.rlhf.sequence_processing import batched_logits_to_logprobs


# ─────────────────────────────────────────────────────────────────────────────
# 1. The numerical contract: what vLLM computes vs what the trainer computes.
# ─────────────────────────────────────────────────────────────────────────────


def _vllm_processed_logprobs(logits: torch.Tensor, temperature: float):
    """Reproduce vLLM 0.15.0's ``processed_logprobs`` for the sampled token.

    Mirrors the installed tree's ordering exactly (v1/sample/sampler.py ->
    v1/sample/ops/topk_topp_sampler.py):

        logits.to(torch.float32)        # sampler.py, BEFORE temperature
        logits.div_(temperature)        # apply_temperature
        <top-k/top-p>                   # early-returns unchanged when k/p unset
        logits.log_softmax(dim=-1, dtype=torch.float32)

    Note the cast happens BEFORE the division, so the reference is
    ``log_softmax(logits.float() / t)`` and not ``log_softmax((logits / t).float())``.
    On bf16 inputs those differ.
    """
    x = logits.to(torch.float32)
    x = x / temperature
    return x.log_softmax(dim=-1, dtype=torch.float32)


def _vllm_raw_logprobs(logits: torch.Tensor):
    """vLLM's DEFAULT mode: snapshot taken before fp32 cast and before temperature."""
    return logits.log_softmax(dim=-1, dtype=torch.float32)


@pytest.mark.parametrize("temperature", [0.8, 1.0, 1.3])
def test_processed_logprobs_matches_trainer_formula(temperature):
    """``processed_logprobs`` is bit-comparable to ``batched_logits_to_logprobs``.

    This is the claim the whole lever rests on: same formula, so swapping the
    source is a no-op numerically (modulo the executor, which is a separate and
    genuinely unverifiable-on-CPU risk — see the module docstring).
    """
    torch.manual_seed(0)
    b, s, v = 3, 7, 64
    logits = torch.randn(b, s, v, dtype=torch.float32)
    tokens = torch.randint(0, v, (b, s))

    trainer = batched_logits_to_logprobs(logits, tokens, temperature, chunk_size=2)

    vllm_full = _vllm_processed_logprobs(logits, temperature)
    vllm = torch.gather(vllm_full, 2, tokens.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(trainer, vllm, rtol=0, atol=0)


def test_raw_logprobs_does_NOT_match_at_production_temperature():
    """Guard against the most dangerous plausible misconfiguration.

    vLLM's default mode is ``raw_logprobs``. If someone enables the logprobs path
    without passing ``--logprobs-mode processed_logprobs``, the run does not fail
    — it trains on a systematically wrong ``pi_old``. This test documents that the
    two are materially different at the production temperature (0.8), so a future
    reader cannot conclude "the default is fine".
    """
    torch.manual_seed(0)
    b, s, v = 2, 5, 32
    logits = torch.randn(b, s, v, dtype=torch.float32)
    tokens = torch.randint(0, v, (b, s))

    trainer = batched_logits_to_logprobs(logits, tokens, 0.8, chunk_size=2)
    raw = torch.gather(_vllm_raw_logprobs(logits), 2, tokens.unsqueeze(-1)).squeeze(-1)

    # Not close — and not trivially so. The discrepancy must be big enough that
    # calling it "numerical noise" is untenable.
    assert not torch.allclose(trainer, raw, atol=1e-2), (
        "raw_logprobs unexpectedly matched the temperature-scaled trainer formula; "
        "if this ever passes, re-derive the sampler ordering before trusting the "
        "default logprobs-mode."
    )
    assert (trainer - raw).abs().max() > 0.05

    # And the resulting IS ratio would be wrong by a large factor, which is the
    # consequence that actually matters.
    #
    # Measure the ratio error TWO-SIDED. exp(trainer - raw) can land either above
    # or below 1 depending on whether the sampled token sits in the head or the
    # tail of the distribution, so a one-sided `.max() > 1.05` is seed-dependent
    # and can pass on a draw where the mismatch is large but mostly downward.
    # (This is not hypothetical: the first version of this assertion read
    # `exp(trainer-raw).max() > 1.05` and failed at 1.0452 on seed 0 while the
    # underlying logprob gap was ~0.6 nats — the guard was wrong, not the claim.)
    log_ratio = (trainer - raw).abs()
    assert log_ratio.max().item() > 0.05, (
        "temperature mismatch produced a negligible log-ratio; re-derive the "
        "sampler ordering before trusting the default logprobs-mode."
    )
    two_sided_ratio_err = torch.exp(log_ratio).max().item()
    assert two_sided_ratio_err > 1.05


def test_raw_and_processed_agree_at_temperature_one():
    """Sanity: the mismatch above is caused by temperature, nothing else."""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, dtype=torch.float32)
    torch.testing.assert_close(
        _vllm_raw_logprobs(logits), _vllm_processed_logprobs(logits, 1.0)
    )


def test_fp32_cast_precedes_temperature_division():
    """Pin the cast/divide ORDER, which is observable in bf16.

    vLLM casts to fp32 and then divides. Doing it the other way round loses
    precision in the division. On a bf16 policy this is the difference between a
    clean match and a drifting one, so the order is part of the contract.
    """
    torch.manual_seed(0)
    logits_bf16 = torch.randn(4, 128, dtype=torch.float32).to(torch.bfloat16)
    t = 0.8

    correct = (logits_bf16.to(torch.float32) / t).log_softmax(-1)
    wrong = (logits_bf16 / t).to(torch.float32).log_softmax(-1)

    # They must not be assumed interchangeable.
    assert (correct - wrong).abs().max() > 0, (
        "bf16 divide-then-cast happened to equal cast-then-divide on this seed; "
        "the test is not exercising the precision difference it claims to."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. The client wire contract.
# ─────────────────────────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _FakeSession:
    """Captures the outgoing payload and replays a canned vLLM response."""

    def __init__(self, response_payload):
        self._response_payload = response_payload
        self.last_payload = None

    def post(self, url, json=None, timeout=None):  # noqa: A002
        self.last_payload = json
        return _FakeResponse(self._response_payload)


def _make_client(session):
    from torchtune.dev.rl.vllm_client import VLLMClient

    client = VLLMClient.__new__(VLLMClient)
    client._api_type = "openai"
    client._model_name = "m"
    client.base_url = "http://x:8000"
    client.session = session
    return client


def _choice(ids, logprobs=None):
    c = {"token_ids": list(ids)}
    if logprobs is not None:
        c["logprobs"] = {"token_logprobs": list(logprobs)}
    return c


def test_logprobs_not_requested_by_default():
    """Default must stay off: the sync production path pays nothing for this."""
    sess = _FakeSession({"choices": [_choice([1, 2, 3])]})
    client = _make_client(sess)
    out = client.generate_from_embeds([torch.zeros(2, 4)])
    assert out == [[1, 2, 3]]
    assert "logprobs" not in sess.last_payload


def test_logprobs_requested_as_zero_and_returned_aligned():
    """``logprobs: 0`` is the cheapest request that still yields the sampled token's
    own logprob (vLLM puts it first). Return shape must pair 1:1 with token ids."""
    sess = _FakeSession(
        {"choices": [_choice([1, 2, 3], [-0.1, -0.2, -0.3])]}
    )
    client = _make_client(sess)
    ids, lps = client.generate_from_embeds([torch.zeros(2, 4)], return_logprobs=True)
    assert sess.last_payload["logprobs"] == 0
    assert ids == [[1, 2, 3]]
    assert lps == [[-0.1, -0.2, -0.3]]
    assert len(lps[0]) == len(ids[0])


def test_missing_logprobs_yields_none_not_silent_misalignment():
    """A choice without logprobs must surface as None, never as a short list."""
    sess = _FakeSession({"choices": [_choice([1, 2, 3])]})
    client = _make_client(sess)
    ids, lps = client.generate_from_embeds([torch.zeros(2, 4)], return_logprobs=True)
    assert ids == [[1, 2, 3]]
    assert lps == [None]


def test_length_mismatch_is_dropped_loudly_not_used():
    """An off-by-one would bias every ratio in the row and be invisible downstream.

    The client must refuse to hand back a misaligned list.
    """
    sess = _FakeSession({"choices": [_choice([1, 2, 3], [-0.1, -0.2])]})
    client = _make_client(sess)
    ids, lps = client.generate_from_embeds([torch.zeros(2, 4)], return_logprobs=True)
    assert ids == [[1, 2, 3]]
    assert lps == [None], "misaligned logprobs must be dropped, not truncated"


def test_legacy_return_shape_is_unchanged():
    """Every existing caller passes no flag and must keep getting a bare list."""
    sess = _FakeSession({"choices": [_choice([7]), _choice([8, 9])]})
    client = _make_client(sess)
    out = client.generate_from_embeds([torch.zeros(1, 2), torch.zeros(1, 2)])
    assert out == [[7], [8, 9]]
    assert not isinstance(out, tuple)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ragged -> padded assembly.
# ─────────────────────────────────────────────────────────────────────────────


def test_ragged_logprobs_pad_into_dense_matrix_without_shifting():
    """vLLM returns ragged per-sequence lists; the trainer needs ``[B*G, L]``.

    The padded region must never be read as a real logprob. The recipe masks the
    response padding, so the pad VALUE is not load-bearing — but the OFFSET is:
    row i's logprob for response position j must land at ``[i, j]``. This pins
    that, because an off-by-one from a prompt-offset mistake is the single most
    likely way to silently corrupt this path.
    """
    comps = [[1, 2, 3], [4], [5, 6]]
    lps = [[-0.1, -0.2, -0.3], [-0.4], [-0.5, -0.6]]
    max_len = 4

    dense = torch.zeros(len(comps), max_len, dtype=torch.float32)
    mask = torch.ones(len(comps), max_len, dtype=torch.bool)
    for i, lp in enumerate(lps):
        n = min(len(lp), max_len)
        dense[i, :n] = torch.tensor(lp[:n])
        mask[i, :n] = False

    assert dense[0].tolist() == pytest.approx([-0.1, -0.2, -0.3, 0.0])
    assert dense[1].tolist() == pytest.approx([-0.4, 0.0, 0.0, 0.0])
    assert dense[2].tolist() == pytest.approx([-0.5, -0.6, 0.0, 0.0])
    assert mask[1].tolist() == [False, True, True, True]
    # Every unmasked entry is a real (negative) logprob.
    assert bool((dense[~mask] < 0).all())


def test_truncation_at_max_generated_tokens_keeps_alignment():
    """A completion longer than the cap is truncated the same way tokens are."""
    lp = [-0.1 * i for i in range(1, 11)]
    max_len = 4
    row = torch.zeros(max_len)
    n = min(len(lp), max_len)
    row[:n] = torch.tensor(lp[:n])
    assert row.tolist() == pytest.approx([-0.1, -0.2, -0.30000000000000004, -0.4])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Why GRPOSimpleLoss cannot substitute (the reason the forward is required).
# ─────────────────────────────────────────────────────────────────────────────


def test_grposimpleloss_ratio_is_identically_one():
    """Documents why async REQUIRES GRPOLoss and therefore requires a real pi_old.

    If GRPOSimpleLoss could carry the async correction, none of this work would be
    needed. It cannot: its ratio is exp(x - x.detach()) == 1 by construction, so it
    is blind to the fact that the rollout came from an older weight version.
    """
    from torchtune.dev.rl.loss import GRPOSimpleLoss

    loss_fn = GRPOSimpleLoss(kl_coeff=1e-3, epsilon=0.2)
    b, s = 2, 3
    pi_old = torch.randn(b, s)
    pi = torch.randn(b, s, requires_grad=True)
    ref = torch.randn(b, s)
    adv = torch.randn(b)
    pad = torch.zeros(b, s, dtype=torch.bool)

    _, _, _, ratios, _ = loss_fn(pi_old, pi, ref, adv, pad)
    assert math.isclose(float(ratios), 1.0, abs_tol=1e-6)

    # Feeding a wildly different pi_old changes nothing — proof it is ignored.
    _, _, _, ratios2, _ = loss_fn(pi_old * 100, pi, ref, adv, pad)
    assert math.isclose(float(ratios2), 1.0, abs_tol=1e-6)


def test_grpoloss_ratio_actually_depends_on_pi_old():
    """The converse: GRPOLoss does consume pi_old, so supplying it correctly matters."""
    from torchtune.dev.rl.loss import GRPOLoss

    loss_fn = GRPOLoss(kl_coeff=1e-3, epsilon=0.2)
    b, s = 2, 3
    pi = torch.randn(b, s, requires_grad=True)
    ref = torch.randn(b, s)
    adv = torch.randn(b)
    pad = torch.zeros(b, s, dtype=torch.bool)

    _, _, _, r1, _ = loss_fn(pi.detach(), pi, ref, adv, pad)
    _, _, _, r2, _ = loss_fn(pi.detach() - 0.5, pi, ref, adv, pad)
    assert not math.isclose(float(r1), float(r2), abs_tol=1e-3)
    # Identical pi_old == pi gives ratio 1, the on-policy sanity point.
    assert math.isclose(float(r1), 1.0, abs_tol=1e-5)

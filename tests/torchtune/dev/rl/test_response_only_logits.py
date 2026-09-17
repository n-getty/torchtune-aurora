# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe equivalence guard for ``TORCHTUNE_RESPONSE_ONLY_LOGITS``.

The response-only path slices the backbone's HIDDEN states to the response span
BEFORE the lm_head, instead of projecting every padded position to
``vocab_size`` and throwing ~78% of the result away in
``gather_response_logits``. At Qwen3-32B scale (V=151936) the discarded slab is
multiple GB of peak memory per microbatch.

This is a pure memory optimization: it must be BIT-IDENTICAL to the default
path. These tests pin that on a tiny real HF Qwen3 backbone (a synthetic
``nn.Linear`` head would not exercise the PEFT/HF plumbing the real model uses),
covering BOTH prompt-length forms that ``gather_response_logits`` accepts:

  * scalar ``int`` — every row shares one prompt length (uniform padding);
  * ``[B]`` tensor — the row-compacted layout produced by
    ``compact_prompt_completion_batch`` (TORCHTUNE_COMPACT_PROMPT_CHUNKS=1,
    ON in production).

No XPU, no distributed init — runs on a login node.
"""
from __future__ import annotations

import pytest
import torch

from torchtune.dev.bioreason.model import BioReasonModel
from torchtune.dev.rl.generation import (
    finish_response_logits,
    gather_response_logits,
    gather_response_span,
    response_only_logits_enabled,
    response_only_logits_kwargs,
)

transformers = pytest.importorskip("transformers")

VOCAB_SIZE = 97
HIDDEN_SIZE = 32
BATCH_SIZE = 2
SEQ_LEN = 11


def _tiny_backbone(tie_word_embeddings: bool = False):
    """Build a tiny real HF Qwen3 causal LM (2 layers, vocab 97).

    ``attn_implementation="eager"`` is LOAD-BEARING, not a style choice: torch's
    CPU SDPA backend is non-deterministic on a many-core login node (measured on
    Aurora, 52 threads — two identical no-grad forwards of this very model differ
    by up to 0.29 in the logits). Under ``sdpa`` a bit-exactness assertion would
    fail on ATTENTION noise that has nothing to do with the lm_head slice under
    test, i.e. a false positive that hides the real signal. ``eager`` is
    reproducible to 0.0 across repeated forwards, so any difference these tests
    report is genuinely the response-only projection. The lm_head slice itself
    was separately confirmed bit-exact under BOTH backends.
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=tie_word_embeddings,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).eval()


def _bioreason_model(backbone) -> BioReasonModel:
    """Wrap a backbone in a BioReasonModel without touching the checkpoint loader."""
    model = BioReasonModel.__new__(BioReasonModel)
    torch.nn.Module.__init__(model)
    model.backbone = backbone
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.hidden_size = HIDDEN_SIZE
    model.vocab_size = VOCAB_SIZE
    return model


def _inputs():
    torch.manual_seed(1)
    inputs_embeds = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
    return inputs_embeds, attention_mask, position_ids


def test_backbone_forward_is_deterministic():
    """Guard the ``eager`` pin in :func:`_tiny_backbone`.

    Every bit-exactness assertion below presumes repeated forwards of the same
    model on the same input agree exactly. If this fails, the backbone has
    drifted back onto a non-deterministic attention backend and the other
    failures in this file are noise, not regressions.
    """
    model = _bioreason_model(_tiny_backbone())
    inputs_embeds, attention_mask, position_ids = _inputs()
    with torch.no_grad():
        runs = [
            model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            for _ in range(4)
        ]
    for run in runs[1:]:
        assert torch.equal(runs[0], run), (
            "tiny backbone forward is non-deterministic — bit-exactness assertions "
            "in this file are meaningless; check attn_implementation"
        )


@pytest.mark.parametrize(
    "prompt_lengths, response_length",
    [
        pytest.param(5, 6, id="scalar_prompt_length"),
        pytest.param(torch.tensor([5, 7]), 4, id="rowwise_prompt_lengths"),
        pytest.param(torch.tensor([3, 3]), 4, id="rowwise_uniform_prompt_lengths"),
        pytest.param(torch.tensor([9, 4]), 2, id="rowwise_clamped_tail"),
    ],
)
def test_response_only_logits_are_bit_identical(prompt_lengths, response_length):
    model = _bioreason_model(_tiny_backbone())
    inputs_embeds, attention_mask, position_ids = _inputs()

    with torch.no_grad():
        full = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        expected = gather_response_logits(full, prompt_lengths, response_length)
        actual = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_prompt_lengths=prompt_lengths,
            response_length=response_length,
        )

    assert full.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert actual.shape == expected.shape
    assert actual.shape[1] == response_length
    assert torch.equal(actual, expected), (
        "response-only logits must be BIT-identical to slicing full-width logits; "
        f"max abs diff {(actual - expected).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "prompt_lengths, response_length",
    [
        pytest.param(5, 6, id="scalar_prompt_length"),
        pytest.param(torch.tensor([5, 7]), 4, id="rowwise_prompt_lengths"),
    ],
)
def test_response_only_logprobs_are_bit_identical(prompt_lengths, response_length):
    """End-to-end: the logprobs the GRPO loss consumes must not move."""
    from torchtune import rlhf

    model = _bioreason_model(_tiny_backbone())
    inputs_embeds, attention_mask, position_ids = _inputs()
    torch.manual_seed(2)
    responses = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, response_length))

    with torch.no_grad():
        full = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        expected = rlhf.batched_logits_to_logprobs(
            gather_response_logits(full, prompt_lengths, response_length),
            responses,
            temperature=0.9,
        )
        actual = rlhf.batched_logits_to_logprobs(
            model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                response_prompt_lengths=prompt_lengths,
                response_length=response_length,
            ),
            responses,
            temperature=0.9,
        )

    assert torch.equal(actual, expected)


def test_response_only_logits_match_under_tied_embeddings():
    """A tied lm_head shares storage with embed_tokens — the hook must still slice."""
    model = _bioreason_model(_tiny_backbone(tie_word_embeddings=True))
    inputs_embeds, attention_mask, position_ids = _inputs()
    prompt_lengths, response_length = torch.tensor([4, 6]), 3

    with torch.no_grad():
        full = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        actual = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_prompt_lengths=prompt_lengths,
            response_length=response_length,
        )

    assert torch.equal(
        actual, gather_response_logits(full, prompt_lengths, response_length)
    )


def test_response_only_logits_match_through_peft_lora():
    """Production runs PEFT-wrap the backbone; the slice must survive the wrapper."""
    peft = pytest.importorskip("peft")

    backbone = peft.get_peft_model(
        _tiny_backbone(),
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model = _bioreason_model(backbone)
    inputs_embeds, attention_mask, position_ids = _inputs()
    prompt_lengths, response_length = torch.tensor([5, 7]), 4

    with torch.no_grad():
        full = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        actual = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_prompt_lengths=prompt_lengths,
            response_length=response_length,
        )

    assert torch.equal(
        actual, gather_response_logits(full, prompt_lengths, response_length)
    )


def test_response_only_logits_gradients_match():
    """The training forward is under autograd — grads must match, not just values."""
    prompt_lengths, response_length = torch.tensor([5, 7]), 4
    inputs_embeds, attention_mask, position_ids = _inputs()

    grads = []
    for response_only in (False, True):
        model = _bioreason_model(_tiny_backbone())
        embeds = inputs_embeds.clone().requires_grad_(True)
        extra = (
            {
                "response_prompt_lengths": prompt_lengths,
                "response_length": response_length,
            }
            if response_only
            else {}
        )
        logits = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **extra,
        )
        if not response_only:
            logits = gather_response_logits(logits, prompt_lengths, response_length)
        logits.square().sum().backward()
        grads.append(embeds.grad.detach().clone())

    torch.testing.assert_close(grads[0], grads[1], rtol=0, atol=0)


def test_hook_is_removed_after_forward():
    """The pre-hook must not leak onto the lm_head and corrupt later forwards."""
    model = _bioreason_model(_tiny_backbone())
    inputs_embeds, attention_mask, position_ids = _inputs()
    lm_head = model.backbone.get_output_embeddings()
    n_hooks_before = len(lm_head._forward_pre_hooks)

    with torch.no_grad():
        baseline = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_prompt_lengths=4,
            response_length=3,
        )
        after = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    assert len(lm_head._forward_pre_hooks) == n_hooks_before
    assert torch.equal(baseline, after)


def test_hook_is_removed_when_forward_raises():
    model = _bioreason_model(_tiny_backbone())
    lm_head = model.backbone.get_output_embeddings()
    n_hooks_before = len(lm_head._forward_pre_hooks)

    with pytest.raises(Exception):
        model(
            inputs_embeds=torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE + 1),
            response_prompt_lengths=4,
            response_length=3,
        )

    assert len(lm_head._forward_pre_hooks) == n_hooks_before


def test_partial_response_args_raise():
    model = _bioreason_model(_tiny_backbone())
    inputs_embeds, _, _ = _inputs()
    with pytest.raises(ValueError, match="must be supplied"):
        model(inputs_embeds=inputs_embeds, response_prompt_lengths=4)
    with pytest.raises(ValueError, match="must be supplied"):
        model(inputs_embeds=inputs_embeds, response_length=3)


def test_scalar_gather_matches_truncate_sequence_for_logprobs():
    """The single-backward path used ``truncate_sequence_for_logprobs``.

    That helper is ``logits[:, ctx-1:-1]``; the wiring replaced it with
    ``gather_response_logits(logits, ctx, S - ctx)`` == ``logits[:, ctx-1:S-1]``.
    Identical for a full-width input — pinned here so the substitution can't
    silently shift the response window by a position.
    """
    from torchtune import rlhf

    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    for context_length in (1, 4, 5, SEQ_LEN - 1):
        assert torch.equal(
            gather_response_logits(logits, context_length, SEQ_LEN - context_length),
            rlhf.truncate_sequence_for_logprobs(logits, context_length),
        )


def test_gather_response_span_matches_gather_response_logits():
    """The hidden-state slice and the logit slice must never diverge."""
    states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    for prompt_lengths in (5, torch.tensor([5, 7])):
        assert torch.equal(
            gather_response_span(states, prompt_lengths, 4),
            gather_response_logits(states, prompt_lengths, 4),
        )


# ── Gate / dispatch helpers ─────────────────────────────────────────────────


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", raising=False)
    assert response_only_logits_enabled() is False
    model = _bioreason_model(_tiny_backbone())
    assert response_only_logits_kwargs(model, 4, 3) == {}


def test_kwargs_empty_for_unsupported_model(monkeypatch):
    """A model with no lm_head (load_backbone=False) must fall back silently."""
    monkeypatch.setenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", "1")
    model = _bioreason_model(None)
    assert model.supports_response_only_logits() is False
    assert response_only_logits_kwargs(model, 4, 3) == {}
    assert response_only_logits_kwargs(torch.nn.Linear(2, 2), 4, 3) == {}


def test_kwargs_populated_when_enabled_and_supported(monkeypatch):
    monkeypatch.setenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", "1")
    model = _bioreason_model(_tiny_backbone())
    kwargs = response_only_logits_kwargs(model, 4, 3)
    assert kwargs == {"response_prompt_lengths": 4, "response_length": 3}


def test_kwargs_resolve_through_wrapper(monkeypatch):
    """FSDP1/DDP proxy unknown attributes to .module — the probe must see through."""
    monkeypatch.setenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", "1")

    class _Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._modules["module"], name)

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    wrapped = _Wrapper(_bioreason_model(_tiny_backbone()))
    assert response_only_logits_kwargs(wrapped, 4, 3) == {
        "response_prompt_lengths": 4,
        "response_length": 3,
    }


def test_finish_response_logits_dispatch():
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    prompt_lengths, response_length = torch.tensor([5, 7]), 4

    sliced = finish_response_logits(logits, {}, prompt_lengths, response_length)
    assert torch.equal(
        sliced, gather_response_logits(logits, prompt_lengths, response_length)
    )

    already = torch.randn(BATCH_SIZE, response_length, VOCAB_SIZE)
    passthrough = finish_response_logits(
        already,
        {"response_prompt_lengths": prompt_lengths, "response_length": response_length},
        prompt_lengths,
        response_length,
    )
    assert passthrough is already


def test_end_to_end_helper_pair_matches_baseline(monkeypatch):
    """The exact call shape the recipe uses, with the flag ON, matches the default."""
    model = _bioreason_model(_tiny_backbone())
    inputs_embeds, attention_mask, position_ids = _inputs()
    prompt_lengths, response_length = torch.tensor([5, 7]), 4

    def _run():
        kwargs = response_only_logits_kwargs(model, prompt_lengths, response_length)
        with torch.no_grad():
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        return finish_response_logits(out, kwargs, prompt_lengths, response_length)

    monkeypatch.delenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", raising=False)
    baseline = _run()
    monkeypatch.setenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", "1")
    optimized = _run()

    assert baseline.shape == (BATCH_SIZE, response_length, VOCAB_SIZE)
    assert torch.equal(baseline, optimized)


# --- engagement marker (blind-vs-fail) -------------------------------------
#
# The equivalence tests above prove the path is CORRECT when it runs. They say
# nothing about whether it runs. On hardware the gate can silently no-op --
# _supports_response_only_logits() returns False for any model without a
# backbone lm_head -- and a silent no-op reads in the log exactly like "the flag
# bought no memory", which would retire a working optimization for the wrong
# reason. Same class as feedback_gate_must_distinguish_blind_from_fail.


def _reset_marker_dedupe():
    """Clear the once-per-state dedupe so each test observes its own emission."""
    from torchtune.dev.rl import generation as _gen

    _gen._RESPONSE_ONLY_LOGGED.clear()


def test_marker_says_engaged_when_the_path_is_live(monkeypatch, caplog):
    """Flag ON + model supports it -> the log must prove the path is live."""
    import logging

    _reset_marker_dedupe()
    model = _bioreason_model(_tiny_backbone())
    monkeypatch.setenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", "1")
    with caplog.at_level(logging.INFO):
        kwargs = response_only_logits_kwargs(model, torch.tensor([5, 7]), 4)
    assert kwargs, "expected the response-only kwargs on a supporting model"
    assert "response_only_logits = engaged" in caplog.text


def test_marker_distinguishes_requested_but_skipped_from_off(monkeypatch, caplog):
    """Flag ON + unsupported model must NOT look like the flag being OFF.

    This is the whole point of the marker: an inert gate and an absent gate
    produce identical forwards, identical memory, and identical step times. Only
    the log can tell them apart.
    """
    import logging

    _reset_marker_dedupe()
    unsupported = torch.nn.Linear(4, 4)  # no supports_response_only_logits probe
    monkeypatch.setenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", "1")
    with caplog.at_level(logging.INFO):
        kwargs = response_only_logits_kwargs(unsupported, 5, 4)
    assert kwargs == {}
    assert "response_only_logits = requested-but-skipped" in caplog.text

    # ...and with the flag OFF the skipped marker must be absent, or the two
    # states would be indistinguishable in the direction that matters.
    _reset_marker_dedupe()
    caplog.clear()
    monkeypatch.delenv("TORCHTUNE_RESPONSE_ONLY_LOGITS", raising=False)
    with caplog.at_level(logging.INFO):
        response_only_logits_kwargs(unsupported, 5, 4)
    assert "requested-but-skipped" not in caplog.text

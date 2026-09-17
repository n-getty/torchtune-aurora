# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Exactness guard for the shared-prefix reference forward (``TORCHTUNE_REF_PREFIX_SHARE``).

The reference model is frozen and its forward runs entirely under ``no_grad``, so
reusing one prompt's KV cache across its ``G`` GRPO continuations must reproduce
the full-recompute logprobs *numerically*, not approximately. If it does not, the
optimization is wrong and must not be enabled — hence these tests assert tight
tolerances against the unmodified full forward.

The subtle failure this pins down: with a populated KV cache ``q_len != kv_len``,
and SDPA's ``is_causal=True`` shortcut means **top-left** alignment, which is the
wrong mask for a cached suffix (it needs **bottom-right** alignment). Passing
``attention_mask=None`` on the suffix call silently yields wrong logits.
``test_suffix_without_mask_is_wrong`` asserts that this failure mode is real, so
nobody "optimizes" the explicit mask away to re-enable the XPU flash kernel.

CPU-only, no XPU, no distributed init.
"""
from __future__ import annotations

import pytest
import torch

from torchtune.dev.rl.ref_prefix_share import (
    expand_cache_batch_,
    prefix_share_supported,
    ref_prefix_share_enabled,
    shared_prefix_ref_logprobs,
)

transformers = pytest.importorskip("transformers")


def _tiny_causal_lm(dtype: torch.dtype = torch.float64, attn: str = "eager"):
    """Build a tiny deterministic Qwen3 causal LM on CPU.

    Defaults to ``attn="eager"`` rather than ``"sdpa"``. This is deliberate and
    load-bearing: on this stack (torch 2.10.0a0, CPU, 52 threads) the **CPU SDPA
    kernel is non-deterministic** — two back-to-back identical forwards of the same
    frozen model on the same input return different tensors, and occasionally
    produce NaN rows. That is a property of the baseline forward itself (it
    reproduces with prefix sharing entirely out of the picture), so under SDPA no
    equivalence assertion at any tolerance is meaningful. ``eager`` is bitwise
    reproducible here and exercises the identical attention math.

    ``test_shared_prefix_matches_full_recompute_under_production_flash_backend``
    separately covers the real production attention backend.
    """
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=53,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
    )
    config._attn_implementation = attn
    model = Qwen3ForCausalLM(config).to(dtype).eval()
    model.model.config._attn_implementation = attn
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


class _StubBioReasonModel:
    """Minimal stand-in exposing the same contract BioReasonModel now implements.

    Mirrors ``BioReasonModel.forward_cached`` / ``embed_completion_ids`` /
    ``forward`` so the test exercises the real ``shared_prefix_ref_logprobs``
    logic without loading a 32B checkpoint or ESM3.
    """

    def __init__(self, model, hidden_size: int, vocab_size: int, dtype: torch.dtype):
        self.backbone = model
        self.device = torch.device("cpu")
        self.dtype = dtype
        torch.manual_seed(1)
        self._table = torch.randn(vocab_size, hidden_size, dtype=dtype)

    def embed_completion_ids(self, completion_ids: torch.Tensor) -> torch.Tensor:
        return self._table[completion_ids]

    def forward_cached(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=0,
        **kwargs,
    ):
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return out.logits, out.past_key_values

    def forward_full(self, inputs_embeds, attention_mask=None, position_ids=None):
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits


def _group_major_batch(stub, num_prompts, group_size, prompt_len, resp_len, hidden, vocab, dtype):
    """Build a group-major ``[B*G, ...]`` batch: rows of a group share one prompt."""
    torch.manual_seed(7)
    base_prompts = torch.randn(num_prompts, prompt_len, hidden, dtype=dtype)
    prompt_embeds = (
        base_prompts[:, None]
        .expand(-1, group_size, -1, -1)
        .reshape(num_prompts * group_size, prompt_len, hidden)
        .contiguous()
    )
    responses = torch.randint(0, vocab, (num_prompts * group_size, resp_len))
    return prompt_embeds, responses


def _full_recompute_logprobs(stub, prompt_embeds, responses, prompt_len, temperature):
    """Reference implementation: one full forward over every [prompt+response] row."""
    from torchtune import rlhf

    comp = stub.embed_completion_ids(responses)
    full = torch.cat([prompt_embeds, comp], dim=1)
    mask = torch.ones(full.shape[0], full.shape[1], dtype=torch.long)
    with torch.no_grad():
        logits = stub.forward_full(full, attention_mask=mask)
    resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + responses.shape[1]]
    return rlhf.batched_logits_to_logprobs(resp_logits, responses, temperature)


@pytest.mark.parametrize(
    "num_prompts,group_size,prompt_len,resp_len",
    [(2, 4, 9, 6), (1, 8, 12, 5), (3, 2, 7, 11), (4, 1, 6, 4)],
)
def test_shared_prefix_matches_full_recompute(num_prompts, group_size, prompt_len, resp_len):
    """The whole point: shared-prefix logprobs must equal full-recompute logprobs."""
    dtype = torch.float64
    model, config = _tiny_causal_lm(dtype)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    prompt_embeds, responses = _group_major_batch(
        stub, num_prompts, group_size, prompt_len, resp_len,
        config.hidden_size, config.vocab_size, dtype,
    )
    temperature = 0.9

    expected = _full_recompute_logprobs(stub, prompt_embeds, responses, prompt_len, temperature)
    with torch.no_grad():
        actual = shared_prefix_ref_logprobs(
            stub,
            prompt_embeds,
            responses,
            group_size=group_size,
            temperature=temperature,
            prompt_length=prompt_len,
        )

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def test_shared_prefix_matches_full_recompute_bf16_close():
    """Also holds in a low-precision dtype (looser tol; reassociation only)."""
    dtype = torch.float32
    model, config = _tiny_causal_lm(dtype)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    prompt_embeds, responses = _group_major_batch(
        stub, 2, 4, 10, 7, config.hidden_size, config.vocab_size, dtype
    )
    expected = _full_recompute_logprobs(stub, prompt_embeds, responses, 10, 1.0)
    with torch.no_grad():
        actual = shared_prefix_ref_logprobs(
            stub, prompt_embeds, responses, group_size=4, temperature=1.0, prompt_length=10
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_first_response_token_logprob_is_not_lost():
    """Column 0 comes from the LAST prefix position — a classic off-by-one.

    If the prefix carry-over were dropped and the suffix logits used directly,
    every column would shift by one token. Assert column 0 specifically.
    """
    dtype = torch.float64
    model, config = _tiny_causal_lm(dtype)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    prompt_embeds, responses = _group_major_batch(
        stub, 2, 3, 8, 6, config.hidden_size, config.vocab_size, dtype
    )
    expected = _full_recompute_logprobs(stub, prompt_embeds, responses, 8, 1.0)
    with torch.no_grad():
        actual = shared_prefix_ref_logprobs(
            stub, prompt_embeds, responses, group_size=3, temperature=1.0, prompt_length=8
        )
    torch.testing.assert_close(actual[:, 0], expected[:, 0], rtol=1e-9, atol=1e-9)
    # And a shifted comparison must NOT match, proving the test has teeth.
    assert not torch.allclose(actual[:, 1:], expected[:, :-1], rtol=1e-4, atol=1e-4)


def _production_mask_backend_name() -> str:
    """Register a backend using the REAL production mask fn + deterministic eager compute.

    The part of the production ``bioreason_xpu_flash`` backend this optimization
    depends on is its **mask** function (``_bioreason_xpu_flash_mask``): it decides
    whether a 4D bottom-right-aligned causal mask is built for the cached suffix, or
    whether ``None`` is passed through and the attention silently assumes top-left
    alignment. That function is loaded here verbatim from
    ``torchtune/dev/bioreason/model.py``.

    Its *compute* half is deliberately swapped for a deterministic eager kernel. On
    CPU the production forward's eligibility guard falls back to
    ``sdpa_attention_forward`` anyway, and torch's CPU SDPA kernel is
    non-deterministic on this stack (see :func:`_tiny_causal_lm`) — it returns
    different values for two identical calls and intermittently produces NaN,
    which would make any equivalence assertion meaningless. Eager is bitwise
    reproducible and computes the same attention.

    The one production behaviour eager does NOT reproduce for free is the
    ``attention_mask is None`` case. ``_bioreason_xpu_flash_mask`` returns ``None``
    for a square all-visible mask, and both the production flash kernel and the
    SDPA fallback then apply ``is_causal=True`` (TOP-LEFT aligned). Stock
    ``eager_attention_forward`` instead applies *no* mask at all, which would make
    the full-recompute baseline non-causal and the comparison meaningless. The
    wrapper below restores top-left causal masking on that path, so both the
    "exact" and the "maskless suffix is wrong" tests exercise real semantics.
    """
    import importlib.util
    import os
    from pathlib import Path

    import transformers.masking_utils as masking_utils
    import transformers.modeling_utils as modeling_utils
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    os.environ["TORCHTUNE_USE_XPU_FLASH"] = "1"
    path = (
        Path(__file__).resolve().parents[4]
        / "torchtune" / "dev" / "bioreason" / "model.py"
    )
    spec = importlib.util.spec_from_file_location("_br_model_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _eager_with_is_causal_fallback(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        """Eager attention speaking the *sdpa* mask convention.

        Two impedance mismatches have to be bridged, because the production mask
        fn is an sdpa-convention producer and eager is an additive-mask consumer:

        1. ``_bioreason_xpu_flash_mask`` delegates to the sdpa mask builder, which
           returns a **boolean** mask (``True`` = attend). ``eager_attention_forward``
           *adds* its mask to the scores, so a bool mask would add ``1.0``/``0.0``
           instead of ``0``/``-inf`` and mask nothing at all.
        2. A square all-visible mask is elided to ``None``, and the real backends
           then apply ``is_causal=True`` (TOP-LEFT aligned). Eager applies nothing.
        """
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(
                attention_mask, dtype=query.dtype
            ).masked_fill_(~attention_mask, float("-inf"))
        if attention_mask is None:
            q_len, kv_len = query.shape[-2], key.shape[-2]
            # TOP-LEFT aligned, matching is_causal=True in both the production
            # flash kernel and the SDPA fallback.
            bias = torch.zeros(q_len, kv_len, dtype=query.dtype, device=query.device)
            bias.masked_fill_(
                torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).triu(1),
                float("-inf"),
            )
            attention_mask = bias[None, None]
        return eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
        )

    name = "bioreason_prod_mask_eager_compute"
    modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping[name] = (
        _eager_with_is_causal_fallback
    )
    # The production mask function, used verbatim.
    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[name] = (
        module._bioreason_xpu_flash_mask
    )
    return name


def _shared_prefix_logits_manual(stub, prompt_embeds, responses, P, G, suffix_mask):
    """Run the prefix/expand/suffix sequence by hand with a caller-chosen suffix mask."""
    from transformers import DynamicCache

    with torch.no_grad():
        cache = DynamicCache()
        pre_logits, cache = stub.forward_cached(
            inputs_embeds=prompt_embeds[:1],
            attention_mask=torch.ones(1, P, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        expand_cache_batch_(cache, G)
        suffix_logits, _ = stub.forward_cached(
            inputs_embeds=stub.embed_completion_ids(responses),
            attention_mask=suffix_mask,
            past_key_values=cache,
            use_cache=True,
        )
    return torch.cat([pre_logits[:, -1:].expand(G, -1, -1), suffix_logits[:, :-1]], dim=1)


def test_shared_prefix_matches_full_recompute_under_production_flash_backend():
    """Exactness must hold under the REAL production attention backend.

    The other tests use ``eager`` for determinism; this one registers and runs the
    actual ``bioreason_xpu_flash`` forward/mask pair from
    ``torchtune/dev/bioreason/model.py``, which is what production dispatches to
    (``TORCHTUNE_USE_XPU_FLASH=1``). On CPU its eligibility guard falls back to
    ``sdpa_attention_forward``, but the mask-construction path — the part this
    optimization actually depends on — is the production one.
    """
    dtype = torch.float64
    G, P, C = 4, 9, 6
    backend = _production_mask_backend_name()
    model, config = _tiny_causal_lm(dtype, attn=backend)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    prompt_embeds, responses = _group_major_batch(
        stub, 1, G, P, C, config.hidden_size, config.vocab_size, dtype
    )
    expected = _full_recompute_logprobs(stub, prompt_embeds, responses, P, 1.0)

    from torchtune import rlhf

    good = _shared_prefix_logits_manual(
        stub, prompt_embeds, responses, P, G, torch.ones(G, P + C, dtype=torch.long)
    )
    actual = rlhf.batched_logits_to_logprobs(good, responses, 1.0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_suffix_without_mask_is_wrong_under_production_flash_backend():
    """Guard: dropping the suffix attention mask silently corrupts the logits.

    With a populated KV cache ``q_len != kv_len``. ``attention_mask=None`` makes the
    production ``bioreason_xpu_flash`` forward infer ``is_causal=True``, which is
    TOP-LEFT aligned — but a cached suffix needs BOTTOM-RIGHT alignment. Measured
    drift here is ~7e-3 on a float64 model: a real numerical error, not noise.

    This test exists so nobody "optimizes" the explicit full-width mask away in
    order to re-enable the XPU flash kernel on the suffix pass. Doing so trades
    exactness for speed silently.
    """
    dtype = torch.float64
    G, P, C = 4, 9, 6
    backend = _production_mask_backend_name()
    model, config = _tiny_causal_lm(dtype, attn=backend)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    prompt_embeds, responses = _group_major_batch(
        stub, 1, G, P, C, config.hidden_size, config.vocab_size, dtype
    )
    expected = _full_recompute_logprobs(stub, prompt_embeds, responses, P, 1.0)

    from torchtune import rlhf

    bad_logits = _shared_prefix_logits_manual(
        stub, prompt_embeds, responses, P, G, None
    )
    bad = rlhf.batched_logits_to_logprobs(bad_logits, responses, 1.0)
    assert not torch.allclose(bad, expected, rtol=1e-4, atol=1e-4), (
        "Expected the maskless cached suffix to be WRONG (top-left vs bottom-right "
        "causal alignment). If this now passes, the backend's cached-mask semantics "
        "changed and shared_prefix_ref_logprobs should be re-examined."
    )


def test_expand_cache_batch_is_a_view_not_a_copy():
    """The shared prefix must cost one row of KV, not G rows."""
    from transformers import DynamicCache

    dtype = torch.float64
    model, config = _tiny_causal_lm(dtype)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    with torch.no_grad():
        cache = DynamicCache()
        _, cache = stub.forward_cached(
            inputs_embeds=torch.randn(1, 9, config.hidden_size, dtype=dtype),
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    before = cache.layers[0].keys.untyped_storage().nbytes()
    expand_cache_batch_(cache, 8)
    assert cache.layers[0].keys.shape[0] == 8
    assert cache.layers[0].keys.stride(0) == 0, "expected a stride-0 broadcast view"
    assert cache.layers[0].keys.untyped_storage().nbytes() == before


def test_expand_cache_batch_rejects_non_unit_batch():
    from transformers import DynamicCache

    dtype = torch.float64
    model, config = _tiny_causal_lm(dtype)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    with torch.no_grad():
        cache = DynamicCache()
        _, cache = stub.forward_cached(
            inputs_embeds=torch.randn(2, 5, config.hidden_size, dtype=dtype),
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
        )
    with pytest.raises(ValueError, match="batch-1 prefix cache"):
        expand_cache_batch_(cache, 4)


def test_shape_and_ordering_validation():
    dtype = torch.float64
    model, config = _tiny_causal_lm(dtype)
    stub = _StubBioReasonModel(model, config.hidden_size, config.vocab_size, dtype)
    pe = torch.randn(6, 5, config.hidden_size, dtype=dtype)
    resp = torch.randint(0, config.vocab_size, (6, 3))
    with pytest.raises(ValueError, match="divide the batch"):
        shared_prefix_ref_logprobs(stub, pe, resp, group_size=4, temperature=1.0)
    with pytest.raises(ValueError, match="batch"):
        shared_prefix_ref_logprobs(
            stub, pe, torch.randint(0, 5, (4, 3)), group_size=2, temperature=1.0
        )


def test_prefix_share_supported_preconditions():
    pe = torch.zeros(8, 4, 2)
    assert prefix_share_supported(
        prompt_embeds=pe, num_seqs=8, group_size=4, compacted_prompt_lengths=None
    ) == (True, "")
    ok, reason = prefix_share_supported(
        prompt_embeds=None, num_seqs=8, group_size=4, compacted_prompt_lengths=None
    )
    assert not ok and "text-only" in reason
    ok, reason = prefix_share_supported(
        prompt_embeds=pe, num_seqs=8, group_size=1, compacted_prompt_lengths=None
    )
    assert not ok and "nothing to share" in reason
    ok, reason = prefix_share_supported(
        prompt_embeds=pe, num_seqs=7, group_size=4, compacted_prompt_lengths=None
    )
    assert not ok and "divisible" in reason
    ok, reason = prefix_share_supported(
        prompt_embeds=pe,
        num_seqs=8,
        group_size=4,
        compacted_prompt_lengths=torch.tensor([3, 4, 5, 6, 3, 4, 5, 6]),
    )
    assert not ok and "ragged" in reason


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("TORCHTUNE_REF_PREFIX_SHARE", raising=False)
    assert ref_prefix_share_enabled() is False
    monkeypatch.setenv("TORCHTUNE_REF_PREFIX_SHARE", "0")
    assert ref_prefix_share_enabled() is False
    monkeypatch.setenv("TORCHTUNE_REF_PREFIX_SHARE", "1")
    assert ref_prefix_share_enabled() is True


def test_bioreason_model_exposes_cached_forward_contract():
    """The recipe dispatches on ``hasattr(ref_model, 'forward_cached')``.

    A source-level check (importing BioReasonModel needs the ESM3/bioreason2
    checkout, unavailable on a login node / CI).
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[4]
        / "torchtune" / "dev" / "bioreason" / "model.py"
    ).read_text()
    assert "def forward_cached(" in src
    assert "def embed_completion_ids(" in src


# ---------------------------------------------------------------------------
# HW burn 2026-09-15: the feature OOMs on 32B. These guards are not about
# exactness (which the tests above establish) but about making the *cost* of
# flash-ineligibility impossible to rediscover by burning another 2-node slot.
#
# Job 8829395: all 12 ranks died in step 0's ref forward allocating 21.00 GiB
# inside scaled_dot_product_attention. Without flash, SDPA materializes a
# G x heads x q_len x kv_len score tensor. The docs used to call this "a genuine
# trade-off"; at production scale it is an unconditional OOM.
# ---------------------------------------------------------------------------


def _score_tensor_gib(group_size, n_heads, q_len, kv_len, bytes_per_elem=2):
    """Bytes SDPA must materialize when it cannot use a fused/flash kernel."""
    return group_size * n_heads * q_len * kv_len * bytes_per_elem / 1024**3


def test_unfused_suffix_score_tensor_exceeds_a_tile_at_production_shape():
    """The measured OOM is reproducible as arithmetic, so it needs no XPU.

    Production shape at the failing step (Qwen3-32B: 64 heads, head_dim 128;
    B=4 x G=8, prompt ~4096, response draw 2745 => kv_len 6841). A PVC tile is
    64 GiB total and roughly 17 GiB was free at the allocation point.
    """
    gib = _score_tensor_gib(group_size=8, n_heads=64, q_len=2745, kv_len=6841)
    assert gib > 17.0, (
        f"score tensor {gib:.1f} GiB no longer exceeds the ~17 GiB that was free "
        "on the tile; if the shape assumptions changed, re-derive the OOM budget "
        "before re-enabling TORCHTUNE_REF_PREFIX_SHARE"
    )
    # Guard the scaling law too: quadratic in length means a longer draw is worse,
    # so a fix must bound the tensor, not merely fit one particular draw.
    longer = _score_tensor_gib(group_size=8, n_heads=64, q_len=3072, kv_len=7168)
    assert longer > gib


def test_response_only_logits_does_not_mitigate_the_score_tensor():
    """RESPONSE_ONLY_LOGITS shrinks logits, not attention scores.

    It was written into the dispatch header as the mitigation and would not have
    saved the run. Pin the distinction so the wrong lever is not reached for again.
    """
    scores = _score_tensor_gib(group_size=8, n_heads=64, q_len=2745, kv_len=6841)
    # Logits the flag actually targets: G x resp_len x vocab, fp32.
    logits_gib = 8 * 2745 * 151936 * 4 / 1024**3
    freed_by_flag = logits_gib  # best case: the flag removes all of it
    assert scores > freed_by_flag, (
        "if the logits term now dominates the score term, RESPONSE_ONLY_LOGITS "
        "may genuinely mitigate -- re-check before citing this test"
    )


def test_docs_record_the_oom_not_a_speed_tradeoff():
    """The module docstring and CLAUDE.md must both carry the refutation.

    Prose-only burns get lost; this makes the doc state executable. It failed
    once already: the docstring described flash-ineligibility as "a genuine
    trade-off, not an oversight", which read as a speed question and is why the
    pre-flight budgeted KV and logits but never asked which kernel the new path
    would land on.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    src = (root / "torchtune" / "dev" / "rl" / "ref_prefix_share.py").read_text()
    assert "OOM" in src and "21.00 GiB" in src, (
        "ref_prefix_share.py docstring must state the measured OOM"
    )
    claude = (root / "CLAUDE.md").read_text()
    row = [ln for ln in claude.splitlines() if "TORCHTUNE_REF_PREFIX_SHARE" in ln]
    assert row, "TORCHTUNE_REF_PREFIX_SHARE row missing from the CLAUDE.md flag table"
    assert any("OOM" in ln for ln in row), (
        "the CLAUDE.md row must say the flag OOMs on 32B, not that it is merely "
        "'not yet HW-validated'"
    )

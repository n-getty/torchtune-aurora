# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Exact prefix sharing for the frozen reference-model forward in GRPO.

GRPO samples ``G`` continuations per prompt. The reference forward therefore
recomputes each prompt prefix ``G`` times. At BioReason 32B (B=4 prompts x G=8
samples, ~4096-token prompt vs ~1101-token response) that is an 8x redundant
recomputation of 79% of the tokens.

**Why this is exact here and not an approximation.** The reference model is
frozen (``requires_grad_(False)``) and the whole forward runs under
``torch.no_grad`` — see ``recipes/dev/grpo_bioreason_distributed_xpu.py``'s ref
construction. There is no gradient to attribute back through a shared prefix, so
reusing one prompt's KV cache across its ``G`` continuations produces the same
numbers as the full recompute, up to floating-point reassociation inside the
attention kernel. This is emphatically **not** true of the policy forward, which
runs under autograd; do not reuse this module there.

Mask semantics (the sharp edge)
-------------------------------
With a KV cache the suffix forward has ``q_len != kv_len``. SDPA's
``is_causal=True`` means **top-left** alignment, which is the *wrong* mask for a
cached suffix -- the correct one is **bottom-right** aligned (query ``i`` is at
absolute position ``prefix_len + i``). Passing ``attention_mask=None`` through
HF in this regime silently produces top-left alignment and therefore wrong
logits (measured drift ~6e-3 on a float64 reference model -- a real numerical
error, not noise). This module therefore **always passes an explicit 2D
attention mask of width ``prefix_len + suffix_len``** on the suffix call, which
makes ``transformers`` build the correctly bottom-right-aligned 4D causal mask.

Consequence for the XPU flash kernel: ``bioreason_xpu_flash``'s eligibility
guard (``torchtune/dev/bioreason/model.py``) requires ``attention_mask is None``
**and** ``query.shape[-2] == key.shape[-2]``. A cached suffix violates both by
construction, so the suffix pass falls back to ``sdpa_attention_forward``. The
prefix pass keeps flash (square, all-ones mask, which ``transformers`` elides to
``None``).

**This is not a speed trade-off. On 32B it is an unconditional OOM.**
HW-measured 2026-09-15 (job 8829395, 2N B4/G8): all 12 ranks died in step 0's ref
forward trying to allocate **21.00 GiB** inside
``torch.nn.functional.scaled_dot_product_attention``. Without flash, SDPA
materializes the score tensor, whose size is::

    G x n_heads x q_len x kv_len x 2 bytes
      = 8 x 64 x 2745 x 6841 x 2  ~= 18 GiB

It grows **quadratically in rollout length and linearly in G**, so it is not a
headroom problem that a smaller batch elsewhere can absorb. Note that
``TORCHTUNE_RESPONSE_ONLY_LOGITS=1`` does **not** help: it shrinks the *logits*,
not the attention scores.

Before reviving this module, in order: (1) chunk the suffix pass at
``ref_forward_batch_size`` rows so the score tensor is bounded; (2) re-derive the
economics, because the premise ("prefix recompute dominates") trades a
flash-accelerated 8x prompt pass for a math-SDPA suffix pass over a ~5200-token
KV, and that balance is no longer obviously favorable; (3) only then spend HW.
Default stays OFF. See
``memory/project_bioreason_ref_prefix_share_oom_flash_ineligible_suffix_20260915.md``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Protocol

import torch

log = logging.getLogger(__name__)

REF_PREFIX_SHARE_ENV = "TORCHTUNE_REF_PREFIX_SHARE"


def ref_prefix_share_enabled() -> bool:
    """Return True when ``TORCHTUNE_REF_PREFIX_SHARE=1`` is set.

    Returns:
        bool: whether the opt-in exact-prefix-sharing ref forward is enabled.
    """
    return os.environ.get(REF_PREFIX_SHARE_ENV, "0") == "1"


class SupportsCachedForward(Protocol):
    """Minimal contract a model must satisfy to use :func:`shared_prefix_ref_logprobs`."""

    def forward_cached(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        logits_to_keep: int = 0,
    ) -> tuple[torch.Tensor, Any]:
        ...

    def embed_completion_ids(self, completion_ids: torch.Tensor) -> torch.Tensor:
        ...


def _cache_layers(cache: Any) -> list:
    """Return the per-layer entries of a ``transformers`` cache object.

    Supports the ``transformers>=4.54`` ``cache.layers[i].keys/.values`` layout.

    Args:
        cache: a ``transformers`` cache instance (e.g. ``DynamicCache``).

    Returns:
        list: the per-layer cache entries.

    Raises:
        TypeError: if the cache does not expose the expected ``.layers`` layout.
    """
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(
            "ref prefix sharing requires a transformers cache exposing `.layers` "
            f"with per-layer `.keys`/`.values` (got {type(cache).__name__}). "
            "This layout landed in transformers 4.54; upgrade or disable "
            f"{REF_PREFIX_SHARE_ENV}."
        )
    return list(layers)


def expand_cache_batch_(cache: Any, repeats: int) -> Any:
    """Broadcast a batch-1 KV cache across ``repeats`` rows, in place.

    Uses ``Tensor.expand`` (stride-0 view) rather than ``repeat``/``contiguous``
    so the shared prefix costs one row of KV, not ``repeats`` rows. The
    subsequent suffix forward's ``cache.update()`` concatenates onto this view
    and materializes the result, so no kernel ever sees the stride-0 tensor as
    its final key/value operand.

    Args:
        cache: cache whose layers currently hold ``[1, kv_heads, prefix, head_dim]``.
        repeats (int): number of continuations sharing this prefix.

    Returns:
        The same cache object, mutated.

    Raises:
        ValueError: if ``repeats`` is not positive, or a layer's batch dim is not 1.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if repeats == 1:
        return cache
    for layer_idx, layer in enumerate(_cache_layers(cache)):
        keys, values = layer.keys, layer.values
        if keys.shape[0] != 1 or values.shape[0] != 1:
            raise ValueError(
                f"expand_cache_batch_ expects a batch-1 prefix cache; layer "
                f"{layer_idx} has key batch {keys.shape[0]}, value batch "
                f"{values.shape[0]}"
            )
        layer.keys = keys.expand(repeats, -1, -1, -1)
        layer.values = values.expand(repeats, -1, -1, -1)
    return cache


def _new_cache() -> Any:
    """Construct an empty ``transformers`` ``DynamicCache``.

    Returns:
        A fresh ``DynamicCache``.
    """
    from transformers import DynamicCache

    return DynamicCache()


def shared_prefix_ref_logprobs(
    model: SupportsCachedForward,
    prompt_embeds: torch.Tensor,
    responses: torch.Tensor,
    *,
    group_size: int,
    temperature: float,
    prompt_length: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    logprob_fn: Optional[Callable[..., torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute reference response logprobs, sharing each prompt's prefix across its group.

    The rows of ``prompt_embeds``/``responses`` must be **group-major**: the
    ``group_size`` continuations of prompt ``b`` occupy rows
    ``[b * group_size, (b + 1) * group_size)``, and every row within a group must
    carry a bit-identical prompt. This is how
    ``grpo_bioreason_distributed_xpu.generate_trajectory`` builds the batch
    (``pe_base[:, None].expand(-1, G, -1, -1).reshape(...)``).

    For each group this runs the prefix **once** (batch 1) with ``use_cache=True``,
    broadcasts the resulting KV cache across the group, and then runs only the
    response span against that cache. The final prefix position's logits -- which
    predict the *first* response token and would otherwise be lost -- are carried
    over explicitly from the prefix pass.

    Args:
        model: object satisfying :class:`SupportsCachedForward`.
        prompt_embeds (torch.Tensor): ``[B*G, P, H]`` prompt embeddings (may be on CPU).
        responses (torch.Tensor): ``[B*G, C]`` response token IDs.
        group_size (int): ``G``, the number of continuations per prompt.
        temperature (float): sampling temperature used to scale logits.
        prompt_length (Optional[int]): prefix width ``P``. Defaults to
            ``prompt_embeds.shape[1]``. Must be uniform across the whole batch;
            row-wise (compacted) prompt lengths are not supported -- see
            :func:`prefix_share_supported`.
        attention_mask (Optional[torch.Tensor]): ``[B*G, P+C]`` mask the equivalent
            full forward would use. ``None`` means "all positions visible", which
            is what the production flash path reduces to for right-padded batches.
        position_ids (Optional[torch.Tensor]): ``[B*G, P+C]`` positions, or ``None``
            to let the backbone derive them from the cache offset.
        logprob_fn (Optional[Callable]): ``(logits, tokens, temperature) -> logprobs``.
            Defaults to ``torchtune.rlhf.batched_logits_to_logprobs``.
        device (Optional[torch.device]): device to place the returned logprobs on.
            Defaults to the device of the computed logits.

    Returns:
        torch.Tensor: ``[B*G, C]`` float logprobs, laid out in the same row order
        as ``responses``.

    Raises:
        ValueError: on shape/ordering violations that would make the result wrong.
    """
    if prompt_embeds.ndim != 3:
        raise ValueError(
            f"prompt_embeds must be [B*G, P, H], got {tuple(prompt_embeds.shape)}"
        )
    if responses.ndim != 2:
        raise ValueError(f"responses must be [B*G, C], got {tuple(responses.shape)}")
    if prompt_embeds.shape[0] != responses.shape[0]:
        raise ValueError(
            f"prompt_embeds batch {prompt_embeds.shape[0]} != responses batch "
            f"{responses.shape[0]}"
        )
    num_rows = responses.shape[0]
    if group_size <= 0 or num_rows % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must be positive and divide the batch "
            f"({num_rows} rows)"
        )

    if logprob_fn is None:
        from torchtune import rlhf

        logprob_fn = rlhf.batched_logits_to_logprobs

    prefix_len = int(prompt_length if prompt_length is not None else prompt_embeds.shape[1])
    response_len = int(responses.shape[1])
    total_len = prefix_len + response_len
    if attention_mask is not None and attention_mask.shape[1] != total_len:
        raise ValueError(
            f"attention_mask width {attention_mask.shape[1]} != prefix+response "
            f"({prefix_len}+{response_len}={total_len})"
        )

    num_groups = num_rows // group_size
    out_chunks: list[torch.Tensor] = []

    for g in range(num_groups):
        lo, hi = g * group_size, (g + 1) * group_size
        group_prompt = prompt_embeds[lo : lo + 1]
        group_responses = responses[lo:hi]

        # --- prefix: one row, square + all-visible => transformers elides the
        # mask to None and the XPU flash kernel stays eligible.
        prefix_mask = (
            attention_mask[lo : lo + 1, :prefix_len] if attention_mask is not None else None
        )
        prefix_positions = (
            position_ids[lo : lo + 1, :prefix_len] if position_ids is not None else None
        )
        cache = _new_cache()
        prefix_logits, cache = model.forward_cached(
            inputs_embeds=group_prompt,
            attention_mask=prefix_mask,
            position_ids=prefix_positions,
            past_key_values=cache,
            use_cache=True,
            # Only the LAST prefix position's logits are ever consumed (it predicts
            # the first response token). Asking the backbone for one row of logits
            # avoids materializing [1, P, vocab].
            logits_to_keep=1,
        )
        # Defensive: honour logits_to_keep having been ignored by an older backbone.
        last_prefix_logits = prefix_logits[:, -1:]

        expand_cache_batch_(cache, group_size)

        # --- suffix: MUST carry an explicit full-width mask so transformers builds
        # the bottom-right-aligned causal mask. See the module docstring.
        if attention_mask is not None:
            suffix_mask = attention_mask[lo:hi]
        else:
            suffix_mask = torch.ones(
                (group_size, total_len),
                dtype=torch.long,
                device=group_responses.device,
            )
        suffix_positions = position_ids[lo:hi, prefix_len:] if position_ids is not None else None

        completion_embeds = model.embed_completion_ids(group_responses)
        suffix_mask = suffix_mask.to(completion_embeds.device)
        if suffix_positions is not None:
            suffix_positions = suffix_positions.to(completion_embeds.device)

        suffix_logits, _ = model.forward_cached(
            inputs_embeds=completion_embeds,
            attention_mask=suffix_mask,
            position_ids=suffix_positions,
            past_key_values=cache,
            use_cache=True,
        )
        del cache

        # Stitch: full-sequence logit index P-1 (predicts response token 0) comes
        # from the prefix pass; indices P..P+C-2 are suffix positions 0..C-2. The
        # suffix's own last position predicts a token beyond the response and is
        # discarded -- dropping the prefix carry-over instead would silently shift
        # every logprob by one token.
        response_logits = torch.cat(
            [last_prefix_logits.expand(group_size, -1, -1), suffix_logits[:, :-1]],
            dim=1,
        )
        del suffix_logits, last_prefix_logits

        group_logprobs = logprob_fn(response_logits, group_responses.to(response_logits.device), temperature)
        del response_logits
        out_chunks.append(group_logprobs if device is None else group_logprobs.to(device))

    return torch.cat(out_chunks, dim=0)


def prefix_share_supported(
    *,
    prompt_embeds: Optional[torch.Tensor],
    num_seqs: int,
    group_size: int,
    compacted_prompt_lengths: Any = None,
) -> tuple[bool, str]:
    """Check the preconditions under which prefix sharing stays exact.

    Args:
        prompt_embeds (Optional[torch.Tensor]): the trajectory's prompt embeddings,
            or ``None`` for the text-only (token-ID) path.
        num_seqs (int): total rows in the batch (``B*G``).
        group_size (int): ``G``.
        compacted_prompt_lengths: the ``context_length`` value produced by
            ``compact_prompt_completion_batch`` -- an ``int`` when every row shares
            one prompt width, or a ``Tensor`` of per-row widths when they differ.

    Returns:
        tuple[bool, str]: ``(supported, reason)``. ``reason`` is empty when supported.
    """
    if prompt_embeds is None:
        return False, "text-only path (no prompt_embeds); prefix sharing is embeds-only"
    if group_size <= 1:
        return False, f"group_size={group_size}; nothing to share"
    if num_seqs % group_size != 0:
        return False, f"num_seqs={num_seqs} not divisible by group_size={group_size}"
    if isinstance(compacted_prompt_lengths, torch.Tensor):
        # Row-compaction produced ragged prompt widths. Within a group the prompt is
        # identical, so this only happens ACROSS groups -- but a ragged batch means
        # the per-group prefix width differs and the single shared `prefix_len`
        # contract above no longer holds for the batch as a whole.
        return False, "row-compacted ragged prompt lengths; prefix width is not uniform"
    return True, ""

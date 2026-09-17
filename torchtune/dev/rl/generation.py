# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch

from torchtune import utils
from torchtune.generation import generate_next_token, get_causal_mask_from_padding_mask
from torchtune.generation._generation import (
    get_position_ids_from_padding_mask,
    update_stop_tokens_tracker,
)
from torchtune.modules import TransformerDecoder

from tqdm.auto import trange


def trim_query_responses_to_global_max(
    query_responses: torch.Tensor,
    context_length: int,
    pad_id: int,
    process_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> tuple[torch.Tensor, int]:
    """Remove trailing response columns that are padding across a process group."""
    responses = query_responses[:, context_length:]
    non_padding_columns = responses.ne(pad_id).any(dim=0).nonzero().flatten()
    active_response_length = (
        int(non_padding_columns[-1].item()) + 1 if non_padding_columns.numel() else 1
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        collective_device = query_responses.device
        if (
            process_group is not None
            and torch.distributed.get_backend(process_group) == "gloo"
        ):
            collective_device = torch.device("cpu")
        global_length = torch.tensor(
            active_response_length,
            dtype=torch.int64,
            device=collective_device,
        )
        torch.distributed.all_reduce(
            global_length,
            op=torch.distributed.ReduceOp.MAX,
            group=process_group,
        )
        active_response_length = int(global_length.item())

    total_length = context_length + active_response_length
    return query_responses[:, :total_length], active_response_length


def get_right_padded_response_length(response_padding_masks: torch.Tensor) -> int:
    """Return the longest active response length, retaining at least one token."""
    if response_padding_masks.ndim != 2:
        raise ValueError(
            "response_padding_masks must have shape [batch, response_length]"
        )
    return max(1, int((~response_padding_masks).sum(dim=-1).max().item()))


def pad_response_logprobs(
    logprobs: torch.Tensor, response_length: int, value: float = 1.0
) -> torch.Tensor:
    """Right-pad response logprobs to a common trajectory width."""
    if logprobs.ndim != 2:
        raise ValueError("logprobs must have shape [batch, response_length]")
    if logprobs.shape[1] > response_length:
        raise ValueError("response_length cannot be smaller than the logprob width")
    if logprobs.shape[1] == response_length:
        return logprobs
    return torch.nn.functional.pad(
        logprobs, (0, response_length - logprobs.shape[1]), value=value
    )


def get_descending_response_chunk_ranges(
    response_padding_masks: torch.Tensor, batch_size: int
) -> list[tuple[int, int]]:
    """Return fixed batch ranges ordered from widest active response to shortest."""
    if response_padding_masks.ndim != 2:
        raise ValueError(
            "response_padding_masks must have shape [batch, response_length]"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ranges = [
        (start, min(start + batch_size, response_padding_masks.shape[0]))
        for start in range(0, response_padding_masks.shape[0], batch_size)
    ]
    return sorted(
        ranges,
        key=lambda bounds: get_right_padded_response_length(
            response_padding_masks[bounds[0] : bounds[1]]
        ),
        reverse=True,
    )


def get_length_sorted_response_chunks(
    response_padding_masks: torch.Tensor, batch_size: int
) -> list[list[int]]:
    """Group response rows by descending active length into fixed-size chunks."""
    if response_padding_masks.ndim != 2:
        raise ValueError(
            "response_padding_masks must have shape [batch, response_length]"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    lengths = (~response_padding_masks).sum(dim=-1).tolist()
    sorted_rows = sorted(range(len(lengths)), key=lambda row: (-lengths[row], row))
    return [
        sorted_rows[start : start + batch_size]
        for start in range(0, len(sorted_rows), batch_size)
    ]


def compact_prompt_completion_batch(
    prompt_embeds: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | torch.Tensor]:
    """Pack each completion directly after its row's right-padded prompt."""
    if prompt_embeds.ndim != 3 or prompt_token_ids.ndim != 2 or completion_ids.ndim != 2:
        raise ValueError("expected prompt embeds [B,P,H] and token IDs [B,L]")
    if prompt_embeds.shape[:2] != prompt_token_ids.shape:
        raise ValueError("prompt embedding and token shapes must match")
    if prompt_embeds.shape[0] != completion_ids.shape[0]:
        raise ValueError("prompt and completion batch sizes must match")

    prompt_lengths = (prompt_token_ids != pad_id).sum(dim=-1)
    if prompt_lengths.numel() and bool((prompt_lengths == 0).any()):
        raise ValueError("prompt rows must contain at least one non-padding token")
    max_prompt_length = int(prompt_lengths.max().item()) if prompt_lengths.numel() else 0
    prompt_embeds = prompt_embeds[:, :max_prompt_length]
    prompt_token_ids = prompt_token_ids[:, :max_prompt_length]

    if prompt_lengths.numel() and bool((prompt_lengths == max_prompt_length).all()):
        token_ids = torch.cat([prompt_token_ids, completion_ids], dim=1)
        attention_mask = (token_ids != pad_id).long()
        position_ids = get_position_ids_from_padding_mask(attention_mask.bool())
        return prompt_embeds, attention_mask, position_ids, max_prompt_length

    batch_size, completion_length = completion_ids.shape
    completion_lengths = (completion_ids != pad_id).sum(dim=-1)
    compact_width = int((prompt_lengths + completion_lengths).max().item())
    source_token_ids = torch.cat(
        [
            prompt_token_ids,
            completion_ids,
            completion_ids.new_full((batch_size, 1), pad_id),
        ],
        dim=1,
    )
    output_positions = torch.arange(
        compact_width,
        device=completion_ids.device,
    ).unsqueeze(0)
    prompt_lengths_device = prompt_lengths.to(completion_ids.device).unsqueeze(1)
    gather_indices = torch.where(
        output_positions < prompt_lengths_device,
        output_positions,
        max_prompt_length + output_positions - prompt_lengths_device,
    )
    gather_indices = gather_indices.clamp_max(source_token_ids.shape[1] - 1)
    token_ids = source_token_ids.gather(1, gather_indices.expand(batch_size, -1))
    attention_mask = (token_ids != pad_id).long()
    position_ids = get_position_ids_from_padding_mask(attention_mask.bool())
    return prompt_embeds, attention_mask, position_ids, prompt_lengths


def gather_response_logits(
    logits: torch.Tensor,
    prompt_lengths: int | torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Select next-token logits for responses with scalar or row-wise prompt lengths."""
    if isinstance(prompt_lengths, int):
        return logits[:, prompt_lengths - 1 : prompt_lengths + response_length - 1]
    if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != logits.shape[0]:
        raise ValueError("prompt_lengths must have shape [B]")
    if bool((prompt_lengths <= 0).any()):
        raise ValueError("prompt lengths must be positive")
    offsets = torch.arange(response_length, device=logits.device).unsqueeze(0)
    indices = prompt_lengths.to(logits.device).unsqueeze(1) - 1 + offsets
    indices = indices.clamp_max(logits.shape[1] - 1)
    return logits.gather(1, indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))


# ── Response-only logits (TORCHTUNE_RESPONSE_ONLY_LOGITS) ────────────────────
# The full-width logits tensor [B, S, V] is the single largest activation in a
# GRPO forward (Qwen3-32B: V=151936; at fbs=2, S=4922, fp32 that is 5.98 GB),
# yet every consumer immediately calls gather_response_logits() and keeps only
# the ~1101 response columns (1.34 GB) — 4.64 GB / 77.6% is projected and then
# discarded. Slicing the HIDDEN states [B, S, H] (H=5120, 30x narrower than V,
# so the full-width hidden tensor is only 0.20 GB) BEFORE the lm_head yields
# bit-identical response logits at a fraction of the peak. The lm_head is only
# ~2.4% of per-token forward FLOPs, so this is a MEMORY win first — its value
# is the headroom it returns for a larger forward_batch_size.

RESPONSE_ONLY_LOGITS_ENV = "TORCHTUNE_RESPONSE_ONLY_LOGITS"


def response_only_logits_enabled() -> bool:
    """Whether ``TORCHTUNE_RESPONSE_ONLY_LOGITS=1`` is set (default OFF).

    Read at call time rather than import time so launchers and tests can toggle
    the gate without re-importing the module.

    Returns:
        bool: True when the response-only lm_head projection is opted in.
    """
    import os

    return os.environ.get(RESPONSE_ONLY_LOGITS_ENV, "0") == "1"


def gather_response_span(
    states: torch.Tensor,
    prompt_lengths: int | torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Select the response span from any ``[B, S, X]`` sequence-major tensor.

    This is exactly :func:`gather_response_logits` — the index math only touches
    the sequence axis, so it applies unchanged to pre-projection hidden states
    ``[B, S, H]``. It delegates rather than duplicating so the hidden-state slice
    and the logit slice can never drift apart.

    Args:
        states (torch.Tensor): Sequence-major tensor of shape ``[B, S, X]``.
        prompt_lengths (int | torch.Tensor): Scalar prompt length shared by every
            row, or a ``[B]`` tensor of per-row prompt lengths (the row-compacted
            ``compact_prompt_completion_batch`` layout).
        response_length (int): Number of response positions to select.

    Returns:
        torch.Tensor: ``[B, response_length, X]`` (scalar prompt lengths may yield
        fewer than ``response_length`` columns when the span runs off the end,
        matching the existing slice semantics).
    """
    return gather_response_logits(states, prompt_lengths, response_length)


def _supports_response_only_logits(model: torch.nn.Module) -> bool:
    """Whether ``model`` can project only the response span inside its forward.

    Resolves through FSDP1/DDP wrappers, which proxy unknown attribute lookups to
    the wrapped module.

    Args:
        model (torch.nn.Module): Possibly wrapped policy or reference model.

    Returns:
        bool: True when the model exposes a working
        ``supports_response_only_logits()`` probe that returns True.
    """
    probe = getattr(model, "supports_response_only_logits", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # pragma: no cover - defensive; never fail a forward here
        utils.get_logger().warning(
            "supports_response_only_logits() raised; falling back to full-width "
            "logits",
            exc_info=True,
        )
        return False


_RESPONSE_ONLY_LOGGED: set = set()


def _log_response_only_state(state: str) -> None:
    """Emit the ``response_only_logits = <state>`` marker once per state.

    Follows the convention CLAUDE.md documents for the other XPU fast-path gates
    (``xpu_flash|xpu_flex|varlen = engaged|requested-but-skipped|disabled``): a
    reader of the log must be able to tell an inert flag from an absent one. The
    dedupe is by state rather than a plain once-flag so a mid-run transition
    (e.g. the reference model advertising support when the policy does not) is
    still visible instead of being swallowed by the first call.

    Args:
        state (str): One of ``engaged``, ``requested-but-skipped``, ``disabled``.
    """
    if state in _RESPONSE_ONLY_LOGGED:
        return
    _RESPONSE_ONLY_LOGGED.add(state)
    if state == "disabled":
        # The default. Logging it at info on every run would be noise, and its
        # absence is not ambiguous (the other two states are what need proof).
        return
    utils.get_logger().info("response_only_logits = %s", state)


def response_only_logits_kwargs(
    model: torch.nn.Module,
    prompt_lengths: int | torch.Tensor,
    response_length: int,
) -> dict:
    """Forward kwargs that make ``model`` project only the response span.

    Returns an empty dict (so the caller's forward is unchanged) unless
    ``TORCHTUNE_RESPONSE_ONLY_LOGITS=1`` AND the model advertises support. Pair
    every call with :func:`finish_response_logits` on the returned logits.

    Args:
        model (torch.nn.Module): Policy or reference model (may be FSDP-wrapped).
        prompt_lengths (int | torch.Tensor): Scalar prompt length, or a ``[B]``
            tensor of per-row prompt lengths under row compaction.
        response_length (int): Number of response positions.

    Returns:
        dict: ``{}`` for the default full-width path, else the response-only
        forward kwargs.
    """
    if not response_only_logits_enabled():
        _log_response_only_state("disabled")
        return {}
    if not _supports_response_only_logits(model):
        # The flag is ON but the model cannot honor it, so every forward silently
        # takes the full-width path. Without this line an A/B reads exactly like
        # "the flag bought nothing" -- the same blind-vs-fail confusion the varlen
        # and xpu_flash gates already log their way out of.
        _log_response_only_state("requested-but-skipped")
        return {}
    _log_response_only_state("engaged")
    return {
        "response_prompt_lengths": prompt_lengths,
        "response_length": response_length,
    }


def finish_response_logits(
    logits: torch.Tensor,
    response_only_kwargs: dict,
    prompt_lengths: int | torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Reduce a forward's logits to the response span, if not already reduced.

    Args:
        logits (torch.Tensor): Model output — ``[B, S, V]`` on the default path,
            already ``[B, response_length, V]`` when ``response_only_kwargs`` is
            non-empty.
        response_only_kwargs (dict): The dict returned by
            :func:`response_only_logits_kwargs` for this same forward.
        prompt_lengths (int | torch.Tensor): As passed to
            :func:`response_only_logits_kwargs`.
        response_length (int): As passed to :func:`response_only_logits_kwargs`.

    Returns:
        torch.Tensor: ``[B, response_length, vocab_size]`` response logits. Both
        paths produce bit-identical values.
    """
    if response_only_kwargs:
        return logits
    return gather_response_logits(logits, prompt_lengths, response_length)


# NOTE: This is almost the same as torchtune.generation.generate, with a few changes necessary for GRPO.
# Namely:
#   1. The `return_logits` argument - we can optionally omit keeping track of logits during generation, which
#        drastically improves generation speed.
#   2. Stop token-based breaking now communicates across multiple devices in a distributed setting.
# NOTE: XPU/GRPO fork of torchtune.generation.generate (adds optional logit
# retention + distributed stop-token sync). Kept local to this Aurora fork; not
# planned for upstream.
@torch.no_grad()
def generate(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[list[int]] = None,
    rng: Optional[torch.Generator] = None,
    custom_generate_next_token: Optional[Callable] = None,
    return_logits: bool = True,
    stop_token_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape either [seq_length] or [bsz x seq_length].
        max_generated_tokens (int): number of tokens to be generated
        pad_id (int): token ID to use for padding, default 0.
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities,
            default None.
        stop_tokens (Optional[list[int]]): If specified, generation is stopped when any of these tokens are generated,
            default None.
        rng (Optional[torch.Generator]): random number generator, default None.
        custom_generate_next_token (Optional[Callable]): If specified, we'll use the
            ``custom_generate_next_token function``. This is generally only useful if
            you want to specify a ``torch.compile`` version of the generate next token for
            performance reasons. If None, we use the default :func:`generate_next_token`.
            Default is None.
        return_logits (bool): whether to return logits associated with the generated tokens, default True.
        stop_token_group (Optional[torch.distributed.ProcessGroup]): Process group used
            to synchronize distributed early stopping. Defaults to the world group.

    Note:
        This function has only been tested with decoder-only models.

    Examples:
        >>> model = torchtune.models.llama3.llama3_8b()
        >>> tokenizer = torchtune.models.llama3.llama3_tokenizer()
        >>> prompt = tokenizer.encode("Hi my name is")
        >>> rng.manual_seed(42)
        >>> output, logits = generate(model, torch.tensor(prompt), max_generated_tokens=100, pad_id=0)
        >>> print(tokenizer.decode(output[0].tolist()))
        Hi my name is Jeremy and I'm a friendly language model assistant!

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - tokens (torch.Tensor): tensor with the generated tokens,
                with shape ``[bsz x seq_len + num_generated_tokens]`` where ``num_generated_tokens``
                may be less than ``max_generated_tokens`` if ``stop_tokens`` are provided.
            - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                with shape ``[bsz x num_generated_tokens x vocab_size]``.
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    bsz, prompt_length = prompt.size()
    total_response_length = prompt_length + max_generated_tokens

    generated_tokens = prompt.clone()
    incremental_decoding = model.caches_are_enabled()

    # grab the correct max_seq_len to generate full causal masks/position ids
    # this is the model's max cache len if incremental decoding, or the sequence
    # length otherwise
    max_seq_len = (
        total_response_length
        if not incremental_decoding
        else model.decoder_max_cache_seq_len
    )

    padding_masks = generated_tokens != pad_id

    if not padding_masks.all():
        # we have padding in the prompt due to varying-length sequences in a batch
        # extend padding masks out to the correct seq len
        padding_masks = torch.nn.functional.pad(
            padding_masks, (0, max_generated_tokens), value=True
        )

        # generate the full causal mask for the whole padding mask with padding ignored
        masks = get_causal_mask_from_padding_mask(
            padding_masks, target_seq_len=max_seq_len
        )

        # right-shift position IDs to account for padding
        input_pos = get_position_ids_from_padding_mask(padding_masks)
    else:
        # just use a regular causal mask if there is no padding
        masks = torch.tril(
            torch.ones(
                total_response_length,
                max_seq_len,
                dtype=torch.bool,
                device=prompt.device,
            )
        ).unsqueeze(0)
        input_pos = torch.arange(
            0, total_response_length, device=generated_tokens.device
        ).unsqueeze(0)

    if incremental_decoding:
        # if KV-caches are enabled, we need a causal mask of shape [bsz, prompt_length, max_cache_len]
        # to match the key/value cache tensor shapes
        curr_masks = masks[:, :prompt_length]
    else:
        # otherwise the causal mask is shape [bsz, prompt_length, prompt_length] because key/value
        # tensors are of identical shape to the prompt
        curr_masks = masks[:, :prompt_length, :prompt_length]

    q = None
    if rng is not None:
        q = torch.empty(
            (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
        ).exponential_(1, generator=rng)
    tokens, generated_logits = generate_next_token(
        model,
        input_pos=input_pos[:, :prompt_length].squeeze(),
        mask=curr_masks,
        x=prompt,
        temperature=temperature,
        top_k=top_k,
        q=q,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    curr_pos = prompt_length

    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    stop_tokens = (
        torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype)
        if stop_tokens
        else None
    )

    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
    )

    # stop early if we reach a stop token in every seq
    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(
            tokens, stop_tokens, stop_token_reached
        )
        if stop_token_reached.all().item():
            return generated_tokens, generated_logits if return_logits else None

    world_size, rank = utils.get_world_size_and_rank()
    # For HSDP, use the shard group for early-stopping all_reduce.
    # The shard group contains ranks that share the same data (FSDP shards).
    # Using the world PG deadlocks when different replicate groups finish
    # generation at different times (XCCL can't mix PG operations).
    _stop_group = stop_token_group
    _stop_group_size = _stop_group.size() if _stop_group is not None else world_size
    for _ in (pbar := trange(max_generated_tokens - 1, leave=False, disable=rank > 0)):
        # update stop_token_mask if we reached a stop token in a previous step
        # by appending the logical not of stop_token_reached to the end of the mask
        # reshaped to be bsz first
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        # if incremental decoding is enabled, we can use the current position
        # otherwise, we take the whole sequence up to the current position
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos].contiguous()
            curr_masks = masks[:, curr_pos, None, :].contiguous()
        else:
            tokens = generated_tokens.clone()
            curr_input_pos = input_pos[:, : curr_pos + 1]
            curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        q = None
        if rng is not None:
            q = torch.empty(
                (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
            ).exponential_(1, generator=rng)
        tokens, logits = custom_generate_next_token(
            model,
            input_pos=curr_input_pos,
            x=tokens.clone(),
            mask=curr_masks,
            temperature=temperature,
            top_k=top_k,
            q=q,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        if return_logits:
            generated_logits = torch.cat([generated_logits, logits], dim=1)
        curr_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if world_size == 1:
                # Single device
                if stop_token_reached.all():
                    break
            else:
                all_done = stop_token_reached.all().int()
                torch.distributed.all_reduce(all_done, group=_stop_group)
                if all_done == _stop_group_size:
                    # Multiple devices
                    break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens *= stop_token_mask
        if return_logits:
            generated_logits *= stop_token_mask[:, -generated_logits.shape[1] :, None]

    return generated_tokens, generated_logits

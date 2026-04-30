# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Runtime sequence packing for GRPO training.

After generating variable-length responses, sequences are bin-packed into
fixed-length "packs" before the training forward/backward pass. This eliminates
padding waste (30-50% of compute at larger G) and produces fixed tensor shapes.

Uses block-diagonal attention masks (via ``create_block_causal_mask``) to prevent
cross-sequence attention within a pack.
"""

import logging
from typing import Optional

import torch


log = logging.getLogger(__name__)


def compute_actual_seq_lens(
    query_responses: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Compute actual (non-padding) sequence lengths for each row.

    Args:
        query_responses: [N, total_len] token IDs
        pad_id: padding token ID

    Returns:
        Tensor of shape [N] with actual lengths (last non-pad + 1).
    """
    # Find rightmost non-pad position per row
    non_pad = query_responses != pad_id  # [N, total_len]
    # Use argmax on reversed tensor to find last non-pad
    # reversed_first_nonpad gives index from end
    any_nonpad = non_pad.any(dim=1)
    # For each row, find the last True position
    indices = torch.arange(query_responses.shape[1], device=query_responses.device)
    # Multiply mask by indices, take max
    lengths = (non_pad * indices.unsqueeze(0)).max(dim=1).values + 1
    # Handle all-pad rows
    lengths[~any_nonpad] = 0
    return lengths


def greedy_bin_pack(
    seq_lens: torch.Tensor,
    pack_capacity: int,
) -> list[list[int]]:
    """
    Greedy first-fit-decreasing bin packing.

    Args:
        seq_lens: [N] actual sequence lengths
        pack_capacity: maximum tokens per pack

    Returns:
        List of bins, each bin is a list of sequence indices.
    """
    N = seq_lens.shape[0]
    # Sort by descending length for better packing
    sorted_indices = torch.argsort(seq_lens, descending=True)

    bins: list[list[int]] = []
    bin_remaining: list[int] = []

    for idx in sorted_indices.tolist():
        length = seq_lens[idx].item()
        if length == 0:
            continue
        if length > pack_capacity:
            log.warning(
                "Sequence %d has length %d > pack_capacity %d, placing alone",
                idx, length, pack_capacity,
            )
            bins.append([idx])
            bin_remaining.append(0)
            continue

        # First-fit
        placed = False
        for b in range(len(bins)):
            if bin_remaining[b] >= length:
                bins[b].append(idx)
                bin_remaining[b] -= length
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_remaining.append(pack_capacity - length)

    return bins


def pack_trajectory_for_training(
    query_responses: torch.Tensor,
    position_ids: torch.Tensor,
    pad_id: int,
    pack_seq_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor]:
    """
    Pack padded sequences into fewer, longer packed sequences for efficient
    model forward/backward.

    Takes the padded ``query_responses`` and ``position_ids`` from a GRPO trajectory,
    bin-packs the non-padding portions into packs of ``pack_seq_len``, and creates
    block-diagonal attention masks.

    Args:
        query_responses: [N, total_len] padded token IDs
        position_ids: [N, total_len] position IDs
        pad_id: padding token ID
        pack_seq_len: target pack length. Defaults to total_len (same as input).

    Returns:
        Tuple of:
        - packed_tokens: [num_packs, pack_seq_len] packed token IDs
        - packed_positions: [num_packs, pack_seq_len] packed position IDs
        - packed_masks: [num_packs, pack_seq_len, pack_seq_len] block-diagonal masks
        - bins: list of lists of original sequence indices per pack
        - actual_lens: [N] actual sequence lengths (for unpacking)
    """
    N, total_len = query_responses.shape
    if pack_seq_len is None:
        pack_seq_len = total_len
    device = query_responses.device

    # Step 1: compute actual lengths
    actual_lens = compute_actual_seq_lens(query_responses, pad_id)

    # Step 2: bin-pack
    bins = greedy_bin_pack(actual_lens, pack_seq_len)
    num_packs = len(bins)

    padding_before = N * total_len
    packed_total = num_packs * pack_seq_len
    actual_total = actual_lens.sum().item()
    log.info(
        "Packing: %d seqs -> %d packs (%.0f%% padding before, %.0f%% after, %.0f%% reduction)",
        N, num_packs,
        100.0 * (1 - actual_total / padding_before),
        100.0 * (1 - actual_total / packed_total) if packed_total > 0 else 0,
        100.0 * (1 - packed_total / padding_before) if padding_before > 0 else 0,
    )

    # Step 3: build packed tensors
    packed_tokens = query_responses.new_full((num_packs, pack_seq_len), pad_id)
    packed_positions = position_ids.new_zeros((num_packs, pack_seq_len))
    # Track sequence lengths within each pack for mask creation
    pack_seq_lens_list: list[torch.Tensor] = []

    for pack_idx, bin_indices in enumerate(bins):
        offset = 0
        seq_lens_in_pack = []
        for seq_idx in bin_indices:
            slen = actual_lens[seq_idx].item()
            if slen == 0:
                continue
            end = offset + slen
            packed_tokens[pack_idx, offset:end] = query_responses[seq_idx, :slen]
            packed_positions[pack_idx, offset:end] = position_ids[seq_idx, :slen]
            seq_lens_in_pack.append(torch.tensor(slen, device=device))
            offset = end
        if seq_lens_in_pack:
            pack_seq_lens_list.append(torch.stack(seq_lens_in_pack))
        else:
            pack_seq_lens_list.append(torch.tensor([0], device=device))

    # Step 4: create block-diagonal attention masks, padded to pack_seq_len.
    # Can't use create_block_causal_mask directly because packs may have
    # different total content lengths (torch.stack would fail).
    packed_masks = torch.zeros(
        (num_packs, pack_seq_len, pack_seq_len), dtype=torch.bool, device=device
    )
    for pack_idx, sl_tensor in enumerate(pack_seq_lens_list):
        # Build block-diagonal causal mask for this pack's sequences
        blocks = [
            torch.tril(
                torch.ones(s, s, dtype=torch.bool, device=device)
            )
            for s in sl_tensor
            if s > 0
        ]
        if blocks:
            block_mask = torch.block_diag(*blocks)
            content_len = block_mask.shape[0]
            packed_masks[pack_idx, :content_len, :content_len] = block_mask

    return packed_tokens, packed_positions, packed_masks, bins, actual_lens


def unpack_tensor(
    packed: torch.Tensor,
    bins: list[list[int]],
    actual_lens: torch.Tensor,
    num_sequences: int,
    total_len: int,
) -> torch.Tensor:
    """
    Unpack a packed tensor back to per-sequence padded format.
    Uses scatter-based approach to maintain autograd compatibility.

    Args:
        packed: [num_packs, pack_seq_len, ...] packed tensor from model
        bins: list of lists of original sequence indices per pack
        actual_lens: [N] actual sequence lengths
        num_sequences: N, number of original sequences
        total_len: original padded sequence length

    Returns:
        Tensor of shape [N, total_len, ...] with values placed back
        in their original positions (padding positions filled with 0).
    """
    extra_dims = packed.shape[2:]
    device = packed.device

    # Build flat index mapping: for each (pack_idx, offset) → (seq_idx, pos)
    # Collect all slices and concatenate (autograd-safe)
    slices = [None] * num_sequences
    for pack_idx, bin_indices in enumerate(bins):
        offset = 0
        for seq_idx in bin_indices:
            slen = actual_lens[seq_idx].item()
            if slen == 0:
                continue
            # Slice is a view into packed — autograd tracks this
            slices[seq_idx] = packed[pack_idx, offset:offset + slen]
            offset += slen

    # Build output: pad each slice to total_len
    result_parts = []
    for seq_idx in range(num_sequences):
        if slices[seq_idx] is not None:
            s = slices[seq_idx]  # [slen, ...]
            slen = s.shape[0]
            if slen < total_len:
                pad_shape = (total_len - slen,) + extra_dims
                padding = torch.zeros(pad_shape, device=device, dtype=s.dtype)
                result_parts.append(torch.cat([s, padding], dim=0))
            else:
                result_parts.append(s[:total_len])
        else:
            result_parts.append(
                torch.zeros((total_len,) + extra_dims, device=device, dtype=packed.dtype)
            )

    return torch.stack(result_parts)  # [N, total_len, ...]

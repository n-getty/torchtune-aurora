# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
from torch import Tensor

from torchtune.utils._device import has_cuda_capability
from torchtune.utils._logging import get_logger, log_once

_log: logging.Logger = get_logger()

# Configuration of MoE
# use grouped_mm in MoE or for loop for experts computation.
use_grouped_mm = True


def should_use_grouped_mm():
    if use_grouped_mm and not has_cuda_capability(9, 0):
        log_once(
            _log,
            "Failed to use grouped mm, which is only supported on SM90 or later",
            level=logging.DEBUG,
        )
        return False
    return use_grouped_mm


def _permute(
    routed_input: Tensor,
    num_tokens_per_expert_group: Tensor,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reorder dispatched tokens from source-rank-major to local-expert-major order.

    After All-to-All dispatch, tokens arrive in source-rank order:
        [ep_r0_exp0_toks, ep_r0_exp1_toks, ..., ep_r1_exp0_toks, ...]

    GroupedExperts.forward() expects tokens in expert-major order:
        [all_exp0_toks, all_exp1_toks, ...]

    This implementation uses pure torch ops (no Triton) and is compatible with
    XPU's loop-based expert path (no grouped_mm alignment padding required).

    Args:
        routed_input: Dispatched tokens, shape ``(total_tokens, dim)``.
        num_tokens_per_expert_group: Token counts per (source_rank, local_expert) pair,
            shape ``(ep_degree * num_local_experts,)`` in source-rank-major order:
            ``[ep0_exp0, ep0_exp1, ..., ep1_exp0, ...]``.
        ep_degree: Number of EP ranks.
        num_local_experts: Number of experts owned by this rank.

    Returns:
        Tuple of:
            - permuted_input: Tokens reordered to expert-major order.
            - local_ntpe: Token count per local expert, shape ``(num_local_experts,)``.
            - perm: Permutation index tensor used to reorder; pass to ``_unpermute``.
    """
    ntpe_matrix = num_tokens_per_expert_group.view(ep_degree, num_local_experts)
    local_ntpe = ntpe_matrix.sum(dim=0)  # (num_local_experts,)

    # Cumulative offsets into the flat routed_input buffer.
    # Convert to int64 first to avoid float32 rounding: histc/bincount values may be
    # stored as float32 (e.g. 44.9999 for a true count of 45) causing arange
    # to produce the wrong number of indices → backward shape mismatch.
    ntpe_int = num_tokens_per_expert_group.round().to(torch.long)
    offsets = torch.zeros(
        ep_degree * num_local_experts + 1, dtype=torch.long, device=routed_input.device
    )
    offsets[1:] = torch.cumsum(ntpe_int, dim=0)
    # Recompute local_ntpe from integer counts so downstream .to(int64) is exact
    local_ntpe = ntpe_int.view(ep_degree, num_local_experts).sum(dim=0)

    # Build permutation: collect token indices in expert-major order
    ntpe_matrix_int = ntpe_int.view(ep_degree, num_local_experts)
    indices: list[Tensor] = []
    for local_exp in range(num_local_experts):
        for ep_r in range(ep_degree):
            flat_idx = ep_r * num_local_experts + local_exp
            start = offsets[flat_idx].item()
            count = int(ntpe_matrix_int[ep_r, local_exp].item())  # exact int
            if count > 0:
                indices.append(
                    torch.arange(start, start + count, device=routed_input.device)
                )

    if indices:
        perm = torch.cat(indices)
    else:
        perm = torch.zeros(0, dtype=torch.long, device=routed_input.device)

    return routed_input[perm], local_ntpe, perm


def _unpermute(routed_output: Tensor, perm: Tensor, total_dispatched: int) -> Tensor:
    """Reverse ``_permute``: scatter expert outputs back to source-rank order.

    Uses argsort-based inverse permutation (fully differentiable) instead of
    index assignment (``out[perm] = src``), which loses gradients because the
    in-place write on a non-leaf tensor breaks autograd.

    Args:
        routed_output: Expert outputs in expert-major order,
            shape ``(total_dispatched, dim)``.
        perm: Permutation indices returned by ``_permute``, shape ``(total_dispatched,)``.
            ``perm[i] = j`` means token at position ``i`` (expert-major) originally
            came from position ``j`` (source-rank-major).
        total_dispatched: Total number of dispatched tokens (= ``perm.shape[0]``).

    Returns:
        Tensor in source-rank-major order, shape ``(total_dispatched, dim)``.
    """
    # inv_perm[j] = i where perm[i] = j → out[j] = routed_output[inv_perm[j]]
    # i.e. out = routed_output[inv_perm]  — pure indexing, autograd-safe.
    inv_perm = torch.argsort(perm)
    return routed_output[: perm.shape[0]][inv_perm]

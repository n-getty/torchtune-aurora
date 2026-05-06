# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# AsymAdamWXPU — AdamW where the FP32 master + moments live only on a subset
# of "spare" XPU ranks, off the trainer/vLLM tiles. Each step:
#   1. gather grads 12->4 onto spare ranks via dist.all_to_all_single
#   2. AdamW math runs only on spare ranks
#   3. scatter params 4->12 back onto trainer ranks
#
# The sharding the wrapped FSDP2 module uses (12-way) is left untouched —
# we maintain a parallel 4-way sharding of FP32 state that lives on the
# spare ranks. Trainer ranks carry no optimizer state.
#
# This is the Phase B counterpart to the Phase A vLLM-rank-subset plumbing
# in vllm_backend.py / weight_sync.py / the recipe. See
# ~/.claude/plans/virtual-orbiting-kite.md.

import logging
from typing import Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist

from torchtune.dev.asym_optim.redistribute import (
    build_a2a_splits,
    compute_overlap_matrix,
)

log = logging.getLogger(__name__)


def _flat_local(p: torch.Tensor) -> torch.Tensor:
    """Return the flat local shard of an FSDP2 DTensor (or the param itself
    when not a DTensor). Always 1-D, contiguous, on the param's device."""
    if hasattr(p, "to_local"):
        local = p.to_local()
    else:
        local = p
    return local.reshape(-1).contiguous()


class AsymAdamWXPU(torch.optim.Optimizer):
    """AdamW where FP32 master + moments live only on a designated set of
    spare ranks (XPU HBM). Trainer ranks call ``step()`` to participate in
    the 12->4 grad gather and 4->12 param scatter, but hold no state.

    Args:
        params: iterable of FSDP2-sharded params (BF16 on XPU).
        spare_ranks: global ranks that own the FP32 state (e.g. [8,9,10,11]).
        optim_pg: a process group containing every rank in
            ``train_ranks ∪ spare_ranks`` (== world for the colocate path).
            ``all_to_all_single`` runs on this PG.
        lr, betas, eps, weight_decay: standard AdamW knobs.

    Notes:
        * Only XCCL backend on XPU is exercised; the CPU smoke uses gloo.
        * FP32 master/exp_avg/exp_avg_sq are lazily allocated on first step
          (we need each param's flat shard size, which is known after the
          first grad pass).
        * Pad/strip math is delegated to ``compute_overlap_matrix`` /
          ``build_a2a_splits``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        spare_ranks: Sequence[int],
        optim_pg: Optional["dist.ProcessGroup"] = None,
        lr: float = 1e-5,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self._spare_ranks: List[int] = list(spare_ranks)
        if not self._spare_ranks:
            raise ValueError("spare_ranks must be non-empty")
        self._optim_pg = optim_pg
        self._world_size = (
            dist.get_world_size(optim_pg)
            if optim_pg is not None and dist.is_initialized()
            else 1
        )
        self._my_rank = (
            dist.get_rank(optim_pg)
            if optim_pg is not None and dist.is_initialized()
            else 0
        )
        # Train ranks = optim_pg ranks minus spare ranks.
        if optim_pg is not None and dist.is_initialized():
            _all = dist.get_process_group_ranks(optim_pg)
        else:
            _all = list(range(max(self._spare_ranks) + 1))
        self._all_pg_ranks: List[int] = list(_all)
        self._train_ranks: List[int] = [
            r for r in _all if r not in self._spare_ranks
        ]
        self._n_src = len(self._train_ranks)
        self._n_dst = len(self._spare_ranks)
        self._is_spare = self._my_rank in self._spare_ranks
        self._initialized = False
        self._first_step_seed_done = False
        self._step_counter = 0

    def _lazy_init_state(self) -> None:
        if self._initialized:
            return
        # Walk every param. Each spare rank holds a per-param fp32 master/moments
        # vector of size ``dst_split_size`` (the per-spare landing buffer for
        # the 12->4 gather). Trainer ranks hold nothing.
        for group in self.param_groups:
            for p in group["params"]:
                local = _flat_local(p)
                src_shard_size = local.numel()
                _, dst_split_size = compute_overlap_matrix(
                    self._n_src, src_shard_size, self._n_dst
                )
                state = self.state[p]
                state["src_shard_size"] = src_shard_size
                state["dst_split_size"] = dst_split_size
                # Index of THIS spare rank inside spare_ranks (None on trainers).
                if self._is_spare:
                    state["fp32_master"] = torch.zeros(
                        dst_split_size, dtype=torch.float32, device=p.device
                    )
                    state["exp_avg"] = torch.zeros_like(state["fp32_master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["fp32_master"])
                    # Seed master from the current bf16 param shard via the
                    # first step's gather (avoids needing a full_tensor() here).
                    state["seeded"] = False
        self._initialized = True

    def _redistribute(
        self,
        flat: torch.Tensor,
        dst_buf: torch.Tensor,
        src_shard_size: int,
        direction: str,
    ) -> None:
        """Run one all_to_all_single in ``direction`` ('gather' or 'scatter').

        Trainer ranks pass ``flat`` (12-shard fragment) as input on gather, and
        receive into ``flat`` on scatter. Spare ranks pass ``dst_buf`` (4-shard
        landing buffer) as output on gather, and as input on scatter.
        """
        overlap, _ = compute_overlap_matrix(
            self._n_src, src_shard_size, self._n_dst
        )
        in_splits, out_splits = build_a2a_splits(
            overlap=overlap,
            pg_ranks=self._all_pg_ranks,
            train_ranks=self._train_ranks,
            spare_ranks=self._spare_ranks,
            my_rank=self._my_rank,
            direction=direction,
        )
        if direction == "gather":
            input_t = flat if self._my_rank in self._train_ranks else flat.new_empty(0)
            output_t = dst_buf if self._is_spare else flat.new_empty(0)
        else:
            input_t = dst_buf if self._is_spare else flat.new_empty(0)
            output_t = flat if self._my_rank in self._train_ranks else flat.new_empty(0)
        dist.all_to_all_single(
            output_t, input_t, out_splits, in_splits, group=self._optim_pg,
        )

    def _writeback_local(self, p: torch.Tensor, recv: torch.Tensor) -> None:
        """Write a flat 1-D recv tensor back into p's local FSDP2 shard."""
        if hasattr(p, "to_local"):
            local = p.to_local()
        else:
            local = p
        local.copy_(recv.view_as(local))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._lazy_init_state()
        self._step_counter += 1

        # On the very first step we also need to seed each spare rank's FP32
        # master from the current bf16 params. That is an extra collective
        # per param, so we batch it as a separate pre-pass to keep the
        # per-step hot loop simple.
        if not self._first_step_seed_done:
            self._seed_masters()
            self._first_step_seed_done = True

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            bias_c1 = 1.0 - beta1 ** self._step_counter
            bias_c2 = 1.0 - beta2 ** self._step_counter
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                src_shard_size = state["src_shard_size"]
                dst_split_size = state["dst_split_size"]

                # Gather grad (12 -> 4). Every rank participates.
                grad_local = _flat_local(p.grad)
                grad_dst = (
                    torch.zeros(dst_split_size, dtype=grad_local.dtype, device=p.device)
                    if self._is_spare
                    else grad_local.new_empty(0)
                )
                self._redistribute(grad_local, grad_dst, src_shard_size, "gather")

                if self._is_spare:
                    g32 = grad_dst.to(torch.float32)
                    if wd != 0.0:
                        state["fp32_master"].mul_(1.0 - lr * wd)
                    state["exp_avg"].mul_(beta1).add_(g32, alpha=1.0 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
                    m_hat = state["exp_avg"] / bias_c1
                    v_hat = state["exp_avg_sq"] / bias_c2
                    update = m_hat / (v_hat.sqrt() + eps)
                    state["fp32_master"].add_(update, alpha=-lr)
                    new_param_dst = state["fp32_master"].to(p.dtype)
                else:
                    new_param_dst = grad_local.new_empty(0)

                # Scatter new param (4 -> 12). Every rank participates.
                recv_buf = (
                    torch.zeros(src_shard_size, dtype=p.dtype, device=p.device)
                    if self._my_rank in self._train_ranks
                    else new_param_dst.new_empty(0)
                )
                self._redistribute(recv_buf, new_param_dst, src_shard_size, "scatter")
                if self._my_rank in self._train_ranks:
                    self._writeback_local(p, recv_buf)

        return loss

    def _seed_masters(self) -> None:
        """One-time gather of bf16 params -> spare-rank fp32 master."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                src_shard_size = state["src_shard_size"]
                dst_split_size = state["dst_split_size"]
                param_local = _flat_local(p)
                param_dst = (
                    torch.zeros(
                        dst_split_size, dtype=param_local.dtype, device=p.device
                    )
                    if self._is_spare
                    else param_local.new_empty(0)
                )
                self._redistribute(param_local, param_dst, src_shard_size, "gather")
                if self._is_spare:
                    state["fp32_master"].copy_(param_dst.to(torch.float32))
                    state["seeded"] = True

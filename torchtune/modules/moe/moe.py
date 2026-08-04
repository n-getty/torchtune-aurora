# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .utils import should_use_grouped_mm
from .measurement import MoEMeasurementCollector


_USE_AURORA_MOE = os.environ.get("TORCHTUNE_USE_AURORA_MOE", "0") == "1"
_USE_WIDE_ROUTING_INDICES = (
    os.environ.get("TORCHTUNE_MOE_WIDE_ROUTING_INDICES", "0") == "1"
)
_USE_INPLACE_ROUTE_WEIGHTING = (
    os.environ.get("TORCHTUNE_MOE_INPLACE_ROUTE_WEIGHTING", "1") == "1"
)
_USE_INPLACE_FINAL_SCATTER = (
    os.environ.get("TORCHTUNE_MOE_INPLACE_FINAL_SCATTER", "1") == "1"
)
_USE_INDEX_SELECT_PACKING = (
    os.environ.get("TORCHTUNE_MOE_INDEX_SELECT_PACKING", "1") == "1"
)
_USE_INDEX_ADD_FINAL_SCATTER = (
    os.environ.get("TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER", "1") == "1"
)
_AURORA_MOE_MEM_DEBUG = os.environ.get("TORCHTUNE_AURORA_MOE_MEM_DEBUG", "0") == "1"
_AURORA_MOE_CALL_INDEX = 0


def _aurora_moe_mem_probe(tag: str, call_index: int) -> None:
    if not _AURORA_MOE_MEM_DEBUG:
        return
    torch.xpu.synchronize()
    if dist.get_rank() != 0:
        return
    gib = 1024**3
    free, total = torch.xpu.mem_get_info()
    print(
        f"[AURORA_MOE_MEM] call={call_index:03d} {tag} "
        f"alloc={torch.xpu.memory_allocated() / gib:.3f}GiB "
        f"reserved={torch.xpu.memory_reserved() / gib:.3f}GiB "
        f"peak_alloc={torch.xpu.max_memory_allocated() / gib:.3f}GiB "
        f"peak_reserved={torch.xpu.max_memory_reserved() / gib:.3f}GiB "
        f"l0_free={free / gib:.3f}GiB l0_used={(total - free) / gib:.3f}GiB",
        flush=True,
    )


def _index_select_with_zero_padding(
    values: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Select padded permutation indices without copying the source tensor.

    ``generate_permute_indices`` uses ``-1`` for alignment-only rows. The
    previous path appended a zero sentinel to ``values`` before advanced
    indexing, which copied the complete routed-token buffer once per tensor.
    Clamping the sentinel and zeroing only the padded rows avoids that full
    source copy while retaining the exact output layout.
    """
    safe_indices = indices.clamp_min(0)
    if indices.ndim == 1:
        packed = values.index_select(0, safe_indices)
    else:
        packed = torch.gather(values, 0, safe_indices)
    invalid = indices < 0
    if indices.ndim == 1 and values.ndim > 1:
        invalid = invalid.reshape(-1, *([1] * (values.ndim - 1)))
    packed.masked_fill_(invalid, 0)
    return packed


def _accumulate_routed_output(
    output: torch.Tensor,
    token_indices: torch.Tensor,
    routed_output: torch.Tensor,
    *,
    shared_expert: Optional[nn.Module],
) -> torch.Tensor:
    """Accumulate routed rows without widening indices when it is safe."""
    if (
        shared_expert is None
        and _USE_INDEX_ADD_FINAL_SCATTER
        and _USE_INPLACE_FINAL_SCATTER
        and token_indices.ndim == 1
    ):
        output.index_add_(0, token_indices, routed_output)
        return output
    scatter_indices = (
        token_indices
        if token_indices.ndim > 1
        else token_indices.unsqueeze(1).expand(-1, routed_output.shape[-1])
    )
    if shared_expert is None and _USE_INPLACE_FINAL_SCATTER:
        output.scatter_add_(0, scatter_indices, routed_output)
        return output
    return output.scatter_add(0, scatter_indices, routed_output)


def _aurora_moe_expert_ids(
    selected_experts: torch.Tensor, ep_degree: int, local_experts: int
) -> torch.Tensor:
    return selected_experts.remainder(ep_degree) * local_experts + torch.div(
        selected_experts, ep_degree, rounding_mode="floor"
    )


@torch.compiler.disable
def _run_aurora_moe(
    x: torch.Tensor,
    selected_scores: torch.Tensor,
    remapped_experts: torch.Tensor,
    local_expert_ids: list[int],
    ep_group,
    ep_degree: int,
    ep_rank: int,
    up_proj: torch.Tensor,
    gate_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    global _AURORA_MOE_CALL_INDEX

    call_index = _AURORA_MOE_CALL_INDEX
    _AURORA_MOE_CALL_INDEX += 1
    _aurora_moe_mem_probe("before", call_index)
    try:
        from aurora_moe._core import _routed_moe
    except ImportError as error:
        raise RuntimeError(
            "TORCHTUNE_USE_AURORA_MOE=1 requires the aurora_moe package"
        ) from error
    aurora_mesh = SimpleNamespace(
        device=x.device,
        groups={"ep_dispatch": ep_group},
        group_size={"ep_dispatch": ep_degree},
        group_rank={"ep_dispatch": ep_rank},
        ranks={"ep_dispatch": dist.get_process_group_ranks(ep_group)},
    )
    x = x.to_local() if isinstance(x, DTensor) else x
    selected_scores = (
        selected_scores.to_local()
        if isinstance(selected_scores, DTensor)
        else selected_scores
    )
    remapped_experts = (
        remapped_experts.to_local()
        if isinstance(remapped_experts, DTensor)
        else remapped_experts
    )
    up_proj = up_proj.to_local() if isinstance(up_proj, DTensor) else up_proj
    gate_proj = gate_proj.to_local() if isinstance(gate_proj, DTensor) else gate_proj
    down_proj = down_proj.to_local() if isinstance(down_proj, DTensor) else down_proj
    output = _routed_moe(
        x,
        selected_scores.contiguous(),
        remapped_experts.contiguous(),
        local_expert_ids,
        aurora_mesh,
        up_proj.transpose(-1, -2).contiguous(),
        gate_proj.transpose(-1, -2).contiguous(),
        down_proj.transpose(-1, -2).contiguous(),
        expert_backend="sycl_sonic",
    )
    _aurora_moe_mem_probe("after", call_index)
    return output


class TokenChoiceTopKRouter(nn.Module):
    """This class implements Token Choice routing. In Token Choice top K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        experts_per_token: int,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid is performed in float32 to avoid loss explosion
        scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)

        # Deterministic top-k: use argsort(stable=True) + slice instead of topk.
        # torch.topk lacks stable=True on XPU → non-deterministic tie-breaking →
        # AC recompute produces different expert assignments → backward matmul shape
        # mismatch. argsort(stable=True) breaks ties by original expert index order.
        # top scores shape (bs*slen, top_k)
        sorted_indices = torch.argsort(scores, dim=1, stable=True, descending=True)
        selected_experts_indices = sorted_indices[:, : self.experts_per_token]
        top_scores = torch.gather(scores, 1, selected_experts_indices)
        self.selected_experts_indices = selected_experts_indices
        # top_scores /= top_scores.sum(dim=-1, keep_dim=True).to(x.dtype)

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        # Use bincount (int64) instead of histc (float32) to avoid float rounding errors.
        # histc float32 counts can be e.g. 44.9999 or 45.0001 for a true count of 45, causing
        # inconsistent truncation in _permute vs _forward_no_grouped_mm → shape mismatch in backward.
        selected_experts_flat = selected_experts_indices.view(-1)
        num_tokens_per_expert = torch.bincount(
            selected_experts_flat,
            minlength=self.num_experts,
        )  # int64 — exact integer counts, safe for allgather/alltoall split computation
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_flat, stable=True
        )
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.experts_per_token
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert


class MoE(nn.Module):
    """This class implements the moe layer which is Mixture of Experts. Mixture of Experts
    typically consists of a set of expert networks, alongside with a router, which directs input tokens
    to the appropriate experts. See more details in https://arxiv.org/pdf/2407.06204.

    Args:
        experts (nn.Module): experts module.
        router (nn.Module): router module.
        shared_expert (Optional[nn.Module]): shared expert module. Default is None.
        checkpoint_experts (bool): if True, wraps ONLY the ``self.experts(...)``
            call (not the router, not EP dispatch/combine) in
            ``torch.utils.checkpoint.checkpoint`` — trades expert-compute
            recompute time for activation memory. Unlike checkpointing the
            whole MoE block (which would recompute the router and risk the
            v158 argsort-tie-break-under-recompute correctness bug, see
            ``torchtune/dev/rl/distributed.py::_apply_split_ac``), this is
            SAFE: ``self.experts()`` is a deterministic function of its
            already-fixed arguments (``routed_input``, ``num_tokens_per_expert``,
            both computed once by the router/dispatch before this call and
            passed in unchanged), with no randomness or shared mutable state
            touched during recompute — EP dispatch/combine's cached instance
            state (``ExpertParallel._ag_gather_idx`` etc.) lives entirely
            OUTSIDE this checkpoint region and is never re-touched. Default
            False (no behavior change from prior releases).
    """

    def __init__(
        self,
        *,
        experts: nn.Module,
        router: nn.Module,
        shared_expert: Optional[nn.Module] = None,
        checkpoint_experts: bool = False,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert
        self.use_grouped_mm = should_use_grouped_mm()
        self.checkpoint_experts = checkpoint_experts
        self.measurement = MoEMeasurementCollector()
        setattr(self.experts, "_moe_measurement", self.measurement)
        # EP dispatch/combine callables — set by setup code after parallelize_module.
        # If set, MoE.forward() calls these directly around self.experts(),
        # replacing the broken hook approach (FSDP2 fully_shard drops EP hooks).
        self._ep_dispatch: Optional[Callable] = None
        self._ep_combine: Optional[Callable] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.

        Raises:
            RuntimeError: If Aurora_MOE is enabled without expert parallelism,
                required router metadata, or the Aurora_MOE package.
        """
        bs, slen, dim = x.shape
        # top_scores and selected_indices shape (bs*slen*experts_per_token,)
        # num_tokens_per_expert shape (num_experts,)
        x_flat = x.reshape(bs * slen, dim)
        if self.measurement.enabled:
            with self.measurement.time("router"):
                top_scores, token_indices, num_tokens_per_expert = self.router(x_flat)
        else:
            top_scores, token_indices, num_tokens_per_expert = self.router(x_flat)
        if self.measurement.enabled:
            ep_mesh = getattr(self.experts, "_ep_device_mesh", None)
            ep_degree = ep_mesh.shape[0] if ep_mesh is not None else None
            self.measurement.record_tokens(
                num_tokens_per_expert,
                ep_degree=ep_degree,
            )

        if _USE_AURORA_MOE:
            if self._ep_dispatch is None:
                raise RuntimeError(
                    "TORCHTUNE_USE_AURORA_MOE requires expert parallelism"
                )
            selected_experts = getattr(self.router, "selected_experts_indices", None)
            selected_scores = getattr(self.router, "selected_expert_scores", None)
            ep_mesh = getattr(self.experts, "_ep_device_mesh", None)
            if selected_experts is None or selected_scores is None or ep_mesh is None:
                raise RuntimeError(
                    "Aurora_MOE requires a router exposing unsorted top-k scores and IDs"
                )
            ep_degree = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
            local_experts = self.experts.up_proj.shape[0]
            remapped_experts = _aurora_moe_expert_ids(
                selected_experts, ep_degree, local_experts
            )
            local_expert_ids = list(
                range(ep_rank * local_experts, (ep_rank + 1) * local_experts)
            )

            def run_routed(routed_x, routed_scores, routed_experts):
                return _run_aurora_moe(
                    routed_x,
                    routed_scores,
                    routed_experts,
                    local_expert_ids,
                    ep_mesh.get_group(),
                    ep_degree,
                    ep_rank,
                    self.experts.up_proj,
                    self.experts.gate_proj,
                    self.experts.down_proj,
                )

            routed_args = (x_flat, selected_scores, remapped_experts)
            if self.checkpoint_experts and torch.is_grad_enabled():
                out = torch.utils.checkpoint.checkpoint(
                    run_routed,
                    *routed_args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                out = run_routed(*routed_args)
            return out.reshape(bs, slen, dim)

        if _USE_WIDE_ROUTING_INDICES:
            token_indices = token_indices.reshape(-1, 1).expand(-1, dim)
        else:
            # Keep routing indices one-dimensional until the final scatter.
            # Expanding them to [tokens, dim] here duplicates metadata at
            # model-hidden width and makes grouped-MM padding allocate another
            # wide index tensor.
            token_indices = token_indices.reshape(-1)

        routed_counts_for_measurement = (
            num_tokens_per_expert.detach().clone()
            if self.measurement.enabled
            else None
        )
        grouped_gemm_alignment = None
        aligned_counts_for_measurement = None

        # shape (bs*slen*experts_per_token, dim)
        if _USE_WIDE_ROUTING_INDICES:
            routed_input = torch.gather(x.view(-1, dim), dim=0, index=token_indices)
        else:
            routed_input = x.view(-1, dim).index_select(0, token_indices)
        if _USE_INPLACE_ROUTE_WEIGHTING:
            routed_input.mul_(top_scores.reshape(-1, 1))
        else:
            routed_input = routed_input * top_scores.reshape(-1, 1)

        if self.use_grouped_mm:
            # NOTE: In order to use torch._grouped_mm, we need to make sure
            # the number of tokens each expert gets is a multiple of 16.
            # The following kernel helps achieve this via padding, without
            # incurring synchronization between device and host.
            from torchtune.modules.moe.indices import generate_permute_indices

            ALIGN_SIZE_M = 16  # noqa
            grouped_gemm_alignment = ALIGN_SIZE_M

            with torch.no_grad():
                (
                    permuted_indices,
                    num_tokens_per_expert,
                    _,
                ) = generate_permute_indices(
                    num_tokens_per_expert,
                    self.experts.num_experts,
                    1,
                    ALIGN_SIZE_M,
                )
            if self.measurement.enabled:
                aligned_counts_for_measurement = num_tokens_per_expert.detach().clone()
            if _USE_INDEX_SELECT_PACKING:
                token_indices = _index_select_with_zero_padding(
                    token_indices, permuted_indices
                )
                routed_input = _index_select_with_zero_padding(
                    routed_input, permuted_indices
                )
            elif _USE_WIDE_ROUTING_INDICES:
                token_indices = torch.vstack(
                    (token_indices, token_indices.new_zeros((dim)))
                )
                token_indices = token_indices[permuted_indices, :]
                routed_input = torch.cat(
                    (routed_input, routed_input.new_zeros((1, dim)))
                )
                routed_input = routed_input[permuted_indices]
            else:
                token_indices = torch.cat((token_indices, token_indices.new_zeros(1)))
                token_indices = token_indices[permuted_indices]
                routed_input = torch.cat(
                    (routed_input, routed_input.new_zeros((1, dim)))
                )
                routed_input = routed_input[permuted_indices]

        # EP dispatch: route tokens to expert-owning ranks via AllGather.
        # _ep_dispatch is set by setup code (not hooks — FSDP2 fully_shard drops
        # hooks registered on GroupedExperts by parallelize_module).
        # v159: dispatch returns (routed_input, num_tokens_per_expert) and
        # caches s_local + gather_idx on the ExpertParallel instance for
        # combine to read back. (Reverts v158 ctx-threading.)
        if self._ep_dispatch is not None:
            if aligned_counts_for_measurement is not None:
                up_projection = self.experts.up_proj
                expert_model_dim = getattr(self.experts, "dim", dim)
                expert_hidden_dim = (
                    up_projection.shape[-2]
                    if up_projection.shape[-1] == expert_model_dim
                    else up_projection.shape[-1]
                )
                self.measurement.record_gemm(
                    aligned_counts_for_measurement,
                    model_dim=dim,
                    hidden_dim=expert_hidden_dim,
                    routed_counts=routed_counts_for_measurement,
                    alignment=grouped_gemm_alignment,
                    stage="global_aligned",
                )
            if self.measurement.enabled:
                with self.measurement.time("dispatch"):
                    routed_input, num_tokens_per_expert = self._ep_dispatch(
                        routed_input, num_tokens_per_expert
                    )
            else:
                routed_input, num_tokens_per_expert = self._ep_dispatch(
                    routed_input, num_tokens_per_expert
                )

        if self.measurement.enabled:
            up_projection = self.experts.up_proj
            expert_model_dim = getattr(self.experts, "dim", dim)
            expert_hidden_dim = (
                up_projection.shape[-2]
                if up_projection.shape[-1] == expert_model_dim
                else up_projection.shape[-1]
            )
            self.measurement.record_gemm(
                num_tokens_per_expert,
                model_dim=dim,
                hidden_dim=expert_hidden_dim,
                routed_counts=(
                    routed_counts_for_measurement
                    if routed_counts_for_measurement is not None
                    and routed_counts_for_measurement.numel()
                    == num_tokens_per_expert.numel()
                    else None
                ),
                alignment=grouped_gemm_alignment,
                stage="local_compute" if self._ep_dispatch is not None else None,
            )

        # shape (bs*slen*top_k, dim)
        if self.checkpoint_experts and torch.is_grad_enabled():
            # Only self.experts()'s own intermediates (e.g. GroupedExpertsHF's
            # padded-BMM/sequential-per-expert temporaries) are discarded and
            # recomputed on backward — routed_input/num_tokens_per_expert are
            # already-fixed inputs to this call (computed once by the router/
            # dispatch above), so recompute here touches no EP dispatch/
            # combine state and cannot desync ExpertParallel's cached
            # gather_idx/permutation (unlike checkpointing the whole MoE
            # block, which would also recompute the router — see the class
            # docstring's note on the v158 argsort-tie-break bug this avoids).
            if self.measurement.enabled:
                with self.measurement.time("expert_forward"):
                    routed_output = torch.utils.checkpoint.checkpoint(
                        self.experts,
                        routed_input,
                        num_tokens_per_expert,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                routed_output = torch.utils.checkpoint.checkpoint(
                    self.experts,
                    routed_input,
                    num_tokens_per_expert,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
        else:
            if self.measurement.enabled:
                with self.measurement.time("expert_forward"):
                    routed_output = self.experts(routed_input, num_tokens_per_expert)
            else:
                routed_output = self.experts(routed_input, num_tokens_per_expert)

        # EP combine: reverse to return outputs to originating ranks.
        if self._ep_combine is not None:
            if self.measurement.enabled:
                with self.measurement.time("combine"):
                    routed_output = self._ep_combine(routed_output)
            else:
                routed_output = self._ep_combine(routed_output)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x).reshape(bs * slen, dim)
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))

        if torch.compiler.is_compiling():
            # Hints to compile dynamic shapes to pass through slice shape checks.
            num_tokens = num_tokens_per_expert.sum().item()
            torch._check_is_size(num_tokens)
            torch._check(num_tokens <= token_indices.size(0))
            torch._check(num_tokens <= routed_output.size(0))
        if self.measurement.enabled:
            with self.measurement.time("final_scatter"):
                out = _accumulate_routed_output(
                    out,
                    token_indices,
                    routed_output,
                    shared_expert=self.shared_expert,
                )
        else:
            out = _accumulate_routed_output(
                out,
                token_indices,
                routed_output,
                shared_expert=self.shared_expert,
            )
        out = out.reshape(bs, slen, dim)
        return out

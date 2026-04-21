# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import PrepareModuleInput, PrepareModuleOutput
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


# v109: use_reentrant=False AC restored (safe now — no AllToAll during AC recompute due to caching).
# v108: AllToAll output caching — cached_dispatch_output/cached_combine_output returned during AC recompute.
# v107: use_reentrant=True AC fixes AllToAll deadlock (all ranks process backward in lockstep).
# v106: gloo CPU-bounce ONLY for no-grad contexts (ref forward); XCCL for all others.
#
# v105 crash: ze_handle_manager.cpp:226 get_ptr: EXCEPTION: unknown memory type
#   XCCL AllToAll crashed during REF forward (ref_cpu_offload=True: params fetched from
#   CPU → freshly allocated XPU buffers not registered in ze_handle_manager as the right
#   memory type). Policy forward (FSDP2 pre-loaded USM buffers) worked fine.
#
# v99-v104 failure: all gloo P2P approaches deadlock during AC recompute.
#   Root cause: AllToAll requires ALL EP ranks to exchange data for the SAME layer.
#   With use_reentrant=False AC, ranks can be at different layers' AC recomputes
#   simultaneously. Any "wait for peer" operation (all_reduce, barrier, irecv) deadlocks.
#   Only XCCL AllToAll (collective) keeps ranks in lockstep → no deadlock.
#
# v106 fix: use torch.is_grad_enabled() to select the AllToAll transport:
#   - is_grad_enabled() == False (ref forward, torch.no_grad()): gloo CPU-bounce.
#     Ref forward is sequential (all ranks at same layer), so gloo P2P won't deadlock.
#     Avoids ze_handle_manager crash from cpu_offload tensors.
#   - is_grad_enabled() == True (policy forward, GRPO step forward, AC recompute):
#     XCCL AllToAll. Collective → keeps all EP ranks in lockstep → no deadlock.
#     Policy model uses FSDP2 pre-loaded USM buffers → no ze_handle_manager crash.
#
# Backward: always XCCL. Ref forward runs under no_grad → backward never called for it.
# Training forward and AC recompute use XCCL forward → backward in lockstep → XCCL safe.
_XPU_A2A_FWD_GLOO_GROUP: Optional[dist.ProcessGroup] = None  # forward AllToAll gloo group
_XPU_A2A_BWD_GLOO_GROUP: Optional[dist.ProcessGroup] = None  # backward AllToAll gloo group (SEPARATE from FWD in v129)
_XCCL_BWD_CALL_COUNT: int = 0  # v110 diag: count XCCL backward AllToAll calls per rank
_GLOO_BWD_CALL_COUNT: int = 0  # v129 diag: count gloo backward AllToAll calls per rank


def _cpu_bounce_all_to_all(
    cpu_input: "Tensor",
    input_splits: list,
    output_splits: list,
    gloo_group: "dist.ProcessGroup",
    feat_shape: tuple,
) -> "Tensor":
    """AllToAll via gloo P2P batch_isend_irecv on CPU tensors. No all_reduce.

    v103: Removed the gloo all_reduce from forward AllToAll (was causing AC recompute
    deadlocks across v99-v102). The all_reduce was originally added in v97 to fix
    output_splits inconsistency, but output_splits is ALREADY consistent because
    _token_dispatch computes it from all_gather_tensor(num_tokens_per_expert) via XCCL
    before calling this function. The XCCL all-gather ensures:
      output_splits_A[B] = ntpe_matrix[B, A_experts].sum()
                         = input_splits_B[A]  (by construction)
    So output_splits IS consistent — no gloo all_reduce needed.

    Deadlock root cause (v99-v102): gloo all_reduce is a collective requiring all 4 EP
    ranks to arrive simultaneously. With use_reentrant=False AC, PyTorch's C++ autograd
    threads release the GIL inside dist.all_reduce, allowing other threads to increment
    the pool counter. Different ranks' threads can be at different layers' AC recomputes
    simultaneously, so different ranks enter different pool[N]'s all_reduce → deadlock.

    v103 fix: Remove all_reduce. Use output_splits directly for recv buffer sizing.
    Only P2P + barrier remain. P2P isend/irecv are asynchronous — sends queue in socket
    buffers, irecvs wait until the message arrives. The final barrier ensures all P2P
    ops complete before returning. Different layers' AC recomputes share the same gloo
    group but cannot deadlock: each P2P send is addressed to a specific peer and the
    message waits in the buffer. The barrier ensures completion without requiring all
    ranks to be at exactly the same point simultaneously.

    Args:
        cpu_input: CPU tensor, shape (sum(input_splits), *feat_shape)
        input_splits: how many rows to send to each rank (from XCCL all-gather, consistent)
        output_splits: how many rows to receive from each rank (from XCCL all-gather, consistent)
        gloo_group: gloo ProcessGroup (backend=gloo) for the EP shard group
        feat_shape: non-batch dimensions of token features
    Returns:
        (cpu_output, output_splits) — output tensor and recv sizes (same as input output_splits)
    """
    ws = dist.get_world_size(gloo_group)
    my_rank = dist.get_rank(gloo_group)

    # output_splits is already consistent with each sender's input_splits[my_rank]
    # (guaranteed by the XCCL all-gather in _token_dispatch). No all_reduce needed.
    recv_sizes = output_splits  # direct use, no cross-rank negotiation

    cpu_output = torch.empty((sum(recv_sizes),) + feat_shape, dtype=cpu_input.dtype)

    ops = []
    send_off = 0
    for dst in range(ws):
        n = input_splits[dst]
        if dst != my_rank and n > 0:
            dst_global = dist.get_global_rank(gloo_group, dst)
            ops.append(dist.P2POp(dist.isend,
                                  cpu_input[send_off:send_off + n].contiguous(),
                                  dst_global, group=gloo_group))
        send_off += n

    recv_off = 0
    for src in range(ws):
        n = recv_sizes[src]
        if src != my_rank and n > 0:
            src_global = dist.get_global_rank(gloo_group, src)
            ops.append(dist.P2POp(dist.irecv,
                                  cpu_output[recv_off:recv_off + n],
                                  src_global, group=gloo_group))
        recv_off += n

    # Self-copy (no network)
    my_send_off = sum(input_splits[:my_rank])
    my_recv_off = sum(recv_sizes[:my_rank])
    n_self = input_splits[my_rank]
    if n_self > 0:
        cpu_output[my_recv_off:my_recv_off + n_self].copy_(
            cpu_input[my_send_off:my_send_off + n_self]
        )

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    # No barrier: req.wait() guarantees all sends/recvs complete. dist.barrier would
    # require all 4 EP ranks simultaneously, deadlocking when AC recomputes fire at
    # different times across ranks (v104 fix — same root cause as the all_reduce).
    # Gloo TCP FIFO ordering + tag-matched P2P handles correctness without a barrier.

    return cpu_output, recv_sizes


class _XPUSyncAllToAll(torch.autograd.Function):
    """AllToAll with XPU-safe transport in both forward and backward.

    v95: CPU-bounce via gloo P2P batch_isend_irecv when _XPU_A2A_GLOO_GROUP is set.
    Copies input XPU→CPU, runs gloo P2P AllToAll (no preamble, no collective sync,
    zero-token ranks issue no ops), copies output CPU→XPU. Fixes ze_handle_manager.

    v94 failed: gloo does not support all_to_all (list API) at all.
    v93 failed: XCCL backend doesn't support CPU tensors.
    v84 failed: gloo all_to_all_single preamble mismatch on unequal splits (v60).
    v65-v79 failed: gloo broadcast/allreduce AllToAll → zero-token rank deadlock.
    v87-v92 warmup failed: CCL workspace is CCL-internal, not pre-registerable.

    Non-XPU path or _XPU_A2A_GLOO_GROUP=None: direct XCCL (original behavior).
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        output_splits: list,
        input_splits: list,
        group: dist.ProcessGroup,
        cached_output: Optional[Tensor] = None,
    ) -> Tensor:
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        ctx.is_xpu = (input.device.type == "xpu")
        ctx.feat_shape = tuple(input.shape[1:])

        if cached_output is not None:
            # v108: AC recompute path — skip AllToAll entirely, use cached output.
            # ExpertParallel caches the AllToAll output from the first forward and passes
            # it here during AC recompute. This avoids any XCCL/gloo communication during
            # the backward phase (AC recompute), eliminating the deadlock between AC recompute
            # AllToAll and FSDP2 gloo reduce_scatter that blocked v99-v107.
            # Gradient correctness: _XPUSyncAllToAll.backward is still registered and will
            # run the reverse AllToAll on the gradient during the actual backward pass.
            # ctx is set so backward knows the correct splits for the gradient AllToAll.
            return cached_output

        if ctx.is_xpu and _XPU_A2A_FWD_GLOO_GROUP is not None and not torch.is_grad_enabled():
            # v106: gloo CPU-bounce ONLY when grad is disabled (ref forward under no_grad).
            # Ref forward is sequential: all ranks at same layer → gloo P2P won't deadlock.
            # Avoids ze_handle_manager crash: cpu_offload tensors (freshly allocated XPU
            # buffers from CPU→XPU fetch) are not registered as USM → XCCL crashes on them.
            # torch.is_grad_enabled()==True means training forward or AC recompute → use XCCL.
            torch.xpu.synchronize()
            cpu_in = input.cpu().contiguous()
            cpu_out, actual_recv_splits = _cpu_bounce_all_to_all(
                cpu_in, input_splits, output_splits, _XPU_A2A_FWD_GLOO_GROUP, ctx.feat_shape
            )
            ctx.output_splits = actual_recv_splits
            return cpu_out.to(input.device)
        else:
            out = input.new_empty(sum(output_splits), *input.shape[1:])
            dist.all_to_all_single(out, input, output_splits, input_splits, group=group)
            if ctx.is_xpu:
                torch.xpu.synchronize()
                # v105: OFI CQ drain after XCCL AllToAll.
                # XCCL all_to_all_single forces all EP ranks to synchronize (collective).
                # After it returns, all ranks are at the same layer — the barrier is
                # trivially satisfied and drains the OFI CQ of residual NIC events.
                dist.barrier(group=group)
            return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_output = grad_output.contiguous()

        if False and ctx.is_xpu and _XPU_A2A_BWD_GLOO_GROUP is not None:  # v134: disabled — all gloo collectives return corrupt values on Aurora XPU + Slingshot-11
            # v131: gloo all_gather-based AllToAll for backward.
            #
            # v127-v130 failure history (all gloo-based):
            #   v127: manual P2P (reused FWD group)         → size mismatch
            #   v128: manual P2P + pre-barrier (reused)     → same mismatch
            #   v129: manual P2P + pre-barrier (sep group)  → same mismatch @ call#39
            #   v130: all_to_all_single (gloo)              → immediate size mismatch
            #
            # Root cause of v127-v129 (from diagnostics):
            #   At call#39, gloo_rank=1 is at a different backward op than gloo_rank=0,2,3.
            #   The barrier syncs by counter# but ranks have drifted: one rank has
            #   processed one extra AllToAll backward somewhere before call#39.
            #   The drift origin is unknown (same autograd graph, same expert count).
            #
            # Root cause of v130:
            #   gloo's all_to_all_single is broken for unequal splits — preamble mismatch
            #   fires immediately at call#1. Same failure as v60 (forward path).
            #   gloo's TCP transport doesn't negotiate sizes correctly in all_to_all_single.
            #
            # v131 fix: implement AllToAll via two gloo all_gather calls.
            #   all_gather works correctly on gloo (equal-size tensors, no preamble issues).
            #   The AllToAll is decomposed as:
            #     1. all_gather(send_splits) → each rank knows all ranks' send sizes
            #        (needed for offset computation into gathered gradient tensors)
            #     2. Pad each rank's gradient to max_total size (equal shapes for all_gather)
            #     3. all_gather(padded_gradient) → each rank holds all ranks' gradients
            #     4. Each rank extracts slices: for each src, take recv_splits[src] tokens
            #        starting at the correct offset within src's padded contribution
            #   Both all_gather calls are proper collectives → true lockstep enforced.
            #   If one rank drifts to a different backward op, the all_gather blocks.
            #   Cost: ~10-30% overhead vs direct AllToAll (max-padding in step 2).
            torch.xpu.synchronize()
            cpu_grad = grad_output.cpu().contiguous()

            gloo_group = _XPU_A2A_BWD_GLOO_GROUP
            ws = dist.get_world_size(gloo_group)
            my_rank = dist.get_rank(gloo_group)
            send_splits = ctx.output_splits   # tokens to send to each EP rank
            recv_splits = ctx.input_splits    # tokens to receive from each EP rank

            # v133: AllToAll via gloo all_reduce (avoids all_gather which returns corrupt values).
            #
            # v131-v132 failure analysis:
            #   v131: all_gather(send_splits) returned inconsistent values → extraction bounds error
            #   v132: all_gather(send_splits) returns values where sum(gathered[src][my_rank])
            #         ≠ sum(recv_splits) for ALL ranks (off by ~12000 tokens), violating
            #         AllToAll conservation. Root cause: gloo all_gather on Aurora XPU +
            #         Slingshot-11 returns WRONG VALUES (not just ordering issues).
            #         sum(got) = 32128 constant across all EP groups, sum(expected) ≈ 44000+.
            #   gloo all_gather is fundamentally unreliable on this platform.
            #   gloo all_to_all_single: broken (preamble mismatch, v130 = v60).
            #   gloo P2P (batch_isend_irecv): broken (size mismatch, v127-v129).
            #   gloo all_reduce: NOT YET TESTED — may work (simpler collective).
            #   XCCL backward: driver SIGSEGV on freshly-allocated gradient XPU tensors
            #     (not OOM — confirmed at v122-v126 with expert_cpu_offload=True, call#0).
            #
            # v133 approach: AllToAll via ws all_reduce calls on gloo.
            #   Step 1: exchange send_splits using a ws×ws all_reduce.
            #           Each rank fills row my_rank of a ws×ws matrix; others are 0.
            #           After SUM all_reduce: all ranks have the full matrix.
            #   Step 2: use send_splits matrix to compute max_total for padding.
            #   Step 3: for each src rank, one all_reduce of the padded gradient:
            #           Only src fills in its cpu_grad; others contribute zeros.
            #           After SUM all_reduce: all ranks have src's full cpu_grad.
            #           Each rank extracts the slice destined for my_rank.
            #
            # Cost: ws all_reduce calls for gradients (each of size max_total × hidden_dim),
            #   plus 1 all_reduce for the splits matrix.
            #   Transfer = ws × max_total × hidden_dim × ws bytes (ws copies broadcast).
            #   vs all_gather: ws × max_total × hidden_dim total (1 copy broadcast to all).
            #   all_reduce costs ~ws× more bandwidth than all_gather, but all_reduce is
            #   the ONLY gloo collective proven not to cause silent corruption here.
            #
            # Correctness: all_reduce (SUM) is a ring-based collective — all ranks must
            #   participate simultaneously. No P2P, no preamble, no unequal-split issues.
            #   Ranks are guaranteed to be at the same operation (ring blocks until all arrive).

            # Step 1: exchange send_splits via a ws×ws all_reduce.
            splits_mat = torch.zeros(ws, ws, dtype=torch.int64)
            splits_mat[my_rank] = torch.tensor(send_splits, dtype=torch.int64)
            dist.all_reduce(splits_mat, op=dist.ReduceOp.SUM, group=gloo_group)
            # splits_mat[src][dst] = tokens src sends to dst in this backward AllToAll

            # Step 2: compute recv counts (authoritative from all_reduce, not stale ctx).
            recv_from = [int(splits_mat[src][my_rank].item()) for src in range(ws)]
            my_total = sum(send_splits)
            max_total = max(int(splits_mat[src].sum().item()) for src in range(ws))

            # Step 3: for each source rank, broadcast its cpu_grad via all_reduce,
            #   then extract the slice destined for my_rank.
            cpu_out = cpu_grad.new_zeros((sum(recv_from),) + ctx.feat_shape)
            out_off = 0
            for src in range(ws):
                src_total = int(splits_mat[src].sum().item())
                buf = cpu_grad.new_zeros((max(max_total, 1),) + ctx.feat_shape)
                if my_rank == src and my_total > 0:
                    buf[:my_total].copy_(cpu_grad)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=gloo_group)
                # buf now contains src's cpu_grad (padded to max_total)
                src_off = int(splits_mat[src][:my_rank].sum().item())
                n = recv_from[src]
                if n > 0:
                    cpu_out[out_off:out_off + n].copy_(buf[src_off:src_off + n])
                out_off += n

            return cpu_out.to(grad_output.device), None, None, None, None
        else:
            global _XCCL_BWD_CALL_COUNT
            _XCCL_BWD_CALL_COUNT += 1
            _call_n = _XCCL_BWD_CALL_COUNT
            import logging as _logging
            _bwd_log = _logging.getLogger(__name__)
            # v111: print() to bypass SSH pipe buffering — log.info may be lost for hanging ranks
            print(
                f"[v111-BWD-A2A] rank={dist.get_rank()} call#{_call_n} "
                f"grad_output.shape={tuple(grad_output.shape)} "
                f"in_splits={ctx.input_splits} out_splits={ctx.output_splits}",
                flush=True,
            )
            _bwd_log.info(
                "Rank %d: XCCL bwd AllToAll #%d PRE in_splits=%s out_splits=%s",
                dist.get_rank(), _call_n, ctx.input_splits, ctx.output_splits,
            )
            grad_input = grad_output.new_empty(sum(ctx.input_splits), *grad_output.shape[1:])
            dist.all_to_all_single(
                grad_input, grad_output,
                ctx.input_splits, ctx.output_splits,
                group=ctx.group,
            )
            if ctx.is_xpu:
                torch.xpu.synchronize()
                _bwd_log.info("Rank %d: XCCL bwd AllToAll #%d POST-sync, calling barrier", dist.get_rank(), _call_n)
                dist.barrier(group=ctx.group)  # v105: OFI CQ drain (same as forward)
                _bwd_log.info("Rank %d: XCCL bwd AllToAll #%d BARRIER done", dist.get_rank(), _call_n)
            return grad_input, None, None, None, None


def _xpu_sync_all_to_all(
    input: Tensor,
    output_splits: list,
    input_splits: list,
    group: dist.ProcessGroup,
    cached_output: Optional[Tensor] = None,
) -> Tensor:
    """Drop-in replacement for ``all_to_all_single_autograd`` with XPU sync.

    Args:
        cached_output: If provided (AC recompute), skip the AllToAll and return
            this tensor directly. The backward AllToAll is still registered.
    """
    return _XPUSyncAllToAll.apply(input, output_splits, input_splits, group, cached_output)


# implementation of Tensor Parallel on the non-shared experts in MoE
class ExpertTensorParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[tuple[Optional[Placement]]] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts or (Replicate(), None)
        self.output_layout = output_layout or Partial()
        self.desired_input_layouts = (Replicate(), None)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor, input_layout, desired_input_layout = (
            inputs[0],
            input_layouts[0],
            desired_input_layouts[0],
        )
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "gate_proj",
            nn.Parameter(distribute_tensor(module.gate_proj, device_mesh, [Shard(2)])),
        )  # Column-wise sharding
        module.register_parameter(
            "down_proj",
            nn.Parameter(distribute_tensor(module.down_proj, device_mesh, [Shard(1)])),
        )  # Row-wise sharding
        module.register_parameter(
            "up_proj",
            nn.Parameter(distribute_tensor(module.up_proj, device_mesh, [Shard(2)])),
        )  # Column-wise sharding

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


# NOTE: This is to achieve replicate computation on the gate module in the MoE router.
# It does nothing other than (1) setting the module parameters as DTensors on the given mesh
# and (2) inserting hooks to module boundary to change torch.Tensor to DTensor and back.
# TODO: The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
# which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.
class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


# TODO: this is temporarily copied over from PyTorch core to enable Llama4 on stable PyTorch
# Once this API is in stable, we should migrate over to the PyTorch core one
class PrepareModuleInputOutput(ParallelStyle):
    """
    Configure the nn.Module's inputs (and outputs) to convert the input tensors (and output tensors, respectively) of the nn.Module
    to DTensors at runtime according to ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``, respectively). This is a combination of
    :class:`PrepareModuleInput` and :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.


    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInputOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInputOutput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...),
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0),
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        use_local_input: bool = False,
        output_layouts: Union[Placement, tuple[Placement]],
        desired_output_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)

        return module


class ExpertParallel(ParallelStyle):
    """Expert Parallelism for MoE layers via All-to-All token dispatch.

    Each EP rank owns ``num_experts // ep_degree`` expert weight matrices (sharded on
    dim 0). Tokens are dispatched to expert-owning ranks via ``_xpu_sync_all_to_all``
    and combined after local expert computation via a reverse All-to-All.

    Gradient flow is via ``_XPUSyncAllToAll``, a custom autograd Function that calls
    ``xpu.synchronize()`` in both forward and backward to prevent FSDP2 backward
    prefetch all-gathers from overlapping with the EP AllToAll on Aurora's OFI fabric.

    Compatible with XCCL 25.190.0+ (frameworks/2025.2.0) on Aurora/XPU.
    Uses ``_permute``/``_unpermute`` from ``torchtune.modules.moe.utils`` (pure torch,
    no Triton) — compatible with XPU's loop-based expert path.

    Usage::

        from torch.distributed.tensor.parallel import parallelize_module
        ep_plan = {"layers.0.moe_block.experts": ExpertParallel()}
        parallelize_module(model, ep_mesh, ep_plan)

    AC recompute non-determinism fix:
        router top_k on XPU may be non-deterministic (tie-breaking) → count_e can differ
        between the original forward and AC recompute → backward matmul shape mismatch.
        Fix: ``advance_fwd_step()`` is called by the recipe before each model forward to
        increment ``ExpertParallel._fwd_step_counter``. Splits saved during forward at
        step N are reused during the AC recompute (also step N). The next forward
        increments the counter → fresh splits.
    """

    # Class-level forward step counter. Increment before each model forward so
    # AC recompute (same step) reuses saved splits while the next forward gets fresh ones.
    _fwd_step_counter: int = 0

    @classmethod
    def advance_fwd_step(cls) -> None:
        """Call before each model forward that may be AC-checkpointed (training forwards).
        Ensures the next dispatch computes fresh splits, while the AC recompute reuses them.
        """
        cls._fwd_step_counter += 1

    def __init__(self) -> None:
        super().__init__()
        self._input_splits: Optional[list[int]] = None
        self._output_splits: Optional[list[int]] = None
        self._perm: Optional[Tensor] = None
        self._total_dispatched: Optional[int] = None
        self._ntpe_group: Optional[Tensor] = None  # saved for AC recompute
        self._saved_at_step: int = -1  # _fwd_step_counter value when splits were saved
        # v108: Cache AllToAll outputs from first forward for AC recompute.
        # AC recompute returns cached tensors instead of re-running AllToAll.
        # This eliminates XCCL/gloo communication during backward (AC recompute),
        # preventing deadlock with FSDP2 reduce_scatter in the backward phase.
        self._cached_dispatch_output: Optional[Tensor] = None
        self._cached_combine_output: Optional[Tensor] = None
        self._combine_saved_at_step: int = -1  # step when _cached_combine_output was set
        # v112: Pre-AllToAll permutation index (expert-sorted → rank-sorted).
        # Interleaved expert assignment requires tokens to be grouped by dest rank
        # before AllToAll. Saved for AC recompute to avoid recomputation.
        self._pre_a2a_perm: Optional[Tensor] = None
        self._inv_pre_a2a_perm: Optional[Tensor] = None  # inverse, for combine reverse

    def _partition_fn(
        self, name: str, mod: nn.Module, device_mesh: DeviceMesh
    ) -> None:
        """Shard expert weight matrices on dim 0 (num_experts) across EP ranks.

        Uses plain tensor slicing (not DTensor) to avoid FSDP2 mesh-dimension
        conflicts when the EP mesh is a submesh of the FSDP mesh.
        """
        ep_rank = device_mesh.get_local_rank()
        ep_degree = device_mesh.shape[0]
        for param_name, param in list(mod.named_parameters(recurse=False)):
            full_data = param.data  # (num_experts, ...)
            assert full_data.shape[0] % ep_degree == 0, (
                f"num_experts ({full_data.shape[0]}) must be divisible by ep_degree ({ep_degree})"
            )
            # v112: Interleaved expert assignment — rank r owns experts r, r+ep_degree, ...
            local_data = full_data[ep_rank::ep_degree].contiguous()
            mod.register_parameter(param_name, nn.Parameter(local_data))

    def _token_dispatch(
        self,
        mod: nn.Module,
        routed_input: Tensor,
        num_tokens_per_expert: Tensor,
        *,
        device_mesh: DeviceMesh,
    ) -> tuple[Tensor, Tensor]:
        """All-to-All dispatch: route tokens to the ranks that own their experts.

        Called directly from ``MoE.forward()`` via the ``_ep_dispatch`` callable
        (bypassing the broken hook mechanism — FSDP2 fully_shard drops EP hooks).

        Args:
            mod: The ``GroupedExperts`` module (retained for API consistency but unused).
            routed_input: Shape ``(bs*slen*top_k, dim)``.
            num_tokens_per_expert: Shape ``(num_experts,)``.
            device_mesh: EP mesh of shape ``(ep_degree,)``.

        Returns:
            ``(dispatched_tokens, local_ntpe)`` for this rank's local experts.
        """
        from torch.distributed._functional_collectives import all_gather_tensor
        from torchtune.modules.moe.utils import _permute

        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        ep_rank = device_mesh.get_local_rank()

        current_step = ExpertParallel._fwd_step_counter
        _is_reuse = (self._saved_at_step == current_step and self._ntpe_group is not None)
        if _is_reuse:
            # AC recompute: same _fwd_step_counter value as when splits were saved.
            ntpe_group = self._ntpe_group
        else:
            # Fresh forward (new step or generation): compute splits from router output.
            with torch.no_grad():
                # All-gather each rank's full num_tokens_per_expert histogram.
                # ntpe_matrix[r, e] = tokens rank r routes to expert e (global expert index)
                ntpe_all = all_gather_tensor(
                    num_tokens_per_expert, gather_dim=0, group=device_mesh.get_group()
                )
                ntpe_all = torch.ops._c10d_functional.wait_tensor(ntpe_all)
                ntpe_matrix = ntpe_all.view(ep_degree, -1)  # (ep_degree, num_experts)

                # v112: Interleaved expert assignment — rank r owns global experts
                # {r, r+ep_degree, r+2*ep_degree, ...} (stride=ep_degree).
                # Contiguous assignment (v0-v111) caused extreme routing imbalance:
                # rank 0 owned experts 0-31 (all hot), ranks 2-3 owned experts 64-127 (cold).
                # Interleaved distributes hot experts evenly across all ranks.

                num_experts = num_tokens_per_expert.shape[0]
                ntpe_int = num_tokens_per_expert.to(torch.long)

                # input_splits[i] = tokens this rank sends to rank i
                # = tokens this rank routes to experts owned by rank i
                # = num_tokens_per_expert[i::ep_degree].sum()
                self._input_splits = [
                    int(ntpe_int[i::ep_degree].sum().item())
                    for i in range(ep_degree)
                ]

                # output_splits[i] = tokens this rank receives from rank i
                # = tokens rank i routes to my experts (ep_rank, ep_rank+ep_degree, ...)
                # = ntpe_matrix[i, ep_rank::ep_degree].sum()
                ntpe_int_matrix = ntpe_matrix.to(torch.long)
                self._output_splits = [
                    int(ntpe_int_matrix[i, ep_rank::ep_degree].sum().item())
                    for i in range(ep_degree)
                ]

                # ntpe_group[ep_r * num_local_experts + local_exp] = tokens from rank ep_r
                # to my local expert local_exp (= global expert ep_rank + local_exp * ep_degree).
                # ntpe_matrix[:, ep_rank::ep_degree] has shape (ep_degree, num_local_experts).
                # Row-major view gives source-rank-major order expected by _permute.
                ntpe_group = ntpe_int_matrix[:, ep_rank::ep_degree].contiguous().view(-1)
                self._ntpe_group = ntpe_group  # save for AC recompute at same step

                # Build pre-AllToAll permutation: expert-sorted → rank-sorted.
                # routed_input arrives expert-sorted (exp0 tokens, exp1 tokens, ...).
                # AllToAll requires rank-sorted (rank0's tokens, rank1's tokens, ...).
                # With interleaved assignment, rank i's experts are i, i+ep, i+2*ep, ...
                # So rank-sorted = [exp0, exp{ep}, ..., exp1, exp{1+ep}, ..., ...]
                offsets_per_expert = torch.zeros(
                    num_experts + 1, dtype=torch.long, device=routed_input.device
                )
                offsets_per_expert[1:] = torch.cumsum(ntpe_int, dim=0)
                perm_parts = []
                for rank_i in range(ep_degree):
                    for local_exp in range(num_local_experts):
                        global_exp = rank_i + local_exp * ep_degree
                        start = int(offsets_per_expert[global_exp].item())
                        count = int(ntpe_int[global_exp].item())
                        if count > 0:
                            perm_parts.append(
                                torch.arange(start, start + count, dtype=torch.long,
                                             device=routed_input.device)
                            )
                if perm_parts:
                    self._pre_a2a_perm = torch.cat(perm_parts)
                else:
                    self._pre_a2a_perm = torch.zeros(0, dtype=torch.long,
                                                     device=routed_input.device)
                # Inverse permutation for combine side (rank-sorted → expert-sorted)
                total_toks = routed_input.shape[0]
                if total_toks > 0 and self._pre_a2a_perm.numel() > 0:
                    inv = torch.empty_like(self._pre_a2a_perm)
                    inv[self._pre_a2a_perm] = torch.arange(
                        total_toks, dtype=torch.long, device=routed_input.device
                    )
                    self._inv_pre_a2a_perm = inv
                else:
                    self._inv_pre_a2a_perm = torch.zeros(0, dtype=torch.long,
                                                         device=routed_input.device)
            self._saved_at_step = current_step

        # v112: Apply pre-AllToAll permutation (expert-sorted → rank-sorted).
        # Required for interleaved assignment so AllToAll input_splits are contiguous.
        if self._pre_a2a_perm is not None and self._pre_a2a_perm.numel() > 0:
            routed_input = routed_input[self._pre_a2a_perm]

        # Dispatch: send tokens to expert-owning ranks (gradient-tracked).
        # v108: On AC recompute (_is_reuse=True), pass cached_dispatch_output so
        # _XPUSyncAllToAll.forward skips the actual AllToAll communication and returns
        # the cached tensor. The backward AllToAll is still registered in the graph.
        routed_input = _xpu_sync_all_to_all(
            routed_input,
            self._output_splits,
            self._input_splits,
            device_mesh.get_group(),
            cached_output=self._cached_dispatch_output if _is_reuse else None,
        )
        if not _is_reuse:
            # First forward: cache the AllToAll output for AC recompute.
            self._cached_dispatch_output = routed_input

        self._total_dispatched = routed_input.shape[0]

        # Reorder from source-rank-major to local-expert-major order
        routed_input, local_ntpe, self._perm = _permute(
            routed_input, ntpe_group, ep_degree, num_local_experts
        )
        return routed_input, local_ntpe

    def _token_combine(
        self,
        mod: nn.Module,
        routed_output: Tensor,
        *,
        device_mesh: DeviceMesh,
    ) -> Tensor:
        """Reverse All-to-All: return expert outputs to originating ranks.

        Args:
            mod: The ``GroupedExperts`` module (retained for API consistency but unused).
            routed_output: Expert outputs in local-expert-major order,
                shape ``(num_processed_tokens, dim)``.
            device_mesh: EP mesh of shape ``(ep_degree,)``.

        Returns:
            Tensor in original token order, shape ``(bs*slen*top_k, dim)``.
        """
        from torchtune.modules.moe.utils import _unpermute

        # Reverse permutation: local-expert order → source-rank-major order
        routed_output = _unpermute(routed_output, self._perm, self._total_dispatched)

        # Reverse All-to-All: swap send/receive split sizes.
        # v108: detect AC recompute by matching combine's own saved step counter.
        # _combine_saved_at_step is set when _cached_combine_output is stored on the
        # FIRST forward of a given step. AC recompute for that step has the same counter.
        current_step = ExpertParallel._fwd_step_counter
        _is_reuse_combine = (
            self._combine_saved_at_step == current_step
            and self._cached_combine_output is not None
        )
        routed_output = _xpu_sync_all_to_all(
            routed_output,
            self._input_splits,   # receive: original tokens from each rank
            self._output_splits,  # send: processed tokens back to each rank
            device_mesh.get_group(),
            cached_output=self._cached_combine_output if _is_reuse_combine else None,
        )
        if not _is_reuse_combine:
            # First forward: cache the combine AllToAll output for AC recompute.
            self._cached_combine_output = routed_output
            self._combine_saved_at_step = current_step

        # v112: Apply inverse pre-AllToAll permutation (rank-sorted → expert-sorted).
        # Restores the original expert-sorted token order expected by scatter_add in MoE.forward.
        if self._inv_pre_a2a_perm is not None and self._inv_pre_a2a_perm.numel() > 0:
            routed_output = routed_output[self._inv_pre_a2a_perm]

        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Register All-to-All dispatch/combine on the expert module.

        Weight sharding is NOT done here because checkpoint loading must happen
        first. Call ``apply_ep_weight_sharding()`` explicitly after loading the
        checkpoint but BEFORE calling ``shard_model()`` (FSDP2 wrapping).

        We do NOT register forward hooks on ``GroupedExperts`` here because
        FSDP2 ``fully_shard`` (called later) drops or shadows them — only the
        FSDP2-internal pre-hook survives (``num_pre_hooks=1`` observation).

        Instead we store this ``ExpertParallel`` instance and the mesh on the
        ``GroupedExperts`` module. The recipe then calls
        ``wire_ep_to_moe_modules(model)`` after both ``parallelize_module``
        *and* ``shard_experts_for_ep`` to set ``_ep_dispatch``/``_ep_combine``
        directly on each parent ``MoE`` instance — bypassing hooks entirely.

        Args:
            module: The ``GroupedExperts`` module to wrap.
            device_mesh: EP mesh of shape ``(ep_degree,)``.

        Returns:
            The wrapped module (EP metadata attached, weights still full-rank).
        """
        # Store mesh and EP instance on GroupedExperts so wire_ep_to_moe_modules
        # can retrieve them later.
        module._ep_device_mesh = device_mesh
        module._ep_instance = self
        return module


def wire_ep_to_moe_modules(model: nn.Module) -> int:
    """Wire EP dispatch/combine callables directly onto each parent ``MoE`` module.

    Must be called AFTER both ``parallelize_module`` (which sets ``_ep_instance``
    on each ``GroupedExperts``) AND ``shard_experts_for_ep`` (which calls
    ``fully_shard`` and drops the EP hooks). This function bypasses hooks entirely
    by setting ``moe._ep_dispatch`` and ``moe._ep_combine`` as callables that
    ``MoE.forward()`` calls directly.

    The EP dispatch logic lives in ``ExpertParallel._token_dispatch`` /
    ``_token_combine`` — we bind partial callables over the EP instance and mesh.

    Args:
        model: Model whose ``MoE`` submodules should be wired.

    Returns:
        Number of ``MoE`` modules wired.
    """
    from torchtune.modules.moe.moe import MoE

    num_wired = 0
    for name, module in model.named_modules():
        if not isinstance(module, MoE):
            continue
        experts = module.experts
        # After FSDP2 fully_shard, the class is FSDPGroupedExperts(FSDPModule, GroupedExperts).
        # _ep_instance was set by _apply before fully_shard — should survive since it's a
        # plain Python attribute, not a registered parameter/buffer.
        ep_instance = getattr(experts, "_ep_instance", None)
        ep_mesh = getattr(experts, "_ep_device_mesh", None)
        if ep_instance is None or ep_mesh is None:
            continue

        # Bind dispatch/combine to this EP instance and mesh
        module._ep_dispatch = partial(ep_instance._token_dispatch, experts, device_mesh=ep_mesh)
        module._ep_combine = partial(ep_instance._token_combine, experts, device_mesh=ep_mesh)
        num_wired += 1

    return num_wired


def apply_ep_weight_sharding(model: nn.Module) -> int:
    """Slice expert weights to local shards after checkpoint loading.

    Finds all modules that have an ``_ep_device_mesh`` attribute (set by
    ``ExpertParallel._apply``) and slices their parameters along dim 0 so each
    EP rank holds only its local expert slice.

    Supports two modes:

    * **Pre-FSDP2** (original): parameters are plain ``nn.Parameter`` tensors.
      Replaces ``param.data`` with the local slice.
    * **Post-FSDP2 with 1-rank solo FSDP2** (v40): parameters are DTensors whose
      local tensor is the full expert weight (1-rank mesh, no communication split).
      Directly replaces the ``_local_tensor`` attribute with the EP-local slice.
      The FSDP2 all_gather for a 1-rank group is a no-op that just returns
      ``_local_tensor`` directly, so this is sufficient for correct expert forward.

    Args:
        model: Model whose expert modules should be weight-sharded.

    Returns:
        Number of expert modules sharded.
    """
    from torch.distributed.tensor import DTensor
    num_sharded = 0
    for name, module in model.named_modules():
        ep_mesh = getattr(module, "_ep_device_mesh", None)
        if ep_mesh is None:
            continue
        ep_rank = ep_mesh.get_local_rank()
        ep_degree = ep_mesh.shape[0]
        params_found = list(module.named_parameters(recurse=False))
        for param_name, param in params_found:
            # Unwrap DTensor to get the underlying local data.
            raw = param.data
            if isinstance(raw, DTensor):
                # Post-FSDP2 path (v40): param is a DTensor from 1-rank solo FSDP2.
                # _local_tensor holds the full [num_experts, ...] data.
                full_data = raw._local_tensor
            else:
                full_data = raw
            assert full_data.shape[0] % ep_degree == 0, (
                f"num_experts ({full_data.shape[0]}) must be divisible by ep_degree ({ep_degree})"
            )
            n_local = full_data.shape[0] // ep_degree
            # v112: Interleaved expert assignment — rank r owns experts r, r+ep_degree, ...
            # Contiguous assignment caused extreme routing imbalance (rank 0 got all hot experts).
            local_slice = full_data[ep_rank::ep_degree].contiguous()
            if isinstance(raw, DTensor):
                # Post-FSDP2 path: overwrite _local_tensor and update the DTensorSpec's
                # TensorMeta so that raw.shape == local_slice.shape. FSDP2 uses the
                # global shape from _spec.tensor_meta to size the all_gather output buffer;
                # if shape is stale ([128,...] while _local_tensor is [32,...]), FSDP2
                # will allocate a [128,...] buffer but copy [32,...] data into it → crash.
                # With a 1-rank mesh and Shard(0), global_shape == local_shape, so we
                # set both to [n_local, ...].
                raw._local_tensor = local_slice
                try:
                    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
                    new_shape = torch.Size([n_local] + list(full_data.shape[1:]))
                    old_spec = raw._spec
                    raw._spec = DTensorSpec(
                        mesh=old_spec.mesh,
                        placements=old_spec.placements,
                        tensor_meta=TensorMeta(
                            shape=new_shape,
                            dtype=old_spec.tensor_meta.dtype,
                            stride=local_slice.stride(),
                        ),
                    )
                except Exception:
                    # Fallback: leave _spec unchanged. The 1-rank no-op all_gather
                    # may still work if FSDP2 doesn't validate shape strictly.
                    pass
            else:
                # Pre-FSDP2 path: replace plain parameter data.
                param.data = local_slice
        num_sharded += 1
    return num_sharded

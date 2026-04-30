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


# AllGather + ReduceScatter EP dispatch (Mula paper, arXiv 2604.00785).
#
# Replaces AllToAll token dispatch which deadlocked or SIGSEGV'd on Aurora XPU
# + Slingshot-11 (v18-v136 saga):
#   AllToAll forward: OFI CQ contamination → FSDP2 EPERM (v18-v39)
#   AllToAll backward: XCCL SIGSEGV (v40-v133), shape mismatch due to caching bug (v134),
#                      deadlock even after caching removal (v136)
#
# AllGather and ReduceScatter are natively optimized in oneCCL for Aurora (topology-aware,
# L0 IPC) and are used by FSDP2 without issues. Their backward is standard PyTorch autograd:
#   AllGather backward = ReduceScatter
#   ReduceScatter backward = AllGather
# No custom split tracking, no OFI CQ drain barriers needed.
#
# v141: Use native XCCL reduce_scatter_tensor, bypassing the recipe's gloo monkey-patch.
#
# The recipe patches dist.reduce_scatter_tensor → gloo CPU AllReduce+slice to work around
# CCL's ze_handle_manager crash on freshly sub-allocated FSDP2 grad tensors.
# But the monkey-patch only replaces the torch.distributed module attribute — the original
# function is still accessible at torch.distributed.distributed_c10d.reduce_scatter_tensor.
#
# EP tensors are explicitly allocated (new_empty, clone, contiguous) — not FSDP2's
# sub-allocated grad buffers — so they should not trigger the IPC handle bug.
#
# v138-v140 used AllReduce+slice which sends 4× more data than needed and deadlocked/stalled
# after ~60 operations during backward (CCL progress-engine stall or resource exhaustion).
#
# v142: Add xpu.synchronize() + dist.barrier(group) after each EP AllGather/ReduceScatter.
#   Result: HUNG for 23 minutes in backward. xpu.synchronize() waits for ALL pending GPU
#   work (not just the collective), causing ~30s/op slowdown × 39 ops ≈ 20 minutes, then
#   op #40 still hung. Cause confirmed: xpu.synchronize() unnecessary and harmful.
#
# v143: Diagnose precise EP backward hang location.
#   - Kept dist.barrier(group) for OFI CQ drain; removed xpu.synchronize().
#   - Added 3-phase per-op logging (ENTER/COLL-DONE/EXIT) to all ep_pg ops.
#   - Result (2026-04-22): ALL 260+ ops completed (ENTER→COLL-DONE→EXIT) with no hang.
#     Backward ops (AG-BWD, RS-BWD) all exited cleanly. Job killed by PBS walltime, not hang.
#   - Conclusion: dist.barrier(group) after each EP collective drains the OFI CQ.
#     v141 hung at backward op #40 because without barriers, the OFI CQ accumulates
#     stale entries across 100+ ops and deadlocks on op ~100 (last backward op).
#
# v144: Production-ready version — barriers kept, diagnostic logging removed. FAILED.
# v145: sleep(10ms) after barrier only. FAILED.
# v146: sleep(50ms) pre+post barrier. FAILED.
# v147: sleep(50ms) pre+post + NTPE sleep. FAILED.
#   All sleep-based approaches failed: backward hangs at first EP collective consistently.
#   sleep() yields CPU but the OFI hang mechanism is not about CPU time.
#
# v149 (FAILED, 2026-04-22): gloo EP-group barrier (4 ranks TCP) after each XCCL collective.
#   v148 (XCCL barrier + prints) hung at op #259. v149 (gloo barrier + prints) ALSO hung at #259.
#   The 6 ENTER RS-BWD lines (0 COLL-DONE) show the XCCL collective itself deadlocks.
#   Per-EP-group gloo barrier (4 ranks) cannot prevent concurrent XCCL between EP groups.
#
# v150: Global 12-rank gloo barrier (all tiles) before + after each XCCL EP collective.
#   Hypothesis: all 3 DP replicas (3 EP groups) run concurrent XCCL EP collectives on
#   the SAME OFI endpoint (oneCCL uses one OFI endpoint per node, shared across all
#   process groups). At op #259, all 3 EP groups converge on RS-BWD simultaneously.
#   Their OFI CQ events cross-contaminate each other's CCL progress engines → deadlock.
#   Per-EP-group gloo barriers (v149) don't prevent this — they only serialize within a
#   group. Only a GLOBAL barrier (all 12 ranks) can prevent concurrent XCCL EP ops.
#
# Root cause: oneCCL OFI collective → Slingshot-11 CXI NIC generates CQ entries.
#   CCL progress thread polls CQ during active collectives but may leave residual entries
#   after completion. Subsequent collectives start while stale entries remain → deadlock.
#   XCCL dist.barrier(ep_pg) adds MORE OFI CQ entries rather than draining existing ones.
#   time.sleep() does not help: it doesn't trigger OFI CQ polling in the CCL worker.
#
# v149 fix: Use gloo TCP barrier (not OFI) after each EP collective.
#   _GLOO_EP_PG: set from recipe after _GLOO_DP_SHARD_PG is created — same 4 EP ranks,
#   but uses TCP sockets (not Slingshot-11 OFI). Provides user-level synchronization
#   without adding more OFI CQ entries. The TCP poll in gloo's barrier forces the kernel
#   to process pending network events, which may include OFI CQ events that the CCL
#   progress thread missed. This avoids the deadlock without adding more CQ events.
#
#   If gloo barrier alone fails: try gloo barrier + XCCL barrier (belt-and-suspenders),
#   or switch CCL_ATL_TRANSPORT=mpi which has stronger OFI CQ completion guarantees.
import sys
from torch.distributed.distributed_c10d import reduce_scatter_tensor as _c10d_reduce_scatter

_EP_OP_N = 0
_GLOO_EP_PG = None      # 4-rank gloo EP group; set from recipe (_GLOO_DP_SHARD_PG mirror).
_GLOO_GLOBAL_PG = None  # 12-rank global gloo group; set from recipe (v150 — failed).


def _ep_mem_probe(tag: str, n: int):
    """v8g diagnostic: print rank-0 XPU L0-free + torch alloc/resv at each EP-OP boundary.

    Goal: localize where L0 IPC handle pressure spikes through train fwd that crashed v3-v8a
    around op #253-261. Pluggable allocator NOT in use on train ranks (XPU_USM_ALLOC_SO unset),
    so torch.xpu.memory_stats and mem_get_info are valid here.
    """
    try:
        if dist.get_rank() != 0:
            return
        import torch
        free_b, total_b = torch.xpu.mem_get_info()
        alloc_b = torch.xpu.memory_allocated()
        resv_b = torch.xpu.memory_reserved()
        gib = 1024 ** 3
        print(
            f"[MEMPROBE] op={n:>4d} {tag:<10s} "
            f"l0_free={free_b/gib:6.2f}GiB "
            f"torch_alloc={alloc_b/gib:6.2f}GiB "
            f"torch_resv={resv_b/gib:6.2f}GiB "
            f"l0_used={(total_b-free_b)/gib:6.2f}GiB",
            flush=True,
        )
    except Exception as e:
        print(f"[MEMPROBE] op={n} {tag} FAIL {e}", flush=True)


def _ep_reduce_scatter(input: Tensor, group: dist.ProcessGroup, label: str = "RS") -> Tensor:
    """EP ReduceScatter via gloo CPU-bounce (v151).

    Replaces XCCL reduce_scatter_tensor with gloo all_reduce (SUM) + local slice.
    Bypasses XCCL/OFI entirely for EP dispatch — eliminates the OFI CQ deadlock at op #259
    that persisted through v144-v150 (XCCL barriers, gloo barriers, global barrier all failed).
    Uses _GLOO_EP_PG (same gloo group as _GLOO_DP_SHARD_PG, already used for FSDP2 grad sync).
    Cost: 4× more bandwidth (all_reduce on full buffer vs native reduce_scatter), but no deadlock.

    Forward (RS-FWD): partial_out(ep_degree×s_local, dim) → all_reduce → slice → (s_local, dim)
    Backward (RS-BWD): same path on grad_output from AG-FWD.
    """
    import time as _time
    global _EP_OP_N
    n = _EP_OP_N; _EP_OP_N += 1
    r = dist.get_rank()
    print(f"[rank{r}] EP-OP #{n} ENTER {label}", flush=True)
    _ep_mem_probe(f"ENTER-{label}", n)
    ep_degree = dist.get_world_size(group)
    ep_rank = dist.get_rank(group)
    out_rows = input.shape[0] // ep_degree

    if input.device.type == "xpu" and _GLOO_EP_PG is not None:
        # gloo CPU-bounce: XPU → CPU → gloo all_reduce(SUM) → slice → XPU
        _t0 = _time.monotonic()
        input_cpu = input.contiguous().cpu()  # (ep_degree * s_local, dim)
        _t_d2h = _time.monotonic() - _t0
        _t1 = _time.monotonic()
        dist.all_reduce(input_cpu, op=dist.ReduceOp.SUM, group=_GLOO_EP_PG)
        _t_coll = _time.monotonic() - _t1
        _t2 = _time.monotonic()
        out_cpu = input_cpu[ep_rank * out_rows : (ep_rank + 1) * out_rows].contiguous()
        out = out_cpu.to(input.device)
        _t_h2d = _time.monotonic() - _t2
        if _t_d2h > 1.0 or _t_coll > 5.0 or _t_h2d > 1.0:
            print(f"[rank{r}] EP-OP #{n} {label} SLOW d2h={_t_d2h:.2f}s coll={_t_coll:.2f}s h2d={_t_h2d:.2f}s shape={tuple(input.shape)}", flush=True)
    else:
        # Fallback: native XCCL reduce_scatter (non-XPU or no gloo group configured)
        out = input.new_empty(out_rows, *input.shape[1:])
        _c10d_reduce_scatter(out, input.contiguous(), op=dist.ReduceOp.SUM, group=group)

    print(f"[rank{r}] EP-OP #{n} COLL-DONE {label}", flush=True)
    _ep_mem_probe(f"EXIT-{label}", n)
    print(f"[rank{r}] EP-OP #{n} EXIT {label}", flush=True)
    return out


def _ep_all_gather(out: Tensor, input: Tensor, group: dist.ProcessGroup, label: str = "AG") -> None:
    """EP AllGather via gloo CPU-bounce (v151).

    Replaces XCCL all_gather_into_tensor with gloo all_gather_into_tensor on CPU tensors.
    Uses _GLOO_EP_PG (same gloo group already used for FSDP2 grad sync via monkey-patch).

    Forward (AG-FWD): (s_local, dim) → all_gather → (ep_degree×s_local, dim)
    Backward (AG-BWD): same path on grad_output from RS-FWD.
    """
    import time as _time
    global _EP_OP_N
    n = _EP_OP_N; _EP_OP_N += 1
    r = dist.get_rank()
    print(f"[rank{r}] EP-OP #{n} ENTER {label}", flush=True)
    _ep_mem_probe(f"ENTER-{label}", n)

    if input.device.type == "xpu" and _GLOO_EP_PG is not None:
        # gloo CPU-bounce: XPU → CPU → gloo all_gather_into_tensor → XPU
        _t0 = _time.monotonic()
        input_cpu = input.contiguous().cpu()  # (s_local, dim)
        _t_d2h = _time.monotonic() - _t0
        _t1 = _time.monotonic()
        out_cpu = torch.zeros(out.shape, dtype=out.dtype, device="cpu")
        dist.all_gather_into_tensor(out_cpu, input_cpu, group=_GLOO_EP_PG)
        _t_coll = _time.monotonic() - _t1
        _t2 = _time.monotonic()
        out.copy_(out_cpu.to(input.device))
        _t_h2d = _time.monotonic() - _t2
        if _t_d2h > 1.0 or _t_coll > 5.0 or _t_h2d > 1.0:
            print(f"[rank{r}] EP-OP #{n} {label} SLOW d2h={_t_d2h:.2f}s coll={_t_coll:.2f}s h2d={_t_h2d:.2f}s shape={tuple(input.shape)}", flush=True)
    else:
        # Fallback: native XCCL all_gather (non-XPU or no gloo group configured)
        dist.all_gather_into_tensor(out, input.contiguous(), group=group)

    print(f"[rank{r}] EP-OP #{n} COLL-DONE {label}", flush=True)
    _ep_mem_probe(f"EXIT-{label}", n)
    print(f"[rank{r}] EP-OP #{n} EXIT {label}", flush=True)


class _AllGatherRS(torch.autograd.Function):
    """AllGather in forward, ReduceScatter in backward."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.ep_degree = dist.get_world_size(group)
        out = input.new_empty(ctx.ep_degree * input.shape[0], *input.shape[1:])
        _ep_all_gather(out, input, group, label="AG-FWD")
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # v153 diagnostic: confirm rank is about to call RS-BWD (op _EP_OP_N).
        # If a rank prints COLL-DONE for AG-BWD but never prints this, it crashed between them.
        print(f"[rank{dist.get_rank()}] PRE-RS-BWD ep_op={_EP_OP_N}", flush=True)
        return _ep_reduce_scatter(grad_output, ctx.group, label="RS-BWD"), None


class _ReduceScatterAG(torch.autograd.Function):
    """ReduceScatter in forward, AllGather in backward."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup) -> Tensor:
        ctx.group = group
        ctx.ep_degree = dist.get_world_size(group)
        return _ep_reduce_scatter(input, group, label="RS-FWD")

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        out = grad_output.new_empty(
            ctx.ep_degree * grad_output.shape[0], *grad_output.shape[1:]
        )
        _ep_all_gather(out, grad_output, ctx.group, label="AG-BWD")
        return out, None


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
    """Expert Parallelism for MoE layers via AllGather + ReduceScatter.

    Implements the FastSparseMoE algorithm from Mula (arXiv 2604.00785), adapted
    for Aurora/XPU where AllToAll deadlocks or SIGSEGVs in the backward pass.

    Algorithm (per MoE layer):
      Forward:
        1. AllGather routed_input across EP ranks → every rank sees all T=EP*S tokens
        2. Each rank selects tokens for its local experts (interleaved assignment)
        3. Local expert computation on selected tokens
        4. Scatter expert outputs into full (T, dim) partial buffer
        5. ReduceScatter → each rank receives its local S tokens' complete outputs

      Backward (automatic via PyTorch autograd):
        AllGather backward = ReduceScatter
        ReduceScatter backward = AllGather

    No AllToAll, no split tracking, no OFI CQ drain. AllGather/ReduceScatter are
    natively optimized in oneCCL (topology-aware, L0 IPC) and used by FSDP2 on Aurora.

    Usage::

        from torch.distributed.tensor.parallel import parallelize_module
        ep_plan = {"layers.0.moe_block.experts": ExpertParallel()}
        parallelize_module(model, ep_mesh, ep_plan)
    """

    def __init__(self) -> None:
        super().__init__()
        # v159: revert v158 ctx-threading. Instance-cache gather_idx + s_local
        # (one ExpertParallel per layer, so no cross-layer aliasing in practice).
        # v161: also cache all_ri so combine can keep the autograd chain alive
        # back to AllGather output even when GroupedExperts short-circuits on
        # an empty dispatch (rank-1/5/9 #237→#238 deadlock root cause).
        self._ag_gather_idx: Optional[Tensor] = None
        self._ag_s_local: Optional[int] = None
        self._ag_all_ri: Optional[Tensor] = None

    def _partition_fn(
        self, name: str, mod: nn.Module, device_mesh: DeviceMesh
    ) -> None:
        """Shard expert weight matrices on dim 0 (num_experts) across EP ranks.

        Interleaved assignment: rank r owns experts {r, r+ep_degree, r+2*ep_degree, ...}.
        This distributes hot experts (typically clustered at low indices for pretrained
        routers) evenly across all ranks.
        """
        ep_rank = device_mesh.get_local_rank()
        ep_degree = device_mesh.shape[0]
        for param_name, param in list(mod.named_parameters(recurse=False)):
            full_data = param.data  # (num_experts, ...)
            assert full_data.shape[0] % ep_degree == 0, (
                f"num_experts ({full_data.shape[0]}) must be divisible by ep_degree ({ep_degree})"
            )
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
        """AllGather dispatch: every rank gets all tokens, selects tokens for local experts.

        Args:
            mod: ``GroupedExperts`` module (unused; retained for API consistency).
            routed_input: Pre-weighted tokens in expert-sorted order, shape ``(S, dim)``
                where ``S = bs * slen * top_k``. Same shape on all EP ranks.
            num_tokens_per_expert: Token counts per global expert, shape ``(num_experts,)``.
            device_mesh: EP mesh of shape ``(ep_degree,)``.

        Returns:
            ``(dispatched_tokens, local_ntpe)`` where dispatched_tokens is in
            expert-major order for the local experts, shape ``(total_local, dim)``.
        """
        ep_degree = device_mesh.shape[0]
        ep_rank = device_mesh.get_local_rank()
        num_experts = num_tokens_per_expert.shape[0]
        num_local_experts = num_experts // ep_degree
        group = device_mesh.get_group()
        s_local = routed_input.shape[0]  # S = bs*slen*top_k (same on all EP ranks)

        # Stage 1: AllGather all tokens across EP ranks (gradient-tracked).
        # all_ri[r*s_local : (r+1)*s_local] = rank r's routed_input (expert-sorted).
        # Backward: ReduceScatter (via _AllGatherRS.backward).
        all_ri = _AllGatherRS.apply(routed_input.contiguous(), group)

        # AllGather num_tokens_per_expert (no grad needed).
        # all_ntpe[r, e] = tokens rank r routes to global expert e.
        all_ntpe_flat = torch.zeros(
            ep_degree * num_experts, dtype=torch.long, device=routed_input.device
        )
        # v153: gloo CPU-bounce for NTPE AllGather (was XCCL).
        # XCCL uses OFI/CXI NIC. After 256+ EP gloo ops, the CXI NIC may have residual
        # OFI CQ entries that contaminate gloo TCP (which also goes through CXI on Aurora).
        # Replacing with gloo CPU-bounce isolates NTPE from the OFI/CXI stack.
        # The GLOO_SOCKET_IFNAME=lo env var forces gloo to use loopback (not CXI).
        # v8: count NTPE-AG against _EP_OP_N. Without this, every per-layer dispatch
        # fires AG-FWD + NTPE-AG + RS-FWD against a counter that increments only twice
        # → forensics are off-by-one-per-layer across all ranks (consistent across
        # ranks, so it doesn't itself cause desync, but it does make logs misleading).
        global _EP_OP_N
        n_ntpe = _EP_OP_N; _EP_OP_N += 1
        r_ntpe = dist.get_rank()
        print(f"[rank{r_ntpe}] EP-OP #{n_ntpe} ENTER NTPE-AG", flush=True)
        if routed_input.device.type == "xpu" and _GLOO_EP_PG is not None:
            ntpe_cpu = num_tokens_per_expert.to(torch.long).contiguous().cpu()
            all_ntpe_cpu = torch.zeros(ep_degree * num_experts, dtype=torch.long, device="cpu")
            dist.all_gather_into_tensor(all_ntpe_cpu, ntpe_cpu, group=_GLOO_EP_PG)
            all_ntpe_flat.copy_(all_ntpe_cpu)
        else:
            dist.all_gather_into_tensor(
                all_ntpe_flat, num_tokens_per_expert.to(torch.long), group=group
            )
        print(f"[rank{r_ntpe}] EP-OP #{n_ntpe} EXIT NTPE-AG", flush=True)
        all_ntpe = all_ntpe_flat.view(ep_degree, num_experts)

        # Cumulative token offsets within each rank's expert-sorted section.
        # all_ntpe_cumsum[r, g] = number of tokens rank r sends to experts 0..g-1
        #                       = start index of expert g in rank r's slice of all_ri.
        all_ntpe_cumsum = torch.zeros(
            ep_degree, num_experts + 1, dtype=torch.long, device=routed_input.device
        )
        all_ntpe_cumsum[:, 1:] = torch.cumsum(all_ntpe, dim=1)

        # Stage 2-3: Build gather indices for local experts (interleaved assignment).
        # Rank ep_rank owns global experts: ep_rank, ep_rank+ep_degree, ..., ep_rank+(NLE-1)*ep_degree.
        # For each local expert g, collect its token positions from all EP ranks.
        with torch.no_grad():
            local_exp_indices: list[Tensor] = []
            local_ntpe_list: list[int] = []

            for local_exp_idx in range(num_local_experts):
                g = ep_rank + local_exp_idx * ep_degree  # global expert index (interleaved)
                parts: list[Tensor] = []
                count_total = 0
                for r in range(ep_degree):
                    count = int(all_ntpe[r, g].item())
                    if count > 0:
                        start_in_r = int(all_ntpe_cumsum[r, g].item())
                        abs_start = r * s_local + start_in_r
                        parts.append(
                            torch.arange(
                                abs_start, abs_start + count,
                                dtype=torch.long, device=routed_input.device,
                            )
                        )
                        count_total += count
                local_exp_indices.append(
                    torch.cat(parts) if parts else
                    torch.zeros(0, dtype=torch.long, device=routed_input.device)
                )
                local_ntpe_list.append(count_total)

            # Concatenate in expert-major order (exp0 tokens, exp1 tokens, ...).
            # Empty expert tensors (numel=0) are handled correctly by torch.cat.
            gather_idx = torch.cat(local_exp_indices)
            local_ntpe = torch.tensor(
                local_ntpe_list, dtype=all_ntpe.dtype, device=routed_input.device
            )

        # Stage 4: Gather tokens for local experts (gradient flows through indexing).
        # v160: always go through index-gather (even with empty gather_idx) to
        # keep an autograd link to all_ri. Without this, an empty-dispatch rank
        # detaches its entire dispatch op from the loss graph → engine skips
        # _AllGatherRS.backward → asymmetric early-exit deadlock at next AG-BWD.
        # all_ri[empty_tensor] returns shape (0, dim) WITH a grad-fn (IndexBackward).
        dispatched = all_ri[gather_idx]

        # v159: cache on the ExpertParallel instance for retrieval in combine.
        # v161: also cache all_ri so combine can re-bind partial_out's autograd
        # to all_ri (and thus _AllGatherRS), preventing rank-1 from skipping
        # backward when expert local count is 0.
        self._ag_gather_idx = gather_idx
        self._ag_s_local = s_local
        self._ag_all_ri = all_ri
        # Per-layer instrumentation: which layer index has empty dispatch on
        # which rank. Cheap (12 ranks * 30 layers * 2 chunks * 4 mb = ~3k lines).
        try:
            r = dist.get_rank()
            n_local = int(gather_idx.shape[0])
            print(
                f"[rank{r}] EP-DISPATCH n_local={n_local} s_local={s_local} dispatched.shape={tuple(dispatched.shape)} requires_grad={dispatched.requires_grad} grad_fn={type(dispatched.grad_fn).__name__ if dispatched.grad_fn is not None else 'None'}",
                flush=True,
            )
        except Exception:
            pass
        return dispatched, local_ntpe

    def _token_combine(
        self,
        mod: nn.Module,
        routed_output: Tensor,
        *,
        device_mesh: DeviceMesh,
    ) -> Tensor:
        """ReduceScatter combine: accumulate local expert outputs and return local slice.

        Args:
            mod: ``GroupedExperts`` module (unused; retained for API consistency).
            routed_output: Expert outputs in expert-major order, shape ``(total_local, dim)``.
            device_mesh: EP mesh of shape ``(ep_degree,)``.

        Returns:
            Combined output in original routed_input order, shape ``(S, dim)``
            where ``S = bs * slen * top_k``.
        """
        ep_degree = device_mesh.shape[0]
        group = device_mesh.get_group()
        # v159: read indices off the ExpertParallel instance (set in _token_dispatch).
        s_local = self._ag_s_local
        gather_idx = self._ag_gather_idx
        all_ri = self._ag_all_ri  # v161: AllGather output, kept alive for autograd binding.

        # Stage 5a: Scatter expert outputs back to their positions in the full (T, dim) buffer.
        # partial_out[i] = expert output for the token that was at all_ri[i].
        # Positions not owned by this rank's local experts remain zero.
        # Since the pre-weighted tokens are already scaled, no extra weighting needed here.
        partial_out = routed_output.new_zeros(ep_degree * s_local, routed_output.shape[-1])
        # v160: ALWAYS run scatter_add (even with empty gather_idx) so partial_out
        # carries an autograd link back to routed_output. With empty gather_idx,
        # scatter_add is a no-op on values but still emits a grad-fn — that
        # grad-fn is what keeps the EP backward chain connected. Without it,
        # _ReduceScatterAG.backward gets skipped on the empty-dispatch rank →
        # asymmetric early-exit deadlock at next AG-BWD (v158/v159 reproduced).
        idx_exp = gather_idx.unsqueeze(1).expand_as(routed_output)
        partial_out = partial_out.scatter_add(0, idx_exp, routed_output)
        # v161: HARD AUTOGRAD ANCHOR. Even with the v160 scatter_add grad-fn,
        # an empty-dispatch rank's _AllGatherRS.backward never fires because
        # downstream gradients arrive at routed_output as zero (no expert
        # produced anything) and the chain back to all_ri only goes via the
        # gather index. By adding 0.0 * all_ri we force partial_out to depend
        # on all_ri at the autograd-graph level — the engine will then ALWAYS
        # call _AllGatherRS.backward (which feeds RS-BWD), keeping rank-1 in
        # lockstep with peers at #237 RS-BWD AND #238 AG-BWD.
        if all_ri is not None and all_ri.requires_grad:
            partial_out = partial_out + all_ri.sum(dim=0, keepdim=True).expand_as(partial_out) * 0.0
        # Diagnostic: confirm partial_out has a grad-fn that reaches AllGather.
        try:
            r = dist.get_rank()
            print(
                f"[rank{r}] EP-COMBINE partial_out.shape={tuple(partial_out.shape)} requires_grad={partial_out.requires_grad} grad_fn={type(partial_out.grad_fn).__name__ if partial_out.grad_fn is not None else 'None'} n_local={int(gather_idx.shape[0])}",
                flush=True,
            )
        except Exception:
            pass

        # Stage 5b: ReduceScatter — sum partial outputs across EP ranks.
        # Rank r receives partial_out[r*s_local:(r+1)*s_local] summed over all EP ranks.
        # Each position has exactly one non-zero contributor (its expert's owning rank),
        # so the sum is that rank's expert output.
        # Backward: AllGather (via _ReduceScatterAG.backward).
        out = _ReduceScatterAG.apply(partial_out, group)  # (s_local, dim)
        return out

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

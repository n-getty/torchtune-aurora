# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
import os
import sys
from contextlib import nullcontext
from functools import partial
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.distributed.distributed_c10d import (
    reduce_scatter_tensor as _c10d_reduce_scatter,
)
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

_EP_OP_N = 0
_DEVICE_ROUTING_METADATA = (
    os.environ.get("TORCHTUNE_EP_DEVICE_ROUTING_METADATA", "0") == "1"
)
_CPU_VECTOR_ROUTING_METADATA = (
    os.environ.get("TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA", "1") == "1"
)
_PACK_ROUTING_METADATA_TRANSFER = (
    os.environ.get("TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER", "1") == "1"
)
_GLOO_EP_PG = None  # 4-rank gloo EP group; set from recipe (_GLOO_DP_SHARD_PG mirror).
_GLOO_GLOBAL_PG = None  # 12-rank global gloo group; set from recipe (v150 — failed).

# Per-op tracing/MEMPROBE/DISPATCH/COMBINE prints are silent unless
# TORCHTUNE_EP_DEBUG=1. SLOW-threshold prints in _ep_all_gather/_ep_reduce_scatter
# remain unconditional — they only fire when a single op exceeds 1s d2h, 5s coll,
# or 1s h2d, which is a real performance signal.
_EP_DEBUG = os.environ.get("TORCHTUNE_EP_DEBUG", "0") == "1"

# Opt-in: route _ep_all_gather and _ep_reduce_scatter through native XCCL on
# the EP process group instead of the v151 gloo CPU-bounce. Default False
# because v141-v150 hit a deterministic OFI CQ deadlock at op #259 (RS-BWD)
# on this exact path. Use only after running the EP smokes (see plan); the
# gloo path remains the safe production default.
_EP_USE_XCCL = os.environ.get("TORCHTUNE_EP_USE_XCCL", "0") == "1"
_USE_INPLACE_AG_ANCHOR = (
    os.environ.get("TORCHTUNE_EP_INPLACE_AG_ANCHOR", "1") == "1"
)
_USE_SINGLE_ROW_AG_ANCHOR = (
    os.environ.get("TORCHTUNE_EP_SINGLE_ROW_AG_ANCHOR", "1") == "1"
)
_USE_ZERO_COST_AG_ANCHOR = (
    os.environ.get("TORCHTUNE_EP_ZERO_COST_AG_ANCHOR", "1") == "1"
)
_USE_UNINITIALIZED_COLLECTIVE_BUFFERS = (
    os.environ.get("TORCHTUNE_EP_UNINITIALIZED_COLLECTIVE_BUFFERS", "1") == "1"
)
_USE_CPU_METADATA_TRANSFER = (
    os.environ.get("TORCHTUNE_EP_CPU_METADATA_TRANSFER", "1") == "1"
)
_USE_DIRECT_CPU_COPY = os.environ.get("TORCHTUNE_EP_DIRECT_CPU_COPY", "1") == "1"
_USE_INDEX_ADD_COMBINE = (
    os.environ.get("TORCHTUNE_EP_INDEX_ADD_COMBINE", "1") == "1"
)
_USE_ROWWISE_ALLTOALL_UNPERMUTE = (
    os.environ.get("TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE", "1") == "1"
)
_USE_UNINITIALIZED_ALLTOALL_COMBINE_BUFFERS = (
    os.environ.get("TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS", "1") == "1"
)
_CONDITIONAL_ALLTOALL_CONTIGUOUS = (
    os.environ.get("TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS", "1") == "1"
)
_USE_FUSED_ALLTOALL_ROUTING = (
    os.environ.get("TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING", "0") == "1"
)


def _materialize_routing_metadata(
    cpu_metadata: Tensor, device: torch.device, *, direct_cpu_transfer: bool
) -> Tensor:
    """Materialize CPU routing metadata on ``device`` using one of the A/B paths."""
    if direct_cpu_transfer:
        return cpu_metadata.to(device=device)
    device_metadata = torch.empty(
        cpu_metadata.numel(), dtype=cpu_metadata.dtype, device=device
    )
    device_metadata.copy_(cpu_metadata)
    return device_metadata


def _materialize_all_to_all_permutations(
    send_perm: Tensor,
    expert_perm: Tensor,
    local_ntpe: Tensor,
    device: torch.device,
    *,
    packed_transfer: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Materialize AllToAll metadata with an optional single host transfer."""
    if not packed_transfer or device.type == "cpu":
        return (
            send_perm.to(device=device),
            expert_perm.to(device=device),
            local_ntpe.to(device=device),
        )
    lengths = (send_perm.numel(), expert_perm.numel(), local_ntpe.numel())
    packed = torch.cat((send_perm, expert_perm, local_ntpe), dim=0).to(device=device)
    send_end = lengths[0]
    expert_end = send_end + lengths[1]
    return packed[:send_end], packed[send_end:expert_end], packed[expert_end:]


def _copy_cpu_collective_output(
    destination: Tensor, source: Tensor, *, direct_cpu_copy: bool
) -> Tensor:
    """Copy a CPU collective result into its destination using the selected A/B path."""
    if direct_cpu_copy:
        destination.copy_(source)
    else:
        destination.copy_(source.to(destination.device))
    return destination

# Opt-in: replace the AllGather+ReduceScatter EP dispatch/combine with a true
# all_to_all_single dispatch on native XCCL (the modern-standard EP pattern used
# by torchtitan/Megatron). AG+RS moves O(ep_degree * S) bytes per rank (every rank
# materializes ALL tokens); all_to_all moves only O(S) — each rank sends a token to
# exactly the one rank that owns its expert. Validated viable on frameworks/2025.3.1
# (2026-07-17): the historic v18-v136 OFI-CQ deadlock at op #259 does NOT reproduce,
# incl. under 3 concurrent EP groups (repro_all2all_concurrent_groups.py). Uses
# torch.distributed._functional_collectives.all_to_all_single (ft_c) — it is
# autograd-aware (its backward is the transposed all_to_all), so unlike the AG+RS
# path this needs NO manual empty-dispatch autograd anchoring: every rank issues
# exactly one all_to_all in dispatch and one in combine, unconditionally, keeping
# ranks in lockstep by construction. Default False (AG+RS remains production default
# until this path is HW-A/B'd on Qwen3-30B-A3B EP=8/16). See CLAUDE.md env table.
_EP_ALL2ALL = os.environ.get("TORCHTUNE_EP_ALL2ALL", "0") == "1"

# v164 (2026-07-24): opt-in periodic gc.collect()+synchronize on the EP gloo
# CPU-bounce path, gated on a raw EP-OP count rather than steps/microbatches
# (mirrors the WS10 weight-sync fix in torchtune/dev/rl/weight_sync.py, which
# solved a structurally similar accumulate-then-die crash with the same
# "gc every N collective ops" pattern). Motivated by an un-chunked SFT
# regime (full_finetune_moe_distributed_xpu.py, single .backward() per step,
# no forward_batch_size chunking) hitting a `banned:1` XPU driver segfault
# DETERMINISTICALLY at EP-OP #329 every time, preceded by one abnormally
# slow (~11s vs sub-100ms) RS-FWD collective — reproduced identically across
# batch_size=1/2, TORCHTUNE_EP_GRAD_RELEASE_XCCL on/off, and GLOO_SOCKET_IFNAME
# routing, ruling those out as the cause and pointing at some per-collective
# CPU-side resource (Python object graph, gloo buffer, or similar) that
# never gets released within a step. Default OFF (0 = disabled, matches
# existing behavior exactly) since this is unvalidated as of authoring;
# EP=8 GRPO's chunked/`forward_batch_size`-gated path has never hit this,
# so there is no evidence yet that gc is even the right fix — only that the
# WS10 precedent is the closest analog. Value N means "gc.collect() +
# torch.xpu.synchronize() every N EP-OPs"; 0 (falsy) means disabled.
_EP_GC_EVERY_N_OPS = int(os.environ.get("TORCHTUNE_EP_GC_EVERY_N_OPS", "0") or "0")


def _ep_maybe_gc(n: int, device_type: str) -> None:
    """Opt-in periodic gc.collect()+synchronize, gated by TORCHTUNE_EP_GC_EVERY_N_OPS."""
    if _EP_GC_EVERY_N_OPS and n > 0 and n % _EP_GC_EVERY_N_OPS == 0:
        import gc

        gc.collect()
        if device_type == "xpu":
            torch.xpu.synchronize()


def _ep_mem_probe(tag: str, n: int):
    """Rank-0 XPU L0-free + torch alloc/resv probe at each EP-OP boundary.

    Originally added (v8g) to localize L0 IPC handle pressure spikes through
    train fwd. Gated behind TORCHTUNE_EP_DEBUG=1; off by default.
    """
    if not _EP_DEBUG:
        return
    try:
        if dist.get_rank() != 0:
            return
        import torch

        free_b, total_b = torch.xpu.mem_get_info()
        alloc_b = torch.xpu.memory_allocated()
        resv_b = torch.xpu.memory_reserved()
        gib = 1024**3
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


def _ep_reduce_scatter(
    input: Tensor, group: dist.ProcessGroup, label: str = "RS"
) -> Tensor:
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
    n = _EP_OP_N
    _EP_OP_N += 1
    r = dist.get_rank()
    if _EP_DEBUG:
        print(f"[rank{r}] EP-OP #{n} ENTER {label}", flush=True)
        _ep_mem_probe(f"ENTER-{label}", n)
    ep_degree = dist.get_world_size(group)
    ep_rank = dist.get_rank(group)
    out_rows = input.shape[0] // ep_degree

    if _EP_USE_XCCL and input.device.type == "xpu":
        # Opt-in: native XCCL reduce_scatter on the EP group, on-device.
        # See module-level _EP_USE_XCCL note for the v141-v150 deadlock history.
        out = input.new_empty(out_rows, *input.shape[1:])
        _t1 = _time.monotonic()
        _c10d_reduce_scatter(out, input.contiguous(), op=dist.ReduceOp.SUM, group=group)
        _t_coll = _time.monotonic() - _t1
        if _t_coll > 5.0:
            print(
                f"[rank{r}] EP-OP #{n} {label} SLOW xccl_coll={_t_coll:.2f}s shape={tuple(input.shape)}",
                flush=True,
            )
    elif input.device.type == "xpu" and _GLOO_EP_PG is not None:
        # gloo CPU-bounce: XPU → CPU → gloo all_reduce(SUM) → slice → XPU
        _t0 = _time.monotonic()
        input_cpu = input.contiguous().cpu()  # (ep_degree * s_local, dim)
        _t_d2h = _time.monotonic() - _t0
        _t1 = _time.monotonic()
        dist.all_reduce(input_cpu, op=dist.ReduceOp.SUM, group=_GLOO_EP_PG)
        _t_coll = _time.monotonic() - _t1
        _t2 = _time.monotonic()
        out = input.new_empty(out_rows, *input.shape[1:])
        out_cpu = input_cpu[ep_rank * out_rows : (ep_rank + 1) * out_rows]
        if _USE_DIRECT_CPU_COPY:
            _copy_cpu_collective_output(out, out_cpu, direct_cpu_copy=True)
        else:
            out = _copy_cpu_collective_output(
                out, out_cpu.contiguous(), direct_cpu_copy=False
            )
        _t_h2d = _time.monotonic() - _t2
        if _t_d2h > 1.0 or _t_coll > 5.0 or _t_h2d > 1.0:
            print(
                f"[rank{r}] EP-OP #{n} {label} SLOW d2h={_t_d2h:.2f}s coll={_t_coll:.2f}s h2d={_t_h2d:.2f}s shape={tuple(input.shape)}",
                flush=True,
            )
    else:
        # Fallback: native XCCL reduce_scatter (non-XPU or no gloo group configured)
        out = input.new_empty(out_rows, *input.shape[1:])
        _c10d_reduce_scatter(out, input.contiguous(), op=dist.ReduceOp.SUM, group=group)

    if _EP_DEBUG:
        print(f"[rank{r}] EP-OP #{n} COLL-DONE {label}", flush=True)
        _ep_mem_probe(f"EXIT-{label}", n)
        print(f"[rank{r}] EP-OP #{n} EXIT {label}", flush=True)
    _ep_maybe_gc(n, input.device.type)
    return out


def _ep_all_gather(
    out: Tensor, input: Tensor, group: dist.ProcessGroup, label: str = "AG"
) -> None:
    """EP AllGather via gloo CPU-bounce (v151).

    Replaces XCCL all_gather_into_tensor with gloo all_gather_into_tensor on CPU tensors.
    Uses _GLOO_EP_PG (same gloo group already used for FSDP2 grad sync via monkey-patch).

    Forward (AG-FWD): (s_local, dim) → all_gather → (ep_degree×s_local, dim)
    Backward (AG-BWD): same path on grad_output from RS-FWD.
    """
    import time as _time

    global _EP_OP_N
    n = _EP_OP_N
    _EP_OP_N += 1
    r = dist.get_rank()
    if _EP_DEBUG:
        print(f"[rank{r}] EP-OP #{n} ENTER {label}", flush=True)
        _ep_mem_probe(f"ENTER-{label}", n)

    if _EP_USE_XCCL and input.device.type == "xpu":
        # Opt-in: native XCCL all_gather on the EP group, on-device.
        # See module-level _EP_USE_XCCL note for the v141-v150 deadlock history.
        _t1 = _time.monotonic()
        dist.all_gather_into_tensor(out, input.contiguous(), group=group)
        _t_coll = _time.monotonic() - _t1
        if _t_coll > 5.0:
            print(
                f"[rank{r}] EP-OP #{n} {label} SLOW xccl_coll={_t_coll:.2f}s shape={tuple(input.shape)}",
                flush=True,
            )
    elif input.device.type == "xpu" and _GLOO_EP_PG is not None:
        # gloo CPU-bounce: XPU → CPU → gloo all_gather_into_tensor → XPU
        _t0 = _time.monotonic()
        input_cpu = input.contiguous().cpu()  # (s_local, dim)
        _t_d2h = _time.monotonic() - _t0
        _t1 = _time.monotonic()
        allocate = torch.empty if _USE_UNINITIALIZED_COLLECTIVE_BUFFERS else torch.zeros
        out_cpu = allocate(out.shape, dtype=out.dtype, device="cpu")
        dist.all_gather_into_tensor(out_cpu, input_cpu, group=_GLOO_EP_PG)
        _t_coll = _time.monotonic() - _t1
        _t2 = _time.monotonic()
        _copy_cpu_collective_output(
            out, out_cpu, direct_cpu_copy=_USE_DIRECT_CPU_COPY
        )
        _t_h2d = _time.monotonic() - _t2
        if _t_d2h > 1.0 or _t_coll > 5.0 or _t_h2d > 1.0:
            print(
                f"[rank{r}] EP-OP #{n} {label} SLOW d2h={_t_d2h:.2f}s coll={_t_coll:.2f}s h2d={_t_h2d:.2f}s shape={tuple(input.shape)}",
                flush=True,
            )
    else:
        # Fallback: native XCCL all_gather (non-XPU or no gloo group configured)
        dist.all_gather_into_tensor(out, input.contiguous(), group=group)

    if _EP_DEBUG:
        print(f"[rank{r}] EP-OP #{n} COLL-DONE {label}", flush=True)
        _ep_mem_probe(f"EXIT-{label}", n)
        print(f"[rank{r}] EP-OP #{n} EXIT {label}", flush=True)
    _ep_maybe_gc(n, input.device.type)


class _AllGatherRS(torch.autograd.Function):
    """AllGather in forward, ReduceScatter in backward."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup, measurement) -> Tensor:
        ctx.group = group
        ctx.measurement = measurement
        ctx.ep_degree = dist.get_world_size(group)
        out = input.new_empty(ctx.ep_degree * input.shape[0], *input.shape[1:])
        if measurement is not None:
            with measurement.collective(
                "allgather_forward",
                scope="ep",
                backend=dist.get_backend(group),
            ):
                _ep_all_gather(out, input, group, label="AG-FWD")
        else:
            _ep_all_gather(out, input, group, label="AG-FWD")
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if _EP_DEBUG:
            # v153 diagnostic: confirm rank is about to call RS-BWD (op _EP_OP_N).
            # If a rank prints COLL-DONE for AG-BWD but never prints this, it crashed between them.
            print(f"[rank{dist.get_rank()}] PRE-RS-BWD ep_op={_EP_OP_N}", flush=True)
        if ctx.measurement is not None:
            with ctx.measurement.collective(
                "reduce_scatter_backward",
                scope="ep",
                backend=dist.get_backend(ctx.group),
            ):
                grad_input = _ep_reduce_scatter(
                    grad_output, ctx.group, label="RS-BWD"
                )
        else:
            grad_input = _ep_reduce_scatter(
                grad_output, ctx.group, label="RS-BWD"
            )
        return grad_input, None, None


class _ReduceScatterAG(torch.autograd.Function):
    """ReduceScatter in forward, AllGather in backward."""

    @staticmethod
    def forward(ctx, input: Tensor, group: dist.ProcessGroup, measurement) -> Tensor:
        ctx.group = group
        ctx.measurement = measurement
        ctx.ep_degree = dist.get_world_size(group)
        if measurement is not None:
            with measurement.collective(
                "reduce_scatter_forward",
                scope="ep",
                backend=dist.get_backend(group),
            ):
                return _ep_reduce_scatter(input, group, label="RS-FWD")
        else:
            return _ep_reduce_scatter(input, group, label="RS-FWD")

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        out = grad_output.new_empty(
            ctx.ep_degree * grad_output.shape[0], *grad_output.shape[1:]
        )
        if ctx.measurement is not None:
            with ctx.measurement.collective(
                "allgather_backward",
                scope="ep",
                backend=dist.get_backend(ctx.group),
            ):
                _ep_all_gather(out, grad_output, ctx.group, label="AG-BWD")
        else:
            _ep_all_gather(out, grad_output, ctx.group, label="AG-BWD")
        return out, None, None


def _raw_all_to_all_single(
    send: Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> Tensor:
    """Non-autograd all_to_all_single (forward primitive only).

    On XPU, stock ``dist.all_to_all_single`` is bit-exact and live on
    frameworks/2025.3.1 (validated 2026-07-17, incl. concurrent EP groups). We
    call it directly and supply our OWN backward via ``_AllToAllSingle`` because
    the ft_c functional-collective autograd wrapper does NOT populate grads on
    this build (grad came back None in isolation — gloo and XPU both).
    """
    out = send.new_empty(int(sum(output_splits)), *send.shape[1:])
    dist.all_to_all_single(
        out,
        (
            send
            if _CONDITIONAL_ALLTOALL_CONTIGUOUS and send.is_contiguous()
            else send.contiguous()
        ),
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )
    return out


class _AllToAllSingle(torch.autograd.Function):
    """all_to_all_single with explicit transposed-all_to_all backward.

    Forward sends ``input_splits`` and receives ``output_splits``; the gradient
    of an all_to_all is another all_to_all with the split roles SWAPPED (each
    rank sends back exactly what it received, to where it came from). This mirrors
    the ``_AllGatherRS`` / ``_ReduceScatterAG`` duality used by the AG+RS path,
    and does not rely on ft_c autograd (which is a no-op on this build).
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        output_splits,
        input_splits,
        group,
        timing_label: str,
        measurement,
    ) -> Tensor:
        ctx.output_splits = list(output_splits)
        ctx.input_splits = list(input_splits)
        ctx.group = group
        ctx.timing_label = timing_label
        ctx.measurement = measurement
        return _raw_all_to_all_single(input, output_splits, input_splits, group)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # backward: send what we received (output_splits) and receive what we
        # originally sent (input_splits) — the transpose.
        if ctx.measurement is not None:
            backend = dist.get_backend(ctx.group)
            timing = ctx.measurement.collective(
                f"{ctx.timing_label}_backward_alltoall",
                scope="ep",
                backend=backend,
            )
        else:
            timing = nullcontext()
        with timing:
            grad_input = _raw_all_to_all_single(
                (
                    grad_output
                    if _CONDITIONAL_ALLTOALL_CONTIGUOUS and grad_output.is_contiguous()
                    else grad_output.contiguous()
                ),
                ctx.input_splits,
                ctx.output_splits,
                ctx.group,
            )
        return grad_input, None, None, None, None, None


def _ep_all_to_all_single(
    send: Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
    timing_label: str,
    measurement=None,
) -> Tensor:
    """Autograd-aware all_to_all_single on the EP group (custom Function).

    Args:
        send: local send buffer, expert/dest-rank contiguous, shape ``(sum(input_splits), dim)``.
        output_splits: per-source-rank row counts THIS rank will receive.
        input_splits: per-dest-rank row counts THIS rank will send.
        group: EP process group.

    Returns:
        Received tensor, shape ``(sum(output_splits), dim)``, source-rank contiguous.
    """
    return _AllToAllSingle.apply(
        send, output_splits, input_splits, group, timing_label, measurement
    )


class _FusedAllToAllRouting(torch.autograd.Function):
    """Fuse dispatch packing, AllToAll, and expert-major unpacking."""

    @staticmethod
    def forward(
        ctx,
        routed_input: Tensor,
        send_perm: Tensor,
        expert_perm: Tensor,
        input_splits,
        output_splits,
        group,
        measurement,
    ) -> Tensor:
        ctx.send_perm = send_perm
        ctx.expert_perm = expert_perm
        ctx.input_splits = list(input_splits)
        ctx.output_splits = list(output_splits)
        ctx.group = group
        ctx.measurement = measurement
        pack_timing = (
            measurement.time("dispatch_pack")
            if measurement is not None
            else nullcontext()
        )
        with pack_timing:
            send = routed_input[send_perm]
        if measurement is not None:
            timing = measurement.collective(
                "dispatch_alltoall",
                scope="ep",
                backend=dist.get_backend(group),
            )
        else:
            timing = nullcontext()
        with timing:
            received = _raw_all_to_all_single(
                send, output_splits, input_splits, group
            )
        unpack_timing = (
            measurement.time("dispatch_unpack")
            if measurement is not None
            else nullcontext()
        )
        with unpack_timing:
            return received[expert_perm]

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        pack_timing = (
            ctx.measurement.time("dispatch_backward_pack")
            if ctx.measurement is not None
            else nullcontext()
        )
        with pack_timing:
            source_major = grad_output.new_empty(
                ctx.expert_perm.numel(), grad_output.shape[-1]
            )
            source_major.index_copy_(0, ctx.expert_perm, grad_output)
        if ctx.measurement is not None:
            timing = ctx.measurement.collective(
                "dispatch_backward_alltoall",
                scope="ep",
                backend=dist.get_backend(ctx.group),
            )
        else:
            timing = nullcontext()
        with timing:
            returned = _raw_all_to_all_single(
                source_major,
                ctx.input_splits,
                ctx.output_splits,
                ctx.group,
            )
        unpack_timing = (
            ctx.measurement.time("dispatch_backward_unpack")
            if ctx.measurement is not None
            else nullcontext()
        )
        with unpack_timing:
            grad_input = grad_output.new_empty(
                ctx.send_perm.numel(), grad_output.shape[-1]
            )
            grad_input.index_copy_(0, ctx.send_perm, returned)
        return grad_input, None, None, None, None, None, None


def _fused_all_to_all_routing(
    routed_input: Tensor,
    send_perm: Tensor,
    expert_perm: Tensor,
    input_splits: list[int],
    output_splits: list[int],
    group: dist.ProcessGroup,
    measurement=None,
) -> Tensor:
    """Apply the opt-in fused dispatch routing operation."""
    return _FusedAllToAllRouting.apply(
        routed_input,
        send_perm,
        expert_perm,
        input_splits,
        output_splits,
        group,
        measurement,
    )


class _FusedAllToAllCombine(torch.autograd.Function):
    """Fuse expert-major combine unpermutation, AllToAll, and final scatter."""

    @staticmethod
    def forward(
        ctx,
        routed_output: Tensor,
        send_perm: Tensor,
        expert_perm: Tensor,
        input_splits,
        output_splits,
        group,
        measurement,
        output_rows: int,
    ) -> Tensor:
        ctx.send_perm = send_perm
        ctx.expert_perm = expert_perm
        ctx.input_splits = list(input_splits)
        ctx.output_splits = list(output_splits)
        ctx.group = group
        ctx.measurement = measurement
        pack_timing = (
            measurement.time("combine_pack")
            if measurement is not None
            else nullcontext()
        )
        with pack_timing:
            source_major = routed_output.new_empty(
                expert_perm.numel(), routed_output.shape[-1]
            )
            source_major.index_copy_(0, expert_perm, routed_output)
        if measurement is not None:
            timing = measurement.collective(
                "combine_alltoall",
                scope="ep",
                backend=dist.get_backend(group),
            )
        else:
            timing = nullcontext()
        with timing:
            returned = _raw_all_to_all_single(
                source_major, output_splits, input_splits, group
            )
        unpack_timing = (
            measurement.time("combine_unpack")
            if measurement is not None
            else nullcontext()
        )
        with unpack_timing:
            output = routed_output.new_empty(output_rows, routed_output.shape[-1])
            output.index_copy_(0, send_perm, returned)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        pack_timing = (
            ctx.measurement.time("combine_backward_pack")
            if ctx.measurement is not None
            else nullcontext()
        )
        with pack_timing:
            returned_grad = grad_output.index_select(0, ctx.send_perm)
        if ctx.measurement is not None:
            timing = ctx.measurement.collective(
                "combine_backward_alltoall",
                scope="ep",
                backend=dist.get_backend(ctx.group),
            )
        else:
            timing = nullcontext()
        with timing:
            source_major_grad = _raw_all_to_all_single(
                returned_grad,
                ctx.input_splits,
                ctx.output_splits,
                ctx.group,
            )
        unpack_timing = (
            ctx.measurement.time("combine_backward_unpack")
            if ctx.measurement is not None
            else nullcontext()
        )
        with unpack_timing:
            grad_routed_output = source_major_grad.index_select(0, ctx.expert_perm)
        return grad_routed_output, None, None, None, None, None, None, None


def _fused_all_to_all_combine(
    routed_output: Tensor,
    send_perm: Tensor,
    expert_perm: Tensor,
    input_splits: list[int],
    output_splits: list[int],
    group: dist.ProcessGroup,
    measurement=None,
    output_rows: int = 0,
) -> Tensor:
    """Apply the opt-in fused combine operation."""
    return _FusedAllToAllCombine.apply(
        routed_output,
        send_perm,
        expert_perm,
        input_splits,
        output_splits,
        group,
        measurement,
        output_rows,
    )


def _build_all_to_all_metadata(
    all_ntpe_cpu: Tensor, ep_rank: int
) -> tuple[Tensor, Tensor, list[int], list[int], list[int]]:
    """Build all-to-all routing permutations from CPU token counts."""
    ep_degree, num_experts = all_ntpe_cpu.shape
    num_local_experts = num_experts // ep_degree
    counts = all_ntpe_cpu.tolist()

    my_counts = counts[ep_rank]
    expert_offsets = [0]
    for count in my_counts:
        expert_offsets.append(expert_offsets[-1] + count)

    send_perm_values = []
    input_splits = []
    for destination in range(ep_degree):
        destination_count = 0
        for local_expert in range(num_local_experts):
            expert = destination + local_expert * ep_degree
            count = my_counts[expert]
            if count:
                send_perm_values.extend(
                    range(expert_offsets[expert], expert_offsets[expert + 1])
                )
                destination_count += count
        input_splits.append(destination_count)

    send_perm = torch.tensor(send_perm_values, dtype=torch.long)

    my_expert_ids = [
        ep_rank + local_expert * ep_degree for local_expert in range(num_local_experts)
    ]
    output_splits = [
        sum(counts[source][expert] for expert in my_expert_ids)
        for source in range(ep_degree)
    ]
    source_offsets = [0]
    for count in output_splits:
        source_offsets.append(source_offsets[-1] + count)

    owned_expert_offsets = []
    for source in range(ep_degree):
        offsets = [0]
        for previous in my_expert_ids[:-1]:
            offsets.append(offsets[-1] + counts[source][previous])
        owned_expert_offsets.append(offsets)

    expert_perm_values = []
    local_ntpe = []
    for local_expert, expert in enumerate(my_expert_ids):
        expert_count = 0
        for source in range(ep_degree):
            count = counts[source][expert]
            if count:
                start = source_offsets[source] + owned_expert_offsets[source][local_expert]
                expert_perm_values.extend(range(start, start + count))
                expert_count += count
        local_ntpe.append(expert_count)

    expert_perm = torch.tensor(expert_perm_values, dtype=torch.long)
    return send_perm, expert_perm, input_splits, output_splits, local_ntpe


def _build_all_to_all_metadata_vectorized(
    all_ntpe: Tensor,
    ep_rank: int,
    topology_indices: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, list[int], list[int], Tensor]:
    """Build all-to-all routing metadata with vectorized tensor operations."""
    ep_degree, num_experts = all_ntpe.shape
    num_local_experts = num_experts // ep_degree
    # Routing metadata is already integer-valued after the collective. Avoid a
    # per-layer device copy when callers provide the canonical int64 tensor;
    # retain the conversion for legacy integer widths.
    counts = all_ntpe if all_ntpe.dtype == torch.long else all_ntpe.to(torch.long)
    local_counts = counts[ep_rank]

    expert_offsets = torch.cumsum(local_counts, dim=0) - local_counts
    if topology_indices is None:
        send_experts = (
            torch.arange(num_experts, device=counts.device)
            .view(num_local_experts, ep_degree)
            .transpose(0, 1)
            .reshape(-1)
        )
        owned_experts = torch.arange(
            ep_rank, num_experts, ep_degree, device=counts.device
        )
    else:
        send_experts, owned_experts = topology_indices
    send_counts = local_counts[send_experts]
    send_starts = expert_offsets[send_experts]
    send_group_ends = torch.cumsum(send_counts, dim=0)
    send_group_begins = send_group_ends - send_counts
    send_perm = _expand_grouped_ranges(send_starts, send_group_begins, send_counts)

    owned_counts = counts[:, owned_experts]
    output_splits = owned_counts.sum(dim=1)
    source_offsets = torch.cumsum(output_splits, dim=0) - output_splits
    owned_offsets = torch.cumsum(owned_counts, dim=1) - owned_counts
    recv_counts = owned_counts.transpose(0, 1).reshape(-1)
    recv_starts = (
        source_offsets[:, None] + owned_offsets
    ).transpose(0, 1).reshape(-1)
    recv_group_ends = torch.cumsum(recv_counts, dim=0)
    recv_group_begins = recv_group_ends - recv_counts
    expert_perm = _expand_grouped_ranges(
        recv_starts, recv_group_begins, recv_counts
    )

    input_splits = local_counts.view(num_local_experts, ep_degree).sum(dim=0)
    local_ntpe = owned_counts.sum(dim=0)
    return (
        send_perm,
        expert_perm,
        input_splits.tolist(),
        output_splits.tolist(),
        local_ntpe,
    )


def _expand_grouped_ranges(
    group_starts: Tensor, group_begins: Tensor, group_counts: Tensor
) -> Tensor:
    """Expand contiguous group ranges without materializing group IDs."""
    group_bases = group_starts - group_begins
    expanded_bases = torch.repeat_interleave(group_bases, group_counts)
    expanded_bases.add_(torch.arange(
        expanded_bases.numel(), device=group_counts.device, dtype=torch.long
    ))
    return expanded_bases


def _build_ag_dispatch_metadata_vectorized(
    all_ntpe: Tensor,
    ep_rank: int,
    s_local: int,
    owned_experts: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build AG/RS local-expert indices without per-token Python loops.

    ``all_ntpe`` is rank-major by global expert. The returned indices are
    expert-major by local expert, then source-rank-major, matching the
    ordering produced by the reference implementation in ``_token_dispatch``.
    """
    ep_degree, num_experts = all_ntpe.shape
    num_local_experts = num_experts // ep_degree
    # The canonical collective metadata is already int64. Reuse it directly;
    # legacy integer widths still normalize without changing the input tensor.
    counts = all_ntpe if all_ntpe.dtype == torch.long else all_ntpe.to(torch.long)
    if owned_experts is None:
        owned_experts = torch.arange(
            ep_rank, num_experts, ep_degree, device=counts.device
        )
    owned_counts = counts[:, owned_experts]

    input_offsets = torch.cumsum(counts, dim=1)
    input_offsets.sub_(counts)
    rank_offsets = torch.arange(
        ep_degree, device=counts.device, dtype=torch.long
    )
    rank_offsets.mul_(s_local)
    rank_offsets.unsqueeze_(1)
    input_starts = rank_offsets + input_offsets[:, owned_experts]

    group_counts = owned_counts.transpose(0, 1).reshape(-1)
    group_starts = input_starts.transpose(0, 1).reshape(-1)
    group_begins = torch.cumsum(group_counts, dim=0)
    group_begins.sub_(group_counts)
    gather_idx = _expand_grouped_ranges(group_starts, group_begins, group_counts)
    local_ntpe = owned_counts.sum(dim=0)
    return gather_idx, local_ntpe


class _AGAutogradIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, partial_out: Tensor, edge: Tensor) -> Tensor:
        return partial_out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, grad_output.new_zeros(())


def _add_ag_autograd_anchor(
    partial_out: Tensor, all_ri: Optional[Tensor]
) -> Tensor:
    if all_ri is None or not all_ri.requires_grad:
        return partial_out
    if _USE_ZERO_COST_AG_ANCHOR and all_ri.numel() > 0:
        return _AGAutogradIdentity.apply(partial_out, all_ri.reshape(-1)[0])
    if _USE_SINGLE_ROW_AG_ANCHOR and all_ri.shape[0] > 0:
        # Keep the edge to the AllGather output with one scalar rather than a
        # hidden-width row. Broadcasting the row-sized anchor into partial_out
        # creates an avoidable [1, hidden] temporary on every MoE layer.
        zero_anchor = all_ri[0, 0] * 0.0
    else:
        zero_anchor = all_ri.sum() * 0.0
    if _USE_INPLACE_AG_ANCHOR:
        partial_out.add_(zero_anchor)
        return partial_out
    return partial_out + zero_anchor


def _accumulate_expert_outputs(
    partial_out: Tensor, gather_idx: Tensor, routed_output: Tensor
) -> Tensor:
    """Accumulate expert rows with a selectable memory-efficient implementation."""
    if _USE_INDEX_ADD_COMBINE:
        partial_out.index_add_(0, gather_idx, routed_output)
    else:
        index = gather_idx.unsqueeze(1).expand_as(routed_output)
        partial_out.scatter_add_(0, index, routed_output)
    return partial_out


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

    No AllToAll, no split tracking. Default transport on Aurora is the gloo
    CPU-bounce path in `_ep_all_gather` / `_ep_reduce_scatter` (the recipe
    monkey-patches `dist.reduce_scatter_tensor` to work around CCL's
    ze_handle_manager IPC bug on FSDP2 sub-allocated grads). Native XCCL on
    `_shard_pg` is opt-in via `TORCHTUNE_EP_USE_XCCL=1` (see CLAUDE.md);
    EP=8 v10f measured -3.9% with XCCL EP AG/RS alone, EP=16 phase B was
    null over Slingshot. The dominant XCCL win on the EP path is grad-release
    (`TORCHTUNE_EP_GRAD_RELEASE_XCCL=1`, -34.5% on v10f).

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
        # all_to_all path (TORCHTUNE_EP_ALL2ALL) — perms + splits for combine reverse.
        self._a2a_send_perm: Optional[Tensor] = None
        self._a2a_expert_perm: Optional[Tensor] = None
        self._a2a_input_splits: Optional[list] = None
        self._a2a_output_splits: Optional[list] = None
        self._a2a_s_local: Optional[int] = None
        self._a2a_recv_rows: Optional[int] = None
        self._measurement = None
        self._topology_index_cache: dict[tuple, tuple[Tensor, Tensor]] = {}

    def _topology_indices(
        self,
        ep_degree: int,
        num_experts: int,
        ep_rank: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        key = (
            device.type,
            device.index,
            ep_degree,
            num_experts,
            ep_rank,
        )
        cached = self._topology_index_cache.get(key)
        if cached is None:
            num_local_experts = num_experts // ep_degree
            send_experts = (
                torch.arange(num_experts, device=device)
                .view(num_local_experts, ep_degree)
                .transpose(0, 1)
                .reshape(-1)
            )
            owned_experts = torch.arange(
                ep_rank, num_experts, ep_degree, device=device
            )
            cached = (send_experts, owned_experts)
            self._topology_index_cache[key] = cached
        return cached

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

        if _EP_ALL2ALL:
            return self._token_dispatch_all2all(
                routed_input,
                num_tokens_per_expert,
                ep_degree,
                ep_rank,
                num_experts,
                num_local_experts,
                group,
            )

        # Stage 1: AllGather all tokens across EP ranks (gradient-tracked).
        # all_ri[r*s_local : (r+1)*s_local] = rank r's routed_input (expert-sorted).
        # Backward: ReduceScatter (via _AllGatherRS.backward).
        all_ri = _AllGatherRS.apply(
            routed_input.contiguous(), group, self._measurement
        )

        # AllGather num_tokens_per_expert (no grad needed).
        # all_ntpe[r, e] = tokens rank r routes to global expert e.
        allocate = (
            torch.empty
            if _USE_UNINITIALIZED_COLLECTIVE_BUFFERS
            else torch.zeros
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
        n_ntpe = _EP_OP_N
        _EP_OP_N += 1
        r_ntpe = dist.get_rank()
        if _EP_DEBUG:
            print(f"[rank{r_ntpe}] EP-OP #{n_ntpe} ENTER NTPE-AG", flush=True)
        if routed_input.device.type == "xpu" and _GLOO_EP_PG is not None:
            ntpe_cpu = num_tokens_per_expert.to(torch.long).contiguous().cpu()
            all_ntpe_cpu = allocate(
                ep_degree * num_experts, dtype=torch.long, device="cpu"
            )
            dist.all_gather_into_tensor(all_ntpe_cpu, ntpe_cpu, group=_GLOO_EP_PG)
            if _USE_CPU_METADATA_TRANSFER:
                all_ntpe = _materialize_routing_metadata(
                    all_ntpe_cpu,
                    routed_input.device,
                    direct_cpu_transfer=True,
                ).view(ep_degree, num_experts)
            else:
                all_ntpe = _materialize_routing_metadata(
                    all_ntpe_cpu,
                    routed_input.device,
                    direct_cpu_transfer=False,
                )
                all_ntpe = all_ntpe.view(ep_degree, num_experts)
        else:
            all_ntpe_flat = allocate(
                ep_degree * num_experts,
                dtype=torch.long,
                device=routed_input.device,
            )
            dist.all_gather_into_tensor(
                all_ntpe_flat, num_tokens_per_expert.to(torch.long), group=group
            )
            all_ntpe = all_ntpe_flat.view(ep_degree, num_experts)
        if _EP_DEBUG:
            print(f"[rank{r_ntpe}] EP-OP #{n_ntpe} EXIT NTPE-AG", flush=True)

        # Build gather indices for local experts (interleaved assignment).
        # Rank ep_rank owns global experts: ep_rank, ep_rank+ep_degree, ..., ep_rank+(NLE-1)*ep_degree.
        # The vectorized path preserves expert-major/source-major ordering while
        # avoiding per-expert scalar reads and Python tensor concatenations.
        with torch.no_grad():
            _, owned_experts = self._topology_indices(
                ep_degree, num_experts, ep_rank, all_ntpe.device
            )
            gather_idx, local_ntpe = _build_ag_dispatch_metadata_vectorized(
                all_ntpe, ep_rank, s_local, owned_experts
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
        if _EP_DEBUG:
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

        if _EP_ALL2ALL:
            return self._token_combine_all2all(routed_output, group)

        # v159: read indices off the ExpertParallel instance (set in _token_dispatch).
        s_local = self._ag_s_local
        gather_idx = self._ag_gather_idx
        all_ri = (
            self._ag_all_ri
        )  # v161: AllGather output, kept alive for autograd binding.

        # Stage 5a: Scatter expert outputs back to their positions in the full (T, dim) buffer.
        # partial_out[i] = expert output for the token that was at all_ri[i].
        # Positions not owned by this rank's local experts remain zero.
        # Since the pre-weighted tokens are already scaled, no extra weighting needed here.
        partial_out = routed_output.new_zeros(
            ep_degree * s_local, routed_output.shape[-1]
        )
        # v160: ALWAYS run indexed accumulation (even with empty gather_idx) so partial_out
        # carries an autograd link back to routed_output. With empty gather_idx,
        # index_add_ is a no-op on values but still emits a grad-fn — that
        # grad-fn is what keeps the EP backward chain connected. Without it,
        # _ReduceScatterAG.backward gets skipped on the empty-dispatch rank →
        # asymmetric early-exit deadlock at next AG-BWD (v158/v159 reproduced).
        # v163 (2026-07-23): use an in-place accumulation on the just-allocated
        # `partial_out` zeros buffer instead of the out-of-place `scatter_add`,
        # which allocated a SECOND full (ep_degree*s_local, dim) tensor right
        # after the first (v162 fixed one full-size temp downstream of this
        # line but the crash simply moved here — this line's own out-of-place
        # semantics is a second, larger full-size alloc every MoE layer).
        # v165: use row-wise index_add_ rather than scatter_add_ with an
        # expanded `(tokens, hidden)` index. Each gathered position belongs to
        # one local expert, so index_add_ has identical accumulation semantics
        # while avoiding a hidden-width-sized index allocation.
        # partial_out doesn't require_grad itself (fresh new_zeros), so the
        # in-place mutation is safe; autograd still records IndexAddBackward
        # via routed_output (the only grad-requiring operand) exactly as with
        # the out-of-place form.
        _accumulate_expert_outputs(partial_out, gather_idx, routed_output)
        # v161: HARD AUTOGRAD ANCHOR. Even with the v160 index_add grad-fn,
        # an empty-dispatch rank's _AllGatherRS.backward never fires because
        # downstream gradients arrive at routed_output as zero (no expert
        # produced anything) and the chain back to all_ri only goes via the
        # gather index. By adding 0.0 * all_ri we force partial_out to depend
        # on all_ri at the autograd-graph level — the engine will then ALWAYS
        # call _AllGatherRS.backward (which feeds RS-BWD), keeping rank-1 in
        # lockstep with peers at #237 RS-BWD AND #238 AG-BWD.
        # v162 (2026-07-23): the original `.expand_as(partial_out) * 0.0` forces
        # eager materialization of a SECOND full (ep_degree*s_local, dim) zero
        # tensor before the `+` even runs (expand_as is a zero-copy view, but
        # multiplying a broadcast view by a scalar is an elementwise op that
        # must produce a real, fully-materialized output). The scalar anchor
        # below keeps the same autograd edge and zero values without creating a
        # hidden-width row or a second full-size broadcast temporary.
        partial_out = _add_ag_autograd_anchor(partial_out, all_ri)
        # Diagnostic: confirm partial_out has a grad-fn that reaches AllGather.
        if _EP_DEBUG:
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
        out = _ReduceScatterAG.apply(
            partial_out, group, self._measurement
        )  # (s_local, dim)
        return out

    # ------------------------------------------------------------------
    # all_to_all_single dispatch path (TORCHTUNE_EP_ALL2ALL=1)
    # ------------------------------------------------------------------
    @torch.compiler.disable
    def _token_dispatch_all2all(
        self,
        routed_input: Tensor,
        num_tokens_per_expert: Tensor,
        ep_degree: int,
        ep_rank: int,
        num_experts: int,
        num_local_experts: int,
        group: dist.ProcessGroup,
    ) -> tuple[Tensor, Tensor]:
        """True all_to_all dispatch: send each token only to the rank owning its expert.

        ``routed_input`` arrives expert-sorted ``(S, dim)`` (experts 0..E-1 contiguous).
        Interleaved ownership: rank ``d`` owns experts ``{d, d+EP, ..., d+(NLE-1)*EP}``.

        Steps:
          1. AllGather ``ntpe`` so every rank knows all_ntpe[src, e]  (small, ints).
          2. Reorder our tokens expert-sorted -> DEST-RANK-major (all tokens for
             rank 0's experts first, then rank 1's, ...). ``input_splits[d]`` =
             tokens we send to rank d = sum over d's owned experts of our ntpe.
          3. ``output_splits[src]`` = tokens we receive from rank src = sum over
             OUR owned experts of all_ntpe[src, our_experts].
          4. all_to_all_single(send, output_splits, input_splits) -> recv, which is
             SOURCE-rank-major (rank0's tokens, rank1's, ...), each block still in
             our-expert order within that source.
          5. Reorder recv source-major -> EXPERT-major (all expert e0 tokens across
             all sources, then e1, ...) which is what GroupedExperts expects.

        Caches inverse permutations + splits on the instance for combine to reverse.
        Returns ``(dispatched_expert_major, local_ntpe)``.
        """
        device = routed_input.device
        s_local = routed_input.shape[0]
        transport_group = group

        # 1. AllGather ntpe (gloo CPU-bounce, same isolation rationale as AG+RS path).
        if device.type == "xpu" and _GLOO_EP_PG is not None:
            ntpe_cpu = num_tokens_per_expert.to(torch.long).contiguous().cpu()
            allocate = (
                torch.empty
                if _USE_UNINITIALIZED_COLLECTIVE_BUFFERS
                else torch.zeros
            )
            all_ntpe_cpu = allocate(
                ep_degree * num_experts, dtype=torch.long, device="cpu"
            )
            metadata_group = _GLOO_EP_PG
        else:
            allocate = (
                torch.empty
                if _USE_UNINITIALIZED_COLLECTIVE_BUFFERS
                else torch.zeros
            )
            all_ntpe_flat = allocate(
                ep_degree * num_experts, dtype=torch.long, device=device
            )
            metadata_group = group
            all_ntpe_cpu = all_ntpe_flat.cpu()

        measurement = self._measurement
        if measurement is not None:
            metadata_timing = measurement.collective(
                "routing_metadata_allgather",
                scope="ep",
                backend=dist.get_backend(metadata_group),
            )
        else:
            metadata_timing = nullcontext()
        with metadata_timing:
            if device.type == "xpu" and _GLOO_EP_PG is not None:
                dist.all_gather_into_tensor(
                    all_ntpe_cpu, ntpe_cpu, group=metadata_group
                )
            else:
                dist.all_gather_into_tensor(
                    all_ntpe_flat,
                    num_tokens_per_expert.to(torch.long),
                    group=metadata_group,
                )
                all_ntpe_cpu = all_ntpe_flat.cpu()

        # Build routing permutations with vectorized device operations while
        # retaining the proven gloo count exchange on XPU. Adding another XCCL
        # collective per MoE layer made the second training step stall on Aurora.
        materialization_timing = (
            measurement.time("routing_metadata_materialization")
            if measurement is not None
            else nullcontext()
        )
        with materialization_timing:
            all_ntpe = (
                all_ntpe_cpu.to(device=device)
                if _DEVICE_ROUTING_METADATA
                else all_ntpe_cpu
            )

        permutation_timing = (
            measurement.time("routing_metadata_permutation")
            if measurement is not None
            else nullcontext()
        )
        with permutation_timing:
            if _DEVICE_ROUTING_METADATA or _CPU_VECTOR_ROUTING_METADATA:
                topology_indices = self._topology_indices(
                    ep_degree, num_experts, ep_rank, all_ntpe.device
                )
                send_perm, expert_perm, input_splits, output_splits, local_ntpe = (
                    _build_all_to_all_metadata_vectorized(
                        all_ntpe.view(ep_degree, num_experts),
                        ep_rank,
                        topology_indices,
                    )
                )
            else:
                (
                    send_perm_cpu,
                    expert_perm_cpu,
                    input_splits,
                    output_splits,
                    local_ntpe_list,
                ) = _build_all_to_all_metadata(
                    all_ntpe.view(ep_degree, num_experts), ep_rank
                )
                send_perm = send_perm_cpu
                expert_perm = expert_perm_cpu
                local_ntpe = torch.tensor(
                    local_ntpe_list, dtype=torch.long, device=device
                )

        if not _DEVICE_ROUTING_METADATA:
            transfer_timing = (
                measurement.time("routing_metadata_materialization")
                if measurement is not None
                else nullcontext()
            )
            with transfer_timing:
                if _CPU_VECTOR_ROUTING_METADATA:
                    (
                        send_perm,
                        expert_perm,
                        local_ntpe,
                    ) = _materialize_all_to_all_permutations(
                        send_perm,
                        expert_perm,
                        local_ntpe,
                        device,
                        packed_transfer=_PACK_ROUTING_METADATA_TRANSFER,
                    )
                else:
                    send_perm = send_perm.to(device=device)
                    expert_perm = expert_perm.to(device=device)
                    local_ntpe = local_ntpe.to(device=device)

        # The fused A/B path keeps packing, transport, and unpacking in one
        # autograd boundary. The default remains the separately timed path.
        if _USE_FUSED_ALLTOALL_ROUTING:
            dispatched = _fused_all_to_all_routing(
                routed_input,
                send_perm,
                expert_perm,
                input_splits,
                output_splits,
                transport_group,
                measurement,
            )
            self._a2a_send_perm = send_perm
            self._a2a_expert_perm = expert_perm
            self._a2a_input_splits = input_splits
            self._a2a_output_splits = output_splits
            self._a2a_s_local = s_local
            self._a2a_recv_rows = int(sum(output_splits))
            self._a2a_transport_group = transport_group
            return dispatched, local_ntpe

        # send buffer, dest-rank-major (gradient flows through index-select).
        pack_timing = (
            measurement.time("dispatch_pack")
            if measurement is not None
            else nullcontext()
        )
        with pack_timing:
            send_buf = routed_input[send_perm]

        # 4. all_to_all: recv is source-rank-major.
        if measurement is not None:
            backend = dist.get_backend(transport_group)
            with measurement.collective(
                "dispatch_alltoall", scope="ep", backend=backend
            ):
                recv = _ep_all_to_all_single(
                    send_buf,
                    output_splits,
                    input_splits,
                    transport_group,
                    "dispatch",
                    measurement,
                )
        else:
            recv = _ep_all_to_all_single(
                send_buf,
                output_splits,
                input_splits,
                transport_group,
                "dispatch",
                measurement,
            )

        # 5. Reorder recv source-major -> expert-major.
        # Within source src's received block, tokens are ordered by our experts
        # (e0's tokens, e1's, ...) because src sent them dest-rank-major which,
        # for OUR rank, is our-expert order. So recv layout is:
        #   [src0: e0.., e1.., ...][src1: e0.., e1.., ...]...
        # We want [e0: src0.., src1..][e1: src0.., src1..]...
        unpack_timing = (
            measurement.time("dispatch_unpack")
            if measurement is not None
            else nullcontext()
        )
        with unpack_timing:
            dispatched = recv[expert_perm]
        # Cache for combine: to reverse we need to undo expert_perm on the output,
        # all_to_all back with swapped splits, then undo send_perm.
        self._a2a_send_perm = send_perm
        self._a2a_expert_perm = expert_perm
        self._a2a_input_splits = input_splits  # dispatch send splits
        self._a2a_output_splits = output_splits  # dispatch recv splits
        self._a2a_s_local = s_local
        self._a2a_recv_rows = int(sum(output_splits))
        self._a2a_transport_group = transport_group
        return dispatched, local_ntpe

    def _token_combine_all2all(
        self, routed_output: Tensor, group: dist.ProcessGroup
    ) -> Tensor:
        """Reverse of _token_dispatch_all2all: expert-major -> original (S, dim).

        Mirror image: undo expert_perm (scatter back to source-major), all_to_all
        with dispatch splits SWAPPED (recv<->send), then undo send_perm (scatter
        back to expert-sorted). All via autograd-safe index ops + ft_c all_to_all.
        """
        send_perm = self._a2a_send_perm
        expert_perm = self._a2a_expert_perm
        # dispatch: send with input_splits, recv with output_splits.
        # combine: send with output_splits (we now send back what we received),
        #          recv with input_splits.
        input_splits = self._a2a_output_splits
        output_splits = self._a2a_input_splits
        measurement = self._measurement

        if _USE_FUSED_ALLTOALL_ROUTING:
            return _fused_all_to_all_combine(
                routed_output,
                send_perm,
                expert_perm,
                input_splits,
                output_splits,
                self._a2a_transport_group,
                self._measurement,
                self._a2a_s_local,
            )

        # 1. Undo expert_perm: place routed_output rows back to source-major order.
        # dispatched = recv[expert_perm]  =>  recv[expert_perm] = routed_output
        # so source-major buffer = scatter routed_output by expert_perm.
        pack_timing = (
            measurement.time("combine_pack")
            if measurement is not None
            else nullcontext()
        )
        with pack_timing:
            if _USE_UNINITIALIZED_ALLTOALL_COMBINE_BUFFERS:
                src_major = routed_output.new_empty(
                    self._a2a_recv_rows, routed_output.shape[-1]
                )
            else:
                src_major = routed_output.new_zeros(
                    self._a2a_recv_rows, routed_output.shape[-1]
                )
            if _USE_ROWWISE_ALLTOALL_UNPERMUTE:
                src_major.index_copy_(0, expert_perm, routed_output)
            else:
                index = expert_perm.unsqueeze(1).expand_as(routed_output)
                src_major.scatter_(0, index, routed_output)

        # 2. all_to_all back (splits swapped) -> dest-rank-major on originating rank.
        transport_group = self._a2a_transport_group
        if measurement is not None:
            backend = dist.get_backend(transport_group)
            with measurement.collective(
                "combine_alltoall", scope="ep", backend=backend
            ):
                back = _ep_all_to_all_single(
                    src_major,
                    output_splits,
                    input_splits,
                    transport_group,
                    "combine",
                    measurement,
                )
        else:
            back = _ep_all_to_all_single(
                src_major,
                output_splits,
                input_splits,
                transport_group,
                "combine",
                measurement,
            )

        # 3. Undo send_perm: dest-rank-major -> expert-sorted (S, dim).
        # send_buf = routed_input[send_perm]  =>  out[send_perm] = back.
        unpack_timing = (
            measurement.time("combine_unpack")
            if measurement is not None
            else nullcontext()
        )
        with unpack_timing:
            if _USE_UNINITIALIZED_ALLTOALL_COMBINE_BUFFERS:
                out = back.new_empty(self._a2a_s_local, back.shape[-1])
            else:
                out = back.new_zeros(self._a2a_s_local, back.shape[-1])
            if _USE_ROWWISE_ALLTOALL_UNPERMUTE:
                out.index_copy_(0, send_perm, back)
            else:
                index = send_perm.unsqueeze(1).expand_as(back)
                out.scatter_(0, index, back)
        return out

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Attach EP metadata onto a ``GroupedExperts`` module.

        Weight sharding is performed by the recipe directly on the checkpoint
        state dict before ``load_from_full_model_state_dict`` (interleaved slice
        ``_ft[_ep_rank::_ep_degree]`` — must match ``_token_dispatch``'s
        ``g = ep_rank + local_exp_idx * ep_degree`` ownership formula).

        We do NOT register forward hooks here because FSDP2 ``fully_shard``
        (called later) drops or shadows them. Instead we store this
        ``ExpertParallel`` instance and the mesh on the module; the recipe
        then calls ``wire_ep_to_moe_modules(model)`` after both
        ``parallelize_module`` and ``fully_shard`` to set
        ``_ep_dispatch``/``_ep_combine`` directly on each parent ``MoE``.

        Args:
            module: The ``GroupedExperts`` module to wrap.
            device_mesh: EP mesh of shape ``(ep_degree,)``.

        Returns:
            The wrapped module (EP metadata attached, weights still full-rank).
        """
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
        module._ep_dispatch = partial(
            ep_instance._token_dispatch, experts, device_mesh=ep_mesh
        )
        module._ep_combine = partial(
            ep_instance._token_combine, experts, device_mesh=ep_mesh
        )
        ep_instance._measurement = (
            module.measurement if module.measurement.enabled else None
        )
        num_wired += 1

    return num_wired

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# XPU distributed infrastructure for GRPO training on Aurora.
#
# Extracted from grpo_full_finetune_distributed_xpu.py so that the recipe file
# can focus on training logic rather than platform-level distributed bookkeeping.
#
# Contents:
#   - Module-level globals (process group handles, degree counters)
#   - Saved originals of torch.distributed ops (captured before monkey-patching)
#   - Activation-checkpointing helpers (_no_reentrant_ac_wrapper, _apply_split_ac)
#   - AllReduce-based reduce_scatter patch (_xpu_reduce_scatter_via_allreduce)
#   - XCCL-based AllToAll wrapper (_xpu_all_to_all_via_gloo)
#   - Post-backward gradient sync for EP (_ep_post_backward_grad_sync[_xccl])
#   - XPU-safe empty-cache overrides (device_empty_cache, _safe_empty_cache)
#   - Trajectory slicing utility (_slice_trajectory)
#   - install_xpu_patches() — call once at module level in the recipe
#   - set_process_groups()  — call from _init_distributed after group creation

# Activation checkpointing history (guides use_reentrant choice):
#
# Applied to full TransformerSelfAttentionLayer (includes MoE) for memory efficiency.
#
# v114: Reverted to use_reentrant=True due to AllToAll backward race condition.
#   v112's interleaved expert assignment changes the pre-AllToAll permutation, creating
#   new tensors at each AC recompute. This causes _XPUSyncAllToAll.apply() to register
#   a new backward node per AC recompute call, so backward IS now invoked (unlike v108-v111
#   where caching returned the original grad_fn). With use_reentrant=False, C++ backward
#   threads can fire different layers' AllToAll backward simultaneously on different ranks
#   → inconsistent XCCL splits (rank 6 sends 1004, rank 7 expects 3012) → buffer overflow
#   → SIGSEGV at call#22. Python sequential backward (use_reentrant=True) ensures all
#   ranks hit the same layer's AllToAll in lockstep.
#   Memory cost: ~13 GiB more HBM vs use_reentrant=False. Offset by v112's interleaved
#   assignment reducing peak from 55.25 GiB to 47.83 GiB. Estimated peak: ~60 GiB.
#
# v109: Reverted to use_reentrant=False — AllToAll caching (v108) made it safe:
#   AC recompute returned cached output immediately, no XCCL communication during recompute.
#   Memory savings: ~13 GiB less HBM.
#
# v107-v108: use_reentrant=True — fixed XCCL AllToAll deadlock by making backward
#   synchronous (sequential layer ordering). v108 caching eliminated AllToAll during
#   AC recompute, making synchronous backward unnecessary.
#
# MoE AC recompute non-determinism (v158 fix):
#   The Gemma4 router runs sigmoid(float32) + argsort(stable, descending). Bitwise
#   differences between the original FWD and an AC recompute can flip ties →
#   bincount(num_tokens_per_expert) shifts by ±1 → ExpertParallel._token_dispatch
#   regenerates _ag_gather_idx with a different row count than the autograd-saved
#   routed_output → ScatterAddBackward0 mismatch on paired EP ranks (e.g. ranks 8/10
#   in v154: got [4579, 2816] vs expected [4578, 2816]).
#   Fix: Gemma4TransformerLayer self-checkpoints attention+dense_MLP only; MoE runs
#   OUTSIDE the AC region (router runs exactly once per FWD). See _apply_split_ac
#   below and torchtune/models/gemma4/_component_builders.py:Gemma4TransformerLayer.
#
# v154: revert to non-reentrant AC. v114 forced use_reentrant=True to work around
# an AllToAll backward SIGSEGV; AllToAll has been gone since v141 (replaced with
# AllGather+ReduceScatter in _parallelism.py), so the original reason is stale.
# Hypothesis: reentrant AC interleaves FWD-recompute with BWD via the Python
# autograd boundary, and on the rank with the smallest routed batch (consistently
# local-index-1 in each EP group after the v110 interleaved routing fix) the
# submodule eval order diverges, producing the deterministic op #259 RS-BWD desync.

import torch
import torch.distributed
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from torchtune import modules, utils
from torchtune.training import device_empty_cache as _orig_device_empty_cache
from torchtune.dev.rl.types import GRPOTrajectory

log = utils.get_logger("DEBUG")

# ---------------------------------------------------------------------------
# Process group globals — set once by set_process_groups() from _init_distributed
# ---------------------------------------------------------------------------
_GLOO_DP_REP_PG = None  # gloo mirror of dp_replicate_pg (3 ranks)
_GLOO_DP_SHARD_PG = None  # gloo mirror of dp_shard_pg (4 ranks)
_GLOO_GLOBAL_PG = (
    None  # gloo global group (all ranks); barrier before post-bwd AllReduce
)
_XCCL_DP_REP_PG = None  # XCCL dp_replicate group (3 ranks, XPU fabric)
_XCCL_DP_SHARD_PG = (
    None  # XCCL dp_shard group (XPU fabric); used by iter2 grad-release fast path
)
_DP_REP_DEGREE = 1  # dp_replicate world size
_DP_SHARD_DEGREE = 1  # dp_shard world size

# Set True by the recipe (via enable_fsdp1_hsdp_inter_node_gloo()) ONLY on the
# dense FSDP1 HYBRID_SHARD path. FSDP1 HYBRID_SHARD fires its inter-node grad
# reduction as `dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)`
# (torch/distributed/fsdp/_runtime_utils.py) on the XCCL replicate group every
# backward. On Aurora that cross-node XCCL/RDMA all_reduce leaks CXI MR handles
# (same class as the Qwen3-32B XCCL-cross leak fixed by the gloo cross-PG reroute)
# → GPU PDE page-fault / banned:1 after ~10 steps at 84 ranks. When this flag is
# set, _xpu_all_reduce_inter_node_gloo CPU-bounces that specific call over
# _GLOO_DP_REP_PG. Left False for EP/FSDP2 (all_reduce must stay unpatched there —
# reduce_grads=False means FSDP2 never fires it) and for flat single-node FSDP1
# (no inter-node PG at all).
_FSDP1_HSDP_INTER_NODE_GLOO = False

# Iter2 opt-in: route _ep_release_fsdp_unsharded_grads's per-param all_reduce
# through native XCCL on the XPU dp_shard PG instead of the gloo CPU-bounce
# (D2H → gloo all_reduce → H2D → chunk). Default off for the same reason
# TORCHTUNE_EP_USE_XCCL is opt-in: native XCCL on the EP fabric historically
# deadlocked at op #259 (RS-BWD), and although iter1 showed that no longer
# reproduces on this codebase, the safer rollout is the same A/B opt-in
# pattern. When unset the helper takes the gloo path unchanged.
import os as _os_iter2

_EP_GRAD_RELEASE_XCCL = (
    _os_iter2.environ.get("TORCHTUNE_EP_GRAD_RELEASE_XCCL", "0") == "1"
)

# v167 (2026-07-24): number of FSDPParamGroups batched into one collective
# per _ep_release_fsdp_unsharded_grads call. A single global batch (all ~97
# groups on Qwen3-30B-A3B/EP=8) held the whole model's unsharded grads +
# a flattened copy simultaneously and HW-crashed with
# UR_RESULT_ERROR_OUT_OF_RESOURCES at step 2 — this bounds peak transient
# memory to one chunk while still cutting the collective count by roughly
# this factor vs the original one-collective-per-param loop. Not yet
# HW-tuned; 8 is a conservative starting point pending an A/B.
_EP_GRAD_RELEASE_CHUNK_GROUPS = int(
    _os_iter2.environ.get("TORCHTUNE_EP_GRAD_RELEASE_CHUNK_GROUPS", "8")
)
_EP_GRAD_RELEASE_LEGACY = (
    _os_iter2.environ.get("TORCHTUNE_EP_GRAD_RELEASE_LEGACY", "0") == "1"
)
_EP_GRAD_RELEASE_STREAMING = (
    _os_iter2.environ.get("TORCHTUNE_EP_GRAD_RELEASE_STREAMING", "0") == "1"
)

# v169 (2026-07-24): opt-in per-component timing breakdown for
# _ep_release_fsdp_unsharded_grads. Added after a codex-review pass on the
# v168 fix flagged that py-spy's earlier ~60%-of-wall-clock share for this
# function's all_reduce could not distinguish real collective-launch
# overhead (what chunked batching actually removes) from communication
# time or cross-rank synchronization wait (which batching does NOT fix).
# Rank-0-only wall-clock breakdown of: leading barrier, per-chunk
# collect/flatten (host-side prep), the all_reduce call itself, and Pass 3
# (unflatten+clone+chunk/pad/cast/accumulate/cleanup). Brackets each
# section with torch.xpu.synchronize() so timings reflect actual XPU
# completion, not just async-launch return time — critical since XPU ops
# are asynchronous and an un-synced wall-clock would just measure how fast
# the CPU could enqueue work, not how long the GPU took.
_EP_GRAD_RELEASE_TIMING = (
    _os_iter2.environ.get("TORCHTUNE_EP_GRAD_RELEASE_TIMING", "0") == "1"
)

# ---------------------------------------------------------------------------
# Saved originals — captured BEFORE monkey-patching (imported by recipe too)
# ---------------------------------------------------------------------------
import torch.distributed as _tdist_patch

_orig_reduce_scatter_tensor = _tdist_patch.reduce_scatter_tensor
_orig_all_reduce = _tdist_patch.all_reduce
_orig_all_to_all_single = _tdist_patch.all_to_all_single

_a2a_call_counter = 0  # v70: counts all_to_all_single calls for diagnostic tagging


# ---------------------------------------------------------------------------
# Activation checkpointing helpers
# ---------------------------------------------------------------------------

import os as _os_ac_diag

# v165 diagnostic (2026-07-24): opt-in reentrant-AC override for the un-chunked
# SFT regime. v154's revert to use_reentrant=False assumed the only reason to
# force reentrant AC (v107/v108) was the since-removed AllToAll dispatch path
# -- but a NEW banned:1 SIGSEGV was found on the current AG+RS dispatch path,
# in a regime (single un-chunked .backward() per step) GRPO's chunked
# forward_batch_size path never exercises. Testing whether reentrant AC
# (synchronous, Python-sequential backward -- the original v107/v108 fix
# mechanism) avoids this NEW crash too. Default 0 preserves exact prior
# behavior (use_reentrant=False) for every existing caller (GRPO, which never
# hits this crash and should not be perturbed).
_AC_USE_REENTRANT = _os_ac_diag.environ.get("TORCHTUNE_AC_USE_REENTRANT", "0") == "1"


def _no_reentrant_ac_wrapper(module):
    return ptd_checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.REENTRANT,
        checkpoint_fn=_torch_checkpoint,
        use_reentrant=_AC_USE_REENTRANT,  # v154 default False; v165 opt-in True diagnostic
        preserve_rng_state=False,
        determinism_check="none",
    )


def _apply_split_ac(model, attention_checkpoint_every: int = 1):
    """v158: Split AC so MoE-bearing layers checkpoint attention only (MoE runs once, never recomputed).

    Both Gemma4TransformerLayer and Qwen3MoeTransformerLayer implement _ac_enabled + the
    self-checkpoint contract. All other TransformerSelfAttentionLayer instances get the
    standard _no_reentrant_ac_wrapper.

    ``attention_checkpoint_every`` controls safe selective checkpointing for
    MoE-bearing layers without wrapping the router: 1 checkpoints every
    attention block, 2 every second block, and so on.

    Returns the number of MoE-bearing attention blocks checkpointed.
    """
    from torchtune.models.gemma4._component_builders import Gemma4TransformerLayer
    from torchtune.models.qwen3_moe._component_builders import Qwen3MoeTransformerLayer

    if attention_checkpoint_every < 1:
        raise ValueError("attention_checkpoint_every must be at least 1")

    moe_layer_ids = set()
    checkpointed_moe_layers = 0
    moe_layer_index = 0
    for m in model.modules():
        if isinstance(m, Gemma4TransformerLayer) and m.moe_block is not None:
            m._ac_enabled = moe_layer_index % attention_checkpoint_every == 0
            moe_layer_ids.add(id(m))
            checkpointed_moe_layers += int(m._ac_enabled)
            moe_layer_index += 1
        elif isinstance(m, Qwen3MoeTransformerLayer):
            m._ac_enabled = moe_layer_index % attention_checkpoint_every == 0
            moe_layer_ids.add(id(m))
            checkpointed_moe_layers += int(m._ac_enabled)
            moe_layer_index += 1

    def _check_fn(submodule):
        if not isinstance(submodule, modules.TransformerSelfAttentionLayer):
            return False
        return id(submodule) not in moe_layer_ids

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=_no_reentrant_ac_wrapper,
        check_fn=_check_fn,
    )
    return checkpointed_moe_layers


def _apply_expert_checkpointing(model) -> int:
    """Set ``checkpoint_experts = True`` on every ``MoE`` module in the model.

    Unlike `_apply_split_ac` (which controls whether the ATTENTION block is
    checkpointed for MoE-bearing layers, and deliberately excludes the router/
    MoE compute from AC entirely for correctness — the v158 fix), this
    targets ``self.experts(...)``'s own compute specifically, via
    `MoE.checkpoint_experts` (see `torchtune/modules/moe/moe.py`'s
    docstring for why recomputing just the expert-compute call, downstream
    of the already-fixed router/dispatch outputs, is safe and does not
    reintroduce the v158 argsort-tie-break bug: the router itself is never
    re-executed by this).

    Motivated by the seq4096 mem_reserved-ratchet investigation (see
    memory/project_moe_sft_profiling_gradrelease_bottleneck_20260724.md) —
    at long sequence lengths, expert-compute intermediates
    (`GroupedExpertsHF`'s padded-BMM or per-expert temporaries) held live
    across all 48 layers until one un-chunked backward are a real
    contributor to the forward-activation peak that XPU can never reclaim
    once cached (`mem_reserved` never drops). Trades expert-compute recompute
    time (roughly doubling that specific sub-phase, per Phase 1's timing
    breakdown showing expert compute at ~10.3% of step time — so a
    recompute costs at most another ~10%, likely less since SEQUENTIAL_EXPERTS'
    forward is cheaper than a full fwd+bwd) for a real reduction in peak
    activation memory.

    Returns the number of MoE modules found and toggled.
    """
    from torchtune.modules.moe.moe import MoE

    n = 0
    for m in model.modules():
        if isinstance(m, MoE):
            m.checkpoint_experts = True
            n += 1
    return n


# ---------------------------------------------------------------------------
# Fake Work object for synchronous ops returning as if async
# ---------------------------------------------------------------------------


class _DoneWork:
    """Fake Work object for synchronous ops masquerading as async."""

    def wait(self):
        pass

    def is_completed(self):
        return True

    def get_future(self):
        import torch.futures as _tf

        f = _tf.Future()
        f.set_result(None)
        return f


# ---------------------------------------------------------------------------
# reduce_scatter_tensor patch — AllReduce-based fallback (XPU/CCL workaround)
# ---------------------------------------------------------------------------


def _xpu_reduce_scatter_via_allreduce(
    output, input, op=None, group=None, async_op=False
):
    """AllReduce-based drop-in for reduce_scatter_tensor (v59: safety net only).

    v59: With reduce_grads=False on ALL FSDPParamGroups, FSDP2 never calls
    reduce_scatter_tensor during backward for grad sync. This patch is retained as
    a safety net for any non-grad reduce_scatter calls (e.g. activation checkpointing).
    In practice it should rarely fire during the grad sync phase.

    Root cause of previous failures (v44-v58): CCL cannot access freshly sub-allocated
    XPU tensors via ANY transport. v59 fixes this by suppressing FSDP2 reduce_scatter
    entirely during backward, then doing post-backward gloo AllReduce manually.
    """
    import torch.distributed as _d

    if op is None:
        op = _d.ReduceOp.SUM
    n = _d.get_world_size(group)
    r = _d.get_rank(group)
    if input.device.type != "cpu":
        # Select gloo group by group size to match the XCCL group dimension.
        if n == _DP_SHARD_DEGREE and _GLOO_DP_SHARD_PG is not None:
            gloo_pg = _GLOO_DP_SHARD_PG
        elif n == _DP_REP_DEGREE and _GLOO_DP_REP_PG is not None:
            gloo_pg = _GLOO_DP_REP_PG
        else:
            gloo_pg = None
        if gloo_pg is not None:
            input_cpu = input.contiguous().to("cpu")
            _orig_all_reduce(input_cpu, op=op, group=gloo_pg)
            input_sum = input_cpu.to(input.device)
        else:
            # Fallback: XCCL group (only safe for dp_replicate=1/dp_shard=world_size).
            input_sum = input.clone()
            _orig_all_reduce(input_sum, op=op, group=group)
    else:
        input_sum = input.clone()
        _orig_all_reduce(input_sum, op=op, group=group)
    # v137: first-dimension slicing (supports multi-dimensional tensors from EP ReduceScatter).
    chunk_rows = output.shape[0]
    output.copy_(input_sum[r * chunk_rows : (r + 1) * chunk_rows])
    if async_op:
        return _DoneWork()


def _xpu_all_reduce_inter_node_gloo(tensor, op=None, group=None, async_op=False):
    """Drop-in for dist.all_reduce that CPU-bounces the FSDP1 HYBRID_SHARD
    inter-node grad reduction over gloo instead of XCCL/RDMA.

    Only active when ``_FSDP1_HSDP_INTER_NODE_GLOO`` is set (dense FSDP1 HSDP).
    FSDP1 HYBRID_SHARD calls ``dist.all_reduce(new_sharded_grad,
    group=state._inter_node_pg)`` every backward on the XCCL replicate group; on
    Aurora that cross-node XCCL/RDMA collective leaks CXI MR handles and crashes
    with a GPU PDE page-fault (banned:1) after ~10 steps at 84 ranks. We detect
    that specific call by matching the group's world size against
    ``_DP_REP_DEGREE`` (the inter-node / replicate dimension) and reroute it
    through ``_GLOO_DP_REP_PG`` (D2H → gloo all_reduce → H2D), mirroring the
    reduce_scatter→gloo patch used for the intra-node collective.

    Everything else — CPU tensors, non-replicate groups, the default/world group,
    and (critically) the EP/FSDP2 path where this flag stays False — falls through
    to the original XCCL all_reduce unchanged.
    """
    import torch.distributed as _d

    if op is None:
        op = _d.ReduceOp.SUM
    if (
        _FSDP1_HSDP_INTER_NODE_GLOO
        and _GLOO_DP_REP_PG is not None
        and group is not None
        and tensor.device.type != "cpu"
    ):
        try:
            _n = _d.get_world_size(group)
        except Exception:
            _n = None
        # Match the inter-node (replicate) collective by group size. The intra-node
        # reduce_scatter is already gloo-routed separately; only the replicate-dim
        # all_reduce reaches here on the XCCL fabric.
        if _n is not None and _n == _DP_REP_DEGREE and _DP_REP_DEGREE > 1:
            tensor_cpu = tensor.contiguous().to("cpu")
            _orig_all_reduce(tensor_cpu, op=op, group=_GLOO_DP_REP_PG)
            tensor.copy_(tensor_cpu.to(tensor.device))
            if async_op:
                return _DoneWork()
            return
    return _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)


def enable_fsdp1_hsdp_inter_node_gloo() -> None:
    """Activate the gloo CPU-bounce for the FSDP1 HYBRID_SHARD inter-node
    all_reduce. Call from the recipe AFTER set_process_groups() and ONLY on the
    dense FSDP1 HSDP path (not EP/FSDP2, not flat single-node)."""
    global _FSDP1_HSDP_INTER_NODE_GLOO
    _FSDP1_HSDP_INTER_NODE_GLOO = True
    _tdist_patch.all_reduce = _xpu_all_reduce_inter_node_gloo
    log.info(
        "Patched dist.all_reduce → gloo CPU-bounce for FSDP1 HSDP inter-node "
        "grad reduction (dp_rep_degree=%d) to avoid XCCL/RDMA CXI MR leak",
        _DP_REP_DEGREE,
    )


# ---------------------------------------------------------------------------
# AllToAll helpers — gloo fallback and XCCL direct routing
# ---------------------------------------------------------------------------


def _gloo_all_to_all_via_allreduce(
    output_cpu, input_cpu, output_split_sizes, input_split_sizes, group, _call_tag="fwd"
):
    """AllToAll via all_reduce split-matrix + sequential broadcasts (v65).

    v64 bug: all_gather of split sizes caused deadlock when gloo queue order diverged
    across ranks (different ranks at different AllToAll calls in the backward).
    Specifically: n_src==0 caused `continue` (skipping broadcast) on some ranks but not
    others → broadcast has too few participants → permanent deadlock.

    v65 fix: replace all_gather with all_reduce(SUM) on a (ws×ws) int64 matrix.
      - Each rank fills row `my_rank` with its input_split_sizes, zeros elsewhere.
      - all_reduce(SUM) combines all rows → same matrix on all ranks (order-invariant).
      - Buffer sizes derived from matrix (consistent → no n_src=0 mismatch between ranks).
      - Broadcast participates ALL ranks every time (no `continue` before broadcast).

    v70: Added diagnostic logging to identify "16 vs 4" crash source in backward.
    """
    import torch.distributed as _d

    ws = _d.get_world_size(group)
    my_rank = _d.get_rank(group)
    global_rank = _d.get_rank()

    splits_matrix = torch.zeros(ws * ws, dtype=torch.int64)
    splits_matrix[my_rank * ws : (my_rank + 1) * ws] = torch.tensor(
        input_split_sizes, dtype=torch.int64
    )
    log.debug(
        "Rank %d [%s]: a2a splits_matrix all_reduce: ws=%d my_rank=%d "
        "input_shape=%s input_splits=%s output_splits=%s nbytes=%d",
        global_rank,
        _call_tag,
        ws,
        my_rank,
        list(input_cpu.shape),
        input_split_sizes,
        output_split_sizes,
        splits_matrix.nbytes,
    )
    try:
        _d.all_reduce(splits_matrix, op=_d.ReduceOp.SUM, group=group)
    except Exception as _e:
        log.error(
            "Rank %d [%s]: a2a splits_matrix all_reduce FAILED: %s | "
            "ws=%d nbytes=%d input_splits=%s",
            global_rank,
            _call_tag,
            _e,
            ws,
            splits_matrix.nbytes,
            input_split_sizes,
        )
        raise
    splits_matrix = splits_matrix.view(ws, ws)

    feat_shape = input_cpu.shape[1:]
    out_off = [0]
    for s in output_split_sizes[:-1]:
        out_off.append(out_off[-1] + s)

    for src in range(ws):
        n_src = int(splits_matrix[src].sum().item())
        n_rows = output_split_sizes[src]
        global_src = _d.get_global_rank(group, src)

        if n_src == 0:
            continue

        if src == my_rank:
            data = input_cpu.contiguous()
            expected_rows = sum(input_split_sizes)
            if data.shape[0] != n_src:
                log.error(
                    "Rank %d [%s]: a2a src=%d SENDER SHAPE MISMATCH: "
                    "data.shape=%s n_src=%d (sum(input_splits)=%d) feat_shape=%s",
                    global_rank,
                    _call_tag,
                    src,
                    list(data.shape),
                    n_src,
                    expected_rows,
                    feat_shape,
                )
        else:
            data = input_cpu.new_zeros((n_src,) + feat_shape)

        log.debug(
            "Rank %d [%s]: a2a broadcast src=%d global_src=%d n_src=%d "
            "data.shape=%s data.nbytes=%d feat_shape=%s",
            global_rank,
            _call_tag,
            src,
            global_src,
            n_src,
            list(data.shape),
            data.nbytes,
            feat_shape,
        )
        try:
            _d.broadcast(data, src=global_src, group=group)
        except Exception as _e:
            log.error(
                "Rank %d [%s]: a2a broadcast FAILED src=%d global_src=%d: %s | "
                "data.shape=%s data.nbytes=%d n_src=%d feat_shape=%s "
                "input_shape=%s input_splits=%s output_splits=%s",
                global_rank,
                _call_tag,
                src,
                global_src,
                _e,
                list(data.shape),
                data.nbytes,
                n_src,
                feat_shape,
                list(input_cpu.shape),
                input_split_sizes,
                output_split_sizes,
            )
            raise

        if n_rows == 0:
            continue

        src_offset = int(splits_matrix[src][:my_rank].sum().item())
        n_rows = min(n_rows, max(0, n_src - src_offset))

        if n_rows > 0:
            output_cpu[out_off[src] : out_off[src] + n_rows].copy_(
                data[src_offset : src_offset + n_rows]
            )


def _xpu_all_to_all_via_gloo(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    """Route EP all_to_all_single via XCCL directly (v80).

    v65-v79: gloo TCP-based AllToAll caused persistent 1800s timeout deadlocks at a2a#241
    (first AllToAll of the backward pass). Root cause: ep_ranks 2,3 (0 tokens due to routing
    imbalance) intermittently fail to participate in gloo TCP collectives. All v65-v79 variants
    (gloo barriers, XCCL step-boundary syncs, pre-ref syncs) failed because the gloo TCP
    path itself is the source of the deadlock.

    v80 fix: remove gloo path entirely. Always use XCCL all_to_all_single directly.
    _XPUSyncAllToAll already adds dist.barrier(group) for OFI CQ drain (added at v47).
    XCCL was confirmed working end-to-end at v47.
    """
    global _a2a_call_counter
    import torch.distributed as _d

    if input.device.type == "xpu" and group is not None:
        n = _d.get_world_size(group)
        if n == _DP_SHARD_DEGREE:
            _a2a_call_counter += 1
    return _orig_all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=async_op,
    )


# ---------------------------------------------------------------------------
# Post-backward gradient sync for EP (reduce_grads=False path)
# ---------------------------------------------------------------------------


def _ep_post_backward_grad_sync(model: nn.Module, dp_rep_degree: int) -> int:
    """Post-backward gradient sync for EP training (v68).

    With reduce_grads=False on all FSDPParamGroups, FSDP2 skips reduce_scatter during
    backward. This function manually syncs ALL param gradients across dp_replicate after
    backward completes.

    v65 bug: `if param.grad is None: continue` caused asymmetric all_reduce participation.
    v66/v67 bug: shape inference from param shape or _local_tensor shape failed due to
      FSDP2 ZeRO-2 vs ZeRO-3 internals.
    v68 fix: two-phase approach — one all_reduce(MAX) to share canonical numels from the
      non-None ranks, then all_reduce each param's grad using the canonical numel.

    Returns number of gradients synced (params with non-None grad on this rank).
    """
    if _GLOO_DP_REP_PG is None:
        return 0

    my_rank = torch.distributed.get_rank()
    param_list = list(model.parameters())
    eff_grads = []
    eff_numels = []
    eff_dtypes = []

    for param in param_list:
        _g = param.grad
        eff_dtypes.append(param.dtype)
        if _g is not None:
            if hasattr(_g, "_local_tensor"):
                _g = _g._local_tensor
            _g_cpu = _g.detach().contiguous().to("cpu").view(-1)
            eff_grads.append(_g_cpu)
            eff_numels.append(_g_cpu.numel())
        else:
            eff_grads.append(None)
            eff_numels.append(0)

    numels_t = torch.tensor(eff_numels, dtype=torch.int64)
    log.debug(
        "Rank %d: grad_sync phase1 numel_exchange: %d params, numels_t.nbytes=%d",
        my_rank,
        len(param_list),
        numels_t.nbytes,
    )
    try:
        _orig_all_reduce(
            numels_t, op=torch.distributed.ReduceOp.MAX, group=_GLOO_DP_REP_PG
        )
    except Exception as _e:
        log.error(
            "Rank %d: grad_sync PHASE1 all_reduce FAILED: %s | "
            "numels_t.shape=%s nbytes=%d",
            my_rank,
            _e,
            numels_t.shape,
            numels_t.nbytes,
        )
        raise
    canonical_numels = numels_t.tolist()

    for i in range(len(param_list)):
        if eff_grads[i] is not None and eff_grads[i].numel() != int(
            canonical_numels[i]
        ):
            log.error(
                "Rank %d: grad_sync param[%d] NUMEL MISMATCH: "
                "eff_numel=%d canonical=%d dtype=%s param_shape=%s grad_has_local=%s",
                my_rank,
                i,
                eff_grads[i].numel(),
                int(canonical_numels[i]),
                eff_dtypes[i],
                list(param_list[i].shape),
                hasattr(param_list[i].grad, "_local_tensor"),
            )

    n_synced = 0
    for i, param in enumerate(param_list):
        numel = int(canonical_numels[i])
        if numel == 0:
            continue

        _g_cpu = eff_grads[i]
        if _g_cpu is not None:
            _g_flat = _g_cpu
            if _g_flat.numel() != numel:
                log.error(
                    "Rank %d: grad_sync param[%d] numel mismatch: "
                    "local=%d canonical=%d — padding to canonical",
                    my_rank,
                    i,
                    _g_flat.numel(),
                    numel,
                )
                if _g_flat.numel() < numel:
                    _g_flat = torch.cat(
                        [_g_flat, _g_flat.new_zeros(numel - _g_flat.numel())]
                    )
                else:
                    _g_flat = _g_flat[:numel]
        else:
            _g_flat = torch.zeros(numel, dtype=eff_dtypes[i])

        try:
            _orig_all_reduce(
                _g_flat, op=torch.distributed.ReduceOp.SUM, group=_GLOO_DP_REP_PG
            )
        except Exception as _e:
            log.error(
                "Rank %d: grad_sync param[%d] all_reduce FAILED: %s | "
                "numel=%d nbytes=%d dtype=%s canonical=%d eff_numel=%s param_shape=%s",
                my_rank,
                i,
                _e,
                _g_flat.numel(),
                _g_flat.nbytes,
                _g_flat.dtype,
                numel,
                eff_grads[i].numel() if eff_grads[i] is not None else "None",
                list(param.shape),
            )
            raise

        if _g_cpu is not None:
            _g_flat.div_(dp_rep_degree)
            _g = param.grad
            if hasattr(_g, "_local_tensor"):
                _g = _g._local_tensor
            _g.copy_(_g_flat[: _g.numel()].view(_g.shape).to(_g.device))
            n_synced += 1

    return n_synced


def _ep_post_backward_grad_sync_xccl(model: nn.Module, dp_rep_degree: int) -> int:
    """Post-backward gradient sync using XCCL dp_replicate group (v75).

    v75 replaces the gloo-based _ep_post_backward_grad_sync with direct XCCL all_reduce
    on XPU tensors via _orig_all_reduce (bypasses our monkey-patch).

    Root cause of v71-v74 failures: all attempted sync mechanisms deadlock because they
    require ALL 12 ranks simultaneously. v75 key insight: dp_replicate XCCL pairs are
    DISJOINT from dp_shard pairs (REP group {0,4,8} vs SHARD group {0,1,2,3} — no overlap),
    so the XCCL REP all_reduce can run concurrently with gloo SHARD AllToAll.

    Returns number of gradients synced.
    """
    if _XCCL_DP_REP_PG is None:
        return 0

    my_rank = torch.distributed.get_rank()
    n_synced = 0

    for param in model.parameters():
        _g = param.grad
        if _g is None:
            continue

        _g_local = _g._local_tensor if hasattr(_g, "_local_tensor") else _g

        # XCCL (oneCCL) does not support ReduceOp.AVG; use SUM + manual div.
        try:
            _orig_all_reduce(
                _g_local, op=torch.distributed.ReduceOp.SUM, group=_XCCL_DP_REP_PG
            )
        except Exception as _e:
            log.error(
                "Rank %d: XCCL grad_sync all_reduce FAILED: %s | "
                "param_shape=%s grad_shape=%s dtype=%s",
                my_rank,
                _e,
                list(param.shape),
                list(_g_local.shape),
                _g_local.dtype,
            )
            raise

        _g_local.div_(dp_rep_degree)
        n_synced += 1

    return n_synced


# ---------------------------------------------------------------------------
# v9: Per-chunk + per-step FSDP2 unsharded-grad release for EP
# ---------------------------------------------------------------------------
#
# v59 sets `FSDPParamGroup.reduce_grads = False` on all FSDPParamGroups in EP
# mode, so FSDP2's `post_backward()` never calls `foreach_reduce()`. Unsharded
# grads stay resident on `FSDPParam._unsharded_param.grad` (and across chunks at
# `unsharded_accumulated_grad`), and `nn.Parameter.grad` is None — which means
# `_ep_post_backward_grad_sync_xccl` above silently no-ops (v9a probe confirmed
# `n_synced=0` on every rank).
#
# This helper does what `foreach_reduce` would have done, but explicitly,
# sequentially, behind a 24-rank gloo barrier so EP-load-imbalance ordering
# (the original v59 motivation) cannot race.
#
# It must be called between every chunk's backward and the next chunk's forward,
# and again as a defensive sweep after `optimizer.zero_grad`. After it runs:
#   - sharded `nn.Parameter.grad` is populated as a DTensor on the param's mesh
#   - `_unsharded_param.grad` and `unsharded_accumulated_grad` are nulled
#   - `_unsharded_param`'s data storage is freed via `free_unsharded_param()`
# `_ep_post_backward_grad_sync_xccl` then has real grads to average across DP
# replicas via XCCL.


def _resolve_pg_for_param_group(pg_obj):
    """Pick the gloo PG for an FSDPParamGroup by inspecting its mesh size.

    Robust to FSDPParamGroup object replacement between init and runtime
    (which is why the helper resolves PGs on every call instead of caching).
    """
    mesh_info = getattr(pg_obj, "_mesh_info", None)
    mesh_size = None
    if mesh_info is not None:
        mesh_obj = getattr(mesh_info, "mesh", None)
        if mesh_obj is not None:
            mesh_size = mesh_obj.size()

    if mesh_size == _DP_SHARD_DEGREE and _GLOO_DP_SHARD_PG is not None:
        return _GLOO_DP_SHARD_PG
    if mesh_size == _DP_REP_DEGREE and _GLOO_DP_REP_PG is not None:
        return _GLOO_DP_REP_PG
    if _GLOO_DP_SHARD_PG is not None:
        return _GLOO_DP_SHARD_PG
    return None


def _ep_release_fsdp_unsharded_grads_streaming(
    param_groups_in_order,
    accumulate_into_grad: bool,
    warn_on_residue: bool,
) -> int:
    """Reduce and release one FSDP parameter at a time.

    This opt-in path avoids both the flattened bucket and the simultaneous
    collection of multiple unsharded gradients. It is intended for hardware
    A/B testing when one individual unsharded gradient is close to the device
    memory ceiling; the default chunked-batched path remains unchanged.
    """
    n_params_reduced = 0
    n_residue_warnings = 0
    for pg_obj in param_groups_in_order:
        gloo_pg = _resolve_pg_for_param_group(pg_obj)
        if gloo_pg is None:
            continue
        degree = gloo_pg.size()
        for fsdp_param in pg_obj.fsdp_params:
            ush_param = getattr(fsdp_param, "_unsharded_param", None)
            ush_grad = ush_param.grad if ush_param is not None else None
            acc_grad = getattr(fsdp_param, "unsharded_accumulated_grad", None)
            grad_full = acc_grad if acc_grad is not None else ush_grad
            if grad_full is None:
                log.error(
                    "_ep_release_streaming: encountered a None gradient for "
                    "an FSDPParam in a %d-rank collective group",
                    degree,
                )
                continue
            if warn_on_residue and n_residue_warnings < 3:
                log.warning(
                    "_ep_release_streaming defensive sweep found residue: numel=%d",
                    grad_full.numel(),
                )
                n_residue_warnings += 1

            shard_dim = 0
            sspec = getattr(fsdp_param, "_sharding_spec", None)
            if sspec is not None:
                for placement in sspec.placements:
                    if hasattr(placement, "dim"):
                        shard_dim = placement.dim
                        break

            use_xccl = (
                _EP_GRAD_RELEASE_XCCL
                and _XCCL_DP_SHARD_PG is not None
                and gloo_pg is _GLOO_DP_SHARD_PG
                and grad_full.device.type == "xpu"
                and degree > 1
            )
            collective_pg = _XCCL_DP_SHARD_PG if use_xccl else gloo_pg
            reduced = grad_full.contiguous()
            if not use_xccl:
                reduced = reduced.cpu()
            if degree > 1:
                _orig_all_reduce(
                    reduced, op=torch.distributed.ReduceOp.SUM, group=collective_pg
                )
                reduced.div_(degree)

            if degree > 1:
                local_rank = torch.distributed.get_rank(collective_pg)
                chunks = list(torch.chunk(reduced, degree, dim=shard_dim))
                if local_rank < len(chunks):
                    local_chunk = chunks[local_rank]
                else:
                    empty_shape = list(reduced.shape)
                    empty_shape[shard_dim] = 0
                    local_chunk = torch.zeros(
                        empty_shape, dtype=reduced.dtype, device=reduced.device
                    )
            else:
                local_chunk = reduced

            sharded_param = fsdp_param.sharded_param
            target_size = fsdp_param.sharded_size
            current = local_chunk.size(shard_dim) if local_chunk.dim() > shard_dim else 0
            target = target_size[shard_dim] if len(target_size) > shard_dim else 0
            if current < target:
                pad_shape = list(local_chunk.shape)
                pad_shape[shard_dim] = target - current
                local_chunk = torch.cat(
                    [
                        local_chunk,
                        torch.zeros(
                            pad_shape,
                            dtype=local_chunk.dtype,
                            device=local_chunk.device,
                        ),
                    ],
                    dim=shard_dim,
                )
            elif current > target:
                local_chunk = local_chunk.narrow(shard_dim, 0, target)
            local_shard = local_chunk.contiguous().to(
                device=sharded_param.device, dtype=sharded_param.dtype
            )
            if sharded_param.grad is None or not accumulate_into_grad:
                sharded_param.grad = fsdp_param.to_sharded_dtensor(local_shard)
            else:
                existing = sharded_param.grad
                existing_local = (
                    existing._local_tensor if hasattr(existing, "_local_tensor") else existing
                )
                existing_local.add_(local_shard)

            if ush_param is not None:
                ush_param.grad = None
            fsdp_param.unsharded_accumulated_grad = None
            try:
                fsdp_param.free_unsharded_param()
            except Exception as free_exc:
                if n_params_reduced < 3:
                    log.warning(
                        "_ep_release_streaming: free_unsharded_param failed: %r",
                        free_exc,
                    )
            del reduced, local_chunk, grad_full, ush_grad, acc_grad
            n_params_reduced += 1
    return n_params_reduced


def _ep_release_fsdp_unsharded_grads(
    model,  # nn.Module — the policy model
    _unused_pg_map,  # legacy arg kept for call-site compat; ignored
    accumulate_into_grad: bool,
    warn_on_residue: bool = False,  # True only for post-step defensive sweep
) -> int:
    """Release FSDP2's unsharded grad pool into sharded `param.grad` (v9, v167 chunked-batched).

    Walks every FSDPParamGroup in fixed order. Behind a global gloo barrier,
    reduce-scatters each FSDPParam's unsharded grad onto the rank-local shard
    of `param.grad` (constructed as a DTensor on the first call, accumulated
    in-place on subsequent calls). Then frees the unsharded pool.

    v166 (2026-07-24): first cut — one `all_reduce` per distinct (resolved
    PG, dtype) bucket instead of one per FSDPParam. py-spy profiling on
    Qwen3-30B-A3B/EP=8 SFT found ~60% of wall-clock time in this function's
    all_reduce, called ~97+ separate times per microbatch — per-collective
    launch/sync overhead, not payload bytes, dominates at this per-call size.
    REVERTED same day: v166 batched ALL ~97 groups' unsharded grads into ONE
    giant flatten+all_reduce with zero interleaved cleanup — every group's
    `.contiguous()` copy plus the flattened buffer were resident
    SIMULTANEOUSLY (vs the old loop's one-group-at-a-time + immediate
    `free_unsharded_param()`), causing a step-2 `UR_RESULT_ERROR_OUT_OF_RESOURCES`
    L0 exhaustion HW-confirmed on Qwen3-30B-A3B/EP=8 (same failure class as
    the v161/v162/v163 full-size-temp-tensor history in this same function).

    v167 (2026-07-24): process FSDPParamGroups in fixed-size CHUNKS
    (`_EP_GRAD_RELEASE_CHUNK_GROUPS`). Each chunk collects its entries,
    batches ONE all_reduce per (PG, dtype) bucket WITHIN the chunk, then
    immediately runs the per-param chunk/pad/slice/cast/accumulate/cleanup
    (including `free_unsharded_param()`) for that chunk before moving to the
    next — bounding peak transient memory to one chunk's worth of unsharded
    grads instead of the whole model, while still cutting the collective
    count by ~chunk_size vs the original per-param loop. All PG-selection,
    XCCL/gloo choice, and post-collective math is UNCHANGED. First HW
    attempt (chunk=8, no clone below) got much further than v166 — 9 clean
    steps vs 2 — then hit a `banned:1` GPU page fault: `_unflatten_dense_tensors`
    returns VIEWS into the flat collective buffer, and without an explicit
    copy those views could reach `sharded_param.grad` (torch.chunk of a
    contiguous tensor stays contiguous; the later `.to()` is a no-op when
    dtype/device already match) — keeping the WHOLE chunk's flat buffer
    alive per param, accumulating across steps. Fixed by `.clone()`-ing each
    param's slice immediately after unflatten, before any further chunk/pad/
    slice reuses the view.

    v169 (2026-07-24): a codex-review pass on v168 flagged that the
    original py-spy 60% share cannot distinguish removable launch overhead
    from real communication/sync wait, and that at the default chunk=8
    the actual reduction is ~97 collectives -> ~13, not ~97 -> ~1 (a much
    smaller change than "1.5% win despite 60% share" implied). Added
    `TORCHTUNE_EP_GRAD_RELEASE_TIMING=1` to measure barrier / collect+flatten
    / all_reduce-itself / unflatten+clone+Pass3 separately, XPU-synchronized,
    so the next lever decision is based on where time actually goes rather
    than another blind chunk-size guess. See
    memory/project_moe_sft_profiling_gradrelease_bottleneck_20260724.md.

    Args:
        model: FSDP2-wrapped policy model.
        mesh_to_pg_map: {id(FSDPParamGroup): gloo PG to reduce-scatter on}.
            Built once at recipe init via `_ep_build_grad_release_pg_map`.
        accumulate_into_grad: True for chunk N>0 (sums into existing grad);
            False for chunk 0 OR for the post-step defensive sweep
            (overwrites; defensive sweep should find zero residue).

    Returns:
        Number of FSDPParamGroups processed (for logging / asserts).
    """
    try:
        from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
    except ImportError:
        log.warning("_ep_release_fsdp_unsharded_grads: FSDP2 not available")
        return 0

    _timing = _EP_GRAD_RELEASE_TIMING and torch.distributed.get_rank() == 0
    if _timing:
        import time as _time_v169

        _t_barrier = _t_collect = _t_allreduce = _t_pass3 = 0.0
        _n_collectives = 0

        def _xpu_sync_v169():
            if torch.xpu.is_available():
                torch.xpu.synchronize()

        _t0 = _time_v169.perf_counter()

    if _GLOO_GLOBAL_PG is not None:
        torch.distributed.barrier(group=_GLOO_GLOBAL_PG)

    if _timing:
        _xpu_sync_v169()
        _t_barrier = _time_v169.perf_counter() - _t0

    seen_pg_ids: set = set()
    param_groups_in_order = []
    for _name, mod in model.named_modules():
        if not isinstance(mod, FSDPModule):
            continue
        fsdp_state = mod._get_fsdp_state()
        if fsdp_state is None:
            continue
        pg_obj = getattr(fsdp_state, "_fsdp_param_group", None)
        if pg_obj is None or id(pg_obj) in seen_pg_ids:
            continue
        seen_pg_ids.add(id(pg_obj))
        param_groups_in_order.append(pg_obj)

    if _EP_GRAD_RELEASE_STREAMING:
        return _ep_release_fsdp_unsharded_grads_streaming(
            param_groups_in_order,
            accumulate_into_grad=accumulate_into_grad,
            warn_on_residue=warn_on_residue,
        )

    # v167: process param groups in bounded-size CHUNKS. Collecting every
    # group's unsharded-grad copy up front (v166) and batching ALL of them
    # into one collective left the entire model's unsharded grads (plus a
    # freshly flattened copy) resident simultaneously — HW-confirmed
    # UR_RESULT_ERROR_OUT_OF_RESOURCES at step 2 on Qwen3-30B-A3B/EP=8.
    # Chunking bounds peak transient memory to one chunk while still
    # collapsing ~chunk_size collectives into 1-2 per chunk.
    n_groups = 0
    n_residue_warnings = 0
    n_params_reduced = 0

    for chunk_start in range(
        0, len(param_groups_in_order), _EP_GRAD_RELEASE_CHUNK_GROUPS
    ):
        chunk_pgs = param_groups_in_order[
            chunk_start : chunk_start + _EP_GRAD_RELEASE_CHUNK_GROUPS
        ]

        if _timing:
            _xpu_sync_v169()
            _t1 = _time_v169.perf_counter()

        # --- Pass 1 (this chunk only): collect, no collective yet ----------
        # Bucketed by (id(collective_pg), dtype) so dp_replicate>1 configs
        # (which can mix dp_shard- and dp_replicate-sized groups needing
        # different PGs) never merge incompatible collectives.
        entries = []
        for pg_obj in chunk_pgs:
            gloo_pg = _resolve_pg_for_param_group(pg_obj)
            if gloo_pg is None:
                continue
            degree = gloo_pg.size()
            for fsdp_param in pg_obj.fsdp_params:
                ush_param = getattr(fsdp_param, "_unsharded_param", None)
                ush_grad = ush_param.grad if ush_param is not None else None
                acc_grad = getattr(fsdp_param, "unsharded_accumulated_grad", None)
                grad_full = acc_grad if acc_grad is not None else ush_grad
                if grad_full is None:
                    # HARDENING NOTE (codex review, 2026-07-25): skipping a None
                    # grad here participates in the "bucket" for this (PG, dtype)
                    # independently per rank — if ranks in the SAME collective
                    # disagreed on which params have a grad, they'd flatten
                    # different tensor lists into different-sized buffers and
                    # the all_reduce below would hang/mismatch. This is the
                    # exact asymmetric-participation bug class
                    # _ep_post_backward_grad_sync had to fix (v65, see that
                    # function's docstring above) with a two-phase numel
                    # exchange. It does NOT currently reproduce on this
                    # architecture: every Qwen3MoeTransformerLayer/GroupedExpertsHF
                    # runs router+attention+experts unconditionally each
                    # forward, and GroupedExpertsHF.forward's zero-token branch
                    # (torchtune/models/qwen3_moe/_experts.py, the `total == 0`
                    # early return) still produces a differentiable output that
                    # touches every expert param — so no FSDPParam should reach
                    # backward with a genuinely absent grad while its peers on
                    # other ranks have one. Logging loudly (not just a silent
                    # continue) so a future model/dataset change that DOES
                    # trigger this is caught immediately instead of manifesting
                    # as a rare hang.
                    log.error(
                        "_ep_release_fsdp_unsharded_grads: encountered a None "
                        "gradient for an FSDPParam in a %d-rank collective group "
                        "— if peer ranks have a non-None grad for the "
                        "corresponding param, this WILL desync the batched "
                        "all_reduce (mismatched flattened buffer sizes). This "
                        "is not expected on the current model architecture; "
                        "investigate before trusting this run's gradients.",
                        degree,
                    )
                    continue

                if warn_on_residue and n_residue_warnings < 3:
                    log.warning(
                        "_ep_release defensive sweep found residue: numel=%d (expected 0 after zero_grad)",
                        grad_full.numel(),
                    )
                    n_residue_warnings += 1

                # FSDP2 shards along dim 0 by default (see FSDPParam._init_sharded_param);
                # `_post_forward_mesh_info.shard_mesh_dim` is the canonical knob, but
                # `sharded_size` already encodes the local shape. Resolve shard_dim from
                # the placements of the existing sharding spec.
                shard_dim = 0
                sspec = getattr(fsdp_param, "_sharding_spec", None)
                if sspec is not None:
                    for plc in sspec.placements:
                        if hasattr(plc, "dim"):
                            shard_dim = plc.dim
                            break

                if degree > 1:
                    # Iter2 opt-in: native XCCL all_reduce on XPU when the dp_shard
                    # XCCL PG is registered AND the gloo PG matches dp_shard. For
                    # dp_replicate>1 we keep the gloo path because we have no
                    # XCCL dp_replicate group on the same mesh shape.
                    use_xccl = (
                        _EP_GRAD_RELEASE_XCCL
                        and _XCCL_DP_SHARD_PG is not None
                        and gloo_pg is _GLOO_DP_SHARD_PG
                        and grad_full.device.type == "xpu"
                    )
                    collective_pg = _XCCL_DP_SHARD_PG if use_xccl else gloo_pg
                    grad_for_reduce = (
                        grad_full.contiguous()
                        if use_xccl
                        else grad_full.contiguous().cpu()
                    )
                else:
                    use_xccl = False
                    collective_pg = None
                    grad_for_reduce = grad_full.contiguous().cpu()

                entries.append(
                    {
                        "fsdp_param": fsdp_param,
                        "grad_for_reduce": grad_for_reduce,
                        "degree": degree,
                        "shard_dim": shard_dim,
                        "use_xccl": use_xccl,
                        "collective_pg": collective_pg,
                    }
                )
            n_groups += 1

        if _timing:
            _xpu_sync_v169()
            _t_collect += _time_v169.perf_counter() - _t1

        # --- Pass 2 (this chunk only): one batched collective per bucket ---
        # degree==1 entries have collective_pg=None — no collective needed,
        # the local_chunk is just grad_for_reduce itself (old `else` branch).
        buckets: dict = {}
        for i, e in enumerate(entries):
            if e["collective_pg"] is None:
                continue
            key = (id(e["collective_pg"]), e["grad_for_reduce"].dtype)
            buckets.setdefault(key, []).append(i)

        for (_pg_id, _dtype), idxs in buckets.items():
            pg = entries[idxs[0]]["collective_pg"]
            if _EP_GRAD_RELEASE_LEGACY:
                for i in idxs:
                    grad = entries[i]["grad_for_reduce"]
                    _orig_all_reduce(
                        grad, op=torch.distributed.ReduceOp.SUM, group=pg
                    )
                    grad.div_(entries[i]["degree"])
            else:
                tensors = [entries[i]["grad_for_reduce"] for i in idxs]
                flat = torch._utils._flatten_dense_tensors(tensors)
                if _timing:
                    _xpu_sync_v169()
                    _t2 = _time_v169.perf_counter()
                _orig_all_reduce(flat, op=torch.distributed.ReduceOp.SUM, group=pg)
                if _timing:
                    _xpu_sync_v169()
                    _t_allreduce += _time_v169.perf_counter() - _t2
                    _n_collectives += 1
                flat.div_(entries[idxs[0]]["degree"])
                reduced = torch._utils._unflatten_dense_tensors(flat, tensors)
                # _unflatten_dense_tensors returns VIEWS into `flat`. Without a
                # forced .clone() here, torch.chunk() (dim-0 slices of a
                # contiguous tensor stay contiguous) and the later no-op .to()
                # can pass that view straight through to sharded_param.grad —
                # keeping the ENTIRE chunk's flat buffer alive per param, every
                # step, until the DTensor is replaced. HW-confirmed: without
                # this clone, a single-chunk-of-8 build ran 9 clean steps then
                # hit a `banned:1` GPU page fault (progressive resource
                # exhaustion), consistent with buffers accumulating instead of
                # being freed each step.
                for i, r in zip(idxs, reduced):
                    entries[i]["grad_for_reduce"] = r.clone()
                del flat, reduced, tensors

        if _timing:
            _xpu_sync_v169()
            _t3 = _time_v169.perf_counter()

        # --- Pass 3 (this chunk only): per-param finish + IMMEDIATE cleanup ---
        # free_unsharded_param() here (before the next chunk's Pass 1 starts
        # collecting) is exactly what bounds peak memory to one chunk.
        for e in entries:
            fsdp_param = e["fsdp_param"]
            grad_reduced = e["grad_for_reduce"]
            degree = e["degree"]
            shard_dim = e["shard_dim"]
            use_xccl = e["use_xccl"]

            sharded_param = fsdp_param.sharded_param
            target_size = fsdp_param.sharded_size

            if degree > 1:
                if use_xccl:
                    local_rank_in_pg = torch.distributed.get_rank(_XCCL_DP_SHARD_PG)
                else:
                    local_rank_in_pg = torch.distributed.get_rank(e["collective_pg"])
                # FSDP2 shards via torch.chunk along shard_dim — chunks may be
                # uneven on the last rank if dim_size % degree != 0, and may
                # produce fewer than `degree` chunks when degree > dim_size
                # (e.g. EP=16 with norm-weight grads of shape [hidden_dim]).
                # FSDP2's sharded_size already accounts for these padded ranks;
                # produce a 0-length local_chunk and let the pad block fill it.
                torch_chunks = list(torch.chunk(grad_reduced, degree, dim=shard_dim))
                if local_rank_in_pg < len(torch_chunks):
                    local_chunk = torch_chunks[local_rank_in_pg]
                else:
                    empty_shape = list(grad_reduced.shape)
                    empty_shape[shard_dim] = 0
                    local_chunk = torch.zeros(
                        empty_shape,
                        dtype=grad_reduced.dtype,
                        device=grad_reduced.device,
                    )
            else:
                local_chunk = grad_reduced

            # Pad/slice along shard_dim to match sharded_size exactly (handles
            # the uneven last-shard case FSDP2 internally pads with zeros).
            cur = local_chunk.size(shard_dim) if local_chunk.dim() > shard_dim else 0
            tgt = target_size[shard_dim] if len(target_size) > shard_dim else 0
            if cur != tgt:
                if cur < tgt:
                    pad_shape = list(local_chunk.shape)
                    pad_shape[shard_dim] = tgt - cur
                    # Match local_chunk's device — under TORCHTUNE_EP_GRAD_RELEASE_XCCL
                    # local_chunk lives on XPU, so a CPU pad would fail torch.cat.
                    pad = torch.zeros(
                        pad_shape, dtype=local_chunk.dtype, device=local_chunk.device
                    )
                    local_chunk = torch.cat([local_chunk, pad], dim=shard_dim)
                else:
                    local_chunk = local_chunk.narrow(shard_dim, 0, tgt)

            local_shard = local_chunk.contiguous().to(
                device=sharded_param.device,
                dtype=sharded_param.dtype,
                non_blocking=False,
            )

            if sharded_param.grad is None or not accumulate_into_grad:
                sharded_param.grad = fsdp_param.to_sharded_dtensor(local_shard)
            else:
                existing = sharded_param.grad
                existing_local = (
                    existing._local_tensor
                    if hasattr(existing, "_local_tensor")
                    else existing
                )
                existing_local.add_(local_shard)

            ush_param = getattr(fsdp_param, "_unsharded_param", None)
            if ush_param is not None:
                ush_param.grad = None
            fsdp_param.unsharded_accumulated_grad = None
            try:
                fsdp_param.free_unsharded_param()
            except Exception as _free_exc:
                if n_params_reduced < 3:
                    log.warning(
                        "_ep_release: free_unsharded_param failed for fsdp_param "
                        "(idx=%d): %r",
                        n_params_reduced,
                        _free_exc,
                    )
            n_params_reduced += 1

        if _timing:
            _xpu_sync_v169()
            _t_pass3 += _time_v169.perf_counter() - _t3

        del entries

    if _timing:
        _t_total = _t_barrier + _t_collect + _t_allreduce + _t_pass3
        log.info(
            "_ep_release TIMING (rank0, %d groups, %d collectives): "
            "barrier=%.4fs collect+flatten=%.4fs all_reduce=%.4fs pass3=%.4fs "
            "total=%.4fs (all_reduce/total=%.1f%%)",
            n_groups,
            _n_collectives,
            _t_barrier,
            _t_collect,
            _t_allreduce,
            _t_pass3,
            _t_total,
            100.0 * _t_allreduce / _t_total if _t_total > 0 else 0.0,
        )

    return n_groups


def _ep_build_grad_release_pg_map(model) -> dict:
    """Build {id(FSDPParamGroup): gloo PG} for `_ep_release_fsdp_unsharded_grads`.

    Each FSDPParamGroup carries a 1D mesh whose size determines whether it's an
    expert group (sharded on dp_shard 4-rank mesh) or non-expert (sharded on
    dp_replicate, but in EP mode this is the dp_shard mesh too — see v59 comment
    at recipe :1440. Both wind up on the gloo SHARD PG.) Walk every group, look
    at its mesh size, and route to the matching gloo PG.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
    except ImportError:
        return {}

    mapping: dict = {}
    seen_pg_ids: set = set()
    for _name, mod in model.named_modules():
        if not isinstance(mod, FSDPModule):
            continue
        fsdp_state = mod._get_fsdp_state()
        if fsdp_state is None:
            continue
        pg_obj = getattr(fsdp_state, "_fsdp_param_group", None)
        if pg_obj is None or id(pg_obj) in seen_pg_ids:
            continue
        seen_pg_ids.add(id(pg_obj))

        mesh = getattr(pg_obj, "_mesh_info", None)
        mesh_size = None
        if mesh is not None:
            mesh_obj = getattr(mesh, "mesh", None)
            if mesh_obj is not None:
                mesh_size = mesh_obj.size()

        if mesh_size == _DP_SHARD_DEGREE and _GLOO_DP_SHARD_PG is not None:
            mapping[id(pg_obj)] = _GLOO_DP_SHARD_PG
        elif mesh_size == _DP_REP_DEGREE and _GLOO_DP_REP_PG is not None:
            mapping[id(pg_obj)] = _GLOO_DP_REP_PG
        elif _GLOO_DP_SHARD_PG is not None:
            mapping[id(pg_obj)] = _GLOO_DP_SHARD_PG
            log.debug(
                "_ep_build_grad_release_pg_map: FSDPParamGroup '%s' mesh_size=%s "
                "matches neither dp_shard (%d) nor dp_replicate (%d); defaulting to dp_shard PG"
                " (benign for CPUOffloadPolicy non-expert groups with reduce_grads=True)",
                _name,
                mesh_size,
                _DP_SHARD_DEGREE,
                _DP_REP_DEGREE,
            )

    log.info(
        "_ep_build_grad_release_pg_map: routed %d FSDPParamGroups to gloo PGs (dp_shard=%d, dp_replicate=%d)",
        len(mapping),
        _DP_SHARD_DEGREE,
        _DP_REP_DEGREE,
    )
    return mapping


# ---------------------------------------------------------------------------
# XPU-safe memory management
# ---------------------------------------------------------------------------


def device_empty_cache(device: torch.device) -> None:
    """XPU-safe drop-in for torchtune.training.device_empty_cache.

    NEVER call empty_cache() on XPU with FSDP. The combination of empty_cache() +
    FSDP storage.resize_() leaks UR handles in Level Zero, causing
    UR_RESULT_ERROR_OUT_OF_RESOURCES after ~70 iters. See
    docs/bugs/intel_xpu_resource_leak_bug_report.md.
    """
    if device.type == "xpu":
        pass
    else:
        _orig_device_empty_cache(device)


def _safe_empty_cache(device: torch.device) -> None:
    """Barrier + synchronize before cache clearing.

    On XPU this is a no-op — empty_cache() + FSDP leaks UR handles.
    """
    torch.distributed.barrier()
    if device.type == "xpu":
        torch.xpu.synchronize()
        return
    _orig_device_empty_cache(device)


# ---------------------------------------------------------------------------
# Trajectory slicing utility
# ---------------------------------------------------------------------------


def _slice_trajectory(
    trajectory: GRPOTrajectory, start: int, end: int
) -> GRPOTrajectory:
    """Slice a GRPOTrajectory along the batch dimension."""
    fields = {}
    for field_name in trajectory._fields:
        val = getattr(trajectory, field_name)
        if isinstance(val, torch.Tensor):
            fields[field_name] = val[start:end]
        elif isinstance(val, list):
            fields[field_name] = val[start:end]
        else:
            fields[field_name] = val
    return GRPOTrajectory(**fields)


# ---------------------------------------------------------------------------
# Patch installation — call once at module level in the recipe
# ---------------------------------------------------------------------------


def install_xpu_patches() -> None:
    """Apply all XPU-specific monkey-patches to torch.distributed ops.

    Must be called after imports but before any distributed ops or model setup.
    Idempotent — safe to call multiple times (subsequent calls re-apply the same patches).
    """
    # Patch 1: FSDP2 gradient divide factor — force SUM reduction (XCCL lacks AVG).
    # FSDP2's reduce_scatter uses ReduceOp.AVG; MTIA upstream uses the same workaround.
    # v56 bug: used _GLOO_DP_REP_PG (3 ranks) for step 1 (dp_shard, 4 ranks) — mismatch.
    # v57 fix: _GLOO_DP_SHARD_PG for reduce_scatter, _GLOO_DP_REP_PG for all_reduce.
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll

        _orig_gdf = _fsdp_coll._get_gradient_divide_factors

        def _xpu_get_gradient_divide_factors(*args, **kwargs):
            if len(args) >= 4 and args[3] == "xpu":
                args = list(args)
                if len(args) >= 6:
                    args[5] = True
                else:
                    kwargs["force_sum_reduction_for_comms"] = True
                args = tuple(args)
            elif kwargs.get("device_type") == "xpu":
                kwargs["force_sum_reduction_for_comms"] = True
            return _orig_gdf(*args, **kwargs)

        _fsdp_coll._get_gradient_divide_factors = _xpu_get_gradient_divide_factors
        log.info(
            "Patched FSDP2 _get_gradient_divide_factors for XPU (force SUM reduction)"
        )
    except Exception as e:
        log.warning("Failed to patch FSDP2 for XPU: %s", e)

    # Patch 2: reduce_scatter_tensor → AllReduce-based fallback.
    # v59: With reduce_grads=False on all FSDPParamGroups, FSDP2 never calls
    # reduce_scatter_tensor during backward. This patch is a safety net only.
    _tdist_patch.reduce_scatter_tensor = _xpu_reduce_scatter_via_allreduce
    log.info(
        "Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)"
    )
    log.info(
        "dist.all_reduce NOT patched (v59: reduce_grads=False on all FSDP2 groups)"
    )

    # Patch 3: all_to_all_single → XCCL direct (v80: gloo path removed).
    _tdist_patch.all_to_all_single = _xpu_all_to_all_via_gloo
    log.info(
        "Patched dist.all_to_all_single → XCCL all_to_all_single for XPU EP tensors (v80)"
    )


def restore_native_xpu_reduce_scatter() -> None:
    """Restore PyTorch's native reduce-scatter after ``install_xpu_patches``."""
    _tdist_patch.reduce_scatter_tensor = _orig_reduce_scatter_tensor
    log.info("Restored native dist.reduce_scatter_tensor for FSDP gradient reduction")


# ---------------------------------------------------------------------------
# Process group registration — call from _init_distributed after group creation
# ---------------------------------------------------------------------------


def set_process_groups(
    gloo_dp_rep_pg,
    gloo_dp_shard_pg,
    gloo_global_pg,
    xccl_dp_rep_pg,
    dp_rep_degree: int,
    dp_shard_degree: int,
    xccl_dp_shard_pg=None,
) -> None:
    """Register process group handles used by distributed op patches.

    Called from GRPOFullFinetuneDistributedXPU._init_distributed() after all gloo
    and XCCL process groups have been created. The registered handles are used by
    _xpu_reduce_scatter_via_allreduce, _ep_post_backward_grad_sync[_xccl], etc.

    `xccl_dp_shard_pg` is optional and only consumed by the iter2 opt-in
    `_ep_release_fsdp_unsharded_grads` XCCL path (TORCHTUNE_EP_GRAD_RELEASE_XCCL=1).
    Pass `device_mesh.get_group("dp_shard")` from the recipe.
    """
    global _GLOO_DP_REP_PG, _GLOO_DP_SHARD_PG, _GLOO_GLOBAL_PG, _XCCL_DP_REP_PG
    global _XCCL_DP_SHARD_PG, _DP_REP_DEGREE, _DP_SHARD_DEGREE
    _GLOO_DP_REP_PG = gloo_dp_rep_pg
    _GLOO_DP_SHARD_PG = gloo_dp_shard_pg
    _GLOO_GLOBAL_PG = gloo_global_pg
    _XCCL_DP_REP_PG = xccl_dp_rep_pg
    _XCCL_DP_SHARD_PG = xccl_dp_shard_pg
    _DP_REP_DEGREE = dp_rep_degree
    _DP_SHARD_DEGREE = dp_shard_degree

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
_GLOO_DP_REP_PG = None    # gloo mirror of dp_replicate_pg (3 ranks)
_GLOO_DP_SHARD_PG = None  # gloo mirror of dp_shard_pg (4 ranks)
_GLOO_GLOBAL_PG = None    # gloo global group (all ranks); barrier before post-bwd AllReduce
_XCCL_DP_REP_PG = None   # XCCL dp_replicate group (3 ranks, XPU fabric)
_DP_REP_DEGREE = 1        # dp_replicate world size
_DP_SHARD_DEGREE = 1      # dp_shard world size

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

def _no_reentrant_ac_wrapper(module):
    return ptd_checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.REENTRANT,
        checkpoint_fn=_torch_checkpoint,
        use_reentrant=False,  # v154: reverted from True (v114 reason — AllToAll BWD SIGSEGV — gone since v141)
        preserve_rng_state=False,
        determinism_check="none",
    )


def _apply_split_ac(model):
    """v158: Split AC so MoE-bearing layers checkpoint attention only (MoE runs once, never recomputed).

    Both Gemma4TransformerLayer and Qwen3MoeTransformerLayer implement _ac_enabled + the
    self-checkpoint contract. All other TransformerSelfAttentionLayer instances get the
    standard _no_reentrant_ac_wrapper.

    Returns the number of MoE-bearing layers detected.
    """
    from torchtune.models.gemma4._component_builders import Gemma4TransformerLayer
    from torchtune.models.qwen3_moe._component_builders import Qwen3MoeTransformerLayer

    moe_layer_ids = set()
    for m in model.modules():
        if isinstance(m, Gemma4TransformerLayer) and m.moe_block is not None:
            m._ac_enabled = True
            moe_layer_ids.add(id(m))
        elif isinstance(m, Qwen3MoeTransformerLayer):
            m._ac_enabled = True
            moe_layer_ids.add(id(m))

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
    return len(moe_layer_ids)


# ---------------------------------------------------------------------------
# Fake Work object for synchronous ops returning as if async
# ---------------------------------------------------------------------------

class _DoneWork:
    """Fake Work object for synchronous ops masquerading as async."""
    def wait(self): pass
    def is_completed(self): return True
    def get_future(self):
        import torch.futures as _tf
        f = _tf.Future()
        f.set_result(None)
        return f


# ---------------------------------------------------------------------------
# reduce_scatter_tensor patch — AllReduce-based fallback (XPU/CCL workaround)
# ---------------------------------------------------------------------------

def _xpu_reduce_scatter_via_allreduce(output, input, op=None, group=None, async_op=False):
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
    if input.device.type != 'cpu':
        # Select gloo group by group size to match the XCCL group dimension.
        if n == _DP_SHARD_DEGREE and _GLOO_DP_SHARD_PG is not None:
            gloo_pg = _GLOO_DP_SHARD_PG
        elif n == _DP_REP_DEGREE and _GLOO_DP_REP_PG is not None:
            gloo_pg = _GLOO_DP_REP_PG
        else:
            gloo_pg = None
        if gloo_pg is not None:
            input_cpu = input.contiguous().to('cpu')
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


# ---------------------------------------------------------------------------
# AllToAll helpers — gloo fallback and XCCL direct routing
# ---------------------------------------------------------------------------

def _gloo_all_to_all_via_allreduce(output_cpu, input_cpu, output_split_sizes, input_split_sizes, group,
                                   _call_tag="fwd"):
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
    log.debug("Rank %d [%s]: a2a splits_matrix all_reduce: ws=%d my_rank=%d "
              "input_shape=%s input_splits=%s output_splits=%s nbytes=%d",
              global_rank, _call_tag, ws, my_rank, list(input_cpu.shape),
              input_split_sizes, output_split_sizes, splits_matrix.nbytes)
    try:
        _d.all_reduce(splits_matrix, op=_d.ReduceOp.SUM, group=group)
    except Exception as _e:
        log.error("Rank %d [%s]: a2a splits_matrix all_reduce FAILED: %s | "
                  "ws=%d nbytes=%d input_splits=%s",
                  global_rank, _call_tag, _e, ws, splits_matrix.nbytes, input_split_sizes)
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
                log.error("Rank %d [%s]: a2a src=%d SENDER SHAPE MISMATCH: "
                          "data.shape=%s n_src=%d (sum(input_splits)=%d) feat_shape=%s",
                          global_rank, _call_tag, src, list(data.shape), n_src, expected_rows, feat_shape)
        else:
            data = input_cpu.new_zeros((n_src,) + feat_shape)

        log.debug("Rank %d [%s]: a2a broadcast src=%d global_src=%d n_src=%d "
                  "data.shape=%s data.nbytes=%d feat_shape=%s",
                  global_rank, _call_tag, src, global_src, n_src,
                  list(data.shape), data.nbytes, feat_shape)
        try:
            _d.broadcast(data, src=global_src, group=group)
        except Exception as _e:
            log.error("Rank %d [%s]: a2a broadcast FAILED src=%d global_src=%d: %s | "
                      "data.shape=%s data.nbytes=%d n_src=%d feat_shape=%s "
                      "input_shape=%s input_splits=%s output_splits=%s",
                      global_rank, _call_tag, src, global_src, _e,
                      list(data.shape), data.nbytes, n_src, feat_shape,
                      list(input_cpu.shape), input_split_sizes, output_split_sizes)
            raise

        if n_rows == 0:
            continue

        src_offset = int(splits_matrix[src][:my_rank].sum().item())
        n_rows = min(n_rows, max(0, n_src - src_offset))

        if n_rows > 0:
            output_cpu[out_off[src]:out_off[src] + n_rows].copy_(
                data[src_offset:src_offset + n_rows]
            )


def _xpu_all_to_all_via_gloo(output, input, output_split_sizes=None,
                               input_split_sizes=None, group=None, async_op=False):
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
    if input.device.type == 'xpu' and group is not None:
        n = _d.get_world_size(group)
        if n == _DP_SHARD_DEGREE:
            _a2a_call_counter += 1
    return _orig_all_to_all_single(
        output, input,
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
            if hasattr(_g, '_local_tensor'):
                _g = _g._local_tensor
            _g_cpu = _g.detach().contiguous().to('cpu').view(-1)
            eff_grads.append(_g_cpu)
            eff_numels.append(_g_cpu.numel())
        else:
            eff_grads.append(None)
            eff_numels.append(0)

    numels_t = torch.tensor(eff_numels, dtype=torch.int64)
    log.debug("Rank %d: grad_sync phase1 numel_exchange: %d params, numels_t.nbytes=%d",
              my_rank, len(param_list), numels_t.nbytes)
    try:
        _orig_all_reduce(numels_t, op=torch.distributed.ReduceOp.MAX, group=_GLOO_DP_REP_PG)
    except Exception as _e:
        log.error("Rank %d: grad_sync PHASE1 all_reduce FAILED: %s | "
                  "numels_t.shape=%s nbytes=%d",
                  my_rank, _e, numels_t.shape, numels_t.nbytes)
        raise
    canonical_numels = numels_t.tolist()

    for i in range(len(param_list)):
        if eff_grads[i] is not None and eff_grads[i].numel() != int(canonical_numels[i]):
            log.error("Rank %d: grad_sync param[%d] NUMEL MISMATCH: "
                      "eff_numel=%d canonical=%d dtype=%s param_shape=%s grad_has_local=%s",
                      my_rank, i, eff_grads[i].numel(), int(canonical_numels[i]),
                      eff_dtypes[i], list(param_list[i].shape),
                      hasattr(param_list[i].grad, '_local_tensor'))

    n_synced = 0
    for i, param in enumerate(param_list):
        numel = int(canonical_numels[i])
        if numel == 0:
            continue

        _g_cpu = eff_grads[i]
        if _g_cpu is not None:
            _g_flat = _g_cpu
            if _g_flat.numel() != numel:
                log.error("Rank %d: grad_sync param[%d] numel mismatch: "
                          "local=%d canonical=%d — padding to canonical",
                          my_rank, i, _g_flat.numel(), numel)
                if _g_flat.numel() < numel:
                    _g_flat = torch.cat([_g_flat, _g_flat.new_zeros(numel - _g_flat.numel())])
                else:
                    _g_flat = _g_flat[:numel]
        else:
            _g_flat = torch.zeros(numel, dtype=eff_dtypes[i])

        try:
            _orig_all_reduce(_g_flat, op=torch.distributed.ReduceOp.SUM, group=_GLOO_DP_REP_PG)
        except Exception as _e:
            log.error("Rank %d: grad_sync param[%d] all_reduce FAILED: %s | "
                      "numel=%d nbytes=%d dtype=%s canonical=%d eff_numel=%s param_shape=%s",
                      my_rank, i, _e, _g_flat.numel(), _g_flat.nbytes, _g_flat.dtype,
                      numel, eff_grads[i].numel() if eff_grads[i] is not None else "None",
                      list(param.shape))
            raise

        if _g_cpu is not None:
            _g_flat.div_(dp_rep_degree)
            _g = param.grad
            if hasattr(_g, '_local_tensor'):
                _g = _g._local_tensor
            _g.copy_(_g_flat[:_g.numel()].view(_g.shape).to(_g.device))
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

        _g_local = _g._local_tensor if hasattr(_g, '_local_tensor') else _g

        # XCCL (oneCCL) does not support ReduceOp.AVG; use SUM + manual div.
        try:
            _orig_all_reduce(_g_local, op=torch.distributed.ReduceOp.SUM,
                             group=_XCCL_DP_REP_PG)
        except Exception as _e:
            log.error("Rank %d: XCCL grad_sync all_reduce FAILED: %s | "
                      "param_shape=%s grad_shape=%s dtype=%s",
                      my_rank, _e, list(param.shape), list(_g_local.shape), _g_local.dtype)
            raise

        _g_local.div_(dp_rep_degree)
        n_synced += 1

    return n_synced


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

def _slice_trajectory(trajectory: GRPOTrajectory, start: int, end: int) -> GRPOTrajectory:
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
        log.info("Patched FSDP2 _get_gradient_divide_factors for XPU (force SUM reduction)")
    except Exception as e:
        log.warning("Failed to patch FSDP2 for XPU: %s", e)

    # Patch 2: reduce_scatter_tensor → AllReduce-based fallback.
    # v59: With reduce_grads=False on all FSDPParamGroups, FSDP2 never calls
    # reduce_scatter_tensor during backward. This patch is a safety net only.
    _tdist_patch.reduce_scatter_tensor = _xpu_reduce_scatter_via_allreduce
    log.info("Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)")
    log.info("dist.all_reduce NOT patched (v59: reduce_grads=False on all FSDP2 groups)")

    # Patch 3: all_to_all_single → XCCL direct (v80: gloo path removed).
    _tdist_patch.all_to_all_single = _xpu_all_to_all_via_gloo
    log.info("Patched dist.all_to_all_single → XCCL all_to_all_single for XPU EP tensors (v80)")


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
) -> None:
    """Register process group handles used by distributed op patches.

    Called from GRPOFullFinetuneDistributedXPU._init_distributed() after all gloo
    and XCCL process groups have been created. The registered handles are used by
    _xpu_reduce_scatter_via_allreduce, _ep_post_backward_grad_sync[_xccl], etc.
    """
    global _GLOO_DP_REP_PG, _GLOO_DP_SHARD_PG, _GLOO_GLOBAL_PG, _XCCL_DP_REP_PG
    global _DP_REP_DEGREE, _DP_SHARD_DEGREE
    _GLOO_DP_REP_PG = gloo_dp_rep_pg
    _GLOO_DP_SHARD_PG = gloo_dp_shard_pg
    _GLOO_GLOBAL_PG = gloo_global_pg
    _XCCL_DP_REP_PG = xccl_dp_rep_pg
    _DP_REP_DEGREE = dp_rep_degree
    _DP_SHARD_DEGREE = dp_shard_degree

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# XPU-adapted variant of grpo_full_finetune_distributed.py with:
# - Device-agnostic memory ops (XPU / CUDA / CPU)
# - Gradient accumulation with no_sync() for multi-node efficiency
# - Production mode sync skipping (FSDP_PRODUCTION_MODE env var)

import io
import os
import struct
import sys
import threading
import time
from functools import partial
from typing import Any, Optional, Union
from warnings import warn

# -- XPU / XCCL compatibility shim ---------------------------------------------
# On Intel XPU (Aurora), running torchtune's ``__init__.py`` while an XCCL
# process group is active corrupts the L0 USM pointer table, causing every
# subsequent collective to fail.  The root cause is an interaction between
# ``torchtune.__init__``'s ``import torchao`` / env-var setup and the XCCL
# backend's device-context bookkeeping.
#
# Workaround:  pre-register the ``torchtune`` package in ``sys.modules``
# (as a plain ``types.ModuleType``) *before* importing any submodules.
# This prevents Python from executing ``torchtune/__init__.py`` while still
# allowing all ``from torchtune.xxx import ...`` statements to work normally.
#
# Each rank uses ``device_id=xpu:{LOCAL_RANK + offset}`` instead of
# ``ZE_AFFINITY_MASK`` — CCL needs to see all device UUIDs for allreduce.
# --------------------------------------------------------------------------

# 1. Tile affinity — use device_id=xpu:{LOCAL_RANK} (no offset).
#
#    IMPORTANT: Do NOT set ZE_AFFINITY_MASK for multi-rank training.
#    CCL's allreduce needs to see all device UUIDs on the node to build the
#    topology. When ZE_AFFINITY_MASK restricts each rank to 1 tile, CCL only
#    discovers 1 UUID and falls back to the "scheduler path" which lacks
#    ReduceOp.AVG support, causing errors for 3+ ranks.
#
#    Instead, leave ZE_AFFINITY_MASK unset and use device_id=xpu:{LOCAL_RANK}.
#    CCL requires device_id index == LOCAL_RANK for correct topology routing.
#    vLLM is launched separately with ZE_AFFINITY_MASK on higher-numbered tiles
#    (e.g., tile 11) to avoid collisions.
#
#    Valid rank counts: must evenly divide CCL's topology view (works: 2, 4, 6,
#    10, 12; fails: 3, 5, 7, 8, 9, 11). CCL maps ranks across 6 cards × 2 tiles.
#
#    Multi-node: ZE_AFFINITY_MASK must be set (e.g., to LOCAL_RANK) so each rank
#    sees only its tile as xpu:0. With ring algorithms, device_id is not needed.
_use_affinity_mask = "ZE_AFFINITY_MASK" in os.environ and os.environ["ZE_AFFINITY_MASK"] != ""
# With single-tile affinity (e.g. "3"), rank sees 1 tile as xpu:0.
# With multi-tile affinity (e.g. "0,1,...,9"), rank sees N tiles and must select by LOCAL_RANK.
# Without affinity, all 12 tiles visible — also select by LOCAL_RANK.
_affinity_tiles = os.environ.get("ZE_AFFINITY_MASK", "").split(",") if _use_affinity_mask else []
_xpu_device_index = 0 if (len(_affinity_tiles) == 1) else int(os.environ.get("LOCAL_RANK", "0"))

import torch

# 2. Pre-register torchtune package to bypass its __init__.py on XPU
import types as _types
import importlib.util as _imp_util

if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
    else:
        # Fallback: assume editable install layout
        _torchtune_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "torchtune",
        )
    if os.path.isdir(_torchtune_path):
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

# 3. Ensure torchao is available (torchtune.__init__ normally checks this)
import torchao  # noqa

from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.datasets import ConcatDataset
from torchtune.dev.rl.generation import generate
from torchtune.dev.rl.rewards import batched_rewards, gene_recall_batched_rewards
from torchtune.dev.rl.types import GRPOStats, GRPOTrajectory
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    device_empty_cache,
    device_record_memory_history,
    disable_dropout,
    DummyProfiler,
    get_xpu_distributed_backend,
    init_xpu_process_group,
    PROFILER_KEY,
    supports_memory_stats,
)
from torchtune.training.lr_schedulers import get_lr
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from tqdm import tqdm

# Activation checkpointing wrapper.
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
#   No prior _fwd_step_counter / _is_reuse cache exists in source — that comment
#   referred to a design that was never landed.
# v154: revert to non-reentrant AC. v114 forced use_reentrant=True to work around
# an AllToAll backward SIGSEGV; AllToAll has been gone since v141 (replaced with
# AllGather+ReduceScatter in _parallelism.py), so the original reason is stale.
# Hypothesis: reentrant AC interleaves FWD-recompute with BWD via the Python
# autograd boundary, and on the rank with the smallest routed batch (consistently
# local-index-1 in each EP group after the v110 interleaved routing fix) the
# submodule eval order diverges, producing the deterministic op #259 RS-BWD desync.
def _no_reentrant_ac_wrapper(module):
    return ptd_checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.REENTRANT,
        checkpoint_fn=_torch_checkpoint,
        use_reentrant=False,  # v154: revert from True (v114 reason — AllToAll BWD SIGSEGV — gone since v141)
        preserve_rng_state=False,
        determinism_check="none",
    )


def _apply_split_ac(model):
    """v158: Split AC wrapping so MoE-bearing Gemma4 layers run their MoE block
    OUTSIDE the checkpointed region.

    v154 (use_reentrant=False) eliminated the v153 op-#259 RS-BWD desync, but
    exposed a paired ScatterAddBackward0 ±1 mismatch on EP ranks. Mechanism:
    `Gemma4MoeRouter` runs `sigmoid(float32) + argsort(stable)`. Bitwise
    differences between FWD and AC recompute can flip ties → bincount differs
    by ±1 → `_ag_gather_idx` regenerated in `_token_dispatch` is off-by-one
    relative to the saved `routed_output` → scatter_add row-count mismatch.

    Fix: the Gemma4 MoE-bearing layer subclass exposes `_ac_enabled` and a
    `_attn_and_dense` helper. Setting `_ac_enabled = True` makes the layer
    checkpoint attention+dense_MLP only — MoE runs once, never recomputed.

    For all other layers (dense Gemma4, non-Gemma4 architectures), keep the
    existing apply_activation_checkpointing on TransformerSelfAttentionLayer.
    """
    from torchtune.models.gemma4._component_builders import Gemma4TransformerLayer

    moe_layer_ids = set()
    for m in model.modules():
        if isinstance(m, Gemma4TransformerLayer) and m.moe_block is not None:
            m._ac_enabled = True
            moe_layer_ids.add(id(m))

    def _check_fn(submodule):
        # Wrap TransformerSelfAttentionLayer instances EXCEPT MoE-bearing Gemma4
        # layers (they self-checkpoint inside forward via _ac_enabled).
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

log = utils.get_logger("DEBUG")

# -- FSDP2 XPU fix: ReduceOp.AVG not supported by XCCL -----------------------
# FSDP2's reduce_scatter uses ReduceOp.AVG which XCCL doesn't support.
# Monkey-patch _get_gradient_divide_factors to force SUM reduction on XPU
# (same approach as MTIA in upstream PyTorch).
try:
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll
    _orig_get_gradient_divide_factors = _fsdp_coll._get_gradient_divide_factors

    def _xpu_get_gradient_divide_factors(*args, **kwargs):
        # args: (reduce_scatter_group, all_reduce_group, reduce_dtype,
        #        device_type, gradient_divide_factor, force_sum_reduction_for_comms)
        # Force SUM reduction for XPU (XCCL doesn't support AVG)
        if len(args) >= 4 and args[3] == "xpu":
            args = list(args)
            if len(args) >= 6:
                args[5] = True  # force_sum_reduction_for_comms
            else:
                kwargs["force_sum_reduction_for_comms"] = True
            args = tuple(args)
        elif kwargs.get("device_type") == "xpu":
            kwargs["force_sum_reduction_for_comms"] = True
        return _orig_get_gradient_divide_factors(*args, **kwargs)

    _fsdp_coll._get_gradient_divide_factors = _xpu_get_gradient_divide_factors
    log.info("Patched FSDP2 _get_gradient_divide_factors for XPU (force SUM reduction)")
except Exception as e:
    log.warning("Failed to patch FSDP2 for XPU: %s", e)

# -- FSDP2 XPU fix: ze_handle_manager "unknown memory type" in reduce_scatter --
# CCL's reduce_scatter_tensor uses Level Zero IPC handles to share GPU memory
# between ranks. On Aurora, PyTorch's XPU caching allocator allocates gradient
# buffers that CCL's ze_handle_manager cannot identify ("unknown memory type"),
# causing ze_handle_manager.cpp:226 get_ptr: EXCEPTION during FSDP2 backward.
#
# AllReduce (ring) works fine — it uses OFI fabric without IPC handles.
# Reduce-scatter = AllReduce (sum all ranks) + local scatter (take our chunk).
#
# This patch replaces dist.reduce_scatter_tensor with an AllReduce-based
# equivalent. Applied globally so it covers both non-expert FSDP2 (dp_replicate)
# and any other reduce_scatter calls. Expert FSDP2 (1-rank solo) already has
# reduce_grads=False so their reduce_scatter is skipped regardless.
#
# v54-v58: CPU AllReduce via gloo. XCCL groups only support XPU tensors.
# We create gloo process groups (set in _init_distributed) mirroring both
# dp_shard_pg and dp_replicate_pg, then use them to AllReduce CPU tensors.
# Host memory OFI needs no XPU memory registration → no ze_handle_manager,
# no OFI GPU-direct EPERM, for all grad-sync collectives.
#
# FSDP2 HSDP grad sync has TWO steps (both must be patched):
#   1. reduce_scatter_tensor(grad_full, group=dp_shard_pg) → sums grad within 4 shard ranks
#   2. all_reduce(grad_shard, group=dp_replicate_pg)       → averages across 3 replica ranks
# Step 1 was patched in v44; step 2 was first patched in v57.
# Without patching step 2, the dp_replicate all_reduce fires on fresh XPU grad_shard
# tensors via XCCL ring → ze_handle_manager crash at first param in backward.
#
# v56 bug: used _GLOO_DP_REP_PG (dp_replicate, 3 ranks) for reduce_scatter which
# is called with group=dp_shard_pg (4 ranks) — mismatched group dimensions.
# v57 fix: _GLOO_DP_SHARD_PG for reduce_scatter, _GLOO_DP_REP_PG for all_reduce.
# v57 failure: gloo preamble size mismatch — different EP groups ({0,1,2,3} and
#   {4,5,6,7}) complete their dp_shard gloo AllReduces at slightly different times
#   due to EP load imbalance. When they reach the dp_replicate gloo AllReduce,
#   they're on different parameters → "op.preamble.length <= op.nbytes" SIGABRT.
# v58 fix: add a global gloo barrier (all 12 ranks) just before the dp_replicate
#   gloo AllReduce in _xpu_all_reduce_via_gloo. The barrier forces all EP groups to
#   synchronize after their dp_shard AllReduces complete, ensuring all dp_replicate
#   ranks are at the same parameter with the same tensor size before the AllReduce.
#   Cost: 1 global gloo barrier per backward (only 1 all_reduce per FSDP group).
# v58 failure: gloo preamble size mismatch STILL — the global barrier fires inside
#   each AllReduce call, but different EP groups fire AllReduces for DIFFERENT params
#   (backward execution order differs by EP load). Even with the barrier, rank 0
#   (from EP group 0) may be AllReducing param N while rank 4 (from EP group 1) is
#   at param N+1 → barrier can't help because they're in different AllReduce calls.
# v59 fix: suppress reduce_grads on ALL FSDPParamGroups (not just expert).
#   With reduce_grads=False on all groups, FSDP2 never calls reduce_scatter or
#   all_reduce during backward for grad sync. Grads accumulate locally (param.grad).
#   After backward completes, a single sequential post-backward pass iterates ALL
#   params with gradients (D2H → gloo AllReduce → H2D), sequentially and synchronously.
#   Sequential iteration ensures all 3 dp_rep ranks hit AllReduces in the same
#   parameter order → no preamble mismatch. No global barrier needed.
#   Cost: ~same total bytes transferred, but serialized (no backward overlap).
_GLOO_DP_REP_PG = None    # gloo mirror of dp_replicate_pg (3 ranks); set in _init_distributed
_GLOO_DP_SHARD_PG = None  # gloo mirror of dp_shard_pg (4 ranks); set in _init_distributed
# v105: No gloo groups for AllToAll — XCCL used directly (keeps EP ranks in lockstep).
# _XPU_A2A_FWD_GLOO_GROUP and _XPU_A2A_BWD_GLOO_GROUP remain None in _parallelism.
_GLOO_GLOBAL_PG = None    # gloo global group (all ranks); barrier before post-backward AllReduce
_XCCL_DP_REP_PG = None   # XCCL dp_replicate group (3 ranks, XPU fabric); set in _init_distributed
_DP_REP_DEGREE = 1        # dp_replicate world size; set in _init_distributed
_DP_SHARD_DEGREE = 1      # dp_shard world size; set in _init_distributed

import torch.distributed as _tdist_patch
_orig_reduce_scatter_tensor = _tdist_patch.reduce_scatter_tensor
_orig_all_reduce = _tdist_patch.all_reduce
_a2a_call_counter = 0  # v70: counts all_to_all_single calls for diagnostic tagging

class _DoneWork:
    """Fake Work object for synchronous ops masquerading as async."""
    def wait(self): pass
    def is_completed(self): return True
    def get_future(self):
        import torch.futures as _tf
        f = _tf.Future()
        f.set_result(None)
        return f

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
        # reduce_scatter_tensor is called with group=dp_shard_pg (size=_DP_SHARD_DEGREE).
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
            # Fallback: XCCL group (only safe for dp_replicate=1/dp_shard=world_size,
            # i.e., non-EP mode without HSDP).
            input_sum = input.clone()
            _orig_all_reduce(input_sum, op=op, group=group)
    else:
        input_sum = input.clone()
        _orig_all_reduce(input_sum, op=op, group=group)
    # Scatter: each rank takes chunk r of the summed tensor.
    # v137: first-dimension slicing (supports multi-dimensional tensors from EP ReduceScatter).
    chunk_rows = output.shape[0]
    output.copy_(input_sum[r * chunk_rows : (r + 1) * chunk_rows])
    if async_op:
        return _DoneWork()

_tdist_patch.reduce_scatter_tensor = _xpu_reduce_scatter_via_allreduce
log.info("Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57: dp_shard gloo)")

# v59: dist.all_reduce is NOT patched.
# With reduce_grads=False on all FSDPParamGroups, FSDP2 never calls dist.all_reduce
# for grad sync during backward. Grad sync is done post-backward via _ep_post_backward_grad_sync().
# dist.all_reduce is used only for: loss/metric CPU tensors (safe with XCCL ring),
# grad norm (XPU scalar, handled via gloo explicitly in train()), optimizer barrier.
log.info("dist.all_reduce NOT patched (v59: reduce_grads=False on all FSDP2 groups)")

# v61: Patch dist.all_to_all_single for EP backward gradients.
# v60 failure: gloo all_to_all_single with unequal splits → preamble size mismatch
#   (gloo::EnforceNotMet pair.cc:456: op.preamble.length <= op.nbytes).
#   Root cause: gloo's all_to_all_single sends total tensor size as preamble, but
#   with unequal splits the receiver allocates based on its output_split_sizes[sender],
#   which may not match the sender's input tensor total size.
# v61 fix: implement AllToAll via gloo P2P (isend/irecv) instead of all_to_all_single.
#   Each P2P message is sized exactly by its split size → no preamble ambiguity.
#   Uses dist.batch_isend_irecv for efficiency (all sends and recvs issued in parallel).
_orig_all_to_all_single = _tdist_patch.all_to_all_single


def _gloo_all_to_all_via_allreduce(output_cpu, input_cpu, output_split_sizes, input_split_sizes, group,
                                   _call_tag="fwd"):
    """AllToAll via all_reduce split-matrix + sequential broadcasts (v65).

    v64 bug: all_gather of split sizes caused deadlock when gloo queue order diverged
    across ranks (different ranks at different AllToAll calls in the backward).
    Specifically: n_src==0 caused `continue` (skipping broadcast) on some ranks but not
    others → broadcast has too few participants → permanent deadlock.
    Root cause: gloo all_gather is order-sensitive; backward AllToAll calls from different
    MoE layers can reach the all_gather in different orders on different ranks.

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

    # Step 1: share all ranks' input_split_sizes via all_reduce(SUM) on (ws×ws) matrix.
    # Each rank contributes row my_rank; SUM gathers all rows → consistent on all ranks.
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
    # splits_matrix[src][j] = src's input_split_sizes[j] = rows src sends to rank j

    feat_shape = input_cpu.shape[1:]

    # Build output offset table using local output_split_sizes
    out_off = [0]
    for s in output_split_sizes[:-1]:
        out_off.append(out_off[-1] + s)

    # Step 2+3+4: for each src, broadcast in a fixed loop (all ranks participate every time)
    for src in range(ws):
        n_src = int(splits_matrix[src].sum().item())  # total rows src sends (consistent)
        n_rows = output_split_sizes[src]              # rows I expect from src (local)
        global_src = _d.get_global_rank(group, src)

        if n_src == 0:
            # all_reduce ensures n_src is consistent across ranks → all skip together.
            # (In v64 with all_gather, n_src could differ → asymmetric skip → deadlock.)
            continue

        # All ranks allocate exactly n_src rows (sender has actual data, receivers have zeros)
        if src == my_rank:
            data = input_cpu.contiguous()
            # Validate: sender's data shape must match (n_src,) + feat_shape
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
            # Broadcast src's data to all ranks (equal-size buffers → no preamble mismatch)
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

        # Extract my slice: src puts rows for rank j at offset sum(input_split_sizes[:j])
        src_offset = int(splits_matrix[src][:my_rank].sum().item())
        n_rows = min(n_rows, max(0, n_src - src_offset))  # cap silently (wrong grads if triggered)

        if n_rows > 0:
            output_cpu[out_off[src]:out_off[src] + n_rows].copy_(
                data[src_offset:src_offset + n_rows]
            )


def _xpu_all_to_all_via_gloo(output, input, output_split_sizes=None,
                               input_split_sizes=None, group=None, async_op=False):
    """Route EP all_to_all_single via XCCL directly (v80).

    v65-v79: gloo TCP-based AllToAll (allreduce+broadcast on _GLOO_DP_SHARD_PG) caused
    persistent 1800s timeout deadlocks at a2a#241 (first AllToAll of the backward pass).
    Root cause: ep_ranks 2,3 (0 tokens due to routing imbalance) intermittently fail to
    participate in gloo TCP collectives → ep_ranks 0,1 wait 1800s → gloo timeout.
    All v65-v79 variants (gloo barriers, XCCL step-boundary syncs, pre-ref syncs) failed
    because the gloo TCP path itself is the source of the deadlock.

    v80 fix: remove gloo path entirely. Always use XCCL all_to_all_single directly.
    _XPUSyncAllToAll already adds dist.barrier(group) for OFI CQ drain (added at v47).
    XCCL was confirmed working end-to-end at v47. No TCP framing issues with XCCL.
    """
    global _a2a_call_counter
    import torch.distributed as _d
    if input.device.type == 'xpu' and group is not None:
        n = _d.get_world_size(group)
        if n == _DP_SHARD_DEGREE:
            _a2a_call_counter += 1  # keep counter for diagnostics
    return _orig_all_to_all_single(
        output, input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=async_op,
    )

_tdist_patch.all_to_all_single = _xpu_all_to_all_via_gloo
log.info("Patched dist.all_to_all_single → XCCL all_to_all_single for XPU EP tensors (v80)")


def _ep_post_backward_grad_sync(model: "nn.Module", dp_rep_degree: int) -> int:
    """Post-backward gradient sync for EP training (v68).

    With reduce_grads=False on all FSDPParamGroups, FSDP2 skips reduce_scatter during
    backward. This function manually syncs ALL param gradients across dp_replicate after
    backward completes.

    v65 bug: `if param.grad is None: continue` caused asymmetric all_reduce participation.
      Expert params on None-grad ranks (zero tokens routed) were skipped → different
      dp_replicate ranks called all_reduce for different params → tensor size mismatch
      → gloo::EnforceNotMet (SIGABRT).

    v66/v67 bug: guessing the zero tensor shape from param.shape or param._local_tensor.shape
      failed because the effective grad shape depends on FSDP2 internals:
      - ZeRO-2 (reshard_after_forward=False): grad is the FULL gathered tensor (= param.shape)
        but param._local_tensor is the shard (1/Nth size). Opposite mismatch to v66.
      - ZeRO-3 (reshard_after_forward=True): grad is a sharded DTensor with _local_tensor.
        param._local_tensor.shape matches, but then padding differences cause off-by-few bytes.
      Both v66 "16 vs 4" (4:1 ratio = dp_shard=4) and v67 "1044874 vs 1044850" (24-byte
      padding difference) confirmed this: shape cannot be reliably inferred from the param.

    v68 fix: Two-phase approach:
      Phase 1: For each param, compute effective_numel (numel of the grad tensor that will be
               all_reduced — after _local_tensor extraction if applicable). None-grad ranks
               contribute 0. One all_reduce(MAX) over all numels shares the canonical numel
               from the non-None ranks to all ranks in the dp_replicate group.
      Phase 2: All_reduce each param's grad using the canonical numel. None-grad ranks create
               zeros of the canonical numel (guaranteed to match non-None ranks). All tensors
               are flattened to 1D to avoid any shape interpretation issues.

    This avoids all shape guessing — the canonical numel comes directly from the rank that
    actually holds the grad. One extra all_reduce per backward (cheap: int64 per param).

    Returns number of gradients synced (params with non-None grad on this rank).
    """
    if _GLOO_DP_REP_PG is None:
        return 0

    my_rank = torch.distributed.get_rank()

    # Phase 1: collect effective grad tensors (already on CPU, flattened 1D).
    param_list = list(model.parameters())
    eff_grads = []   # CPU 1D tensor or None
    eff_numels = []  # numel of eff_grad (0 if None)
    eff_dtypes = []  # dtype of param (for zero tensor creation)

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

    # One all_reduce(MAX) to share canonical numels.
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

    # Log any discrepancy: non-None rank whose eff_numel != canonical (cross-replica mismatch)
    for i in range(len(param_list)):
        if eff_grads[i] is not None and eff_grads[i].numel() != int(canonical_numels[i]):
            log.error("Rank %d: grad_sync param[%d] NUMEL MISMATCH: "
                      "eff_numel=%d canonical=%d dtype=%s param_shape=%s grad_has_local=%s",
                      my_rank, i, eff_grads[i].numel(), int(canonical_numels[i]),
                      eff_dtypes[i], list(param_list[i].shape),
                      hasattr(param_list[i].grad, '_local_tensor'))

    # Phase 2: all_reduce each param's grad using canonical numel.
    n_synced = 0
    for i, param in enumerate(param_list):
        numel = int(canonical_numels[i])
        if numel == 0:
            continue  # No rank in dp_replicate group has grad → skip entirely

        _g_cpu = eff_grads[i]
        if _g_cpu is not None:
            _g_flat = _g_cpu  # already 1D
            if _g_flat.numel() != numel:
                # Numel mismatch between this rank and canonical — pad/truncate with warning.
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
            # Write back to device grad (reshape from 1D back to original shape).
            _g = param.grad
            if hasattr(_g, '_local_tensor'):
                _g = _g._local_tensor
            _g.copy_(_g_flat[:_g.numel()].view(_g.shape).to(_g.device))
            n_synced += 1

    return n_synced


def _ep_post_backward_grad_sync_xccl(model: "nn.Module", dp_rep_degree: int) -> int:
    """Post-backward gradient sync using XCCL dp_replicate group (v75).

    v75 replaces the gloo-based _ep_post_backward_grad_sync with direct XCCL all_reduce
    on XPU tensors via _orig_all_reduce (bypasses our monkey-patch).

    Root cause of v71-v74 failures:
      All attempted sync mechanisms (gloo global barrier, XCCL dist.barrier(), XCCL
      _orig_all_reduce on all-12 ranks) deadlock because they require ALL 12 ranks to
      participate simultaneously. When fast EP groups finish backward first and try to
      sync, slow EP group ranks are still inside gloo SHARD AllToAll and cannot join the
      12-rank sync. The 12-rank XCCL all_reduce (v74) then waits indefinitely for those
      ranks, while gloo waits indefinitely for ranks that are stuck in the XCCL all_reduce
      → mutual deadlock → 1800s timeout.

    v75 key insight: dp_replicate TCP/XCCL pairs are DISJOINT from dp_shard pairs.
      SHARD group {0,1,2,3}: pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
      REP   group {0,4,8}:   pairs (0,4),(0,8),(4,8)  ← no overlap with SHARD
      XCCL uses the XPU Slingshot fabric. With disjoint rank pairs, the XCCL REP
      all_reduce CAN run concurrently with gloo SHARD AllToAll without any interference.
      No pre-sync barrier is needed — XCCL naturally waits for the 3 dp_rep ranks.

    Implementation:
      Iterate model.parameters() in order (same order on all dp_rep ranks). For each
      param with a gradient, call _orig_all_reduce(grad, SUM, group=_XCCL_DP_REP_PG)
      then divide by dp_rep_degree — no CPU copies, no numel exchange, no gloo.
      (XCCL/oneCCL does not support ReduceOp.AVG; SUM + manual div is equivalent.)
      Params with None grad are skipped (no asymmetry risk since all dp_rep ranks for
      a given param either all have grad or all don't — same expert assignments).

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

        # Extract local shard if DTensor (ZeRO-3 style).
        _g_local = _g._local_tensor if hasattr(_g, '_local_tensor') else _g

        # all_reduce(SUM) on XPU tensor via XCCL — bypasses gloo monkey-patch.
        # Note: XCCL (oneCCL) does not support ReduceOp.AVG; use SUM + manual div.
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


# Original device_empty_cache imported above. When colocated vLLM engines
# are present, torch.xpu.empty_cache() can deadlock if called while another
# rank is inside an FSDP collective. We replace it with synchronize() which
# forces completion of pending XPU ops without risking the cache-clearing
# deadlock, and still frees some memory via the allocator.
_orig_device_empty_cache = device_empty_cache
_colocate_vllm_mode = False

def device_empty_cache(device: torch.device) -> None:
    if device.type == "xpu":
        # NEVER call empty_cache() on XPU with FSDP. The combination of
        # empty_cache() + FSDP storage.resize_() leaks UR handles in Level
        # Zero, causing UR_RESULT_ERROR_OUT_OF_RESOURCES after ~70 iters.
        # The caching allocator reuses blocks without touching L0 if we
        # skip empty_cache(). See docs/bugs/intel_xpu_resource_leak_bug_report.md
        pass
    else:
        _orig_device_empty_cache(device)


def _safe_empty_cache(device: torch.device) -> None:
    """Empty cache at a synchronized point (after a barrier).

    On XPU, this is a no-op — empty_cache() + FSDP leaks UR handles.
    """
    torch.distributed.barrier()
    if device.type == "xpu":
        torch.xpu.synchronize()
        return
    _orig_device_empty_cache(device)


def _slice_trajectory(
    trajectory: GRPOTrajectory, start: int, end: int
) -> GRPOTrajectory:
    """Slice a GRPOTrajectory along the batch dimension.

    Args:
        trajectory: The full trajectory.
        start: Start index (inclusive).
        end: End index (exclusive).

    Returns:
        A new GRPOTrajectory with only the [start:end] samples.
    """
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



class GRPOFullFinetuneDistributedXPU(FTRecipeInterface):
    """
    Distributed GRPO full-finetune recipe adapted for Intel XPU (Aurora HPC).

    Key differences from the CUDA-only version:
    - Device-agnostic memory management (empty_cache, memory stats)
    - Gradient accumulation with ``no_sync()`` to reduce AllReduce overhead
    - XPU-safe distributed init (no ``device_id`` for XCCL)
    - Production mode sync skipping via ``FSDP_PRODUCTION_MODE`` env var
    """

    def __init__(self, cfg: DictConfig) -> None:
        _usm_so = os.environ.get("XPU_USM_ALLOC_SO")
        if _usm_so:
            from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
            _usm_alloc = XPUPluggableAllocator(_usm_so, "xpu_usm_malloc", "xpu_usm_free")
            change_current_allocator(_usm_alloc)
            log.info(f"USM arena allocator registered: {_usm_so}")
            # Pluggable allocator doesn't support getDeviceStats/emptyCache.
            # Monkeypatch memory query functions to return 0 instead of crashing.
            torch.xpu.memory_allocated = lambda device=None: 0
            torch.xpu.memory_reserved = lambda device=None: 0
            torch.xpu.max_memory_allocated = lambda device=None: 0
            torch.xpu.max_memory_reserved = lambda device=None: 0
            torch.xpu.reset_peak_memory_stats = lambda device=None: None
            torch.xpu.empty_cache = lambda: None
            torch.xpu.memory_stats = lambda device=None: {}

        # Use the correct XPU device index (LOCAL_RANK + tile offset)
        if cfg.device == "xpu":
            self._device = torch.device(f"xpu:{_xpu_device_index}")
            torch.xpu.set_device(_xpu_device_index)
        else:
            self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        self._output_dir = cfg.output_dir

        # Logging attributes
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and not supports_memory_stats(self._device):
            log.info(
                "log_peak_memory_stats was set to True, however, the device does "
                "not support memory stats. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Initialize the distributed environment
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        # ref_cpu_offload: keep ref model on CPU, policy on GPU (needed for inline gen).
        # Frees ~12 GiB XPU HBM by offloading ref model params to CPU; FSDP2 moves
        # them to GPU on demand during ref forward passes only.
        self._ref_cpu_offload = cfg.get("ref_cpu_offload", False)
        self._disable_prefetch = cfg.get("disable_prefetch", False)
        self._fsdp_diagnostics = cfg.get("fsdp_diagnostics", False)
        self._empty_cache_before_backward = cfg.get("empty_cache_before_backward", False)
        self.distributed_backend = get_xpu_distributed_backend(
            self._device.type, offload_ops_to_cpu=self.fsdp_cpu_offload
        )

        # Colocated vLLM must be initialized BEFORE the training PG because
        # vLLM creates gloo sub-groups that crash if the default PG was
        # initialized with device_id=xpu:0.
        vllm_mode = cfg.get("vllm_mode", None)
        if cfg.get("vllm_url", None) is not None and vllm_mode is None:
            vllm_mode = "server"
        if vllm_mode in ("colocate", "colocate_sleep"):
            self._init_vllm_early(cfg)

        # If using MPI transport (CCL_ATL_TRANSPORT=mpi), pre-init MPI so CCL can
        # use it. This follows the official Aurora DDP pattern.
        if os.environ.get("CCL_ATL_TRANSPORT") == "mpi":
            try:
                from mpi4py import MPI
                MPI.COMM_WORLD.Barrier()
            except ImportError:
                pass

        if not torch.distributed.is_initialized():
            init_xpu_process_group(self.distributed_backend, device_index=_xpu_device_index)
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        # SDPA backend selection for XPU.
        # Default: force math-only SDPA (disable flash/mem_efficient) as a
        # precaution against UR handle leaks observed with broken CCL config.
        # Set force_math_sdpa=False to re-enable optimized SDPA backends.
        if self._device.type == "xpu" and cfg.get("force_math_sdpa", True):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            log.info("Rank %d: forced math-only SDPA (set force_math_sdpa=False to test optimized backends)", self.rank)
        elif self._device.type == "xpu":
            log.info("Rank %d: using default SDPA backends (flash/mem_efficient enabled)", self.rank)

        # Production mode: skip non-essential barriers/synchronize() calls.
        # Multi-node XPU REQUIRES production mode because world-level barriers
        # conflict with FSDP's sub-PG operations on XCCL (both HSDP sub-PGs and
        # FSDP2's per-module sub-communicators trigger broadcast_scaleout failures).
        _is_multinode = self.world_size > int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self._production_mode = (
            os.environ.get("FSDP_PRODUCTION_MODE", "0") == "1"
            or cfg.get("data_parallel_replicate_dim", 1) > 1
            or (_is_multinode and self._device.type == "xpu")
        )

        # Training attributes
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._compile = cfg.get("compile", False)
        # dynamic=True uses symbolic shapes to avoid recompilation on
        # variable-length sequences (essential for RL workloads).
        # Default True on XPU (matches PRISM's proven approach).
        self._compile_dynamic = cfg.get(
            "compile_dynamic", True if self._device.type == "xpu" else False
        )

        # Compile + multi-node on XPU: historically deadlocked with oneCCL,
        # but may work with fixed CCL config (WORKER_COUNT=1, MPI transport).
        # Set allow_compile_multinode=True to test.
        if self._compile and self._device.type == "xpu":
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
            if self.world_size > local_world_size:
                if cfg.get("allow_compile_multinode", False):
                    log.warning(
                        "compile + multi-node XPU: EXPERIMENTAL (allow_compile_multinode=True)"
                    )
                else:
                    log.warning(
                        "torch.compile disabled for multi-node XPU. "
                        "Set allow_compile_multinode=True to test."
                    )
                    self._compile = False

        # HSDP: dp_replicate × dp_shard mesh for multi-node
        # dp_replicate=1 (default): pure FSDP across all ranks
        # dp_replicate=NUM_NODES: FSDP within node, DDP across nodes (HSDP)
        self._dp_replicate = cfg.get("data_parallel_replicate_dim", 1)
        if self._dp_replicate > 1:
            from torch.distributed.device_mesh import init_device_mesh
            self._dp_shard = self.world_size // self._dp_replicate
            log.info(
                "HSDP enabled: dp_replicate=%d × dp_shard=%d (world_size=%d)",
                self._dp_replicate, self._dp_shard, self.world_size,
            )
            # Create a simple 2D mesh directly (avoid ParallelDims.build_mesh
            # which creates many submeshes that can deadlock XCCL).
            self._dp_mesh = init_device_mesh(
                self._device.type,
                (self._dp_replicate, self._dp_shard),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
            # Get shard PG for generation early-stopping all_reduce.
            # The shard group = ranks within same node (FSDP shards of same model).
            self._shard_pg = self._dp_mesh.get_group("dp_shard")
            # v57: Create gloo process groups mirroring BOTH dp_replicate AND dp_shard.
            # FSDP2 HSDP grad sync has two CCL steps that both need to be intercepted:
            #   1. reduce_scatter_tensor(grad, group=dp_shard_pg) — our monkey-patch
            #      converts to gloo AllReduce across dp_shard, then scatters.
            #   2. all_reduce(grad_shard, group=dp_replicate_pg) — separate FSDP2 step
            #      to average across replicas; also patched to use gloo.
            # v56 bug: used _GLOO_DP_REP_PG (3 ranks) for step 1 which uses dp_shard (4 ranks).
            # v57 fix: _GLOO_DP_SHARD_PG for step 1, _GLOO_DP_REP_PG for step 2.
            # bound_device_id must be cleared (v56 fix) before any new_group(backend="gloo")
            # to prevent PyTorch from trying pg._get_backend(xpu_device) on the gloo group.
            global _GLOO_DP_REP_PG, _GLOO_DP_SHARD_PG, _GLOO_GLOBAL_PG, _XCCL_DP_REP_PG, _DP_REP_DEGREE, _DP_SHARD_DEGREE
            _n_dp_rep = self._dp_replicate   # 3
            _n_dp_shd = self._dp_shard       # 4 (= ep_degree)
            _DP_REP_DEGREE = _n_dp_rep
            _DP_SHARD_DEGREE = _n_dp_shd
            import torch.distributed.distributed_c10d as _dc10d
            _default_pg = _dc10d._get_default_group()
            _orig_bound_device_id = _default_pg.bound_device_id
            _default_pg.bound_device_id = None
            try:
                # dp_replicate gloo groups (3 ranks each, one per shard position):
                # e.g. shard_idx=0: [0,4,8], shard_idx=1: [1,5,9], etc.
                for _shard_idx in range(_n_dp_shd):
                    _gloo_ranks = [_shard_idx + _n_dp_shd * j for j in range(_n_dp_rep)]
                    _gloo_pg = torch.distributed.new_group(_gloo_ranks, backend="gloo")
                    if self.rank in _gloo_ranks:
                        _GLOO_DP_REP_PG = _gloo_pg
                # dp_shard gloo groups (4 ranks each, one per replicate group):
                # e.g. rep_idx=0: [0,1,2,3], rep_idx=1: [4,5,6,7], rep_idx=2: [8,9,10,11]
                for _rep_idx in range(_n_dp_rep):
                    _gloo_ranks = [_rep_idx * _n_dp_shd + j for j in range(_n_dp_shd)]
                    _gloo_pg = torch.distributed.new_group(_gloo_ranks, backend="gloo")
                    if self.rank in _gloo_ranks:
                        _GLOO_DP_SHARD_PG = _gloo_pg
                    # v152: EP dispatch needs its OWN gloo communicator — separate from
                    # _GLOO_DP_SHARD_PG. FSDP2 grad sync (monkey-patched) uses _GLOO_DP_SHARD_PG
                    # concurrently with EP RS-BWD. Sharing the same gloo communicator causes
                    # sequence number collision → TCP timeout at op #259 (same hang as XCCL).
                    # new_group() with same ranks but called again → independent communicator,
                    # own sequence counters, own TCP connections. No shared state.
                    # v153: 120s timeout (vs default 1800s) for faster failure diagnostics.
                    import datetime as _dt
                    _ep_gloo_pg = torch.distributed.new_group(
                        _gloo_ranks, backend="gloo",
                        timeout=_dt.timedelta(seconds=120),
                    )
                    if self.rank in _gloo_ranks:
                        try:
                            from torchtune.modules.moe import _parallelism as _ep_par
                            _ep_par._GLOO_EP_PG = _ep_gloo_pg
                        except Exception:
                            pass
                # v106: gloo CPU-bounce ONLY for no-grad (ref forward); XCCL for all else.
                # One gloo FWD group per DP replica (4 ranks = EP ranks for that replica).
                # Ref forward runs under torch.no_grad() → _XPUSyncAllToAll.forward checks
                # torch.is_grad_enabled()==False → uses gloo CPU-bounce → avoids
                # ze_handle_manager crash (cpu_offload freshly-allocated XPU buffers ≠ USM).
                # Training forward + AC recompute: is_grad_enabled()==True → XCCL (lockstep).
                # v116: Backward uses XCCL (reverted from v115 gloo BWD).
                #   v115 hypothesis was wrong: use_reentrant=True serializes WITHIN a rank
                #   but gloo P2P is NOT a collective — provides no cross-rank synchronization.
                #   Different EP ranks can be at different backward layers → gloo size mismatch.
                #   v114 crash (SIGSEGV call#27-30, consistent splits) was OOM, not FSDP2+XCCL
                #   conflict. XCCL backward is safe: it IS a collective (forces lockstep).
                #   v116 fix: keep XCCL backward + halve fbs (4→2) to reduce backward peak.
                #   _XPU_A2A_BWD_GLOO_GROUP remains None → backward uses XCCL path.
                # v127: Backward uses gloo CPU-bounce (same group as FWD).
                #   v116-v126 XCCL backward crashes with SIGSEGV at call#0 (v122-v126) or
                #   call#42 (v116-v119). All attempts to fix XCCL backward have failed:
                #   IPC handle caching (v124), naive AllGather (v125), requires_grad_(False)
                #   for ref (v126). The SIGSEGV is in XPU driver XCCL code, not Python.
                #   v127 fix: force gloo CPU-bounce for backward AllToAll too.
                #   Why v115 failed but v127 should work:
                #     v115 attempted separate BWD gloo group (v101 implementation). This
                #     deadlocked because AC recompute forward AllToAll and backward AllToAll
                #     were concurrent on different TCP socket pairs.
                #   v127 reuses the same FWD group for backward. This is safe because:
                #     (1) AC recompute is CACHED (returns cached_output, no communication).
                #         The FWD gloo group is completely idle during backward.
                #     (2) With use_reentrant=True, backward is sequential per rank — all
                #         12 ranks proceed in lockstep (same barrier discipline as forward).
                #     (3) The same P2P send/recv split logic in backward (ctx.output_splits →
                #         ctx.input_splits) is proven correct (v101 implementation).
                #   Setting _XPU_A2A_BWD_GLOO_GROUP = FWD group activates gloo CPU-bounce
                #   in _XPUSyncAllToAll.backward instead of XCCL all_to_all_single.
                # AllGather+ReduceScatter EP dispatch (v137+): no AllToAll gloo groups needed.
                # EP communication uses dist.all_gather_into_tensor and dist.reduce_scatter_tensor
                # (XCCL) which are natively optimized on Aurora and have correct autograd.
                # Global gloo group (all ranks): used for barrier before dp_replicate
                # AllReduce to synchronize EP groups (v58 fix for gloo size mismatch).
                _GLOO_GLOBAL_PG = torch.distributed.new_group(
                    list(range(self.world_size)), backend="gloo"
                )
                # v150: inject global gloo group into _parallelism for cross-EP serialization.
                # All 12 ranks form one gloo group; barrier before/after each XCCL EP
                # collective prevents concurrent XCCL ops on different ep_pg groups
                # from cross-contaminating each other's OFI CQ events.
                try:
                    from torchtune.modules.moe import _parallelism as _ep_par
                    _ep_par._GLOO_GLOBAL_PG = _GLOO_GLOBAL_PG
                except Exception:
                    pass
            finally:
                _default_pg.bound_device_id = _orig_bound_device_id
            # v75: Get the XCCL dp_replicate process group from the device mesh.
            _XCCL_DP_REP_PG = self._dp_mesh.get_group("dp_replicate")
            log.info(
                "Created gloo dp_replicate (%d-rank), dp_shard (%d-rank), global gloo. "
                "EP dispatch: AllGather+ReduceScatter (no AllToAll gloo groups). "
                "dp_rep_pg=%s dp_shard_pg=%s global_pg=%s xccl_rep_pg=%s",
                _n_dp_rep, _n_dp_shd,
                _GLOO_DP_REP_PG, _GLOO_DP_SHARD_PG,
                _GLOO_GLOBAL_PG, _XCCL_DP_REP_PG,
            )
            # Shard group leader = local rank 0 within each shard group (= each node).
            # For replicated vLLM, each shard leader talks to its local vLLM.
            self._shard_rank = torch.distributed.get_rank(self._shard_pg)
            self._is_shard_leader = (self._shard_rank == 0)
            # Global rank of this shard group's leader (for broadcast src).
            # torch.distributed.broadcast requires global rank as src, even with group.
            shard_ranks = torch.distributed.get_process_group_ranks(self._shard_pg)
            self._shard_leader_global_rank = shard_ranks[0]
            self._dp_degree = self.world_size
            self._dp_rank = self.rank
        else:
            self._dp_shard = self.world_size
            self._dp_degree = self.world_size
            self._dp_rank = self.rank
            self._dp_mesh = None
            self._shard_pg = None
            self._shard_rank = self.rank
            self._is_shard_leader = self._is_rank_zero

            # v206: non-HSDP single-replicate path also needs a gloo group for the
            # _xpu_reduce_scatter_via_allreduce patch to use as CPU-bounce backend.
            # Without this, the patch falls through to its XCCL fallback (line 306)
            # which calls all_reduce on freshly-allocated XPU grad buffers — exactly
            # the v44-v58 failure pattern. Step 0 sometimes works (warmup buffers),
            # but step 1 reliably hits UR:40 with vLLM colocated.
            # (Globals already declared at line 889 in HSDP branch — same function,
            # so Python's static analysis already treats them as global here.)
            _DP_SHARD_DEGREE = self.world_size
            _DP_REP_DEGREE = 1
            import torch.distributed.distributed_c10d as _dc10d
            _default_pg = _dc10d._get_default_group()
            _orig_bound_device_id = _default_pg.bound_device_id
            _default_pg.bound_device_id = None
            try:
                _all_ranks = list(range(self.world_size))
                _GLOO_DP_SHARD_PG = torch.distributed.new_group(
                    _all_ranks, backend="gloo",
                )
                _GLOO_GLOBAL_PG = _GLOO_DP_SHARD_PG
            finally:
                _default_pg.bound_device_id = _orig_bound_device_id
            log.info(
                "v206: non-HSDP gloo PG initialized (world=%d) for "
                "_xpu_reduce_scatter_via_allreduce CPU-bounce path",
                self.world_size,
            )

        # Expert Parallelism (reuses dp_shard process group — no new communicators)
        self._expert_parallel_degree = cfg.get("expert_parallel_degree", 1)
        self._expert_cpu_offload = cfg.get("expert_cpu_offload", False)
        if self._expert_parallel_degree > 1:
            assert self._dp_replicate > 1, (
                "expert_parallel_degree > 1 requires data_parallel_replicate_dim > 1"
            )
            assert self._expert_parallel_degree == self._dp_shard, (
                f"expert_parallel_degree={self._expert_parallel_degree} must equal "
                f"dp_shard={self._dp_shard} (EP reuses the dp_shard process group)"
            )
            log.info(
                "Expert Parallelism enabled: ep_degree=%d (reuses dp_shard group)",
                self._expert_parallel_degree,
            )

        # Gradient accumulation
        self._gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

        # vLLM generation (optional — None means use native generation)
        # mode: "server" = external vLLM HTTP server on separate tile(s)
        #        "colocate" = in-process vLLM engine per rank (TRL-style)
        #        "colocate_sleep" = colocate with sleep/wake memory management
        self._vllm_mode = cfg.get("vllm_mode", None)  # None, "server", "colocate", or "colocate_sleep"
        self._vllm_url = cfg.get("vllm_url", None)
        # Multi-URL support: comma-separated URLs for DP vLLM replicas
        if self._vllm_url and "," in self._vllm_url:
            self._vllm_urls = [u.strip() for u in self._vllm_url.split(",")]
        elif self._vllm_url:
            self._vllm_urls = [self._vllm_url]
        else:
            self._vllm_urls = []
        self._vllm_group_port = cfg.get("vllm_group_port", 51216)
        self._vllm_weight_sync = cfg.get("vllm_weight_sync", True)
        self._vllm_weight_sync_interval = cfg.get("vllm_weight_sync_interval", 1)
        # Weight sync method: "raw_bytes" (file-based, default) or "shm" (POSIX shared memory,
        # faster for large models — eliminates the file read step on the vLLM side).
        self._vllm_weight_sync_method = cfg.get("vllm_weight_sync_method", "raw_bytes")
        self._vllm_clients = []  # initialized in setup() if vllm_mode == "server"
        self._vllm_client = None  # backward compat: first client
        # _vllm_llm may already be set by _init_vllm_early() for colocate mode
        if not hasattr(self, "_vllm_llm"):
            self._vllm_llm = None
        self._vllm_gpu_memory_utilization = cfg.get("vllm_gpu_memory_utilization", 0.3)
        self._vllm_max_model_len = cfg.get("vllm_max_model_len", 2048)
        self._vllm_tp_size = cfg.get("vllm_tensor_parallel_size", 1)

        # Backward compat: if vllm_url is set but vllm_mode is not, infer "server"
        if self._vllm_url is not None and self._vllm_mode is None:
            self._vllm_mode = "server"

        # Eval config
        self._eval_every_n_steps = cfg.get("eval_every_n_steps", 0)
        self._eval_max_examples = cfg.get("eval_max_examples", 50)
        self._eval_grpo_samples = cfg.get("eval_grpo_samples", None)
        self._eval_enabled = False  # set in setup() if eval_dataset configured

        # Step-based checkpointing
        self._save_every_n_steps = cfg.get("save_every_n_steps", None)
        self._save_final_checkpoint = cfg.get("save_final_checkpoint", True)

        # Recipe state attributes
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = 0
        self._epochs_run = 0
        # torch.Generator does not support XPU — use CPU generator instead
        _rng_device = self._device if self._device.type == "cuda" else torch.device("cpu")
        self._rng = torch.Generator(_rng_device).manual_seed(self.seed)

    def _init_vllm_early(self, cfg):
        """Initialize colocated vLLM engine(s) before the training PG.

        Two modes based on ``vllm_tensor_parallel_size``:

        **TP=1** (default): Each rank creates its own isolated vLLM engine
        using a gloo PG (world_size=1). Sequential init via file barriers.

        **TP>1**: Ranks are grouped into DP groups of size tp_size. Each group
        creates one shared vLLM engine via ``external_launcher`` with an XCCL
        PG of size tp_size. All ranks in a group call generate() together.
        """
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = _xpu_device_index  # The actual XPU tile for this rank

        vllm_mode = cfg.get("vllm_mode", "colocate")
        tp_size = cfg.get("vllm_tensor_parallel_size", 1)

        # For colocate_sleep mode, apply XPU sleep patches BEFORE importing vLLM LLM
        if vllm_mode == "colocate_sleep":
            from torchtune.dev.xpu_sleep import patch_vllm_for_xpu_sleep
            patch_vllm_for_xpu_sleep()

        from vllm import LLM

        model_path = cfg.base_model_path
        gpu_mem = cfg.get("vllm_gpu_memory_utilization", 0.3)
        max_model_len = cfg.get("vllm_max_model_len", 2048)

        log.info(
            "Rank %d: initializing colocated vLLM engine on xpu:%d (tp=%d, gpu_mem=%.2f, max_model_len=%d)",
            rank, local_rank, tp_size, gpu_mem, max_model_len,
        )

        # vLLM V1: disable multiprocessing to avoid EngineCore subprocess hangs.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Disable torch.compile for vLLM init (not viable on XPU for vLLM).
        _prev_compile_disable = os.environ.get("TORCH_COMPILE_DISABLE")
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

        # Each rank generates all grpo_samples for its own prompts (no split)
        max_num_seqs = max(cfg.batch_size * cfg.grpo_samples, 1)

        if tp_size > 1:
            self._init_vllm_tp(
                cfg, rank, world_size, local_rank, tp_size,
                model_path, gpu_mem, max_model_len, max_num_seqs, vllm_mode, LLM,
            )
        else:
            self._init_vllm_tp1(
                cfg, rank, world_size, local_rank,
                model_path, gpu_mem, max_model_len, max_num_seqs, vllm_mode, LLM,
            )

        # Restore TORCH_COMPILE_DISABLE so training model can use torch.compile
        if _prev_compile_disable is not None:
            os.environ["TORCH_COMPILE_DISABLE"] = _prev_compile_disable
        elif "TORCH_COMPILE_DISABLE" in os.environ:
            del os.environ["TORCH_COMPILE_DISABLE"]

        # Re-set the XPU device to our tile (vLLM may have called set_device)
        torch.xpu.set_device(local_rank)

        log.info("Rank %d: colocated vLLM engine initialized on xpu:%d, PG reset for training", rank, local_rank)

    def _init_vllm_tp1(self, cfg, rank, world_size, local_rank,
                        model_path, gpu_mem, max_model_len, max_num_seqs,
                        vllm_mode, LLM):
        """Initialize TP=1 colocated vLLM: one isolated engine per rank.

        Uses gloo PG (world_size=1) with sequential init via file barriers.
        """
        # Sequential init to avoid port/resource conflicts.
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
        barrier_dir = f"/tmp/torchtune/vllm_init_barriers_{run_id}"
        os.makedirs(barrier_dir, exist_ok=True)
        if rank > 0:
            prev_barrier = os.path.join(barrier_dir, f"rank_{rank - 1}_done")
            log.info("Rank %d: waiting for rank %d to finish vLLM init...", rank, rank - 1)
            while not os.path.exists(prev_barrier):
                time.sleep(0.5)

        # Override torchrun env vars so vLLM sees world_size=1
        saved_env = {}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                    "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID"):
            saved_env[key] = os.environ.pop(key, None)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29599 + rank)

        # Pre-init a gloo PG (world_size=1) so vLLM skips its own init.
        import tempfile
        _store_file = tempfile.mktemp(prefix=f"vllm_gloo_store_r{rank}_")
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{_store_file}",
            world_size=1,
            rank=0,
        )

        # Monkey-patch new_group to force gloo backend.
        _orig_new_group = torch.distributed.new_group
        def _gloo_new_group(*args, **kwargs):
            kwargs["backend"] = "gloo"
            return _orig_new_group(*args, **kwargs)
        torch.distributed.new_group = _gloo_new_group

        # Monkey-patch all_reduce to skip XPU tensor ops on gloo.
        _orig_all_reduce = torch.distributed.all_reduce
        def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM,
                             group=None, async_op=False):
            if group is not None and group.size() == 1:
                return None
            if tensor.is_xpu:
                return None
            return _orig_all_reduce(tensor, op=op, group=group,
                                    async_op=async_op)
        torch.distributed.all_reduce = _safe_all_reduce

        # Monkey-patch _distributed_args for correct local_rank.
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        _orig_distributed_args = UniProcExecutor._distributed_args
        _correct_local_rank = local_rank
        def _patched_distributed_args(self_exec):
            method, _rank, _lr = _orig_distributed_args(self_exec)
            return method, _rank, _correct_local_rank
        UniProcExecutor._distributed_args = _patched_distributed_args

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            dtype="bfloat16",
            disable_custom_all_reduce=True,
        )
        if vllm_mode == "colocate_sleep":
            llm_kwargs["enable_sleep_mode"] = True

        self._vllm_llm = LLM(**llm_kwargs)

        # Restore monkey-patches
        UniProcExecutor._distributed_args = _orig_distributed_args
        torch.distributed.new_group = _orig_new_group
        torch.distributed.all_reduce = _orig_all_reduce

        # Destroy vLLM's gloo PG so we can init the training XCCL PG.
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._WORLD = None

        try:
            os.unlink(_store_file)
        except OSError:
            pass

        # Restore torchrun env vars for the training PG
        for key, val in saved_env.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]

        # Signal next rank that we're done
        my_barrier = os.path.join(barrier_dir, f"rank_{rank}_done")
        with open(my_barrier, "w") as f:
            f.write("done")

        # Last rank cleans up barrier files
        if rank == world_size - 1:
            time.sleep(0.5)
            for i in range(world_size):
                try:
                    os.unlink(os.path.join(barrier_dir, f"rank_{i}_done"))
                except OSError:
                    pass

    def _init_vllm_tp(self, cfg, rank, world_size, local_rank, tp_size,
                       model_path, gpu_mem, max_model_len, max_num_seqs,
                       vllm_mode, LLM):
        """Initialize TP>1 colocated vLLM: one engine per DP group via external_launcher.

        Groups ranks into DP groups of tp_size. Each group creates an XCCL PG
        of size tp_size and a shared vLLM engine with external_launcher.
        All ranks in a group must call generate() with identical prompts.

        Pattern:
        1. File-based barrier (all ranks in DP group signal ready)
        2. Override env: WORLD_SIZE, RANK, LOCAL_RANK, MASTER_PORT; disable elastic agent store
        3. Create TP-sized XCCL PG per DP group (TCP rendezvous, unique port)
        4. Create vLLM LLM with external_launcher
        5. Destroy TP PG, restore env (training PG created later via elastic agent store)
        """
        assert world_size % tp_size == 0, (
            f"world_size={world_size} must be divisible by vllm_tensor_parallel_size={tp_size}"
        )
        dp_size = world_size // tp_size
        dp_rank = rank // tp_size
        tp_rank = rank % tp_size

        # Store TP/DP info for later use
        self._vllm_dp_rank = dp_rank
        self._vllm_tp_rank = tp_rank
        self._vllm_dp_size = dp_size

        log.info(
            "Rank %d: TP init — dp_rank=%d, tp_rank=%d, dp_size=%d, tp_size=%d",
            rank, dp_rank, tp_rank, dp_size, tp_size,
        )

        # Step 1: File-based barrier — all ranks signal ready before any creates PG.
        # torchrun sets TORCHELASTIC_USE_AGENT_STORE=True; consuming it for a
        # throwaway global PG would prevent reuse for the training PG later.
        _run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
        _barrier_dir = f"/tmp/torchtune/vllm_tp_barrier_{_run_id}"
        os.makedirs(_barrier_dir, exist_ok=True)
        # Signal this rank is ready
        with open(os.path.join(_barrier_dir, f"rank_{rank}"), "w") as f:
            f.write("ready")
        # Wait for all ranks in our DP group to be ready
        dp_group_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        for r in dp_group_ranks:
            while not os.path.exists(os.path.join(_barrier_dir, f"rank_{r}")):
                time.sleep(0.1)
        log.info("Rank %d: all %d ranks in DP group %d ready", rank, tp_size, dp_rank)

        # Step 2: Override env for TP subgroup
        saved_env = {}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_PORT",
                    "TORCHELASTIC_USE_AGENT_STORE"):
            saved_env[key] = os.environ.get(key)

        original_port = os.environ.get("MASTER_PORT", "29500")
        os.environ["WORLD_SIZE"] = str(tp_size)
        os.environ["RANK"] = str(tp_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        # Unique port per DP group to avoid collisions
        os.environ["MASTER_PORT"] = str(int(original_port) + dp_rank + 1)
        # Disable elastic agent store — use TCP rendezvous for TP subgroup
        os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)

        # Step 3: Create XCCL PG for this TP subgroup
        torch.distributed.init_process_group(
            backend="xccl",
            world_size=tp_size,
            rank=tp_rank,
        )
        log.info("Rank %d: TP subgroup PG initialized (dp_group=%d, size=%d)",
                 rank, dp_rank, torch.distributed.get_world_size())

        # Step 5: Create vLLM LLM engine
        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp_size,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            dtype="bfloat16",
        )
        if vllm_mode == "colocate_sleep":
            llm_kwargs["enable_sleep_mode"] = True

        self._vllm_llm = LLM(**llm_kwargs)

        # Step 6: Destroy TP PG — training will create its own global XCCL PG
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        import vllm.distributed.parallel_state as vllm_ps
        vllm_ps._WORLD = None

        # Restore env vars
        for key, val in saved_env.items():
            if val is not None:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]

        # Clean up barrier files (best-effort)
        if rank == 0:
            import shutil
            try:
                shutil.rmtree(_barrier_dir, ignore_errors=True)
            except OSError:
                pass

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        # Setup model to train
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        # Reshard parameters after forward to avoid keeping the full unsharded
        # model in memory. For 31B+ with FSDP-10/12, full unshard = 62+ GiB
        # which exceeds tile capacity. Always reshard except in colocate mode
        # (non-sleep) where model stays on GPU anyway.
        # Can be overridden via config: reshard_after_forward: false
        reshard_policy = cfg.get("reshard_after_forward",
                                 self._vllm_mode != "colocate")
        # MEMPROBE: baseline (pre-policy)
        try:
            import sys as _sys
            _mp_path = "/lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b"
            if _mp_path not in _sys.path:
                _sys.path.insert(0, _mp_path)
            from mem_probe import dump_mem as _dump_mem_init
            _dump_mem_init("INIT pre-policy")
        except Exception as _e:
            log.warning("mem_probe pre-policy failed: %r", _e)
            _dump_mem_init = None
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_sd=checkpoint_dict[training.MODEL_KEY],
            reshard_after_forward=reshard_policy,
        )
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-policy")
            except Exception:
                pass
        # Free the policy state-dict references that load_from_full_model_state_dict
        # nullified internally (release_sd=True). Without this, the checkpoint_dict
        # outer keys + the FullModelHFCheckpointer's own buffers stay live on rank 0.
        # Test T showed rank 0/1 carry +10 GiB external HBM through training; this is
        # the targeted release.
        import gc as _gc_post_policy
        try:
            checkpoint_dict.clear()
        except Exception:
            pass
        del checkpoint_dict
        _gc_post_policy.collect()
        if self._device.type == "xpu":
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-policy-cleanup")
            except Exception:
                pass
        # Setup reference model
        ref_checkpoint_dict = self.load_checkpoint(
            cfg_checkpointer=cfg.ref_checkpointer
        )
        # ref model: use ref_cpu_offload if set (saves ~12 GiB XPU HBM).
        # Policy model stays on GPU for inline generation; ref can live on CPU.
        _ref_offload = self.fsdp_cpu_offload or self._ref_cpu_offload
        self._ref_model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=_ref_offload,
            model_sd=ref_checkpoint_dict[training.MODEL_KEY],
            eval_mode=True,
            reshard_after_forward=True,
        )
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-ref")
            except Exception:
                pass
        # Free ref state-dict references (mirror of policy cleanup above).
        try:
            ref_checkpoint_dict.clear()
        except Exception:
            pass
        del ref_checkpoint_dict
        _gc_post_policy.collect()
        if self._device.type == "xpu":
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
        if _dump_mem_init is not None:
            try:
                _dump_mem_init("INIT post-ref-cleanup")
            except Exception:
                pass
        if _ref_offload and not getattr(self, "_ref_no_fsdp2", False):
            # FSDP2 cpu_offload manages parameters only; registered buffers
            # (layer_scalar, rope cache, etc.) are left on CPU after setup.
            # Move them to XPU so they don't cause device mismatch against XPU
            # inputs during ref forward passes. Params still live on CPU and are
            # fetched to XPU on-demand by FSDP2's param-fetch hook.
            # v120: skip for _ref_no_fsdp2 case — the whole model (params+buffers)
            # is on CPU; generate_trajectory() does model.to(device) before fwd.
            for _buf_name, _buf in self._ref_model.named_buffers():
                if _buf is not None and _buf.device.type == "cpu":
                    _buf.data = _buf.data.to(self._device)
            if self._is_rank_zero:
                log.info(
                    "ref_cpu_offload: moved registered buffers to XPU "
                    "(params remain on CPU for FSDP2 on-demand fetch)"
                )
        # Skip barrier on multi-node (XCCL sub-communicator creation deadlocks
        # on 2D mesh). FSDP will implicitly sync during first forward pass.
        if not self._production_mode:
            torch.distributed.barrier()

        # Utilize the same tokenizer for both models (hack)
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        # Detect chunked output loss (takes raw logits instead of logprobs)
        self._use_chunked_loss = hasattr(self._loss_fn, "num_output_chunks")
        if self._use_chunked_loss:
            log.info(
                "Using chunked output loss with %d chunks",
                self._loss_fn.num_output_chunks,
            )
        if self._compile:
            _saved_tcd = os.environ.pop("TORCH_COMPILE_DISABLE", None)
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)
            if _saved_tcd is not None:
                os.environ["TORCH_COMPILE_DISABLE"] = _saved_tcd

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        collate_name = cfg.get(
            "collate_fn", "torchtune.dev.grpo.data.padded_collate_rl"
        )
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            dataloader_state_dict=(
                checkpoint_dict[training.DATALOADER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # Setup eval dataset (if configured)
        cfg_eval_dataset = cfg.get("eval_dataset", None)
        if cfg_eval_dataset is not None and self._eval_every_n_steps > 0:
            eval_ds = config.instantiate(cfg_eval_dataset, self._tokenizer)
            max_ex = min(self._eval_max_examples, len(eval_ds))
            self._eval_examples = [eval_ds[i] for i in range(max_ex)]
            self._eval_enabled = True
            if self._is_rank_zero:
                log.info("Eval dataset loaded: %d examples (from %d total), eval every %d steps",
                         max_ex, len(eval_ds), self._eval_every_n_steps)
        else:
            self._eval_examples = []

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        self._steps_per_epoch = len(self._dataloader)
        self.global_step = self._epochs_run * self._steps_per_epoch

        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # RL params
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size

        # Reward mode: "math" (default) or "gene_recall"
        self._reward_mode = cfg.get("reward_mode", "math")
        self._gene_reward_metric = cfg.get("gene_reward_metric", "f1")

        # Sequence packing: bin-pack variable-length sequences to eliminate
        # padding waste in the training forward/backward pass.
        self._enable_packing = cfg.get("enable_packing", False)
        if self._enable_packing:
            log.info("Sequence packing enabled for GRPO training forward/backward")

        self._ppo_epochs = cfg.ppo_epochs
        self._save_every_n_epochs = cfg.save_every_n_epochs
        self._total_steps = cfg.num_steps

        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer, "stop_tokens"):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

        # --- vLLM initialization ---
        if self._vllm_mode == "server":
            self._setup_vllm_server_mode()
        elif self._vllm_mode in ("colocate", "colocate_sleep"):
            self._setup_vllm_colocate_mode(cfg)
            # Use synchronize-only mode for device_empty_cache — the full
            # torch.xpu.empty_cache() can deadlock when vLLM engines are
            # present alongside FSDP collectives on the same XPU tile.
            global _colocate_vllm_mode
            _colocate_vllm_mode = True

    def _setup_vllm_server_mode(self):
        """Initialize vLLM in server mode.

        Only global rank 0 creates HTTP clients and calls vLLM for generation.
        Results are broadcast to all ranks via the world process group.

        NOTE: XCCL on Aurora deadlocks when using shard-level sub-communicators,
        so per-node vLLM generation with shard-level broadcast is not possible.
        Only rank 0's vLLM server is used; other nodes' vLLM servers are idle.
        """
        should_init_client = self._is_rank_zero

        if should_init_client:
            from torchtune.dev.grpo.vllm_client import VLLMClient

            # Create clients for all vLLM URLs (supports DP vLLM replicas)
            self._vllm_clients = []
            for url in self._vllm_urls:
                client = VLLMClient(
                    base_url=url,
                    group_port=self._vllm_group_port,
                    connection_timeout=900.0,
                )
                self._vllm_clients.append(client)
            self._vllm_client = self._vllm_clients[0]  # backward compat

            if self._vllm_weight_sync:
                # On XPU, creating a second ProcessGroupXCCL (for weight sync)
                # alongside the training XCCL PG causes SIGABRT. Use file-based
                # weight sync instead: save to /tmp, POST to vLLM to reload.
                self._build_tune_to_hf_map()
                # Use /dev/shm (RAM-backed tmpfs, 504 GB on Aurora) for fast I/O.
                # Raw bytes format (.raw) replaces safetensors — uses memcpy at DDR5
                # bandwidth instead of CPU serialization. For 62 GB BF16: ~2s vs ~47s.
                # Async: gather is synchronous (FSDP collective), but save+HTTP runs
                # in a background thread overlapping with the next generation step.
                self._weight_sync_path = "/dev/shm/torchtune/weight_update.raw"
                # Async sync state: event is set when no sync is in progress (safe to generate).
                self._sync_done_event = threading.Event()
                self._sync_done_event.set()  # initially clear (no sync running)
                self._sync_error = None      # captured exception from background thread
                log.info(
                    "Rank %d: vLLM %d client(s) initialized: %s (%d params mapped, async raw-bytes sync via %s, interval=%d)",
                    self.rank,
                    len(self._vllm_clients),
                    ",".join(self._vllm_urls),
                    len(self._tune_to_hf_map),
                    self._weight_sync_path,
                    self._vllm_weight_sync_interval,
                )
            else:
                log.info(
                    "Rank %d: vLLM %d client(s) initialized (generation only, no weight sync): %s",
                    self.rank, len(self._vllm_clients), ",".join(self._vllm_urls),
                )

        # NOTE: Do NOT use shard_pg for any explicit collectives on Aurora.
        # XCCL deadlocks when creating sub-communicators from device_mesh groups.
        # FSDP2 internally manages its own communicators and works fine, but
        # explicit operations on dp_mesh.get_group("dp_shard") deadlock.
        if not self._production_mode:
            torch.distributed.barrier()

    def _setup_vllm_colocate_mode(self, cfg):
        """Initialize vLLM in colocate mode: every rank has its own vLLM engine.

        All ranks have vLLM engines (created in _init_vllm_early). Each rank
        generates all completions for its own prompts locally (no cross-rank
        communication during generation). Weight sync loads gathered FSDP2
        params into each rank's local engine.

        Called from setup() AFTER model init.
        """
        # Build param name mapping for weight sync
        self._build_tune_to_hf_map()

        log.info(
            "Rank %d: colocated vLLM engine ready (%d params mapped, local generation)",
            self.rank, len(self._tune_to_hf_map),
        )
        torch.distributed.barrier()

    def _build_tune_to_hf_map(self):
        """Build torchtune -> HuggingFace parameter name mapping for weight sync.

        Uses named_parameters() instead of state_dict() to avoid triggering
        FSDP1's AllGather collective (which would deadlock if only shard leaders
        call this function).
        """
        from torchtune.models.convert_weights import get_mapped_key
        from torchtune.training.checkpointing._utils import ModelType

        # Select the model-specific _FROM_HF map based on checkpointer model_type.
        # This avoids hardcoding Qwen2's map and works for Gemma4, etc.
        _model_type = getattr(self._checkpointer, "_model_type", None)
        if _model_type == ModelType.GEMMA4:
            from torchtune.models.gemma4._convert_weights import _GEMMA4_FROM_HF as _FROM_HF
        elif _model_type == ModelType.GEMMA2:
            from torchtune.models.gemma2._convert_weights import _FROM_HF
        elif _model_type in (None, ModelType.QWEN2):
            from torchtune.models.qwen2._convert_weights import _FROM_HF
        elif _model_type == ModelType.QWEN3:
            from torchtune.models.qwen3._convert_weights import _FROM_HF
        else:
            # Fallback: use standard HF->tune mapping dict
            from torchtune.models.qwen2._convert_weights import _FROM_HF
            log.warning(
                "Unknown model type %s for weight sync mapping, falling back to Qwen2 _FROM_HF",
                _model_type,
            )
        inverted = {v: k for k, v in _FROM_HF.items() if v is not None}
        self._tune_to_hf_map = {}
        for tune_name, _ in self._model.named_parameters():
            # Strip FSDP and activation checkpoint wrapper prefixes for mapping.
            # state_dict() returns clean names, so key the map by clean name
            # so lookups in _sync_weights_to_vllm hit correctly.
            clean_name = tune_name.replace("_fsdp_wrapped_module.", "")
            clean_name = clean_name.replace("_checkpoint_wrapped_module.", "")
            self._tune_to_hf_map[clean_name] = get_mapped_key(
                clean_name, inverted
            )

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                log.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        optimizer = self._optimizer

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]
                self.profiler_num_cycles = profiler_cfg["num_cycles"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        fsdp_cpu_offload: bool,
        model_sd: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        eval_mode: bool = False,
        reshard_after_forward: bool = True,
    ) -> tuple[nn.Module, nn.Module]:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``

        HSDP mode (dp_replicate > 1):
           Uses FSDP1 FullyShardedDataParallel with HYBRID_SHARD strategy.
           FSDP2's per-layer fully_shard() creates too many XCCL sub-communicators
           which hang on Aurora XPU. FSDP1's single wrapping call avoids this.
        """
        # HSDP mode: use FSDP1 path for small models (fits on single tile).
        # For large models (>60 GiB), FSDP1 OOMs during flatten (needs full model
        # on device + flat buffer simultaneously). Fall through to FSDP2 FULL_SHARD
        # which uses meta device init and never materializes the full model.
        # FSDP2 FULL_SHARD across all ranks (no HSDP mesh) avoids the sub-PG deadlock.
        model_bytes = sum(v.numel() * v.element_size() for v in model_sd.values())
        model_gib = model_bytes / (1024 ** 3)
        # EP requires FSDP2: FSDP1 flattening is incompatible with post-load EP weight slicing.
        _ep_active = self._expert_parallel_degree > 1 and self._dp_mesh is not None
        if self._dp_replicate > 1 and model_gib < 50.0 and not _ep_active:
            self._use_fsdp1 = True
            return self._setup_model_fsdp1_hsdp(
                cfg_model=cfg_model,
                enable_activation_checkpointing=enable_activation_checkpointing,
                model_sd=model_sd,
                eval_mode=eval_mode,
                reshard_after_forward=reshard_after_forward,
            )
        elif self._dp_replicate > 1:
            reason = "EP active" if _ep_active else f"model too large ({model_gib:.1f} GiB)"
            utils.log_rank_zero(
                log,
                f"Using FSDP2 with HSDP mesh ({reason}): "
                f"dp_replicate={self._dp_replicate} × dp_shard={self._dp_shard}.",
            )

        self._use_fsdp1 = False
        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if eval_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        # v120: early-return path for ref model when EP active + ref_cpu_offload.
        # At this point the model is on meta device with clean (pre-AC, pre-FSDP2) param names
        # that match model_sd keys directly. We skip all EP/AC/FSDP2 wrapping and do a plain
        # to_empty('cpu') + load_state_dict. The expert params in model_sd are pre-sliced later
        # (below), but here we do it before the early return since we won't reach that code.
        # Ref forward uses gloo AllToAll (no_grad path) — no EP FSDP2 needed.
        if self._compile:
            # Temporarily allow torch.compile (TORCH_COMPILE_DISABLE=1 set for vLLM)
            _saved_tcd = os.environ.pop("TORCH_COMPILE_DISABLE", None)
            training.compile_model(
                model, verbose=self._is_rank_zero, dynamic=self._compile_dynamic
            )
            if _saved_tcd is not None:
                os.environ["TORCH_COMPILE_DISABLE"] = _saved_tcd

        # EP v42: Pre-shrink expert meta params BEFORE activation checkpointing wrapping.
        # v41 bug: meta param shrinking was done AFTER set_activation_checkpointing, which
        # inserts "_checkpoint_wrapped_module" into module names. model_sd uses clean names
        # (e.g. "layers.0.moe_block.experts.gate_proj") but named_modules() after AC wrapping
        # yields "layers.0._checkpoint_wrapped_module.moe_block.experts.gate_proj" → 0 keys
        # matched in model_sd pre-slicing → full [128,...] tensors loaded into [32,...] params
        # → "start (0) + length (128) exceeds dimension size (32)".
        # v42 fix: shrink meta params BEFORE AC wrapping so names match model_sd keys.
        _ep_active = self._expert_parallel_degree > 1 and self._dp_mesh is not None
        _expert_param_names: set = set()  # expert param full names for model_sd pre-slicing
        if _ep_active:
            from torchtune.modules.moe.experts import GroupedExperts as _GE_pre
            ep_mesh = self._dp_mesh["dp_shard"]  # 4-rank submesh per DP replica
            _ep_rank = ep_mesh.get_local_rank()
            _ep_degree = ep_mesh.shape[0]
            # Collect expert param names and pre-shrink meta params from [128,...] → [32,...].
            # At this point model has original (clean) module names matching model_sd keys.
            for _ename, _emod in model.named_modules():
                if not (_ename.endswith(".experts") and isinstance(_emod, _GE_pre)):
                    continue
                for _pname, _param in list(_emod.named_parameters(recurse=False)):
                    _full_shape = _param.shape
                    assert _full_shape[0] % _ep_degree == 0, (
                        f"num_experts ({_full_shape[0]}) not divisible by ep_degree ({_ep_degree})"
                    )
                    _n_local = _full_shape[0] // _ep_degree
                    _new_shape = torch.Size([_n_local] + list(_full_shape[1:]))
                    setattr(_emod, _pname, nn.Parameter(
                        torch.empty(_new_shape, dtype=_param.dtype, device="meta"),
                        requires_grad=_param.requires_grad,
                    ))
                    _expert_param_names.add(f"{_ename}.{_pname}")
            utils.log_rank_zero(
                log,
                f"EP v42: pre-shrunk {len(_expert_param_names)} expert meta params "
                f"[{_full_shape[0]},...] → [{_n_local},...] "
                f"(EP rank {_ep_rank}/{_ep_degree}, before AC wrapping)",
            )

        if enable_activation_checkpointing:
            _n_moe_self_ac = _apply_split_ac(model)
            utils.log_rank_zero(
                log,
                f"v158 split AC: {_n_moe_self_ac} Gemma4 MoE layers self-checkpoint "
                f"(MoE outside AC); other layers wrapped by apply_activation_checkpointing.",
            )

        # Expert Parallelism: install All-to-All dispatch/combine hooks on GroupedExperts.
        # Must happen BEFORE shard_model so hooks are registered on the original modules.
        if _ep_active:
            from torch.distributed.tensor.parallel import parallelize_module
            from torchtune.models.gemma4._parallelism import gemma4_ep_plan
            from torchtune.modules.moe.experts import GroupedExperts
            ep_plan = gemma4_ep_plan(model)
            if ep_plan:
                parallelize_module(model, ep_mesh, ep_plan)
                utils.log_rank_zero(
                    log,
                    f"EP={self._expert_parallel_degree}: registered all-to-all hooks on "
                    f"{len(ep_plan)} module(s)",
                )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # Initialize here so it's defined even when _ep_active=False (EP=1 path).
        _expert_cpu_offload = False

        if _ep_active:
            # EP model setup (v40: expert FSDP2 on dp_replicate, not ep_mesh):
            # Expert params are EP-partitioned (32/rank). Previously we wrapped them
            # with FSDP2 on ep_mesh (ep_pg, 4-rank). This caused:
            # - v18-v31: OFI EPERM — AllToAll backward contaminates ep_pg's OFI CQ;
            #   FSDP2 all_gather on same ep_pg reads stale error.
            # - v32-v39: separate fsdp2_ep_pg (same 4 EP ranks, different CCL comm)
            #   fixed EPERM but introduced DEADLOCK: FSDP2 all_gather (fsdp2_ep_pg)
            #   and AllToAll backward (ep_pg) run concurrently over the same 4 ranks.
            #   With CCL_WORKER_COUNT=1, different ranks advance at different speeds →
            #   some call all_gather, others call AllToAll → both block → deadlock.
            #
            # Fix (v40): wrap experts with FSDP2 on dp_replicate (3-rank) instead.
            # dp_replicate groups ({0,4,8}, {1,5,9}, etc.) are DISJOINT from ep_mesh
            # groups ({0,1,2,3}, {4,5,6,7}, {8,9,10,11}).
            # Expert FSDP2 all_gather and AllToAll can NEVER deadlock — they operate
            # on different, non-overlapping sets of ranks.
            # Expert FSDP2 semantics: each rank shards its 32-expert EP slice across
            # 3 dp_replicate ranks (32/3 ≈ 10-11 per rank at rest). Before expert
            # forward (dispatch+compute), FSDP2 all_gathers → each rank gets its full
            # 32 experts. Expert compute is correct. Reduce_scatter after backward
            # averages grads across 3 dp_replicate replicas — correct for DP.
            # Wire EP dispatch/combine directly onto each MoE module.
            # parallelize_module stores _ep_instance on GroupedExperts; wire_ep_to_moe_modules
            # reads it and sets moe._ep_dispatch/_ep_combine as plain callables so that
            # MoE.forward() calls them directly (no FSDP2 hooks needed).
            from torchtune.modules.moe import wire_ep_to_moe_modules
            n_ep_wired = wire_ep_to_moe_modules(model)
            utils.log_rank_zero(
                log,
                f"EP: wired dispatch/combine callables on {n_ep_wired} MoE modules",
            )
            # v42: wrap expert modules with trivial 1-rank FSDP2 (no communication).
            # Meta params were pre-shrunk to [32,...] BEFORE AC wrapping (above),
            # so FSDPParam is initialized with the correct EP-local sizes.
            # v123: skip for ref model (eval_mode=True). The ref model uses 1-rank solo FSDP2
            # for the entire model (root shard_model call below), so separate expert FSDP2
            # is not needed. Expert dispatch is wired by parallelize_module above.
            if not eval_mode:
                from torch.distributed._composable.fsdp import fully_shard as _fully_shard
                from torchtune.modules.moe.experts import GroupedExperts as _GE
                # Wrap with 1-rank solo FSDP2. FSDPParam sees [32,...] shapes.
                # ALL ranks must call new_group for each 1-rank group (one per rank) in the
                # same order. 12 new_group calls total (world_size), each of size 1.
                _solo_groups = []
                for _r in range(self.world_size):
                    _sg = torch.distributed.new_group([_r])
                    _solo_groups.append(_sg)
                _my_solo_pg = _solo_groups[self.rank]
                from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh
                _solo_mesh = _DeviceMesh.from_group(_my_solo_pg, "xpu")
                # Use isinstance only (not name suffix) — after AC wrapping the module path
                # contains "_checkpoint_wrapped_module" so endswith(".experts") would miss them.
                _n_solo_wrapped = 0
                _solo_wrapped_mods = []
                for _ename, _emod in model.named_modules():
                    if isinstance(_emod, _GE):
                        # reshard_after_forward=False: no-op (1-rank group, nothing to gather)
                        _fully_shard(_emod, mesh=_solo_mesh, reshard_after_forward=False)
                        _solo_wrapped_mods.append(_emod)
                        _n_solo_wrapped += 1
                utils.log_rank_zero(
                    log,
                    f"EP v43: wrapped {_n_solo_wrapped} expert modules with trivial 1-rank FSDP2 "
                    f"(no communication, excludes them from root dp_replicate FSDP2)",
                )
                # Suppress reduce_grads on expert 1-rank FSDP2 groups.
                # CCL's reduce_scatter_tensor for 1-rank groups tries to register L0 IPC handles
                # for the grad buffer → ze_handle_manager "unknown memory type" crash (v42).
                # With reduce_grads=False, FSDP2 skips reduce_scatter in post_backward entirely;
                # expert grads accumulate in param.grad for manual AllReduce in train().
                # After _fully_shard(), each expert module is an FSDPModule — use ._get_fsdp_state()
                # which is the stable public API on FSDPModule.
                from torch.distributed.fsdp import FSDPModule as _FSDPModule
                _n_grads_suppressed = 0
                for _emod in _solo_wrapped_mods:
                    if isinstance(_emod, _FSDPModule):
                        _fsdp_state = _emod._get_fsdp_state()
                        if _fsdp_state is not None and _fsdp_state._fsdp_param_group is not None:
                            _fsdp_state._fsdp_param_group.reduce_grads = False
                            _n_grads_suppressed += 1
                utils.log_rank_zero(
                    log,
                    f"EP v43: suppressed reduce_grads on {_n_grads_suppressed} expert FSDPParamGroups "
                    f"(prevents CCL ze_handle_manager crash for 1-rank reduce_scatter)",
                )
            else:
                # Ref model (eval_mode=True): create NEW solo process groups for the root
                # 1-rank FSDP2 (below). All 12 ranks must participate in all 12 new_group
                # calls (PyTorch collective constraint). Store as instance attr so
                # shard_model() sees it via fsdp2_mesh.
                # v123: using 1-rank solo mesh for root FSDP2 instead of dp_replicate.
                from torch.distributed.device_mesh import DeviceMesh as _DeviceMeshRef
                _ref_root_solo_groups = []
                for _r in range(self.world_size):
                    _ref_root_sg = torch.distributed.new_group([_r])
                    _ref_root_solo_groups.append(_ref_root_sg)
                self._ref_root_solo_mesh = _DeviceMeshRef.from_group(
                    _ref_root_solo_groups[self.rank], "xpu"
                )
                utils.log_rank_zero(
                    log, "EP v123: created 1-rank solo mesh for ref root FSDP2 (no dp_replicate all-gather)"
                )

        # Standard shard conditions — solo-wrapped experts are already FSDP2-wrapped;
        # shard_model will skip them (inner FSDP units are opaque to outer).
        fsdp_shard_conditions = [partial(training.get_shard_conditions, names_to_match=custom_sharded_layers)]

        # Policy doesn't reshard after forward for faster generation.
        # Reference net reshards after forward because it never calls .backward()
        # EP-mode: use dp_replicate (3-rank) mesh for non-expert FSDP2 instead of the
        # EP+FSDP2 communicator conflict fix:
        # XCCL cannot handle concurrent collectives from different process groups on the
        # same OFI fabric. In EP mode, the EP AllToAll backward (on ep_mesh/dp_shard
        # 4-rank group) fires concurrently with FSDP2's pre_backward all-gather (on
        # dp_replicate 3-rank group) → OFI EPERM (err=265) → allgatherv EXCEPTION.
        #
        # Root cause: FSDP2 reshard_after_forward=True causes each layer to reshard
        # after forward, then re-all-gather during backward (pre_backward hook). This
        # all-gather collides with EP AllToAll backward at the same time.
        #
        # Fix: Use reshard_after_forward=False (ZeRO-2 style) for non-expert params
        # in EP mode. Parameters stay gathered (not resharded) after forward, so there
        # is NO all-gather during backward — only reduce-scatter (post_backward).
        # Reduce-scatter runs AFTER backward completes, when EP AllToAll is already done.
        # Cost: ~2× more memory for non-expert params vs reshard_after_forward=True.
        # At 12.96 GiB alloc with 24 GiB HBM, this is acceptable.
        #
        # Without EP: use full dp_mesh (HSDP) with standard reshard_after_forward.
        # FSDP2 mesh and reshard strategy:
        # - Policy (eval_mode=False, EP active): dp_replicate (3-rank) mesh, ZeRO-2
        #   (reshard_after_forward=False) — params stay gathered after fwd, no all-gather
        #   during bwd (prevents XCCL conflict with EP AllToAll backward).
        # - Ref (eval_mode=True, EP active): 1-rank solo mesh, ZeRO-3 (reshard_after_forward=True)
        #   + cpu_offload=True. Each layer's params are fetched CPU→XPU individually during fwd
        #   via simple memory copy (no XCCL, no IPC handles). Released back to CPU after each layer.
        #   v123 rationale:
        #     v118: ZeRO-3 on dp_replicate + cpu_offload=True → FSDP2 bug: 14.94 GiB stays in
        #       XPU allocated after reshard (XCCL IPC handle cache holds reference).
        #     v120-v122: .to(device)/.cpu() approach → XCCL IPC handle corruption → SIGSEGV.
        #     v123: 1-rank solo mesh → no inter-rank communication → no XCCL IPC handles
        #       involved in ref FSDP2 all-gather. Pure CPU→XPU memory copy per layer.
        #       Pre-bwd: 12.23 + 15.57 ≈ 27.8 GiB. Peak backward: 27.8 + 16.18 ≈ 44 GiB << 64.
        if _ep_active and self._dp_mesh.ndim > 1:
            if eval_mode:
                # Ref model: 1-rank solo FSDP2 (created in EP block above).
                fsdp2_mesh = self._ref_root_solo_mesh
                fsdp2_raf = True  # reshard after each layer → CPU→XPU→CPU per layer
                # v126: set requires_grad=False on all ref model params BEFORE FSDP2 wrapping.
                # With 1-rank solo FSDP2, FSDPParams inherit requires_grad from original params.
                # If requires_grad=True (default), FSDP2 registers AccumulateGrad hooks on
                # FSDPParams → pre_backward unshard hooks fire at policy backward() start →
                # try to all-gather CPU ref params back to XPU → crash (state machine conflict
                # or null pointer in FSDP2 C++ hook since no gradient flows through ref model).
                # With requires_grad=False: NO AccumulateGrad hooks → NO FSDP2 backward hooks
                # registered → ref model FSDP2 is transparent to policy backward.
                # Forward (unshard + compute + reshard) still works — hooks are module-level
                # (forward hooks), not gradient-level.
                model.requires_grad_(False)
                log.info(
                    "EP v126: ref model requires_grad_(False) before FSDP2 wrapping — "
                    "prevents FSDP2 backward hooks from firing during policy backward."
                )
                log.info(
                    "EP v123: ref model using 1-rank solo FSDP2 mesh + cpu_offload=True + "
                    "reshard_after_forward=True. Per-layer CPU→XPU copy, no XCCL."
                )
            else:
                # Policy model: dp_replicate ZeRO-2.
                fsdp2_mesh = self._dp_mesh["dp_replicate"]
                fsdp2_raf = False
                log.info(
                    "EP active: using reshard_after_forward=False (ZeRO-2) for non-expert "
                    "FSDP2 on dp_replicate mesh to prevent XCCL conflict with EP AllToAll."
                )
        else:
            fsdp2_mesh = self._dp_mesh
            fsdp2_raf = reshard_after_forward

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=fsdp2_raf,
            dp_mesh=fsdp2_mesh,
            disable_prefetch=self._disable_prefetch,
        )

        # Disable FSDP2 backward prefetch for ALL wrapped modules when EP is active.
        # v40: expert FSDP2 is on dp_replicate (disjoint from ep_pg) — backward prefetch
        # would fire async all_gathers on dp_replicate during AllToAll backward. While
        # these are on different groups (no deadlock possible), disabling prefetch ensures
        # sequential unshard → AllToAll → next layer, matching the backward graph order.
        # Disabling prefetch also avoids the pre-v40 issue where prefetch triggered
        # expert all_gathers on ep_pg during AllToAll backward → OFI EPERM.
        if _ep_active:
            training.disable_fsdp2_backward_prefetch(model)

        # v59: Suppress reduce_grads on ALL FSDPParamGroups when EP is active.
        # Expert groups already have reduce_grads=False (set above, v43).
        # Non-expert groups (on dp_replicate 1D mesh) still fire reduce_scatter_tensor
        # during backward, which goes through _xpu_reduce_scatter_via_allreduce and
        # calls gloo AllReduce on _GLOO_DP_REP_PG. But different EP replica groups
        # ({0,4,8}, {1,5,9}, ...) may be at different params when the AllReduce fires
        # (EP load imbalance → different backward execution order per replica group) →
        # gloo preamble size mismatch → SIGABRT.
        # Fix: suppress reduce_grads=False on ALL FSDPParamGroups. FSDP2 then skips
        # reduce_scatter entirely during backward. Grads accumulate in param.grad locally.
        # Post-backward: _ep_post_backward_grad_sync() iterates all params sequentially
        # and does gloo AllReduce in a fixed parameter order — no ordering race possible.
        if _ep_active:
            from torch.distributed.fsdp import FSDPModule as _FSDPModuleV59
            _n_all_suppressed = 0
            for _mod in model.modules():
                if isinstance(_mod, _FSDPModuleV59):
                    _fsdp_state = _mod._get_fsdp_state()
                    if _fsdp_state is not None and _fsdp_state._fsdp_param_group is not None:
                        if _fsdp_state._fsdp_param_group.reduce_grads:
                            _fsdp_state._fsdp_param_group.reduce_grads = False
                            _n_all_suppressed += 1
            utils.log_rank_zero(
                log,
                f"EP v59: suppressed reduce_grads on {_n_all_suppressed} additional "
                f"non-expert FSDPParamGroups (post-backward manual gloo AllReduce instead)",
            )

        # Load checkpoint.
        # v41: expert meta params were pre-shrunk to [32,...] before fully_shard, so
        # FSDPParam expects [32,...] tensors. Pre-slice model_sd expert params here so
        # load_from_full_model_state_dict copies the correct [32,...] EP shard.
        # Non-expert params: FSDP2 DTensors on dp_replicate (3-rank), auto-sliced.
        if _ep_active:
            _n_sd_sliced = 0
            for _sd_name in list(model_sd.keys()):
                if _sd_name in _expert_param_names:
                    _ft = model_sd[_sd_name]
                    assert _ft.shape[0] % _ep_degree == 0, (
                        f"Expert param {_sd_name}: shape[0]={_ft.shape[0]} not divisible by ep_degree={_ep_degree}"
                    )
                    _n_local = _ft.shape[0] // _ep_degree
                    model_sd[_sd_name] = _ft[_ep_rank * _n_local : (_ep_rank + 1) * _n_local].contiguous()
                    _n_sd_sliced += 1
            utils.log_rank_zero(
                log,
                f"EP v41: pre-sliced {_n_sd_sliced} expert params in model_sd "
                f"(full [128,...] → [{_n_local},...] for EP rank {_ep_rank}/{_ep_degree})",
            )
        training.load_from_full_model_state_dict(
            model,
            model_sd,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )
        # v41: EP weight slicing was done via model_sd pre-slicing above.
        # apply_ep_weight_sharding is no longer called post-load.

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        # Store vocab size for OOB clamping (FSDP2 with use_orig_params keeps shapes)
        if not hasattr(self, '_vocab_size') and hasattr(model, 'tok_embeddings'):
            self._vocab_size = model.tok_embeddings.weight.shape[0]

        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            try:
                memory_stats = training.get_memory_stats(device=self._device)
                training.log_memory_stats(memory_stats)
            except RuntimeError:
                pass

        # Verify allocator config is applied (PYTORCH_ALLOC_CONF, not TORCH_XPU_ALLOC_CONF)
        if self._is_rank_zero:
            _alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "<NOT SET>")
            _bad_conf = os.environ.get("TORCH_XPU_ALLOC_CONF")
            log.info("PYTORCH_ALLOC_CONF=%s", _alloc_conf)
            if _bad_conf:
                log.warning(
                    "TORCH_XPU_ALLOC_CONF=%s is set but NOT recognized by PyTorch. "
                    "Use PYTORCH_ALLOC_CONF instead.", _bad_conf
                )

        # FSDP diagnostic: log wrapping structure and per-unit reshard settings
        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_structure(model, log=log)
            training.verify_activation_checkpointing(model, log=log)
            if self._disable_prefetch:
                log.info("FSDP prefetch DISABLED (reshard_after_forward at root = per-layer setting)")
            else:
                log.info("FSDP prefetch ENABLED (reshard_after_forward=None at root)")

        # Register per-layer memory hooks for diagnostics (rank 0 only)
        if self._fsdp_diagnostics and self._is_rank_zero:
            self._layer_mem_hooks = training.register_per_layer_memory_hooks(
                model, self._device, log, sample_every=10,
            )
        else:
            self._layer_mem_hooks = []

        disable_dropout(model)

        # v40: no manual FSDP2 param group tracking needed.
        # Non-expert: FSDP2 reduce_scatter on dp_replicate handles grad sync natively.
        # Expert: 1-rank solo FSDP2 (no-op); manual AllReduce in train() after grpo_step.
        self._expert_fsdp_modules = []
        self._fsdp2_param_groups_meta = []  # empty — no manual sync needed in v40

        return model

    def _setup_model_fsdp1_hsdp(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_sd: dict[str, Any],
        eval_mode: bool = False,
        reshard_after_forward: bool = True,
    ) -> nn.Module:
        """FSDP1-based HSDP model setup.

        Uses FullyShardedDataParallel with HYBRID_SHARD strategy and a 2D
        DeviceMesh. This is proven to work on Aurora XPU (used by PRISM).

        FSDP2's composable fully_shard() creates per-layer XCCL sub-communicators
        which hang on Aurora. FSDP1's single FSDP() call avoids this by managing
        all sub-communicators internally in one initialization.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        utils.log_rank_zero(
            log,
            f"HSDP (FSDP1): Instantiating model on device "
            f"(dp_replicate={self._dp_replicate} × dp_shard={self._dp_shard})...",
        )
        init_start = time.perf_counter()

        # Instantiate on CPU first, load state dict, then move to device.
        # For large models (32B), ZE_AFFINITY_MASK must be unset so the
        # allocator can spill across tiles during FSDP1 flatten.
        with training.set_default_dtype(self._dtype):
            model = config.instantiate(cfg_model)

        if eval_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        # Load checkpoint BEFORE wrapping with FSDP1
        model.load_state_dict(model_sd, strict=True)
        del model_sd  # free CPU memory

        # Move to device
        model = model.to(device=self._device, dtype=self._dtype)

        # Store vocab size BEFORE FSDP wrapping (FSDP shards tok_embeddings,
        # making .weight.shape[0] return shard size instead of vocab size)
        if hasattr(model, 'tok_embeddings'):
            self._vocab_size = model.tok_embeddings.weight.shape[0]

        # Initialize RoPE
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

        if enable_activation_checkpointing:
            _n_moe_self_ac = _apply_split_ac(model)
            utils.log_rank_zero(
                log,
                f"v158 split AC: {_n_moe_self_ac} Gemma4 MoE layers self-checkpoint "
                f"(MoE outside AC); other layers wrapped by apply_activation_checkpointing.",
            )

        # Mixed precision policy
        mp_policy = MixedPrecision(
            param_dtype=self._dtype,
            reduce_dtype=self._dtype,
            buffer_dtype=self._dtype,
        )

        # Sharding strategy: reshard_after_forward=True → HYBRID_SHARD (FULL_SHARD intra-node)
        #                     reshard_after_forward=False → _HYBRID_SHARD_ZERO2 (SHARD_GRAD_OP intra-node)
        if reshard_after_forward:
            sharding = ShardingStrategy.HYBRID_SHARD
            shard_label = "HYBRID_SHARD"
        else:
            try:
                sharding = ShardingStrategy._HYBRID_SHARD_ZERO2
                shard_label = "_HYBRID_SHARD_ZERO2"
            except AttributeError:
                sharding = ShardingStrategy.HYBRID_SHARD
                shard_label = "HYBRID_SHARD (fallback)"

        utils.log_rank_zero(log, f"HSDP: Using {shard_label} strategy")

        # Wrap with FSDP1 — single call, minimal sub-communicator creation
        model = FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp_policy,
            device_mesh=self._dp_mesh,
            use_orig_params=True,
            limit_all_gathers=True,
        )

        utils.log_rank_zero(
            log,
            f"HSDP model setup took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            try:
                memory_stats = training.get_memory_stats(device=self._device)
                training.log_memory_stats(memory_stats)
            except RuntimeError:
                pass

        disable_dropout(model)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )
        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        # Instantiate collate_fn
        collate_fn = _get_component_from_path(collate_fn)

        # When using TP>1 for vLLM, all ranks in a TP group must see identical
        # data so they generate the same prompts (deterministic scheduling).
        # Use dp_size/dp_rank instead of world_size/rank for the sampler.
        #
        # For HSDP: ranks within the same dp_shard group process the same batch
        # (they're FSDP shards of the same model copy). Only dp_replicate copies
        # need different data. So sampler_replicas = dp_replicate, sampler_rank =
        # replicate index.
        if self._vllm_mode == "server":
            # vLLM server mode: rank 0 generates and broadcasts to all ranks.
            # All ranks must see the same batch for matching tensor shapes.
            sampler_replicas = 1
            sampler_rank = 0
        elif self._dp_replicate > 1:
            # HSDP: each replicate group sees different data
            sampler_replicas = self._dp_replicate
            sampler_rank = self.rank // self._dp_shard
        elif self._vllm_tp_size > 1:
            sampler_replicas = self.world_size // self._vllm_tp_size
            sampler_rank = self.rank // self._vllm_tp_size
        else:
            sampler_replicas = self.world_size
            sampler_rank = self.rank

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            shuffle=shuffle,
            seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)
            list(dataloader)
        return dataloader

    def _gather_cpu_state_dict_safe(
        self,
        model,
        is_rank_zero: bool,
        device=None,
    ) -> dict:
        """Like training.gather_cpu_state_dict but uses shard PG for barriers.

        On multi-node XPU, world-level barriers crash after FSDP2 sub-communicators
        are created (broadcast_scaleout_sycl.cpp:65). This replaces the per-parameter
        world barrier with a shard PG barrier when available.
        """
        from torchtune.training._distributed import _gather_nf4_tensor
        try:
            from torchao.dtypes import NF4Tensor
        except ImportError:
            NF4Tensor = None

        cpu_state_dict = {}
        sharded_sd = model.state_dict()
        for param_name, param in sharded_sd.items():
            if param.is_cpu:
                param = param.to(device)
            if hasattr(param, "_local_tensor"):
                if NF4Tensor is not None and isinstance(param._local_tensor, NF4Tensor):
                    param = _gather_nf4_tensor(param)
                else:
                    param = param.full_tensor()
            if NF4Tensor is not None and isinstance(param, NF4Tensor):
                param = param.to(param.dtype)
            if is_rank_zero:
                cpu_state_dict[param_name] = param.cpu()
            # Skip barrier — FSDP full_tensor() is itself a synchronization point
            if not self._production_mode:
                torch.distributed.barrier()
        return cpu_state_dict

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0.
        # Use shard PG for barriers when available (multi-node XPU: world barriers
        # crash after FSDP2 sub-communicators are created).
        cpu_state_dict = self._gather_cpu_state_dict_safe(
            self._model,
            self._is_rank_zero,
            device=self._device,
        )

        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if intermediate_checkpoint:
            start = time.perf_counter()
            utils.log_rank_zero(log, "Getting optimizer state dict...")
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._model,
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
            utils.log_rank_zero(
                log,
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs",
            )
        else:
            opt_state_dict = None

        if self._is_rank_zero:
            start = time.perf_counter()
            checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self._epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.RNG_KEY: self._rng.get_state(),
                        training.DATALOADER_KEY: self._dataloader.state_dict(),
                    }
                )

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        # Skip barrier in production mode — checkpoint save is rank-0 only,
        # other ranks just need to stay out of the way during full_tensor() gather.
        if not self._production_mode:
            torch.distributed.barrier()

    def _generate_with_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Call vLLM server for generation, broadcast results to all ranks.

        Only global rank 0 calls vLLM and broadcasts to all ranks via the world
        process group. Uses a fixed buffer size (tokenizer.max_seq_len) so all
        ranks have matching tensor shapes regardless of per-rank batch variation.

        Returns:
            query_responses: ``[B*G, max_seq_len]`` (padded to fixed length)
        """
        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        if self._is_rank_zero:
            # Strip padding and convert to Python lists for HTTP
            prompts = []
            for i in range(bsz):
                ids = batch_input_ids[i].cpu().tolist()
                ids = [t for t in ids if t != self._tokenizer.pad_id]
                prompts.append(ids)

            gen_kwargs = dict(
                n=1,  # prompts already expanded by grpo_samples
                max_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k or 0,
            )

            t0 = time.perf_counter()
            num_clients = len(self._vllm_clients)
            if num_clients > 1:
                # DP vLLM: split sequences round-robin across replicas, call in parallel
                from concurrent.futures import ThreadPoolExecutor, as_completed
                chunks = [prompts[i::num_clients] for i in range(num_clients)]

                def _call_vllm(client, chunk):
                    return client.generate(prompts=chunk, **gen_kwargs) if chunk else []

                with ThreadPoolExecutor(max_workers=num_clients) as pool:
                    futures = {
                        pool.submit(_call_vllm, client, chunk): idx
                        for idx, (client, chunk) in enumerate(zip(self._vllm_clients, chunks))
                    }
                    chunk_results = [None] * num_clients
                    for future in as_completed(futures):
                        idx = futures[future]
                        chunk_results[idx] = future.result()

                # Interleave results back to original prompt order
                completions = [None] * bsz
                for i in range(bsz):
                    client_idx = i % num_clients
                    within_idx = i // num_clients
                    completions[i] = chunk_results[client_idx][within_idx]
            else:
                completions = self._vllm_client.generate(prompts=prompts, **gen_kwargs)
            gen_time = time.perf_counter() - t0

            # Build query_responses tensor: [prompt | completion | padding]
            query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
            query_responses[:, :context_length] = batch_input_ids
            for i, comp in enumerate(completions):
                length = min(len(comp), self._max_generated_tokens)
                query_responses[i, context_length : context_length + length] = torch.tensor(
                    comp[:length], dtype=batch_input_ids.dtype, device=self._device
                )

            total_tokens = sum(len(c) for c in completions)
            log.info(
                "Rank %d: vLLM generation: %d sequences (%d clients), %d tokens in %.1fs (%.1f tok/s)",
                self.rank, bsz, num_clients, total_tokens, gen_time, total_tokens / max(gen_time, 0.01),
            )
        else:
            query_responses = batch_input_ids.new_empty(bsz, total_len)

        # Broadcast from rank 0 to all ranks via world PG.
        # Broadcast from rank 0 to all ranks via world PG.
        # XCCL on Aurora deadlocks when creating sub-communicators for shard PG,
        # so we always use the world PG for explicit collectives.
        torch.distributed.broadcast(query_responses, src=0)
        return query_responses

    def _generate_with_colocated_vllm(
        self,
        batch_input_ids: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Generate using this rank's colocated vLLM engine.

        With a DistributedSampler, each rank already has its own subset of
        prompts. Each rank generates ALL grpo_samples completions for its own
        prompts locally — no cross-rank communication needed.

        Returns:
            query_responses: ``[B*G, context_length + max_generated_tokens]``
        """
        from vllm import SamplingParams

        bsz = batch_input_ids.shape[0]
        total_len = context_length + self._max_generated_tokens

        sampling_params = SamplingParams(
            max_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k if self._top_k else -1,
            detokenize=False,
        )

        # Strip padding and convert to Python lists
        prompts = []
        for i in range(bsz):
            ids = batch_input_ids[i].cpu().tolist()
            ids = [t for t in ids if t != self._tokenizer.pad_id]
            prompts.append(ids)

        t0 = time.perf_counter()
        outputs = self._vllm_llm.generate(
            prompts=[{"prompt_token_ids": p} for p in prompts],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        gen_time = time.perf_counter() - t0

        # Build query_responses: [prompt | completion | padding]
        query_responses = batch_input_ids.new_full((bsz, total_len), self._tokenizer.pad_id)
        query_responses[:, :context_length] = batch_input_ids
        total_tokens = 0
        for i, output in enumerate(outputs):
            comp = output.outputs[0].token_ids
            total_tokens += len(comp)
            length = min(len(comp), self._max_generated_tokens)
            query_responses[i, context_length : context_length + length] = torch.tensor(
                comp[:length], dtype=batch_input_ids.dtype, device=self._device
            )

        log.info(
            "Rank %d: generated %d sequences, %d tokens in %.1fs (%.1f tok/s)",
            self.rank, bsz, total_tokens, gen_time, total_tokens / max(gen_time, 0.01),
        )

        # Re-set XPU device context after vLLM generation (vLLM may have
        # called torch.xpu.set_device internally, shifting the default device).
        torch.xpu.set_device(_xpu_device_index)
        torch.xpu.synchronize()

        return query_responses

    _file_barrier_gen = 0

    def _serialized_empty_cache(self) -> None:
        """Empty XPU cache one rank at a time using file-based serialization.

        NOTE: On XPU with FSDP, empty_cache() leaks UR handles (see
        docs/bugs/intel_xpu_resource_leak_bug_report.md). This method is now a
        no-op on XPU — kept for non-XPU use or future driver fix.
        """
        if self._device.type == "xpu":
            # Skip — empty_cache + FSDP storage.resize_ leaks UR handles
            if not self._production_mode:
                torch.distributed.barrier()
            return

        gen = self._file_barrier_gen
        self.__class__._file_barrier_gen = gen + 1
        barrier_dir = f"/tmp/torchtune/empty_cache_barriers/gen{gen}"
        os.makedirs(barrier_dir, exist_ok=True)

        for r in range(self.world_size):
            if r == self.rank:
                # My turn to call empty_cache
                torch.xpu.synchronize()
                log.info("Rank %d: serialized empty_cache gen%d start", self.rank, gen)
                torch.xpu.empty_cache()
                log.info("Rank %d: serialized empty_cache gen%d done", self.rank, gen)
                # Signal done
                with open(os.path.join(barrier_dir, f"r{r}_done"), "w") as f:
                    f.write("done")
            else:
                # Wait for rank r to finish
                done_file = os.path.join(barrier_dir, f"r{r}_done")
                while not os.path.exists(done_file):
                    time.sleep(0.001)

        # No cleanup — leave barrier files (they're in /tmp and tiny)

    def _sync_colocated_weights(self) -> None:
        """Sync FSDP2 weights to colocated vLLM via load_weights().

        For each param: full_tensor() → load_weights() → del. One param
        at a time to minimize peak GPU memory (critical for 32B where
        optimizer states + vLLM weights leave <1 GiB free).

        vLLM's load_weights() handles TP slicing and QKV/gate_up merging
        internally via per-param weight_loader functions.
        """
        import gc

        t0 = time.perf_counter()
        llm_model = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model

        n_synced = 0
        for tune_name, param in self._model.named_parameters():
            clean = tune_name.replace("_checkpoint_wrapped_module.", "")
            hf_name = self._tune_to_hf_map.get(
                clean, self._tune_to_hf_map.get(tune_name, clean)
            )

            if hasattr(param, 'full_tensor'):
                weight_data = param.full_tensor()
            else:
                weight_data = param.data

            llm_model.load_weights([(hf_name, weight_data)])
            n_synced += 1
            del weight_data

            # Sync + gc every 5 params to bound UR handle pressure
            # (707 full_tensor all-gathers create UR handles that must be
            # reclaimed before the subsequent FSDP backward pass).
            if n_synced % 5 == 0 and torch.xpu.is_available():
                gc.collect()
                torch.xpu.synchronize(self._device)

        self._vllm_llm.llm_engine.reset_prefix_cache()

        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.synchronize(self._device)

        log.info(
            "Rank %d: weight sync: %d params in %.1fs",
            self.rank, n_synced, time.perf_counter() - t0,
        )

    @staticmethod
    def _save_raw_bytes(state_dict: dict, path: str) -> int:
        """Write tensors as raw bytes with a simple JSON header.

        Format: [8-byte header_len][JSON header][raw tensor bytes...]

        BF16 tensors are stored as int16 bytes (same bit pattern, no conversion)
        and tagged with dtype "torch.bfloat16" in the header so the reader can
        reinterpret correctly.

        This is ~40× faster than safetensors for large models:
        - safetensors: ~1.3 GB/s (CPU serialization bottleneck)
        - raw bytes: view(int16).numpy().tobytes() = memcpy at DDR5 bandwidth (~50 GB/s)
        - For 62 GB BF16: ~2s vs ~47s

        Returns number of params written.
        """
        import json

        header_entries = []
        np_arrays = []
        offset = 0

        for name, tensor in state_dict.items():
            dtype_str = str(tensor.dtype)  # e.g. "torch.bfloat16"
            shape = list(tensor.shape)

            # BF16 not supported by numpy — view as int16 (same bits, just reinterpreted).
            if tensor.dtype == torch.bfloat16:
                np_arr = tensor.view(torch.int16).numpy()
            else:
                np_arr = tensor.numpy()

            # Ensure contiguous layout so buffer protocol write is a single memcpy.
            if not np_arr.flags['C_CONTIGUOUS']:
                import numpy as _np
                np_arr = _np.ascontiguousarray(np_arr)

            nbytes = np_arr.nbytes

            header_entries.append({
                "name": name,
                "shape": shape,
                "dtype": dtype_str,
                "offset": offset,
                "nbytes": nbytes,
            })
            np_arrays.append(np_arr)  # zero-copy view — no tobytes() allocation
            offset += nbytes

        header_json = json.dumps(header_entries).encode("utf-8")
        header_len_bytes = struct.pack("<Q", len(header_json))

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(header_len_bytes)
            f.write(header_json)
            for arr in np_arrays:
                f.write(arr)  # buffer protocol: kernel copies directly from numpy memory

        return len(header_entries)

    def _post_weights_to_vllm(self, save_path: str, n_params: int, t_gather: float, t0: float) -> None:
        """POST to all vLLM replicas to reload from /dev/shm raw bytes file.

        Called from background thread — runs while next step's generation proceeds.
        Uses load_weights_from_raw endpoint (fast frombuffer path).
        Falls back to load_weights_from_path (safetensors) if raw not available.
        """
        import requests

        t_http0 = time.perf_counter()
        for url in self._vllm_urls:
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={"method": "load_weights_from_raw", "args": [save_path]},
                    timeout=300,
                )
                if r.status_code != 200:
                    log.warning("vLLM weight reload (raw) failed (%s): %s %s", url, r.status_code, r.text[:200])
                else:
                    result = r.json()
                    results = result.get("results", [{}])
                    first = results[0] if results else {}
                    if isinstance(first, dict) and first.get("status") != "ok":
                        log.warning("vLLM weight reload (raw) error (%s): %s", url, first)
            except Exception as e:
                log.error("vLLM weight reload HTTP error (%s): %s", url, e)
        t_http = time.perf_counter() - t_http0

        for client in self._vllm_clients:
            client.reset_prefix_cache()

        log.info(
            "Rank %d: weight sync %d vLLM replica(s): %d params %.1fs total "
            "(gather=%.1fs [sync] save=%.1fs http=%.1fs) [async raw-bytes: %s]",
            self.rank, len(self._vllm_urls), n_params, time.perf_counter() - t0,
            t_gather, time.perf_counter() - t0 - t_gather - t_http, t_http, save_path,
        )

    def _sync_weights_to_vllm(self) -> None:
        """Dispatch weight sync to the configured method (raw_bytes, shm, or xccl).

        Dispatches to:
          raw_bytes — file-based /dev/shm write + HTTP POST (default, ~9s for 3B)
          shm       — POSIX shared memory, zero-copy on vLLM side (~5s for 3B, ~3s for 31B)
          xccl      — GPU→GPU XCCL broadcast, no CPU staging (~14 GB/s on Aurora)

        Both methods are async: Phase 1 (FSDP gather) is synchronous across all ranks,
        Phase 2 (copy + POST) runs in a background thread overlapping with generation.
        Call _wait_for_sync_complete() before generate_trajectory().
        """
        _method = getattr(self, "_vllm_weight_sync_method", "raw_bytes")
        if _method == "xccl":
            return self._sync_weights_to_vllm_xccl()
        if _method == "shm":
            return self._sync_weights_to_vllm_shm()
        # raw_bytes path below:
        t0 = time.perf_counter()
        hf_state_dict = {}

        if getattr(self, '_use_fsdp1', False) and self._dp_replicate > 1:
            # FSDP1 path: state_dict() handles gathering within shard group.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
            with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
                full_sd = self._model.state_dict()
            if self._is_shard_leader:
                for param_name, param in full_sd.items():
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.cpu()
            del full_sd
        else:
            # FSDP2 path: gather DTensor → full tensor.
            sharded_sd = self._model.state_dict()
            for param_name, param in sharded_sd.items():
                if param.is_cpu:
                    param = param.to(self._device)
                if hasattr(param, "_local_tensor"):
                    param = param.full_tensor()
                if self._is_shard_leader:
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.cpu()
            del sharded_sd

        # Barrier: all ranks finish full_tensor() before shard leader saves
        if not self._production_mode:
            torch.distributed.barrier()

        t_gather = time.perf_counter() - t0

        if self._is_shard_leader:
            save_path = self._weight_sync_path
            n_params = len(hf_state_dict)

            # Mark sync in-progress BEFORE spawning thread (prevent race with
            # _wait_for_sync_complete checking the event).
            self._sync_done_event.clear()
            self._sync_error = None

            def _bg_save_and_post(state_dict=hf_state_dict):
                try:
                    t_save0 = time.perf_counter()
                    self._save_raw_bytes(state_dict, save_path)
                    t_save = time.perf_counter() - t_save0
                    del state_dict  # free ~62 GB as soon as written
                    log.info(
                        "Rank %d: async sync save done: %d params in %.1fs (%.1f GB/s) → %s",
                        self.rank, n_params, t_save,
                        (n_params and os.path.getsize(save_path) / 1024**3 / t_save if t_save > 0 else 0),
                        save_path,
                    )
                    self._post_weights_to_vllm(save_path, n_params, t_gather, t0)
                except Exception as e:
                    log.error("Rank %d: async weight sync failed: %s", self.rank, e, exc_info=True)
                    self._sync_error = e
                finally:
                    self._sync_done_event.set()

            t = threading.Thread(target=_bg_save_and_post, daemon=True, name="weight_sync")
            t.start()
            log.info(
                "Rank %d: weight sync gather done in %.1fs, async save+POST started (%d params → %s)",
                self.rank, t_gather, n_params, save_path,
            )

        device_empty_cache(self._device)

    def _init_xccl_weight_sync(self) -> None:
        """Create a cross-process XCCL group between training rank 0 and vLLM worker(s).

        Called once on first weight sync. Training is rank 0 (TCPStore master),
        vLLM worker(s) join as rank 1..N via /collective_rpc POST.

        Ordering: TCPStore (master) first, then POST to vLLM in a background
        thread (vLLM enters PG constructor, blocks waiting for training), then
        training enters PG constructor — both sides unblock simultaneously.
        """
        import datetime
        import threading
        import torch.distributed as dist
        import torch.distributed.distributed_c10d as c10d
        import requests

        xccl_port = getattr(self, "_vllm_xccl_port", 51217)
        tp_size = getattr(self, "_vllm_tp_size", 1)
        num_replicas = len(self._vllm_urls)
        # Total ranks: 1 (training) + num_replicas * tp_size (all vLLM TP workers)
        # Replica r gets base_rank = 1 + r * tp_size, occupying ranks [base_rank, base_rank + tp_size)
        world_size = 1 + num_replicas * tp_size

        # Address that vLLM workers use to reach this TCPStore master.
        # TORCHTUNE_XCCL_HOST takes priority — set this to the training node's HSN
        # address when using --standalone (which overrides MASTER_ADDR to 127.0.0.1).
        import socket as _socket
        _xccl_host = (
            os.environ.get("TORCHTUNE_XCCL_HOST")
            or os.environ.get("MASTER_ADDR")
            or _socket.gethostname()
        )

        log.info(
            "Rank %d: initializing XCCL weight sync (host=%s, port=%d, world=%d, tp=%d, replicas=%d)",
            self.rank, _xccl_host, xccl_port, world_size, tp_size, num_replicas,
        )

        store = dist.TCPStore(
            host_name="0.0.0.0",
            port=xccl_port,
            world_size=world_size,
            is_master=True,
            timeout=datetime.timedelta(seconds=120),
            wait_for_workers=False,
        )

        # POST to all vLLM replicas CONCURRENTLY — each POST blocks until that
        # replica's TP workers enter the XCCL PG constructor. All replicas must
        # join simultaneously (world_size barrier), so sequential posting would
        # deadlock: replica 0 waits for the full group while we never reach replica 1.
        # 2-hop: training rank 0 sends to vLLM rank 1 (cross-node Slingshot),
        # rank 1 distributes to ranks 2..N via intra-node XeLink broadcast.
        # Reduces sync time from ~38s (12 sequential Slingshot sends) to ~3s.
        use_two_hop = num_replicas > 0  # always use when there are cross-node vLLM workers
        self._xccl_two_hop = use_two_hop

        vllm_errors = []
        def _post_replica(r_idx, url):
            base_rank = 1 + r_idx * tp_size
            try:
                r = requests.post(
                    f"{url}/collective_rpc",
                    json={
                        "method": "init_xccl_communicator",
                        "args": [_xccl_host, xccl_port, world_size, base_rank, use_two_hop],
                    },
                    timeout=120,
                )
                if r.status_code != 200:
                    vllm_errors.append(f"init_xccl_communicator failed ({url}): {r.status_code} {r.text}")
                    return
                result = r.json().get("results", [{}])
                first = result[0] if result else {}
                if isinstance(first, dict) and first.get("status") != "ok":
                    vllm_errors.append(f"init_xccl_communicator error ({url}): {first}")
            except Exception as e:
                vllm_errors.append(str(e))

        replica_threads = [
            threading.Thread(target=_post_replica, args=(r_idx, url), daemon=True)
            for r_idx, url in enumerate(self._vllm_urls)
        ]

        for t in replica_threads:
            t.start()

        # Training enters PG constructor — unblocks vLLM side when all replicas join.
        # 2-hop: training uses a 2-rank cross PG (rank 0 + vLLM rank 1) so broadcast
        # sends exactly one cross-Slingshot copy instead of 12 sequential copies.
        opts = c10d.ProcessGroupXCCL.Options()
        if use_two_hop:
            cross_prefixed = c10d.PrefixStore("wsync_cross", store)
            self._xccl_wsync_pg = c10d.ProcessGroupXCCL(
                store=cross_prefixed, rank=0, size=2, options=opts,
            )
        else:
            prefixed = c10d.PrefixStore("wsync", store)
            self._xccl_wsync_pg = c10d.ProcessGroupXCCL(
                store=prefixed, rank=0, size=world_size, options=opts,
            )

        for t in replica_threads:
            t.join(timeout=120)
        if vllm_errors:
            raise RuntimeError(f"vLLM XCCL init failed: {vllm_errors}")

        log.info("Rank %d: XCCL weight sync communicator ready", self.rank)

    def _sync_weights_to_vllm_xccl(self) -> None:
        """Gather sharded params then broadcast to vLLM via XCCL (GPU→GPU).

        GPU-direct path: full_tensor() returns a GPU tensor, so we concat
        directly into a flat GPU buffer — no CPU staging. The background
        thread only does the HTTP POST + broadcast.

        For 32B+ models where the full flat buffer won't fit on one tile,
        falls back to CPU staging (same as SHM path).
        """
        import json
        import requests

        t0 = time.perf_counter()

        if self._is_shard_leader:
            if not hasattr(self, '_xccl_wsync_pg'):
                self._init_xccl_weight_sync()

        use_fsdp1 = getattr(self, '_use_fsdp1', False) and self._dp_replicate > 1

        if use_fsdp1:
            # FSDP1: state_dict() handles gathering; result is on CPU already
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
            with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
                full_sd = self._model.state_dict()
            hf_state_dict = {}
            if self._is_shard_leader:
                for param_name, param in full_sd.items():
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.to(self._device)
            del full_sd

            if not self._production_mode:
                torch.distributed.barrier()
            t_gather = time.perf_counter() - t0

            if self._is_shard_leader:
                tensors_meta = []
                total_elements = 0
                for hf_name, tensor in hf_state_dict.items():
                    numel = tensor.numel()
                    tensors_meta.append({"name": hf_name, "shape": list(tensor.shape),
                                         "dtype": str(tensor.dtype), "numel": numel})
                    total_elements += numel
                flat_gpu = torch.empty(total_elements, dtype=torch.bfloat16, device=self._device)
                offset = 0
                for tensor in hf_state_dict.values():
                    flat_gpu[offset:offset + tensor.numel()] = tensor.to(torch.bfloat16).flatten()
                    offset += tensor.numel()
                del hf_state_dict
        else:
            # FSDP2 weight sync: gather sharded params and broadcast to vLLM.
            # Two sub-modes selected by TORCHTUNE_XCCL_BATCHED_AG:
            #
            #   0 (default): per-param full_tensor() + batched XCCL broadcast
            #      AllGather per param: <0.12ms for 3B, ~0.8ms for 32B (negligible).
            #      Real bottleneck: XCCL broadcast bandwidth (1.7 GB/s at 12 receivers
            #      = 35.9s for 61 GiB). Tests CD/CE/CF confirm this floor.
            #
            #   1 (BROKEN): batched all_gather_into_tensor() + reconstruct + broadcast
            #      Saves ~0.2s for 3B (<1% for 32B). Leaves FSDP2 shard state
            #      inconsistent → checkpoint save hangs after sync. Do not use.
            _USE_BATCHED_AG = os.environ.get("TORCHTUNE_XCCL_BATCHED_AG", "0") == "1"
            if _USE_BATCHED_AG:
                log.warning(
                    "TORCHTUNE_XCCL_BATCHED_AG=1 is BROKEN: it leaves FSDP2 shard state "
                    "inconsistent, causing the post-training checkpoint save to hang. "
                    "Savings are <1%% for 32B (0.2s vs 38s floor). Do not use."
                )
            import threading as _threading

            _BATCH_MAX_NUMEL = 512 * 1024 * 1024  # 512M bf16 elements = 1 GiB

            sharded_sd = self._model.state_dict()

            # Build manifest from DTensor global shapes (no collective needed).
            # Same format for both sub-modes — vLLM side is unchanged.
            if self._is_shard_leader:
                tensors_meta = []
                for param_name, param in sharded_sd.items():
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    tensors_meta.append({
                        "name": hf_name,
                        "shape": list(param.shape),
                        "numel": param.numel(),
                    })
                meta_json = json.dumps({
                    "tensors": tensors_meta,
                    "batch_max_numel": _BATCH_MAX_NUMEL,
                })

                self._sync_done_event.clear()
                self._sync_error = None

                post_errors = []
                def _post_manifest(url):
                    try:
                        r = requests.post(
                            f"{url}/collective_rpc",
                            json={"method": "receive_weights_xccl_streaming",
                                  "args": [meta_json]},
                            timeout=600,
                        )
                        if r.status_code != 200:
                            post_errors.append(
                                f"streaming init failed ({url}): {r.status_code} {r.text}")
                        else:
                            result = r.json().get("results", [{}])
                            first = result[0] if result else {}
                            if isinstance(first, dict) and first.get("status") != "ok":
                                post_errors.append(f"streaming error ({url}): {first}")
                    except Exception as e:
                        post_errors.append(str(e))

                manifest_threads = [
                    _threading.Thread(target=_post_manifest, args=(url,), daemon=True)
                    for url in self._vllm_urls
                ]
                for mt in manifest_threads:
                    mt.start()

                time.sleep(0.3)  # let vLLM workers enter first broadcast.wait()

            t_ag = 0.0
            t_bcast = 0.0
            n_params = 0
            n_batches = 0

            if _USE_BATCHED_AG:
                # Mode 1: batched AllGather — reduces AllGather calls from N_params to
                # N_batches by manually gathering local shards with all_gather_into_tensor().
                # All training ranks contribute local shards; rank 0 reconstructs full
                # params from the interleaved output then broadcasts to vLLM.
                _training_ws = torch.distributed.get_world_size()

                # local batch: local shards from THIS rank, accumulated until threshold
                _lb_parts: list = []      # list of bf16 contiguous local shard tensors
                _lb_numel = 0             # total local numel accumulated so far
                _lb_metas: list = []      # (local_numel, full_numel) per param in batch
                _batch_full_numel = 0     # full numel accumulated (for threshold check)

                def _flush_batched_ag():
                    nonlocal t_ag, t_bcast, n_batches
                    local_flat = torch.cat([p.flatten() for p in _lb_parts])
                    # AllGather: shape [world_size * _lb_numel], layout:
                    #   [rank0_local | rank1_local | ... | rank_{W-1}_local]
                    full_batch = torch.empty(
                        _training_ws * _lb_numel,
                        dtype=torch.bfloat16, device=self._device,
                    )
                    tb0 = time.perf_counter()
                    torch.distributed.all_gather_into_tensor(full_batch, local_flat)
                    t_ag += time.perf_counter() - tb0
                    del local_flat

                    if self._is_shard_leader:
                        # Reconstruct full params from interleaved AllGather output.
                        # For param_i with local_numel=lnu at local_offset=cum_lnu:
                        #   full_param = cat(full_batch[r*_lb_numel+cum : r*_lb_numel+cum+lnu]
                        #                    for r in range(_training_ws))[:full_numel]
                        bcast_parts = []
                        cum_lnu = 0
                        for lnu, fnu in _lb_metas:
                            slices = [
                                full_batch[r * _lb_numel + cum_lnu:
                                           r * _lb_numel + cum_lnu + lnu]
                                for r in range(_training_ws)
                            ]
                            bcast_parts.append(torch.cat(slices)[:fnu])
                            cum_lnu += lnu
                        bcast_flat = torch.cat([p.flatten() for p in bcast_parts])
                        bcast_parts.clear()
                        del full_batch

                        tb1 = time.perf_counter()
                        self._xccl_wsync_pg.broadcast(bcast_flat, root=0).wait()
                        t_bcast += time.perf_counter() - tb1
                        del bcast_flat
                    else:
                        del full_batch
                    n_batches += 1

                for param_name, param in sharded_sd.items():
                    if param.is_cpu:
                        param = param.to(self._device)
                    fnu = param.numel()
                    if hasattr(param, "_local_tensor"):
                        local_shard = param._local_tensor.to(torch.bfloat16).contiguous()
                    else:
                        local_shard = param.to(torch.bfloat16).contiguous()
                    lnu = local_shard.numel()

                    # Flush if adding this param would exceed batch threshold
                    if _batch_full_numel > 0 and _batch_full_numel + fnu > _BATCH_MAX_NUMEL:
                        _flush_batched_ag()
                        _lb_parts.clear()
                        _lb_metas.clear()
                        _lb_numel = 0
                        _batch_full_numel = 0

                    _lb_parts.append(local_shard)
                    _lb_numel += lnu
                    _lb_metas.append((lnu, fnu))
                    _batch_full_numel += fnu
                    n_params += 1
                    del param

                if _lb_parts:
                    _flush_batched_ag()

            else:
                # Mode 0 (default): per-param full_tensor() + batched XCCL broadcast.
                #
                # 2-hop mode (use_two_hop=True, default):
                #   Training uses a 2-rank cross PG (rank 0 training + rank 1 vLLM).
                #   broadcast(root=0) sends one cross-Slingshot copy (~2.4s/61GiB at ~25 GB/s).
                #   vLLM rank 1 then distributes to ranks 2-12 via XeLink intra PG (~0.6s).
                #   Total: ~3s vs 38s for flat 13-rank broadcast (12 sequential Slingshot sends).
                #
                # Legacy flat broadcast (use_two_hop=False):
                #   Training rank 0 broadcasts to all vLLM ranks directly in 13-rank PG.

                if self._is_shard_leader:
                    batch_parts: list = []
                    batch_numel = 0

                for param_name, param in sharded_sd.items():
                    if param.is_cpu:
                        param = param.to(self._device)
                    if hasattr(param, "_local_tensor"):
                        param = param.full_tensor()  # collective: all FSDP ranks participate
                    if self._is_shard_leader:
                        param_bf16 = param.to(torch.bfloat16).contiguous()
                        pn = param_bf16.numel()
                        if batch_numel > 0 and batch_numel + pn > _BATCH_MAX_NUMEL:
                            flat = torch.cat([t.flatten() for t in batch_parts])
                            batch_parts = []
                            tb0 = time.perf_counter()
                            self._xccl_wsync_pg.broadcast(flat, root=0).wait()
                            t_bcast += time.perf_counter() - tb0
                            del flat
                            batch_numel = 0
                            n_batches += 1
                        batch_parts.append(param_bf16)
                        batch_numel += pn
                        n_params += 1
                    del param

                if self._is_shard_leader and batch_parts:
                    flat = torch.cat([t.flatten() for t in batch_parts])
                    batch_parts = []
                    tb0 = time.perf_counter()
                    self._xccl_wsync_pg.broadcast(flat, root=0).wait()
                    t_bcast += time.perf_counter() - tb0
                    del flat
                    n_batches += 1

            del sharded_sd

            if not self._production_mode:
                torch.distributed.barrier()
            t_gather = time.perf_counter() - t0  # gather+bcast interleaved

            if self._is_shard_leader:
                # Wait for all manifest threads (return after vLLM applies all params)
                for mt in manifest_threads:
                    mt.join(timeout=600)

                if post_errors:
                    log.error("Rank %d: XCCL streaming sync errors: %s", self.rank, post_errors)
                    self._sync_error = RuntimeError(str(post_errors))

                for client in self._vllm_clients:
                    client.reset_prefix_cache()

                total_gb = sum(e["numel"] for e in tensors_meta) * 2 / 1024**3
                mode_tag = "batched-AG" if _USE_BATCHED_AG else "streaming"
                log.info(
                    "Rank %d: XCCL %s sync: %d params %d batches %.2f GiB in %.1fs "
                    "(ag=%.1fs bcast=%.1fs %.1f GB/s)",
                    self.rank, mode_tag, n_params, n_batches, total_gb,
                    time.perf_counter() - t0, t_ag, t_bcast,
                    total_gb / t_bcast if t_bcast > 0 else 0,
                )
                self._sync_done_event.set()

        device_empty_cache(self._device)

    def _sync_weights_to_vllm_shm(self) -> None:
        """Gather sharded params then async-dispatch to vLLM via POSIX shared memory.

        Faster alternative to _sync_weights_to_vllm (raw bytes file) for large models:

          Phase 1 (synchronous): FSDP full_tensor() gather — same as raw bytes path.
          Phase 2 (async, shard leader only):
            - Allocate a single SharedMemory block (one mmap, one OS call).
            - Copy all tensors in via ctypes.memmove (DDR5 bandwidth, ~0.2s for 6 GB,
              ~1.8s for 62 GB). No Python object allocation — avoids GC pressure.
            - Write metadata JSON (tensor names/shapes/offsets) to /dev/shm.
            - POST metadata path to vLLM load_weights_from_shm.
            - vLLM maps the SAME physical RAM pages zero-copy via frombuffer(shm.buf).
            - After vLLM confirms, unlink the SHM block.

        Total for 62 GB BF16 model:
          Raw bytes: write=24s + HTTP(read=6s+load=?s) = 30s+
          SHM:       copy=1.8s + HTTP(map=0s+to_xpu=1.2s+load=?s) = 3s+
          Savings: ~27s per sync for 31B.

        Enable with config: vllm_weight_sync_method: shm
        """
        import ctypes
        import json
        from multiprocessing.shared_memory import SharedMemory

        t0 = time.perf_counter()
        hf_state_dict = {}

        # Phase 1: FSDP gather (same as raw bytes path)
        if getattr(self, '_use_fsdp1', False) and self._dp_replicate > 1:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
            with FSDP.state_dict_type(self._model, StateDictType.FULL_STATE_DICT):
                full_sd = self._model.state_dict()
            if self._is_shard_leader:
                for param_name, param in full_sd.items():
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.cpu()
            del full_sd
        else:
            sharded_sd = self._model.state_dict()
            for param_name, param in sharded_sd.items():
                if param.is_cpu:
                    param = param.to(self._device)
                if hasattr(param, "_local_tensor"):
                    param = param.full_tensor()
                if self._is_shard_leader:
                    hf_name = self._tune_to_hf_map.get(param_name, param_name)
                    hf_state_dict[hf_name] = param.cpu()
            del sharded_sd

        if not self._production_mode:
            torch.distributed.barrier()

        t_gather = time.perf_counter() - t0

        if self._is_shard_leader:
            n_params = len(hf_state_dict)
            meta_path = "/dev/shm/torchtune/wsync_meta.json"

            self._sync_done_event.clear()
            self._sync_error = None

            def _bg_shm_and_post(state_dict=hf_state_dict):
                try:
                    import numpy as _np

                    # Compute total bytes and build numpy views
                    tensors_meta = []
                    np_arrays = []
                    total_bytes = 0
                    for hf_name, tensor in state_dict.items():
                        if tensor.dtype == torch.bfloat16:
                            np_arr = tensor.view(torch.int16).numpy()
                        else:
                            np_arr = tensor.numpy()
                        if not np_arr.flags['C_CONTIGUOUS']:
                            np_arr = _np.ascontiguousarray(np_arr)
                        nbytes = np_arr.nbytes
                        tensors_meta.append({
                            "name": hf_name,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "offset": total_bytes,
                            "nbytes": nbytes,
                        })
                        np_arrays.append(np_arr)
                        total_bytes += nbytes

                    del state_dict  # free gathered tensors ASAP

                    # Reuse the persistent SHM block if size matches — avoids page-fault
                    # overhead (~4s for 6 GB) caused by OS demand-paging fresh SHM pages.
                    # On first call or size change, allocate a new block and warm it up.
                    shm_name = "torchtune_weights"
                    existing = getattr(self, '_shm_block', None)
                    if existing is not None and existing.size == total_bytes:
                        shm = existing
                        is_new = False
                    else:
                        # Clean up old block (size change or first run)
                        if existing is not None:
                            try:
                                existing.close()
                                existing.unlink()
                            except Exception:
                                pass
                        # Also clean up any stale SHM from a previous process
                        try:
                            stale = SharedMemory(name=shm_name, create=False)
                            stale.close()
                            stale.unlink()
                        except FileNotFoundError:
                            pass
                        shm = SharedMemory(name=shm_name, create=True, size=max(total_bytes, 1))
                        self._shm_block = shm
                        is_new = True

                    # Copy tensors into SHM via ctypes.memmove — no Python object allocation.
                    # On reused blocks, pages are already faulted → DDR5 bandwidth (~30 GB/s).
                    # On new blocks, first-touch faults limit to ~1-2 GB/s (one-time cost).
                    t_copy0 = time.perf_counter()
                    for arr, entry in zip(np_arrays, tensors_meta):
                        dst = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf, entry["offset"]))
                        src = arr.ctypes.data
                        ctypes.memmove(dst, src, entry["nbytes"])
                    del np_arrays
                    t_copy = time.perf_counter() - t_copy0

                    gb = total_bytes / 1024**3
                    log.info(
                        "Rank %d: shm copy done: %d params %.2f GiB in %.1fs (%.1f GB/s) → shm:%s%s",
                        self.rank, n_params, gb, t_copy, gb / t_copy if t_copy > 0 else 0,
                        shm_name, " [new block, page-fault warmup]" if is_new else " [reused]",
                    )

                    # Build and POST metadata to vLLM
                    meta_json = json.dumps({
                        "shm_name": shm_name,
                        "total_bytes": total_bytes,
                        "tensors": tensors_meta,
                    })

                    import requests
                    t_http0 = time.perf_counter()
                    for url in self._vllm_urls:
                        try:
                            r = requests.post(
                                f"{url}/collective_rpc",
                                json={"method": "load_weights_from_shm", "args": [meta_json]},
                                timeout=300,
                            )
                            if r.status_code != 200:
                                log.warning("vLLM shm reload failed (%s): %s %s", url, r.status_code, r.text[:200])
                            else:
                                result = r.json()
                                results = result.get("results", [{}])
                                first = results[0] if results else {}
                                if isinstance(first, dict) and first.get("status") != "ok":
                                    log.warning("vLLM shm reload error (%s): %s", url, first)
                        except Exception as e:
                            log.error("vLLM shm reload HTTP error (%s): %s", url, e)
                    t_http = time.perf_counter() - t_http0

                    for client in self._vllm_clients:
                        client.reset_prefix_cache()

                    log.info(
                        "Rank %d: weight sync shm %d vLLM replica(s): %d params %.1fs total "
                        "(gather=%.1fs [sync] copy=%.1fs http=%.1fs) [shm:%s]",
                        self.rank, len(self._vllm_urls), n_params, time.perf_counter() - t0,
                        t_gather, t_copy, t_http, shm_name,
                    )

                except Exception as e:
                    log.error("Rank %d: shm weight sync failed: %s", self.rank, e, exc_info=True)
                    self._sync_error = e
                finally:
                    # SHM block is kept alive (self._shm_block) for reuse next step.
                    # Cleanup happens in teardown() or on size change above.
                    self._sync_done_event.set()

            t = threading.Thread(target=_bg_shm_and_post, daemon=True, name="weight_sync_shm")
            t.start()
            log.info(
                "Rank %d: weight sync (shm) gather done in %.1fs, async copy+POST started (%d params)",
                self.rank, t_gather, n_params,
            )

        device_empty_cache(self._device)

    def _wait_for_sync_complete(self) -> None:
        """Block until the background weight sync thread finishes.

        Call this immediately before generate_trajectory() to ensure vLLM has
        up-to-date weights. If sync finishes before generation is ready to start,
        wait time is zero. If sync takes longer than the inter-step gap, we wait
        only the remaining time.
        """
        if not self._vllm_weight_sync or not self._is_shard_leader:
            return
        if not hasattr(self, "_sync_done_event"):
            return
        if not self._sync_done_event.is_set():
            t_wait0 = time.perf_counter()
            self._sync_done_event.wait()
            waited = time.perf_counter() - t_wait0
            if waited > 0.05:
                log.info("Rank %d: waited %.1fs for async weight sync to complete", self.rank, waited)
        if self._sync_error is not None:
            log.error("Rank %d: previous async weight sync had an error: %s", self.rank, self._sync_error)
            self._sync_error = None  # reset so training continues

    def generate_trajectory(
        self, input_ids: torch.Tensor, answers: list[str]
    ) -> GRPOTrajectory:
        """
        Generates a trajectory given the current policy model, the reference policy model,
        the reward function, and batch of inputs.
        """
        # Synchronize XPU between steps.
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if not _colocate_vllm_mode:
            device_empty_cache(self._device)
        elif self._vllm_mode == "colocate_sleep" and self._vllm_llm is not None and hasattr(self, '_vllm_is_sleeping') and self._vllm_is_sleeping:
            # Sleep mode: wake weights → sync updated FSDP weights → wake KV cache
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            log.info("Rank %d: waking up vLLM for generation", self.rank)
            t_wake = time.perf_counter()
            # 1. Restore vLLM weight storage (from CPU backup — old weights)
            self._vllm_llm.wake_up(tags=["weights"])
            # 2. Overwrite with updated FSDP weights
            self._sync_colocated_weights()
            # 3. Reallocate KV cache
            self._vllm_llm.wake_up(tags=["kv_cache"])
            self._vllm_is_sleeping = False
            log.info("Rank %d: vLLM wake_up + weight sync completed in %.2fs", self.rank, time.perf_counter() - t_wake)
        elif self._vllm_llm is not None and hasattr(self, '_vllm_kv_cache_shapes'):
            # Non-sleep colocate: reclaim cached training memory, reallocate KV cache.
            # NOTE: skip empty_cache() — leaks UR handles with FSDP.
            import gc
            gc.collect()
            torch.xpu.synchronize()
            torch.distributed.barrier()
            kv_caches = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
            for i, (shape, dtype) in enumerate(self._vllm_kv_cache_shapes):
                kv_caches[i] = torch.zeros(shape, dtype=dtype, device=self._device)
            del self._vllm_kv_cache_shapes
            self._vllm_llm.llm_engine.reset_prefix_cache()

        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        # step 1: generate responses using the current policy (or vLLM)
        _vllm_t0 = time.perf_counter()
        if self._vllm_mode in ("colocate", "colocate_sleep"):
            query_responses = self._generate_with_colocated_vllm(batch_input_ids, context_length)
        elif self._vllm_mode == "server":
            query_responses = self._generate_with_vllm(batch_input_ids, context_length)
        else:
            with local_kv_cache(
                model=self._model,
                batch_size=batch_size * grpo_size,
                device=self._device,
                dtype=self._dtype,
                decoder_max_seq_len=context_length + self._max_generated_tokens,
            ):
                # For HSDP: disable early stopping during generation.
                # With HSDP, different replicate groups process different data
                # and finish at different times. Early stopping via all_reduce
                # on the shard PG works per-node, but then the world-level
                # barrier after generation deadlocks because one node finishes
                # before the other. Always generating max tokens ensures all
                # ranks take the same number of steps (stop_token_mask zeros
                # out tokens past EOS).
                _stop_tokens = (
                    None if self._dp_replicate > 1
                    else self._tokenizer.stop_tokens
                )
                query_responses, _ = generate(
                    model=self._model,
                    prompt=batch_input_ids,
                    max_generated_tokens=self._max_generated_tokens,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    pad_id=self._tokenizer.pad_id,
                    rng=self._rng if self._device.type == "cuda" else None,
                    stop_tokens=_stop_tokens,
                    return_logits=False,
                )

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _vllm_time = time.perf_counter() - _vllm_t0

        # Barrier: all ranks must finish generation before FSDP forward passes.
        # For vLLM server mode, the world broadcast in _generate_with_vllm
        # already synchronizes all ranks. For other modes, use world barrier.
        if self._vllm_mode != "server" and not self._production_mode:
            torch.distributed.barrier()

        # Free vLLM GPU memory to reclaim space for training forward/backward passes.
        if _colocate_vllm_mode and self._vllm_llm is not None:
            if torch.xpu.is_available():
                mem_before = torch.xpu.memory_allocated(self._device) / 1024**3

            if self._vllm_mode == "colocate_sleep":
                # Sleep mode: offload weights to CPU and release all GPU storage
                # (weights + KV cache). This frees maximum memory for training.
                log.info("Rank %d: sleeping vLLM (weights + KV cache) for training", self.rank)
                t_free = time.perf_counter()
                self._vllm_llm.sleep(level=1)
                self._vllm_is_sleeping = True
            else:
                # Non-sleep colocate: only free KV cache, weights stay on GPU
                log.info("Rank %d: freeing vLLM KV cache for training", self.rank)
                t_free = time.perf_counter()
                kv_caches = self._vllm_llm.llm_engine.model_executor.driver_worker.model_runner.kv_caches
                self._vllm_kv_cache_shapes = []
                for i, cache in enumerate(kv_caches):
                    self._vllm_kv_cache_shapes.append((cache.shape, cache.dtype))
                    kv_caches[i] = torch.empty(0, device="cpu")

            if torch.xpu.is_available():
                mem_after = torch.xpu.memory_allocated(self._device) / 1024**3
                log.info("Rank %d: vLLM memory freed in %.1fs (%.2f -> %.2f GiB, freed %.2f GiB)",
                         self.rank, time.perf_counter() - t_free,
                         mem_before, mem_after, mem_before - mem_after)
            else:
                log.info("Rank %d: vLLM memory freed in %.1fs", self.rank, time.perf_counter() - t_free)

        responses = query_responses[:, context_length:].clone()

        # Clamp token IDs to valid vocab range (XPU scatter kernel crashes on OOB)
        vocab_size = getattr(self, '_vocab_size', None)
        if vocab_size is not None and vocab_size > 0:
            oob_mask = responses >= vocab_size
            if oob_mask.any():
                log.warning("Clamping %d OOB token IDs (max=%d, vocab=%d)",
                            oob_mask.sum().item(), responses.max().item(), vocab_size)
                responses = responses.clamp(max=vocab_size - 1)
                query_responses = torch.cat([query_responses[:, :context_length], responses], dim=1)

        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy
        # Chunk the forward pass to avoid OOM with many sequences (Config B).
        # Process forward_batch_size sequences at a time through the model.
        _policy_fwd_t0 = time.perf_counter()
        num_seqs = query_responses.shape[0]
        fwd_bs = self._forward_batch_size
        if fwd_bs >= num_seqs:
            # Single forward pass (no chunking needed)
            log.info("Rank %d: policy forward start (shape=%s)", self.rank, list(query_responses.shape))
            logits = self._model(query_responses, input_pos=position_ids, mask=masks)
            log.info("Rank %d: policy forward done", self.rank)
            logits = logits[:, context_length - 1 :]
            logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
            del logits
        else:
            # Chunked forward pass
            log.info("Rank %d: policy forward start CHUNKED (total=%d, chunk=%d)", self.rank, num_seqs, fwd_bs)
            if self.rank == 0 and self._device.type == "xpu":
                log.info("Rank 0: PRE-policy-fwd memory: alloc=%.2f GiB, resv=%.2f GiB",
                         torch.xpu.memory_allocated() / 1024**3, torch.xpu.memory_reserved() / 1024**3)
            logprobs_chunks = []
            for cs in range(0, num_seqs, fwd_bs):
                ce = min(cs + fwd_bs, num_seqs)
                chunk_logits = self._model(
                    query_responses[cs:ce],
                    input_pos=position_ids[cs:ce],
                    mask=masks[cs:ce],
                )
                if self.rank == 0 and self._device.type == "xpu":
                    log.info("Rank 0: POST-chunk[%d:%d] memory: alloc=%.2f GiB, resv=%.2f GiB",
                             cs, ce, torch.xpu.memory_allocated() / 1024**3, torch.xpu.memory_reserved() / 1024**3)
                chunk_logits = chunk_logits[:, context_length - 1 :]
                logprobs_chunks.append(
                    rlhf.batched_logits_to_logprobs(chunk_logits, responses[cs:ce], self._temperature)
                )
                del chunk_logits
                # NOTE: empty_cache() between forward chunks causes GPU segfaults
                # on XPU — FSDP2 internal handles reference the freed blocks.
                # Instead, use fbs >= grpo_samples to avoid chunking entirely.
                device_empty_cache(self._device)
            logprobs = torch.cat(logprobs_chunks, dim=0)
            del logprobs_chunks
            log.info("Rank %d: policy forward done (chunked)", self.rank)
        device_empty_cache(self._device)

        if self._device.type == "xpu":
            torch.xpu.synchronize()
        _policy_fwd_time = time.perf_counter() - _policy_fwd_t0

        # NOTE: Removed policy→ref defrag. With fbs >= grpo_samples (no chunking),
        # post-policy reserved ~29 GiB leaves enough headroom for ref forward.
        # Each empty_cache() accelerates UR handle leak, so minimize calls.

        # step 2.1 estimate logprobs of the responses using the reference policy
        # Barrier: sync before ref model forward. FSDP2 uses lazy allgather
        # which is itself a synchronization point — skip world barrier for
        # multi-node to avoid CCL broadcast_scaleout failures.
        _ref_fwd_t0 = time.perf_counter()
        log.info("Rank %d: pre-ref forward", self.rank)
        # v79: XCCL SHARD barrier before ref forward (EP case).
        # Root cause of v78 a2a#241 deadlock: after the post-optimizer XCCL SHARD
        # sync (v78), all SHARD members start step N+1 generate() together. But
        # policy fwd has 60 gloo AllToAlls — when token routing imbalance causes
        # ep_rank 0 to process 32k tokens vs ep_ranks 1-3 processing ~0, the 60
        # policy fwd AllToAlls finish but the TRANSITION to ref fwd has no sync
        # in production mode. Fast ep_ranks (1-3) enter ref fwd and start
        # a2a#241 (first ref fwd AllToAll). ep_rank 0 is still processing its
        # policy fwd tokens → never calls a2a#241 → 1800s timeout.
        # Fix: XCCL SHARD all_reduce before ref forward. Forces all 4 SHARD
        # members to rendezvous between policy fwd and ref fwd. ep_ranks 1-3
        # wait here for ep_rank 0 to finish policy fwd before any rank starts
        # ref fwd AllToAlls. No XCCL SHARD collision: the only other XCCL SHARD
        # op in this step (post-optimizer, v78) fired in the PREVIOUS step and
        # is fully complete. Zero gloo AllToAlls are active when this fires.
        if self._expert_parallel_degree > 1 and self._shard_pg is not None:
            _pre_ref_sync = torch.zeros(1, dtype=torch.float32, device=self._device)
            _orig_all_reduce(_pre_ref_sync, op=torch.distributed.ReduceOp.SUM,
                             group=self._shard_pg)
            log.info("Rank %d: EP v79 pre-ref XCCL SHARD sync done", self.rank)
        elif not self._production_mode:
            torch.distributed.barrier()
        if fwd_bs >= num_seqs:
            log.info("Rank %d: ref forward start", self.rank)
            ref_logits = self._ref_model(
                query_responses, input_pos=position_ids, mask=masks
            )
            ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
            ref_logprobs = rlhf.batched_logits_to_logprobs(
                ref_logits, responses, self._temperature
            )
            del ref_logits
        else:
            log.info("Rank %d: ref forward start CHUNKED (total=%d, chunk=%d)", self.rank, num_seqs, fwd_bs)
            ref_logprobs_chunks = []
            for cs in range(0, num_seqs, fwd_bs):
                ce = min(cs + fwd_bs, num_seqs)
                chunk_ref_logits = self._ref_model(
                    query_responses[cs:ce],
                    input_pos=position_ids[cs:ce],
                    mask=masks[cs:ce],
                )
                chunk_ref_logits = rlhf.truncate_sequence_for_logprobs(chunk_ref_logits, context_length)
                ref_logprobs_chunks.append(
                    rlhf.batched_logits_to_logprobs(chunk_ref_logits, responses[cs:ce], self._temperature)
                )
                del chunk_ref_logits
                device_empty_cache(self._device)
            ref_logprobs = torch.cat(ref_logprobs_chunks, dim=0)
            del ref_logprobs_chunks
            log.info("Rank %d: ref forward done (chunked)", self.rank)
        device_empty_cache(self._device)
        if self._device.type == "xpu":
            torch.xpu.synchronize()
        if self._is_rank_zero:
            _alloc_after_ref = torch.xpu.memory_allocated(self._device) / 1e9
            _resv_after_ref = torch.xpu.memory_reserved(self._device) / 1e9
            log.info(
                "Rank 0: post-ref-fwd alloc=%.2f GiB resv=%.2f GiB",
                _alloc_after_ref, _resv_after_ref,
            )
        _ref_fwd_time = time.perf_counter() - _ref_fwd_t0

        log.info(
            "Rank %d: GENTIMING vllm=%.1fs policy_fwd=%.1fs ref_fwd=%.1fs",
            self.rank, _vllm_time, _policy_fwd_time, _ref_fwd_time,
        )

        # step 4. replace tokens after first stop token with padding
        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # Compute rewards
        responses = responses.reshape(batch_size, grpo_size, -1)
        if self._reward_mode == "gene_recall":
            rewards, successes, metadata = gene_recall_batched_rewards(
                self._tokenizer, responses, answers, device=self._device,
                reward_metric=self._gene_reward_metric,
            )
        else:
            rewards, successes, metadata = batched_rewards(
                self._tokenizer, responses, answers, device=self._device
            )
        rewards = rewards.to(self._device)
        successes = successes.to(self._device)

        # Aggregate rewards and successes across reward functions
        rewards = rewards.sum(dim=-1)
        successes = successes.sum(dim=-1)

        # Log sample responses for verification (rank 0 only, first sample)
        if self._is_rank_zero:
            try:
                sample_resp = responses[0, 0]  # first prompt, first sample
                # Decode only non-pad tokens
                non_pad = sample_resp[sample_resp != self._tokenizer.pad_id]
                decoded = self._tokenizer.decode(non_pad.tolist())
                log.info(
                    "SAMPLE_RESPONSE step=%d reward=%.1f success=%.1f answer=%s response=%s",
                    self._steps_run,
                    rewards[0, 0].item(),
                    successes[0, 0].item(),
                    answers[0][:80],
                    decoded[:200],
                )
            except Exception as e:
                log.warning("Could not decode sample response: %s", e)

        advantages = (rewards - rewards.mean(1, keepdim=True)) / (
            rewards.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.reshape(batch_size * grpo_size)
        del responses
        device_empty_cache(self._device)

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        # Use masked_fill_ to avoid boolean index L0 sub-allocation (UR:40 risk).
        logprobs.masked_fill_(response_padding_masks, 1.0)
        ref_logprobs.masked_fill_(response_padding_masks, 1.0)

        return GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=rewards.reshape(batch_size * grpo_size),
            successes=successes.reshape(batch_size * grpo_size),
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
            answers=answers,
        )

    def generate_trajectory_batched(
        self, input_ids: torch.Tensor, answers: list[str]
    ) -> GRPOTrajectory:
        """
        Generates trajectories using forward_batch_size micro-batches.
        """
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_answers = answers[
                    batch_start : batch_start + self._forward_batch_size
                ]
                device_empty_cache(self._device)
                trajectories.append(
                    self.generate_trajectory(batch_input_ids, batch_answers)
                )
                device_empty_cache(self._device)

        # Concatenate all trajectory fields except answers (which is a list of strings)
        concatenated_fields = {}
        for field_name in trajectories[0]._fields:
            if field_name == "answers":
                concatenated_fields[field_name] = []
                for traj in trajectories:
                    concatenated_fields[field_name].extend(traj.answers)
            else:
                concatenated_fields[field_name] = torch.cat(
                    [getattr(traj, field_name) for traj in trajectories]
                )

        return GRPOTrajectory(**concatenated_fields)

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """
        Perform a single GRPO optimization step over a batch of trajectories.
        """
        # Synchronize + flush UR resources before forward pass (prevents
        # UR_RESULT_ERROR_OUT_OF_RESOURCES after ~240 total sequences)
        if self._device.type == "xpu":
            torch.xpu.synchronize()

        # FSDP memory diagnostics: track memory at each phase
        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "pre_forward", log=log)
            # Reset peak stats to measure per-phase peaks
            if self._device.type == "xpu":
                try:
                    torch.xpu.reset_peak_memory_stats()
                except RuntimeError:
                    pass

        # estimate logprobs from the policy at the current optimisation step
        _fwd_t0 = time.perf_counter()

        if self._enable_packing:
            # Pack sequences to eliminate padding waste in forward/backward
            from torchtune.dev.grpo.packing import (
                pack_trajectory_for_training,
                unpack_tensor,
            )
            packed_tokens, packed_positions, packed_masks, bins, actual_lens = (
                pack_trajectory_for_training(
                    trajectory.query_responses,
                    trajectory.position_ids,
                    self._tokenizer.pad_id,
                )
            )
            log.info(
                "Rank %d: grpo_step packed forward start (%d seqs -> %d packs)",
                self.rank, trajectory.query_responses.shape[0], packed_tokens.shape[0],
            )
            packed_logits = self._model(
                packed_tokens, input_pos=packed_positions, mask=packed_masks,
            )
            del packed_tokens, packed_positions, packed_masks
            # Unpack back to per-sequence layout
            pi_logits = unpack_tensor(
                packed_logits, bins, actual_lens,
                num_sequences=trajectory.query_responses.shape[0],
                total_len=trajectory.query_responses.shape[1],
            )
            del packed_logits
        elif (
            os.environ.get("TORCHTUNE_USE_CHUNKED_LOSS") == "1"
            and self._expert_parallel_degree <= 1
        ):
            # Single forward + single backward (pre-acdc7c9f pattern).
            # For non-EP runs where the per-chunk fwd+bwd loop causes excessive
            # activation memory pinning (FSDP2 grad-sync suppression hypothesis).
            # Gated by TORCHTUNE_USE_CHUNKED_LOSS=1 so EP runs use the chunked loop.
            total_seqs = trajectory.query_responses.shape[0]
            grad_scale = max(1, self._gradient_accumulation_steps)

            log.info("Rank %d: single-backward forward start (total=%d seqs)", self.rank, total_seqs)
            _fwd_t0_sb = time.perf_counter()
            pi_logits = self._model(
                trajectory.query_responses,
                input_pos=trajectory.position_ids,
                mask=trajectory.masks,
            )
            pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
            pi_logprobs = rlhf.batched_logits_to_logprobs(
                pi_logits,
                trajectory.query_responses[:, context_length:],
                self._temperature,
                chunk_size=1,
            )
            pi_logprobs.masked_fill_(trajectory.response_padding_masks, 1.0)
            del pi_logits
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _fwd_time_sb = time.perf_counter() - _fwd_t0_sb
            log.info("Rank %d: single-backward forward=%.1fs", self.rank, _fwd_time_sb)

            loss, policy_loss, kl_loss, ratios, clipfrac = self._loss_fn(
                trajectory.logprobs,
                pi_logprobs,
                trajectory.ref_logprobs,
                trajectory.advantages,
                padding_masks=~trajectory.response_padding_masks,
            )

            # MEMPROBE: per-rank L0-truth snapshot before backward.
            try:
                import sys as _sys_sb
                _mp_path_sb = "/lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b"
                if _mp_path_sb not in _sys_sb.path:
                    _sys_sb.path.insert(0, _mp_path_sb)
                from mem_probe import dump_mem as _dump_mem_sb
                _dump_mem_sb(f"PRE-BWD step={self._steps_run} single-bwd")
            except Exception:
                _dump_mem_sb = None

            log.info("Rank %d: single-backward backward start", self.rank)
            _bwd_t0_sb = time.perf_counter()
            # Bypass gloo CPU-AllReduce reduce_scatter patch: XCCL reduce_scatter is
            # correct and fast for non-EP single-node FSDP2 (W-old: 7.2s vs gloo: 130s).
            # The module-level patch routes reduce_scatter_tensor through D2H+gloo+H2D
            # per layer, adding ~2s × 64 layers = 130s to backward for non-EP runs.
            import torch.distributed as _tdist_sb_fix
            _rsc_patch_saved = _tdist_sb_fix.reduce_scatter_tensor
            _tdist_sb_fix.reduce_scatter_tensor = _orig_reduce_scatter_tensor
            try:
                try:
                    (loss / grad_scale).backward()  # loss passed to GRPOStats is UNSCALED
                except Exception as _bwd_exc_sb:
                    try:
                        if _dump_mem_sb is not None:
                            _dump_mem_sb(f"BWD-FAIL step={self._steps_run} single-bwd exc={type(_bwd_exc_sb).__name__}")
                    except Exception:
                        pass
                    log.error("Rank %d: SINGLE-BWD FAILED step=%d exc=%r", self.rank, self._steps_run, _bwd_exc_sb)
                    raise
            finally:
                _tdist_sb_fix.reduce_scatter_tensor = _rsc_patch_saved
            if self._device.type == "xpu":
                torch.xpu.synchronize()
            _bwd_total = time.perf_counter() - _bwd_t0_sb
            try:
                if _dump_mem_sb is not None:
                    _dump_mem_sb(f"POST-BWD step={self._steps_run} single-bwd")
            except Exception:
                pass
            log.info("Rank %d: single-backward backward=%.1fs", self.rank, _bwd_total)
            _fwd_time = _fwd_time_sb

        else:
            # Chunked training forward+backward: process self._forward_batch_size
            # sequences at a time. Each chunk gets its own backward() call so
            # autograd only holds one chunk's activation graph in memory at once.
            # Gradients accumulate across chunks before the optimizer step.
            # Mathematically equivalent to a single forward+backward over all seqs.
            total_seqs = trajectory.query_responses.shape[0]
            fwd_bs = self._forward_batch_size
            num_fwd_chunks = (total_seqs + fwd_bs - 1) // fwd_bs
            grad_scale = num_fwd_chunks * max(1, self._gradient_accumulation_steps)

            # FSDP2 reduce-scatter suppression for multi-chunk backward.
            # With num_fwd_chunks > 1, each backward() call would normally
            # fire reduce-scatter (AllReduce for FSDP1). That's num_fwd_chunks×
            # the communication cost. Suppress for all but the final chunk:
            # gradients accumulate locally on each rank; final chunk fires once.
            # Use set_requires_gradient_sync() for FSDP2, no_sync() for FSDP1.
            _use_fsdp2_grad_sync = (
                num_fwd_chunks > 1
                and hasattr(self._model, 'set_requires_gradient_sync')
                and not self._use_fsdp1
            )
            _use_fsdp1_no_sync = (
                num_fwd_chunks > 1
                and self._use_fsdp1
                and hasattr(self._model, 'no_sync')
            )

            _chunk_losses, _chunk_policy_losses, _chunk_kl_losses = [], [], []
            _chunk_ratios, _chunk_clipfracs, _chunk_pi_logprobs = [], [], []
            _bwd_total = 0.0

            for _cs in range(0, total_seqs, fwd_bs):
                _is_last_chunk = (_cs + fwd_bs >= total_seqs)
                _ce = min(_cs + fwd_bs, total_seqs)
                if self._device.type == "xpu" and self._is_rank_zero:
                    _pre_fwd_alloc = torch.xpu.memory_allocated() / 1024**3
                    _pre_fwd_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: PRE-train-fwd[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce, _pre_fwd_alloc, _pre_fwd_resv,
                    )
                log.info("Rank %d: grpo_step chunk[%d:%d] fwd", self.rank, _cs, _ce)
                _c_logits = self._model(
                    trajectory.query_responses[_cs:_ce],
                    input_pos=trajectory.position_ids[_cs:_ce],
                    mask=trajectory.masks[_cs:_ce],
                )
                _c_logits = rlhf.truncate_sequence_for_logprobs(_c_logits, context_length)
                _c_pi_lp = rlhf.batched_logits_to_logprobs(
                    _c_logits,
                    trajectory.query_responses[_cs:_ce, context_length:],
                    self._temperature,
                    chunk_size=1,
                )
                # Use masked_fill_ instead of boolean index assignment.
                # Boolean indexing triggers L0 sub-allocation (gather/scatter)
                # while FSDP storage is live, causing UR:40 handle exhaustion.
                _c_pi_lp.masked_fill_(trajectory.response_padding_masks[_cs:_ce], 1.0)
                del _c_logits
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                if self._device.type == "xpu" and self._is_rank_zero:
                    _post_fwd_alloc = torch.xpu.memory_allocated() / 1024**3
                    _post_fwd_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: POST-train-fwd[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce, _post_fwd_alloc, _post_fwd_resv,
                    )

                _c_loss, _c_pol, _c_kl, _c_rat, _c_clip = self._loss_fn(
                    trajectory.logprobs[_cs:_ce],
                    _c_pi_lp,
                    trajectory.ref_logprobs[_cs:_ce],
                    trajectory.advantages[_cs:_ce],
                    padding_masks=~trajectory.response_padding_masks[_cs:_ce],
                )

                # Pre-backward memory logging (no empty_cache — leaks UR handles).
                # With expert_cpu_offload=True, base resv is ~1.2 GiB so no
                # need to return cached blocks to L0 before backward.
                if _cs == 0 and self._device.type == "xpu":
                    torch.xpu.synchronize()
                    if self._is_rank_zero:
                        _pre_bwd_alloc = torch.xpu.memory_allocated() / 1024**3
                        _pre_bwd_resv = torch.xpu.memory_reserved() / 1024**3
                        log.info(
                            "Rank 0: pre-bwd (no empty_cache): alloc=%.2f GiB, resv=%.2f GiB",
                            _pre_bwd_alloc, _pre_bwd_resv,
                        )
                elif self._device.type == "xpu" and self._is_rank_zero:
                    _pre_bwd_alloc = torch.xpu.memory_allocated() / 1024**3
                    _pre_bwd_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: PRE-backward[%d:%d] alloc=%.2f GiB, resv=%.2f GiB",
                        _cs, _ce, _pre_bwd_alloc, _pre_bwd_resv,
                    )
                # v84: CPU-bounce AllToAll (via gloo) bypasses XCCL/OFI/ze_handle_manager.
                # v83 pre-backward XCCL sync removed — CCL worker drain no longer needed
                # since AllToAll no longer uses XCCL at all.
                log.info("Rank %d: chunk[%d:%d] backward start", self.rank, _cs, _ce)
                _bwd_chunk_t0 = time.perf_counter()
                # MEMPROBE v1: per-rank L0-truth memory snapshot before backward.
                try:
                    import sys as _sys, os as _os
                    _mp_path = "/lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b"
                    if _mp_path not in _sys.path:
                        _sys.path.insert(0, _mp_path)
                    from mem_probe import dump_mem as _dump_mem
                    _dump_mem(f"PRE-BWD step={self._steps_run} chunk[{_cs}:{_ce}]")
                except Exception as _mp_e:
                    log.warning("Rank %d: mem_probe import/PRE-BWD failed: %r", self.rank, _mp_e)
                    _dump_mem = None
                # v111 diag: log token dispatch counts for all ranks (first step only).
                # _ep_dispatch is partial(ep_instance._token_dispatch, ...) — access ep_instance
                # via _ep_dispatch.func.__self__ (bound method's __self__).
                if self._steps_run == 0 and _cs == 0 and self._expert_parallel_degree > 1:
                    from torchtune.modules.moe.moe import MoE as _MoE
                    _total_recv = 0
                    _total_send = 0
                    _n_layers_found = 0
                    for _m in self._model.modules():
                        if not isinstance(_m, _MoE):
                            continue
                        _ep_disp = getattr(_m, '_ep_dispatch', None)
                        if _ep_disp is None:
                            continue
                        # partial.func = bound method; .func.__self__ = ExpertParallel instance
                        _ep_inst = getattr(getattr(_ep_disp, 'func', None), '__self__', None)
                        if _ep_inst is None:
                            continue
                        _splits_out = getattr(_ep_inst, '_output_splits', None)
                        _splits_in = getattr(_ep_inst, '_input_splits', None)
                        if _splits_out is not None:
                            _total_recv += sum(_splits_out)
                            _n_layers_found += 1
                        if _splits_in is not None:
                            _total_send += sum(_splits_in)
                    log.info(
                        "Rank %d: EP TOKEN LOAD step=0 total_recv=%d total_send=%d "
                        "layers=%d (via _ep_dispatch.func.__self__)",
                        self.rank, _total_recv, _total_send, _n_layers_found,
                    )
                # v59: EP mode — all FSDPParamGroups have reduce_grads=False permanently.
                # Do NOT call set_requires_gradient_sync: it would re-enable reduce_grads
                # on the last chunk, undoing our v59 suppression and triggering XCCL.
                # Post-backward gloo AllReduce in _ep_post_backward_grad_sync handles sync.
                # Non-EP path: standard grad accumulation (suppress all but last chunk).
                if not self._fsdp2_param_groups_meta and _use_fsdp2_grad_sync and \
                        self._expert_parallel_degree <= 1:
                    # Non-EP FSDP2: standard grad accumulation (suppress all but last chunk).
                    self._model.set_requires_gradient_sync(_is_last_chunk)
                try:
                    if _use_fsdp1_no_sync and not _is_last_chunk:
                        with self._model.no_sync():
                            (_c_loss / grad_scale).backward()
                    else:
                        (_c_loss / grad_scale).backward()
                except Exception as _bwd_exc:
                    try:
                        if _dump_mem is not None:
                            _dump_mem(f"BWD-FAIL step={self._steps_run} chunk[{_cs}:{_ce}] exc={type(_bwd_exc).__name__}")
                    except Exception:
                        pass
                    log.error("Rank %d: BACKWARD FAILED step=%d chunk[%d:%d] exc=%r",
                              self.rank, self._steps_run, _cs, _ce, _bwd_exc)
                    raise
                # MEMPROBE v1: per-rank L0-truth snapshot after backward returns.
                try:
                    if _dump_mem is not None:
                        _dump_mem(f"POST-BWD step={self._steps_run} chunk[{_cs}:{_ce}]")
                except Exception:
                    pass
                # v110 diag: log immediately after backward() returns (before xpu.sync).
                log.info(
                    "Rank %d: chunk[%d:%d] backward() returned (pre-sync), elapsed=%.1fs",
                    self.rank, _cs, _ce, time.perf_counter() - _bwd_chunk_t0,
                )
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _bwd_total += time.perf_counter() - _bwd_chunk_t0

                _chunk_losses.append(_c_loss.detach())
                _chunk_policy_losses.append(_c_pol.detach() if torch.is_tensor(_c_pol) else _c_pol)
                _chunk_kl_losses.append(_c_kl.detach() if torch.is_tensor(_c_kl) else _c_kl)
                _chunk_ratios.append(_c_rat.detach())
                _chunk_clipfracs.append(_c_clip.detach() if torch.is_tensor(_c_clip) else _c_clip)
                _chunk_pi_logprobs.append(_c_pi_lp.detach())

            if self._device.type == "xpu":
                torch.xpu.synchronize()

            # EP v75 post-backward grad sync via XCCL dp_replicate group.
            # With reduce_grads=False on all FSDPParamGroups, FSDP2 did NOT fire
            # reduce_scatter during backward. All param grads are local. Now sync
            # all param grads across dp_replicate via XCCL all_reduce (AVG).
            #
            # v71-v74 all failed because any sync requiring ALL 12 ranks deadlocks:
            #   fast EP groups finish backward first → their ranks try to sync →
            #   slow EP group ranks are still in gloo SHARD AllToAll → can't join
            #   the 12-rank sync → mutual deadlock → 1800s timeout.
            #
            # v75 fix: use XCCL all_reduce on dp_replicate group (3 ranks) only.
            #   dp_replicate pairs (e.g. (0,4),(0,8),(4,8)) are DISJOINT from
            #   dp_shard/SHARD pairs (within {0,1,2,3},{4,5,6,7},{8,9,10,11}).
            #   XCCL REP all_reduce operates on a different rank-pair set than
            #   gloo SHARD AllToAll → zero interference, no pre-sync needed.
            #   When slow EP group {0,1,2,3} finishes backward, ranks 0,1,2,3
            #   each call XCCL all_reduce on their respective REP groups:
            #     rank 0 → REP {0,4,8}  (pairs (0,4),(0,8) — disjoint from SHARD)
            #     rank 1 → REP {1,5,9}  (pairs (1,5),(1,9) — disjoint from SHARD)
            #   Since fast ranks 4,8 (etc.) are already waiting in their XCCL
            #   all_reduce for this call, the all_reduce completes immediately.
            if self._expert_parallel_degree > 1 and self._dp_replicate > 1:
                _grad_sync_t0 = time.perf_counter()
                _n_synced = _ep_post_backward_grad_sync_xccl(
                    self._model, self._dp_replicate
                )
                _grad_sync_time = time.perf_counter() - _grad_sync_t0
                log.info(
                    "Rank %d: EP v75 XCCL post-bwd grad sync: %d params in %.2fs",
                    self.rank, _n_synced, _grad_sync_time,
                )

            _fwd_time = time.perf_counter() - _fwd_t0 - _bwd_total
            log.info("Rank %d: grpo_step forward=%.1fs", self.rank, _fwd_time)
            log.info("Rank %d: backward=%.1fs", self.rank, _bwd_total)

            loss = torch.stack(_chunk_losses).mean()
            policy_loss = (
                torch.stack(_chunk_policy_losses).mean()
                if torch.is_tensor(_chunk_policy_losses[0])
                else sum(_chunk_policy_losses) / len(_chunk_policy_losses)
            )
            kl_loss = (
                torch.stack(_chunk_kl_losses).mean()
                if torch.is_tensor(_chunk_kl_losses[0])
                else sum(_chunk_kl_losses) / len(_chunk_kl_losses)
            )
            ratios = torch.stack(_chunk_ratios).mean()
            clipfrac = (
                torch.stack(_chunk_clipfracs).mean()
                if torch.is_tensor(_chunk_clipfracs[0])
                else sum(_chunk_clipfracs) / len(_chunk_clipfracs)
            )
            pi_logprobs = torch.cat(_chunk_pi_logprobs, dim=0)

        if self._fsdp_diagnostics and self._is_rank_zero:
            training.log_fsdp_memory_per_phase(self._device, "post_backward", log=log)

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return GRPOStats(
            loss,  # already unscaled (chunks averaged, not summed)
            policy_loss,
            kl_loss,
            ratios,
            clipfrac,
            approx_policy_kls,
            None,  # metadata
        )

    def train(self) -> None:
        """
        The core training loop with gradient accumulation and no_sync() support.
        """
        # clean up before training begins
        try:
            training.cleanup_before_training()
        except RuntimeError:
            pass

        # v86b: Pre-register XPU L0 root blocks with CCL ze_handle_manager before training.
        #
        # v86 finding: 64 MiB × ~576 chunks hit UR_RESULT_ERROR_OUT_OF_RESOURCES (UR:40) —
        # too many concurrent L0 memory objects. Fix: use 2 GiB chunks (~18 total), one
        # XCCL all_reduce per chunk to register each root block individually.
        #
        # Root cause of SIGABRT: At peak HBM during AC recompute backward, FSDP2 AllGather
        # for expert params forces a NEW L0 root block allocation (beyond pool). CCL's
        # ze_handle_manager doesn't know about new root blocks → ZE_MEMORY_TYPE_UNKNOWN → crash.
        #
        # Fix: allocate 2 GiB chunks until pool is at expected peak (~50.5 GiB target),
        # run one XCCL all_reduce on a VIEW of each chunk (same memory → same root block)
        # to register each root block with ze_handle_manager, then free all chunks.
        # During training, no new root blocks needed → no crash.
        if self._device.type == "xpu" and self._expert_parallel_degree > 1:
            # v95: No XPU warmup needed — CPU-bounce AllToAll (gloo P2P) eliminates
            # ze_handle_manager by removing all XPU AllToAll calls entirely.
            #
            # v87-v92 warmup history (all exhausted):
            #   v87: 50.50 GiB pool → +3.38 GiB during gen → crash at backward
            #   v90: 53.50 GiB pool → +3.38 GiB during gen → crash at backward
            #   v91: 58.00 GiB pool → +3.38 GiB during gen → crash at backward
            #   v92: 57.35 GiB (ONE 45 GiB chunk) → +3.38 GiB during gen → crash at backward
            # Root cause: CCL's AllToAll internal workspace is CCL-internal (not pre-registerable).
            # v93: XCCL doesn't support CPU tensors — failed.
            # v94: gloo all_to_all list-API not supported — failed.
            # v95: gloo P2P (batch_isend_irecv) — correct approach.
            log.info(
                "Rank %d: v95 — gloo P2P CPU-bounce AllToAll active, no XPU warmup needed. "
                "cur_resv=%.2f GiB",
                self.rank, torch.xpu.memory_reserved() / 1024**3,
            )

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        grad_norm = None

        training_completed = False
        self._profiler.start()
        for curr_epoch in range(self._epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                # Start tracking memory for active steps
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=True)

                tokens = batch["tokens"]
                answers = batch["answers"]
                tokens = tokens.to(self._device)

                _, context_length = tokens.shape

                # NOTE: No empty_cache() here. With PYTORCH_ALLOC_CONF=expandable_segments:True,
                # the allocator handles fragmentation internally without UR handle leaks.

                # Memory diagnostics before each step
                if self._device.type == "xpu" and self.rank == 0:
                    _alloc = torch.xpu.memory_allocated() / 1024**3
                    _resv = torch.xpu.memory_reserved() / 1024**3
                    log.info("Rank 0: PRE-STEP %d memory: allocated=%.2f GiB, reserved=%.2f GiB",
                             self._steps_run, _alloc, _resv)

                # Wait for the previous step's async weight sync to complete before
                # generating — ensures vLLM has up-to-date weights. If sync finished
                # during the previous step's compute/gather (typical case), wait=0.
                if self._vllm_mode == "server" and self._vllm_weight_sync:
                    self._wait_for_sync_complete()

                _step_t0 = time.perf_counter()
                trajectory = self.generate_trajectory_batched(tokens, answers)
                if self._device.type == "xpu":
                    torch.xpu.synchronize()
                _gen_time = time.perf_counter() - _step_t0
                if not self._production_mode:
                    torch.distributed.barrier()

                if self._device.type == "xpu" and self._is_rank_zero:
                    _alloc = torch.xpu.memory_allocated() / 1024**3
                    _resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank 0: post-gen memory: alloc=%.2f GiB, resv=%.2f GiB",
                        _alloc, _resv,
                    )

                grpo_stats: list[GRPOStats] = []
                _grpo_t0 = time.perf_counter()

                for _ in range(self._ppo_epochs):
                    total_samples = trajectory.query_responses.shape[0]

                    if self._gradient_accumulation_steps > 1:
                        # Gradient accumulation: disable gradient sync for
                        # non-final micro-batches.
                        # FSDP2 uses set_requires_gradient_sync()
                        # FSDP1 (HSDP) uses no_sync() context manager
                        micro_batch_size = total_samples // self._gradient_accumulation_steps
                        for ga_step in range(self._gradient_accumulation_steps):
                            start_idx = ga_step * micro_batch_size
                            end_idx = start_idx + micro_batch_size
                            micro_trajectory = _slice_trajectory(
                                trajectory, start_idx, end_idx
                            )

                            is_last = (ga_step == self._gradient_accumulation_steps - 1)
                            if hasattr(self._model, 'set_requires_gradient_sync'):
                                # FSDP2 path
                                self._model.set_requires_gradient_sync(is_last)
                                step_stats = self.grpo_step(
                                    micro_trajectory, context_length
                                )
                            elif not is_last and hasattr(self._model, 'no_sync'):
                                # FSDP1 path
                                with self._model.no_sync():
                                    step_stats = self.grpo_step(
                                        micro_trajectory, context_length
                                    )
                            else:
                                step_stats = self.grpo_step(
                                    micro_trajectory, context_length
                                )
                            grpo_stats.append(step_stats)

                            # Memory cleanup between GA steps — gc only,
                            # skip empty_cache (leaks UR handles with FSDP)
                            if not is_last and self._device.type == "xpu":
                                import gc
                                gc.collect()
                    else:
                        # No gradient accumulation — single step.
                        # NOTE: empty_cache() is now called INSIDE grpo_step before
                        # the first backward chunk (after training forward). This is
                        # more effective because the training forward itself pushes
                        # resv from 11.9 → 18.3 GiB, and we need to reclaim that
                        # before backward can allocate recompute + AllGather buffers.
                        step_stats = self.grpo_step(trajectory, context_length)
                        grpo_stats.append(step_stats)

                    # Sync device before timing grad clip
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _grpo_time = time.perf_counter() - _grpo_t0

                    # v59: ALL grad averaging (expert + non-expert) is now done inside
                    # grpo_step() via _ep_post_backward_grad_sync() using gloo AllReduce.
                    # reduce_grads=False on all FSDPParamGroups prevents FSDP2 from firing
                    # reduce_scatter during backward. The post-backward pass does a single
                    # sequential gloo AllReduce over all params in model.parameters() order,
                    # ensuring consistent ordering across dp_replicate ranks.
                    # No separate expert grad averaging needed here.

                    _clip_t0 = time.perf_counter()
                    if self._clip_grad_norm is not None:
                        if self._expert_parallel_degree > 1:
                            # EP mixes two DTensor meshes: non-expert params on dp_mesh
                            # (2D: dp_replicate×dp_shard) and expert params on ep_mesh
                            # (1D: dp_shard). torch.nn.utils.clip_grad_norm_ calls
                            # torch.stack on per-param norm DTensors, which fails when
                            # the tensors have different meshes.
                            #
                            # v77: compute both non-EP and EP norms purely locally.
                            # v76 root cause: clip_grad_norm_(_non_ep_params, inf) on FSDP2
                            # DTensor params triggers an XCCL all_reduce on self._shard_pg
                            # internally (for DTensor norm aggregation). The subsequent
                            # explicit _orig_all_reduce(_ep_norm_sq_xpu, group=self._shard_pg)
                            # is then a SECOND XCCL collective on the same communicator.
                            # With token routing imbalance (ep_rank 0: 32k tokens, ep_ranks 2,3:
                            # 0 tokens), SHARD groups finish backward at different times → the
                            # two sequential XCCL ops on shard_pg arrive at different clock offsets
                            # across the 4 ranks → communicator sequence mismatch → 1800s timeout.
                            # Fix: compute all norms manually (local tensors only, no distributed
                            # collectives in the norm computation). Each rank computes its local
                            # approximation of the global norm and applies clip_coef locally.
                            # This is already a valid approximation for FSDP2 sharded grads.
                            from torchtune.modules.moe.experts import GroupedExperts
                            _ep_param_ids = set()
                            _ep_params = []
                            for _mn, _mm in self._model.named_modules():
                                if _mn.endswith(".experts") and isinstance(_mm, GroupedExperts):
                                    for _p in _mm.parameters(recurse=False):
                                        _ep_param_ids.add(id(_p))
                                        _ep_params.append(_p)
                            _non_ep_params = [
                                _p for _p in self._model.parameters()
                                if id(_p) not in _ep_param_ids
                            ]
                            # Compute local norm_sq for non-EP params (no XCCL, no DTensor norm).
                            _non_ep_norm_sq_val = 0.0
                            for _p in _non_ep_params:
                                if _p.grad is not None:
                                    _g = _p.grad
                                    if hasattr(_g, '_local_tensor'):
                                        _g = _g._local_tensor
                                    _non_ep_norm_sq_val += float(_g.float().norm().item() ** 2)
                            # Compute local norm_sq for EP params.
                            _ep_norm_sq_val = 0.0
                            for _p in _ep_params:
                                if _p.grad is not None:
                                    _g = _p.grad
                                    if hasattr(_g, '_local_tensor'):
                                        _g = _g._local_tensor
                                    _ep_norm_sq_val += float(_g.float().norm().item() ** 2)
                            # Local total norm (no cross-rank all_reduce — see v77 comment above).
                            _total_norm_f = (_non_ep_norm_sq_val + _ep_norm_sq_val) ** 0.5
                            _max_norm_f = float(self._clip_grad_norm)
                            _clip_coef = _max_norm_f / max(_total_norm_f, 1e-6)
                            if _clip_coef < 1.0:
                                for _p in self._model.parameters():
                                    if _p.grad is not None:
                                        _g = _p.grad
                                        if hasattr(_g, '_local_tensor'):
                                            _g._local_tensor.detach().mul_(_clip_coef)
                                        else:
                                            _g.detach().mul_(_clip_coef)
                            grad_norm = torch.tensor(_total_norm_f, device=self._device)
                        else:
                            # Non-EP path: avoid torch.nn.utils.clip_grad_norm_ on FSDP2
                            # DTensor params (its internal XCCL all_reduce on shard_pg
                            # deadlocks with vLLM L0/fabric usage when colocated).
                            # ALSO avoid per-param .item() — that's ~700 D2H syncs which
                            # also deadlocks with concurrent vLLM activity (Test D
                            # 2026-04-22). Compute the norm fully on-device with a
                            # single .item() at the end.
                            _local_norm_sq = torch.zeros((), device=self._device, dtype=torch.float32)
                            _grads_to_clip = []
                            for _p in self._model.parameters():
                                if _p.grad is not None:
                                    _g = _p.grad
                                    if hasattr(_g, '_local_tensor'):
                                        _g = _g._local_tensor
                                    _local_norm_sq = _local_norm_sq + _g.float().pow(2).sum()
                                    _grads_to_clip.append(_p.grad)
                            grad_norm = _local_norm_sq.sqrt()
                            _max_norm = float(self._clip_grad_norm)
                            _clip_coef = (_max_norm / (grad_norm + 1e-6)).clamp(max=1.0)
                            for _g in _grads_to_clip:
                                if hasattr(_g, '_local_tensor'):
                                    _g._local_tensor.detach().mul_(_clip_coef)
                                else:
                                    _g.detach().mul_(_clip_coef)
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _clip_time = time.perf_counter() - _clip_t0

                    if not self._production_mode:
                        torch.distributed.barrier()

                    # Synchronize before optimizer — no empty_cache in
                    # colocate mode (it hangs with vLLM engines).
                    if _colocate_vllm_mode and self._device.type == "xpu":
                        log.info("Rank %d: pre-optimizer sync", self.rank)
                        torch.xpu.synchronize()

                    _opt_t0 = time.perf_counter()
                    log.info("Rank %d: optimizer.step()", self.rank)
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    if self._device.type == "xpu":
                        torch.xpu.synchronize()
                    _opt_time = time.perf_counter() - _opt_t0
                    log.info("Rank %d: optimizer done", self.rank)

                    if self._fsdp_diagnostics and self._is_rank_zero:
                        training.log_fsdp_memory_per_phase(self._device, "post_optimizer", log=log)

                    # Remove per-layer hooks after first step to avoid noise
                    if self._fsdp_diagnostics and self._steps_run == 0 and self._layer_mem_hooks:
                        for h in self._layer_mem_hooks:
                            h.remove()
                        log.info("Removed per-layer memory hooks after step 0")
                        self._layer_mem_hooks = []

                    if not self._production_mode or self._expert_parallel_degree > 1:
                        # With EP, ranks finish optimizer at different times (ep_mesh
                        # AllGather costs differ by replica group). Without this barrier,
                        # rank 0 exits the training loop and starts Python teardown of
                        # FSDP2/ZE objects while other ranks are still in optimizer.step(),
                        # causing L0 use-after-free (SIGABRT in drm_neo.cpp:426).
                        #
                        # v78: replace dist.barrier() with XCCL SHARD all_reduce.
                        # dist.barrier() on the default (12-rank) gloo PG shares TCP
                        # pairs with SHARD gloo AllToAll → concurrent ops → deadlock
                        # (same root cause as v72 gloo global barrier deadlock).
                        #
                        # Root cause of v77 step-boundary deadlock:
                        #   EP token routing imbalance: ep_rank 0 (ranks 0,4,8) processes
                        #   ~32k tokens → backward takes 20+ seconds longer than ep_ranks
                        #   1-3 (~0-500 tokens). After XCCL REP grad_sync, fast ep_ranks
                        #   (1-3) finish optimizer and start step N+1 forward → call gloo
                        #   SHARD AllToAll at a2a#241. ep_rank 0 is still in optimizer →
                        #   never calls a2a#241 gloo AllToAll → 1800s timeout.
                        #
                        # Fix: XCCL SHARD all_reduce (4-rank) after optimizer.
                        #   - Uses self._shard_pg (XCCL, Slingshot fabric — not TCP).
                        #   - Forces all 4 SHARD members to meet here before step N+1.
                        #   - Safe: v77 removed all other XCCL SHARD calls post-backward,
                        #     so this is the ONLY XCCL SHARD collective in this phase.
                        #     No sequence mismatch possible.
                        #   - ep_ranks 1,2,3 wait for ep_rank 0 here → all proceed
                        #     to step N+1 generate() together → synchronized → no race.
                        if self._expert_parallel_degree > 1 and self._shard_pg is not None:
                            _ep_sync_t = torch.zeros(1, dtype=torch.float32, device=self._device)
                            _orig_all_reduce(_ep_sync_t, op=torch.distributed.ReduceOp.SUM,
                                             group=self._shard_pg)
                            log.info("Rank %d: EP v78 post-optimizer XCCL SHARD sync done", self.rank)
                        else:
                            torch.distributed.barrier()

                    self.global_step += 1

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                # Sync updated weights to vLLM (after all ppo_epochs)
                # For colocate_sleep, sync happens during wake_up in generate_trajectory
                if self._vllm_mode == "colocate":
                    self._sync_colocated_weights()
                elif self._vllm_mode == "server" and self._vllm_weight_sync:
                    if self._steps_run % self._vllm_weight_sync_interval == 0:
                        self._sync_weights_to_vllm()

                # Stop tracking memory
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and supports_memory_stats(self._device)
                ):
                    device_record_memory_history(self._device, enabled=None)

                _step_time = time.perf_counter() - _step_t0
                if self._is_rank_zero:
                    log.info(
                        "TIMING step=%d  total=%.1fs  gen=%.1fs  grpo=%.1fs  clip=%.1fs  opt=%.1fs  other=%.1fs",
                        self._steps_run, _step_time, _gen_time, _grpo_time,
                        _clip_time, _opt_time,
                        _step_time - _gen_time - _grpo_time - _clip_time - _opt_time,
                    )

                self._steps_run += 1
                if self._steps_run % self._log_every_n_steps == 0:
                    extra_metrics = {}
                    extra_metrics["lr"] = get_lr(self._optimizer)
                    if grad_norm is not None:
                        extra_metrics["grad_norm"] = grad_norm

                    # Concatenate GRPOStats fields properly
                    concatenated_stats = {}
                    for field_name in grpo_stats[0]._fields:
                        if field_name == "metadata":
                            concatenated_stats[field_name] = None
                        else:
                            concatenated_stats[field_name] = torch.stack(
                                [getattr(stat, field_name) for stat in grpo_stats]
                            )

                    self.log_metrics(
                        trajectory,
                        GRPOStats(**concatenated_stats),
                        **extra_metrics,
                    )

                self.cleanup_after_step(trajectory, grpo_stats)

                # NOTE: No empty_cache() between steps. With PYTORCH_ALLOC_CONF=
                # expandable_segments:True, the allocator avoids fragmentation by
                # expanding existing segments in-place. This avoids the UR handle
                # leak that empty_cache() causes with FSDP on XPU.
                if self._device.type == "xpu" and not self.fsdp_cpu_offload:
                    torch.xpu.synchronize()
                    _mem_alloc = torch.xpu.memory_allocated() / 1024**3
                    _mem_resv = torch.xpu.memory_reserved() / 1024**3
                    log.info(
                        "Rank %d: between-step memory: allocated=%.2f GiB, "
                        "reserved=%.2f GiB, gap=%.2f GiB",
                        self.rank, _mem_alloc, _mem_resv, _mem_resv - _mem_alloc,
                    )

                # Periodic evaluation on held-out data
                if (self._eval_enabled and self._steps_run > 0 and
                        self._steps_run % self._eval_every_n_steps == 0):
                    self.run_eval()

                # Step-based checkpointing
                if (self._save_every_n_steps is not None and self._steps_run > 0 and
                        self._steps_run % self._save_every_n_steps == 0):
                    try:
                        self.save_checkpoint(curr_epoch)
                    except Exception as e:
                        utils.log_rank_zero(
                            log,
                            f"WARNING: Step checkpoint save failed: {e}. "
                            "Continuing training.",
                        )

                self._profiler.step()

                pbar.update(1)

                if self._steps_run == self._total_steps:
                    # Final eval (skip if periodic eval already ran this step)
                    if (self._eval_enabled and
                            self._steps_run % self._eval_every_n_steps != 0):
                        self.run_eval()
                    if self._save_final_checkpoint:
                        try:
                            self.save_checkpoint(curr_epoch)
                        except Exception as e:
                            utils.log_rank_zero(
                                log,
                                f"WARNING: Final checkpoint save failed: {e}. "
                                "Training results are still valid.",
                            )
                    training_completed = True
                    break

            self._epochs_run += 1
            if self._epochs_run % self._save_every_n_epochs == 0:
                try:
                    self.save_checkpoint(curr_epoch)
                except Exception as e:
                    utils.log_rank_zero(
                        log,
                        f"WARNING: Epoch checkpoint save failed: {e}. "
                        "Continuing training.",
                    )
            if training_completed:
                if self._expert_parallel_degree > 1:
                    # Sync all ranks before teardown. Without this, fast ranks may start
                    # Python GC of FSDP2/ZE objects while slow ranks are still in
                    # train(), causing L0 use-after-free (SIGABRT in drm_neo.cpp:426).
                    # NOTE: destroy_process_group() doesn't help — nested ep_mesh/dp_mesh
                    # groups fail when the world group is destroyed first. The SIGABRT
                    # is a known L0 driver issue with simultaneous ZE context destruction
                    # by 12 processes; it's cosmetic (METRICS are always logged first).
                    torch.distributed.barrier()
                return

        self._profiler.stop()

    def log_metrics(
        self, trajectory: GRPOTrajectory, grpo_stats: GRPOStats, **extras
    ) -> None:
        rewards = trajectory.rewards.mean()
        # HSDP: skip world-level reduce to avoid mixing world PG with FSDP1
        # sub-PGs on XCCL. Each replicate group processes the same model with
        # different data, so rank 0's local metrics are representative.
        if self._shard_pg is None:
            torch.distributed.reduce(rewards, dst=0, op=torch.distributed.ReduceOp.SUM)
            rewards /= self.world_size

        successes = trajectory.successes.mean()
        if self._shard_pg is None:
            torch.distributed.reduce(successes, dst=0, op=torch.distributed.ReduceOp.SUM)
            successes /= self.world_size

        log_dict = {
            "rewards": rewards,
            "successes": successes,
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "loss": grpo_stats.loss.mean(),
            "policy_loss": grpo_stats.policy_loss.mean(),
            "kl_loss": grpo_stats.kl_loss.mean(),
            "clipfrac": grpo_stats.clipfrac.mean(),
            "ratios": grpo_stats.ratios.mean(),
            "approx_policy_kl": grpo_stats.approx_policy_kls.mean(),
            "response_lengths": trajectory.seq_lens.float().mean(),
            **extras,
        }

        if supports_memory_stats(self._device) and self._log_peak_memory_stats:
            try:
                log_dict.update(training.get_memory_stats(device=self._device))
            except RuntimeError:
                pass
        if self._is_rank_zero:
            self._metric_logger.log_dict(log_dict, step=self.global_step)
            # Also log key metrics to stdout for verification
            log.info(
                "METRICS step=%d  loss=%.4f  policy_loss=%.4f  kl_loss=%.6f  "
                "rewards=%.3f  successes=%.3f  grad_norm=%.4f  "
                "clipfrac=%.4f  ratios=%.4f  approx_kl=%.6f  resp_len=%.1f",
                self.global_step,
                log_dict["loss"].item() if hasattr(log_dict["loss"], "item") else log_dict["loss"],
                log_dict["policy_loss"].item() if hasattr(log_dict["policy_loss"], "item") else log_dict["policy_loss"],
                log_dict["kl_loss"].item() if hasattr(log_dict["kl_loss"], "item") else log_dict["kl_loss"],
                log_dict["rewards"].item() if hasattr(log_dict["rewards"], "item") else log_dict["rewards"],
                log_dict["successes"].item() if hasattr(log_dict["successes"], "item") else log_dict["successes"],
                log_dict.get("grad_norm", 0.0),
                log_dict["clipfrac"].item() if hasattr(log_dict["clipfrac"], "item") else log_dict["clipfrac"],
                log_dict["ratios"].item() if hasattr(log_dict["ratios"], "item") else log_dict["ratios"],
                log_dict["approx_policy_kl"].item() if hasattr(log_dict["approx_policy_kl"], "item") else log_dict["approx_policy_kl"],
                log_dict["response_lengths"].item() if hasattr(log_dict["response_lengths"], "item") else log_dict["response_lengths"],
            )

    def run_eval(self) -> None:
        """Run evaluation on held-out data. Generates via vLLM, computes rewards.

        All ranks participate (required for vLLM broadcast), but only rank 0 logs.
        """
        if not self._eval_enabled:
            return

        eval_t0 = time.perf_counter()
        if self._is_rank_zero:
            log.info("EVAL starting at step %d (%d examples)",
                     self._steps_run, len(self._eval_examples))

        all_rewards = []
        all_successes = []
        all_response_lengths = []
        eval_grpo_samples = self._eval_grpo_samples or self.grpo_samples

        collate_fn = _get_component_from_path("torchtune.dev.grpo.data.padded_collate_rl")

        for eval_idx, example in enumerate(self._eval_examples):
            batch = collate_fn([example], padding_idx=self._tokenizer.pad_id)
            tokens = batch["tokens"].to(self._device)
            answers = batch["answers"]
            _, context_length = tokens.shape

            # Expand for grpo_samples
            batch_input_ids = tokens[:, None, :].expand(-1, eval_grpo_samples, -1)
            batch_input_ids = batch_input_ids.reshape(eval_grpo_samples, -1)

            with torch.no_grad():
                if self._vllm_mode == "server":
                    query_responses = self._generate_with_vllm(batch_input_ids, context_length)
                elif self._vllm_mode in ("colocate", "colocate_sleep"):
                    query_responses = self._generate_with_colocated_vllm(batch_input_ids, context_length)
                else:
                    # Native generation
                    with local_kv_cache(
                        model=self._model,
                        batch_size=eval_grpo_samples,
                        device=self._device,
                        dtype=self._dtype,
                        decoder_max_seq_len=context_length + self._max_generated_tokens,
                    ):
                        query_responses, _ = generate(
                            model=self._model,
                            prompt=batch_input_ids,
                            max_generated_tokens=self._max_generated_tokens,
                            temperature=self._temperature,
                            top_k=self._top_k,
                            pad_id=self._tokenizer.pad_id,
                            stop_tokens=self._tokenizer.stop_tokens if hasattr(self._tokenizer, 'stop_tokens') else None,
                            return_logits=False,
                        )

                responses = query_responses[:, context_length:].clone()

                # Truncate at stop tokens
                response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
                    responses, self._stop_token_ids, self._tokenizer.pad_id
                )

                # Compute rewards
                responses_reshaped = responses.reshape(1, eval_grpo_samples, -1)
                if self._reward_mode == "gene_recall":
                    rewards, successes, _ = gene_recall_batched_rewards(
                        self._tokenizer, responses_reshaped, answers, device=self._device,
                        reward_metric=self._gene_reward_metric,
                    )
                else:
                    rewards, successes, _ = batched_rewards(
                        self._tokenizer, responses_reshaped, answers, device=self._device
                    )
                # Sum across reward functions, mean across samples
                rewards = rewards.sum(dim=-1).mean().item()
                successes_mean = successes[:, :, -1].mean().item()

                all_rewards.append(rewards)
                all_successes.append(successes_mean)

                seq_lens = response_padding_masks.any(-1).sum()
                resp_len = (~response_padding_masks).sum(dim=-1).float().mean().item()
                all_response_lengths.append(resp_len)

            del tokens, batch_input_ids, query_responses, responses, response_padding_masks
            if self._device.type == "xpu":
                import gc
                gc.collect()

        eval_time = time.perf_counter() - eval_t0
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        avg_success = sum(all_successes) / len(all_successes) if all_successes else 0.0
        avg_resp_len = sum(all_response_lengths) / len(all_response_lengths) if all_response_lengths else 0.0

        eval_log_dict = {
            "eval/rewards": avg_reward,
            "eval/successes": avg_success,
            "eval/response_lengths": avg_resp_len,
            "eval/num_examples": len(self._eval_examples),
            "eval/time_seconds": eval_time,
        }

        if self._is_rank_zero:
            self._metric_logger.log_dict(eval_log_dict, step=self.global_step)
            log.info(
                "EVAL step=%d  rewards=%.3f  successes=%.3f  resp_len=%.1f  time=%.1fs  (%d examples)",
                self._steps_run, avg_reward, avg_success, avg_resp_len, eval_time,
                len(self._eval_examples),
            )

    def cleanup(self) -> None:
        if self._vllm_client is not None and self._vllm_weight_sync and self._vllm_client.communicator is not None:
            self._vllm_client.close_communicator()
        # Abort the XCCL weight-sync PG on both sides before exiting. The PG is created
        # via c10d.ProcessGroupXCCL directly (not dist.new_group), so dist.destroy_process_group
        # doesn't work (not in registry). Use abort() which is unilateral — no collective needed.
        if getattr(self, "_xccl_wsync_pg", None) is not None:
            import threading as _threading

            def _close_xccl_replica(url):
                try:
                    requests.post(
                        f"{url}/collective_rpc",
                        json={"method": "close_xccl_communicator", "args": []},
                        timeout=30,
                    )
                except Exception:
                    pass

            if self._is_shard_leader:
                close_threads = [
                    _threading.Thread(target=_close_xccl_replica, args=(url,), daemon=True)
                    for url in self._vllm_urls
                ]
                for t in close_threads:
                    t.start()
                # Wait for vLLM abort() to complete BEFORE training abort().
                # Intel XCCL abort() appears to require both sides to call it
                # in overlapping time windows — calling training abort() first
                # (before vLLM has entered abort()) deadlocks.
                for t in close_threads:
                    t.join(timeout=30)

            # Abort training-side PG in a daemon thread with a hard timeout so
            # that even if Intel XCCL abort() hangs we don't block indefinitely.
            _pg = self._xccl_wsync_pg
            self._xccl_wsync_pg = None
            _abort_done = _threading.Event()

            def _do_abort(pg=_pg):
                try:
                    pg.abort()
                except Exception:
                    pass
                _abort_done.set()

            _threading.Thread(target=_do_abort, daemon=True).start()
            if not _abort_done.wait(timeout=15):
                log.warning("XCCL abort() timed out after 15s — forcing os._exit(0)")
                os._exit(0)
            log.info("XCCL wsync PG aborted cleanly.")
        if self._vllm_llm is not None:
            del self._vllm_llm
            self._vllm_llm = None
        # Release persistent SHM block allocated by _sync_weights_to_vllm_shm()
        shm_block = getattr(self, "_shm_block", None)
        if shm_block is not None:
            try:
                shm_block.close()
                shm_block.unlink()
            except Exception:
                pass
            self._shm_block = None
        if self._is_rank_zero:
            self._metric_logger.close()
        if getattr(self, "_vllm_weight_sync_method", None) == "xccl":
            # XCCL abort() leaves oneCCL in a state where the subsequent
            # destroy_process_group() collective hangs indefinitely on all ranks.
            # Checkpoint is already saved before cleanup() is called, so it is
            # safe to skip normal teardown and exit immediately.
            os._exit(0)
        destroy_process_group()

    def cleanup_after_step(
        self,
        trajectory: GRPOTrajectory,
        l_grpo_stats: list[GRPOStats],
    ) -> None:
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats
        # Memory cleanup — gc only, skip empty_cache (leaks UR handles with FSDP)
        if self._device.type == "xpu":
            import gc
            gc.collect()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """

    recipe = GRPOFullFinetuneDistributedXPU(cfg=cfg)
    config.log_config(recipe_name="GRPOFullFinetuneDistributedXPU", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

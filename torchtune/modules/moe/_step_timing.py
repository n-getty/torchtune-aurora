# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Opt-in, rank-0-only sub-phase wall-clock timing for a single MoE training
step (``TORCHTUNE_MOE_STEP_TIMING=1``).

Context: py-spy statistical sampling on Qwen3-30B-A3B/EP=8 SFT (see
``memory/project_moe_sft_profiling_gradrelease_bottleneck_20260724.md``) found
~29.2% of samples in a generic ``backward()`` frame it could not resolve
further (native C++ autograd kernel execution). This module extends the
methodology that DID work for the grad-release investigation
(``TORCHTUNE_EP_GRAD_RELEASE_TIMING=1`` in ``torchtune/dev/rl/distributed.py``,
``torch.xpu.synchronize()``-bracketed rank-0 wall-clock, not statistical
sampling) to the rest of a MoE layer's forward and — via paired
gradient hooks on the tensors flowing into/out of ``self.experts(...)`` —
the expert module's OWN backward, isolated from everything else running in
the single combined ``.backward()`` call.

Usage (see call sites in ``torchtune/modules/moe/moe.py``,
``torchtune/models/qwen3_moe/_component_builders.py``, and
``recipes/dev/full_finetune_moe_distributed_xpu.py``):

    from torchtune.modules.moe._step_timing import timed, backward_mark, \\
        reset_step, report_step, STEP_TIMING_ENABLED

    with timed("router_fwd"):
        ...
    tensor.register_hook(backward_mark("expert_bwd"))  # paired start/end marks

Safe to leave in permanently: fully gated behind the env var, zero overhead
(no synchronize calls, no hook registration) when disabled.
"""
import os
import time
from contextlib import contextmanager

import torch

STEP_TIMING_ENABLED = os.environ.get("TORCHTUNE_MOE_STEP_TIMING", "0") == "1"

_accum: dict = {}
_counts: dict = {}
_bwd_pending_start: dict = {}  # name -> perf_counter() timestamp of the last "start" mark


def _is_rank0() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _sync():
    if torch.xpu.is_available():
        torch.xpu.synchronize()


def reset_step() -> None:
    """Call once at the start of each optimizer step (not each microbatch —
    accumulates across gradient_accumulation_steps microbatches for a single
    step's total, matching how time_per_step_s is reported elsewhere)."""
    if not (STEP_TIMING_ENABLED and _is_rank0()):
        return
    _accum.clear()
    _counts.clear()
    _bwd_pending_start.clear()


@contextmanager
def timed(name: str):
    """Bracket a forward-side operation with XPU-synchronized wall-clock timing.

    No-op (zero overhead) unless TORCHTUNE_MOE_STEP_TIMING=1 and this is rank 0.
    """
    if not (STEP_TIMING_ENABLED and _is_rank0()):
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _accum[name] = _accum.get(name, 0.0) + (time.perf_counter() - t0)
        _counts[name] = _counts.get(name, 0) + 1


_bwd_bad_pairing_warned: set = set()


def backward_mark(name: str, role: str):
    """Return a tensor gradient hook that marks one half of a start/end pair
    for `name`. `role` must be "start" or "end".

    Backward runs in the REVERSE of forward order: a hook registered on
    tensor T fires when the gradient flowing INTO T is ready — i.e. right
    when the operation that CONSUMED T (in forward) finishes its backward,
    and right BEFORE T's own producing op runs its backward. So to time a
    forward-side sub-block ``y = block(x)`` sitting between two other ops,
    register a "start" hook on ``y`` (fires the instant the op downstream of
    ``y`` finishes backward, i.e. right as ``block``'s own backward begins)
    and an "end" hook on ``x`` (fires once ``block``'s backward has produced
    ``x``'s gradient, i.e. right as ``block``'s backward finishes).

    Safe for repeated layers in a single synchronous (non-chunked,
    non-reentrant-interleaved) backward pass: each layer's start/end pair
    fires fully before the next layer's, so accumulation across N layers is
    just N additive (start, end) pairs for the same `name`.
    """
    assert role in ("start", "end")

    def _hook(grad):
        if not (STEP_TIMING_ENABLED and _is_rank0()):
            return grad
        _sync()
        now = time.perf_counter()
        if role == "start":
            if name in _bwd_pending_start and name not in _bwd_bad_pairing_warned:
                _bwd_bad_pairing_warned.add(name)
                print(
                    f"[moe_step_timing] WARNING: '{name}' start fired while a "
                    "prior start was still pending (unexpected backward "
                    "interleaving) — discarding the stale one.",
                    flush=True,
                )
            _bwd_pending_start[name] = now
        else:  # role == "end"
            start = _bwd_pending_start.pop(name, None)
            if start is None:
                if name not in _bwd_bad_pairing_warned:
                    _bwd_bad_pairing_warned.add(name)
                    print(
                        f"[moe_step_timing] WARNING: '{name}' end fired with "
                        "no pending start — skipping this sample.",
                        flush=True,
                    )
                return grad
            _accum[name] = _accum.get(name, 0.0) + (now - start)
            _counts[name] = _counts.get(name, 0) + 1
        return grad
    return _hook


def report_step(logger, rank_zero: bool = True, prefix: str = "moe_step_timing") -> dict:
    """Log and return the accumulated per-phase timings for the just-completed
    step. Call AFTER backward (and optionally after the optimizer step) but
    BEFORE the next reset_step(). Returns {} and logs nothing if disabled."""
    if not (STEP_TIMING_ENABLED and _is_rank0()):
        return {}
    if rank_zero and logger is not None:
        parts = ", ".join(
            f"{k}={v:.3f}s(n={_counts.get(k, 0)})" for k, v in sorted(_accum.items())
        )
        logger.info("%s: %s", prefix, parts if parts else "(no phases recorded)")
    return dict(_accum)


def step_record() -> dict:
    """Return opt-in phase timings and invocation counts for the current step."""
    if not (STEP_TIMING_ENABLED and _is_rank0()):
        return {"timings_s": {}, "timing_counts": {}}
    return {
        "timings_s": dict(_accum),
        "timing_counts": dict(_counts),
    }

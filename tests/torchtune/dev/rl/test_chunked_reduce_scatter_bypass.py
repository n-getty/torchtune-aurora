# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test: the CHUNKED backward path in ``grpo_step`` must
bypass the gloo CPU-AllReduce ``reduce_scatter_tensor`` patch on NON-EP runs.

Background (see docs/reports/chunked_reduce_scatter_bypass_fix_20260617.md):

``install_xpu_patches()`` swaps ``dist.reduce_scatter_tensor`` for a gloo
CPU-bounce (D2H -> gloo AllReduce -> H2D), required for Expert Parallelism but
catastrophic for non-EP FSDP2: ~2s/layer x 64 layers = ~130s added to backward
(it corrupted a 4B benchmark to 274s/step). The SINGLE_BACKWARD path already
restores the native ``_orig_reduce_scatter_tensor`` around its ``.backward()``;
the CHUNKED path historically did NOT, so a non-EP chunked dense run sent its
single (final-chunk) reduce_scatter through the 130s gloo path.

This test parses the recipe ``grpo_step`` source via AST and pins:
  (a) the chunked-backward path installs the ``_orig_reduce_scatter_tensor``
      bypass (save -> swap -> backward -> finally-restore),
  (b) the bypass is gated on a NON-EP condition
      (``self._expert_parallel_degree <= 1``), so EP runs keep the gloo patch,
  (c) the SINGLE_BACKWARD bypass is still present (we did not regress it).

Static (AST string scan), import-free: the recipe pulls torchao + XPU backends
at import time and crashes on a login node.
"""
import ast

import pytest


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "grpo_full_finetune_distributed_xpu.py"
)

_CLASS = "GRPOFullFinetuneDistributedXPU"
_METHOD = "grpo_step"

# Marker for the native (un-patched) reduce_scatter restore.
_BYPASS_MARKER = "_orig_reduce_scatter_tensor"
# The non-EP gate used throughout the recipe.
_EP_GATE = "self._expert_parallel_degree <= 1"


def _method_source() -> str:
    with open(_RECIPE_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == _CLASS:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == _METHOD:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f"Could not find {_CLASS}.{_METHOD}")


def _split_paths(method_src: str) -> tuple[str, str]:
    """Split grpo_step into (single_backward_block, chunked_backward_block).

    The chunked block is everything from the ``else:`` that opens the chunked
    branch (its hallmark is the ``num_fwd_chunks`` loop) onward. The
    single-backward block is the region containing the
    ``single-backward backward start`` log line.
    """
    lines = method_src.splitlines()
    # Locate the chunked branch by its unique loop variable header.
    chunk_idx = next(
        i for i, ln in enumerate(lines) if "for _cs in range(0, total_seqs, fwd_bs)" in ln
    )
    # Walk back to the nearest enclosing comment that opens the chunked branch.
    chunk_start = next(
        i for i in range(chunk_idx, -1, -1)
        if "Chunked training forward+backward" in lines[i]
    )
    single_block = "\n".join(lines[:chunk_start])
    chunked_block = "\n".join(lines[chunk_start:])
    return single_block, chunked_block


@pytest.fixture(scope="module")
def paths() -> tuple[str, str]:
    return _split_paths(_method_source())


def test_chunked_path_has_reduce_scatter_bypass(paths):
    """The chunked backward path must restore native reduce_scatter."""
    _single, chunked = paths
    assert _BYPASS_MARKER in chunked, (
        "Chunked backward path no longer references "
        f"{_BYPASS_MARKER!r}. Without the bypass, a non-EP chunked dense run "
        "routes its final-chunk reduce_scatter through the ~130s gloo "
        "CPU-bounce patch (corrupted a 4B benchmark to 274s/step)."
    )


def test_chunked_bypass_is_non_ep_gated(paths):
    """The bypass must be gated on a non-EP condition so EP runs keep gloo.

    We assert the non-EP gate appears in the same statement region as the
    reduce_scatter swap, by checking the gate variable assignment precedes the
    swap and references the non-EP condition.
    """
    _single, chunked = paths
    assert _EP_GATE in chunked, (
        f"Chunked path lost the non-EP gate {_EP_GATE!r}."
    )
    # Find the bypass swap line and confirm a non-EP gate guards it.
    lines = chunked.splitlines()
    swap_idx = next(
        i for i, ln in enumerate(lines)
        if "reduce_scatter_tensor = _orig_reduce_scatter_tensor" in ln
    )
    # The gate variable must be assigned from the non-EP condition shortly
    # before the swap, and the swap must be inside an `if <gate>:` block.
    preceding = "\n".join(lines[max(0, swap_idx - 8):swap_idx])
    assert _EP_GATE in preceding, (
        "The reduce_scatter swap in the chunked path is not guarded by the "
        f"non-EP gate {_EP_GATE!r}. EP runs MUST keep the gloo patch "
        "(native XCCL reduce_scatter on the EP mesh has the op#259 deadlock)."
    )
    # The swap must be restored in a finally (exception-safe, mirrors
    # SINGLE_BACKWARD). Confirm a `finally:` and a restore appear after the swap.
    trailing = "\n".join(lines[swap_idx:swap_idx + 45])
    assert "finally:" in trailing and "_rsc_patch_saved_ck" in trailing, (
        "The chunked bypass must restore the saved patch in a finally block "
        "so the gloo patch is reinstated even if backward raises."
    )


def test_chunked_bypass_logs_on_rank_zero(paths):
    """A rank-0 log line must announce the chunked bypass (self-documenting runs)."""
    _single, chunked = paths
    assert "chunked backward: non-EP reduce_scatter bypass ACTIVE" in chunked, (
        "The chunked bypass must emit a rank-0 log line so future runs are "
        "self-documenting (supports the run-health gate)."
    )


def test_single_backward_bypass_not_regressed(paths):
    """The pre-existing SINGLE_BACKWARD bypass must still be present."""
    single, _chunked = paths
    assert _BYPASS_MARKER in single, (
        "SINGLE_BACKWARD path lost its reduce_scatter bypass — regression."
    )


def test_ep_path_keeps_gloo_patch(paths):
    """When EP is active the chunked path must NOT swap to native XCCL.

    Structurally: the swap lives inside an `if <non-EP gate>:` block, so when
    expert_parallel_degree > 1 the gate is False and no swap occurs. We assert
    the gate variable is assigned `self._expert_parallel_degree <= 1` (i.e. the
    EP-active case yields False) immediately before the bypass region.
    """
    _single, chunked = paths
    assert f"_rsc_bypass_chunk = {_EP_GATE}" in chunked, (
        "The chunked bypass gate must be `_rsc_bypass_chunk = "
        f"{_EP_GATE}` so that EP (degree > 1) disables the swap and the gloo "
        "patch stays in force (byte-identical EP behavior)."
    )

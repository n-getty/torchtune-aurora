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


_METHOD = "grpo_step"

# Marker for the native (un-patched) reduce_scatter restore.
_BYPASS_MARKER = "_orig_reduce_scatter_tensor"
# The non-EP gate used throughout the recipe.
_EP_GATE = "self._expert_parallel_degree <= 1"


def _method_source(recipe_path: str, class_name: str) -> str:
    with open(recipe_path) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == _METHOD:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f"Could not find {class_name}.{_METHOD}")


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
        i
        for i, ln in enumerate(lines)
        if "for _cs in range(0, total_seqs, fwd_bs)" in ln
    )
    # Walk back to the nearest enclosing comment that opens the chunked branch.
    chunk_start = next(
        i
        for i in range(chunk_idx, -1, -1)
        if "Chunked training forward+backward" in lines[i]
    )
    single_block = "\n".join(lines[:chunk_start])
    chunked_block = "\n".join(lines[chunk_start:])
    return single_block, chunked_block


@pytest.fixture(
    scope="module",
    params=[
        (
            "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
            "grpo_full_finetune_distributed_xpu.py",
            "GRPOFullFinetuneDistributedXPU",
        ),
        (
            "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
            "grpo_bioreason_distributed_xpu.py",
            "GRPOBioReasonDistributedXPU",
        ),
    ],
    ids=["base", "bioreason"],
)
def paths(request) -> tuple[str, str]:
    return _split_paths(_method_source(*request.param))


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
    assert _EP_GATE in chunked, f"Chunked path lost the non-EP gate {_EP_GATE!r}."
    # Find the bypass swap line and confirm a non-EP gate guards it.
    lines = chunked.splitlines()
    swap_idx = next(
        i
        for i, ln in enumerate(lines)
        if "reduce_scatter_tensor = _orig_reduce_scatter_tensor" in ln
    )
    # The gate variable must be assigned from the non-EP condition shortly
    # before the swap, and the swap must be inside an `if <gate>:` block.
    preceding = "\n".join(lines[max(0, swap_idx - 8) : swap_idx])
    assert _EP_GATE in preceding, (
        "The reduce_scatter swap in the chunked path is not guarded by the "
        f"non-EP gate {_EP_GATE!r}. EP runs MUST keep the gloo patch "
        "(native XCCL reduce_scatter on the EP mesh has the op#259 deadlock)."
    )
    # The swap must be restored in a finally (exception-safe, mirrors
    # SINGLE_BACKWARD). Confirm a `finally:` and a restore appear after the swap.
    trailing = "\n".join(lines[swap_idx : swap_idx + 45])
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
    assert (
        _BYPASS_MARKER in single
    ), "SINGLE_BACKWARD path lost its reduce_scatter bypass — regression."


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


def test_bioreason_postdivide_only_preserves_total_divisor():
    """The XPU workaround must move, not remove, FSDP gradient averaging."""
    recipe_path = (
        "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
        "grpo_bioreason_distributed_xpu.py"
    )
    with open(recipe_path) as f:
        source = f.read()
    assert "TORCHTUNE_FSDP_POSTDIVIDE_ONLY" in source
    assert "_fsdp_state._gradient_predivide_factor = 1.0" in source
    assert "_predivide * _postdivide" in source


def test_bioreason_syncs_fsdp_ignored_projector_gradients():
    with open(
        "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
        "grpo_bioreason_distributed_xpu.py"
    ) as source_file:
        source = source_file.read()

    # The projector/LoRA grads are FSDP-ignored (replicated, not sharded), so FSDP
    # never reduces them -- the recipe must do it by hand over the gloo groups.
    assert "def _sync_ignored_trainable_grads" in source
    assert "_gloo_dp_shard_pg" in source
    assert "_gloo_dp_replicate_pg" in source
    assert "torch.distributed.all_reduce(flat_cpu, group=shard_pg)" in source
    assert "torch.distributed.all_reduce(flat_cpu, group=replicate_pg)" in source
    assert "TORCHTUNE_FSDP_REPLICATE_TRAINABLES" in source
    assert '{"ignored_states": _ignored_states}' in source
    assert "if param.requires_grad" in source
    assert "param.data = param.data.to(self._device)" in source

    # 2026-09-14: the divide moved from `flat_cpu.div_(self._dp_shard)` to a
    # per-param `_divisor` so a param present on only SOME ranks is averaged over its
    # contributors instead of being silently scaled down by n_present/world. Assert the
    # scaling still happens, without pinning the old literal text.
    sync = source.split("def _sync_ignored_trainable_grads", 1)[1].split(
        "\n    def ", 1
    )[0]
    assert "_divisor" in sync
    assert "self._dp_shard" in sync and "self._dp_replicate" in sync

    # THE LOAD-BEARING INVARIANT (job 8826889): the grad buckets must be keyed on
    # `param.dtype` (structural, identical on every rank), NEVER on `param.grad.dtype`
    # (which does not exist when a grad is missing). Bucketing on grad.dtype made both
    # the bucket set and the all_reduce call count rank-dependent -- rank 0 had no grads,
    # issued zero collectives, and its 11 peers blocked until the 1800s gloo timeout
    # (Exit_status=143, zero steps completed). Behavioral proof lives in
    # tests/torchtune/dev/rl/test_sync_ignored_grads_collective_match.py.
    assert "params_by_dtype" in sync, (
        "grads must be bucketed by the structural param.dtype on a rank-independent "
        "param list; see the job-8826889 deadlock."
    )
    assert "grads_by_dtype" not in sync, (
        "grads_by_dtype keys on param.grad.dtype, making the collective count depend "
        "on local grad presence -- this is the job-8826889 deadlock. Bucket on "
        "param.dtype and zero-fill missing grads instead."
    )
    # Missing grads must be zero-filled so every rank builds the same-shaped buffer.
    assert "torch.zeros(param.numel()" in sync


def test_bioreason_cpu_postdivide_replaces_both_xpu_buffer_divides():
    with open(
        "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
        "grpo_bioreason_distributed_xpu.py"
    ) as source_file:
        source = source_file.read()

    assert "TORCHTUNE_FSDP_CPU_POSTDIVIDE" in source
    assert "_fsdp_state._gradient_predivide_factor = 1.0" in source
    assert "_fsdp_state._gradient_postdivide_factor = 1.0" in source
    assert "set_fsdp1_hsdp_cpu_postdivide(_cpu_postdivide)" in source
    assert "are mutually exclusive" in source
    assert "_adjusted_states == 0" in source


def test_both_fsdp_branches_pass_ignored_states_not_ignored_modules():
    """Both FSDP1 wrap sites must honor parameter-level `ignored_states`.

    Jobs 8827075/8827354/8827618: the non-HSDP (dp_replicate=1) branch hardcoded
    `ignored_modules=_ignored`, which excludes only the encoder MODULES. The LoRA
    adapters were therefore swept INTO the per-layer flat params and sharded, while the
    HSDP branch (which passes `**_fsdp_ignore_kwargs`) kept them replicated. The named
    grad census showed each rank holding grads for only 1-3 projection types across all
    64 layers (counts 0/128/384 summing to exactly 896), and the param names carried
    `_fsdp_wrapped_module` twice — once for the root, once for the decoder layer that had
    claimed the adapter.

    Downstream that made `_sync_ignored_trainable_grads` average disjoint shards, which
    aborted in gloo on a buffer-length mismatch. Diagnosing it cost three capacity jobs,
    partly because the pre-wrap log line claimed ignored_states was "ACTIVE" when it had
    only been BUILT.
    """
    with open(
        "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
        "grpo_bioreason_distributed_xpu.py"
    ) as source_file:
        source = source_file.read()

    # No FSDP( call may hardcode the module-level form while a param-level list exists.
    # Match the call-site form only (leading whitespace + trailing comma) so the
    # explanatory comments that quote the old code do not trip this.
    assert "\n                    ignored_modules=_ignored,\n" not in source, (
        "an FSDP() call still hardcodes `ignored_modules=_ignored`; it must pass the "
        "same **_fsdp_ignore_kwargs form the HSDP branch uses so `ignored_states` wins "
        "when _replicate_trainables is set (job 8827618)."
    )
    # Both branches build the kwargs dict the same way.
    assert source.count('{"ignored_states": _ignored_states}') >= 2, (
        "both the HSDP and non-HSDP FSDP wraps must construct "
        '`{"ignored_states": _ignored_states}` when the param-level list exists.'
    )

    # The pre-wrap log must not claim ACTIVE — it fires at list-construction time and
    # reading it as proof of a passed list is what discarded a correct diagnosis.
    # Match the LOG FORMAT STRING, not prose: the surrounding comments legitimately
    # quote the old wording when explaining why it was misleading.
    assert 'ignored_states ACTIVE for %d' not in source, (
        "the pre-wrap log must say BUILT, not ACTIVE: it is emitted when the list is "
        "constructed, before FSDP sees it."
    )
    assert 'ignored_states BUILT for %d' in source, (
        "keep the pre-wrap log, reworded to BUILT."
    )

    # And a post-wrap check must actually verify FSDP honored it.
    assert "FSDP ignored_states NOT honored" in source, (
        "keep the post-wrap verification that flags LoRA params swept into wrapped "
        "decoder layers."
    )


def test_ignored_states_verification_queries_fsdp_not_param_names():
    """The post-wrap check must ask FSDP, not pattern-match qualified names.

    The first version of this check inferred "swept into the flat param" from the name
    containing `_fsdp_wrapped_module` after the `layers.N` segment. That is wrong by
    construction: `ignored_states` excludes a param from FLATTENING, not from the module
    TREE, so an *ignored* adapter still lives under a wrapped decoder layer and its
    qualified name gains `_fsdp_wrapped_module` either way. On job 8827805 it reported
    `NOT honored for 896` with the fix in place — a false positive that could have sent
    the next session chasing a non-bug.

    Ground truth is FSDP's resolved `_ignored_params` set (`_init_utils.py` builds it;
    `_get_orig_params(module, state._ignored_params)` excludes it; it reaches
    auto-wrapped children via `root_kwargs["ignored_states"]`).
    """
    with open(
        "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
        "grpo_bioreason_distributed_xpu.py"
    ) as source_file:
        source = source_file.read()

    assert "_ignored_params" in source, (
        "the post-wrap verification must consult FSDP's resolved _ignored_params set."
    )
    # The discredited heuristic must not come back.
    assert '"_fsdp_wrapped_module" in _n.split("layers.", 1)[-1]' not in source, (
        "name-based 'swept in' detection is a false-positive generator: ignored params "
        "are still nested under wrapped layers. Query _ignored_params instead."
    )

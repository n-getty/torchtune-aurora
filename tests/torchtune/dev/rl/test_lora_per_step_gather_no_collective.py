# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe regression test: the per-step LoRA publish gather must NOT enter an
FSDP ``FULL_STATE_DICT`` collective.

Background (see docs/reports/lora_grpo_perf_hardening_20260617.md):

The 2026-05-04 ``_cache_lora_base_weights`` optimization moved the only
``FSDP.state_dict_type(FULL_STATE_DICT, ...)`` gather to setup() (called once,
guarded by ``not self._lora_use_runtime``). Both per-step publish gathers are
now rank-0-only with no cross-rank collective:

  * Path A (default, ``use_runtime_lora=False``): ``_gather_merged_lora_weights``
    reads ``self._cached_base_weights`` (set at setup) and the *replicated*
    ``lora_a`` / ``lora_b`` adapter tensors live from the model. Adapter params
    are in FSDP ``ignored_states`` so they are not sharded — no all-gather.
  * Path B (legacy, ``use_runtime_lora=True``): ``_gather_lora_state_dict``
    reads ``self._model.named_parameters()`` filtered for adapter keys.

If a future edit re-introduces an FSDP ``state_dict_type`` / ``FULL_STATE_DICT``
context into either *per-step* gather, the ~7 s/step full-model gather (and its
suspected ``banned:1`` PDE trigger on Aurora XPU) comes back silently. This test
parses the recipe source via AST and asserts that neither per-step gather method
contains the collective markers, while ``_cache_lora_base_weights`` (the
once-at-setup gather) still does.

This is a *static* guard (AST string scan), deliberately import-free: the recipe
module pulls torchao + XPU backends at import time and crashes on a login node.
"""
import ast
import textwrap

import pytest


_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "lora_grpo_full_finetune_distributed_xpu.py"
)

# Markers that indicate an FSDP whole-model gather collective.
_COLLECTIVE_MARKERS = ("FULL_STATE_DICT", "state_dict_type")

# Per-step publish gathers — must be rank-0-only, no collective.
_PER_STEP_GATHERS = ("_gather_merged_lora_weights", "_gather_lora_state_dict")

# Once-at-setup gather — collective is expected and correct here.
_SETUP_GATHER = "_cache_lora_base_weights"


def _strip_docstring(fn: ast.FunctionDef, src: str) -> str:
    """Return the *code* source of a function with its leading docstring removed.

    The per-step gather docstrings explicitly mention ``FULL_STATE_DICT`` and
    ``state_dict_type`` to document what they deliberately AVOID. We scan only
    executable code, so docstring prose must not trip the marker check.
    """
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # drop the docstring statement
    return "\n".join(
        textwrap.dedent(ast.get_source_segment(src, stmt)) for stmt in body
    )


def _method_sources() -> dict[str, str]:
    """Return {method_name: code_source} (docstring stripped) for methods of
    LoRAGRPODistributedXPU."""
    with open(_RECIPE_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LoRAGRPODistributedXPU":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    out[item.name] = _strip_docstring(item, src)
    if not out:
        raise RuntimeError("Could not find LoRAGRPODistributedXPU class body")
    return out


@pytest.fixture(scope="module")
def methods() -> dict[str, str]:
    return _method_sources()


@pytest.mark.parametrize("name", _PER_STEP_GATHERS)
def test_per_step_gather_has_no_fsdp_collective(methods, name):
    assert name in methods, f"{name} not found on LoRAGRPODistributedXPU"
    body = methods[name]
    for marker in _COLLECTIVE_MARKERS:
        assert marker not in body, (
            f"{name} contains FSDP-collective marker {marker!r}. The per-step "
            f"publish gather must stay rank-0-only (no FULL_STATE_DICT). "
            f"The full-model gather belongs in {_SETUP_GATHER} (once at setup)."
        )


def test_setup_gather_still_collective(methods):
    """The once-at-setup base-weight cache MUST keep the FULL_STATE_DICT gather —
    if it loses it, the cache is empty/sharded and the merged path breaks."""
    assert _SETUP_GATHER in methods
    body = methods[_SETUP_GATHER]
    assert "FULL_STATE_DICT" in body, (
        f"{_SETUP_GATHER} lost its FULL_STATE_DICT gather — base-weight cache "
        f"would be incomplete/sharded and the merged publish path would corrupt."
    )


def test_merged_gather_reads_cached_base_weights(methods):
    """Path A gather must source base weights from the rank-0 cache, not a
    live re-gather."""
    body = methods["_gather_merged_lora_weights"]
    assert "_cached_base_weights" in body, (
        "_gather_merged_lora_weights no longer references _cached_base_weights — "
        "it may have reverted to a per-step gather."
    )


def test_per_step_gathers_are_rank0_guarded(methods):
    """Both per-step gathers early-return on non-rank-0 (rank-0-only contract).
    A collective, by contrast, requires *all* ranks to participate; a rank-0
    guard is the structural signature of a non-collective path."""
    for name in _PER_STEP_GATHERS:
        body = methods[name]
        assert "self._is_rank_zero" in body and "return None" in body, (
            f"{name} must early-return None on non-rank-0 ranks (rank-0-only, "
            f"no collective). Missing the guard suggests a collective path."
        )

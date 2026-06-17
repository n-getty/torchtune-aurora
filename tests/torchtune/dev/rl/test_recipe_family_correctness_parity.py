# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe drift guard for the GRPO recipe family.

The GRPO recipes are a family: one base (``GRPOFullFinetuneDistributedXPU``)
plus model/variant-specific recipes. By convention model-specific recipes
SUBCLASS the base so shared correctness fixes apply once. ``lora_grpo`` was
historically a copy-paste FORK, and it silently drifted away from two base
correctness fixes:

  * Llama-family Q/K un-permute at weight-sync time (``_maybe_unpermute_qk``).
    Without it, weights published to vLLM are scrambled for LLAMA* models —
    the exact bug that cost a multi-day bake-off investigation on the base
    recipe (see test_wsync_qk_unpermute.py and
    bugs/project_wsync_qk_permute_llama_family_bug.md).
  * Batch-level advantage normalization (``batch_level_advantages``), which
    keeps the learning signal alive when a prompt-group's rewards are
    degenerate.

This test fails if any GRPO recipe loses access to those invariants — whether
by inheriting from a base that has them, or (for a standalone recipe) by
referencing the shared helpers directly. It is the guard that, had it existed,
would have caught the lora_grpo drift the day it happened.

Implementation: pure source/AST inspection. No torch, no XPU, no device init.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RECIPES_DIR = Path(__file__).resolve().parents[4] / "recipes" / "dev"

# GRPO recipes that must carry the base correctness invariants. Each is either
# the base itself, a subclass of it, or (tolerated) a standalone recipe that
# references the shared helpers directly.
GRPO_RECIPES = [
    "grpo_full_finetune_distributed_xpu.py",      # base
    "grpo_bioreason_distributed_xpu.py",          # subclass
    "lora_grpo_full_finetune_distributed_xpu.py",  # subclass (was a fork)
]

BASE_CLASS = "GRPOFullFinetuneDistributedXPU"

# Tokens that prove the invariant is reachable in a recipe's own source.
QK_UNPERMUTE_TOKENS = ("_maybe_unpermute_qk", "_needs_qk_unpermute")
ADVANTAGE_TOKENS = ("batch_level_advantages",)


def _recipe_source(name: str) -> str:
    path = RECIPES_DIR / name
    assert path.exists(), f"recipe not found: {path}"
    return path.read_text()


def _top_level_class_bases(source: str) -> dict[str, list[str]]:
    """Map each top-level class name -> list of base-class names (as text)."""
    tree = ast.parse(source)
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            out[node.name] = bases
    return out


def _inherits_base(source: str) -> bool:
    """True if the recipe defines a class that subclasses the GRPO base."""
    for _cls, bases in _top_level_class_bases(source).items():
        if BASE_CLASS in bases:
            return True
    return False


def _is_base_recipe(name: str, source: str) -> bool:
    return BASE_CLASS in _top_level_class_bases(source)


def _references(source: str, tokens) -> bool:
    return any(tok in source for tok in tokens)


@pytest.mark.parametrize("recipe", GRPO_RECIPES)
def test_recipe_carries_qk_unpermute_invariant(recipe):
    """Every GRPO recipe must reach the Q/K un-permute fix.

    Either it IS the base, or it subclasses the base (inherits it), or it
    references the shared helper directly. A standalone fork that does none of
    these scrambles Llama-family weights at vLLM sync — the drift this guards.
    """
    source = _recipe_source(recipe)
    if _is_base_recipe(recipe, source):
        assert _references(source, QK_UNPERMUTE_TOKENS), (
            f"{recipe} is the base recipe but no longer references "
            f"{QK_UNPERMUTE_TOKENS} — the Q/K un-permute fix regressed."
        )
        return
    if _inherits_base(source):
        # Inherits the base's bound helpers; nothing more required.
        return
    assert _references(source, QK_UNPERMUTE_TOKENS), (
        f"{recipe} neither subclasses {BASE_CLASS} nor references "
        f"{QK_UNPERMUTE_TOKENS}. If it publishes weights to vLLM it will "
        "scramble Llama-family Q/K. Subclass the base or call the shared "
        "weight_sync helper."
    )


@pytest.mark.parametrize("recipe", GRPO_RECIPES)
def test_recipe_carries_batch_level_advantages(recipe):
    """Every GRPO recipe must honor batch_level_advantages.

    Subclasses inherit the base's grpo_step/trajectory path; standalone recipes
    must reference the shared helper so the config flag is wired.
    """
    source = _recipe_source(recipe)
    if _is_base_recipe(recipe, source):
        assert _references(source, ADVANTAGE_TOKENS), (
            f"{recipe} is the base recipe but no longer references "
            f"{ADVANTAGE_TOKENS} — batch-level advantage normalization regressed."
        )
        return
    if _inherits_base(source):
        return
    assert _references(source, ADVANTAGE_TOKENS), (
        f"{recipe} neither subclasses {BASE_CLASS} nor references "
        f"{ADVANTAGE_TOKENS}. Wire batch_level_advantages (gate on "
        "cfg.get('batch_level_advantages', True)) or subclass the base."
    )


def test_lora_grpo_specifically_has_both_fixes():
    """Explicit pin for the recipe that drifted, independent of the matrix above.

    lora_grpo computes advantages and publishes merged weights in its own code
    (the merged-weight publish path has no base equivalent), so even once it is
    a subclass it must still reference both helpers in its own source.
    """
    source = _recipe_source("lora_grpo_full_finetune_distributed_xpu.py")
    assert _references(source, QK_UNPERMUTE_TOKENS), (
        "lora_grpo merges weights and publishes to vLLM in its own "
        "_gather_merged_lora_weights — it must apply _maybe_unpermute_qk."
    )
    assert _references(source, ADVANTAGE_TOKENS), (
        "lora_grpo computes advantages in its own generate_trajectory — it "
        "must honor batch_level_advantages."
    )


# Attributes the SHARED _setup_vllm_server_mode (vllm_backend.py) reads
# unconditionally (i.e. before the `if self._vllm_weight_sync:` gate). Any
# recipe that BINDS that shared method but does NOT subclass the base recipe
# must set these in its own __init__, or setup() AttributeErrors at launch.
# This pins the real bug the Phase-2 smoke caught: 8b5f0f3f added _dp_replicate
# / _is_shard_leader reads to the shared helper, and the standalone lora_grpo
# fork (which binds the helper but isn't a subclass) had set neither — it was
# crash-on-setup from 2026-06-16 until this fix.
SERVER_MODE_REQUIRED_ATTRS = ("_dp_replicate", "_is_shard_leader")


def _binds_shared_server_mode(source: str) -> bool:
    return "_setup_vllm_server_mode = _vllm_backend_module._setup_vllm_server_mode" in source


def _sets_attr(source: str, attr: str) -> bool:
    # Matches `self._attr = ...` (an assignment, not just a read).
    return re.search(rf"self\.{re.escape(attr)}\s*=", source) is not None


@pytest.mark.parametrize("recipe", GRPO_RECIPES)
def test_standalone_recipe_sets_shared_server_mode_attrs(recipe):
    """A standalone recipe binding _setup_vllm_server_mode must set its attrs.

    Subclasses inherit the base __init__ that sets them; the base itself sets
    them. Only a standalone recipe that binds the shared helper is at risk.
    """
    source = _recipe_source(recipe)
    if _is_base_recipe(recipe, source) or _inherits_base(source):
        return  # base sets them; subclasses inherit them
    if not _binds_shared_server_mode(source):
        return  # doesn't use the shared helper, no contract to meet
    missing = [a for a in SERVER_MODE_REQUIRED_ATTRS if not _sets_attr(source, a)]
    assert not missing, (
        f"{recipe} binds the shared _setup_vllm_server_mode but does not set "
        f"{missing} in __init__. That helper reads them unconditionally "
        "(added 8b5f0f3f) — setup() will AttributeError at launch. Set them to "
        "single-replicate defaults (_dp_replicate=1, _is_shard_leader=_is_rank_zero)."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

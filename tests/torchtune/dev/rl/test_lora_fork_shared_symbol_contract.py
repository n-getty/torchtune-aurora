# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe symbol-existence guard for the standalone LoRA-GRPO fork.

``lora_grpo_full_finetune_distributed_xpu.py`` is DELIBERATELY a standalone
recipe (NOT a subclass of ``GRPOFullFinetuneDistributedXPU`` — see CLAUDE.md).
To stay correct it BINDS several shared symbols from the base recipe and the
``torchtune.dev.rl`` modules, two ways:

  * class-body injection:  ``name = _module.name``  (the "injected method"
    pattern), e.g. ``_maybe_unpermute_qk = _weight_sync_module._maybe_unpermute_qk``
  * function-scoped import: ``from torchtune.dev.rl.<mod> import <name>``
    e.g. ``from torchtune.dev.rl.vllm_client import vllm_http_generate``

Because it is a fork, a RENAME or REMOVAL of one of those shared symbols does
not break the fork at import time — it breaks at *runtime on a compute node*,
which is exactly how the ``_dp_replicate``/``_is_shard_leader`` drift cost a
day and the dense-4B launcher dropped ``--worker-extension-cls``.

This test enumerates every shared symbol the fork binds and asserts (by AST
inspection of the SOURCE module — never importing it; the modules pull XPU/vLLM)
that the symbol still exists where the fork expects it. A rename in shared code
then fails this login-node test loudly instead of at launch.

Complements ``test_recipe_family_correctness_parity.py``:
  * that test pins the *correctness invariants* (Q/K unpermute, batch-level
    advantages) are reachable, and that a server-mode binder sets two specific
    attrs.
  * THIS test pins that *every shared binding resolves* — a broader, mechanical
    "the wires are still connected" check, plus a generalized version of the
    server-mode attr contract driven off the helper's real source.

Implementation: pure source/AST inspection. No torch, no XPU, no device init.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPES_DIR = REPO_ROOT / "recipes" / "dev"
RL_DIR = REPO_ROOT / "torchtune" / "dev" / "rl"

LORA_RECIPE = "lora_grpo_full_finetune_distributed_xpu.py"

# Map the module-alias used in the LoRA recipe's class-body injections to the
# real source file the alias points at. Keep this in sync with the recipe's
# top-of-file ``import torchtune.dev.rl.<mod> as _<alias>`` lines.
MODULE_ALIAS_TO_FILE = {
    "_weight_sync_module": RL_DIR / "weight_sync.py",
    "_vllm_backend_module": RL_DIR / "vllm_backend.py",
}


def _read(path: Path) -> str:
    assert path.exists(), f"expected file not found: {path}"
    return path.read_text()


def _module_defines(source: str, name: str) -> bool:
    """True if `source` binds `name` at module top level.

    AST-based so we never import the (XPU/vLLM-pulling) module. Covers every way
    a top-level name can be bound: ``def`` / ``async def`` / ``class``, plain
    and annotated assignment (``x = ...`` / ``x: T = ...``), and re-exports
    (``from m import name`` / ``import m as name``) so a symbol the fork imports
    that the target module itself re-exports still resolves.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return True
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound == name:
                    return True
    return False


# ---------------------------------------------------------------------------
# 1a. Class-body injected bindings: `name = _alias.name`
# ---------------------------------------------------------------------------
def _injected_bindings(source: str) -> list[tuple[str, str, str]]:
    """Return (lhs_name, module_alias, attr_name) for every class-body
    assignment of the form ``name = _module_alias.attr``."""
    tree = ast.parse(source)
    out: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            val = stmt.value
            if not (
                isinstance(val, ast.Attribute)
                and isinstance(val.value, ast.Name)
                and val.value.id in MODULE_ALIAS_TO_FILE
            ):
                continue
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    out.append((tgt.id, val.value.id, val.attr))
    return out


# ---------------------------------------------------------------------------
# 1b. Function-scoped imports of shared rl symbols
#     `from torchtune.dev.rl.<mod> import <name>`
# ---------------------------------------------------------------------------
def _rl_imports(source: str) -> list[tuple[str, str]]:
    """Return (module_dotted, imported_name) for every (top- or function-level)
    ``from torchtune.dev.rl.<mod> import <name>`` in the recipe."""
    tree = ast.parse(source)
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("torchtune.dev.rl."):
                for alias in node.names:
                    out.append((node.module, alias.name))
    return out


def _rl_module_path(dotted: str) -> Path:
    """torchtune.dev.rl.vllm_client -> .../torchtune/dev/rl/vllm_client.py"""
    leaf = dotted.split(".")[-1]
    return RL_DIR / f"{leaf}.py"


@pytest.fixture(scope="module")
def lora_source() -> str:
    return _read(RECIPES_DIR / LORA_RECIPE)


def test_lora_fork_binds_some_shared_symbols(lora_source):
    """Sanity: the grep-driven assumptions still hold (the fork DOES bind
    shared symbols). If this drops to zero the rest of the file silently
    passes — guard against that."""
    injected = _injected_bindings(lora_source)
    imported = _rl_imports(lora_source)
    assert injected, (
        "Expected the LoRA fork to inject shared methods via "
        "`name = _module.name`; found none. Did the binding pattern change? "
        "Update MODULE_ALIAS_TO_FILE / this test."
    )
    assert imported, (
        "Expected the LoRA fork to import shared symbols from "
        "torchtune.dev.rl.*; found none."
    )


def test_injected_bindings_resolve_in_source(lora_source):
    """Every `name = _module.attr` class-body binding must point at a symbol
    that actually exists in the aliased source module.

    A rename of e.g. `_maybe_unpermute_qk` in weight_sync.py would otherwise
    only blow up at runtime on a compute node, after FSDP/vLLM init.
    """
    failures = []
    for lhs, alias, attr in _injected_bindings(lora_source):
        mod_file = MODULE_ALIAS_TO_FILE[alias]
        mod_src = _read(mod_file)
        if not _module_defines(mod_src, attr):
            failures.append(
                f"  {LORA_RECIPE}: `{lhs} = {alias}.{attr}` but "
                f"{mod_file.name} no longer defines `{attr}`"
            )
    assert not failures, (
        "LoRA fork binds shared symbols that no longer exist in their source "
        "module (rename/removal drift):\n" + "\n".join(failures)
    )


def test_rl_imports_resolve_in_source(lora_source):
    """Every `from torchtune.dev.rl.<mod> import <name>` in the fork must point
    at a symbol the target module actually defines.

    Covers the function-scoped imports the AST-inline-equiv checks miss
    (vllm_http_generate, batch_level_advantages, _save_raw_bytes, ...).
    """
    failures = []
    for dotted, name in _rl_imports(lora_source):
        mod_file = _rl_module_path(dotted)
        if not mod_file.exists():
            failures.append(f"  import `{name}` from missing module {dotted}")
            continue
        if not _module_defines(_read(mod_file), name):
            failures.append(
                f"  `from {dotted} import {name}` but {mod_file.name} "
                f"no longer defines `{name}`"
            )
    assert not failures, (
        "LoRA fork imports shared rl symbols that no longer exist "
        "(rename/removal drift):\n" + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# 2. Generalized server-mode attr contract.
#
# The shared `_setup_vllm_server_mode` reads several `self.<attr>` BEFORE the
# `if self._vllm_weight_sync:` gate. A standalone binder must set every such
# unconditional attr in its own __init__/setup, or setup() AttributeErrors at
# launch. The existing parity test hardcodes (_dp_replicate, _is_shard_leader);
# here we DERIVE the required attrs from the helper's real source so a newly
# added unconditional read is caught automatically.
# ---------------------------------------------------------------------------
def _unconditional_self_reads(func_src: str, func_name: str) -> set[str]:
    """Return the set of `self.<attr>` names READ before the first
    `if self._vllm_weight_sync:` gate inside `func_name`.

    We treat everything up to that gate as the unconditional prologue. Reads
    (ast.Load) only — assignments to self.<attr> are not requirements.
    """
    tree = ast.parse(func_src)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            target = node
            break
    assert target is not None, f"{func_name} not found in source"

    # Find the line of the first `if self._vllm_weight_sync:` gate, if any.
    gate_line = None
    for node in ast.walk(target):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Attribute)
                and isinstance(test.value, ast.Name)
                and test.value.id == "self"
                and test.attr == "_vllm_weight_sync"
            ):
                if gate_line is None or node.lineno < gate_line:
                    gate_line = node.lineno

    reads: set[str] = set()
    for node in ast.walk(target):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Load)
        ):
            if gate_line is None or node.lineno < gate_line:
                reads.add(node.attr)
    return reads


def _binds_server_mode(source: str) -> bool:
    return any(
        attr == "_setup_vllm_server_mode"
        for _lhs, _alias, attr in _injected_bindings(source)
    )


def _sets_self_attr(source: str, attr: str) -> bool:
    """True if the recipe assigns `self.<attr> = ...` anywhere."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and tgt.attr == attr
                ):
                    return True
    return False


def test_lora_fork_sets_all_unconditional_server_mode_attrs(lora_source):
    """The fork binds `_setup_vllm_server_mode`; assert it sets every
    `self.<attr>` that helper reads unconditionally (before the weight-sync
    gate). Derived from the helper's source so new unconditional reads added
    upstream are caught here, not at launch on a compute node.
    """
    assert _binds_server_mode(lora_source), (
        "Expected the LoRA fork to bind _setup_vllm_server_mode; it no longer "
        "does — update this test (the contract may have moved)."
    )
    helper_src = _read(MODULE_ALIAS_TO_FILE["_vllm_backend_module"])
    required = _unconditional_self_reads(helper_src, "_setup_vllm_server_mode")

    # Methods/attrs the helper resolves via the class itself are not __init__
    # state the fork must seed (e.g. bound helper methods, clients it creates).
    # Restrict to the plain scalar config/state attrs the prologue depends on.
    # `_build_tune_to_hf_map` etc. are method calls, not the risk class.
    ignore = {
        # created/owned by the helper itself or are bound methods, not state
        "_vllm_urls",  # set in fork __init__ already; harmless if present
    }
    required = {a for a in required if not a.endswith("__") and a not in ignore}

    # The two that historically drifted MUST be in the derived set — guards
    # against a refactor that hides them behind the gate (which would silently
    # weaken this check).
    for must in ("_dp_replicate", "_is_shard_leader"):
        assert must in required, (
            f"{must} is no longer read unconditionally by "
            "_setup_vllm_server_mode — verify the refactor is intentional and "
            "update this guard."
        )

    missing = sorted(a for a in required if not _sets_self_attr(lora_source, a))
    assert not missing, (
        f"{LORA_RECIPE} binds _setup_vllm_server_mode but never sets "
        f"{missing}, which the helper reads unconditionally. setup() will "
        "AttributeError at launch. Set them in __init__ (single-replicate "
        "defaults: _dp_replicate=1, _is_shard_leader=_is_rank_zero)."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

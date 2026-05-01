# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Parity / structural tests for the train() refactor that removed the
180-line BioReason subclass override of train().

History: the subclass previously copy-pasted the base train() loop and
silently dropped the post-optimizer ``_sync_weights_to_vllm`` block. vLLM
served stale (SFT-init) weights for an entire run before the bug was caught
(see memory: project_bioreason_train_missing_wsync). To make that class of
regression structurally impossible, the subclass now overrides only a small
``_extract_batch_kwargs(batch) -> dict`` hook, and the base train() splats
the result into ``generate_trajectory_batched``.

These tests pin the structural invariants without importing the recipe
modules (recipes is intentionally not an importable package).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_PATH = REPO_ROOT / "recipes/dev/grpo_full_finetune_distributed_xpu.py"
SUB_PATH = REPO_ROOT / "recipes/dev/grpo_bioreason_distributed_xpu.py"
BASE_CLS = "GRPOFullFinetuneDistributedXPU"
SUB_CLS = "GRPOBioReasonDistributedXPU"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)
    }


@pytest.fixture(scope="module")
def base_cls() -> ast.ClassDef:
    return _find_class(_parse(BASE_PATH), BASE_CLS)


@pytest.fixture(scope="module")
def sub_cls() -> ast.ClassDef:
    return _find_class(_parse(SUB_PATH), SUB_CLS)


def test_base_defines_train_and_hook(base_cls):
    methods = _methods(base_cls)
    assert "train" in methods, "base must define train()"
    assert "_extract_batch_kwargs" in methods, (
        "base must define the _extract_batch_kwargs hook"
    )


def test_subclass_does_not_override_train(sub_cls):
    """Subclass MUST NOT define train(). If it does, the post-optimizer
    weight-sync block can drift out of sync silently — that exact regression
    has happened once already (project_bioreason_train_missing_wsync)."""
    methods = _methods(sub_cls)
    assert "train" not in methods, (
        "GRPOBioReasonDistributedXPU must inherit train() from the base "
        "recipe. Override _extract_batch_kwargs / generate_trajectory_batched "
        "instead — see project_bioreason_train_missing_wsync."
    )


def test_subclass_overrides_extract_batch_kwargs(sub_cls):
    methods = _methods(sub_cls)
    assert "_extract_batch_kwargs" in methods, (
        "subclass must override _extract_batch_kwargs to forward "
        "protein_sequences into generate_trajectory_batched"
    )
    src = ast.unparse(methods["_extract_batch_kwargs"])
    assert "protein_sequences" in src, (
        "_extract_batch_kwargs must mention protein_sequences (multimodal "
        "batch field forwarded to generate_trajectory_batched)"
    )


def test_base_extract_batch_kwargs_default_empty(base_cls):
    """The base hook returns {} so non-multimodal recipes call
    generate_trajectory_batched(tokens, answers) with no extra kwargs."""
    src = ast.unparse(_methods(base_cls)["_extract_batch_kwargs"])
    # Must contain a Return with an empty dict literal.
    assert "return {}" in src.replace(" ", "").replace("\n", "") or \
        "return{}" in src.replace(" ", "").replace("\n", ""), (
        "base _extract_batch_kwargs must return {} by default"
    )


def test_train_calls_extract_batch_kwargs_and_splats(base_cls):
    """The hook must be called and splat-applied into generate_trajectory_batched —
    otherwise multimodal subclasses lose protein_sequences silently."""
    train = _methods(base_cls)["train"]
    src = ast.unparse(train)
    assert "self._extract_batch_kwargs(batch)" in src, (
        "train() must call self._extract_batch_kwargs(batch)"
    )
    # The kwargs must reach generate_trajectory_batched as **splat.
    # Accept either of: generate_trajectory_batched(tokens, answers, **_extra_gen_kwargs)
    # or any name bound from _extract_batch_kwargs.
    assert "generate_trajectory_batched" in src
    # Find Call to generate_trajectory_batched and confirm it has a Starred
    # double-star (**) keyword expansion.
    found = False
    for node in ast.walk(train):
        if isinstance(node, ast.Call):
            fn = node.func
            name = (
                fn.attr if isinstance(fn, ast.Attribute) else
                getattr(fn, "id", None)
            )
            if name == "generate_trajectory_batched":
                if any(kw.arg is None for kw in node.keywords):
                    found = True
                    break
    assert found, (
        "generate_trajectory_batched must be called with **kwargs splat — "
        "otherwise multimodal kwargs are dropped"
    )


def test_train_syncs_weights_after_optimizer(base_cls):
    """Pin the post-optimizer weight-sync sequence — the exact thing the
    BioReason subclass once silently dropped."""
    train = _methods(base_cls)["train"]
    src = ast.unparse(train)
    # All three vLLM modes must wire in a sync after optimizer.step().
    assert "self._optimizer.step()" in src, "train() must call optimizer.step()"
    assert "_sync_colocated_weights" in src, (
        "colocate vLLM mode must call _sync_colocated_weights"
    )
    assert "_sync_dedicated_vllm_weights" in src, (
        "dedicated_rank vLLM mode must call _sync_dedicated_vllm_weights"
    )
    assert "_sync_weights_to_vllm" in src, (
        "server vLLM mode must call _sync_weights_to_vllm"
    )
    # Order: optimizer.step() must appear before the sync calls in source.
    opt_idx = src.index("self._optimizer.step()")
    for sync_name in (
        "_sync_colocated_weights",
        "_sync_dedicated_vllm_weights",
        "_sync_weights_to_vllm",
    ):
        sync_idx = src.index(sync_name)
        assert sync_idx > opt_idx, (
            f"{sync_name} must appear AFTER optimizer.step() in train() — "
            "weight sync precedes optimizer step is a correctness bug"
        )


def test_train_short_circuits_vllm_rank(base_cls):
    """vLLM ranks must skip training entirely — this short-circuit lives in
    train() (or its setup helpers it calls). We assert the marker is present."""
    train = _methods(base_cls)["train"]
    src = ast.unparse(train)
    # The marker is the dedicated short-circuit branch around line 3207.
    # The subclass used to dup this; with the refactor it must live in base train.
    assert "_is_vllm_rank" in src, (
        "train() must reference _is_vllm_rank to short-circuit dedicated vLLM "
        "ranks (otherwise they enter the train loop and deadlock)"
    )

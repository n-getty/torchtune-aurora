# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe structural guard for the colocate warmup-at-max fix.

The LoRA-GRPO colocate path crashes with ``banned:1`` once a mid-run rollout
first exceeds all prior sequence lengths: on XPU we never ``empty_cache`` (the
FSDP UR-handle-leak guard), so the larger vLLM-KV + FSDP activation buffers grow
``reserved`` monotonically until a GPU page-fault. ``_warmup_at_max`` front-loads
those peak buffers at step 0 by running a max-length generate + ref forward +
real ``grpo_step`` fwd/bwd before the train loop, so the curve is flat from
step 0 (see ``docs/reports/lora_colocate_4b_20260618.md``).

The recipe imports torchao + XPU backends, so it cannot be imported on a login
node. These tests inspect the recipe SOURCE/AST instead (same approach as
``test_recipe_family_correctness_parity.py``) to pin down that the warmup:

  * exists as a method,
  * is gated to colocate AND honors the documented disable flag,
  * is actually CALLED from train() (and on all ranks, i.e. NOT behind an
    ``if self._is_rank_zero`` guard — the FSDP fwd/bwd inside it is collective),
  * runs the real ``grpo_step`` (so it allocates the true training-backward peak,
    not a cheap proxy) and discards the warmup gradients.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RECIPE = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "dev"
    / "lora_grpo_full_finetune_distributed_xpu.py"
)
FLAG = "TORCHTUNE_COLOCATE_WARMUP_AT_MAX"


def _source() -> str:
    assert RECIPE.exists(), f"recipe not found: {RECIPE}"
    return RECIPE.read_text()


def _func_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def test_warmup_method_exists():
    tree = ast.parse(_source())
    _func_node(tree, "_warmup_at_max")  # raises if missing


def test_warmup_gated_to_colocate_and_flag():
    """_warmup_at_max must early-return off colocate and respect the disable flag."""
    src = ast.get_source_segment(_source(), _func_node(ast.parse(_source()), "_warmup_at_max"))
    assert src is not None
    assert "self._colocate" in src, "warmup must be gated on self._colocate"
    assert "return" in src, "warmup must early-return when not applicable"
    assert FLAG in src, f"warmup must honor the documented {FLAG} disable flag"


def test_warmup_runs_real_grpo_step_and_clears_grads():
    """Warmup must exercise the true backward peak (grpo_step) then zero grads.

    A cheap proxy forward would NOT allocate the FSDP reduce-scatter / activation
    buffers that the real backward does — defeating the front-loading. And the
    warmup grads are garbage (synthetic batch), so they must be discarded before
    step 0.
    """
    src = ast.get_source_segment(
        _source(), _func_node(ast.parse(_source()), "_warmup_at_max")
    )
    assert "self.grpo_step(" in src, "warmup must run the real grpo_step (true bwd peak)"
    assert "zero_grad" in src, "warmup must discard its synthetic-batch gradients"
    # vLLM KV peak: a max-length generate with ignore_eos so it runs full length.
    assert "ignore_eos" in src, "warmup vLLM generate must ignore_eos to hit peak KV"


def test_train_calls_warmup_on_all_ranks():
    """train() must call _warmup_at_max, and NOT behind an is_rank_zero guard.

    The FSDP forward/backward inside the warmup is a collective — every rank must
    enter it together or the warmup deadlocks. We assert the call statement is at
    the same nesting level as the surrounding train() body (not inside an
    ``if self._is_rank_zero`` block).
    """
    tree = ast.parse(_source())
    train = _func_node(tree, "train")

    # Find the call to self._warmup_at_max() and check none of its ancestor
    # statements within train() is an `if self._is_rank_zero` test.
    found = False
    for node in ast.walk(train):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_warmup_at_max"
        ):
            found = True
    assert found, "train() must call self._warmup_at_max()"

    # Structural rank-zero-guard check: collect line ranges of `if self._is_rank_zero`
    # blocks and ensure the warmup call line is outside all of them.
    src = _source()
    warmup_call_line = None
    for i, line in enumerate(src.splitlines(), start=1):
        if "_warmup_at_max()" in line and "def " not in line:
            warmup_call_line = i
            break
    assert warmup_call_line is not None

    for node in ast.walk(train):
        if isinstance(node, ast.If):
            test = node.test
            is_rank_zero = (
                isinstance(test, ast.Attribute)
                and test.attr == "_is_rank_zero"
            )
            if is_rank_zero and node.body:
                start = node.lineno
                end = node.end_lineno
                assert not (start <= warmup_call_line <= end), (
                    "self._warmup_at_max() must NOT be inside an "
                    "`if self._is_rank_zero` block — the FSDP fwd/bwd it runs is "
                    "a collective and would deadlock on non-zero ranks."
                )


def test_flag_documented_in_claude_md():
    """The disable flag must appear in the CLAUDE.md env-var table (doc<->code)."""
    claude = RECIPE.resolve().parents[2] / "CLAUDE.md"
    assert claude.exists(), f"CLAUDE.md not found: {claude}"
    assert FLAG in claude.read_text(), (
        f"{FLAG} must be documented in the CLAUDE.md env-var table "
        "(test_documented_env_flags_exist.py enforces this separately too)."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

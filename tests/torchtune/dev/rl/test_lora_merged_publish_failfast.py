# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe guard for the LoRA-GRPO merged-weight publish fail-fast contract.

Background — the correctness hazard this pins down:
  The DEFAULT adapter-delivery path is the merged-weight publish
  (``use_runtime_lora=False``). Each step rank 0 builds W_eff and POSTs it to
  every vLLM tile via ``_publish_merged_weights_background`` → ``_post_one``.

  If a POST fails (HTTP != 200, or a transport exception), ``_post_one`` must
  NOT swallow the failure silently. The recipe's safety net is the next-step
  join on ``_publish_thread`` which inspects ``self._publish_error`` and aborts
  training. That net only fires if ``_publish_merged_weights_background`` (and
  therefore the ``_bg`` wrapper that sets ``_publish_error``) actually raises.

  The original code logged a warning/error in ``_post_one`` and returned
  normally — so a failed publish left ``_publish_error=None``, the join passed,
  and training silently continued generating rollouts against STALE vLLM
  weights. GRPO semantics require the generation policy to track the training
  policy; stale weights make every subsequent rollout off-policy without any
  signal. This is a silent correctness hazard, and it is the default path.

  The runtime/legacy path (``_publish_lora_background``) already raises on
  POST failure (RuntimeError on the failed-tile set) — only the merged path
  drifted.

This test enforces that the merged publish path surfaces POST failures so the
fail-fast join can catch them. It is pure source inspection — no torch, no
device, no HTTP.
"""
from __future__ import annotations

import ast
from pathlib import Path

RECIPE = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "dev"
    / "lora_grpo_full_finetune_distributed_xpu.py"
)


def _func_source(method_name: str) -> str:
    """Return the source of a top-level method of the recipe class by name."""
    src = RECIPE.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"method {method_name} not found in {RECIPE}")


def test_merged_publish_tracks_post_failures():
    """_publish_merged_weights_background must track per-POST failures and raise.

    Either the inner _post_one returns a status the caller inspects, or the
    function accumulates failures and raises a RuntimeError. A bare
    log.warning/log.error + return that is never re-raised is the regression.
    """
    body = _func_source("_publish_merged_weights_background")
    # The function must contain an explicit `raise` that fires on POST failure.
    # We look for a RuntimeError raise that references failures/tiles/urls.
    assert "raise RuntimeError" in body, (
        "_publish_merged_weights_background swallows POST failures: no "
        "`raise RuntimeError` found. A failed merged-weight POST leaves "
        "_publish_error=None, the next-step join passes, and training "
        "silently continues against STALE vLLM weights (off-policy GRPO). "
        "Track per-URL failures and raise so the fail-fast join catches it."
    )


def test_runtime_publish_still_fails_fast():
    """Sanity: the legacy runtime path already raises on POST failure.

    This guards against a 'fix' to the merged path accidentally removing the
    runtime path's existing fail-fast (the two share the join safety net).
    """
    body = _func_source("_publish_lora_background")
    assert "raise RuntimeError" in body, (
        "_publish_lora_background lost its fail-fast raise on load_lora_adapter "
        "failure — the runtime path would now train against stale vLLM weights."
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

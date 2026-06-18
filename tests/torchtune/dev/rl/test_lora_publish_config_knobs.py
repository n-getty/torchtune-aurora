# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe guard that LoRA-GRPO publish timeouts are config-driven, not literal.

These pin the config-ergonomics fixes from the 2026-06-17 hardening sweep:
the HTTP/collective_rpc/rsync/ssh timeouts and the merged-weight POST timeout
must read from cfg (with sensible defaults) so a slow shared FS or a large
model can be tuned per-run instead of requiring a code edit. The defaults are
preserved, so off-path behaviour is byte-identical to before the change.

Pure source inspection — no torch, no device, no HTTP.
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


def _src() -> str:
    return RECIPE.read_text()


def _init_assigns() -> set[str]:
    """Set of `self._x` attribute names assigned in __init__."""
    src = _src()
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and isinstance(
                    sub.value, ast.Name
                ):
                    if sub.value.id == "self" and isinstance(
                        getattr(sub, "ctx", None), ast.Store
                    ):
                        names.add(sub.attr)
    return names


def test_publish_timeouts_are_config_driven():
    """The merged collective_rpc and publish-join timeouts must be cfg-driven.

    After the hardening sweep, __init__ should define dedicated timeout attrs
    (e.g. _publish_join_timeout, _collective_rpc_timeout) sourced from cfg.
    """
    assigned = _init_assigns()
    expected = {
        "_publish_join_timeout",
        "_collective_rpc_timeout",
    }
    missing = expected - assigned
    assert not missing, (
        f"LoRA-GRPO __init__ does not define config-driven timeout attrs "
        f"{missing}. These replace the hardcoded 120s join / 600s "
        f"collective_rpc literals so a slow FS or large model is tunable "
        f"without a code edit."
    )


def test_collective_rpc_uses_config_attr_not_literal_600():
    """The merged-weight POST must use the config attr, not a literal 600."""
    src = _src()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_publish_merged_weights_background"
        ):
            body = ast.get_source_segment(src, node)
            assert "_collective_rpc_timeout" in body, (
                "_publish_merged_weights_background still uses a hardcoded "
                "timeout for the collective_rpc POST; route it through "
                "self._collective_rpc_timeout."
            )
            return
    raise AssertionError("_publish_merged_weights_background not found")


def test_join_timeout_uses_config_attr():
    """The train-loop and cleanup publish joins must use the config attr."""
    src = _src()
    # Both join sites should reference the config attr rather than `join(timeout=120)`.
    assert "_publish_join_timeout" in src, (
        "publish thread join sites still hardcode 120s; route through "
        "self._publish_join_timeout."
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

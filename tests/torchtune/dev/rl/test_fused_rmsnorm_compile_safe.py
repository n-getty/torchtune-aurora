# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU-safe regression guard for the fused-RMSNorm / torch.compile interaction.

WHY A SOURCE CHECK (not a compile test): the fused path only runs on XPU (on CPU
``FusedRMSNorm.forward`` takes the eager fallback), so a CPU ``torch.compile`` never
traces into the ``@triton.jit`` kernel and cannot reproduce the failure. The bug is
therefore invisible to any CPU runtime test and to the eager-only autokernel suite.

THE BUG (HW-caught, fused+compile A/B job 8676886, 2026-07-16): under torch.compile
(a BioReason SFT production default), Inductor re-codegens any ``@triton.jit`` kernel
reached during tracing into a fresh module that imports triton as ``triton`` and
triton.language as ``tl``. If the kernel *source* references private aliases
(``_tl`` / ``_triton``), the regenerated code raises
``torch._inductor.exc.InductorError: NameError: name '_tl' is not defined`` at Inductor
compile time — crashing step 0. The invariant below pins the conventional aliasing so a
future refactor can't silently reintroduce the private names.
"""
import ast
import inspect

import torchtune.modules._fused_rmsnorm_xpu as fused_mod


def _kernel_source_blocks():
    """Return the source of every top-level triton.jit-decorated kernel in the module."""
    src = inspect.getsource(fused_mod)
    tree = ast.parse(src)
    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            deco_src = " ".join(
                ast.dump(d) for d in node.decorator_list
            )
            if "jit" in deco_src or node.name.endswith("_kernel"):
                blocks.append(ast.get_source_segment(src, node))
    return blocks


def test_triton_kernels_use_conventional_aliases():
    """The @triton.jit kernel bodies must use ``tl.`` / ``triton.`` (never ``_tl``/``_triton``)
    so Inductor's re-codegen under torch.compile resolves the names. This is the exact
    invariant that job 8676886 violated."""
    blocks = _kernel_source_blocks()
    assert blocks, "no triton kernels found — did the module structure change?"
    joined = "\n".join(blocks)
    # The kernel BODIES must not reference the private aliases.
    assert "_tl." not in joined, (
        "fused RMSNorm triton kernel uses private alias `_tl` — Inductor re-codegen under "
        "torch.compile will NameError. Use the conventional `tl` alias (see job 8676886)."
    )
    assert "_triton." not in joined, (
        "fused RMSNorm triton kernel uses private alias `_triton` — use `triton`."
    )
    # And they SHOULD use the conventional ones (sanity that the check isn't vacuous).
    assert "tl." in joined, "expected conventional `tl.` usage in the kernel body"


def test_module_level_triton_aliases_are_conventional():
    """Module-level triton imports must bind the conventional names so the kernel source
    (which Inductor copies verbatim) resolves at re-codegen time."""
    src = inspect.getsource(fused_mod)
    # The import must bind `triton` and `tl`, not `_triton` / `_tl`.
    assert "import triton.language as tl" in src, (
        "expected `import triton.language as tl` (conventional alias for Inductor codegen)"
    )
    assert "import triton.language as _tl" not in src, (
        "private alias `_tl` reintroduced — breaks torch.compile (job 8676886)"
    )

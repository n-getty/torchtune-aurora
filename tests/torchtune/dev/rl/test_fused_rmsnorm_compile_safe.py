# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CPU-safe regression guard for the fused-kernel / torch.compile interaction.

Covers BOTH fused Triton modules that a recipe can swap in before sharding:
``_fused_rmsnorm_xpu`` and ``_fused_rope_xpu``.

WHY A SOURCE CHECK (not a compile test): the fused paths only run on XPU (on CPU the
fused modules take the eager fallback), so a CPU ``torch.compile`` never traces into the
``@triton.jit`` kernels and cannot reproduce the failure. The bug is therefore invisible
to any CPU runtime test and to the eager-only autokernel suite.

THE BUG (RMSNorm HW-caught, fused+compile A/B job 8676886, 2026-07-16; RoPE was the
identical latent bug, never surfaced only because fused RoPE is off by default): under
torch.compile (a BioReason SFT production default), Inductor re-codegens any
``@triton.jit`` kernel reached during tracing into a fresh module that imports triton as
``triton`` and triton.language as ``tl``. If the kernel *source* references private
aliases (``_tl`` / ``_triton``), the regenerated code raises
``torch._inductor.exc.InductorError: NameError: name '_tl' is not defined`` at Inductor
compile time — crashing step 0. The invariant below pins the conventional aliasing so a
future refactor can't silently reintroduce the private names in either module.
"""
import ast
import inspect

import pytest

import torchtune.modules._fused_rmsnorm_xpu as fused_rmsnorm_mod
import torchtune.modules._fused_rope_xpu as fused_rope_mod

_FUSED_MODULES = [
    pytest.param(fused_rmsnorm_mod, id="rmsnorm"),
    pytest.param(fused_rope_mod, id="rope"),
]


def _kernel_source_blocks(mod):
    """Return the source of every top-level triton.jit-decorated kernel in the module."""
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            deco_src = " ".join(ast.dump(d) for d in node.decorator_list)
            if "jit" in deco_src or node.name.endswith("_kernel"):
                blocks.append(ast.get_source_segment(src, node))
    return blocks


@pytest.mark.parametrize("mod", _FUSED_MODULES)
def test_triton_kernels_use_conventional_aliases(mod):
    """The @triton.jit kernel bodies must use ``tl.`` / ``triton.`` (never ``_tl``/``_triton``)
    so Inductor's re-codegen under torch.compile resolves the names. This is the exact
    invariant that job 8676886 violated (RMSNorm) and that RoPE shared latently."""
    blocks = _kernel_source_blocks(mod)
    assert blocks, f"no triton kernels found in {mod.__name__} — did the structure change?"
    joined = "\n".join(blocks)
    assert "_tl." not in joined, (
        f"{mod.__name__} triton kernel uses private alias `_tl` — Inductor re-codegen under "
        "torch.compile will NameError. Use the conventional `tl` alias (see job 8676886)."
    )
    assert "_triton." not in joined, (
        f"{mod.__name__} triton kernel uses private alias `_triton` — use `triton`."
    )
    # And they SHOULD use the conventional ones (sanity that the check isn't vacuous).
    assert "tl." in joined, f"expected conventional `tl.` usage in {mod.__name__} kernel body"


@pytest.mark.parametrize("mod", _FUSED_MODULES)
def test_module_level_triton_aliases_are_conventional(mod):
    """Module-level triton imports must bind the conventional names so the kernel source
    (which Inductor copies verbatim) resolves at re-codegen time."""
    src = inspect.getsource(mod)
    assert "import triton.language as tl" in src, (
        f"{mod.__name__}: expected `import triton.language as tl` (conventional alias)"
    )
    assert "import triton.language as _tl" not in src, (
        f"{mod.__name__}: private alias `_tl` reintroduced — breaks torch.compile (job 8676886)"
    )

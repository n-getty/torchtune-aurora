# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-safe regression tests for the XPU SFT recipes.

The two recipes (``recipes/dev/full_finetune_distributed_xpu.py`` and
``recipes/dev/lora_finetune_distributed_xpu.py``) contain an XPU/XCCL
compatibility shim at module top that pre-registers ``torchtune`` in
``sys.modules`` and calls ``install_xpu_patches()`` before any torchtune
submodule import. The shim must:

  * Not require an XPU to import (login-node + CI compatibility).
  * Leave ``sys.modules['torchtune']`` populated as a ``ModuleType``.
  * Expose the renamed recipe class symbol.

These tests run in a fresh subprocess each so that the
``sys.modules['torchtune']`` shim installed by the recipe does not leak
into other tests in the same pytest run.
"""

import os
import subprocess
import sys

import pytest


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _run_import(recipe_path: str, class_name: str) -> subprocess.CompletedProcess:
    """Load the recipe module by file path in a clean subprocess and assert the
    class exists. Recipes live under ``recipes/`` which intentionally raises on
    package import, so we use ``importlib.util.spec_from_file_location``."""
    abs_path = os.path.join(REPO_ROOT, recipe_path)
    script = (
        "import importlib.util, sys, types\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        f"spec = importlib.util.spec_from_file_location('_sft_xpu_under_test', {abs_path!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['_sft_xpu_under_test'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"assert hasattr(mod, {class_name!r}), 'missing class: ' + {class_name!r}\n"
        "tt = sys.modules.get('torchtune')\n"
        "assert isinstance(tt, types.ModuleType), 'torchtune sys.modules entry invalid'\n"
        f"print('OK:', {recipe_path!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        cwd=REPO_ROOT,
        timeout=120,
    )


@pytest.mark.parametrize(
    "recipe_path,classname",
    [
        (
            "recipes/dev/full_finetune_distributed_xpu.py",
            "FullFinetuneRecipeDistributedXPU",
        ),
        (
            "recipes/dev/lora_finetune_distributed_xpu.py",
            "LoRAFinetuneRecipeDistributedXPU",
        ),
        (
            "recipes/dev/full_finetune_moe_distributed_xpu.py",
            "FullFinetuneMoEDistributedXPU",
        ),
    ],
)
def test_xpu_sft_recipe_imports_on_cpu_host(recipe_path: str, classname: str) -> None:
    """Recipe must import on a no-XPU host without raising."""
    cp = _run_import(recipe_path, classname)
    assert cp.returncode == 0, (
        f"Import of {recipe_path} failed.\n"
        f"--- stdout ---\n{cp.stdout}\n"
        f"--- stderr ---\n{cp.stderr}"
    )
    assert f"OK: {recipe_path}" in cp.stdout


def test_auroragpt_tokenizer_has_sft_interface() -> None:
    """AuroraGPTTokenizer must implement tokenize_messages + __call__ so the
    standard ``SFTDataset`` / ``SFTTransform`` can consume it (without an HF
    chat template). Regression: the original tokenizer only inherited the
    SentencePiece base interface and the GRPO custom dataset bypassed this
    contract, so SFT use was silently broken until the SFT methods landed."""
    from torchtune.dev.rl.ezpz_tasks import AuroraGPTTokenizer

    assert hasattr(AuroraGPTTokenizer, "tokenize_messages")
    assert hasattr(AuroraGPTTokenizer, "__call__")

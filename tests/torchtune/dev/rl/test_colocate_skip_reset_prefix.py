# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe guard test for TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX.

The LoRA-GRPO colocate weight-publish path calls
``self._vllm_llm.llm_engine.reset_prefix_cache()`` once per publish. That call
mutates the vLLM KV block table and was a suspect for the colocate generation
page fault, so it is gated behind ``TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX`` for
A/B isolation (experiments/colocate/run_colocate_ab.sh, cell=noreset).

The A/B established reset_prefix_cache is NOT the trigger (the run still faults
with it skipped), so the gate stays DEFAULT-OFF (the call still fires by
default). This test pins two things so the default-safe behavior cannot
silently regress:

  1. the env-gate predicate (`os.environ.get(flag,"0") != "1"`) — default fires,
     only "1" skips;
  2. the call site in the recipe is actually wrapped by that predicate (source
     scan — no recipe import, which would pull torchao + XPU backends).

Runs on a login node: no torch, no XPU, no distributed init.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE = (
    REPO_ROOT / "recipes" / "dev" / "lora_grpo_full_finetune_distributed_xpu.py"
)
FLAG = "TORCHTUNE_COLOCATE_SKIP_RESET_PREFIX"


def _gate(env_val):
    """Mirror the recipe predicate: the call fires UNLESS the flag == '1'."""
    return os.environ.get(FLAG, "0") != "1" if env_val is None else env_val != "1"


def test_default_fires_reset_prefix():
    # Unset / "0" / anything-not-"1" → reset_prefix_cache() STILL runs (default-safe).
    assert _gate("0") is True
    assert _gate(None) is True
    assert _gate("false") is True


def test_flag_one_skips():
    assert _gate("1") is False


def test_recipe_call_site_is_gated():
    """The colocate in-process reset_prefix_cache() call must be guarded by the flag.

    Target ONLY the in-process colocate call
    ``self._vllm_llm.llm_engine.reset_prefix_cache()`` — the two server-mode
    ``pool.map(lambda c: c.reset_prefix_cache(), ...)`` calls are a different
    (HTTP-client) path and are intentionally ungated.
    """
    src = RECIPE.read_text()
    assert FLAG in src, f"{FLAG} not found in recipe — guard removed?"
    m = re.search(r"self\._vllm_llm\.llm_engine\.reset_prefix_cache\(\)", src)
    assert m is not None, "in-process colocate reset_prefix_cache() call not found"
    # The guarding `if os.environ.get(FLAG...)` must appear in the lines just
    # before the call (immediately preceding statement).
    preceding = src[max(0, m.start() - 200) : m.start()]
    assert FLAG in preceding, (
        "in-process reset_prefix_cache() is not guarded by "
        f"{FLAG} — default-safe gate missing or moved"
    )
    assert "os.environ.get" in preceding


def test_default_safe_call_still_present():
    """Guard must not have deleted the call — default behavior unchanged."""
    src = RECIPE.read_text()
    assert "reset_prefix_cache()" in src

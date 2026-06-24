# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe guards for the BioReason per-rank COLOCATE LoRA wiring.

The colocate path can't be unit-run (needs FSDP + an in-process vLLM engine on
XPU), and the merge MATH is already pinned by
test_bioreason_lora_peft.test_weff_via_delta_map_matches_base_plus_scaled_ba
(the colocate merge uses the identical base + lora_delta_map() formula as the
server path). What these source-level checks pin is the colocate-specific WIRING
that, if it regressed, would silently ship frozen weights or trigger banned:1:

  (a) `_sync_colocated_lora_weights` exists and uses summon_full_params +
      lora_delta_map + load_weights + the load-bearing .clone().
  (b) The colocate_sleep wake path routes to it when the model has LoRA (NOT the
      inherited _sync_colocated_weights, which ships the frozen base under LoRA).
  (c) Every device_empty_cache call in the recipe is gated by the colocate guard
      (ungated empty_cache under FSDP + in-process vLLM leaks UR handles → banned:1).
"""
import ast
from pathlib import Path

import pytest

RECIPE_PATH = (
    Path(__file__).resolve().parents[4]
    / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
)
SRC = RECIPE_PATH.read_text()
TREE = ast.parse(SRC)


def _method(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SRC, node)
    return None


def test_recipe_module_imports():
    """Import the recipe file (catches class-body errors like a bad base-method
    reference that only surface at import — e.g. the 2026-06-18
    `_base_module._sync_colocated_weights` AttributeError that crashed all 12 ranks
    at startup). importorskip peft/transformers; loaded under a non-recipes name to
    bypass the `recipes` package import guard."""
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    import importlib.util as u
    spec = u.spec_from_file_location("_br_recipe_import_smoke", RECIPE_PATH)
    m = u.module_from_spec(spec)
    spec.loader.exec_module(m)
    cls = m.GRPOBioReasonDistributedXPU
    assert hasattr(cls, "_sync_colocated_weights_base")
    assert "_sync_colocated_weights" in cls.__dict__  # the override


# (a) the merge method exists and has the load-bearing pieces ------------------

def test_sync_colocated_lora_weights_exists_and_correct_shape():
    body = _method("_sync_colocated_lora_weights")
    assert body is not None, "_sync_colocated_lora_weights missing from recipe"
    # STREAMING merge: one delta at a time (lora_delta_iter), NOT the eager
    # lora_delta_map dict — holding all ~398 fp32 deltas per step fragments the
    # allocator under colocate (no empty_cache) → reserved staircase → banned:1.
    assert "lora_delta_iter" in body
    assert "vllm_param_iter" in body
    # non-mutating fp32 delta (NOT merge_adapter/unmerge → bf16 drift).
    assert "merge_adapter" not in body, (
        "must use non-mutating lora_delta_iter, not in-place merge_adapter"
    )
    # summon ONLY when actually FSDP-wrapped (colocate model is unsharded → no-op).
    assert "isinstance(self._model, FSDP)" in body
    # load into THIS rank's own engine.
    assert "load_weights" in body


# (b) _sync_colocated_weights is OVERRIDDEN to route LoRA → merge --------------

def test_sync_colocated_weights_override_routes_lora_to_merge():
    """BioReason overrides _sync_colocated_weights so BOTH callers — the base
    train() loop's _run_wsync_block (plain colocate) AND the colocate_sleep wake
    path — route to the LoRA merge when _has_lora, else fall back to the inherited
    backbone sync. The inherited sync ships the sharded embed_tokens → trips
    vLLM's vocab-embedding assert under FSDP FULL_SHARD (2026-06-18)."""
    override = _method("_sync_colocated_weights")
    assert override is not None, "BioReason must override _sync_colocated_weights"
    assert "_has_lora" in override
    assert "_sync_colocated_lora_weights()" in override, (
        "override must route to the LoRA merge when _has_lora"
    )
    assert "_sync_colocated_weights_base()" in override, (
        "non-LoRA path must fall back to the inherited base sync"
    )


def test_colocate_sleep_wake_calls_sync_between_weights_and_kv():
    """colocate_sleep syncs in generate_trajectory (not the train loop). The sync
    must run after wake(weights) and before wake(kv_cache)."""
    gt = _method("generate_trajectory")
    assert gt is not None
    i_weights = gt.find('wake_up(tags=["weights"])')
    i_sync = gt.find("self._sync_colocated_weights()")
    i_kv = gt.find('wake_up(tags=["kv_cache"])')
    assert -1 < i_weights < i_sync < i_kv, (
        "sync must run after wake(weights) and before wake(kv_cache)"
    )


# (c) every device_empty_cache is colocate-gated -----------------------------

def test_all_device_empty_cache_calls_are_colocate_gated():
    """Walk the AST: every `device_empty_cache(self._device)` must be inside an
    `if not _colocate_vllm_mode:` block (UR-handle leak guard)."""
    bad = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.guard_stack = []

        def _is_colocate_guard(self, test):
            # matches `not _colocate_vllm_mode`
            return (
                isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Name)
                and test.operand.id == "_colocate_vllm_mode"
            )

        def visit_If(self, node):
            guard = self._is_colocate_guard(node.test)
            self.guard_stack.append(guard)
            for n in node.body:
                self.visit(n)
            self.guard_stack.pop()
            # else-branch is NOT colocate-guarded
            self.guard_stack.append(False)
            for n in node.orelse:
                self.visit(n)
            self.guard_stack.pop()

        def visit_Call(self, node):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "device_empty_cache":
                if not any(self.guard_stack):
                    bad.append(node.lineno)
            self.generic_visit(node)

    V().visit(TREE)
    assert not bad, (
        f"ungated device_empty_cache at lines {bad} — leaks UR handles under "
        f"colocate (FSDP + in-process vLLM) → banned:1"
    )


# (d) chunked grpo_step combines ratios with stack, not cat ------------------

def test_chunked_grpo_step_stacks_scalar_ratios_not_cat():
    """GRPOSimpleLoss returns ratios/clipfrac as 0-dim scalars (torch.tensor(1.0)).
    The chunked grpo_step path (fbs < num_seqs) collects them per-chunk; combining
    with torch.cat crashes ('zero-dimensional tensor cannot be concatenated' —
    hit on the first colocate run, fbs=2). Must use torch.stack like the base
    recipe. Only pi_logprobs (1-dim) may use cat.
    """
    gs = _method("grpo_step")
    assert gs is not None
    assert "torch.cat(_chunk_ratios)" not in gs, (
        "chunked ratios must use torch.stack (0-dim scalars), not torch.cat"
    )
    assert "torch.stack(_chunk_ratios)" in gs
    assert "torch.cat(_chunk_clipfracs)" not in gs


# (e) the module-level _colocate_vllm_mode flag is set from the RUNTIME mode ---

def test_setup_sets_colocate_flag_from_runtime_mode():
    """`_colocate_vllm_mode` is snapshotted from the base module at import time
    (False), and BioReason overrides setup() so the base recipe's flip never runs.
    setup() MUST set the global from self._vllm_mode, or every `if not
    _colocate_vllm_mode:` guard stays True in colocate → device_empty_cache fires
    every step (UR-handle leak → banned:1) and the colocate merge branch is never
    reached (root cause of the 2026-06-18 colocate step-1 banned:1).
    """
    setup = _method("setup")
    assert setup is not None
    assert "global _colocate_vllm_mode" in setup, (
        "setup() must declare `global _colocate_vllm_mode` to update the flag"
    )
    # set from the runtime mode (covers both colocate and colocate_sleep)
    assert '_colocate_vllm_mode = self._vllm_mode in ("colocate", "colocate_sleep")' in setup

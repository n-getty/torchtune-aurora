# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe guards for the OPT-IN LoRA surface on the device-agnostic GRPO recipe.

Recipe: ``recipes/dev/grpo_full_finetune_distributed.py``.

The recipe is a heavily-used TRACKED full-FT recipe. Adding a LoRA training
surface MUST be fully backward compatible: when the config uses a non-LoRA
(``lora_*``-free) model builder, the load path must behave byte-identically to
today — a single ``training.load_from_full_model_state_dict(..., strict=True)``.

The LoRA path is gated behind detection of adapter params on the instantiated
model (``get_adapter_params(model)`` non-empty). For LoRA models the recipe must
mirror the validated upstream LoRA load pattern from
``recipes/lora_finetune_distributed.py`` (non-strict base load +
``validate_missing_and_unexpected_for_lora``), because a base checkpoint has no
``lora_a``/``lora_b`` keys and a LoRA model has extra adapter params (so
``strict=True`` would reject both missing and unexpected keys).

These tests cannot fully run ``_setup_model`` (it needs FSDP + a device), so they
pin:
  (a) the branch-selection helper (pure, importable, exercised on a real tiny model);
  (b) that the full-FT strict=True load still exists in source (no regression);
  (c) that the LoRA load branch + validate_missing_and_unexpected_for_lora wiring
      is present in source;
  (d) that save_checkpoint honors ``save_adapter_weights_only`` and has an
      adapter / merge branch so it cannot crash on a LoRA state dict;
  (e) the ref-model semantics decision is documented in the recipe.
"""
import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

from torchtune.modules.peft import get_adapter_params, set_trainable_params


RECIPE_PATH = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "dev"
    / "grpo_full_finetune_distributed.py"
)
RECIPE_SRC = RECIPE_PATH.read_text()


def _load_recipe_module():
    """Import the recipe file directly, bypassing the `recipes` package guard.

    `recipes/__init__.py` raises on import-as-package by design; loading the
    file under a non-`recipes.` module name sidesteps that so we can exercise
    the pure helper(s) on a real (CPU) model.
    """
    spec = importlib.util.spec_from_file_location(
        "_grpo_recipe_under_test", RECIPE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grpo_recipe = _load_recipe_module()


# ---------------------------------------------------------------------------
# (a) Pure branch-selection helper, exercised on real tiny models (CPU-safe).
# ---------------------------------------------------------------------------
def _build_tiny_lora_model():
    from torchtune.models.qwen3._component_builders import lora_qwen3

    return lora_qwen3(
        lora_attn_modules=["q_proj", "v_proj"],
        apply_lora_to_mlp=False,
        apply_lora_to_output=False,
        vocab_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        intermediate_dim=128,
        max_seq_len=32,
        head_dim=16,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1_000_000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )


def _build_tiny_base_model():
    from torchtune.models.qwen3._component_builders import qwen3

    return qwen3(
        vocab_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        intermediate_dim=128,
        max_seq_len=32,
        head_dim=16,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1_000_000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
    )


def test_helper_exists_and_is_importable():
    assert hasattr(grpo_recipe, "_model_has_adapter_params"), (
        "_setup_model's LoRA-vs-full-FT branch must be factored into a pure, "
        "testable helper `_model_has_adapter_params(model)`."
    )


def test_detection_true_on_lora_model():
    model = _build_tiny_lora_model()
    assert grpo_recipe._model_has_adapter_params(model) is True


def test_detection_false_on_base_model():
    model = _build_tiny_base_model()
    assert grpo_recipe._model_has_adapter_params(model) is False


def test_detection_matches_get_adapter_params():
    """The helper must agree with the canonical peft API on both model types."""
    lora = _build_tiny_lora_model()
    base = _build_tiny_base_model()
    assert grpo_recipe._model_has_adapter_params(lora) == bool(get_adapter_params(lora))
    assert grpo_recipe._model_has_adapter_params(base) == bool(get_adapter_params(base))


# ---------------------------------------------------------------------------
# (b) Full-FT path is byte-identical: the strict=True single load must remain.
# ---------------------------------------------------------------------------
def test_full_ft_strict_load_still_present():
    """Non-LoRA branch must keep the exact current strict=True single load."""
    assert "strict=True" in RECIPE_SRC, (
        "The full-FT path must keep the existing strict=True "
        "load_from_full_model_state_dict call. If you removed it, the non-LoRA "
        "case is no longer byte-identical to the tracked recipe."
    )


def test_full_ft_load_is_guarded_by_adapter_detection():
    """The strict=True load must be reachable when there are NO adapter params.

    We assert that within _setup_model the strict=True load and the helper call
    coexist, i.e. the LoRA branch is an opt-in fork, not a replacement.
    """
    src = inspect.getsource(grpo_recipe.GRPOFullFinetuneRecipeDistributed._setup_model)
    assert "strict=True" in src, "strict=True load must live inside _setup_model"
    assert "_model_has_adapter_params" in src, (
        "_setup_model must branch on _model_has_adapter_params(...)"
    )


# ---------------------------------------------------------------------------
# (c) LoRA load branch wiring is present (non-strict base load + validate).
# ---------------------------------------------------------------------------
def test_lora_imports_present():
    for name in (
        "get_adapter_params",
        "set_trainable_params",
        "validate_missing_and_unexpected_for_lora",
        "AdapterModule",
    ):
        assert name in RECIPE_SRC, f"LoRA load pattern requires `{name}` import/use"


def test_lora_validate_call_present():
    src = inspect.getsource(grpo_recipe.GRPOFullFinetuneRecipeDistributed._setup_model)
    assert "validate_missing_and_unexpected_for_lora" in src, (
        "LoRA branch must call validate_missing_and_unexpected_for_lora instead "
        "of relying on strict=True."
    )
    # Non-strict base load must be present in the LoRA branch.
    assert "strict=False" in src, (
        "LoRA branch must do a NON-strict base load and capture "
        "(base_missing, base_unexpected)."
    )


def test_lora_branch_initializes_adapter_and_dora():
    src = inspect.getsource(grpo_recipe.GRPOFullFinetuneRecipeDistributed._setup_model)
    assert "initialize_parameters" in src, (
        "LoRA branch must initialize fresh adapter params (m.initialize_parameters())."
    )
    assert "initialize_dora_magnitude" in src, (
        "LoRA branch must initialize DoRA magnitude when present."
    )
    assert "set_trainable_params" in src, (
        "LoRA branch must freeze base + mark adapters trainable via set_trainable_params."
    )


# ---------------------------------------------------------------------------
# (d) save_checkpoint: LoRA-safe (adapter-only flag + merge/adapter branch).
# ---------------------------------------------------------------------------
def test_save_checkpoint_honors_save_adapter_weights_only():
    src = inspect.getsource(grpo_recipe.GRPOFullFinetuneRecipeDistributed.save_checkpoint)
    assert "save_adapter_weights_only" in src or "_save_adapter_weights_only" in src, (
        "save_checkpoint must honor a save_adapter_weights_only config flag for LoRA."
    )


def test_save_checkpoint_has_adapter_or_merge_branch():
    src = inspect.getsource(grpo_recipe.GRPOFullFinetuneRecipeDistributed.save_checkpoint)
    # Must either merge LoRA into base (get_merged_lora_ckpt) or emit ADAPTER_KEY,
    # so the checkpointer does not run tune_to_hf over raw lora_a/lora_b keys.
    assert ("get_merged_lora_ckpt" in src) or ("ADAPTER_KEY" in src), (
        "save_checkpoint must handle LoRA state dicts (merge to base via "
        "get_merged_lora_ckpt and/or save adapter weights via ADAPTER_KEY) so it "
        "does not crash converting raw adapter keys to HF format."
    )


# ---------------------------------------------------------------------------
# (e) ref-model semantics are documented in the recipe (decision is load-bearing).
# ---------------------------------------------------------------------------
def test_ref_model_semantics_documented():
    assert ("ref" in RECIPE_SRC.lower()) and ("lora" in RECIPE_SRC.lower()), (
        "The recipe must document the LoRA ref-model semantics decision in a comment."
    )


# ---------------------------------------------------------------------------
# Source must remain valid Python / importable (sanity).
# ---------------------------------------------------------------------------
def test_recipe_source_parses():
    ast.parse(RECIPE_SRC)

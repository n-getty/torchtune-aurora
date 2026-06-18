# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe structural + hygiene invariants for the LoRA SFT XPU path.

Three groups:

1. Adapter-only checkpoint structure. The configs set
   ``save_adapter_weights_only: true`` and build an ``_adapter_config`` from the
   model's LoRA hparams. The checkpoint the GRPO recipe consumes must contain
   ONLY adapter tensors (lora_a/lora_b/magnitude) for the configured target
   modules — no base weights — and the ``_adapter_config['target_modules']``
   must match the layers that actually carry LoRA. A mismatch here ships a
   checkpoint the downstream PEFT loader silently mis-keys.

2. SFT loader path-discovery hygiene. ``glob.glob`` hangs on DAOS/dfuse
   (CLAUDE.md). The SFT dataset module must not contain it (mirrors the
   existing BioReason path-discovery guard).

3. Recipe XPU-correctness hazards. The LoRA SFT XPU recipe must not reintroduce
   the known XPU footguns: ``empty_cache`` in the training loop, ``glob.glob``,
   per-module FSDP wrapping, or ``device_id=`` in process-group init. These are
   grep-level guards — cheap insurance against a future copy-paste from the
   CUDA recipe.
"""
from pathlib import Path

import pytest

import torch

from torchtune.models.llama3._component_builders import lora_llama3
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    set_trainable_params,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE = REPO_ROOT / "recipes" / "dev" / "lora_finetune_distributed_xpu.py"
SFT_LOADER = REPO_ROOT / "torchtune" / "dev" / "sft" / "auroragpt_math_mix.py"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Adapter-only checkpoint structure
# ──────────────────────────────────────────────────────────────────────────────

LORA_ATTN = ["q_proj", "v_proj"]  # matches the SFT-LoRA configs


def _tiny_lora_llama3(apply_mlp=False, apply_out=False):
    return lora_llama3(
        lora_attn_modules=LORA_ATTN,
        apply_lora_to_mlp=apply_mlp,
        apply_lora_to_output=apply_out,
        vocab_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        max_seq_len=32,
        intermediate_dim=128,
        lora_rank=8,
        lora_alpha=16,
    )


def test_adapter_state_dict_is_adapter_only():
    model = _tiny_lora_llama3()
    set_trainable_params(model, get_adapter_params(model))

    adapter_sd = get_adapter_state_dict(model.state_dict())
    assert len(adapter_sd) > 0, "no adapter tensors found"

    for k in adapter_sd:
        # Only LoRA / DoRA tensors — never a plain base weight.
        assert ("lora_a" in k or "lora_b" in k or "magnitude" in k), (
            f"adapter checkpoint leaked a non-adapter tensor: {k}"
        )
        # Only the configured attn target modules carry adapters here.
        assert ("q_proj" in k or "v_proj" in k), (
            f"adapter key outside configured target modules: {k}"
        )


def test_adapter_config_target_modules_match_lora_layers():
    """Mirror the recipe's _adapter_config construction and assert the declared
    target_modules cover every module that actually has an adapter tensor."""
    target_modules = get_lora_module_names(
        LORA_ATTN, apply_lora_to_mlp=False, apply_lora_to_output=False
    )
    adapter_config = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": target_modules,
        "peft_type": "LORA",
    }
    assert adapter_config["target_modules"] == LORA_ATTN

    model = _tiny_lora_llama3()
    set_trainable_params(model, get_adapter_params(model))
    adapter_sd = get_adapter_state_dict(model.state_dict())

    # Every adapter key's module must be in target_modules.
    declared = set(adapter_config["target_modules"])
    for k in adapter_sd:
        mod = next((m for m in declared if m in k), None)
        assert mod is not None, (
            f"adapter tensor {k} belongs to a module not in target_modules "
            f"{declared} — PEFT loader would mis-key it"
        )


def test_apply_lora_to_mlp_adds_mlp_targets():
    """When apply_lora_to_mlp=True the adapter keys and target_modules must both
    expand to include the MLP projections — a guard that the two stay in lockstep."""
    target_modules = get_lora_module_names(
        LORA_ATTN, apply_lora_to_mlp=True, apply_lora_to_output=False
    )
    model = _tiny_lora_llama3(apply_mlp=True)
    set_trainable_params(model, get_adapter_params(model))
    adapter_sd = get_adapter_state_dict(model.state_dict())

    mlp_modules = {"w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"}
    has_mlp_key = any(any(m in k for m in mlp_modules) for k in adapter_sd)
    assert has_mlp_key, "apply_lora_to_mlp=True produced no MLP adapter tensors"
    # And target_modules grew accordingly.
    assert len(target_modules) > len(LORA_ATTN)


# ──────────────────────────────────────────────────────────────────────────────
# 2. SFT loader path-discovery hygiene
# ──────────────────────────────────────────────────────────────────────────────


def test_sft_loader_has_no_glob_glob():
    src = SFT_LOADER.read_text()
    assert "glob.glob" not in src, (
        "auroragpt_math_mix.py uses glob.glob — it hangs on DAOS/dfuse (CLAUDE.md)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Recipe XPU-correctness hazards (grep-level guards)
# ──────────────────────────────────────────────────────────────────────────────


def _recipe_src():
    return RECIPE.read_text()


def test_recipe_no_empty_cache_in_loop():
    src = _recipe_src()
    # The recipe must never call empty_cache (FSDP + empty_cache leaks UR handles).
    assert "empty_cache" not in src, (
        "lora_finetune_distributed_xpu.py calls empty_cache — banned under FSDP on XPU."
    )


def test_recipe_no_glob_glob():
    assert "glob.glob" not in _recipe_src()


def test_recipe_no_device_id_in_pg_init():
    src = _recipe_src()
    # device_id= in init_process_group hangs DataLoader workers on XPU multi-node.
    # The recipe goes through init_xpu_process_group (device_index, not device_id).
    assert "init_process_group(" in src or "init_xpu_process_group(" in src
    assert "device_id=" not in src, (
        "device_id= in process-group init hangs DataLoader workers on XPU."
    )


def test_recipe_uses_top_level_fsdp_shard():
    """Per-module FSDP wrapping is catastrophic on XPU; the recipe must use the
    shared training.shard_model path (top-level), not fully_shard per submodule."""
    src = _recipe_src()
    assert "training.shard_model(" in src
    # No raw per-module fully_shard loop.
    assert "fully_shard(" not in src, (
        "recipe calls fully_shard directly — use training.shard_model (top-level)."
    )


def test_recipe_routes_pg_init_through_xpu_helper():
    src = _recipe_src()
    assert "init_xpu_process_group(" in src, (
        "recipe must init the PG via init_xpu_process_group on the xpu path."
    )


def test_recipe_no_torch_cuda_memory_record_on_xpu_path():
    """Upstream uses torch.cuda.memory._record_memory_history; the XPU fork must
    route memory profiling through the device-agnostic helper instead."""
    src = _recipe_src()
    assert "torch.cuda.memory._record_memory_history" not in src, (
        "recipe still calls the CUDA-only memory recorder; use "
        "device_record_memory_history."
    )
    assert "device_record_memory_history" in src

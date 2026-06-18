# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe config-validity invariants for the AuroraGPT LoRA SFT configs.

These configs feed ``recipes/dev/lora_finetune_distributed_xpu.py`` and produce
the adapter checkpoints the AGPT-2B GRPO recipe consumes. The recipe has almost
no XPU test coverage, so the config surface is the cheapest place to catch a
broken handoff before a node is ever held.

What each test pins (no XPU, no distributed init):
  * Every SFT-LoRA YAML parses with OmegaConf and resolves interpolations.
  * The mandatory keys the recipe reads in ``__init__`` / ``setup`` are present
    (the XPU recipe makes ``lr_scheduler`` MANDATORY — see the smoke YAML's
    own comment — so a missing block would crash at setup, not parse).
  * ``save_adapter_weights_only`` is True (the configs advertise adapter-only
    handoff; if this silently flips to False the GRPO side gets a full merged
    checkpoint it isn't wired to consume cheaply).
  * The model component is a LoRA builder and exposes the LoRA hyperparameters
    the recipe reads unconditionally in ``_setup_model``
    (``lora_rank`` / ``lora_alpha`` / ``lora_attn_modules`` / ``apply_lora_to_mlp``).
  * ``device: xpu`` and ``dtype`` is bf16/fp32 (the recipe raises on fp16).
  * ``tensor_parallel_dim`` (if set) is 1 — the recipe asserts this.
"""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PROD_DIR = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "configs"
    / "dev"
    / "production"
)

# The two configs explicitly paired (via header comment) with
# recipes/dev/lora_finetune_distributed_xpu.py.
SFT_LORA_CONFIGS = [
    "auroragpt_2b_sft_lora_gsm8k_xpu.yaml",
    "auroragpt_2b_sft_alpaca_smoke_lora_xpu.yaml",
]


def _load(name: str):
    path = PROD_DIR / name
    assert path.is_file(), f"expected SFT-LoRA config missing: {path}"
    cfg = OmegaConf.load(str(path))
    # Force interpolation resolution so a broken ${...} surfaces here.
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_config_parses_and_resolves(name):
    cfg = _load(name)
    assert cfg is not None


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_mandatory_recipe_keys_present(name):
    cfg = _load(name)
    # Keys the recipe reads via cfg.<key> (NOT cfg.get) — a missing one is a
    # hard crash in __init__/setup, not a graceful default.
    for key in (
        "device",
        "dtype",
        "model",
        "tokenizer",
        "checkpointer",
        "dataset",
        "optimizer",
        "loss",
        "lr_scheduler",  # XPU recipe: mandatory (cfg.lr_scheduler, not cfg.get)
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps_per_epoch",
        "seed",
        "shuffle",
        "resume_from_checkpoint",
        "output_dir",
        "metric_logger",
    ):
        assert key in cfg, f"{name}: missing mandatory key '{key}'"


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_adapter_only_handoff(name):
    cfg = _load(name)
    assert cfg.get("save_adapter_weights_only", False) is True, (
        f"{name}: save_adapter_weights_only must be True — these configs "
        "advertise an adapter-only checkpoint for the GRPO handoff."
    )


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_model_is_lora_with_required_hparams(name):
    cfg = _load(name)
    comp = cfg.model.get("_component_")
    assert comp is not None and "lora" in comp.lower(), (
        f"{name}: model._component_ ({comp}) is not a LoRA builder"
    )
    # Read exactly the attrs _setup_model touches unconditionally.
    for attr in ("lora_rank", "lora_alpha", "lora_attn_modules", "apply_lora_to_mlp"):
        assert attr in cfg.model, f"{name}: model missing LoRA hparam '{attr}'"
    assert int(cfg.model.lora_rank) > 0
    assert float(cfg.model.lora_alpha) > 0
    assert len(list(cfg.model.lora_attn_modules)) > 0


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_device_and_dtype(name):
    cfg = _load(name)
    assert cfg.device == "xpu", f"{name}: device must be xpu"
    # Recipe raises ValueError on fp16.
    assert str(cfg.dtype).lower() in ("bf16", "fp32", "float32"), (
        f"{name}: dtype {cfg.dtype} is not supported (fp16 raises)."
    )


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_tensor_parallel_dim_is_one(name):
    cfg = _load(name)
    # Recipe asserts tensor_parallel_dim == 1.
    assert int(cfg.get("tensor_parallel_dim", 1)) == 1, (
        f"{name}: tensor_parallel_dim must be 1 (recipe asserts this)."
    )


@pytest.mark.parametrize("name", SFT_LORA_CONFIGS)
def test_loss_is_instantiable_component(name):
    cfg = _load(name)
    comp = cfg.loss.get("_component_")
    assert comp is not None, f"{name}: loss._component_ missing"
    # Must be importable (caught at parse-time of the component path).
    from torchtune.config._utils import _get_component_from_path

    obj = _get_component_from_path(comp)
    assert obj is not None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Pure helper functions for LoRA-GRPO on Aurora/XPU.
#
# Importable from the recipe and (future) BioReason-LoRA subclass.
# No distributed imports at module level — safe to import on login nodes.

import json
import os
import re
import shutil
from typing import Optional

import torch
from torch import nn

# torchtune parameter name → HuggingFace parameter name for attention + MLP
# Derived from torchtune/models/qwen3/_convert_weights.py _FROM_HF (inverted).
# Keys are the torchtune module path (without lora suffix), values are HF module path.
_TUNE_MODULE_TO_HF: dict[str, str] = {
    "attn.q_proj":     "self_attn.q_proj",
    "attn.k_proj":     "self_attn.k_proj",
    "attn.v_proj":     "self_attn.v_proj",
    "attn.output_proj": "self_attn.o_proj",   # output_proj → o_proj
    "mlp.w1":          "mlp.gate_proj",       # w1 = gate_proj
    "mlp.w2":          "mlp.down_proj",       # w2 = down_proj
    "mlp.w3":          "mlp.up_proj",         # w3 = up_proj
}

# PEFT target_modules used in adapter_config.json (HF names)
_ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
_MLP_TARGET_MODULES  = ["gate_proj", "up_proj", "down_proj"]


def build_qwen3_lora_model(cfg) -> nn.Module:
    """Instantiate a LoRA-wrapped Qwen3 model and set adapter-only trainability.

    Calls ``cfg.model._component_`` (typically ``lora_qwen3_4b_base``), then
    marks only adapter parameters as trainable.  Returns the model ready for
    FSDP wrap — caller is responsible for sharding.

    Args:
        cfg: OmegaConf DictConfig with a ``model`` subsection.

    Returns:
        nn.Module with base weights frozen and LoRA adapter weights trainable.
    """
    from torchtune import config
    from torchtune.modules.peft import get_adapter_params, set_trainable_params

    model = config.instantiate(cfg.model)
    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)
    return model


def adapter_optimizer_params(model: nn.Module) -> list:
    """Return the list of adapter ``nn.Parameter`` objects that should be optimized.

    Uses ``get_adapter_params`` to find all LoRA weights, returns the values
    (parameters) in a flat list suitable for passing to an optimizer.

    Args:
        model: The LoRA-wrapped model (possibly FSDP-wrapped).

    Returns:
        list[nn.Parameter]: adapter parameters only.
    """
    from torchtune.modules.peft import get_adapter_params

    return list(get_adapter_params(model).values())


def _strip_fsdp_prefixes(key: str) -> str:
    """Remove FSDP and activation-checkpoint wrapper prefixes from a state-dict key."""
    key = key.replace("_fsdp_wrapped_module.", "")
    key = key.replace("_checkpoint_wrapped_module.", "")
    return key


def _translate_lora_key(tune_key: str) -> Optional[str]:
    """Translate one torchtune LoRA adapter key to PEFT format.

    torchtune: ``layers.{i}.attn.q_proj.lora_a.weight``
    PEFT:      ``base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight``

    Returns None for keys that are not LoRA adapter keys (e.g. base weights).
    """
    key = _strip_fsdp_prefixes(tune_key)

    # Match: layers.{i}.<module_path>.lora_{a|b}.weight
    m = re.match(r"^layers\.(\d+)\.(.+)\.(lora_[ab]\.weight)$", key)
    if m is None:
        return None

    layer_idx  = m.group(1)
    module_path = m.group(2)
    lora_suffix = m.group(3)

    hf_module = _TUNE_MODULE_TO_HF.get(module_path)
    if hf_module is None:
        return None

    # lora_a → lora_A, lora_b → lora_B (PEFT convention)
    hf_lora_suffix = lora_suffix.replace("lora_a.", "lora_A.").replace("lora_b.", "lora_B.")

    # PEFT prefix: base_model.model. + HF model prefix (model.)
    return f"base_model.model.model.layers.{layer_idx}.{hf_module}.{hf_lora_suffix}"


def torchtune_to_peft_state_dict(
    adapter_sd: dict,
    model_name: str,
    rank: int,
    alpha: float,
    target_modules: list[str],
) -> tuple[dict, dict]:
    """Translate a torchtune LoRA adapter state dict into PEFT format.

    Produces both the PEFT weight dict and the ``adapter_config.json`` content
    that vLLM's ``_create_merged_loras_inplace`` expects.  The q/k/v → qkv_proj
    fusion (if any) is handled by vLLM automatically; we ship them unfused.

    Args:
        adapter_sd: Flat state dict of adapter tensors only (keys contain "lora").
        model_name: Base model path string (stored in adapter_config).
        rank: LoRA rank (``r`` in PEFT).
        alpha: LoRA alpha scaling factor.
        target_modules: HF module names that LoRA is applied to
            (e.g. ``["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]``).

    Returns:
        ``(peft_state_dict, adapter_config_dict)`` — both suitable for writing
        directly to an adapter directory via :func:`write_peft_adapter_dir`.

    Raises:
        ValueError: if any key in ``adapter_sd`` cannot be translated.
    """
    peft_sd: dict[str, torch.Tensor] = {}
    untranslatable = []

    for tune_key, tensor in adapter_sd.items():
        peft_key = _translate_lora_key(tune_key)
        if peft_key is None:
            untranslatable.append(tune_key)
            continue
        peft_sd[peft_key] = tensor.cpu().to(torch.float32)

    if untranslatable:
        raise ValueError(
            f"Could not translate {len(untranslatable)} adapter key(s) to PEFT format. "
            f"First few: {untranslatable[:5]}"
        )

    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }

    return peft_sd, adapter_config


def write_peft_adapter_dir(
    peft_sd: dict,
    adapter_config: dict,
    path: str,
) -> None:
    """Write a PEFT adapter directory atomically.

    Writes ``adapter_model.safetensors`` and ``adapter_config.json`` to
    a temporary directory (``path + ".tmp"``), then renames it to ``path``
    so readers never see a partially-written directory.

    Uses ``os.listdir`` + filter instead of ``glob.glob`` to avoid DAOS/dfuse
    hang (Aurora platform constraint).

    Args:
        peft_sd: PEFT-format state dict (from :func:`torchtune_to_peft_state_dict`).
        adapter_config: Adapter config dict (from :func:`torchtune_to_peft_state_dict`).
        path: Target directory path (will be created / replaced atomically).
    """
    try:
        from safetensors.torch import save_file as _save_safetensors
        _has_safetensors = True
    except ImportError:
        _has_safetensors = False

    tmp_path = path + ".tmp"

    # Remove stale tmp dir if present (crash recovery)
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    # Write weights
    if _has_safetensors:
        # safetensors requires contiguous float32 tensors
        sd_contiguous = {k: v.contiguous().float() for k, v in peft_sd.items()}
        _save_safetensors(sd_contiguous, os.path.join(tmp_path, "adapter_model.safetensors"))
    else:
        # Fallback: save as PyTorch binary (vLLM can load both)
        torch.save(peft_sd, os.path.join(tmp_path, "adapter_model.bin"))

    # Write config
    with open(os.path.join(tmp_path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Atomic rename: replace any existing dir at `path`
    if os.path.isdir(path):
        # os.rename cannot replace a non-empty dir on Linux; remove first
        shutil.rmtree(path)
    os.rename(tmp_path, path)


def make_lora_request(name: str, lora_int_id: int, path: str):
    """Create a vLLM ``LoRARequest`` for offline engine use.

    Args:
        name: Human-readable adapter name (e.g. ``"rl_step_5"``).
        lora_int_id: Integer ID used by vLLM's internal adapter cache.
        path: Local filesystem path to the PEFT adapter directory.

    Returns:
        ``vllm.lora.request.LoRARequest`` instance.
    """
    from vllm.lora.request import LoRARequest
    return LoRARequest(lora_name=name, lora_int_id=lora_int_id, lora_path=path)


def load_lora_adapter_http(
    session,
    base_url: str,
    lora_name: str,
    lora_path: str,
    timeout: int = 120,
) -> bool:
    """Load a LoRA adapter on a running vLLM HTTP server.

    Calls ``POST /v1/load_lora_adapter`` with the adapter name and shared-FS
    path.  The path must be visible from the vLLM server process.

    Args:
        session: ``requests.Session`` from VLLMClient.
        base_url: vLLM server base URL (e.g. ``"http://10.1.2.3:8001"``).
        lora_name: Adapter name as it will appear in model field of future requests.
        lora_path: Shared-FS path to the PEFT adapter directory.
        timeout: HTTP timeout in seconds.

    Returns:
        True on success (HTTP 200 or 201), False otherwise.
    """
    url = f"{base_url}/v1/load_lora_adapter"
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    try:
        r = session.post(url, json=payload, timeout=timeout)
        return r.status_code in (200, 201)
    except Exception:
        return False


def unload_lora_adapter_http(
    session,
    base_url: str,
    lora_name: str,
    timeout: int = 30,
) -> bool:
    """Unload a LoRA adapter from a running vLLM HTTP server.

    Args:
        session: ``requests.Session`` from VLLMClient.
        base_url: vLLM server base URL.
        lora_name: Adapter name to remove.
        timeout: HTTP timeout in seconds.

    Returns:
        True on success, False otherwise.
    """
    url = f"{base_url}/v1/unload_lora_adapter"
    payload = {"lora_name": lora_name}
    try:
        r = session.post(url, json=payload, timeout=timeout)
        return r.status_code in (200, 201)
    except Exception:
        return False

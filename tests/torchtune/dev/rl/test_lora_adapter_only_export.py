# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe parity test for the rank-0 adapter-only export path.

Pins the contract that replacing the FSDP ``FULL_STATE_DICT`` gather with a
rank-0 ``named_parameters()`` snapshot produces a byte-identical PEFT state
dict (modulo the upstream fp32 cast that both paths perform). This is the
core invariant behind P1 of the LoRA-GRPO performance plan.

What we guard:
  - The set of PEFT keys produced by the new path is identical to the legacy
    full-gather path.
  - Tensor values are bit-equal after the dtype cast both paths apply.
  - The adapter tensor count matches ``adapter_optimizer_params`` (the
    surface the optimizer trains and the manual all-reduce sweeps).
  - Snapshot keys carrying simulated FSDP wrapping prefixes still translate
    cleanly via ``_strip_fsdp_prefixes`` inside ``_translate_lora_key``.
"""
import torch

from torchtune.dev.rl.lora_helpers import (
    adapter_optimizer_params,
    torchtune_to_peft_state_dict,
)
from torchtune.modules.peft import (
    get_adapter_params,
    get_adapter_state_dict,
    set_trainable_params,
)
from torchtune.models.qwen3._component_builders import lora_qwen3


RANK = 4
ALPHA = 8.0
NUM_LAYERS = 2
ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "output_proj"]
MLP_TARGET_MODULES_HF = ["gate_proj", "down_proj", "up_proj"]


def _build_tiny_lora_model():
    """Tiny Qwen3-LoRA model on CPU (matches recipe wiring: attn + MLP)."""
    return lora_qwen3(
        lora_attn_modules=ATTN_TARGET_MODULES,
        apply_lora_to_mlp=True,
        apply_lora_to_output=False,
        vocab_size=64,
        num_layers=NUM_LAYERS,
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
        lora_rank=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
    )


def _new_path_snapshot(model):
    """Mirror the new ``_gather_lora_state_dict`` body — rank-0 named_parameters
    filtered for adapter keys, dtype-cast on CPU."""
    return {
        n: p.detach().cpu().to(torch.float32)
        for n, p in model.named_parameters()
        if "lora_a" in n or "lora_b" in n
    }


def _legacy_path_snapshot(model):
    """Mirror the legacy ``FULL_STATE_DICT`` → ``get_adapter_state_dict`` path
    (single-process equivalent — no FSDP wrap). Casts to fp32 on CPU to match
    what the legacy path produced before handing to ``torchtune_to_peft_state_dict``."""
    full_sd = model.state_dict()
    adapter_sd = get_adapter_state_dict(full_sd, device="cpu")
    return {k: v.to(torch.float32) for k, v in adapter_sd.items()}


def _peft_targets():
    return sorted(ATTN_TARGET_MODULES + MLP_TARGET_MODULES_HF)


def test_new_path_tensor_count_matches_adapter_optimizer_params():
    model = _build_tiny_lora_model()
    set_trainable_params(model, get_adapter_params(model))

    snapshot = _new_path_snapshot(model)
    expected = len(list(adapter_optimizer_params(model)))

    assert expected > 0, "model must expose adapter_optimizer_params"
    assert len(snapshot) == expected, (
        f"snapshot tensor count ({len(snapshot)}) does not match "
        f"adapter_optimizer_params ({expected}) — recipe assertion would fire"
    )
    # 4 attn + 3 mlp modules × 2 (lora_a/b) × 2 layers = 28
    assert len(snapshot) == (len(ATTN_TARGET_MODULES) + 3) * 2 * NUM_LAYERS


def test_new_path_peft_keys_match_legacy():
    model = _build_tiny_lora_model()
    set_trainable_params(model, get_adapter_params(model))

    new_sd = _new_path_snapshot(model)
    legacy_sd = _legacy_path_snapshot(model)

    new_peft, _ = torchtune_to_peft_state_dict(
        new_sd,
        model_name="base",
        rank=RANK,
        alpha=ALPHA,
        target_modules=_peft_targets(),
    )
    legacy_peft, _ = torchtune_to_peft_state_dict(
        legacy_sd,
        model_name="base",
        rank=RANK,
        alpha=ALPHA,
        target_modules=_peft_targets(),
    )

    assert set(new_peft.keys()) == set(legacy_peft.keys()), (
        f"PEFT key set diverged. only_in_new={set(new_peft) - set(legacy_peft)}; "
        f"only_in_legacy={set(legacy_peft) - set(new_peft)}"
    )


def test_new_path_peft_tensors_bit_equal_to_legacy():
    model = _build_tiny_lora_model()
    set_trainable_params(model, get_adapter_params(model))
    # Perturb adapters off zero so equality is non-trivial.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_a" in n or "lora_b" in n:
                p.copy_(torch.randn_like(p))

    new_sd = _new_path_snapshot(model)
    legacy_sd = _legacy_path_snapshot(model)
    new_peft, _ = torchtune_to_peft_state_dict(
        new_sd, model_name="base", rank=RANK, alpha=ALPHA,
        target_modules=_peft_targets(),
    )
    legacy_peft, _ = torchtune_to_peft_state_dict(
        legacy_sd, model_name="base", rank=RANK, alpha=ALPHA,
        target_modules=_peft_targets(),
    )

    assert set(new_peft.keys()) == set(legacy_peft.keys())
    for k in new_peft:
        a, b = new_peft[k], legacy_peft[k]
        assert a.dtype == b.dtype == torch.float32
        assert a.shape == b.shape, f"shape mismatch at {k}: {a.shape} vs {b.shape}"
        assert torch.equal(a, b), f"tensor diverged at PEFT key {k}"


def test_new_path_handles_fsdp_wrapped_keys():
    """Simulate FSDP1 ``use_orig_params=True`` keys carrying
    ``_fsdp_wrapped_module.`` prefixes — translation must succeed unchanged."""
    model = _build_tiny_lora_model()
    set_trainable_params(model, get_adapter_params(model))

    raw = _new_path_snapshot(model)
    # Simulate FSDP wrap by prepending the wrapper prefix to every key.
    wrapped = {f"_fsdp_wrapped_module.{k}": v for k, v in raw.items()}

    raw_peft, _ = torchtune_to_peft_state_dict(
        raw, model_name="base", rank=RANK, alpha=ALPHA,
        target_modules=_peft_targets(),
    )
    wrapped_peft, _ = torchtune_to_peft_state_dict(
        wrapped, model_name="base", rank=RANK, alpha=ALPHA,
        target_modules=_peft_targets(),
    )

    assert set(raw_peft.keys()) == set(wrapped_peft.keys())
    for k in raw_peft:
        assert torch.equal(raw_peft[k], wrapped_peft[k]), (
            f"FSDP-prefixed key produced different PEFT tensor at {k}"
        )

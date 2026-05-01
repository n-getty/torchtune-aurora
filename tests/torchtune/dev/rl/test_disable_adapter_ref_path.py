# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe test: disable_adapter() as the ref-model path for LoRA-GRPO.

In LoRA-GRPO, we avoid a separate FSDP-wrapped ref model copy by using
``disable_adapter(model)`` as a context manager. This saves ~8 GiB HBM.

What we guard:
  - ``disable_adapter`` is a functional context manager (enter + exit work).
  - Model forward outputs differ between adapter-enabled and adapter-disabled
    modes (i.e., the adapter is actually contributing to the computation).
  - After exiting the context, ``module.disabled`` is reset to False on all
    LoRA modules (no state leak across calls).
  - Re-entering the context on the next step also works (idempotent).
"""
import torch
import pytest

from torchtune.modules.peft import (
    disable_adapter,
    get_adapter_params,
    set_trainable_params,
)
from torchtune.models.qwen3._component_builders import lora_qwen3


NUM_LAYERS = 2
SEQ_LEN = 4
BATCH = 1


def _build_tiny_lora_model():
    model = lora_qwen3(
        lora_attn_modules=["q_proj", "v_proj"],
        apply_lora_to_mlp=False,
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
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )
    adapter_params = get_adapter_params(model)
    set_trainable_params(model, adapter_params)
    return model


def _forward(model, tokens):
    with torch.no_grad():
        return model(tokens)


def test_disable_adapter_context_manager_is_callable():
    model = _build_tiny_lora_model()
    tokens = torch.randint(0, 64, (BATCH, SEQ_LEN))
    # Should not raise
    with disable_adapter(model):
        _forward(model, tokens)


def test_adapter_and_base_outputs_differ():
    """LoRA lora_b is zero-init, so adapter mode == base mode at init time.

    We manually set lora_a and lora_b to non-zero values to force divergence.
    """
    model = _build_tiny_lora_model()

    # Non-zero-init the adapters so they actually affect the output.
    adapter_params = get_adapter_params(model)
    for name, param in adapter_params.items():
        if "lora_b" in name:
            torch.nn.init.normal_(param, std=0.5)

    tokens = torch.randint(0, 64, (BATCH, SEQ_LEN))

    with torch.no_grad():
        out_adapted = _forward(model, tokens)

    with disable_adapter(model):
        with torch.no_grad():
            out_base = _forward(model, tokens)

    assert not torch.allclose(out_adapted, out_base, atol=1e-4), (
        "Adapter and base outputs should differ when lora_b is non-zero. "
        "If they are identical, disable_adapter may not be working or the "
        "adapter is not contributing to the forward pass."
    )


def test_disable_adapter_restores_state():
    """After the context exits, all adapter modules must be re-enabled."""
    model = _build_tiny_lora_model()

    from torchtune.modules.peft.lora import LoRALinear

    with disable_adapter(model):
        for module in model.modules():
            if isinstance(module, LoRALinear):
                assert getattr(module, "disabled", False), (
                    "LoRALinear.disabled must be True inside disable_adapter context"
                )

    # After context exit, all adapters must be enabled again
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert not getattr(module, "disabled", False), (
                "LoRALinear.disabled must be False after disable_adapter context exits"
            )


def test_disable_adapter_idempotent_across_steps():
    """Simulates two sequential training steps using disable_adapter for ref logprobs."""
    model = _build_tiny_lora_model()
    tokens = torch.randint(0, 64, (BATCH, SEQ_LEN))

    for step in range(3):
        with disable_adapter(model):
            _forward(model, tokens)  # ref logprob forward

        # Normal adapter-enabled forward (policy logprobs)
        _forward(model, tokens)

    # No state leak after 3 iterations
    from torchtune.modules.peft.lora import LoRALinear
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert not getattr(module, "disabled", False), (
                f"State leaked after step {step}: LoRALinear.disabled=True outside context"
            )


def test_base_output_reproducible_across_disable_calls():
    """The ref forward must be deterministic — same base output every call."""
    model = _build_tiny_lora_model()
    tokens = torch.randint(0, 64, (BATCH, SEQ_LEN))

    with disable_adapter(model):
        with torch.no_grad():
            out1 = _forward(model, tokens)

    with disable_adapter(model):
        with torch.no_grad():
            out2 = _forward(model, tokens)

    assert torch.allclose(out1, out2, atol=0.0), (
        "Ref model outputs should be bit-exact across two disable_adapter calls "
        "(base weights are frozen; no stochastic layers in eval mode)."
    )

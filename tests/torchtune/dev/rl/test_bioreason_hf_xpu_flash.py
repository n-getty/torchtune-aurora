# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch

from torchtune.dev.bioreason import model


def test_repeat_kv_for_xpu_flash_matches_repeat_interleave():
    key = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)

    actual = model._repeat_kv_for_xpu_flash(key, 4)
    expected = torch.repeat_interleave(key, 4, dim=1)

    torch.testing.assert_close(actual, expected)


def test_to_bshd_memory_preserves_values_and_layout():
    tensor = torch.randn(2, 8, 16, 32)

    actual = model._to_bshd_memory(tensor)

    torch.testing.assert_close(actual, tensor)
    assert actual.transpose(1, 2).is_contiguous()
    assert model._to_bshd_memory(actual) is actual


def test_hf_xpu_flash_falls_back_for_cpu():
    query = torch.randn(2, 8, 16, 32)
    module = mock.Mock(num_key_value_groups=1, is_causal=True)
    expected = torch.randn(2, 16, 8, 32)

    with mock.patch(
        "transformers.integrations.sdpa_attention.sdpa_attention_forward",
        return_value=(expected, None),
    ) as fallback:
        actual, weights = model._bioreason_xpu_flash_attention_forward(
            module, query, query, query, None
        )

    assert actual is expected
    assert weights is None
    fallback.assert_called_once()


def test_hf_xpu_flash_engages_when_grad_disabled():
    query = torch.randn(2, 8, 16, 128, dtype=torch.bfloat16)
    module = mock.Mock(num_key_value_groups=1, is_causal=True)

    class _FakeDevice:
        type = "xpu"

    class _FlashContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    with mock.patch.object(type(query), "device", _FakeDevice()), mock.patch(
        "torch.nn.attention.sdpa_kernel", return_value=_FlashContext()
    ) as kernel, mock.patch(
        "torch.nn.functional.scaled_dot_product_attention", return_value=query
    ) as sdpa, torch.no_grad():
        actual, weights = model._bioreason_xpu_flash_attention_forward(
            module, query, query, query, None
        )

    assert actual.shape == (2, 16, 8, 128)
    assert weights is None
    kernel.assert_called_once()
    sdpa.assert_called_once()


def test_hf_xpu_flash_expands_gqa_and_forces_flash():
    query = torch.randn(2, 8, 16, 128, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(2, 2, 16, 128, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(2, 2, 16, 128, dtype=torch.bfloat16, requires_grad=True)
    module = mock.Mock(num_key_value_groups=4, is_causal=True)

    class _FakeDevice:
        type = "xpu"

    class _FlashContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    def _fake_sdpa(actual_query, actual_key, actual_value, **kwargs):
        assert actual_query.shape == (2, 8, 16, 128)
        assert actual_key.shape == (2, 8, 16, 128)
        assert actual_value.shape == (2, 8, 16, 128)
        assert actual_query.transpose(1, 2).is_contiguous()
        assert actual_key.transpose(1, 2).is_contiguous()
        assert actual_value.transpose(1, 2).is_contiguous()
        assert kwargs["attn_mask"] is None
        assert kwargs["is_causal"] is True
        return actual_query

    with mock.patch.object(type(query), "device", _FakeDevice()), mock.patch(
        "torch.nn.attention.sdpa_kernel", return_value=_FlashContext()
    ) as kernel, mock.patch(
        "torch.nn.functional.scaled_dot_product_attention", side_effect=_fake_sdpa
    ) as sdpa:
        output, weights = model._bioreason_xpu_flash_attention_forward(
            module, query, key, value, None
        )

    assert output.shape == (2, 16, 8, 128)
    assert weights is None
    kernel.assert_called_once()
    sdpa.assert_called_once()


def test_register_hf_xpu_flash_attention_registers_mask_policy():
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = model._register_hf_xpu_flash_attention()

    assert ALL_ATTENTION_FUNCTIONS[name] is model._bioreason_xpu_flash_attention_forward
    assert ALL_MASK_ATTENTION_FUNCTIONS[name] is model._bioreason_xpu_flash_mask


def test_hf_xpu_flash_mask_skips_synthetic_packed_sequence_mask():
    assert (
        model._bioreason_xpu_flash_mask(
            attention_mask=None,
            mask_function=mock.Mock(),
        )
        is None
    )


def test_hf_xpu_flash_mask_preserves_explicit_mask():
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    actual = model._bioreason_xpu_flash_mask(
        attention_mask=attention_mask,
        batch_size=2,
        cache_position=torch.arange(5),
        kv_length=5,
    )

    assert actual.shape == (2, 1, 5, 5)


def test_right_padding_mask_detection():
    assert model._is_unpadded_or_right_padded(torch.ones(2, 5, dtype=torch.long))
    assert model._is_unpadded_or_right_padded(
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    )
    assert not model._is_unpadded_or_right_padded(
        torch.tensor([[0, 1, 1, 1, 1]])
    )
    assert not model._is_unpadded_or_right_padded(
        torch.tensor([[[1, 1, 0, 0]]])
    )


def test_forward_drops_position_ids_with_safe_xpu_padding_mask():
    inputs_embeds = torch.randn(2, 5, 8)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    position_ids = attention_mask.cumsum(-1) - 1
    backbone = mock.Mock(return_value=mock.Mock(logits=torch.randn(2, 5, 11)))
    bioreason_model = mock.Mock(backbone=backbone)

    class _FakeDevice:
        type = "xpu"

    with mock.patch.object(type(inputs_embeds), "device", _FakeDevice()), mock.patch.object(
        model, "_USE_HF_XPU_FLASH", True
    ):
        model.BioReasonModel.forward(
            bioreason_model,
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    backbone.assert_called_once_with(
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch

from torchtune.dev.bioreason.model import BioReasonModel
from torchtune.dev.rl.loss import GRPOSimpleLoss
from torchtune.dev.rl.generation import (
    compact_prompt_completion_batch,
    gather_response_logits,
    get_descending_response_chunk_ranges,
    get_length_sorted_response_chunks,
    get_right_padded_response_length,
    pad_response_logprobs,
    trim_query_responses_to_global_max,
)


def test_compact_prompt_completion_batch_removes_shared_prompt_padding():
    prompt_embeds = torch.randn(2, 5, 3)
    prompt_ids = torch.tensor([[11, 12, 13, 0, 0], [21, 22, 23, 0, 0]])
    completions = torch.tensor([[31, 32, 0], [41, 42, 43]])

    embeds, mask, positions, context_length = compact_prompt_completion_batch(
        prompt_embeds, prompt_ids, completions, pad_id=0
    )

    assert embeds.shape == (2, 3, 3)
    assert context_length == 3
    assert mask.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
    assert positions.tolist() == [[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 5]]


def test_compact_prompt_completion_batch_packs_mixed_prompt_rows():
    prompt_embeds = torch.randn(2, 5, 3)
    prompt_ids = torch.tensor([[11, 12, 13, 0, 0], [21, 22, 23, 24, 0]])
    completions = torch.tensor([[31, 32], [41, 0]])

    embeds, mask, _, context_length = compact_prompt_completion_batch(
        prompt_embeds, prompt_ids, completions, pad_id=0
    )

    assert embeds.shape == (2, 4, 3)
    torch.testing.assert_close(context_length, torch.tensor([3, 4]))
    assert mask.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]


def test_gather_response_logits_uses_each_prompt_length_and_preserves_gradients():
    logits = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    logits.requires_grad_()

    gathered = gather_response_logits(logits, torch.tensor([3, 5]), 2)

    torch.testing.assert_close(gathered[0], logits[0, 2:4])
    torch.testing.assert_close(gathered[1], logits[1, 4:6])
    gathered.sum().backward()
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, 2:4] = 1
    expected_grad[1, 4:6] = 1
    torch.testing.assert_close(logits.grad, expected_grad)


def test_gather_response_logits_clamps_only_padded_response_positions():
    logits = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)

    gathered = gather_response_logits(logits, torch.tensor([3, 4]), 3)

    torch.testing.assert_close(gathered[0], logits[0, 2:5])
    torch.testing.assert_close(gathered[1, :2], logits[1, 3:5])
    torch.testing.assert_close(gathered[1, 2], logits[1, 4])


def test_bioreason_build_full_embeds_packs_completions_after_each_prompt():
    model = BioReasonModel.__new__(BioReasonModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model._embed = torch.nn.Embedding.from_pretrained(
        torch.arange(100, dtype=torch.float32).unsqueeze(1)
    )
    prompt_embeds = torch.tensor([[[11.0], [12.0], [13.0], [0.0]], [[21.0], [22.0], [23.0], [24.0]]])
    completion_ids = torch.tensor([[31, 32], [41, 0]])

    packed = model.build_full_embeds(
        prompt_embeds, completion_ids, torch.tensor([3, 4]), packed_width=5
    )

    assert packed.squeeze(-1).tolist() == [
        [11.0, 12.0, 13.0, 31.0, 32.0],
        [21.0, 22.0, 23.0, 24.0, 41.0],
    ]


def test_mixed_row_compaction_preserves_valid_causal_response_states():
    prompt_ids = torch.tensor([[11, 12, 13, 0, 0], [21, 22, 23, 24, 0]])
    completion_ids = torch.tensor([[31, 32, 0], [41, 0, 0]])
    prompt_embeds = prompt_ids.float().unsqueeze(-1)
    original = torch.cat([prompt_embeds, completion_ids.float().unsqueeze(-1)], dim=1)
    original_mask = torch.cat(
        [prompt_ids != 0, completion_ids != 0], dim=1
    ).unsqueeze(-1)
    original_states = (original * original_mask).cumsum(dim=1)

    packed_prompts, mask, _, prompt_lengths = compact_prompt_completion_batch(
        prompt_embeds, prompt_ids, completion_ids, pad_id=0
    )
    model = BioReasonModel.__new__(BioReasonModel)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model._embed = torch.nn.Embedding.from_pretrained(
        torch.arange(100, dtype=torch.float32).unsqueeze(1)
    )
    packed = model.build_full_embeds(
        packed_prompts, completion_ids, prompt_lengths, packed_width=mask.shape[1]
    )
    packed_states = (packed * mask.unsqueeze(-1)).cumsum(dim=1)

    original_response_states = gather_response_logits(
        original_states, torch.tensor([5, 5]), completion_ids.shape[1]
    )
    packed_response_states = gather_response_logits(
        packed_states, prompt_lengths, completion_ids.shape[1]
    )
    valid = completion_ids != 0
    torch.testing.assert_close(
        packed_response_states.squeeze(-1)[valid],
        original_response_states.squeeze(-1)[valid],
    )


def test_trim_uses_cpu_scalar_for_gloo_process_group(monkeypatch):
    process_group = object()
    observed = {}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")

    def fake_all_reduce(tensor, op, group):
        observed["device"] = tensor.device.type
        observed["group"] = group

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    query_responses = torch.tensor([[1, 2, 3, 0]])
    trimmed, response_length = trim_query_responses_to_global_max(
        query_responses, context_length=2, pad_id=0, process_group=process_group
    )

    assert observed == {"device": "cpu", "group": process_group}
    assert response_length == 1
    assert trimmed.tolist() == [[1, 2, 3]]


def test_trim_query_responses_uses_local_active_length():
    query_responses = torch.tensor(
        [
            [11, 12, 21, 22, 0, 0],
            [13, 14, 31, 32, 33, 0],
        ]
    )

    trimmed, response_length = trim_query_responses_to_global_max(
        query_responses, context_length=2, pad_id=0
    )

    assert response_length == 3
    torch.testing.assert_close(trimmed, query_responses[:, :5])


def test_trim_query_responses_keeps_one_empty_response_column():
    query_responses = torch.tensor([[11, 12, 0, 0]])

    trimmed, response_length = trim_query_responses_to_global_max(
        query_responses, context_length=2, pad_id=0
    )

    assert response_length == 1
    torch.testing.assert_close(trimmed, query_responses[:, :3])


def test_trim_query_responses_uses_distributed_global_max():
    query_responses = torch.tensor([[11, 12, 21, 0, 0, 0]])

    def set_remote_max(length, **kwargs):
        length.fill_(3)

    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_backend", return_value="gloo"),
        mock.patch(
            "torch.distributed.all_reduce", side_effect=set_remote_max
        ) as reduce,
    ):
        trimmed, response_length = trim_query_responses_to_global_max(
            query_responses, context_length=2, pad_id=0, process_group="training"
        )

    assert response_length == 3
    assert trimmed.shape == (1, 5)
    reduce.assert_called_once_with(
        mock.ANY,
        op=torch.distributed.ReduceOp.MAX,
        group="training",
    )


def test_get_right_padded_response_length_uses_longest_sequence():
    padding_masks = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, True],
        ]
    )

    assert get_right_padded_response_length(padding_masks) == 3


def test_get_right_padded_response_length_keeps_one_empty_column():
    padding_masks = torch.ones((2, 4), dtype=torch.bool)

    assert get_right_padded_response_length(padding_masks) == 1


def test_pad_response_logprobs_restores_neutral_width_and_gradient():
    local_logprobs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True)

    padded = pad_response_logprobs(local_logprobs, response_length=4)

    torch.testing.assert_close(
        padded,
        torch.tensor([[-1.0, -2.0, 1.0, 1.0], [-3.0, -4.0, 1.0, 1.0]]),
    )
    padded.sum().backward()
    torch.testing.assert_close(local_logprobs.grad, torch.ones_like(local_logprobs))


def test_response_trim_helpers_reject_invalid_shapes_and_widths():
    with pytest.raises(ValueError, match="response_padding_masks"):
        get_right_padded_response_length(torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="logprobs"):
        pad_response_logprobs(torch.ones(4), response_length=4)
    with pytest.raises(ValueError, match="smaller"):
        pad_response_logprobs(torch.ones((2, 4)), response_length=3)


def test_get_descending_response_chunk_ranges_preserves_fixed_batches():
    padding_masks = torch.tensor(
        [
            [False, False, True, True, True],
            [False, True, True, True, True],
            [False, False, False, False, True],
            [False, False, False, True, True],
            [False, False, False, False, False],
        ]
    )

    assert get_descending_response_chunk_ranges(padding_masks, 2) == [
        (4, 5),
        (2, 4),
        (0, 2),
    ]


def test_get_descending_response_chunk_ranges_rejects_bad_batch_size():
    with pytest.raises(ValueError, match="positive"):
        get_descending_response_chunk_ranges(torch.zeros((2, 3), dtype=torch.bool), 0)


def test_get_length_sorted_response_chunks_pairs_similar_lengths():
    padding_masks = torch.tensor(
        [
            [False, False, True, True, True],
            [False, True, True, True, True],
            [False, False, False, False, True],
            [False, False, False, True, True],
            [False, False, False, False, False],
        ]
    )

    assert get_length_sorted_response_chunks(padding_masks, 2) == [
        [4, 2],
        [3, 0],
        [1],
    ]


def test_get_length_sorted_response_chunks_is_stable_and_validates_inputs():
    equal_lengths = torch.tensor(
        [[False, True], [False, True], [False, False]]
    )
    assert get_length_sorted_response_chunks(equal_lengths, 2) == [[2, 0], [1]]
    with pytest.raises(ValueError, match="response_padding_masks"):
        get_length_sorted_response_chunks(torch.zeros(3, dtype=torch.bool), 2)
    with pytest.raises(ValueError, match="positive"):
        get_length_sorted_response_chunks(equal_lengths, 0)


def test_length_sorted_chunks_preserve_simple_grpo_loss_and_gradients():
    padding_masks = torch.tensor(
        [
            [False, False, False, False, True],
            [False, True, True, True, True],
            [False, False, False, True, True],
            [False, False, True, True, True],
        ]
    )
    old_logprobs = torch.randn(4, 5)
    ref_logprobs = torch.randn(4, 5)
    advantages = torch.randn(4)
    initial_logprobs = torch.randn(4, 5)
    loss_fn = GRPOSimpleLoss(kl_coeff=1e-3)

    def run(chunks):
        logprobs = initial_logprobs.clone().requires_grad_()
        loss = sum(
            loss_fn(
                old_logprobs[rows],
                logprobs[rows],
                ref_logprobs[rows],
                advantages[rows],
                padding_masks=~padding_masks[rows],
            )[0]
            * (len(rows) / len(padding_masks))
            for rows in chunks
        )
        loss.backward()
        return loss.detach(), logprobs.grad

    fixed_loss, fixed_grad = run([[0, 1], [2, 3]])
    sorted_loss, sorted_grad = run(
        get_length_sorted_response_chunks(padding_masks, batch_size=2)
    )

    torch.testing.assert_close(sorted_loss, fixed_loss)
    torch.testing.assert_close(sorted_grad, fixed_grad)

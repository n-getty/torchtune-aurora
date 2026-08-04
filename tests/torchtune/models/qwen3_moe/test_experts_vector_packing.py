import torch

from torchtune.models.qwen3_moe._experts import (
    _expert_padded_row_indices,
    _pack_expert_tokens_vectorized,
)


def test_vectorized_expert_packing_handles_uneven_and_zero_counts():
    counts = torch.tensor([0, 2, 1, 0, 3], dtype=torch.int64)
    inputs = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)

    packed, rows = _pack_expert_tokens_vectorized(inputs, counts, max_count=3)

    expected = torch.zeros(5, 3, 4)
    expected[1, :2] = inputs[:2]
    expected[2, :1] = inputs[2:3]
    expected[4, :3] = inputs[3:]
    torch.testing.assert_close(packed, expected)
    torch.testing.assert_close(packed.reshape(-1, 4).index_select(0, rows), inputs)


def test_vectorized_expert_row_indices_preserve_sorted_token_order():
    counts = torch.tensor([1, 0, 2, 3], dtype=torch.int64)
    rows = _expert_padded_row_indices(counts, max_count=3, total=6)
    assert rows.tolist() == [0, 6, 7, 9, 10, 11]


def test_vectorized_expert_row_indices_match_reference_for_zero_and_uneven_counts():
    counts = torch.tensor([2, 0, 4, 1, 0, 3], dtype=torch.int64)
    max_count = 4
    total = int(counts.sum())
    starts = torch.cumsum(counts, dim=0) - counts
    expected = torch.cat(
        [
            torch.arange(start, start + count) + expert * max_count - start
            for expert, (start, count) in enumerate(zip(starts.tolist(), counts.tolist()))
            if count
        ]
    )
    actual = _expert_padded_row_indices(counts, max_count, total)
    torch.testing.assert_close(actual, expected)


def test_vectorized_expert_row_indices_normalize_non_long_counts():
    counts = torch.tensor([1, 0, 2, 3], dtype=torch.int32)
    rows = _expert_padded_row_indices(counts, max_count=3, total=6)
    assert rows.dtype == torch.int64
    assert rows.tolist() == [0, 6, 7, 9, 10, 11]


def test_vectorized_qwen_experts_matches_legacy_forward_and_backward(monkeypatch):
    import torchtune.models.qwen3_moe._experts as experts_module

    torch.manual_seed(3)
    counts = torch.tensor([0, 2, 1, 0, 3], dtype=torch.int64)
    legacy = experts_module.GroupedExpertsHF(dim=4, hidden_dim=7, num_experts=5)
    candidate = experts_module.GroupedExpertsHF(dim=4, hidden_dim=7, num_experts=5)
    for parameter in legacy.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    candidate.load_state_dict(legacy.state_dict())
    legacy_input = torch.randn(6, 4, requires_grad=True)
    candidate_input = legacy_input.detach().clone().requires_grad_(True)

    monkeypatch.setattr(experts_module, "_VECTOR_PACKING", False)
    legacy_output = legacy(legacy_input, counts)
    legacy_output.square().mean().backward()
    monkeypatch.setattr(experts_module, "_VECTOR_PACKING", True)
    candidate_output = candidate(candidate_input, counts)
    candidate_output.square().mean().backward()

    torch.testing.assert_close(candidate_output, legacy_output)
    torch.testing.assert_close(candidate_input.grad, legacy_input.grad)
    for legacy_parameter, candidate_parameter in zip(
        legacy.parameters(), candidate.parameters()
    ):
        torch.testing.assert_close(candidate_parameter.grad, legacy_parameter.grad)

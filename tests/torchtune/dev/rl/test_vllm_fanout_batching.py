# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Guard for the BioReason vLLM generation fan-out BATCHING fix (2026-06-22).

The old fan-out submitted one request per prompt round-robin'd across all 12 vLLM
engines -> ~1 seq/engine -> single-stream decode (~50 tok/s). vLLM batches concurrent
seqs at ~175 tok/s (Running:3-4), 3-4x faster, but only if each engine gets MULTIPLE
seqs. The fix GROUPS the bsz embeds into per-engine batches
(TORCHTUNE_VLLM_SEQS_PER_ENGINE, default 4) and submits one multi-embed call per engine.

This test pins the grouping invariants (the part that's pure logic and could silently
break: every index covered exactly once, engine count bounded, batch size respected)
and that the recipe still uses the client's list-batched generate_from_embeds.
"""
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]


def _group(bsz, seqs_per_engine, num_clients=12):
    """Reproduces the grouping logic in _generate_with_vllm_server_embeds."""
    spe = max(1, seqs_per_engine)
    n_eng = max(1, min(num_clients, (bsz + spe - 1) // spe))
    groups = [[] for _ in range(n_eng)]
    for i in range(bsz):
        groups[i % n_eng].append(i)
    return groups


def _prompt_n_request_indices(group, grpo_size, enabled=True):
    prompt_n = grpo_size if enabled and len(group) % grpo_size == 0 else 1
    return prompt_n, group[::prompt_n]


def _split_prompt_n_requests(
    request_plan,
    num_clients=6,
    engine_stride=0,
    engine_phase=0,
    prompts_per_request=1,
):
    prompts_per_request = max(1, prompts_per_request)
    return [
        (
            (engine_id + engine_phase + split_idx * engine_stride) % num_clients,
            indices[start : start + request_n * prompts_per_request],
            request_n,
        )
        for engine_id, indices, request_n in request_plan
        for split_idx, start in enumerate(
            range(0, len(indices), request_n * prompts_per_request)
        )
    ]


def _split_choice_requests(request_plan, num_clients=6, engine_stride=0):
    if engine_stride == 0:
        return request_plan
    return [
        (
            (engine_id + (choice_idx % request_n) * engine_stride) % num_clients,
            [index],
            1,
        )
        for engine_id, indices, request_n in request_plan
        for choice_idx, index in enumerate(indices)
    ]


@pytest.mark.parametrize("bsz", [1, 8, 16, 64, 96])
@pytest.mark.parametrize("spe", [1, 4, 8])
def test_grouping_covers_every_index_once(bsz, spe):
    groups = _group(bsz, spe)
    covered = sorted(j for g in groups for j in g)
    assert covered == list(range(bsz)), "grouping must cover each seq exactly once"


def test_prompt_n_deduplicates_prompt_major_grpo_expansion():
    prompt_n, request_indices = _prompt_n_request_indices(list(range(16)), 8)

    assert prompt_n == 8
    assert request_indices == [0, 8]


def test_prompt_n_falls_back_when_group_splits_a_grpo_set():
    prompt_n, request_indices = _prompt_n_request_indices([0, 1, 2, 3], 8)

    assert prompt_n == 1
    assert request_indices == [0, 1, 2, 3]


def test_split_prompt_n_keeps_one_prompt_per_request():
    request_plan = _split_prompt_n_requests([(3, list(range(8)), 4)])

    assert request_plan == [(3, [0, 1, 2, 3], 4), (3, [4, 5, 6, 7], 4)]


def test_split_prompt_n_engine_stride_decorrelates_prompt_requests():
    request_plan = _split_prompt_n_requests([(1, list(range(8)), 4)], engine_stride=3)

    assert request_plan == [(1, [0, 1, 2, 3], 4), (4, [4, 5, 6, 7], 4)]


def test_split_prompt_n_engine_stride_balances_production_requests():
    requests = [
        request
        for replica_idx in range(14)
        for request in _split_prompt_n_requests(
            [(replica_idx % 6, list(range(8)), 4)], engine_stride=3
        )
    ]
    engine_counts = [
        sum(engine_id == index for engine_id, _, _ in requests) for index in range(6)
    ]

    assert engine_counts == [5, 5, 4, 5, 5, 4]
    assert max(engine_counts) - min(engine_counts) == 1


def test_split_prompt_n_supports_four_prompts_with_two_choices():
    request_plan = _split_prompt_n_requests([(2, list(range(8)), 2)])

    assert request_plan == [
        (2, [0, 1], 2),
        (2, [2, 3], 2),
        (2, [4, 5], 2),
        (2, [6, 7], 2),
    ]


def test_split_prompt_n_groups_two_prompts_per_request():
    request_plan = _split_prompt_n_requests(
        [(2, list(range(8)), 2)], engine_stride=1, prompts_per_request=2
    )

    assert request_plan == [(2, [0, 1, 2, 3], 2), (3, [4, 5, 6, 7], 2)]


def test_two_prompt_requests_balance_b4g2_production_sequences():
    requests = [
        request
        for replica_idx in range(14)
        for request in _split_prompt_n_requests(
            [(replica_idx % 6, list(range(8)), 2)],
            engine_stride=1,
            engine_phase=replica_idx * 3,
            prompts_per_request=2,
        )
    ]
    engine_sequences = [
        sum(len(indices) for engine_id, indices, _ in requests if engine_id == index)
        for index in range(6)
    ]

    assert sorted(engine_sequences) == [16, 16, 20, 20, 20, 20]
    assert sum(engine_sequences) == 112


def test_split_prompt_n_stride_one_balances_b4g2_production_requests():
    requests = [
        request
        for replica_idx in range(14)
        for request in _split_prompt_n_requests(
            [(replica_idx % 6, list(range(8)), 2)],
            engine_stride=1,
            engine_phase=replica_idx * 3,
        )
    ]
    engine_counts = [
        sum(engine_id == index for engine_id, _, _ in requests) for index in range(6)
    ]

    assert engine_counts == [10, 10, 9, 9, 9, 9]
    assert max(engine_counts) - min(engine_counts) == 1


def test_split_choices_preserves_order_and_separates_pair():
    request_plan = _split_choice_requests([(2, [4, 5], 2)], engine_stride=3)

    assert request_plan == [(2, [4], 1), (5, [5], 1)]


def test_split_choices_balances_b4g2_production_sequences():
    requests = [
        request
        for replica_idx in range(14)
        for request in _split_choice_requests(
            _split_prompt_n_requests(
                [(replica_idx % 6, list(range(8)), 2)],
                engine_stride=1,
                engine_phase=replica_idx * 3,
            ),
            engine_stride=3,
        )
    ]
    engine_sequences = [
        sum(len(indices) for engine_id, indices, _ in requests if engine_id == index)
        for index in range(6)
    ]

    assert sorted(engine_sequences) == [18, 18, 19, 19, 19, 19]
    assert sum(engine_sequences) == 112


def test_recipe_has_split_prompt_n_guard_in_both_generation_paths():
    src = (_REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py").read_text()

    assert src.count("TORCHTUNE_VLLM_SPLIT_PROMPT_N") == 2
    assert src.count("TORCHTUNE_VLLM_SPLIT_ENGINE_STRIDE") == 2
    assert src.count("TORCHTUNE_VLLM_SPLIT_ENGINE_PHASE_PER_REPLICA") == 2
    assert src.count("TORCHTUNE_VLLM_SPLIT_PROMPTS_PER_REQUEST") == 1
    assert src.count("TORCHTUNE_VLLM_SPLIT_CHOICE_ENGINE_STRIDE") == 2


@pytest.mark.parametrize(
    "bsz,spe,want_engines",
    [
        (16, 4, 4),  # the prod sweet spot: 4 engines x 4 seqs (Running:4 -> ~175 tok/s)
        (8, 4, 2),
        (16, 1, 12),  # spe=1 restores spread-thin (<=12 engines)
        (96, 4, 12),  # capped at num_clients
    ],
)
def test_engine_count(bsz, spe, want_engines):
    assert len(_group(bsz, spe)) == want_engines


def test_max_seqs_per_engine_respected():
    # at spe=4, no engine should hold more than ceil(bsz/n_eng) which is <= spe
    # until bsz exceeds 12*spe (then it grows, capped by 12 engines).
    groups = _group(16, 4)
    assert max(len(g) for g in groups) == 4


def test_recipe_uses_grouped_multi_embed_call():
    src = (_REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py").read_text()
    # the env knob + grouped submission must be present
    assert "TORCHTUNE_VLLM_SEQS_PER_ENGINE" in src
    assert "_call_group" in src
    # the client call must pass a LIST of embeds (batched), not a single [embed]
    assert "prompt_embeds=embeds" in src


def test_spe1_restores_spread_behavior():
    # escape hatch: spe=1 spreads bsz across up to 12 engines (old behavior)
    assert len(_group(16, 1)) == 12
    assert all(len(g) <= 2 for g in _group(16, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

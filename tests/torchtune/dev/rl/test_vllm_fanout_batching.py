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


@pytest.mark.parametrize("bsz", [1, 8, 16, 64, 96])
@pytest.mark.parametrize("spe", [1, 4, 8])
def test_grouping_covers_every_index_once(bsz, spe):
    groups = _group(bsz, spe)
    covered = sorted(j for g in groups for j in g)
    assert covered == list(range(bsz)), "grouping must cover each seq exactly once"


@pytest.mark.parametrize("bsz,spe,want_engines", [
    (16, 4, 4),    # the prod sweet spot: 4 engines x 4 seqs (Running:4 -> ~175 tok/s)
    (8, 4, 2),
    (16, 1, 12),   # spe=1 restores spread-thin (<=12 engines)
    (96, 4, 12),   # capped at num_clients
])
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

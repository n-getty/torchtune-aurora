"""Guard for the BioReason vLLM REPLICA-DISJOINT engine assignment fix (2026-06-23).

Root cause (train_mpiexec_20260623_000609.log, 4N HSDP dp_replicate=3 x dp_shard=12):
under HSDP every shard leader (ranks 0, 12, 24) runs _generate_with_vllm_server_embeds
CONCURRENTLY and all of them point at the SAME 12 vLLM client URLs. The pre-fix code
selected engines via clients[g % num_clients] starting at g=0 for every leader, so all
3 leaders piled onto engines 0-3 (each carrying 3x4=12 concurrent seqs) while engines
4-11 (8 of 12, 67% of the gen node) sat 100% IDLE. Measured: gen ~92s/step, ~27s mean
spread across the 3 leaders.

The fix partitions the engine pool into dp_replicate disjoint contiguous bands and gives
each replica leader its own band, so the R*bsz seqs/step spread uniformly across ALL
num_clients engines with zero cross-leader contention.

This test pins the pure-logic invariants that could silently regress:
  1. across all replicas, the engine sets are DISJOINT (no engine serves two leaders),
  2. the union covers as many engines as work allows (no idle engines when R*want >= N),
  3. every seq index is still covered exactly once per leader,
  4. dp_replicate<=1 reduces to the old base-0 behavior (byte-identical validated path),
  5. the recipe source still wires replica-banded selection (not the old g % num_clients).
"""
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]


def _assign(bsz, seqs_per_engine, replica_idx, n_rep, num_clients=12):
    """Reproduces the engine-selection + grouping logic in the recipe.

    Returns (engine_ids, groups) for ONE replica leader.
    """
    spe = max(1, seqs_per_engine)
    want_engines = max(1, (bsz + spe - 1) // spe)
    n_rep = max(1, n_rep)
    ridx = replica_idx if n_rep > 1 else 0
    band = max(1, num_clients // n_rep)
    eng_base = (ridx % n_rep) * band
    is_last_band = (ridx == n_rep - 1)
    band_size = (num_clients - eng_base) if is_last_band else band
    n_engines = max(1, min(want_engines, band_size))
    engine_ids = [(eng_base + e) % num_clients for e in range(n_engines)]
    groups = [[] for _ in range(n_engines)]
    for i in range(bsz):
        groups[i % n_engines].append(i)
    return engine_ids, groups


# --- prod envelope: 4N HSDP, dp_replicate=3, bsz=16, spe=4 --------------------

def test_prod_envelope_uses_all_12_engines_disjoint():
    """3 replicas x 4 engines = all 12 engines, no overlap (the actual bug)."""
    n_rep, num_clients = 3, 12
    all_ids = []
    for r in range(n_rep):
        ids, groups = _assign(bsz=16, seqs_per_engine=4, replica_idx=r,
                              n_rep=n_rep, num_clients=num_clients)
        assert len(ids) == 4, f"replica {r} should use 4 engines, got {ids}"
        all_ids.extend(ids)
    assert sorted(all_ids) == list(range(12)), (
        f"3 leaders must cover all 12 engines exactly once; got {sorted(all_ids)}"
    )


@pytest.mark.parametrize("n_rep,num_clients", [(2, 12), (3, 12), (4, 12), (6, 12), (1, 12)])
def test_engine_bands_are_disjoint(n_rep, num_clients):
    """No engine may serve two replica leaders simultaneously."""
    seen = {}
    for r in range(n_rep):
        ids, _ = _assign(bsz=16, seqs_per_engine=4, replica_idx=r,
                        n_rep=n_rep, num_clients=num_clients)
        for e in ids:
            assert e not in seen or n_rep == 1, (
                f"engine {e} assigned to replicas {seen.get(e)} and {r} (contention)"
            )
            seen[e] = r


@pytest.mark.parametrize("n_rep", [2, 3, 4, 6])
def test_no_idle_engine_when_work_saturates_pool(n_rep):
    """When R*want_engines >= num_clients, every engine should be used."""
    num_clients = 12
    # bsz=16, spe=4 -> want=4 engines per leader; R*4 >= 12 for R in {3,4,6}, and
    # R=2 -> 8 engines used (band=6, want=4 -> 4 each -> 8 total, not 12: expected,
    # work doesn't saturate so this asserts coverage == min(R*want, num_clients)).
    want = 4
    used = set()
    for r in range(n_rep):
        ids, _ = _assign(bsz=16, seqs_per_engine=4, replica_idx=r,
                        n_rep=n_rep, num_clients=num_clients)
        used.update(ids)
    assert len(used) == min(n_rep * want, num_clients)


@pytest.mark.parametrize("bsz", [1, 8, 16, 24])
@pytest.mark.parametrize("n_rep", [1, 2, 3])
def test_grouping_covers_every_index_once(bsz, n_rep):
    for r in range(n_rep):
        _, groups = _assign(bsz, 4, r, n_rep)
        covered = sorted(j for g in groups for j in g)
        assert covered == list(range(bsz)), (
            f"replica {r}: grouping must cover each seq exactly once"
        )


def test_single_replica_is_base0_identical():
    """dp_replicate<=1 must reduce to the old base-0 behavior (validated path)."""
    # bsz=16, spe=4, single leader -> engines 0..3 (same as the pre-HSDP path).
    ids, groups = _assign(bsz=16, seqs_per_engine=4, replica_idx=0, n_rep=1)
    assert ids == [0, 1, 2, 3]
    assert len(groups) == 4 and max(len(g) for g in groups) == 4


def test_spe1_spread_thin_within_band():
    """spe=1 escape hatch: each leader spreads across its whole band."""
    # 3 replicas, band=4 each. spe=1 -> want=16 engines but capped to band_size=4.
    ids, _ = _assign(bsz=16, seqs_per_engine=1, replica_idx=1, n_rep=3)
    assert ids == [4, 5, 6, 7]  # replica 1's band


def test_last_band_absorbs_remainder():
    """num_clients not divisible by n_rep: last replica gets the leftover engines."""
    # 5 replicas over 12 engines: band=2, last band = 12 - 4*2 = 4 engines.
    n_rep, num_clients = 5, 12
    bands = [_assign(16, 1, r, n_rep, num_clients)[0] for r in range(n_rep)]
    flat = [e for b in bands for e in b]
    assert sorted(flat) == list(range(12)), f"must cover all engines; got {sorted(flat)}"
    assert bands[-1] == [8, 9, 10, 11], f"last band should absorb remainder; got {bands[-1]}"


# --- source guard -------------------------------------------------------------

def test_recipe_wires_replica_banded_selection():
    src = (_REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py").read_text()
    # the replica-disjoint band logic must be present
    assert "_engine_ids" in src, "recipe must compute per-replica engine ids"
    assert "_eng_base" in src, "recipe must offset engines by replica band"
    assert "_replica_idx" in src and "_dp_shard" in src, (
        "recipe must derive replica index from rank // dp_shard"
    )
    # the OLD contention bug (every leader starting at clients[g % num_clients]) is gone
    assert "self._vllm_clients[g % num_clients]" not in src, (
        "recipe still uses the buggy g % num_clients selection (all leaders collide on 0..)"
    )
    assert "self._vllm_clients[_engine_ids[g]]" in src, (
        "recipe must select clients via the per-replica disjoint _engine_ids band"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

"""
CPU-safe test: WS10 parallel recv intra-TP fanout fix.

Validates that the _ws10_pre_recv batch-indexed lookup is built correctly
and contains bit-exact data matching the original tensors. No XPU, no
distributed init, no gloo PGs required.

Root cause (2026-05-05): _ws10_parallel_recv() ran only on vLLM TP0 (which
has _xccl_sharded_cross_pgs). TP1/2/3 don't have that attribute and fell
through to _xccl_intra_pg.broadcast(recv_buf, root=0).wait() — blocking for
120 s (DistStoreError timeout) because TP0 returned early without calling the
intra broadcast.

Fix: TP0 no longer returns early. Instead it builds _ws10_pre_recv (a
batch-indexed CPU buffer lookup) and falls through to the serial receive
loop. In the serial loop, gloo cross-recv is replaced by a CPU memcpy from
_ws10_pre_recv[batch_start]. The subsequent intra-TP broadcast runs normally,
distributing data to TP1/2/3.

This test verifies that the batch boundaries produced by the lookup builder
match those the serial loop computes, and that every tensor value is
preserved exactly.
"""

import re
import torch
import pytest


def build_ws10_pre_recv(tensors_meta: list, batch_max_numel: int, p_ne: list, p_shards: dict) -> dict:
    """Replicate the _ws10_pre_recv building logic from vllm_weight_sync_worker.py."""
    ws10_ntf: dict = {}
    for name, tensor in p_ne:
        ws10_ntf[name] = tensor.view(-1)

    ws10_ere = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight")
    for li, lyr in p_shards.items():
        for R, s in lyr.get("w13", {}).items():
            ws10_ntf[(li, "w13", R)] = s.view(-1)
        for R, s in lyr.get("w2", {}).items():
            ws10_ntf[(li, "w2", R)] = s.view(-1)

    ws10_pre_recv: dict = {}
    j = 0
    n_params = len(tensors_meta)
    while j < n_params:
        bs = j
        bn = 0
        Rr = tensors_meta[j].get("trainer_ep_rank")
        while j < n_params:
            pn = tensors_meta[j]["numel"]
            Rn = tensors_meta[j].get("trainer_ep_rank")
            if bn > 0 and Rn != Rr:
                break
            if bn > 0 and bn + pn > batch_max_numel:
                break
            bn += pn
            j += 1
        pack = torch.empty(bn, dtype=torch.bfloat16)
        off = 0
        for e in tensors_meta[bs:j]:
            en = e["numel"]
            em = ws10_ere.match(e["name"])
            if em:
                k = (int(em.group(1)), em.group(2), int(e.get("trainer_ep_rank", 0)))
            else:
                k = e["name"]
            src = ws10_ntf.get(k)
            if src is not None:
                pack[off:off + en].copy_(src[:en])
            off += en
        ws10_pre_recv[bs] = pack
    return ws10_pre_recv


def serial_batch_boundaries(tensors_meta: list, batch_max_numel: int) -> list:
    """Replicate the serial loop's batch boundary computation — returns list of batch_start values."""
    boundaries = []
    i = 0
    n_params = len(tensors_meta)
    while i < n_params:
        boundaries.append(i)
        batch_start = i
        batch_numel = 0
        Rr = tensors_meta[i].get("trainer_ep_rank")
        while i < n_params:
            pn = tensors_meta[i]["numel"]
            Rn = tensors_meta[i].get("trainer_ep_rank")
            if batch_numel > 0 and Rn != Rr:
                break
            if batch_numel > 0 and batch_numel + pn > batch_max_numel:
                break
            batch_numel += pn
            i += 1
    return boundaries


def make_tensors_meta(n_expert_layers: int, ep_degree: int, n_non_expert: int,
                      expert_numel: int = 100, non_expert_numel: int = 50) -> list:
    """Build a synthetic tensors_meta matching WS10 manifest structure."""
    meta = []
    # Expert entries: grouped by trainer_ep_rank, all layers for each rank
    for R in range(ep_degree):
        for li in range(n_expert_layers):
            meta.append({
                "name": f"model.layers.{li}.mlp.experts.w13_weight",
                "numel": expert_numel,
                "shape": [expert_numel],
                "trainer_ep_rank": R,
                "ep_degree": ep_degree,
            })
            meta.append({
                "name": f"model.layers.{li}.mlp.experts.w2_weight",
                "numel": expert_numel,
                "shape": [expert_numel],
                "trainer_ep_rank": R,
                "ep_degree": ep_degree,
            })
    # Non-expert entries from rank 0 (no trainer_ep_rank tag)
    for idx in range(n_non_expert):
        meta.append({
            "name": f"model.layers.{idx}.self_attn.q_proj.weight",
            "numel": non_expert_numel,
            "shape": [non_expert_numel],
        })
    return meta


def make_parallel_recv_results(tensors_meta: list, ep_degree: int):
    """Simulate _ws10_parallel_recv() output with known tensor values."""
    ws10_ere = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight")
    p_shards: dict = {}
    p_ne: list = []

    # Assign unique values: expert (li, kind, R) -> float32 seed, non-expert name -> seed
    for entry in tensors_meta:
        n = entry["numel"]
        m = ws10_ere.match(entry["name"])
        if m:
            li = int(m.group(1))
            kind = m.group(2)
            R = int(entry.get("trainer_ep_rank", 0))
            val = float(li * 1000 + (0 if kind == "w13" else 100) + R + 1) / 10000
            t = torch.full((n,), val, dtype=torch.bfloat16)
            layer = p_shards.setdefault(li, {
                "w13": {}, "w2": {}, "ep_degree": ep_degree, "n_local_trainer": 1,
            })
            layer[kind][R] = t.reshape(entry["shape"])
        else:
            name = entry["name"]
            val = float(hash(name) % 10000 + 1) / 10000
            t = torch.full((n,), val, dtype=torch.bfloat16)
            p_ne.append((name, t.reshape(entry["shape"])))

    return p_shards, p_ne


class TestWs10PreRecv:

    def test_batch_boundaries_match_serial_loop(self):
        """_ws10_pre_recv must have keys matching serial loop's batch_start values."""
        meta = make_tensors_meta(n_expert_layers=4, ep_degree=4, n_non_expert=3,
                                 expert_numel=80, non_expert_numel=40)
        batch_max = 250
        p_shards, p_ne = make_parallel_recv_results(meta, ep_degree=4)

        pre_recv = build_ws10_pre_recv(meta, batch_max, p_ne, p_shards)
        serial_starts = serial_batch_boundaries(meta, batch_max)

        assert set(pre_recv.keys()) == set(serial_starts), (
            f"Batch start mismatch: pre_recv={sorted(pre_recv.keys())} "
            f"serial={sorted(serial_starts)}"
        )

    def test_expert_values_preserved(self):
        """Expert tensors packed into _ws10_pre_recv must match original shard values."""
        ep_degree = 4
        n_layers = 3
        expert_numel = 60
        meta = make_tensors_meta(n_expert_layers=n_layers, ep_degree=ep_degree,
                                 n_non_expert=2, expert_numel=expert_numel)
        batch_max = 200
        p_shards, p_ne = make_parallel_recv_results(meta, ep_degree=ep_degree)
        pre_recv = build_ws10_pre_recv(meta, batch_max, p_ne, p_shards)

        ws10_ere = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight")
        i = 0
        while i < len(meta):
            bs = list(pre_recv.keys())
            # Find which batch this entry belongs to
            batch_start = max(s for s in bs if s <= i)
            pack = pre_recv[batch_start]
            # Find offset of entry i within its batch
            off = sum(meta[j]["numel"] for j in range(batch_start, i))
            entry = meta[i]
            n = entry["numel"]
            m = ws10_ere.match(entry["name"])
            if m:
                li = int(m.group(1))
                kind = m.group(2)
                R = int(entry.get("trainer_ep_rank", 0))
                expected = p_shards[li][kind][R].view(-1)
                actual = pack[off:off + n]
                assert torch.allclose(actual, expected, atol=0), (
                    f"Mismatch at layer={li} kind={kind} R={R}: "
                    f"expected={expected[:5]} actual={actual[:5]}"
                )
            i += 1

    def test_non_expert_values_preserved(self):
        """Non-expert tensors packed into _ws10_pre_recv must match original values."""
        meta = make_tensors_meta(n_expert_layers=2, ep_degree=2, n_non_expert=4,
                                 expert_numel=50, non_expert_numel=30)
        batch_max = 300
        p_shards, p_ne = make_parallel_recv_results(meta, ep_degree=2)
        pre_recv = build_ws10_pre_recv(meta, batch_max, p_ne, p_shards)

        ne_dict = {name: t.view(-1) for name, t in p_ne}
        n_params = len(meta)
        i = 0
        while i < n_params:
            entry = meta[i]
            if entry.get("trainer_ep_rank") is None:
                batch_starts = sorted(k for k in pre_recv if k <= i)
                batch_start = batch_starts[-1]
                off = sum(meta[j]["numel"] for j in range(batch_start, i))
                n = entry["numel"]
                expected = ne_dict.get(entry["name"])
                actual = pre_recv[batch_start][off:off + n]
                if expected is not None:
                    assert torch.allclose(actual, expected, atol=0), (
                        f"Non-expert mismatch for {entry['name']}: "
                        f"expected={expected[:5]} actual={actual[:5]}"
                    )
            i += 1

    def test_batch_size_respects_max_numel(self):
        """No batch in _ws10_pre_recv should exceed batch_max_numel elements."""
        meta = make_tensors_meta(n_expert_layers=6, ep_degree=8, n_non_expert=5,
                                 expert_numel=100, non_expert_numel=70)
        batch_max = 350
        p_shards, p_ne = make_parallel_recv_results(meta, ep_degree=8)
        pre_recv = build_ws10_pre_recv(meta, batch_max, p_ne, p_shards)

        for bs, pack in pre_recv.items():
            assert pack.numel() <= batch_max, (
                f"Batch at {bs} has {pack.numel()} elements > max {batch_max}"
            )

    def test_rank_boundary_enforced(self):
        """Entries from different trainer_ep_ranks must not share a batch."""
        meta = make_tensors_meta(n_expert_layers=2, ep_degree=3, n_non_expert=1,
                                 expert_numel=100)
        batch_max = 10000  # large enough that size alone never splits
        p_shards, p_ne = make_parallel_recv_results(meta, ep_degree=3)
        pre_recv = build_ws10_pre_recv(meta, batch_max, p_ne, p_shards)

        for bs, pack in pre_recv.items():
            # Collect trainer_ep_rank values for this batch
            # Find how many entries are in this batch
            total = pack.numel()
            acc = 0
            ranks_in_batch = set()
            for j, entry in enumerate(meta[bs:], start=bs):
                if acc >= total:
                    break
                acc += entry["numel"]
                R = entry.get("trainer_ep_rank")
                if R is not None:
                    ranks_in_batch.add(R)
            # Each batch should have at most 1 distinct EP rank (plus possibly None for non-expert)
            assert len(ranks_in_batch) <= 1, (
                f"Batch at {bs} mixes EP ranks: {ranks_in_batch}"
            )

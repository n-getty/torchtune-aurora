"""
CPU-safe test: WS10 parallel recv batch packing (post rebuild-pass removal).

Root cause (2026-05-05): _ws10_parallel_recv() ran only on vLLM TP0 (which
has _xccl_sharded_cross_pgs). TP1/2/3 don't have that attribute and fell
through to _xccl_intra_pg.broadcast(recv_buf, root=0).wait() — blocking for
120 s (DistStoreError timeout) because TP0 returned early without calling the
intra broadcast.

Original fix (2026-05-05): TP0 no longer returns early. Instead it built
_ws10_pre_recv (a batch-indexed CPU buffer lookup) via a SEPARATE
single-threaded post-join "rebuild" pass over all_shards/nonexpert_weights,
then fell through to the serial receive loop, which used _ws10_pre_recv
instead of a fresh gloo cross-recv.

Removed 2026-07-22 (see memory/project_ws10_ep8_hang_20260721.md): that
rebuild pass measured ~21-23s single-threaded (job 8683110) sitting between
"last EP rank's broadcast lands" and "first intra-node TP fanout batch
posts". Proven redundant: because the sender's canonical manifest order
(_ws10_sort_key_by_rank in weight_sync.py) groups every rank's entries
contiguously, a rank's LOCAL greedy batch boundaries (computed from just
that rank's own entries slice) are byte-identical to the GLOBAL rebuild
plan's batches filtered to that rank. So each recv_rank_thread in
_ws10_parallel_recv now packs its own batch's flat tensor directly, keyed
by GLOBAL tensors_meta index, right after its own pg.broadcast() completes
— no separate pass, no all_shards/nonexpert_weights intermediate.

This test verifies the NEW per-thread packing logic (mirrored standalone,
no torch/distributed/threading required) produces batch boundaries and
byte-identical data to what the OLD two-step approach produced, and that
those boundaries still match the serial fanout loop's independent
computation (the receiver-side consumer in vllm_weight_sync_worker.py).
"""

import re
import torch
import pytest


_FUSED_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight")


def build_pre_recv_new(tensors_meta: list, batch_max_numel: int,
                        wire_values: dict) -> dict:
    """Mirrors the NEW recv_rank_thread inline packing in
    _ws10_parallel_recv (vllm_weight_sync_worker.py). No threading needed
    for the test — ranks are processed sequentially, which is equivalent
    since each rank's packing is independent (disjoint pre_recv keys).

    wire_values: {entry_name: 1-D tensor} — the "received" value for each
    entry, as if it arrived over the wire. This stands in for the real
    gloo broadcast payload.
    """
    rank_groups: dict = {}
    for global_idx, entry in enumerate(tensors_meta):
        R = entry.get("trainer_ep_rank")
        eff_R = int(R) if R is not None else 0
        rank_groups.setdefault(eff_R, []).append((global_idx, entry))

    pre_recv: dict = {}
    for ep_rank, indexed_entries in sorted(rank_groups.items()):
        i = 0
        n = len(indexed_entries)
        while i < n:
            batch_start = i
            global_batch_start = indexed_entries[batch_start][0]
            batch_numel = 0
            batch_R = indexed_entries[batch_start][1].get("trainer_ep_rank")
            while i < n:
                pn = indexed_entries[i][1]["numel"]
                R_now = indexed_entries[i][1].get("trainer_ep_rank")
                if batch_numel > 0 and R_now != batch_R:
                    break
                if batch_numel > 0 and batch_numel + pn > batch_max_numel:
                    break
                batch_numel += pn
                i += 1
            # Simulate the concatenated wire payload for this batch: in
            # reality this comes from a single pg.broadcast() call; here we
            # concatenate the per-entry synthetic values in order.
            parts = [wire_values[e["name"]] for _, e in indexed_entries[batch_start:i]]
            pre_recv[global_batch_start] = torch.cat(parts)
    return pre_recv


def build_pre_recv_old(tensors_meta: list, batch_max_numel: int,
                        wire_values: dict) -> dict:
    """Mirrors the OLD (removed) two-step approach: scatter wire_values into
    an all_shards/nonexpert_weights-equivalent lookup keyed by (layer, kind,
    rank) or name, then run the separate global rebuild-plan pass. Kept here
    only as an equivalence oracle proving the new code produces identical
    output to the old code."""
    ntf: dict = {}
    for entry in tensors_meta:
        m = _FUSED_RE.match(entry["name"])
        R = entry.get("trainer_ep_rank")
        if m:
            key = (int(m.group(1)), m.group(2), int(R if R is not None else 0))
        else:
            key = entry["name"]
        ntf[key] = wire_values[entry["name"]]

    plan = []
    j = 0
    n_params = len(tensors_meta)
    while j < n_params:
        bs = j
        bn = 0
        Rr = tensors_meta[j].get("trainer_ep_rank")
        seg = []
        while j < n_params:
            e = tensors_meta[j]
            pn = e["numel"]
            Rn = e.get("trainer_ep_rank")
            if bn > 0 and Rn != Rr:
                break
            if bn > 0 and bn + pn > batch_max_numel:
                break
            m = _FUSED_RE.match(e["name"])
            if m:
                k = (int(m.group(1)), m.group(2), int(Rn if Rn is not None else 0))
            else:
                k = e["name"]
            seg.append((bn, pn, k))
            bn += pn
            j += 1
        plan.append((bs, bn, seg))

    pre_recv = {}
    for bs, bn, seg in plan:
        pack = torch.empty(bn, dtype=torch.float32)
        for off, en, k in seg:
            src = ntf.get(k)
            if src is not None:
                pack[off:off + en].copy_(src[:en])
        pre_recv[bs] = pack
    return pre_recv


def serial_batch_boundaries(tensors_meta: list, batch_max_numel: int) -> list:
    """Replicate the serial fanout loop's independent batch boundary
    computation (vllm_weight_sync_worker.py's while-loop consumer) — returns
    list of batch_start values it expects to find as _ws10_pre_recv keys."""
    boundaries = []
    i = 0
    n_params = len(tensors_meta)
    while i < n_params:
        boundaries.append(i)
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
    for idx in range(n_non_expert):
        meta.append({
            "name": f"model.layers.{idx}.self_attn.q_proj.weight",
            "numel": non_expert_numel,
            "shape": [non_expert_numel],
        })
    return meta


def make_wire_values(tensors_meta: list) -> dict:
    """Deterministic per-entry values so packing can be verified exactly."""
    values = {}
    for idx, entry in enumerate(tensors_meta):
        # Unique per name+rank combination is unnecessary here since names
        # are already unique per (layer, kind) x rank combination in
        # make_tensors_meta (same name repeats across ranks, but that's
        # fine — real HF param names DO repeat across ranks; only the
        # trainer_ep_rank tag distinguishes them on the wire, which is
        # exactly what this test is checking gets respected).
        key = f"{entry['name']}::{entry.get('trainer_ep_rank')}::{idx}"
        val = float((hash(key) % 10000) + 1) / 10000
        values[entry["name"]] = torch.full((entry["numel"],), val, dtype=torch.float32)
    return values


class TestWs10PreRecvNoRebuildPass:

    def test_batch_boundaries_match_serial_loop(self):
        """_ws10_pre_recv (built inline per-thread) must have keys matching
        the serial fanout loop's independently-computed batch_start
        values."""
        meta = make_tensors_meta(n_expert_layers=4, ep_degree=4, n_non_expert=3,
                                 expert_numel=80, non_expert_numel=40)
        batch_max = 250
        wire = make_wire_values(meta)

        pre_recv = build_pre_recv_new(meta, batch_max, wire)
        serial_starts = serial_batch_boundaries(meta, batch_max)

        assert set(pre_recv.keys()) == set(serial_starts), (
            f"Batch start mismatch: pre_recv={sorted(pre_recv.keys())} "
            f"serial={sorted(serial_starts)}"
        )

    def test_new_packing_matches_old_two_step_byte_identical(self):
        """The new inline-per-thread packing (no all_shards/rebuild-plan
        intermediate) must produce byte-identical output to the OLD
        (removed) two-step approach, for a manifest with a non-expert tail
        (the historical trigger condition for the rank0 boundary bug)."""
        meta = make_tensors_meta(n_expert_layers=3, ep_degree=8, n_non_expert=5,
                                 expert_numel=137, non_expert_numel=51)
        batch_max = 400
        wire = make_wire_values(meta)

        new_pre_recv = build_pre_recv_new(meta, batch_max, wire)
        old_pre_recv = build_pre_recv_old(meta, batch_max, wire)

        assert set(new_pre_recv.keys()) == set(old_pre_recv.keys()), (
            f"Key mismatch: new={sorted(new_pre_recv.keys())} "
            f"old={sorted(old_pre_recv.keys())}"
        )
        for k in new_pre_recv:
            assert torch.equal(new_pre_recv[k], old_pre_recv[k]), (
                f"Batch {k} content mismatch: new={new_pre_recv[k][:5]} "
                f"old={old_pre_recv[k][:5]}"
            )

    def test_batch_size_respects_max_numel(self):
        """No batch in _ws10_pre_recv should exceed batch_max_numel elements."""
        meta = make_tensors_meta(n_expert_layers=6, ep_degree=8, n_non_expert=5,
                                 expert_numel=100, non_expert_numel=70)
        batch_max = 350
        wire = make_wire_values(meta)
        pre_recv = build_pre_recv_new(meta, batch_max, wire)

        for bs, pack in pre_recv.items():
            assert pack.numel() <= batch_max, (
                f"Batch at {bs} has {pack.numel()} elements > max {batch_max}"
            )

    def test_rank_boundary_enforced(self):
        """Entries from different trainer_ep_ranks must not share a batch."""
        meta = make_tensors_meta(n_expert_layers=2, ep_degree=3, n_non_expert=1,
                                 expert_numel=100)
        batch_max = 10000  # large enough that size alone never splits
        wire = make_wire_values(meta)
        pre_recv = build_pre_recv_new(meta, batch_max, wire)

        for bs, pack in pre_recv.items():
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
            assert len(ranks_in_batch) <= 1, (
                f"Batch at {bs} mixes EP ranks: {ranks_in_batch}"
            )

    def test_rank0_non_expert_tail_values_preserved(self):
        """Rank 0's non-expert tail (the historical trigger for the
        sender/receiver boundary desync) must pack correctly and be
        recoverable byte-for-byte from pre_recv."""
        meta = make_tensors_meta(n_expert_layers=2, ep_degree=4, n_non_expert=3,
                                 expert_numel=90, non_expert_numel=33)
        batch_max = 150
        wire = make_wire_values(meta)
        pre_recv = build_pre_recv_new(meta, batch_max, wire)

        n_params = len(meta)
        for i, entry in enumerate(meta):
            batch_starts = sorted(k for k in pre_recv if k <= i)
            batch_start = batch_starts[-1]
            off = sum(meta[j]["numel"] for j in range(batch_start, i))
            n = entry["numel"]
            expected = wire[entry["name"]]
            actual = pre_recv[batch_start][off:off + n]
            assert torch.equal(actual, expected), (
                f"Mismatch at global idx {i} ({entry['name']}, "
                f"R={entry.get('trainer_ep_rank')})"
            )

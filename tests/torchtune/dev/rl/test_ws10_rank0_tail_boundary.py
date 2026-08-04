"""
CPU-safe regression test: WS10 sender/receiver batch-boundary desync for
rank 0's non-expert tail.

Root cause (2026-07-22): trainer rank 0 is the only WS10 EP rank carrying
non-expert params (embeddings, norms, router, attention). The sender
(``weight_sync.py::_xccl_gather_and_stage_fsdp2``) builds rank 0's batches
in two independent greedy passes: expert shards via
``_ws10_build_local_payload``, then a FRESH pass for the non-expert tail
that always starts a new batch at ``_cur_n = 0`` — even when the last
expert batch had leftover room under ``batch_max_numel``. The receiver's
actual wire loop (``vllm_weight_sync_worker.py::_ws10_parallel_recv``'s
inner ``recv_rank_thread``) greedily packed by byte cap ALONE, with no
forced break at the expert/non-expert transition, so it disagreed with the
sender about where batch N ends whenever the last expert entry didn't
exactly fill a batch. gloo's ``pg.broadcast()`` doesn't raise cleanly on a
receive-size mismatch — it just hangs the corresponding rank until gloo's
30-minute TCP timeout fires (observed via
``gloo::EnforceNotMet ... unbound_buffer.cc:129 Timed out waiting
1800000ms``, job 8683288, and a similarly-shaped size-mismatch crash in an
earlier run misdiagnosed at the time as an int32 preamble overflow, job
8683165).

This was latent at the validated 1 GiB default (the specific tensor sizes
there happened not to trigger a boundary split at the transition) and only
surfaced when the batch size was changed. Two OTHER loops in the same file
(the diagnostic batch counter and the ``_ws10_pre_recv`` rebuild-plan
loop) already forced this exact break via a ``trainer_ep_rank`` presence
check; only the real wire-receive loop was missing it. Fix: mirror the
same forced break in ``recv_rank_thread``.

This test replicates both algorithms standalone (no torch/distributed
required) and checks that the fixed boundary logic exactly matches the
sender's, using a case where the last expert entry does NOT exactly fill
a batch (reproducing the trigger condition).
"""


def sender_batches_rank0(expert_meta, non_expert_meta, batch_max_numel):
    """Mirrors weight_sync.py: _ws10_build_local_payload's greedy batching
    for the expert region, THEN a SEPARATE fresh greedy pass for the
    non-expert tail (lines ~2699-2742)."""
    batches = []
    cur = 0
    cur_start = 0
    i = 0
    n = len(expert_meta)
    while i < n:
        pn = expert_meta[i]["numel"]
        if cur > 0 and cur + pn > batch_max_numel:
            batches.append((cur_start, i, cur))
            cur_start = i
            cur = 0
        cur += pn
        i += 1
    if cur > 0:
        batches.append((cur_start, i, cur))

    cur = 0
    cur_start = 0
    j = 0
    m = len(non_expert_meta)
    while j < m:
        pn = non_expert_meta[j]["numel"]
        if cur > 0 and cur + pn > batch_max_numel:
            batches.append((cur_start, j, cur))
            cur_start = j
            cur = 0
        cur += pn
        j += 1
    if cur > 0:
        batches.append((cur_start, j, cur))
    return batches


def receiver_batches_fixed(entries, batch_max_numel):
    """Current (post-fix) recv_rank_thread boundary logic — forces a break
    whenever trainer_ep_rank changes (None for non-expert != int for
    expert), matching the sender's forced expert/non-expert split."""
    boundaries = []
    i = 0
    n = len(entries)
    while i < n:
        start = i
        bn = 0
        R = entries[start].get("trainer_ep_rank")
        while i < n:
            pn = entries[i]["numel"]
            Rn = entries[i].get("trainer_ep_rank")
            if bn > 0 and Rn != R:
                break
            if bn > 0 and bn + pn > batch_max_numel:
                break
            bn += pn
            i += 1
        boundaries.append((start, i, bn))
    return boundaries


def receiver_batches_broken(entries, batch_max_numel):
    """Pre-fix recv_rank_thread boundary logic — byte-cap only, no forced
    break at the expert/non-expert transition. Kept here only to prove the
    repro actually triggers the historical bug."""
    boundaries = []
    i = 0
    n = len(entries)
    while i < n:
        start = i
        bn = 0
        while i < n:
            pn = entries[i]["numel"]
            if bn > 0 and bn + pn > batch_max_numel:
                break
            bn += pn
            i += 1
        boundaries.append((start, i, bn))
    return boundaries


def _make_scenario():
    # Expert region totals 70 elements (doesn't exactly fill batch_max=100)
    # followed by a non-expert tail that the sender ALWAYS starts fresh --
    # this is the trigger condition for the historical desync.
    batch_max = 100
    expert_meta = [
        {"name": "model.layers.0.mlp.experts.w13_weight", "numel": 40,
         "trainer_ep_rank": 0},
        {"name": "model.layers.0.mlp.experts.w2_weight", "numel": 30,
         "trainer_ep_rank": 0},
    ]
    non_expert_meta = [
        {"name": "model.embed_tokens.weight", "numel": 20},
        {"name": "model.norm.weight", "numel": 15},
    ]
    return expert_meta, non_expert_meta, batch_max


class TestWs10Rank0TailBoundary:

    def test_broken_receiver_diverges_from_sender(self):
        """Sanity: confirm the scenario actually triggers the historical
        bug (byte-cap-only receiver disagrees with the sender's forced
        break) — otherwise this test would pass trivially without
        exercising the real failure mode."""
        expert_meta, non_expert_meta, batch_max = _make_scenario()
        sender = sender_batches_rank0(expert_meta, non_expert_meta, batch_max)
        combined = expert_meta + non_expert_meta
        broken = receiver_batches_broken(combined, batch_max)
        assert [b[2] for b in sender] != [b[2] for b in broken], (
            "scenario failed to reproduce the sender/receiver boundary "
            "mismatch — broken receiver accidentally agrees with sender"
        )

    def test_fixed_receiver_matches_sender_exactly(self):
        """The actual regression guard: fixed receiver batch byte-counts
        (and thus wire sizes) must exactly match the sender's, including
        at the expert -> non-expert transition."""
        expert_meta, non_expert_meta, batch_max = _make_scenario()
        sender = sender_batches_rank0(expert_meta, non_expert_meta, batch_max)
        combined = expert_meta + non_expert_meta
        fixed = receiver_batches_fixed(combined, batch_max)
        assert [b[2] for b in sender] == [b[2] for b in fixed], (
            f"fixed receiver still diverges from sender: "
            f"sender={sender} receiver={fixed}"
        )

    def test_non_rank0_peer_unaffected(self):
        """Peer EP ranks (no non-expert tail) never hit the boundary case
        — their batches must be unaffected by the fix (regression guard
        against overcorrecting)."""
        batch_max = 100
        expert_meta = [
            {"name": "model.layers.0.mlp.experts.w13_weight", "numel": 60,
             "trainer_ep_rank": 3},
            {"name": "model.layers.0.mlp.experts.w2_weight", "numel": 30,
             "trainer_ep_rank": 3},
            {"name": "model.layers.1.mlp.experts.w13_weight", "numel": 50,
             "trainer_ep_rank": 3},
        ]
        sender = sender_batches_rank0(expert_meta, [], batch_max)
        fixed = receiver_batches_fixed(expert_meta, batch_max)
        broken = receiver_batches_broken(expert_meta, batch_max)
        assert [b[2] for b in sender] == [b[2] for b in fixed]
        assert [b[2] for b in sender] == [b[2] for b in broken], (
            "peer-rank-only batching should be identical between broken "
            "and fixed receiver logic — the fix must only change behavior "
            "at the expert/non-expert transition"
        )

    def test_exact_fill_at_transition_is_a_noop_case(self):
        """When the expert region exactly fills a batch (no leftover
        room), broken and fixed logic coincide — this is why the bug was
        latent at the validated 1 GiB default and only surfaced at other
        batch sizes."""
        batch_max = 70
        expert_meta = [
            {"name": "model.layers.0.mlp.experts.w13_weight", "numel": 40,
             "trainer_ep_rank": 0},
            {"name": "model.layers.0.mlp.experts.w2_weight", "numel": 30,
             "trainer_ep_rank": 0},
        ]
        non_expert_meta = [
            {"name": "model.embed_tokens.weight", "numel": 20},
        ]
        sender = sender_batches_rank0(expert_meta, non_expert_meta, batch_max)
        combined = expert_meta + non_expert_meta
        broken = receiver_batches_broken(combined, batch_max)
        fixed = receiver_batches_fixed(combined, batch_max)
        assert [b[2] for b in sender] == [b[2] for b in broken] == [
            b[2] for b in fixed
        ]

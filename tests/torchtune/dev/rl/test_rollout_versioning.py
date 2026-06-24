# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Version-bump precision regression test for `_wait_for_sync_complete`.

Before the fix, the rank-0 weight version bumped on every call to
`_wait_for_sync_complete` because `_sync_done_event.is_set()` is True at
construction time. Two consecutive calls without an intervening dispatch
inflated the version counter, and the async producer could mis-tag rollouts
with phantom weight generations.

The fix adds `_pending_sync_id` (incremented at every dispatch site,
cleared after the bump). This test exercises the corrected lifecycle on a
hand-built fake recipe — no XPU, no distributed init.
"""
import logging
import threading

import pytest

from torchtune.dev.rl.async_rollout import WeightVersionTracker
from torchtune.dev.rl.weight_sync import _wait_for_sync_complete


class _FakeRecipe:
    """Minimum surface area `_wait_for_sync_complete` reads from `self`."""

    def __init__(self):
        self.rank = 0
        self._is_rank_zero = True
        self._vllm_weight_sync = True
        self._weight_versions = WeightVersionTracker()
        self._sync_done_event = threading.Event()
        self._sync_done_event.set()  # matches construction-time invariant
        self._pending_sync_id = None
        self._sync_id_counter = 0
        self._sync_error = None

    def dispatch_sync(self):
        """Mirror what _sync_dedicated_vllm_weights / xccl / shm do."""
        self._sync_done_event.clear()
        self._sync_id_counter += 1
        self._pending_sync_id = self._sync_id_counter

    def signal_sync_done(self):
        """Mirror the worker thread setting the event when the bcast returns."""
        self._sync_done_event.set()


def test_no_bump_when_no_sync_dispatched():
    recipe = _FakeRecipe()
    assert recipe._weight_versions.version == 0
    _wait_for_sync_complete(recipe)
    assert recipe._weight_versions.version == 0, (
        "version must NOT bump when _wait_for_sync_complete is called with no "
        "pending sync (the bug was bumping on every call because the event "
        "starts in the set state)"
    )


def test_one_bump_per_dispatch_completion_cycle():
    recipe = _FakeRecipe()

    recipe.dispatch_sync()
    recipe.signal_sync_done()
    _wait_for_sync_complete(recipe)
    assert recipe._weight_versions.version == 1
    assert recipe._pending_sync_id is None, (
        "_pending_sync_id must be cleared after the bump so a follow-up call "
        "without a new dispatch does not double-count"
    )

    # Second wait with no new dispatch — version must NOT change.
    _wait_for_sync_complete(recipe)
    assert recipe._weight_versions.version == 1


def test_repeated_dispatch_completion_cycles_increment_one_at_a_time():
    recipe = _FakeRecipe()
    for expected in range(1, 6):
        recipe.dispatch_sync()
        recipe.signal_sync_done()
        _wait_for_sync_complete(recipe)
        assert recipe._weight_versions.version == expected


def test_warning_emitted_when_no_pending_sync(caplog):
    recipe = _FakeRecipe()
    with caplog.at_level(logging.WARNING, logger="torchtune.dev.rl.weight_sync"):
        _wait_for_sync_complete(recipe)
    assert any(
        "no pending sync" in rec.message for rec in caplog.records
    ), "expected warning when wait is called with no pending sync"


def test_failed_sync_does_not_bump_version():
    """A failed background sync (e.g. POST/broadcast raised) must NOT bump
    the version counter. Without this guard, telemetry and the async producer
    would believe vLLM holds new weights when the broadcast never landed.
    """
    recipe = _FakeRecipe()
    recipe.dispatch_sync()
    recipe.signal_sync_done()
    recipe._sync_error = RuntimeError("simulated bcast failure")
    _wait_for_sync_complete(recipe)
    assert recipe._weight_versions.version == 0, (
        "failed sync must not bump the weight version"
    )
    assert recipe._pending_sync_id == 1, (
        "_pending_sync_id must be retained on failure so a successful retry "
        "can still bump"
    )
    assert recipe._sync_error is None, "error flag must be reset for next step"


def test_failed_then_successful_sync_bumps_once():
    """After a failed sync, a subsequent successful sync (same _pending_sync_id
    or a new dispatch) bumps the version exactly once.
    """
    recipe = _FakeRecipe()
    recipe.dispatch_sync()
    recipe.signal_sync_done()
    recipe._sync_error = RuntimeError("first attempt failed")
    _wait_for_sync_complete(recipe)
    assert recipe._weight_versions.version == 0

    # Retry: clear the event, re-signal completion, no new dispatch needed.
    recipe._sync_done_event.clear()
    recipe.signal_sync_done()
    _wait_for_sync_complete(recipe)
    assert recipe._weight_versions.version == 1
    assert recipe._pending_sync_id is None


def test_noop_when_vllm_weight_sync_disabled():
    recipe = _FakeRecipe()
    recipe._vllm_weight_sync = False
    recipe.dispatch_sync()
    recipe.signal_sync_done()
    _wait_for_sync_complete(recipe)
    # Early return — no bump, pending state untouched.
    assert recipe._weight_versions.version == 0
    assert recipe._pending_sync_id == 1


# ---------------------------------------------------------------------------
# Staleness pin (=1): the BioReason async lookahead tags each rollout with the
# weight version live at the MAIN-THREAD post point (the _weight_version key on
# the mailbox dict), NOT a producer-pickup snapshot. This bounds the consume-time
# lag to exactly 1 (the HW bug plateaued at 2 because the pickup snapshot drifted
# by the queue depth). These tests pin the RolloutProducer honouring that tag.
# ---------------------------------------------------------------------------
def test_producer_honours_pretagged_weight_version():
    from queue import Queue

    from torchtune.dev.rl.async_rollout import RolloutProducer

    versions = WeightVersionTracker()
    inbox: Queue = Queue(maxsize=1)

    def batch_iter_fn():
        return inbox.get()

    p = RolloutProducer(
        produce_fn=lambda w: ({"qr": w["id"]}, {}),
        batch_iter_fn=batch_iter_fn,
        weight_versions=versions,
        max_staleness=1,
        warmup=False,
    )
    p.start()
    try:
        # Bump the tracker AFTER posting work tagged with the CURRENT (pre-bump)
        # version. If the producer re-snapshotted at pickup it could read the
        # bumped value; with the pin it must use the posted tag.
        inbox.put({"id": 0, "_weight_version": versions.version})  # tag = 0
        versions.bump()  # tracker -> 1 (simulate a wsync racing the pickup)
        item0 = p.get(timeout=5.0)
        assert item0.weight_version == 0, (
            "producer must honour the pre-tagged _weight_version (post-time), "
            f"not re-snapshot at pickup; got {item0.weight_version}"
        )
        inbox.put(None)
        assert p.get(timeout=5.0) is None
    finally:
        p.stop()


def test_pinned_lag_never_exceeds_one_under_simulated_loop():
    """Simulate the recipe's main-loop ordering (post i, consume i-1, bump) and
    assert the consume-time lag is bounded to 1 for every rollout when the tag is
    taken at post time (the staleness pin). Mirrors _async_lookahead_iter_impl."""
    from queue import Queue

    from torchtune.dev.rl.async_rollout import RolloutProducer

    versions = WeightVersionTracker()
    inbox: Queue = Queue(maxsize=1)

    p = RolloutProducer(
        produce_fn=lambda w: ({"id": w["id"]}, {}),
        batch_iter_fn=lambda: inbox.get(),
        weight_versions=versions,
        max_staleness=1,
        warmup=False,
    )
    p.start()
    try:
        pending = None
        lags = []
        for i in range(6):
            # post i tagged with the version live NOW (before training i-1).
            inbox.put({"id": i, "_weight_version": versions.version})
            if pending is not None:
                item = p.get(timeout=5.0)
                lag = max(0, versions.version - item.weight_version)
                lags.append(lag)
                versions.bump()  # end-of-step wsync for the consumed batch
            pending = i
        # drain last
        item = p.get(timeout=5.0)
        lags.append(max(0, versions.version - item.weight_version))
        inbox.put(None)
        assert p.get(timeout=5.0) is None
        assert all(l <= 1 for l in lags), f"lag must be pinned to <=1, got {lags}"
    finally:
        p.stop()

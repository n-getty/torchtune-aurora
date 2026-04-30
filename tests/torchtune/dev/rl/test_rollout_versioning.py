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


def test_noop_when_vllm_weight_sync_disabled():
    recipe = _FakeRecipe()
    recipe._vllm_weight_sync = False
    recipe.dispatch_sync()
    recipe.signal_sync_done()
    _wait_for_sync_complete(recipe)
    # Early return — no bump, pending state untouched.
    assert recipe._weight_versions.version == 0
    assert recipe._pending_sync_id == 1

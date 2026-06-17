# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe lifecycle tests for RolloutProducer (async GRPO overlap).

``WeightVersionTracker`` is already covered by test_rollout_versioning.py.
This file covers the producer THREAD contract, which has no other coverage and
is on the async-GRPO hot path:
  * clean exhaustion: ``batch_iter_fn() -> None`` ends iteration with a sentinel,
    not a hang;
  * error propagation: an exception in ``produce_fn`` surfaces on ``get()`` as
    ``RuntimeError("Producer thread died")`` rather than silently stalling;
  * weight-version tagging at produce time (staleness accounting);
  * ``max_staleness`` bounds the queue (back-pressure);
  * the ``max_staleness < 1`` guard.

Pure threading + simple callables — no XPU, no vLLM, no distributed init. All
waits are bounded so a regression that hangs fails as a timeout, never blocks
the suite.
"""
from __future__ import annotations

import threading

import pytest

from torchtune.dev.rl.async_rollout import RolloutProducer, WeightVersionTracker


def _producer(batches, versions=None, fail_on=None, max_staleness=2):
    """Build a producer over a fixed list of batches.

    fail_on: 1-based produce call index at which produce_fn raises.
    """
    versions = versions or WeightVersionTracker()
    it = iter(batches)
    calls = {"n": 0}

    def batch_iter_fn():
        return next(it, None)

    def produce_fn(batch):
        calls["n"] += 1
        if fail_on is not None and calls["n"] == fail_on:
            raise RuntimeError("simulated produce failure")
        return {"out": batch}, {"telem_key": 1.0}

    p = RolloutProducer(
        produce_fn=produce_fn,
        batch_iter_fn=batch_iter_fn,
        weight_versions=versions,
        max_staleness=max_staleness,
        warmup=False,
    )
    return p


def test_invalid_max_staleness_raises():
    with pytest.raises(ValueError, match="max_staleness"):
        RolloutProducer(
            produce_fn=lambda b: (b, {}),
            batch_iter_fn=lambda: None,
            weight_versions=WeightVersionTracker(),
            max_staleness=0,
        )


def test_consumes_all_then_exhausts():
    p = _producer([{"x": 0}, {"x": 1}, {"x": 2}])
    p.start()
    try:
        items = list(p)  # __iter__ stops at the exhaustion sentinel
    finally:
        p.stop()
    assert len(items) == 3
    assert [it.batch_meta["batch"]["x"] for it in items] == [0, 1, 2]


def test_empty_iter_returns_none_immediately():
    p = _producer([])
    p.start()
    try:
        assert p.get(timeout=5.0) is None
    finally:
        p.stop()


def test_producer_error_surfaces_on_get():
    p = _producer([{"x": 0}, {"x": 1}], fail_on=1)
    p.start()
    try:
        with pytest.raises(RuntimeError, match="Producer thread died"):
            # Drain until the dead-thread error is raised.
            for _ in range(5):
                p.get(timeout=5.0)
    finally:
        p.stop()


def test_weight_version_tagged_at_produce_time():
    versions = WeightVersionTracker()
    versions.bump()  # version == 1 before the producer runs
    p = _producer([{"x": 0}], versions=versions, max_staleness=1)
    p.start()
    try:
        item = p.get(timeout=5.0)
        assert item is not None
        assert item.weight_version == 1
    finally:
        p.stop()


def test_max_staleness_bounds_queue():
    # With max_staleness=1 the queue holds at most 1 item; the producer blocks
    # on put until the consumer drains. Producing 3 must still yield all 3.
    p = _producer([{"x": i} for i in range(3)], max_staleness=1)
    p.start()
    try:
        # Give the producer a moment; the bounded queue must not exceed 1.
        first = p.get(timeout=5.0)
        assert first is not None
        assert p.qsize() <= 1
        rest = []
        while True:
            it = p.get(timeout=5.0)
            if it is None:
                break
            rest.append(it)
        assert len(rest) == 2  # 3 produced, 1 already consumed
    finally:
        p.stop()


def test_stop_is_idempotent_and_safe_before_exhaustion():
    p = _producer([{"x": i} for i in range(100)], max_staleness=1)
    p.start()
    try:
        assert p.get(timeout=5.0) is not None
    finally:
        p.stop()
        p.stop()  # second stop must not raise
    assert not p._thread.is_alive()  # thread joined


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

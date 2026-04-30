# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async rollout producer for GRPO.

Phase 1 (server mode only):
    Single producer thread on rank 0 that overlaps the vLLM HTTP generate()
    call with the previous step's training. The trajectory tensors are
    produced on rank 0 and broadcast to all training ranks at consume time
    via the existing world process group.

Phase 2 (planned):
    Continuous producer with bounded queue (max_staleness > 1), an
    independent vLLM-side dataloader replica, and importance-sampling
    correction with epsilon_high (DAPO-style dual clip).

Design constraints (Aurora XPU + dedicated_rank vLLM):
    The dedicated_rank generation path uses broadcast_object_list over the
    training PG, which requires every training rank to call together. That
    mode is therefore *not* a target for Phase 1; only `vllm_mode == "server"`
    (HTTP) is. A producer thread can drive HTTP completions independently
    while the rest of the training step proceeds.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class RolloutItem:
    """A single rollout produced by the async producer.

    Attributes:
        batch_meta: Opaque per-batch metadata (input_ids, answers,
            protein_sequences). Pulled from the dataloader at produce time
            and forwarded to the consumer for ref/policy fwd + reward calc.
        weight_version: Monotonic counter incremented at each successful
            vLLM weight sync. Tagged at produce time so the consumer can
            compute staleness against the current trainer version.
        produce_time: Wall-clock time the rollout was placed on the queue
            (for telemetry).
        produce_latency_s: Time spent inside `_generate_with_vllm` for
            this rollout (telemetry only).
    """

    batch_meta: dict
    weight_version: int
    produce_time: float
    produce_latency_s: float


class WeightVersionTracker:
    """Monotonic version counter for weight syncs.

    Updated by the training loop after each successful vLLM weight sync.
    Read by the producer to tag in-flight rollouts.
    """

    def __init__(self):
        self._version = 0
        self._lock = threading.Lock()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def bump(self) -> int:
        with self._lock:
            self._version += 1
            return self._version


class RolloutProducer:
    """Single-thread rollout producer for Phase 1 (k=1) async generation.

    The producer thread drains an iterator of dataloader batches, runs the
    vLLM HTTP generate path, and pushes results onto a bounded queue. The
    training loop pulls from the queue at the top of each step.

    The producer is driven by `Callable[[batch], RolloutItem]` so the
    recipe can keep its existing `_generate_with_vllm` and ref/policy fwd
    code paths inline at consume time. Phase 1 only async-overlaps the
    vLLM HTTP roundtrip; ref/policy fwd run synchronously on the consumer
    side (they are XPU collectives that need every training rank).

    Args:
        produce_fn: Closure that takes a dataloader batch dict and returns
            a (query_responses_cpu_tensor, telemetry_dict) tuple. Runs
            inside the producer thread; must be thread-safe vs the rest
            of the trainer.
        batch_iter_fn: Closure that returns the next dataloader batch dict
            (called inside the producer thread). The recipe owns the
            actual iterator (sampler invariants live there) and just hands
            the producer a thunk.
        weight_versions: Tracker so each rollout gets tagged with the
            currently-active vLLM weight version.
        max_staleness: Queue capacity (= max steps the rollout can lag
            the current weights). Phase 1 fixes this at 1.
        warmup: If True, the producer immediately starts producing the
            first rollout so step 0 finds it ready.
        name: Thread name (for logs).
    """

    def __init__(
        self,
        produce_fn: Callable[[dict], tuple[Any, dict]],
        batch_iter_fn: Callable[[], Optional[dict]],
        weight_versions: WeightVersionTracker,
        max_staleness: int = 1,
        warmup: bool = True,
        name: str = "rollout_producer",
    ):
        if max_staleness < 1:
            raise ValueError(f"max_staleness must be >= 1, got {max_staleness}")
        self._produce_fn = produce_fn
        self._batch_iter_fn = batch_iter_fn
        self._weight_versions = weight_versions
        self._queue: Queue[RolloutItem] = Queue(maxsize=max_staleness)
        self._max_staleness = max_staleness
        self._stop_event = threading.Event()
        self._exhausted = False
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._warmup = warmup
        self._produced = 0
        self._consumed = 0
        # Telemetry: cumulative time the producer was blocked on a full
        # queue. Read+reset by the consumer each step (Step 3 telemetry).
        self._time_blocked_on_put_s = 0.0
        self._tel_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        log.info(
            "RolloutProducer.start: max_staleness=%d warmup=%s",
            self._max_staleness, self._warmup,
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        log.info("RolloutProducer.stop: signalling thread")
        self._stop_event.set()
        try:
            # Drain any held slot so the producer can observe the stop event.
            while True:
                self._queue.get_nowait()
        except Empty:
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            log.warning("RolloutProducer.stop: thread did not exit within %.1fs", timeout)

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                batch = self._batch_iter_fn()
                if batch is None:
                    log.info("RolloutProducer: batch_iter_fn returned None — exiting")
                    self._exhausted = True
                    # Wake any consumer blocked on get().
                    try:
                        self._queue.put(None, timeout=5.0)  # type: ignore[arg-type]
                    except Full:
                        pass
                    return
                t0 = time.perf_counter()
                # Snapshot weight version BEFORE producing so the rollout
                # is tagged with the version it actually generated under.
                weight_version_at_start = self._weight_versions.version
                rollout_payload, telem = self._produce_fn(batch)
                latency = time.perf_counter() - t0
                item = RolloutItem(
                    batch_meta={
                        "batch": batch,
                        "rollout_payload": rollout_payload,
                        **telem,
                    },
                    weight_version=weight_version_at_start,
                    produce_time=time.time(),
                    produce_latency_s=latency,
                )
                # blocking put — back-pressure when the consumer is slow.
                _put_t0 = time.perf_counter()
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(item, timeout=1.0)
                        break
                    except Full:
                        continue
                _put_blocked = time.perf_counter() - _put_t0
                if _put_blocked > 0.0:
                    with self._tel_lock:
                        self._time_blocked_on_put_s += _put_blocked
                self._produced += 1
                log.info(
                    "RolloutProducer: produced #%d (weight_version=%d, latency=%.2fs, qsize=%d)",
                    self._produced, weight_version_at_start, latency, self._queue.qsize(),
                )
        except BaseException as exc:  # noqa: BLE001 — propagate everything
            log.exception("RolloutProducer: fatal exception in producer thread")
            self._error = exc

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------
    def get(self, timeout: float = 600.0) -> Optional[RolloutItem]:
        """Block until the next rollout is available.

        Returns None when the producer's batch_iter_fn has been exhausted
        (clean end-of-data). Raises whatever exception killed the producer
        thread (if any), or TimeoutError if no rollout arrives within
        ``timeout`` seconds.
        """
        if self._error is not None:
            raise RuntimeError("Producer thread died") from self._error
        try:
            item = self._queue.get(timeout=timeout)
        except Empty as exc:
            if self._error is not None:
                raise RuntimeError("Producer thread died") from self._error
            raise TimeoutError(
                f"RolloutProducer.get timed out after {timeout:.1f}s"
            ) from exc
        if item is None:
            # Exhaustion sentinel; do not increment consumed.
            return None
        self._consumed += 1
        return item

    def __iter__(self):
        """Iterate `(batch, rollout_payload, item)` tuples until exhausted.

        Convenience wrapper for `for item in producer:` style consumption
        from the train loop. The full :class:`RolloutItem` is yielded so
        callers can read weight_version / latency telemetry.
        """
        while True:
            item = self.get()
            if item is None:
                return
            yield item

    def qsize(self) -> int:
        return self._queue.qsize()

    def read_blocked_on_put_ms(self) -> float:
        """Read and reset the cumulative producer-block-on-put counter.

        Called once per consumer step so the metrics line shows blocking
        since the last step rather than total since start.
        """
        with self._tel_lock:
            ms = self._time_blocked_on_put_s * 1000.0
            self._time_blocked_on_put_s = 0.0
        return ms

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def stats(self) -> dict:
        return {
            "produced": self._produced,
            "consumed": self._consumed,
            "qsize": self._queue.qsize(),
            "error": str(self._error) if self._error else None,
            "exhausted": self._exhausted,
        }

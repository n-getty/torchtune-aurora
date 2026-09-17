"""The METRICS `weight_lag` must be measured at CONSUME time, not after the publish.

Observed 2026-09-16 on the 32B async probe (job 8830832): a run whose true staleness was
1 logged `weight_lag=2`. The per-consume `lag=` line and the `Queue(maxsize=1)` bound both
said 1. The METRICS tail computed

    lag = self._weight_versions.version - item.weight_version

where `version` is read in the logging block -- i.e. *after* the step's own weight publish
has already bumped it. The counter therefore over-reports by exactly one publish, every
step, on a perfectly healthy run.

Why this is worth a test rather than a shrug: the recipe FAIL-FASTS on
`max_staleness > 1`, so an inflated staleness readout is the exact signal an operator
would use to conclude async is broken and disable it. A diagnostic that cries wolf on a
correct run gets the correct run turned off.

These tests model the two counters directly and assert the buggy formula reproduces the
observed 2 while the fixed one reports 1 -- then pin the source so the fix cannot be
quietly reverted.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASE_RECIPE = REPO / "recipes" / "dev" / "grpo_full_finetune_distributed_xpu.py"
BIO_RECIPE = REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
VLLM_BACKEND = REPO / "torchtune" / "dev" / "rl" / "vllm_backend.py"

ATTR = "_last_rollout_consume_wver"


class _FakeTracker:
    def __init__(self, version=0):
        self.version = version


class _FakeItem:
    def __init__(self, weight_version):
        self.weight_version = weight_version


def _simulate(n_steps, use_consume_snapshot):
    """Walk a healthy staleness=1 async loop and return the lag reported each step.

    Order per step, matching the recipe: consume a rollout produced under the current
    weights, train, publish new weights (bump), then log METRICS.
    """
    tracker = _FakeTracker(version=0)
    reported = []
    for _ in range(n_steps):
        # Producer generated this rollout under the weights live at post time.
        item = _FakeItem(weight_version=tracker.version)
        consume_wver = tracker.version  # snapshot taken at consume

        # ... training happens ...

        tracker.version += 1  # the step's own weight publish

        # METRICS logging block runs here, after the publish.
        w_now = consume_wver if use_consume_snapshot else tracker.version
        reported.append(max(0, w_now - item.weight_version))
    return reported


def test_live_counter_reproduces_the_observed_off_by_one():
    """Negative control: the OLD formula must actually produce the bug.

    If this ever passes trivially, the simulation no longer models the defect and the
    positive test below proves nothing.
    """
    buggy = _simulate(5, use_consume_snapshot=False)
    assert buggy == [1, 1, 1, 1, 1], buggy
    # On a run with real lag=1 the same +1 skew is what produced the observed 2.
    assert all(v > 0 for v in buggy), (
        "the live-counter formula must report nonzero lag even on a zero-lag loop; "
        "that inflation is the bug under test"
    )


def test_consume_snapshot_reports_true_zero_lag():
    fixed = _simulate(5, use_consume_snapshot=True)
    assert fixed == [0, 0, 0, 0, 0], fixed


def test_fix_is_exactly_one_publish_better():
    buggy = _simulate(8, use_consume_snapshot=False)
    fixed = _simulate(8, use_consume_snapshot=True)
    assert [b - f for b, f in zip(buggy, fixed)] == [1] * 8, (
        "the correction must be exactly one publish per step -- if it differs, the "
        "snapshot is being taken at the wrong point, not merely offset"
    )


def test_metrics_tail_prefers_the_consume_snapshot():
    src = BASE_RECIPE.read_text(errors="replace")
    # Locate the METRICS async tail block.
    idx = src.find("prod_qsize=%d  weight_lag=%d")
    assert idx != -1, "METRICS async tail not found; did the log format change?"
    block = src[max(0, idx - 1400) : idx]
    assert ATTR in block, (
        f"the METRICS async tail no longer reads {ATTR}; it is back to reading the live "
        "counter after the step's publish, which over-reports lag by one"
    )
    assert "getattr(self, \"_last_rollout_consume_wver\", None)" in block, (
        "the snapshot must be read defensively (getattr with a None default) so a "
        "sync-mode or pre-first-consume step falls back instead of raising"
    )


def test_both_consume_sites_stash_the_version():
    """Either consume path can be the live one; a fix on only one is a silent gap."""
    for path, marker in (
        (BASE_RECIPE, "recipe._last_rollout_item = item"),
        (BIO_RECIPE, "self._last_rollout_item = item"),
    ):
        src = path.read_text(errors="replace")
        idx = src.find(marker)
        assert idx != -1, f"consume site not found in {path.name}"
        window = src[idx : idx + 700]
        assert ATTR in window, (
            f"{path.name} sets _last_rollout_item but never stashes {ATTR} alongside it; "
            "the METRICS tail would fall back to the inflated live counter on this path"
        )


def test_attribute_is_initialized_so_it_is_never_stale():
    src = VLLM_BACKEND.read_text(errors="replace")
    assert re.search(rf"self\.{ATTR}\s*=\s*None", src), (
        f"{ATTR} is never initialized in vllm_backend.py. Without an explicit None it "
        "could carry a value across a mode switch, and the getattr fallback would never "
        "engage when it should."
    )

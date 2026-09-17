# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""The async 32B config is a COPY of the sync one; pin the copy to its parent.

`bioreason_32b_lora_grpo_hsdp_xpu_async.yaml` was produced by copying
`bioreason_32b_lora_grpo_hsdp_xpu.yaml` and changing four things. Its own header
says "everything else is byte-identical so an A/B against the sync parent isolates
the overlap, not a config drift" -- but nothing enforced that, and the *parent* is
the file the campaign actually edits (batch_size, max_gen, fbs, LoRA rank, reward
knobs all live there). A tuning change lands in the parent, the async copy keeps the
old value, and the next async-vs-sync A/B silently measures the config delta instead
of the overlap. That is the same failure class as
memory/feedback_a_new_branch_must_set_what_its_siblings_set.md: the invariant is the
sibling INTERSECTION, and a copy is a sibling that stops being told things.

`test_async_loss_combo.py` does NOT cover this. It checks the async/loss/logprob
triple *within* one file; it is satisfied by an async config whose batch_size has
drifted to half the parent's.

WHAT COUNTS AS SANCTIONED. Exactly the four deltas the async header documents:
  1-2. name / output_dir  -- must differ (a shared output_dir clobbers; that is its
       own known burn), and the async value must be derived from the parent's.
  3.   async_generation.* + always_compute_rollout_logprobs -- keys ADDED by async.
  4.   loss._component_   -- GRPOSimpleLoss -> GRPOLoss.
Plus any key whose value is the sync value with the sync output_dir rewritten to the
async one (`metric_logger.log_dir`, `profiler.output_dir`, ...). Those are not a
separate decision -- they are output_dir delta #2 propagating, and they MUST track it
or the async run writes its metrics into the sync campaign's directory. This is
computed, not enumerated: a config that later adds a third derived path is covered,
while a path that FAILS to track output_dir is caught as drift. Both directions
matter, which is why the rule is a substitution rather than a key allow-list.

Any other differing or missing key is drift and fails, naming the key.
"""
from pathlib import Path

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf

PROD_DIR = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "configs"
    / "dev"
    / "production"
)
SYNC = PROD_DIR / "bioreason_32b_lora_grpo_hsdp_xpu.yaml"
ASYNC = PROD_DIR / "bioreason_32b_lora_grpo_hsdp_xpu_async.yaml"

# Keys the async variant is allowed to differ on. Anything else is drift.
SANCTIONED_DIFFERENT = {"name", "output_dir", "loss._component_"}
# Keys the async variant is allowed to ADD (absent from the sync parent).
SANCTIONED_ADDED = {
    "async_generation.enabled",
    "async_generation.max_staleness",
    "always_compute_rollout_logprobs",
}


def _flatten(cfg, prefix=""):
    """Config -> {dotted.key: value}. Lists are compared whole, not per element."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, DictConfig):
            out.update(_flatten(v, prefix=f"{key}."))
        elif isinstance(v, ListConfig):
            out[key] = list(v)
        else:
            out[key] = v
    return out


def _load_pair():
    for p in (SYNC, ASYNC):
        if not p.exists():
            pytest.skip(f"{p.name} not present")
    return _flatten(OmegaConf.load(str(SYNC))), _flatten(OmegaConf.load(str(ASYNC)))


def _is_derived_path(sync_val, async_val, sync_out, async_out):
    """True if async_val is sync_val with the sync output_dir swapped for the async
    one -- i.e. the value is a path DERIVED from output_dir, tracking it correctly."""
    if not (isinstance(sync_val, str) and isinstance(async_val, str)):
        return False
    if not (isinstance(sync_out, str) and isinstance(async_out, str)):
        return False
    if sync_out not in sync_val:
        return False
    return sync_val.replace(sync_out, async_out) == async_val


def _drift(sync, asy):
    """Return (changed, added, removed) keys that are NOT sanctioned."""
    sync_out, async_out = sync.get("output_dir"), asy.get("output_dir")
    changed = sorted(
        k
        for k in set(sync) & set(asy)
        if sync[k] != asy[k]
        and k not in SANCTIONED_DIFFERENT
        and not _is_derived_path(sync[k], asy[k], sync_out, async_out)
    )
    added = sorted(set(asy) - set(sync) - SANCTIONED_ADDED)
    removed = sorted(set(sync) - set(asy))
    return changed, added, removed


def test_async_config_has_not_drifted_from_its_sync_parent():
    sync, asy = _load_pair()
    changed, added, removed = _drift(sync, asy)
    assert not changed, (
        "async config DRIFTED from its sync parent on: "
        + ", ".join(f"{k} (sync={sync[k]!r} async={asy[k]!r})" for k in changed)
        + ". An async-vs-sync A/B now measures this difference, not the overlap. "
        "Re-copy the value from the parent, or add the key to SANCTIONED_DIFFERENT "
        "with a reason."
    )
    assert not added, (
        f"async config has unsanctioned extra keys: {added}. If a new async-only "
        "knob is intended, add it to SANCTIONED_ADDED so the intent is reviewable."
    )
    assert not removed, (
        f"async config is MISSING keys its sync parent sets: {removed}. A missing "
        "key silently falls back to a recipe default, which is the "
        "sibling-intersection burn."
    )


def test_sanctioned_deltas_are_actually_present():
    """Non-vacuity: the four deltas must exist, else the test above passes on a
    file that is merely a duplicate of the sync config (async never enabled)."""
    sync, asy = _load_pair()
    assert asy.get("async_generation.enabled") is True
    assert asy.get("async_generation.max_staleness") == 1
    assert asy.get("always_compute_rollout_logprobs") is True
    assert asy.get("loss._component_") == "torchtune.dev.rl.loss.GRPOLoss"
    assert sync.get("loss._component_") == "torchtune.dev.rl.loss.GRPOSimpleLoss"


def test_output_dir_and_name_differ():
    """Two configs sharing an output_dir clobber each other's checkpoints."""
    sync, asy = _load_pair()
    assert sync["output_dir"] != asy["output_dir"]
    assert sync["name"] != asy["name"]


def test_drift_detector_is_not_vacuous():
    """Inject a realistic drift (a tuning knob changed in the parent only) and
    assert the detector fires. Without this the test above can pass because the
    comparison is broken, not because the configs agree."""
    sync, asy = _load_pair()
    key = "batch_size" if "batch_size" in sync else sorted(set(sync) & set(asy))[0]
    poisoned = dict(asy)
    poisoned[key] = "___injected_drift___"
    changed, _, _ = _drift(sync, poisoned)
    assert key in changed, (
        f"drift detector did NOT fire on an injected change to {key!r} -- the "
        "comparison is broken, so a green run above means nothing."
    )


def test_derived_path_exemption_does_not_swallow_a_stale_path():
    """The output_dir-substitution exemption must not become a blanket pass for any
    differing path. A derived path left pointing at the SYNC output_dir is the exact
    bug it is meant to catch (async metrics written into the sync campaign's dir),
    so poison one to the sync value and assert it is reported as drift."""
    sync, asy = _load_pair()
    derived = [
        k
        for k in set(sync) & set(asy)
        if sync[k] != asy[k]
        and _is_derived_path(sync[k], asy[k], sync["output_dir"], asy["output_dir"])
    ]
    assert derived, (
        "no derived-path keys found -- this test is vacuous; if the configs no "
        "longer derive any path from output_dir, delete it."
    )
    key = sorted(derived)[0]
    poisoned = dict(asy)
    poisoned[key] = sync[key]  # stale: still points into the SYNC output_dir
    changed, _, _ = _drift(sync, poisoned)
    assert key not in changed, "identical values are not drift"
    # ...and a path that points somewhere else entirely is drift.
    poisoned[key] = "/somewhere/else/entirely"
    changed, _, _ = _drift(sync, poisoned)
    assert key in changed, (
        f"the derived-path exemption swallowed an arbitrary change to {key!r}; it "
        "must only exempt an exact output_dir substitution."
    )


def test_removed_key_detector_is_not_vacuous():
    sync, asy = _load_pair()
    key = sorted(set(sync) & set(asy))[0]
    poisoned = {k: v for k, v in asy.items() if k != key}
    _, _, removed = _drift(sync, poisoned)
    assert key in removed

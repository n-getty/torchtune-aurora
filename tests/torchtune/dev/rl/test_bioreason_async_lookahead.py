# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe guards for BioReason async generation/training overlap (server mode).

The BioReason recipe (recipes/dev/grpo_bioreason_distributed_xpu.py) gains an
opt-in async path: a rank-0 RolloutProducer thread overlaps the pure vLLM HTTP
`generate_from_embeds` round-trip with the previous step's training, while the
collective prompt_embeds build stays on the main thread. These tests pin down the
SAFETY CONTRACT without needing XPU / vLLM / distributed init:

  * (a) async refuses to engage in non-server vLLM modes (dedicated_rank / colocate
        are world collectives that cannot overlap);
  * (b) async_generation.max_staleness > 1 raises (rollout logprobs are recomputed
        on current weights -> biased IS ratios; only k=1 is correct);
  * (c) the new async YAML satisfies the loss-combo contract (GRPOLoss +
        always_compute_rollout_logprobs) AND the sync YAML is left untouched;
  * (d) producer/consumer weight-version tagging at staleness=1 over the
        HTTP-mailbox producer pattern the recipe uses (one-step lag);
  * (e) the sync path is unchanged when async is disabled — the generate_trajectory
        server branch only consumes a pending async rollout when one is stashed,
        and the inline embeds path is reached otherwise.

The recipe module itself imports the base recipe (which runs XPU shim setup at
import time and is not importable on a login node), so the guard SEMANTICS are
verified against the recipe SOURCE (string/AST, same approach as
test_bioreason_path_discovery.py) plus a faithful re-implementation of the guard
gate, and the producer behaviour is verified against the real RolloutProducer.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from torchtune.dev.rl.async_rollout import RolloutProducer, WeightVersionTracker

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
_PROD = _REPO / "recipes" / "configs" / "dev" / "production"
_ASYNC_YAML = _PROD / "bioreason_4b_lora_grpo_2node_server_xpu_async.yaml"
_SYNC_YAML = _PROD / "bioreason_4b_lora_grpo_2node_server_xpu.yaml"


# ---------------------------------------------------------------------------
# Faithful re-implementation of the recipe's async config gate (setup()).
# Mirrors grpo_bioreason_distributed_xpu.py: must stay in sync (the source
# checks below assert the real recipe still contains the same guards).
# ---------------------------------------------------------------------------
def _apply_async_gate(async_cfg, vllm_mode, dp_replicate=1):
    enabled = bool(async_cfg.get("enabled", False))
    max_staleness = int(async_cfg.get("max_staleness", 1))
    # NOTE: there is NO LONGER a dp_replicate>1 force-disable. Async is HSDP-aware:
    # each replica's shard-leader runs its own producer + node-local broadcast.
    if enabled and vllm_mode != "server":
        enabled = False  # only server mode is async-overlappable
    if enabled and max_staleness > 1:
        raise ValueError("async_generation.max_staleness>1 is not safe yet")
    return enabled


# ---------------------------------------------------------------------------
# (a) async refuses non-server modes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["dedicated_rank", "colocate", "colocate_sleep"])
def test_async_disabled_in_non_server_mode(mode):
    enabled = _apply_async_gate({"enabled": True, "max_staleness": 1}, vllm_mode=mode)
    assert enabled is False, f"async must NOT engage in vllm_mode={mode}"


def test_async_engages_in_server_mode():
    enabled = _apply_async_gate({"enabled": True, "max_staleness": 1}, vllm_mode="server")
    assert enabled is True


def test_async_engages_under_hsdp():
    # Async is now HSDP-aware (per-shard-leader producer + node-local broadcast).
    # It must ENGAGE under dp_replicate>1, not force-disable.
    enabled = _apply_async_gate(
        {"enabled": True, "max_staleness": 1}, vllm_mode="server", dp_replicate=3
    )
    assert enabled is True, "async must ENGAGE under HSDP (per-replica shard-leader gen)"


# ---------------------------------------------------------------------------
# (b) staleness > 1 raises
# ---------------------------------------------------------------------------
def test_staleness_gt_1_raises():
    with pytest.raises(ValueError, match="max_staleness>1"):
        _apply_async_gate({"enabled": True, "max_staleness": 2}, vllm_mode="server")


def test_staleness_1_ok():
    assert _apply_async_gate({"enabled": True, "max_staleness": 1}, vllm_mode="server")


def test_disabled_with_high_staleness_does_not_raise():
    # If async is off, staleness is irrelevant and must not raise.
    assert _apply_async_gate({"enabled": False, "max_staleness": 4}, vllm_mode="server") is False


# ---------------------------------------------------------------------------
# (c) the new async YAML satisfies the loss-combo contract; sync YAML untouched.
# (test_async_loss_combo.py already parametrizes over ALL prod YAMLs; these are
#  targeted assertions so a regression names BioReason specifically.)
# ---------------------------------------------------------------------------
def test_async_yaml_exists():
    assert _ASYNC_YAML.exists(), f"missing {_ASYNC_YAML}"


def test_async_yaml_satisfies_loss_combo():
    cfg = OmegaConf.load(str(_ASYNC_YAML))
    assert bool(cfg.async_generation.enabled) is True
    assert int(cfg.async_generation.max_staleness) == 1
    assert cfg.loss._component_ == "torchtune.dev.rl.loss.GRPOLoss", (
        "async requires GRPOLoss (GRPOSimpleLoss collapses IS ratio to 1.0)"
    )
    assert bool(cfg.get("always_compute_rollout_logprobs", False)) is True
    assert cfg.vllm_mode == "server"
    assert cfg.model_type == "bioreason"


def test_sync_yaml_unchanged_no_async_overlap():
    # The validated sync config must NOT have async enabled and must keep
    # always_compute_rollout_logprobs falsey (or absent) — otherwise it would
    # silently change the production path and fail the global loss-combo test.
    cfg = OmegaConf.load(str(_SYNC_YAML))
    async_cfg = cfg.get("async_generation", None)
    enabled = bool(async_cfg.get("enabled", False)) if async_cfg is not None else False
    assert enabled is False, "sync config must keep async disabled"
    assert not bool(cfg.get("always_compute_rollout_logprobs", False)), (
        "sync config must not force rollout-logprob fwd (wasted work + loss-combo)"
    )
    assert cfg.loss._component_ == "torchtune.dev.rl.loss.GRPOSimpleLoss"


# ---------------------------------------------------------------------------
# (d) producer/consumer version tagging at staleness=1 over the HTTP-mailbox
#     producer pattern the recipe uses. The recipe's producer does NOT iterate
#     the dataloader; the MAIN thread fills a bounded mailbox with pre-built CPU
#     work, the producer transforms mailbox item -> HTTP result, and the rollout
#     is tagged with the weight version active at HTTP dispatch.
# ---------------------------------------------------------------------------
def test_mailbox_producer_one_step_lag_version_tagging():
    from queue import Queue

    versions = WeightVersionTracker()
    inbox: Queue = Queue(maxsize=1)

    def batch_iter_fn():
        return inbox.get()  # None sentinel -> exhaustion

    def produce_fn(work):
        # Pure transform: echo the work id as the "query_responses".
        return {"qr": work["id"]}, {}

    p = RolloutProducer(
        produce_fn=produce_fn,
        batch_iter_fn=batch_iter_fn,
        weight_versions=versions,
        max_staleness=1,
        warmup=False,
    )
    p.start()
    try:
        # Simulate the recipe loop: post batch i, then (after a wsync bump)
        # consume batch i-1. Producer tags each rollout with the version live
        # when it pulled the work item.
        consumed = []
        # Step 0: post 0 (version 0), no consume yet.
        inbox.put({"id": 0})
        # Step 1: post 1; consume 0 (was produced under v0); then wsync bump -> v1.
        inbox.put({"id": 1})
        item0 = p.get(timeout=5.0)
        consumed.append(item0)
        versions.bump()  # v0 -> v1 (mirrors end-of-step wsync)
        # Step 2: post 2; consume 1 (produced under whatever version was live
        # when the producer pulled id=1, i.e. v0 since it pulled before the bump).
        inbox.put({"id": 2})
        item1 = p.get(timeout=5.0)
        consumed.append(item1)
        versions.bump()  # v1 -> v2
        # Drain.
        item2 = p.get(timeout=5.0)
        consumed.append(item2)
        inbox.put(None)  # exhaustion sentinel
        assert p.get(timeout=5.0) is None

        ids = [it.batch_meta["rollout_payload"]["qr"] for it in consumed]
        assert ids == [0, 1, 2], "rollouts must be consumed in FIFO order"
        # Every rollout was generated under a version <= the version live when it
        # is consumed: staleness is bounded (>= 0) and never negative.
        # item0 produced under v0, consumed before any bump (lag 0 at consume).
        assert all(it.weight_version >= 0 for it in consumed)
        # The producer tagged at dispatch, so versions are monotonic non-decreasing.
        tagged = [it.weight_version for it in consumed]
        assert tagged == sorted(tagged), f"version tags must be monotonic, got {tagged}"
    finally:
        p.stop()


def test_mailbox_producer_clean_exhaustion_via_sentinel():
    from queue import Queue

    inbox: Queue = Queue(maxsize=1)

    def batch_iter_fn():
        return inbox.get()

    p = RolloutProducer(
        produce_fn=lambda w: ({"qr": w}, {}),
        batch_iter_fn=batch_iter_fn,
        weight_versions=WeightVersionTracker(),
        max_staleness=1,
        warmup=False,
    )
    p.start()
    try:
        inbox.put(None)  # immediate end-of-data
        assert p.get(timeout=5.0) is None
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# (e) sync path unchanged when async disabled — source-level structural guards.
# ---------------------------------------------------------------------------
def _recipe_src():
    return _RECIPE.read_text()


def test_recipe_has_async_gate_guards():
    src = _recipe_src()
    # The setup() gate must keep the staleness>1 and server-mode guards.
    assert "max_staleness>1 is not safe yet" in src, "staleness>1 guard missing"
    assert 'self._vllm_mode != "server"' in src, "server-mode guard missing"


def test_recipe_no_longer_force_disables_async_under_hsdp():
    """The old dp_replicate>1 force-disable ('Running synchronously') must be GONE —
    async is now HSDP-aware via per-shard-leader producers."""
    src = _recipe_src()
    assert "Running synchronously" not in src, (
        "the HSDP force-disable must be removed (async is now per-replica)"
    )


def test_async_impl_gates_on_shard_leader_not_rank_zero():
    """The async lookahead impl must drive the producer / consume off the replica
    SHARD LEADER (_is_shard_leader), not global rank 0, so each HSDP replica
    generates its own slice. Single-replica: _is_shard_leader == rank 0."""
    src = _recipe_src()
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_async_lookahead_iter_impl":
            fn = node
            break
    assert fn is not None, "_async_lookahead_iter_impl not found"
    lines = src.splitlines()
    code = "\n".join(lines[fn.lineno - 1 : fn.end_lineno])
    assert "_is_shard_leader" in code, (
        "async impl must gate generation on _is_shard_leader for HSDP per-replica gen"
    )


def test_recipe_pending_async_checked_before_inline_embeds():
    """In generate_trajectory's server branch, the pending-async rollout MUST be
    consumed BEFORE the inline `_is_bioreason and prompt_embeds` path, otherwise
    async generation is dead code (the embeds path would always re-run the HTTP).
    """
    src = _recipe_src()
    i_pending = src.find('_async_consume = getattr(self, "_async_consume_active", False)')
    i_inline = src.find('elif getattr(self, "_is_bioreason", False) and prompt_embeds is not None')
    assert i_pending != -1, "async-consume gate not found in generate_trajectory"
    assert i_inline != -1, "inline embeds branch not found"
    assert i_pending < i_inline, (
        "async-consume branch must be checked BEFORE the inline embeds branch"
    )


def test_producer_path_is_xpu_and_collective_free():
    """The producer-thread entrypoint (_http_generate_from_embeds_cpu) must not
    touch the XPU device or any distributed collective — the load-bearing
    thread-safety property (no concurrent XCCL from a side thread; no empty_cache).
    """
    src = _recipe_src()
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_http_generate_from_embeds_cpu":
            fn = node
            break
    assert fn is not None, "_http_generate_from_embeds_cpu not found"
    # Reconstruct the function's source lines, skipping the docstring (which
    # legitimately MENTIONS broadcasting/the device as the caller's job).
    lines = src.splitlines()
    body_stmts = fn.body
    first_exec = body_stmts[0]
    if (
        isinstance(first_exec, ast.Expr)
        and isinstance(getattr(first_exec, "value", None), ast.Constant)
        and isinstance(first_exec.value.value, str)
        and len(body_stmts) > 1
    ):
        start = body_stmts[1].lineno - 1
    else:
        start = first_exec.lineno - 1
    end = fn.end_lineno
    code = "\n".join(lines[start:end])
    # No XPU device tensor placement, no FSDP gather, no distributed collective
    # CALL in the producer-thread path. (Read-only ints like self.rank /
    # self._dp_shard are fine — they touch no device and no communicator.)
    assert "self._device" not in code, (
        "_http_generate_from_embeds_cpu must stay on CPU (no self._device) — it "
        "runs in the producer thread"
    )
    assert "torch.distributed" not in code, (
        "producer HTTP path must not touch torch.distributed (no collective)"
    )
    assert "summon_full_params" not in code, "producer path must not run the FSDP gather"
    assert "empty_cache" not in code, "producer path must not call empty_cache"


def test_async_impl_hook_present_and_delegated():
    """The base _async_lookahead_iter must delegate to a recipe-provided
    _async_lookahead_iter_impl when present, and BioReason must define it.
    """
    base = (_REPO / "recipes" / "dev" / "grpo_full_finetune_distributed_xpu.py").read_text()
    assert "_async_lookahead_iter_impl" in base, (
        "base iter must delegate to the subclass hook"
    )
    assert "def _async_lookahead_iter_impl" in _recipe_src(), (
        "BioReason must define _async_lookahead_iter_impl"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

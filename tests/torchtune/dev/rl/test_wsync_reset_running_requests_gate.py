# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Source-level guard for the delta_tp publish's ``reset_running_requests`` gate.

BACKGROUND
----------
The ``delta_tp`` LoRA publish ends by calling
``reset_prefix_cache(reset_running_requests=...)`` on every vLLM client.

Under SYNCHRONOUS generation, nothing is in flight at publish time, so
``reset_running_requests=True`` is free and simply guarantees no sequence keeps
decoding across a weight change.

Under ASYNC generation it is not free. The producer posts step N+1's rollout
before the consumer finishes step N, and the publish fires at the END of step N,
so the producer's request is mid-decode when the reset lands. In vLLM 0.15.0
(verified in the INSTALLED tree at /opt/aurora/..., not the vendored copy), that
preempts every running request: KV freed, ``num_computed_tokens`` zeroed, request
prepended to the waiting queue. Emitted tokens are preserved and the request
auto-resumes, so the cost is a full re-prefill of prompt+emitted rather than a
lost completion — but at prompt~4096 with n=8 that re-prefill is exactly the work
async exists to hide, and the resumed tokens then continue under weights their
prefix was not sampled from.

``TORCHTUNE_WSYNC_RESET_RUNNING_REQUESTS=0`` keeps the prefix-cache invalidation
(the part that matters for correctness across a weight change) while letting an
in-flight rollout finish under the weights it started with.

WHY A SOURCE-LEVEL TEST: exercising the real publish needs XPU, a live vLLM
pool, and distributed init. These assertions are cheap, run on a login node, and
pin the two properties that would silently regress — the env gate existing at
all, and its default.
"""
from pathlib import Path

import pytest

WSYNC = (
    Path(__file__).resolve().parents[4]
    / "torchtune"
    / "dev"
    / "rl"
    / "weight_sync.py"
)

FLAG = "TORCHTUNE_WSYNC_RESET_RUNNING_REQUESTS"


@pytest.fixture(scope="module")
def source() -> str:
    return WSYNC.read_text()


def test_flag_is_wired(source):
    """A ghost flag (documented/exported but absent from code) is silently inert."""
    assert FLAG in source, (
        f"{FLAG} is not referenced in weight_sync.py. The async dispatch exports "
        "it; if the code never reads it the knob is a no-op and the obstacle-3 "
        "A/B measures nothing."
    )


def test_default_preserves_legacy_sync_behavior(source):
    """Default MUST be '1' so the synchronous production path is unchanged.

    The live 100-step campaign runs without this variable set. If the default
    flipped to 0, every sync run would silently stop resetting running requests
    — a behavior change to a validated production path made by a knob added for
    an unrelated experiment.
    """
    assert f'os.environ.get("{FLAG}", "1")' in source, (
        f"{FLAG} must default to '1' (legacy behavior). Found a different or "
        "missing default."
    )


def test_reset_prefix_cache_still_called_unconditionally(source):
    """The prefix-cache reset itself must NOT become conditional on the flag.

    Only the ``reset_running_requests`` ARGUMENT is gated. Skipping
    ``reset_prefix_cache`` entirely would let vLLM serve blocks cached under the
    previous adapter after a weight change — a real correctness bug, and an easy
    mistake to make when "optimizing" this path later.

    SCOPED TO THE delta_tp PUBLISH METHOD ONLY. weight_sync.py has several other
    ``reset_prefix_cache`` calls that are deliberately NOT part of this gate:
    in-process ``llm_engine.reset_prefix_cache()`` on the colocate /
    dedicated-rank paths, and bare ``client.reset_prefix_cache()`` on the XCCL
    streaming, shm, and WS10 paths. Those run under synchronous generation where
    nothing is in flight, so the argument is moot. Two earlier versions of this
    test asserted too broadly (all calls in the file, then all ``client.`` calls)
    and failed on those — the assertions were wrong, not the code.
    """
    method = source.split("def _publish_bioreason_lora_delta", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "client.reset_prefix_cache(" in method, (
        "the delta_tp publish no longer invalidates the prefix cache — vLLM can "
        "then serve blocks cached under the previous adapter after a weight "
        "change."
    )
    idx = method.index("client.reset_prefix_cache(")
    call_text = method[idx : method.index(")", idx) + 1]
    assert "reset_running_requests" in call_text, (
        "the delta_tp publish's reset_prefix_cache lost its explicit "
        "reset_running_requests kwarg; relying on the client default hides the "
        "async trade-off."
    )


def test_gate_is_a_plain_env_read_not_a_config_field(source):
    """Keep it an env read.

    It must be settable per-dispatch without editing the shared, untracked
    launcher or a production YAML that a live job is reading — editing either
    mid-campaign is a documented drift hazard in this repo.
    """
    idx = source.index(FLAG)
    window = source[max(0, idx - 200) : idx + 200]
    assert "os.environ.get" in window

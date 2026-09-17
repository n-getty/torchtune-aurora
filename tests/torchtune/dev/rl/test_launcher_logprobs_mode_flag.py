# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pin ``--logprobs-mode processed_logprobs`` on the launcher's api_server spawn.

BioReason GRPO does **not** serve from an in-process ``LLM(...)`` object. It spawns
``vllm.entrypoints.openai.api_server`` from the shell launcher, so the five
``_logprobs_engine_kwargs`` call sites in ``vllm_backend.py`` — and the CPU test that
asserts they all carry the mode — say nothing about what production actually runs.
Before this flag landed, every production engine ran vLLM's default ``raw_logprobs``
while the trainer computed ``log_softmax(logits / T)``, a systematic ``pi_old``
mismatch (worst IS-ratio error 4.09x at V=151936, T=0.8) rather than drift.

``api_server`` exposes only ``/load`` and ``/version``; neither reports engine config,
so **there is no runtime check that can catch a regression here**. A static test on
the spawn line is the only available guard.

The flag must be written literally, not as ``${VAR:-processed_logprobs}``: this
launcher is shared across every BioReason arm, so a defaulted variable is a
fleet-wide flip that some other caller can silently override.

See memory ``project_vllm_logprobs_mode_http_servers_miss_the_helper_20260917``.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "experiments" / "bioreason" / "run_bioreason_32b_Nnode_hsdp.sh"

# The exact literal the spawn line must carry.
REQUIRED_FLAG = "--logprobs-mode processed_logprobs"

# Valid values per vLLM 0.15.0 `LogprobsMode` (config/model.py). Only the
# "processed" logprobs variant matches the trainer's log_softmax(logits / T).
_WRONG_MODES = ("raw_logits", "raw_logprobs", "processed_logits")


def _launcher_text() -> str:
    if not LAUNCHER.exists():
        pytest.skip(f"launcher not present: {LAUNCHER}")
    return LAUNCHER.read_text()


def _spawn_line_numbers(text: str) -> list[int]:
    """Line numbers of api_server *spawn* lines (not pkill cleanup lines)."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if "vllm.entrypoints.openai.api_server" not in line:
            continue
        if "pkill" in line:  # teardown, not a spawn
            continue
        out.append(i)
    return out


def test_launcher_has_exactly_one_api_server_spawn():
    """If a second spawn site appears, the flag must be added there too.

    This test exists so that adding an engine spawn cannot silently ship an
    unflagged server: the count assertion fails and forces the author here.
    """
    text = _launcher_text()
    spawns = _spawn_line_numbers(text)
    assert len(spawns) == 1, (
        f"expected exactly 1 api_server spawn in {LAUNCHER.name}, found "
        f"{len(spawns)} at lines {spawns}. Every spawn must carry "
        f"'{REQUIRED_FLAG}' — update this test and all spawn sites together."
    )


def test_spawn_carries_processed_logprobs():
    text = _launcher_text()
    assert REQUIRED_FLAG in text, (
        f"{LAUNCHER.name} must pass '{REQUIRED_FLAG}' to api_server. Without it "
        "vLLM serves the default raw_logprobs (unscaled by temperature) while the "
        "trainer computes log_softmax(logits / T), so behavior logprobs used as "
        "pi_old are systematically wrong."
    )


def test_flag_is_pinned_not_a_shell_default():
    """No ``${VAR:-...}`` / ``$VAR`` form — this launcher is shared across arms."""
    text = _launcher_text()
    for m in re.finditer(r"--logprobs-mode[ =]+(\S+)", text):
        value = m.group(1)
        assert "$" not in value, (
            f"--logprobs-mode must be pinned literally, got {value!r}. A shell "
            "variable here is a fleet-wide flip: this launcher is shared by every "
            "BioReason arm and a caller could override it without review."
        )


def test_no_wrong_logprobs_mode_anywhere():
    text = _launcher_text()
    for bad in _WRONG_MODES:
        assert f"--logprobs-mode {bad}" not in text, (
            f"{LAUNCHER.name} passes --logprobs-mode {bad}; only "
            "'processed_logprobs' matches the trainer's log_softmax(logits / T)."
        )


def test_flag_is_inside_the_spawn_command():
    """Guard against the flag being added as a stray line outside the invocation.

    The spawn is a multi-line backslash continuation; a flag that lands after the
    redirect (or outside the block) parses as a no-op or a syntax error at job
    start on the compute node, where it is expensive to discover.
    """
    text = _launcher_text()
    lines = text.splitlines()
    spawns = _spawn_line_numbers(text)
    assert spawns, "no api_server spawn found"
    start = spawns[0] - 1

    # Walk the backslash-continued command to its final line.
    end = start
    while end < len(lines) and lines[end].rstrip().endswith("\\"):
        end += 1

    block = "\n".join(lines[start : end + 1])
    assert REQUIRED_FLAG in block, (
        f"'{REQUIRED_FLAG}' is present in {LAUNCHER.name} but not within the "
        f"api_server invocation (lines {start + 1}-{end + 1}); it would not reach "
        "the server process."
    )

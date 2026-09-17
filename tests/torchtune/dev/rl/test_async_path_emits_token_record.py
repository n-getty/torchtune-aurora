# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""The ASYNC generation path must emit the same token record as the sync path.

WHY. `_http_generate_from_embeds_cpu` is the generation path used when
`async_generation.enabled=true`: it runs on the RolloutProducer thread and, until
2026-09-17, logged NOTHING. Consequences, both real:

  1. `experiments/bioreason/rolog_token_totals.sh` gates on
     "vLLM engine=N request done sequences=N output_tokens=N" and so exited 3
     BLIND on every async log -- correctly refusing to guess, but leaving the
     async-vs-sync A/B with exactly ONE token estimator (BIOREASON_DIAG
     len_mean*n) and no independent cross-check. A single estimator with no
     cross-check is what produced the rolog arity burn
     (memory/feedback_a_tool_built_to_fix_a_bug_can_be_the_bug).
  2. Any per-request throughput reading (tok/s, output_length_max, elapsed) was
     simply unavailable on the async arm, so "async is faster" could only ever be
     argued from step totals.

An absent log line and an inert one look identical downstream, which is why this
is pinned by a test rather than by the comment at the call site.

WHAT IS PINNED: the async site emits a record whose GRAMMAR matches the sync
site's, so one parser serves both arms. The test asserts against the same regex
`rolog_token_totals.sh` uses -- if someone reworded either site, this fails.
"""
import ast
import re
import subprocess
from pathlib import Path

import pytest

RECIPE = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "dev"
    / "grpo_bioreason_distributed_xpu.py"
)
ROLOG = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "bioreason"
    / "rolog_token_totals.sh"
)

# The grammar rolog_token_totals.sh gates on. Kept as a literal here on purpose:
# importing it from the shell script would make both sides move together and the
# test would pass through a rename that breaks every existing log.
ROLOG_GATE = re.compile(
    r"vLLM engine=[0-9]+ request done sequences=[0-9]+ output_tokens=[0-9]+"
)


def _source():
    if not RECIPE.exists():
        pytest.skip(f"{RECIPE.name} not present")
    return RECIPE.read_text()


def _fmt_strings(src):
    """Every string literal in the file that looks like the token record."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "request done sequences=" in node.value:
                out.append(node.value)
        # Implicit concatenation arrives as a single Constant after parsing, so
        # the walk above already sees the joined string.
    return out


def _async_fn_source(src):
    """Source of _http_generate_from_embeds_cpu, the async-only generation path."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_http_generate_from_embeds_cpu"
        ):
            return ast.get_source_segment(src, node)
    return None


def test_async_generation_path_exists():
    """Guard the guard: if the function is renamed, the tests below would pass
    vacuously by finding nothing to check."""
    assert _async_fn_source(_source()) is not None, (
        "_http_generate_from_embeds_cpu not found -- the async generation path "
        "was renamed or removed; update this test rather than deleting it."
    )


def test_async_path_emits_a_token_record():
    fn_src = _async_fn_source(_source())
    assert "request done sequences=" in fn_src, (
        "the ASYNC generation path emits no 'request done' token record. "
        "rolog_token_totals.sh will exit 3 BLIND on every async log, leaving the "
        "async-vs-sync A/B with a single unchecked token estimator."
    )
    assert "output_tokens=" in fn_src


def test_async_record_grammar_matches_the_rolog_gate():
    """The emitted line must match the regex the parser gates on. A record that
    is present but worded differently is worse than absent: it looks fixed."""
    fn_src = _async_fn_source(_source())
    fmts = [s for s in _fmt_strings(fn_src) if "request done sequences=" in s]
    assert fmts, "no token-record format string inside the async path"
    for fmt in fmts:
        # Render the %-format with plausible values, as logging would.
        rendered = fmt.replace("%d", "123").replace("%.1f", "12.3").replace("%s", "x")
        assert ROLOG_GATE.search(rendered), (
            f"async token record does not match the rolog gate.\n"
            f"  rendered: {rendered!r}\n"
            f"  gate    : {ROLOG_GATE.pattern}\n"
            "One parser must serve both arms; reword both sites or neither."
        )


def test_both_sync_and_async_sites_use_the_same_grammar():
    """There are two _call_group closures (sync fanout + async HTTP). Both must
    emit the record, in the same grammar, or a cross-arm token comparison is
    comparing two different things."""
    src = _source()
    fmts = [s for s in _fmt_strings(src) if "output_tokens=" in s]
    assert len(fmts) >= 2, (
        f"expected a token record at BOTH the sync and async generation sites, "
        f"found {len(fmts)}. If a site was intentionally dropped, the A/B loses "
        "its independent token estimator on that arm."
    )
    normalized = {re.sub(r"\s+", " ", f).strip() for f in fmts}
    assert len(normalized) == 1, (
        "the sync and async token records use DIFFERENT grammars:\n  "
        + "\n  ".join(repr(n) for n in sorted(normalized))
        + "\nrolog_token_totals.sh parses one pattern; divergence makes one arm "
        "silently BLIND while the other reports."
    )


def test_rolog_groups_async_on_the_producer_marker_not_diag():
    """Emitting the record was only half the fix.

    Under async the record is emitted by the RolloutProducer THREAD, which runs a
    step ahead of training, so its records interleave with the consumer's output
    and a generation's records can land AFTER the BIOREASON_DIAG line of the step
    being trained -- the margin in the n=1 async log was 3 seconds. Grouping on
    DIAG therefore attributes records to the wrong generation. The tool must group
    async logs on `RolloutProducer: produced #N`, which the producer emits for
    itself right after the generation it describes (group on a delimiter the
    PRODUCER emits -- memory/feedback_a_tool_built_to_fix_a_bug_can_be_the_bug).
    """
    if not ROLOG.exists():
        pytest.skip("rolog_token_totals.sh not present")
    text = ROLOG.read_text()
    assert "RolloutProducer: produced #" in text, (
        "rolog_token_totals.sh does not group async logs on the producer marker; "
        "records will be attributed to the wrong generation."
    )
    assert 'mode == "async" && /BIOREASON_DIAG' in text, (
        "async mode must SKIP the consumer's DIAG line as a delimiter, or the "
        "producer's records get split across generations."
    )
    assert "NO CROSS-CHECK" in text, (
        "async rows have no len_mean and so no independent estimate; the tool "
        "must say so on the row rather than print them as if verified."
    )


def test_rolog_selftest_passes_including_the_async_cases():
    """Calibrate the instrument on constructed answers, both modes."""
    if not ROLOG.exists():
        pytest.skip("rolog_token_totals.sh not present")
    r = subprocess.run(
        ["bash", str(ROLOG), "--selftest"], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, f"rolog selftest failed:\n{r.stdout}\n{r.stderr}"
    assert "selftest: PASS" in r.stdout
    # Pin the async cases specifically so the selftest cannot later be thinned
    # back to the sync-only set that let this bug through.
    for case in (
        "async fixture detected as async",
        "async groups on the producer marker, not DIAG",
        "a DIAG line does not split an async generation",
        "async emits no DIAG-grouped step= rows",
        # Observed live: records present, no delimiter yet -> RUN TOTAL 0 at
        # exit 0. A zero that a caller can script on must read BLIND.
        "all-in-flight reads BLIND, not a zero total",
    ):
        assert case in r.stdout, f"the rolog selftest no longer covers: {case}"


def test_rolog_gate_literal_still_matches_the_script():
    """If rolog_token_totals.sh changes its gate, the literal above goes stale and
    every assertion here would be checking a pattern nothing uses."""
    if not ROLOG.exists():
        pytest.skip("rolog_token_totals.sh not present")
    text = ROLOG.read_text()
    assert "request done sequences=" in text and "output_tokens=" in text, (
        "rolog_token_totals.sh no longer gates on the 'request done' record; the "
        "ROLOG_GATE literal in this test is stale."
    )

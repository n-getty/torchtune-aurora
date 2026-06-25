# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe doc<->code consistency guard for TORCHTUNE_* feature flags.

CLAUDE.md documents an opt-in env-var table that every session loads as the
contract for the codebase. Flags drift two ways:

  * GHOST: a flag is documented but exists in NO code file. This is the same
    failure class as the orphaned IS-correction tests — the doc tells a future
    session to set a flag that was never wired (e.g. TORCHTUNE_XPU_SDPA_UPCAST,
    found in MEMORY.md referencing 0 code files). This test FAILS on ghosts.

  * UNDOCUMENTED: a flag is wired in code but absent from CLAUDE.md. Less
    dangerous (it just won't be discovered), so this is a NON-FAILING report
    printed for awareness, with the debug-only TORCHTUNE_SKIP_*/MEMPROBE family
    excluded.

Implementation: source-text scan. No torch, no XPU. Uses pathlib.rglob (NOT
stdlib glob.glob, which hangs on DAOS/dfuse per CLAUDE.md) over the package
source — fine on a login node.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
CODE_DIRS = [REPO_ROOT / "torchtune", REPO_ROOT / "recipes"]

FLAG_RE = re.compile(r"TORCHTUNE_[A-Z0-9_]+")

# The "supported, documented" set is the flags that appear as the FIRST cell of
# a CLAUDE.md table row: `| `TORCHTUNE_X=...` | ...`. We deliberately parse only
# table rows (not prose) so explanatory mentions of prefixes/families elsewhere
# (e.g. "the `TORCHTUNE_SKIP_*` family") are not mistaken for real flags. The
# trailing `=` or backtick after the name distinguishes a concrete flag from a
# `TORCHTUNE_PREFIX_*` glob.
DOC_ROW_FLAG_RE = re.compile(r"^\|\s*`(TORCHTUNE_[A-Z0-9_]+)(?:=[^`]*)?`", re.MULTILINE)

# Debug-only / diagnostic flags that intentionally stay out of the supported
# table — do not report these as "undocumented".
UNDOCUMENTED_OK_PREFIXES = ("TORCHTUNE_SKIP_",)
UNDOCUMENTED_OK_EXACT = {
    "TORCHTUNE_ASYM_MEMPROBE",
    "TORCHTUNE_COLOCATE_MEM_PROBE",  # diagnostic: per-step free-HBM leak probe in colocate LoRA sync
    "TORCHTUNE_LEAK_CENSUS",  # diagnostic: per-step live-XPU-tensor census (names the server step-6 leak)
    "TORCHTUNE_EP_DEBUG",  # documented, but guard anyway
}


def _documented_flags() -> set[str]:
    text = CLAUDE_MD.read_text()
    return set(DOC_ROW_FLAG_RE.findall(text))


def _code_flags() -> set[str]:
    found: set[str] = set()
    for d in CODE_DIRS:
        for py in d.rglob("*.py"):
            try:
                found.update(FLAG_RE.findall(py.read_text()))
            except (UnicodeDecodeError, OSError):
                continue
    return found


def test_documented_flags_are_wired_in_code():
    """Every TORCHTUNE_* flag in the CLAUDE.md table must exist in code."""
    documented = _documented_flags()
    assert documented, "No documented TORCHTUNE_* flags parsed from CLAUDE.md — regex broke?"
    code = _code_flags()
    ghosts = sorted(documented - code)
    assert not ghosts, (
        "CLAUDE.md documents flags that exist in NO code file under torchtune/ "
        f"or recipes/ (ghost flags): {ghosts}. Either wire them or remove them "
        "from the doc — a documented-but-absent flag misleads every future "
        "session (see the TORCHTUNE_XPU_SDPA_UPCAST / IS-correction precedent)."
    )


def test_report_undocumented_flags():
    """Non-failing: surface code flags missing from CLAUDE.md for awareness."""
    documented = _documented_flags()
    code = _code_flags()
    undocumented = sorted(
        f
        for f in (code - documented)
        if not f.startswith(UNDOCUMENTED_OK_PREFIXES)
        and f not in UNDOCUMENTED_OK_EXACT
    )
    if undocumented:
        print(
            "\n[info] TORCHTUNE_* flags wired in code but not in the CLAUDE.md "
            "table (consider documenting):\n  " + "\n  ".join(undocumented)
        )
    # Intentionally always passes — this is a report, not a gate.
    assert True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))

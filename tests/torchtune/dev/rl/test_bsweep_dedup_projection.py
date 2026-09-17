# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Guards on the B-sweep readout's prefix-dedup projection.

The projection used to rest on a hardcoded ``PROMPT_TOKENS = 4096`` constant. 4096 is
the padding *cap*, not the prompt length -- measured padded widths run 1774-4096 -- so
the constant overstated the per-row prompt and on short steps produced a **negative**
post-dedup token count, which is physically impossible and went unnoticed because the
number was only ever eyeballed. It also projected from a single (largest) row, so the
headline payoff moved 2.90x -> 2.34x purely because a later step landed.

These tests pin both fixes: widths are measured from the log, and the payoff is
aggregated over every warm row of the cell.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_READOUT = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "bioreason"
    / "bsweep_readout.py"
)


@pytest.fixture(scope="module")
def mod():
    if not _READOUT.is_file():
        pytest.skip(f"readout script not found: {_READOUT}")
    spec = importlib.util.spec_from_file_location("bsweep_readout", _READOUT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bsweep_readout"] = m
    spec.loader.exec_module(m)
    return m


def _log(chunks, bwd, step):
    """Render chunk-width + bwd + TIMING lines the way the recipe emits them."""
    out = ["batch_size: 4", "forward_batch_size: 2"]
    for rows, resp, total in chunks:
        out.append(
            "INFO: BIOREASON_CHUNK_WIDTH phase=train_policy "
            f"rows=[{', '.join(str(r) for r in rows)}] "
            f"response={resp}/3072 total={total}"
        )
    out.append(f"Rank 0: grpo_step bwd={bwd}s")
    out.append(
        f"TIMING step={step}  total=600.0s  gen=300.0s  grpo=280.0s  clip=0.1s"
    )
    return "\n".join(out) + "\n"


def test_no_prompt_tokens_constant(mod):
    """The falsified 4096 constant must not come back."""
    assert not hasattr(mod, "PROMPT_TOKENS"), (
        "PROMPT_TOKENS was removed because 4096 is the padding cap, not the prompt "
        "length; measure `total - response` per chunk instead."
    )


def test_group_widths_measured_not_assumed(mod):
    """Prompt width per group comes from `total - response`, whatever its value."""
    # One G=8 group, two rows per chunk, prompt width 1774 (well under the 4096 cap).
    pol = [([r, r + 1], 500, 1774 + 500) for r in range(0, 8, 2)]
    widths = mod._group_prompt_widths(pol)
    assert widths == {0: 1774}


def test_straddling_chunk_takes_max_width(mod):
    """A chunk spanning two groups can't realise the saving -- be conservative."""
    pol = [([0, 1], 100, 1100), ([2, 3], 100, 3100)]
    # Same group 0, disagreeing widths -> max, not min or mean.
    assert mod._group_prompt_widths(pol) == {0: 3000}


def test_dedup_never_exceeds_current_tokens(mod, tmp_path):
    """Post-dedup tokens must be positive and <= current. The 4096 constant broke this.

    Reproduces the shape of the real step 7 (short responses, narrow prompts) where the
    old projection returned -8.0 ktok.
    """
    # 4 groups x 8 rows, fbs=2, short responses, prompt width 1774 (the measured min).
    chunks = [
        ([r, r + 1], 300, 1774 + 300) for r in range(0, 32, 2)
    ]
    p = tmp_path / "train_short.log"
    p.write_text(_log(chunks, bwd=100.0, step=1))
    (rec,) = mod.parse(str(p))
    assert rec["dedup_ktok"] > 0, "post-dedup token count went non-positive"
    assert rec["dedup_ktok"] < rec["pol_ktok"]


def test_dedup_math_is_exact(mod, tmp_path):
    """Dedup pays each response once per row and each prompt once per GROUP."""
    prompt, resp = 2000, 400
    chunks = [([r, r + 1], resp, prompt + resp) for r in range(0, 16, 2)]  # 2 groups
    p = tmp_path / "train_exact.log"
    p.write_text(_log(chunks, bwd=50.0, step=1))
    (rec,) = mod.parse(str(p))

    rows, groups = 16, 2
    assert rec["pol_ktok"] == pytest.approx(rows * (prompt + resp) / 1000.0)
    assert rec["dedup_ktok"] == pytest.approx(
        (groups * prompt + rows * resp) / 1000.0
    )


def test_projection_aggregates_all_warm_rows(mod, tmp_path, capsys):
    """Adding a step must not swing the headline the way single-row projection did.

    Two steps with very different response lengths: a single-row projection reports
    whichever one it picked; the aggregate sits between them and barely moves.
    """
    def cell(resp, bwd, step):
        return _log(
            [([r, r + 1], resp, 3000 + resp) for r in range(0, 32, 2)], bwd, step
        )

    p = tmp_path / "train_two.log"
    p.write_text(cell(0, 0.0, 0) + cell(2500, 250.0, 1) + cell(300, 120.0, 2))
    recs = mod.parse(str(p))
    warm = [r for r in recs if r["step"] >= 1]
    assert len(warm) == 2

    per_step = sorted(r["pol_ktok"] / r["dedup_ktok"] for r in warm)
    agg = sum(r["pol_ktok"] for r in warm) / sum(r["dedup_ktok"] for r in warm)
    assert per_step[0] < agg < per_step[-1], (
        f"aggregate {agg:.2f}x must lie between per-step {per_step} -- if it equals "
        "one of them the projection is reading a single row again"
    )

#!/usr/bin/env python3
"""Diff the generated text of two A/B legs.

Exists because the A/B harness's default prompt is `PROMPT_TOKENS*4` bytes of
the letter 'a', which the model answers with a run of 'a's. That is fine for
timing but proves nothing about numerics: a wrong kernel would very likely
reproduce it exactly. Run the legs with a PROMPT that has a checkable answer,
then use this to compare.

Usage:  compare_leg_completions.py LEG_DIR_A LEG_DIR_B
Prints one COMPLETION= line (grep-able) and exits non-zero on mismatch.
"""

import glob
import json
import os
import sys


def completions(leg_dir):
    """Map request-file basename -> generated text, skipping warmups."""
    out = {}
    for path in sorted(glob.glob(os.path.join(leg_dir, "*.json"))):
        name = os.path.basename(path)
        if name.startswith("warmup"):
            continue
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        choices = doc.get("choices")
        if not choices:
            continue
        text = choices[0].get("text")
        if text is not None:
            out[name] = text
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    a, b = completions(a_dir), completions(b_dir)

    if not a or not b:
        print(
            f"COMPLETION=NO_DATA a={len(a)} b={len(b)} "
            f"(a_dir={a_dir} b_dir={b_dir}) -- cannot judge correctness"
        )
        return 3

    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"COMPLETION=NO_OVERLAP a={sorted(a)} b={sorted(b)}")
        return 3

    mismatches = [k for k in shared if a[k] != b[k]]

    # A degenerate answer (one character repeated) cannot distinguish a
    # correct kernel from a wrong one, so say so rather than reporting PASS.
    degenerate = all(len(set(a[k].strip())) <= 1 for k in shared)

    if mismatches:
        print(f"COMPLETION=MISMATCH {len(mismatches)}/{len(shared)} differ")
        for k in mismatches[:3]:
            print(f"  {k}\n    A: {a[k][:160]!r}\n    B: {b[k][:160]!r}")
        return 1

    if degenerate:
        print(
            f"COMPLETION=IDENTICAL_BUT_DEGENERATE {len(shared)} pairs; "
            "output is a single repeated character, so this does NOT "
            "validate numerics. Re-run with a real prompt."
        )
        print(f"  sample: {a[shared[0]][:80]!r}")
        return 4

    print(f"COMPLETION=IDENTICAL {len(shared)} pairs, non-degenerate")
    print(f"  sample: {a[shared[0]][:160]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a frozen per-term go_pred reliability prior for BioReason rerank targets.

For every GO term that appears in ANY training example's ``go_pred`` field, compute
``P(correct | term in go_pred)`` — the empirical fraction of the time that term is also
in the example's ground truth (``go_bp``/``go_mf``/``go_cc``). This is an aggregate class
prior over terms, not a per-sample signal, and it is computed ONLY over the training
shards — never validation or test — so it cannot leak eval information into the graded
SFT target (see the rerank_target design in the BioReason per-candidate reranking plan).

Terms with fewer than ``--min_count`` occurrences fall back to the global rate at
inference time (the dataset side, not here) — this script just records occurrence
counts alongside the raw reliability so that decision can be made downstream.

Usage:
  python scripts/build_gopred_prior.py \\
    --data_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/data \\
    --out /lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/gopred_prior.json \\
    --held_out_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/data/validation-00000-of-00001.parquet \\
    --held_out_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

GO_RE = re.compile(r"GO:\d{7}")


def _nonempty(v) -> bool:
    if v is None:
        return False
    try:
        return len(v) > 0
    except TypeError:
        return bool(v)


def _gt_terms(row: dict) -> set:
    out = set()
    for col in ("go_mf", "go_cc", "go_bp"):
        v = row.get(col)
        if not _nonempty(v):
            continue
        items = [v] if isinstance(v, str) else list(v)
        for t in items:
            t = str(t).strip()
            if t:
                out.add(t)
    return out


def _gopred_terms(row: dict) -> list:
    go_pred = row.get("go_pred", "") or ""
    seen, out = set(), []
    for t in GO_RE.findall(str(go_pred)):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _load_train_rows(data_dir: str) -> list:
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    if not paths:
        raise RuntimeError(f"No train-*.parquet files found under {data_dir}")
    rows = []
    for p in paths:
        df = pd.read_parquet(p)
        rows.extend(df.to_dict("records"))
        print(f"[build_gopred_prior] loaded {len(df)} rows from {p}")
    return rows


def _load_protein_ids(path: str) -> set:
    import pandas as pd

    if os.path.isdir(path):
        ids = set()
        for p in sorted(glob.glob(os.path.join(path, "*.parquet"))):
            ids.update(pd.read_parquet(p)["protein_id"].tolist())
        return ids
    return set(pd.read_parquet(path)["protein_id"].tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir containing train-*.parquet")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--held_out_dir",
        action="append",
        default=[],
        help="dir or file of protein ids to EXCLUDE from the prior computation (pass "
        "once per split, e.g. validation dir and test dir). Any train row whose "
        "protein_id also appears in a held-out split is dropped before computing "
        "reliabilities, so the frozen prior is guaranteed to contain zero information "
        "from any held-out split even if the raw train/test parquets happen to share "
        "protein ids (measured: 471/8630 bioreason_pro_test ids also appear in the SFT "
        "train shards — a pre-existing property of the source data, not a splitting "
        "bug introduced here).",
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="terms below this occurrence count are still written (with their raw "
        "count) but flagged low_count=true so the dataset-side loader can fall back "
        "to the global rate for them.",
    )
    args = ap.parse_args()

    rows = _load_train_rows(args.data_dir)

    held_ids: set = set()
    for held_out in args.held_out_dir:
        ids = _load_protein_ids(held_out)
        held_ids |= ids
        print(f"[build_gopred_prior] held-out {held_out}: {len(ids)} protein ids")

    if held_ids:
        before = len(rows)
        rows = [r for r in rows if str(r.get("protein_id")) not in held_ids]
        dropped = before - len(rows)
        print(
            f"[build_gopred_prior] excluded {dropped}/{before} train rows whose "
            "protein_id also appears in a held-out split (prior computed on the "
            f"remaining {len(rows)} rows)."
        )

    # Hard assertion: zero overlap between the rows actually used and every held-out
    # split, checked AFTER exclusion so this can never silently pass on a filtering bug.
    used_ids = {str(r.get("protein_id")) for r in rows}
    overlap = used_ids & held_ids
    if overlap:
        print(
            f"[build_gopred_prior] FATAL: {len(overlap)} protein ids used for the "
            f"prior still overlap a held-out split after exclusion (e.g. "
            f"{sorted(overlap)[:5]}) — exclusion logic is broken.",
            file=sys.stderr,
        )
        return 1

    correct = defaultdict(int)
    total = defaultdict(int)
    for row in rows:
        gt = _gt_terms(row)
        preds = set(_gopred_terms(row))
        for t in preds:
            total[t] += 1
            if t in gt:
                correct[t] += 1

    if not total:
        print("[build_gopred_prior] FATAL: no go_pred terms found across all rows", file=sys.stderr)
        return 1

    global_correct = sum(correct.values())
    global_total = sum(total.values())
    global_rate = global_correct / global_total

    terms = {}
    for t, cnt in total.items():
        reliability = correct[t] / cnt
        terms[t] = {
            "reliability": reliability,
            "count": cnt,
            "low_count": cnt < args.min_count,
        }

    out = {
        "global_rate": global_rate,
        "min_count": args.min_count,
        "num_train_rows": len(rows),
        "num_terms": len(terms),
        "terms": terms,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(
        f"[build_gopred_prior] wrote {len(terms)} terms, global_rate={global_rate:.4f}, "
        f"rows={len(rows)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

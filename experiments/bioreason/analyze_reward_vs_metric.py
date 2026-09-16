#!/usr/bin/env python3
"""Score GRPO rollouts and greedy eval generations with the *training* reward.

Why this exists
---------------
Run 8829513 showed the training reward *falling* (-0.0024/step, GT-controlled) while
F_max stayed flat-to-slightly-up. The natural hypothesis was that the reward is
misaligned with the metric: the reward is an **unweighted** propagated F1, while the
eval is an **IA-weighted** F_max, and GO propagation makes generic high-frequency
ancestors free to predict. This script tests that directly, two ways:

1. ``--mode rank``  — within-group Spearman between the training reward and an
   IA-weighted F1 on the *same* rollouts. GRPO only ever sees the within-group
   contrast, so that is the quantity that decides whether the gradient points toward
   the metric. Measured 2026-09-16: **rho = 0.95 mean / 1.00 median**. Misalignment
   refuted.

2. ``--mode bridge`` — apply the training reward to the **greedy eval generations**.
   Same 127 test proteins, same scoring code across snapshots, so it is comparable
   step-to-step and sits alongside F_max. Measured: 0.3165 / 0.3227 / 0.3268 at steps
   0/20/40 — monotone **rising**, same direction as F_max, opposite to the temp-0.8
   rollouts.

Conclusion: the reward tracks the metric fine. What differs between the two readings is
the *distribution the reward is averaged over* (temp 0.8 vs greedy). See
``memory/project_bioreason_reward_falls_at_temp08_rises_greedy_20260916.md``.

Comparability rules enforced by refusing to print cross-column deltas
---------------------------------------------------------------------
* Rollout scores are the TRAIN split; eval scores are the TEST split. Never difference
  a rollout mean against an eval mean
  (``project_bioreason_rollout_fmax_is_trainset_not_comparable_to_eval_20260915``).
* The training reward (unweighted propagated F1) is not F_max (IA-weighted, tau-swept).
  Each column is read *within itself, across steps*.

Implementation note: ``goatools`` is not installed in the ``frameworks`` module, so the
GO ``is_a`` closure is parsed straight out of ``go-basic.obo`` here. ``--self-test``
checks that closure against a few known ontology facts before any scoring runs.

Usage (login node, no allocation)::

    module load frameworks
    python3 experiments/bioreason/analyze_reward_vs_metric.py --self-test
    python3 experiments/bioreason/analyze_reward_vs_metric.py --mode rank \
        --rollouts experiments/bioreason/outputs/<run>/rollouts.jsonl
    python3 experiments/bioreason/analyze_reward_vs_metric.py --mode bridge \
        --eval-dir experiments/bioreason/eval_out/grpo_step0_rep1 \
        --eval-dir experiments/bioreason/eval_out/grpo_epoch_0_step40_snapshot_rep1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from statistics import mean

GO_RE = re.compile(r"GO:\d{7}")

DEFAULT_OBO = "/lus/flare/projects/ModCon/ngetty/BioReason-Pro/bioreason2/dataset/go-basic.obo"
DEFAULT_IA = "/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt"


# ─────────────────────────── ontology ───────────────────────────


def load_obo(path: str):
    """Parse ``is_a`` parents + ``alt_id`` aliases. Obsolete terms are dropped.

    goatools would do this, but it is absent from the frameworks env and the only
    thing needed here is the ancestor closure.
    """
    parents: dict[str, set[str]] = {}
    alt2main: dict[str, str] = {}
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:].strip()
            parents.setdefault(cur, set())
        elif cur and line.startswith("alt_id: GO:"):
            alt2main[line[8:].strip()] = cur
        elif cur and line.startswith("is_a: "):
            parents[cur].add(line[6:].split("!")[0].strip())
        elif cur and line.startswith("is_obsolete: true"):
            parents.pop(cur, None)
    return parents, alt2main


class Ontology:
    def __init__(self, parents, alt2main):
        self.parents = parents
        self.alt2main = alt2main
        self._cache: dict[str, frozenset] = {}

    def ancestors(self, term: str) -> frozenset:
        term = self.alt2main.get(term, term)
        hit = self._cache.get(term)
        if hit is not None:
            return hit
        out, stack, seen = set(), [term], set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            out.add(node)
            stack.extend(self.parents.get(node, ()))
        self._cache[term] = frozenset(out)
        return self._cache[term]

    def propagate(self, terms) -> set:
        out: set = set()
        for t in terms:
            out |= self.ancestors(t)
        return out


def load_ia(path: str) -> dict[str, float]:
    ia = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            try:
                ia[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return ia


# ─────────────────────────── scoring ───────────────────────────


def f1(pred: set, gt: set, weights: dict | None = None) -> float:
    """F1 over term sets; ``weights`` switches to information-accretion weighting.

    Mirrors ``torchtune/dev/bioreason/reward.py::weighted_f_score`` for the unweighted
    case (empty-vs-empty is not reachable here since GT is always non-empty).
    """
    if not pred or not gt:
        return 0.0
    tp = pred & gt
    if weights is None:
        precision = len(tp) / len(pred)
        recall = len(tp) / len(gt)
    else:
        sp = sum(weights.get(x, 0.0) for x in pred)
        sg = sum(weights.get(x, 0.0) for x in gt)
        st = sum(weights.get(x, 0.0) for x in tp)
        if sp <= 0 or sg <= 0:
            return 0.0
        precision, recall = st / sp, st / sg
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def spearman(xs, ys):
    """Tie-corrected Spearman. Returns None when a side is constant (rank undefined)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    a, b = ranks(xs), ranks(ys)
    ma, mb = mean(a), mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    da = math.sqrt(sum((p - ma) ** 2 for p in a))
    db = math.sqrt(sum((q - mb) ** 2 for q in b))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


# ─────────────────────────── modes ───────────────────────────


def mode_rank(onto: Ontology, ia: dict, rollouts: str) -> int:
    recs = [json.loads(l) for l in open(rollouts)]
    rhos, degenerate = [], 0
    pooled_r, pooled_w = [], []
    for rec in recs:
        gt = onto.propagate(set(GO_RE.findall(rec["answer"])))
        rs, ws = [], []
        for s in rec["samples"]:
            pred = onto.propagate(set(GO_RE.findall(s["text"])))
            rs.append(f1(pred, gt))
            ws.append(f1(pred, gt, ia))
        pooled_r += rs
        pooled_w += ws
        if len(set(rs)) < 2 or len(set(ws)) < 2:
            degenerate += 1
            continue
        rho = spearman(rs, ws)
        if rho is not None:
            rhos.append(rho)
    if not rhos:
        print("no usable groups", file=sys.stderr)
        return 1
    print(f"groups={len(recs)}  usable={len(rhos)}  degenerate={degenerate}")
    print(
        "WITHIN-GROUP spearman(training reward, IA-weighted F1): "
        f"mean={mean(rhos):.4f}  median={sorted(rhos)[len(rhos) // 2]:.4f}"
    )
    print(
        f"  frac rho<0.5: {sum(1 for r in rhos if r < 0.5) / len(rhos):.3f}"
        f"   frac rho<0: {sum(1 for r in rhos if r < 0) / len(rhos):.3f}"
    )
    print(f"pooled spearman = {spearman(pooled_r, pooled_w):.4f}")
    print(
        "\nGRPO only sees the within-group contrast. rho near 1 means the reward ranks "
        "candidates\nthe way the IA-weighted metric does, i.e. IA weighting is NOT the "
        "train/eval gap."
    )
    return 0


def _eval_gt(item: dict) -> set:
    """Ground truth lives in go_bp/go_mf/go_cc. The ``ground_truth`` key is often ''."""
    gt: set = set()
    for key in ("go_bp", "go_mf", "go_cc"):
        val = item.get(key) or []
        if isinstance(val, list):
            gt |= {x for x in val if isinstance(x, str) and GO_RE.fullmatch(x)}
    raw = item.get("ground_truth")
    if isinstance(raw, str):
        gt |= set(GO_RE.findall(raw))
    return gt


def mode_bridge(onto: Ontology, eval_dirs) -> int:
    print("Training reward (propagated F1) applied to GREEDY eval generations.")
    print("Read DOWN a column across steps; do NOT difference against rollout means")
    print("(train vs test split) or against F_max (different metric).\n")
    print(f"{'snapshot':>46} {'reward':>9} {'zero%':>7} {'GOids':>7} {'n':>5}")
    for d in eval_dirs:
        if not os.path.isdir(d):
            print(f"{os.path.basename(d):>46}  MISSING")
            continue
        vals, zero, ids = [], 0, []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            try:
                j = json.load(open(os.path.join(d, fn)))
            except Exception:
                continue
            item = j[0] if isinstance(j, list) else j
            if not isinstance(item, dict):
                continue
            gt = _eval_gt(item)
            if not gt:
                continue
            pred = set(GO_RE.findall(item.get("generated_response") or ""))
            if not pred:
                zero += 1
            ids.append(len(pred))
            vals.append(f1(onto.propagate(pred), onto.propagate(gt)))
        if not vals:
            print(f"{os.path.basename(d):>46}  no scorable generations")
            continue
        print(
            f"{os.path.basename(d):>46} {mean(vals):9.4f} {zero / len(vals):6.1%} "
            f"{mean(ids):7.1f} {len(vals):5d}"
        )
    return 0


def _score_dir_by_protein(onto: Ontology, d: str) -> dict:
    """Map protein_id -> mean training-reward score over that protein's generations.

    A directory may hold several generations per protein (the paper's released dirs
    have 330 files for 250 proteins). Averaging within protein first makes the pairing
    one-per-protein regardless of how many generations each arm emitted.
    """
    acc: dict = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            j = json.load(open(os.path.join(d, fn)))
        except Exception:
            continue
        item = j[0] if isinstance(j, list) else j
        if not isinstance(item, dict):
            continue
        pid = item.get("protein_id")
        gt = _eval_gt(item)
        if not pid or not gt:
            continue
        pred = set(GO_RE.findall(item.get("generated_response") or ""))
        acc.setdefault(pid, []).append(f1(onto.propagate(pred), onto.propagate(gt)))
    return {k: mean(v) for k, v in acc.items()}


def mode_paired(onto: Ontology, before: str, after: str) -> int:
    """Paired within-arm delta on the training reward.

    This is the statistic that calibrates our RL against the paper's. Cross-ARM
    *levels* are not comparable when the protein populations differ (ours overlaps the
    paper's released set by only 3 of 250), but a within-arm paired delta is immune to
    that: each arm is differenced against itself on an identical protein set.

    See ``memory/project_bioreason_paper_rl_gain_paired_calibration_20260916.md``.
    """
    a, b = _score_dir_by_protein(onto, before), _score_dir_by_protein(onto, after)
    common = sorted(set(a) & set(b))
    if not common:
        print("no proteins in common -- these two dirs cannot be paired")
        return 1
    deltas = [b[p] - a[p] for p in common]
    n = len(deltas)
    m = mean(deltas)
    if n < 2:
        print(f"n={n}: too few pairs for a t statistic")
        return 1
    sd = math.sqrt(sum((x - m) ** 2 for x in deltas) / (n - 1))
    t = m / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    up = sum(1 for x in deltas if x > 1e-12)
    dn = sum(1 for x in deltas if x < -1e-12)
    print(f"  before : {os.path.basename(before.rstrip('/')):>44}  mean {mean(a[p] for p in common):.4f}")
    print(f"  after  : {os.path.basename(after.rstrip('/')):>44}  mean {mean(b[p] for p in common):.4f}")
    print(f"  paired : n={n}  delta={m:+.4f}  sd={sd:.4f}  t={t:+.2f}")
    print(f"           improved {up}   worsened {dn}   unchanged {n - up - dn}")
    print("  NOTE: compare this delta to the paper's +0.0225 (t=+5.72). Do NOT compare")
    print("        levels across arms with different protein populations.")
    return 0


def self_test(onto: Ontology) -> int:
    """Sanity-check the hand-rolled closure before trusting any score."""
    ok = True
    # Every branch terminates at one of the three ontology roots.
    roots = {"GO:0008150", "GO:0003674", "GO:0005575"}
    anc = onto.ancestors("GO:0006468")  # protein phosphorylation
    for label, cond in [
        ("closure is self-inclusive", "GO:0006468" in anc),
        ("closure reaches a root", bool(anc & roots)),
        ("closure includes GO:0008152 metabolic process", "GO:0008152" in anc),
        ("root has no non-self ancestors", onto.ancestors("GO:0008150") == frozenset({"GO:0008150"})),
        ("unknown term passes through", onto.ancestors("GO:9999999") == frozenset({"GO:9999999"})),
        ("f1 of disjoint sets is 0", f1({"GO:1"}, {"GO:2"}) == 0.0),
        ("f1 of identical sets is 1", f1({"GO:1"}, {"GO:1"}) == 1.0),
        ("empty prediction scores 0", f1(set(), {"GO:1"}) == 0.0),
        # Tolerance, not equality: the rank sums are exact but the correlation is a
        # ratio of floating-point sums and lands 2e-16 off 1.0.
        ("spearman of a monotone pair is 1", abs(spearman([1, 2, 3], [4, 5, 9]) - 1.0) < 1e-9),
        ("spearman of an antitone pair is -1", abs(spearman([1, 2, 3], [9, 5, 4]) + 1.0) < 1e-9),
        ("spearman is None when a side is constant", spearman([1, 2, 3], [5, 5, 5]) is None),
    ]:
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")
        ok &= bool(cond)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["rank", "bridge", "paired"])
    ap.add_argument("--rollouts", help="rollouts.jsonl from a run OUTPUT_DIR")
    ap.add_argument("--eval-dir", action="append", default=[], help="repeatable; an eval_out/<tag> dir")
    ap.add_argument("--before", help="--mode paired: the earlier eval dir")
    ap.add_argument("--after", help="--mode paired: the later eval dir")
    ap.add_argument("--obo", default=DEFAULT_OBO)
    ap.add_argument("--ia", default=DEFAULT_IA)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    onto = Ontology(*load_obo(args.obo))
    if args.self_test:
        return self_test(onto)
    if args.mode == "rank":
        if not args.rollouts:
            ap.error("--mode rank requires --rollouts")
        return mode_rank(onto, load_ia(args.ia), args.rollouts)
    if args.mode == "bridge":
        if not args.eval_dir:
            ap.error("--mode bridge requires at least one --eval-dir")
        return mode_bridge(onto, args.eval_dir)
    if args.mode == "paired":
        if not (args.before and args.after):
            ap.error("--mode paired requires --before and --after")
        return mode_paired(onto, args.before, args.after)
    ap.error("pass --mode or --self-test")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build IA.txt (information accretion) for weighted F_max, from the GO DAG + a corpus
of protein GO annotations. NO gated wanglab/cafa5 needed — uses go-basic.obo (in repo)
+ the local RL parquet's ground-truth go_ids.

IA definition (Clark & Radivojac 2013, the CAFA weighted-F_max weight):
    IA(v) = -log2 P(v | Parents(v))
          = -log2 ( count_propagated(v) / count_propagated(Parents(v)) )
where counts are over a corpus with each protein's annotations PROPAGATED up the DAG to
all ancestors (the standard CAFA convention). Parents(v) co-occurrence is approximated by
the count of proteins annotated to ALL parents of v (= count at the intersection); with
single-parent terms this is just the parent count. For multi-parent terms we use the
count of proteins annotated to every parent (propagated annotations guarantee a protein
annotated to v is annotated to all parents, so this is well-defined and <= each parent).

Output format (what cafaeval.parser.ia_parser expects): "<GO_term>\t<ia_value>" per line.

Reuses cafaeval's obo_parser so the DAG (is_a + part_of, per namespace, obsolete excluded)
is BIT-IDENTICAL to what the scorer uses — no DAG drift between IA and evaluation.

Usage:
  PYTHONPATH=$BIOREASON_DEPS python build_ia.py \
    --obo .../go-basic.obo --parquet .../train-00000-of-00001.parquet --out .../IA.txt
"""
import argparse
import math
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--go_col", default="go_ids",
                    help="parquet column with the per-protein GO term list (ground truth)")
    args = ap.parse_args()

    from cafaeval.parser import obo_parser
    import pandas as pd
    import numpy as np

    # 1. Parse the DAG exactly as the scorer will (per-namespace graphs).
    ontologies = obo_parser(args.obo, valid_rel=("is_a", "part_of"), ia_file=None, orphans=True)
    # Build global maps: term -> (namespace, index), and term -> set(parent terms).
    term_ns = {}
    term_parents = {}          # term -> set(parent term ids) within its namespace
    alt_to_canon = {}          # alt_id -> canonical term
    for ns, graph in ontologies.items():
        # graph.terms_list[i] = {'id','adj':set(parent idxs),'children':...}
        idx_to_id = {graph.terms_dict[t]['index']: t for t in graph.terms_dict}
        for entry in graph.terms_list:
            tid = entry['id']
            term_ns[tid] = ns
            term_parents[tid] = {idx_to_id[j] for j in entry['adj']}
        for a_id, canon_set in graph.terms_dict_alt.items():
            for c in canon_set:
                alt_to_canon[a_id] = c
    print(f"[ia] parsed DAG: {len(term_ns)} terms across {len(ontologies)} namespaces",
          flush=True)

    def canon(t):
        return alt_to_canon.get(t, t)

    # 2. Propagated annotation counts. For each protein, take its GO terms, map alt->canon,
    #    propagate to ALL ancestors (transitive closure over parents), count each term once.
    # Precompute ancestors per term (memoized DFS over term_parents).
    _anc_cache = {}
    def ancestors(t):
        if t in _anc_cache:
            return _anc_cache[t]
        acc = set()
        stack = list(term_parents.get(t, ()))
        while stack:
            p = stack.pop()
            if p not in acc:
                acc.add(p)
                stack.extend(term_parents.get(p, ()))
        _anc_cache[t] = acc
        return acc

    df = pd.read_parquet(args.parquet)
    if args.go_col not in df.columns:
        print(f"[ia] ERROR: column {args.go_col} not in parquet {list(df.columns)}",
              file=sys.stderr)
        sys.exit(2)

    import ast
    def _as_list(terms):
        # Handle both schemas: train parquet stores numpy arrays/lists; test parquet
        # stores a STRING repr "['GO:...', ...]". NaN/None -> empty.
        if terms is None:
            return []
        if isinstance(terms, float):  # NaN
            return []
        if isinstance(terms, str):
            s = terms.strip()
            if not s:
                return []
            try:
                return list(ast.literal_eval(s))
            except Exception:
                return []
        return list(terms)

    count = defaultdict(int)   # propagated count per term
    n_prot = 0
    n_unknown = 0
    for terms in df[args.go_col]:
        _lst = _as_list(terms)
        if not _lst:
            continue
        # parquet stores numpy arrays / lists of strings / string-repr
        tset = set()
        for raw in _lst:
            t = canon(str(raw))
            if t in term_ns:
                tset.add(t)
                tset |= ancestors(t)
            else:
                n_unknown += 1
        for t in tset:
            count[t] += 1
        n_prot += 1
    print(f"[ia] propagated {n_prot} proteins; {len(count)} terms with counts; "
          f"{n_unknown} unknown/obsolete term mentions skipped", flush=True)

    # 3. IA(v) = -log2( count(v) / count_of_all_parents(v) ).
    # count_of_all_parents = number of proteins annotated to EVERY parent of v. Under full
    # propagation a protein annotated to v is annotated to all parents, so this >= count(v).
    # We approximate the joint-parent count by the MIN over parents (tight upper bound on
    # the true joint; exact for single-parent terms, the vast majority). Roots (no parents)
    # get IA = -log2(count(v)/N) (accretion vs the whole corpus), the standard convention.
    N = n_prot
    out_lines = []
    n_zero_parent = 0
    for t, c in count.items():
        if c <= 0:
            continue
        parents = term_parents.get(t, set())
        parents_in_corpus = [count[p] for p in parents if count.get(p, 0) > 0]
        if parents_in_corpus:
            denom = min(parents_in_corpus)   # tight bound on joint-parent count
        else:
            denom = N                        # root / no counted parent: vs corpus size
            n_zero_parent += 1
        # guard: propagation should give denom >= c; clamp to avoid negative IA from the
        # min-approximation on rare multi-parent edge cases.
        p_cond = min(1.0, c / denom) if denom > 0 else 1.0
        ia = -math.log2(p_cond) if p_cond > 0 else 0.0
        if ia < 0:
            ia = 0.0
        out_lines.append((t, ia))

    out_lines.sort()
    with open(args.out, "w") as f:
        for t, ia in out_lines:
            f.write(f"{t}\t{ia:.6f}\n")
    ias = [ia for _, ia in out_lines]
    print(f"[ia] wrote {len(out_lines)} terms to {args.out} "
          f"({n_zero_parent} root/no-parent). IA stats: "
          f"min={min(ias):.3f} max={max(ias):.3f} mean={sum(ias)/len(ias):.3f}",
          flush=True)


if __name__ == "__main__":
    main()

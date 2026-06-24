#!/usr/bin/env python
"""Collate grpo_bench results/*.json into one markdown table."""
import glob, json, os

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def fmt(x, n=1):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def main():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(fp) as f:
            r = json.load(f)
        rows.append(r)

    hdr = ("| tag | platform | nodes | gen | G | max_gen | micro | step_med(s) | "
           "CoV | gen_tok/s node | gen_tok/s dev | mean_cmpl_len | peak_mem(GB) | node |")
    sep = "|" + "|".join(["---"] * 14) + "|"
    print(hdr)
    print(sep)
    for r in rows:
        nodes = (r.get("world_size", 0) // max(1, r.get("visible_devices_per_rank", 1)))
        print("| {tag} | {plat} | {nodes} | {gen} | {G} | {mg} | {micro} | {st} | "
              "{cov} | {gtn} | {gtd} | {mcl} | {mem} | {node} |".format(
                  tag=r.get("tag"), plat=r.get("platform"),
                  nodes=nodes or "?", gen=r.get("gen_backend"),
                  G=r.get("num_generations"), mg=r.get("max_completion_length"),
                  micro=r.get("micro_bsz"),
                  st=fmt(r.get("step_time_median_s"), 2),
                  cov=fmt(r.get("step_time_cov"), 3),
                  gtn=fmt(r.get("gen_tok_per_sec_node"), 0),
                  gtd=fmt(r.get("gen_tok_per_sec_device"), 0),
                  mcl=fmt(r.get("mean_completion_len"), 0),
                  mem=fmt(r.get("peak_mem_gb"), 1),
                  node=r.get("node", "?")))


if __name__ == "__main__":
    main()

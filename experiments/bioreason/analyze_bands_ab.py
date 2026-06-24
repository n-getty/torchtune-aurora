#!/usr/bin/env python3
"""Analyze the BioReason straggler-band A/B (same-node, back-to-back legs).

Parses the two train logs (bandsOFF, bandsON) produced by batch_bands_ab.sh and
reports, per leg:
  - engine spread: which engine ids each replica leader used (ids=[...] lines) —
    the DIRECT check that bands ON => 3 disjoint 4-engine bands covering 0..11,
    bands OFF => all leaders pile on 0..3.
  - per-step generation time per leader (min/median/max across leaders => the
    straggler spread the step barrier pays for).
  - per-step total step time (steady-state median, dropping step 0/1 warmup).
  - the headline: median step time OFF vs ON and the % delta.

Same-node back-to-back => the delta is immune to Aurora node-to-node variance.

Usage:
  python analyze_bands_ab.py <bandsOFF.log> <bandsON.log>
  python analyze_bands_ab.py --auto   # picks the newest *_bandsOFF / *_bandsON
"""
import re
import sys
import glob
import os
import statistics as st

GEN_RE = re.compile(
    r"Rank (\d+): vLLM-embeds (?:HTTP|generation): (\d+) seqs?.*?(\d+) tokens in ([\d.]+)s"
)
IDS_RE = re.compile(r"Rank (\d+): vLLM-embeds HTTP: \d+ seqs over (\d+) engines \(ids=(\[[^\]]*\])\)")
# Explicit rank-0 per-step timing line:
#   TIMING step=0  total=185.9s  gen=124.3s  grpo=59.5s  clip=0.2s  opt=0.0s  other=1.9s
TIMING_RE = re.compile(
    r"TIMING step=(\d+)\s+total=([\d.]+)s\s+gen=([\d.]+)s\s+grpo=([\d.]+)s"
)
DIAG_RE = re.compile(
    r"len_mean=([\d.]+).*?trunc_rate=([\d.]+)\s+stop_rate=([\d.]+)"
)


def _parse(path):
    leader_gen = {}           # rank -> [gen_s,...]
    leader_ids = {}           # rank -> [ids_str,...]
    steps = []                # list of (step, total, gen, grpo)
    diags = []                # list of (len_mean, trunc_rate, stop_rate)
    with open(path, errors="replace") as f:
        for line in f:
            m = GEN_RE.search(line)
            if m:
                rank = int(m.group(1)); gen_s = float(m.group(4))
                leader_gen.setdefault(rank, []).append(gen_s)
            m = IDS_RE.search(line)
            if m:
                rank = int(m.group(1)); ids = m.group(3)
                leader_ids.setdefault(rank, []).append(ids)
            m = TIMING_RE.search(line)
            if m:
                steps.append((int(m.group(1)), float(m.group(2)),
                              float(m.group(3)), float(m.group(4))))
            m = DIAG_RE.search(line)
            if m:
                diags.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return leader_gen, leader_ids, steps, diags


def _summ(name, leader_gen, leader_ids, steps, diags):
    print(f"\n===== {name} =====")
    if diags:
        lm = st.median([d[0] for d in diags]); tr = st.median([d[1] for d in diags])
        sr = st.median([d[2] for d in diags])
        print(f"  DIAG (median): len_mean={lm:.1f} trunc_rate={tr:.3f} stop_rate={sr:.3f}"
              + ("   <- stop tokens ENGAGED" if sr > 0 else "   <- NO EOS stops (max_gen-bound)"))
    if leader_ids:
        print("  engine-id usage per leader rank (last occurrence):")
        all_ids = set()
        for rank in sorted(leader_ids):
            last = leader_ids[rank][-1]
            print(f"    rank {rank}: ids={last}  (n={len(leader_ids[rank])} steps)")
            for tok in re.findall(r"\d+", last):
                all_ids.add(int(tok))
        cov = sorted(all_ids)
        print(f"  union of engines used: {cov}  ({len(cov)}/12)")
        if len(cov) >= 11:
            print("    -> GOOD: bands cover (nearly) all 12 engines")
        else:
            print("    -> engines IDLE:", sorted(set(range(12)) - all_ids))
    # straggler spread: per "round" (align leaders by index)
    if leader_gen:
        ranks = sorted(leader_gen)
        n = min(len(v) for v in leader_gen.values())
        spreads, slowest = [], []
        for i in range(n):
            row = [leader_gen[r][i] for r in ranks]
            spreads.append(max(row) - min(row))
            slowest.append(max(row))
        if spreads:
            print(f"  per-step leader gen spread (max-min): "
                  f"median={st.median(spreads):.1f}s max={max(spreads):.1f}s")
            print(f"  per-step SLOWEST leader gen (barrier pays this): "
                  f"median={st.median(slowest):.1f}s")
        for r in ranks:
            v = leader_gen[r]
            print(f"    rank {r} gen: median={st.median(v):.1f}s n={len(v)}")
    med_total = med_gen = None
    if len(steps) >= 2:
        steady = [s for s in steps if s[0] >= 1]  # drop step 0 warmup
        if not steady:
            steady = steps
        totals = [s[1] for s in steady]; gens = [s[2] for s in steady]
        grpos = [s[3] for s in steady]
        med_total, med_gen = st.median(totals), st.median(gens)
        print(f"  TIMING (steady, n={len(steady)}): total median={med_total:.1f}s "
              f"(min {min(totals):.1f}/max {max(totals):.1f}) | "
              f"gen median={med_gen:.1f}s | grpo median={st.median(grpos):.1f}s")
    return med_total, med_gen


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    if not args or args[0] in ("--auto", "--auto-stop"):
        kind = "stop" if (args and args[0] == "--auto-stop") else "bands"
        off = sorted(glob.glob(os.path.join(here, f"train_mpiexec_*_{kind}OFF.log")))
        on = sorted(glob.glob(os.path.join(here, f"train_mpiexec_*_{kind}ON.log")))
        if not off or not on:
            print(f"No {kind}OFF/{kind}ON logs found yet.", file=sys.stderr); sys.exit(2)
        off, on = off[-1], on[-1]
    else:
        off, on = args[0], args[1]
    print(f"OFF log: {off}\nON  log: {on}")
    off_total, off_gen = _summ("LEG OFF (old path)", *_parse(off))
    on_total, on_gen = _summ("LEG ON (fix)", *_parse(on))
    print("\n===== HEADLINE (same-node, variance-immune) =====")
    if off_total and on_total:
        dt_ = (off_total - on_total) / off_total * 100
        dg_ = (off_gen - on_gen) / off_gen * 100 if (off_gen and on_gen) else float("nan")
        print(f"  step total: OFF={off_total:.1f}s  ON={on_total:.1f}s  "
              f"delta={dt_:+.1f}% ({'FIX FASTER' if dt_>0 else 'NO WIN / REGRESSION'})")
        print(f"  gen time:   OFF={off_gen:.1f}s  ON={on_gen:.1f}s  delta={dg_:+.1f}%")
    else:
        print("  (insufficient TIMING samples in one leg — check both legs ran >=2 steps)")


if __name__ == "__main__":
    main()

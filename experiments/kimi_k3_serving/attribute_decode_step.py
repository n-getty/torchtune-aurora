"""Attribute a K3 c=1 decode step from a DEVICE-ONLY Kineto trace.

`analyze_decode_trace.py` refuses these traces, and rightly so: with no CPU
rows and no collective kernels it cannot tell host dispatch from time blocked
in oneCCL, and those imply opposite optimizations. This script does the
attribution anyway, using two measurements taken OUTSIDE the trace to break
that tie:

  * per-launch dispatch cost, measured on-node with a tiny-kernel burst
    (6.5 us; unaffected by ZE_ENABLE_API_TRACING, tested both ways);
  * isolated collective latency, measured with the 32-rank benchmark
    (0.557 ms for 14 KiB).

The key observation that makes a device-only trace usable: XCCL's collectives
appear in it as `Memcpy M2D (MEMORY(Unknown) -> DEVICE)`, and their count
matches the collective count per token 1:1 (464 vs 463 on job 8748640). So
the long gap that ENDS at an M2D is the collective's real in-situ cost --
rendezvous, staging and all -- which is the thing the isolated benchmark
cannot see.

Reported per token:
    compute      sum of kernel durations
    dispatch     launches x measured per-launch cost
    collectives  in-situ, from the D2M -> gap -> M2D pattern
    remainder    everything left over, stated as unexplained

Usage:
  python attribute_decode_step.py TRACE.json.gz --tokens 12 [--step-ms 1249]
"""

import argparse
import gzip
import json
from collections import Counter

# Measured on a compute node (probe_launch_overhead.py): a burst of tiny
# elementwise kernels costs 6.50-6.58 us each to launch, with
# ZE_ENABLE_API_TRACING either on or off.
PER_LAUNCH_US = 6.5
# Measured by bench_collective_latency_3node.py at 32 ranks / 3 nodes.
ISOLATED_COLLECTIVE_MS = 0.557
# Counted from the model definition (93 attn o_proj + 92 MoE-final + 92
# routed up + 92 shared down + 92 routed-down gather + embedding/logits).
COLLECTIVES_PER_STEP = 463
# A gap this long between kernels is a stall, not scheduling jitter.
STALL_US = 400.0


def load(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace.get("traceEvents", [])
    return sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and "dur" in e
            and str(e.get("cat", "")) in ("kernel", "gpu_memcpy")
        ),
        key=lambda e: e["ts"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--tokens", type=int, required=True,
                        help="completion_tokens generated inside the profiled window")
    parser.add_argument("--step-ms", type=float, default=None,
                        help="known-good step time to attribute against; "
                             "default is this trace's own span/tokens")
    args = parser.parse_args()

    events = load(args.trace)
    if not events:
        raise SystemExit("no device rows in this trace")
    tokens = args.tokens

    span_ms = (
        max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)
    ) / 1000.0
    compute_ms = sum(e["dur"] for e in events) / 1000.0
    launches = len(events)

    # In-situ collective cost: the stall that ends at an M2D memcpy.
    insitu_us = 0.0
    insitu_n = 0
    for first, second in zip(events, events[1:]):
        gap = second["ts"] - (first["ts"] + first["dur"])
        if gap > STALL_US and "M2D" in second.get("name", ""):
            insitu_us += gap
            insitu_n += 1

    m2d = sum(1 for e in events if "M2D" in e.get("name", ""))

    per_tok_span = span_ms / tokens
    step = args.step_ms if args.step_ms else per_tok_span
    per_tok = {
        "compute": compute_ms / tokens,
        "dispatch": launches * PER_LAUNCH_US / 1000.0 / tokens,
        "collectives_insitu": insitu_us / 1000.0 / tokens,
    }
    per_tok["remainder"] = step - sum(per_tok.values())

    print(f"trace            {args.trace}")
    print(f"tokens in window {tokens}")
    print(f"trace span       {span_ms:.0f} ms  ({per_tok_span:.0f} ms/token)")
    print(f"attributing against step = {step:.0f} ms/token"
          f"{'  (given)' if args.step_ms else '  (this trace)'}")
    print()
    print(f"launches/token   {launches / tokens:,.0f}")
    print(f"M2D/token        {m2d / tokens:,.0f}   "
          f"(vs {COLLECTIVES_PER_STEP} collectives/step"
          f"{' -- 1:1' if abs(m2d / tokens - COLLECTIVES_PER_STEP) < 30 else ''})")
    print(f"in-situ stalls   {insitu_n / tokens:,.0f}/token, mean "
          f"{insitu_us / insitu_n if insitu_n else 0:.0f} us")
    print()
    print(f"{'term':<22}{'ms/token':>10}{'share':>9}")
    for name, value in per_tok.items():
        print(f"{name:<22}{value:>10.0f}{value / step:>9.1%}")
    print()

    isolated_ms = ISOLATED_COLLECTIVE_MS * COLLECTIVES_PER_STEP
    print(f"collectives isolated:  {isolated_ms:.0f} ms/token "
          f"({ISOLATED_COLLECTIVE_MS:.3f} ms x {COLLECTIVES_PER_STEP})")
    print(f"collectives in situ :  {per_tok['collectives_insitu']:.0f} ms/token")
    if isolated_ms:
        print(f"  in-situ / isolated = "
              f"{per_tok['collectives_insitu'] / isolated_ms:.2f}x -- the isolated "
              f"benchmark syncs first, so it cannot see rendezvous or staging.")
    print()
    print("CAVEAT: dispatch is inferred (launches x a separately measured")
    print("per-launch cost), not read from the trace -- this trace has no CPU")
    print("rows. The remainder is what no measurement accounts for; do not")
    print("relabel it as any one term.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

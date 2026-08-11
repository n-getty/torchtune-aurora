"""Attribute one c=1 K3 decode step from a Kineto trace, and apply the
pre-registered decision rules.

WHY THIS EXISTS
---------------
Every K3 optimization decision so far has been sized by a *model* of where the
1249 ms c=1 decode step goes, not a measurement. The two candidate models
disagree about what to do next:

  collectives  ~460-930 ms   ~463 all_reduce/all_gather per step x 1-2 ms
                             (32 ranks, 3 nodes, 14 KiB each -- latency-bound)
  dispatch     ~145 ms       ~20,700 eager op dispatches x ~7 us
  host syncs   ~62 ms        torch.unique / torch.all / .item() in KDA+MLA
  kernels      remainder     not separately measured

Graph capture removes dispatch and syncs but REPLAYS the same 463 network
round-trips. So if collectives dominate, capture cannot reach the target and
the collective count must be cut first. This script decides which.

DECISION RULES ARE PRE-REGISTERED. They were written down before any trace was
looked at (see the plan doc), and are encoded here rather than in prose so the
verdict is not chosen after seeing the numbers:

  collectives      > 40%  -> cut collectives first; capture is secondary
  gaps + syncs     > 40%  -> capture is the main event
  both     25-40%         -> collectives first (cheaper, no venv migration)
  gpu_busy         > 60%  -> the premise is WRONG; stop and re-derive

One clarification to the last rule, forced by a synthetic test before this saw
real data: "gpu_busy" is read as busy WITH COMPUTE, i.e. collectives excluded.
On XPU a blocking all_reduce occupies a device queue and appears as a device
row, so a trace that is 85% all_reduce would otherwise score 85% "GPU busy"
and fire the premise-wrong rule at exactly the moment it confirmed the
collective hypothesis. Both numbers are reported.

Usage:
  python analyze_decode_trace.py TRACE.json[.gz] [--steps N] [--json OUT]

Reads a torch profiler trace (the worker-side one, which has both CPU and XPU
rows). Reports per-category wall time, the inter-kernel gap, and the verdict.
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

# Substring patterns, checked lowercase, first match wins. Order matters:
# collectives must be checked before the generic kernel bucket, because an
# XCCL op also shows up as a device kernel.
CATEGORY_PATTERNS = [
    (
        "collective",
        (
            "allreduce",
            "all_reduce",
            "allgather",
            "all_gather",
            "reduce_scatter",
            "reducescatter",
            "broadcast",
            "alltoall",
            "all_to_all",
            "xccl",
            "oneccl",
            "ccl::",
            "c10d::",
        ),
    ),
    (
        "host_sync",
        (
            "unique",
            "aten::item",
            "_local_scalar_dense",
            "aten::all",
            "aten::any",
            "synchronize",
            "memcpydtoh",
            "memcpy dtoh",
        ),
    ),
    (
        "moe",
        (
            "moe",
            "grouped_gemm",
            "cutlass_grouped",
            "expert",
            "topk",
            "situ",
        ),
    ),
    (
        "attention",
        (
            "kda",
            "mla",
            "conv1d",
            "attention",
            "flash",
            "sdpa",
            "recurrent",
        ),
    ),
]


def categorize(name: str) -> str:
    lowered = name.lower()
    for category, patterns in CATEGORY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return category
    return "other_kernel"


def load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    return [e for e in events if e.get("ph") == "X" and "dur" in e]


def is_device_event(event: dict) -> bool:
    """Device-side rows, i.e. rows that consume GPU time.

    Kineto labels XPU rows variously across versions ("gpu_user_annotation",
    "kernel", "xpu_op", "gpu_memcpy"). Match on the category prefix rather
    than an exact string so a version bump degrades to "unclassified" (loud,
    visible in the report) instead of "zero device time" (silent, and would
    trip the gpu_busy>60% rule backwards).
    """
    cat = str(event.get("cat", "")).lower()
    return (
        cat.startswith("kernel")
        or cat.startswith("gpu")
        or cat.startswith("xpu")
        or "memcpy" in cat
        or "memset" in cat
    )


def merged_busy_span(intervals: list[tuple[float, float]]) -> float:
    """Union of intervals, in us. Overlapping kernels must not double-count."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def analyze(events: list[dict], steps: int) -> dict:
    device_events = [e for e in events if is_device_event(e)]
    if not device_events:
        raise SystemExit(
            "No device-side rows found in this trace. Either the profiler ran "
            "with activities=['CPU'] only, or Kineto's XPU category names "
            "changed -- check `cat` values before trusting any verdict."
        )

    span_start = min(e["ts"] for e in device_events)
    span_end = max(e["ts"] + e["dur"] for e in device_events)
    wall_us = span_end - span_start

    by_category: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    intervals: list[tuple[float, float]] = []
    for event in device_events:
        category = categorize(event.get("name", ""))
        by_category[category] += event["dur"]
        counts[category] += 1
        intervals.append((event["ts"], event["ts"] + event["dur"]))

    busy_us = merged_busy_span(intervals)
    gap_us = max(0.0, wall_us - busy_us)

    # "GPU busy" must mean busy WITH COMPUTE for the premise rule to mean what
    # it was written to mean. On XPU a blocking collective occupies a device
    # queue and shows up as a device row, so a trace that is 85% all_reduce
    # reads as 85% "GPU busy" -- which would fire the >60% premise-wrong rule
    # and tell us to abandon the collective hypothesis at the exact moment the
    # trace confirms it. (Caught by a synthetic collective-heavy trace before
    # this script ever saw real data.) Exclude collectives from the compute
    # fraction; keep the raw one for reference.
    compute_intervals = [
        (e["ts"], e["ts"] + e["dur"])
        for e in device_events
        if categorize(e.get("name", "")) != "collective"
    ]
    compute_busy_us = merged_busy_span(compute_intervals)

    # Host syncs are mostly CPU-side stalls; count them from the CPU rows too.
    host_sync_cpu_us = sum(
        e["dur"]
        for e in events
        if not is_device_event(e) and categorize(e.get("name", "")) == "host_sync"
    )

    return {
        "steps": steps,
        "wall_us": wall_us,
        "wall_ms_per_step": wall_us / 1000.0 / steps,
        "busy_us": busy_us,
        "compute_busy_us": compute_busy_us,
        "gap_us": gap_us,
        "gpu_busy_frac": busy_us / wall_us if wall_us else 0.0,
        "compute_busy_frac": compute_busy_us / wall_us if wall_us else 0.0,
        "gap_frac": gap_us / wall_us if wall_us else 0.0,
        "host_sync_cpu_us": host_sync_cpu_us,
        "by_category_us": dict(by_category),
        "by_category_frac": {
            k: v / wall_us for k, v in by_category.items() if wall_us
        },
        "counts": dict(counts),
        "collectives_per_step": counts.get("collective", 0) / steps,
    }


def verdict(report: dict) -> tuple[str, str]:
    """Apply the pre-registered rules. Returns (verdict, rationale)."""
    collective = report["by_category_frac"].get("collective", 0.0)
    gaps_and_syncs = report["gap_frac"] + report["by_category_frac"].get(
        "host_sync", 0.0
    )
    # COMPUTE busy, not raw device busy: a blocking collective occupies a
    # device queue, so raw busy counts the very thing the collective rule is
    # about. See the note where compute_busy_us is computed.
    compute_busy = report["compute_busy_frac"]

    # Checked FIRST: if the GPU is genuinely doing compute, both models are
    # wrong and the shares below are shares of the wrong thing.
    if compute_busy > 0.60:
        return (
            "PREMISE-WRONG",
            f"GPU busy with compute {compute_busy:.1%} > 60% (collectives "
            "excluded): the step is not dominated by collectives OR launch "
            "overhead. Stop and re-derive from the trace -- neither Step 2 "
            "nor Step 5 is indicated.",
        )
    if collective > 0.40:
        return (
            "CUT-COLLECTIVES",
            f"collectives {collective:.1%} > 40%: Step 2 is the main event. "
            "Graph capture replays the same round-trips and is secondary.",
        )
    if gaps_and_syncs > 0.40:
        return (
            "CAPTURE",
            f"gaps+syncs {gaps_and_syncs:.1%} > 40%: launch overhead "
            "dominates. Go to Step 3 then Step 5 (needs the 2.11 venv).",
        )
    if 0.25 <= collective <= 0.40 and 0.25 <= gaps_and_syncs <= 0.40:
        return (
            "COLLECTIVES-FIRST",
            f"collectives {collective:.1%} and gaps+syncs "
            f"{gaps_and_syncs:.1%} both 25-40%: do Step 2 first (cheaper, no "
            "venv migration), then Step 3/5.",
        )
    return (
        "INCONCLUSIVE",
        f"collectives {collective:.1%}, gaps+syncs {gaps_and_syncs:.1%}, "
        f"compute_busy {compute_busy:.1%} match no pre-registered rule. Do NOT "
        "pick a branch by eye -- widen the trace (more steps) or report the "
        "split as-is.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="worker Kineto trace (.json or .json.gz)")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="decode steps captured in this trace (for per-step normalization)",
    )
    parser.add_argument("--json", help="also write the report as JSON here")
    args = parser.parse_args()

    report = analyze(load_events(args.trace), args.steps)
    call, rationale = verdict(report)
    report["verdict"] = call
    report["rationale"] = rationale

    print(f"trace           {args.trace}")
    print(f"steps           {args.steps}")
    print(f"wall            {report['wall_us'] / 1000:.1f} ms "
          f"({report['wall_ms_per_step']:.1f} ms/step)")
    print(f"gpu busy        {report['busy_us'] / 1000:.1f} ms "
          f"({report['gpu_busy_frac']:.1%})")
    print(f"  of which cmpt {report['compute_busy_us'] / 1000:.1f} ms "
          f"({report['compute_busy_frac']:.1%}, collectives excluded)")
    print(f"inter-kernel gap{report['gap_us'] / 1000:>9.1f} ms "
          f"({report['gap_frac']:.1%})")
    print(f"host syncs (cpu){report['host_sync_cpu_us'] / 1000:>9.1f} ms")
    print(f"collectives/step{report['collectives_per_step']:>9.1f}")
    print("")
    print(f"{'category':<16}{'ms':>10}{'share':>9}{'count':>9}")
    for category, micros in sorted(
        report["by_category_us"].items(), key=lambda kv: -kv[1]
    ):
        print(
            f"{category:<16}{micros / 1000:>10.1f}"
            f"{report['by_category_frac'][category]:>9.1%}"
            f"{report['counts'][category]:>9}"
        )
    print("")
    print(f"VERDICT  {call}")
    print(f"         {rationale}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare two same-topology sealed MoE optimization summaries."""

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchtune.modules.moe.measurement import compare_optimization_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--varying-control", action="append", default=[])
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("optimization summary JSON must contain mappings")
    result = compare_optimization_summaries(
        baseline, candidate, varying_controls=args.varying_control
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

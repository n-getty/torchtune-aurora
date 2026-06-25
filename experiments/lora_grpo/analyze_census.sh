#!/bin/bash
# Extract + diff LEAK_CENSUS output to name the per-step retained tensor.
# Usage: bash analyze_census.sh <run_log>
set -eo pipefail
L="${1:?usage: analyze_census.sh <run_log>}"

echo "=== gc_total vs alloc vs gc_unseen per step (is the leak gc-visible?) ==="
grep -E "LEAK_CENSUS step=[0-9]+ live_xpu_tensors" "$L" | \
  grep -oE "step=[0-9]+ live_xpu_tensors gc_total=[0-9.]+ GiB alloc=[0-9.]+ GiB gc_unseen=[0-9.]+ GiB groups=[0-9]+" | sort -u

echo
echo "=== top tensor groups per step (look for the group whose GiB or count grows ~+2.5/step) ==="
grep -E "LEAK_CENSUS step=[0-9]+ +[0-9]" "$L" | \
  grep -oE "step=[0-9]+ +[0-9.]+ GiB x[0-9]+ +shape=.* (torch\.[a-z0-9]+)" | sort -t= -k2 -n || \
  grep -E "LEAK_CENSUS step=[0-9]+ +[0-9.]+ GiB" "$L"

echo
echo "=== DIFF: same-shape group, bytes at step 1 vs last (the accumulator) ==="
# Pull (step, shape, gib, count) tuples and show per-shape growth
grep -E "LEAK_CENSUS step=[0-9]+ +[0-9.]+ GiB x" "$L" | \
  sed -E 's/.*step=([0-9]+) +([0-9.]+) GiB x([0-9]+) +shape=(\([^)]*\)) (torch\.[a-z0-9]+).*/\1|\4 \5|\2|\3/' | \
  awk -F'|' '{key=$2; step=$1; gib=$3; cnt=$4;
              if (!(key in first)) {first[key]=gib; firstc[key]=cnt; firsts[key]=step}
              last[key]=gib; lastc[key]=cnt; lasts[key]=step}
       END{ printf "%-34s %8s %8s %8s %8s\n","shape dtype","s"firsts["x"]"_GiB","last_GiB","s_cnt","last_cnt";
            for (k in first) printf "%-34s %8s %8s %8s %8s\n", k, first[k], last[k], firstc[k], lastc[k] }'

#!/usr/bin/env bash
# Compare the DAOS-mounted K3 checkpoint against the Lustre copy, byte for byte.
#
# WHY
# ---
# Every run that actually showed the decode degeneration (all six, 2026-08-05)
# served the checkpoint from a **dfuse mount of a DAOS container** at
# /tmp/AuroraGPT/prism_models, mounted with launch-dfuse-with-caching.sh. Every
# run since has read Lustre directly -- and none of those has generated more
# than a single token, so none of them has re-observed the bug.
#
# Silently wrong bytes from a cached or partial DAOS read reproduce the exact
# observed signature:
#   * the model loads cleanly (96/96 shards present, strict tracking passes),
#   * the FIRST token is correct, because it exercises a narrow slice of the
#     weights and most of them are fine,
#   * the tail degenerates as more of the corrupted region is touched,
#   * and every component / equivalence / prefill-vs-decode test passes,
#     because the CODE is correct and only the DATA is wrong.
# That last point matters: it is consistent with everything found so far.
# Router, attention residual, SiTU, latent MoE, fused MoE at M=1, prefill
# vs decode at TP=1 and TP=8+EP -- all clean, on Lustre weights.
#
# Critically, the degenerate runs did NOT set enable_weights_track (absent from
# their metadata; present in every later run), so nothing in them would have
# noticed corrupt weights.
#
# WHAT THIS DOES
# --------------
# Hashes each safetensors shard on both paths and reports any mismatch. Reads
# are streamed, and it hashes a configurable subset by default because 1.5 TiB
# twice is not free -- pass --all to do the whole checkpoint.
#
# Run this ON A NODE THAT HAS THE DAOS MOUNT (the 3-node hold scripts create
# it; see hold_3node.sh). It needs no XPU and no server.
#
# Usage:
#   verify_daos_vs_lustre.sh [--all] [--shards N]

set -o pipefail   # NOT -u: see the Lmod/ZSH_EVAL_CONTEXT note in the suite scripts

DAOS=${DAOS_PATH:-/tmp/AuroraGPT/prism_models}
LUSTRE=${LUSTRE_PATH:-/flare/ModCon/ngetty/models/Kimi-K3}
N=8
ALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all) ALL=1 ;;
    --shards) shift; N="$1" ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "=== DAOS vs Lustre checkpoint comparison $(date -Is) ==="
echo "  daos  : $DAOS"
echo "  lustre: $LUSTRE"

[ -d "$DAOS" ]   || { echo "FATAL: DAOS path not present. Run this on a node with the dfuse mount." >&2; exit 1; }
[ -d "$LUSTRE" ] || { echo "FATAL: Lustre path not present" >&2; exit 1; }

# config.json / index first: cheap, and a mismatch here alone would explain a lot.
for f in config.json model.safetensors.index.json; do
  a=$(md5sum "$DAOS/$f" 2>/dev/null | awk '{print $1}')
  b=$(md5sum "$LUSTRE/$f" 2>/dev/null | awk '{print $1}')
  if [ "$a" = "$b" ] && [ -n "$a" ]; then
    echo "  MATCH    $f"
  else
    echo "  MISMATCH $f  daos=$a lustre=$b"
  fi
done

mapfile -t SHARDS < <(cd "$LUSTRE" && ls -1 model-*.safetensors 2>/dev/null | sort)
echo "  ${#SHARDS[@]} shards on Lustre"

if [ "$ALL" = "1" ]; then
  PICK=("${SHARDS[@]}")
else
  # Sample across the range rather than the first N: a partial stage tends to
  # damage the tail, so first-N would be the least informative choice.
  PICK=()
  total=${#SHARDS[@]}
  step=$(( total / N )); [ "$step" -lt 1 ] && step=1
  for ((i=0; i<total; i+=step)); do PICK+=("${SHARDS[$i]}"); done
  PICK+=("${SHARDS[$((total-1))]}")   # always include the last one
fi

echo "  comparing ${#PICK[@]} shards"
bad=0
for s in "${PICK[@]}"; do
  a=$(md5sum "$DAOS/$s" 2>/dev/null | awk '{print $1}')
  b=$(md5sum "$LUSTRE/$s" 2>/dev/null | awk '{print $1}')
  if [ -z "$a" ]; then
    echo "  UNREADABLE-ON-DAOS $s"; bad=$((bad+1)); continue
  fi
  if [ "$a" = "$b" ]; then
    echo "  MATCH    $s"
  else
    echo "  MISMATCH $s  daos=$a lustre=$b"
    bad=$((bad+1))
  fi
done

echo
if [ "$bad" -gt 0 ]; then
  echo "VERDICT: $bad/${#PICK[@]} shards DIFFER. The DAOS-served weights are not"
  echo "the Lustre weights -- this alone can explain correct-first-token /"
  echo "degenerate-tail with every code-level test passing. Re-stage and re-test"
  echo "before pursuing any further code hypothesis."
  exit 1
fi
echo "VERDICT: all ${#PICK[@]} compared shards identical."
echo "NULL RESULT on a SAMPLE -- rerun with --all before treating DAOS as fully"
echo "exonerated, and note this compares bytes at rest, not what a cached dfuse"
echo "read returned during the failing run."

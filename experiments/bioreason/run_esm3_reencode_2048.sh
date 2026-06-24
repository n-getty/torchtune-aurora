#!/bin/bash
# ESM3 re-encode at max_protein_len=2048 (faithful inputs). One tile, single process.
# Writes a DISTINCT cache (esm3_cache_2048.pt) so the 128 cache/sidecar is untouched.
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1
export ZE_AFFINITY_MASK=0
export HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
cd $TT
echo "=== esm3 reencode start $(date) on $(hostname) ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK ==="
python experiments/bioreason/precompute_esm3_cache.py \
  --data_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl \
  --out /lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt \
  --max_protein_len 2048 \
  --log_every 200 --flush_every 1000
RC=$?
echo "=== esm3 reencode end rc=$RC $(date) ==="
exit $RC

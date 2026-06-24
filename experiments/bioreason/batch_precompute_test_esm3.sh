#!/bin/bash
# One-shot PBS batch: precompute the ESM3 cache for the bioreason_pro_test proteins, so the
# go_pred-fix eval can run from cache (frees ~5.5 GiB/tile -> no UR:40 OOM that killed 6/12
# shards on the live-ESM3 run). The existing esm3_cache_2048 is TRAIN-only (KeyError on test).
# Single tile, ESM3 fp32 encoder over the test sequences (<=2048), sha1-keyed, resumable.
#
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_esm3_test
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_precompute_test_esm3.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
module load frameworks
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export ZE_AFFINITY_MASK=0
cd $TT
echo "=== precompute test ESM3 cache start $(date) job=${PBS_JOBID} ==="
python experiments/bioreason/precompute_esm3_cache.py \
  --data_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test \
  --out /lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt \
  --max_protein_len 2048 --log_every 200
RC=$?
echo "=== precompute test ESM3 cache end rc=$RC $(date) ==="
exit $RC

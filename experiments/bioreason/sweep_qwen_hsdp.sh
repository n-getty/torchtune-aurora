#!/bin/bash
# HSDP shard-width sweep for Qwen3-32B BioReason SFT. Safe-first ordering so a clean
# baseline lands before the aggressive (smaller-shard) cells that risk OOM.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_qwen_hsdp_sweep_cell.sh
TAG=shard12_full DP_SHARD=-1 DP_REPLICATE=1 STEPS=12 bash $R || true   # baseline (Qwen fits-test = clean)
TAG=shard3_rep4  DP_SHARD=3  DP_REPLICATE=4 STEPS=12 bash $R || true   # safe
TAG=shard2_rep6  DP_SHARD=2  DP_REPLICATE=6 STEPS=12 bash $R || true   # aggressive/fastest-if-fits
echo "=== HSDP SWEEP DONE ==="
grep -h HSDP_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/qwen_hsdp_*/run.log 2>/dev/null

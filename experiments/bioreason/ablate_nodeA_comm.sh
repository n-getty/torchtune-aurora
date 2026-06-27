#!/bin/bash
# Node A: communication levers (full-shard baseline vs HSDP). Sequential cells.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=A0_fullshard      NPROC=12 STEPS=20 OVERRIDES=""                                                  bash $R || true
TAG=A1_hsdp_shard4    NPROC=12 STEPS=20 OVERRIDES="data_parallel_shard_dim=4 data_parallel_replicate_dim=3" bash $R || true
TAG=A2_hsdp_shard2    NPROC=12 STEPS=20 OVERRIDES="data_parallel_shard_dim=2 data_parallel_replicate_dim=6" bash $R || true
echo "=== NODE A ABLATIONS DONE ==="
grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_A*/run.log 2>/dev/null

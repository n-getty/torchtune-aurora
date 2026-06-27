#!/bin/bash
# Node B: at the safe seq=4096 / 12 tiles, test throughput levers that don't blow memory.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
BASE="tokenizer.max_seq_len=4096 dataset.max_seq_len=4096"
TAG=B0_seq4096_base NPROC=12 STEPS=20 OVERRIDES="$BASE" bash $R || true
TAG=B1_seq4096_hsdp4 NPROC=12 STEPS=20 OVERRIDES="$BASE data_parallel_shard_dim=4 data_parallel_replicate_dim=3" bash $R || true
TAG=B2_seq4096_bs2 NPROC=12 STEPS=20 OVERRIDES="$BASE batch_size=2" bash $R || true
echo "=== NODE B DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_B*/run.log 2>/dev/null

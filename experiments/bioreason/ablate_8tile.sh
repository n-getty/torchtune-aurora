#!/bin/bash
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=H0_8tile_seq8192 NPROC=8 STEPS=12 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192" bash $R || true
echo "=== 8-TILE DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_H0*/run.log 2>/dev/null

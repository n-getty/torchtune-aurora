#!/bin/bash
# Node A: max_seq_len sweep at 12 tiles, full AC. Safe-first (4096) so we always get one
# clean number before risking the OOM-prone 8192. A banned:1 wedges the node, so if a
# cell hard-crashes the later cells will fail — that's acceptable (safe cell ran first).
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=A0_seq4096 NPROC=12 STEPS=20 OVERRIDES="tokenizer.max_seq_len=4096 dataset.max_seq_len=4096" bash $R || true
TAG=A1_seq6144 NPROC=12 STEPS=20 OVERRIDES="tokenizer.max_seq_len=6144 dataset.max_seq_len=6144" bash $R || true
TAG=A2_seq8192 NPROC=12 STEPS=20 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192" bash $R || true
echo "=== NODE A DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_A*/run.log 2>/dev/null

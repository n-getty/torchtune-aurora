#!/bin/bash
# Confirm imbalance-is-the-cause: full-shard 12 tiles, SDPA on, but cap protein_len=512 +
# seq=4096 so EVERY rank's sequence fits well under the tile. If this trains clean past
# step 3, the crash was per-rank long-sequence activation, not the 12-tile topology.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=E0_full12_plen512_seq4096 NPROC=12 STEPS=12 OVERRIDES="tokenizer.max_seq_len=4096 dataset.max_seq_len=4096 dataset.max_protein_len=512" bash $R || true
echo "=== CONFIRM DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_E*/run.log 2>/dev/null

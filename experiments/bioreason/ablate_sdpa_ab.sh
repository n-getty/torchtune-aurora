#!/bin/bash
# Clean A/B on the known-good 6-tile topology, seq=8192: SDPA on vs off. Memory delta
# proves SDPA engaged + quantifies the S^2 savings. 6 tiles avoids the 12-tile imbalance OOM.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
export TORCHTUNE_GEMMA4_SDPA=1
TAG=F_sdpa_on_6t  NPROC=6 STEPS=8 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192" bash $R || true
export TORCHTUNE_GEMMA4_SDPA=0
TAG=F_sdpa_off_6t NPROC=6 STEPS=8 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192" bash $R || true
echo "=== SDPA A/B DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_F_*/run.log 2>/dev/null

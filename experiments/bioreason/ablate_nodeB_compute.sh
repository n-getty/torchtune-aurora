#!/bin/bash
# Node B: compute/memory levers (baseline variance-check, selective AC, batch_size=2).
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=B0_fullshard      NPROC=12 STEPS=20 OVERRIDES=""                                                          bash $R || true
TAG=B1_selac_every2   NPROC=12 STEPS=20 OVERRIDES="enable_activation_checkpointing=False ac_mode=selective ac_option=2" bash $R || true
TAG=B2_bs2            NPROC=12 STEPS=20 OVERRIDES="batch_size=2"                                              bash $R || true
echo "=== NODE B ABLATIONS DONE ==="
grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_B*/run.log 2>/dev/null

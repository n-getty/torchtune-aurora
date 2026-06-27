#!/bin/bash
# Confirm the IPC-handle mechanism: single-node HSDP shard=2 (each FSDP group = 2 ranks,
# far fewer IPC handles) vs the 12-tile full-shard that banned:1'd at step 3.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=D0_hsdp_shard2_seq8192 NPROC=12 STEPS=12 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192 data_parallel_shard_dim=2 data_parallel_replicate_dim=6" bash $R || true
echo "=== HSDP TEST DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_D*/run.log 2>/dev/null

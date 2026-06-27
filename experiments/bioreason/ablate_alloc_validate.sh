#!/bin/bash
# Validate the USM-allocator fix at the EXACT config that banned:1'd at step 3:
# 12 tiles, seq=8192, SDPA on. Run 12 steps — surviving past step ~3-6 (the
# MR-accumulation crash window) = fix confirmed. Memory logs will show 0 (allocator
# monkeypatches the query API) — judge by survival + step time, not the mem number.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
TAG=G0_alloc_12t_seq8192 NPROC=12 STEPS=12 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192" bash $R || true
echo "=== ALLOC VALIDATE DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_G0*/run.log 2>/dev/null

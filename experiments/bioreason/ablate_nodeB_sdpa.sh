#!/bin/bash
# Node B: the SDPA-fix validation at the config that OOM'd before (seq=8192, 12 tiles).
# Then an A/B: same config with the legacy math attention (expect OOM) to prove causation.
set -eo pipefail
R=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
# C0: SDPA ON (default), full node, seq=8192 — should NOT OOM now.
TAG=C0_sdpa_seq8192_12t NPROC=12 STEPS=20 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192" bash $R || true
# C1: SDPA ON, seq=8192, batch_size=2 — now that memory is O(S), can we batch?
TAG=C1_sdpa_seq8192_bs2 NPROC=12 STEPS=20 OVERRIDES="tokenizer.max_seq_len=8192 dataset.max_seq_len=8192 batch_size=2" bash $R || true
echo "=== NODE B (SDPA) DONE ==="; grep -h ABLATE_RESULT /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/ablate_C*/run.log 2>/dev/null

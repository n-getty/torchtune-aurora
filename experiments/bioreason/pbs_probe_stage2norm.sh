#!/bin/bash
#PBS -N br_probe_s2n
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/probe_stage2norm.out
#
# ★ RUN 2 PROBE GATE. Isolate whether Stage-2 backbone (LoRA @ lr 2e-5) on the CLEAN LayerNorm
# projector collapses to ':' at generation (the prior Stage-2 did, on the OLD unbounded projector)
# or reasons coherently. Decode-variant x splice probe on the FIRST stage2norm checkpoint.
# If A/B/C (splice) collapse to ':' / loops -> STOP Run 2 (LayerNorm didn't save Stage-2; fall
# back to the 0.6008 stage1norm as shippable). If coherent -> let Run 2 run to walltime.
# 32B via HF device_map across 2 tiles (one shard, one process).
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
module load frameworks
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$TT
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
cd $TT

CKPT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_stage2norm/epoch_0
TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048_l37.safetensors

# 2 tiles for the device_map shard. max_new_tokens 200 (not 80) so we can see whether the
# reasoning REACHES a GO answer vs over-reasons/loops (Run 1's failure mode).
ZE_AFFINITY_MASK=0,1 python experiments/bioreason/probe_collapse.py \
  --ckpt_dir "$CKPT" --proj_dir "$CKPT" \
  --esm3_cache_path "$CACHE" --local_parquet "$TEST" \
  --protein_token_id 151643 --go_token_id 151644 \
  --num_go_tokens 200 --max_protein_len 2048 --n 3 --max_new_tokens 200
echo "=== probe done $(date) ==="

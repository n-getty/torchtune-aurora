#!/usr/bin/env bash
# SFT GPU stress test on Qwen/Qwen3.6-27B, restricted to physical GPUs 2 and 3
# only (per explicit instruction), to get an independent real-training-shaped
# fault signal on a brand new model architecture (hybrid Gated-DeltaNet linear
# attention + full attention, Qwen3_5ForConditionalGeneration) distinct from
# the BioReason/Qwen3-32B recipes already run on this node many times.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3
VENV=/raid/ngetty/nemo_rl_work/nemo-rl/.venv
SCRIPT=/raid/ngetty/nemo_rl_work/sft_qwen3_6_27b_gpu_stress.py
LOG=/raid/ngetty/nemo_rl_work/qwen3_6_27b_gpu23_stress.log
OUT_JSON=/raid/ngetty/nemo_rl_work/qwen3_6_27b_gpu23_stress.json

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== GPU state before launch (physical 2,3 only pinned via CUDA_VISIBLE_DEVICES) ===" | tee "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$LOG"

echo "=== Launching torchrun --nproc_per_node=2 (maps to physical GPUs 2,3) ===" | tee -a "$LOG"
"$VENV/bin/torchrun" --standalone --nproc_per_node=2 "$SCRIPT" \
  --steps 60 \
  --batch_size 1 \
  --grad_accum 4 \
  --max_seq_len 1024 \
  --lora_rank 16 \
  --out_json "$OUT_JSON" \
  2>&1 | tee -a "$LOG"

echo "=== GPU state after run ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$LOG"

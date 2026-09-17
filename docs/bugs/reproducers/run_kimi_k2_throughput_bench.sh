#!/usr/bin/env bash
# Heavy throughput benchmark: load Kimi-K2-Thinking (moonshotai, ~1T-param
# DeepSeek-V3-arch MoE, INT4 compressed-tensors quant, 594GB on disk) across
# all 8 H200 NVL GPUs (TP=8) on rbdgx3 and run vLLM's offline throughput
# benchmark. Purpose: heavy sustained multi-GPU load, matching the shape of
# workload that was running (another user's vLLM + Kimi K2) when the
# intermittent "busy or unavailable" GPU fault was first observed on this
# node -- a real, heavy, multi-GPU vLLM job is a better stress condition
# than a synthetic matmul probe.
#
# Weights are read-only from another user's world-readable cache
# (/raid/mcashdollar/model_cache/moonshotai/Kimi-K2-Thinking) -- nothing is
# written there, no download needed.
set -euo pipefail

MODEL_DIR=/raid/mcashdollar/model_cache/moonshotai/Kimi-K2-Thinking
VENV=/raid/ngetty/nemo_rl_work/nemo-rl/.venv
LOG=/raid/ngetty/nemo_rl_work/kimi_k2_throughput_bench.log

export VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Compat shim: Kimi-K2's vendored tokenization_kimi.py imports
# bytes_to_unicode from a transformers path that moved in this installed
# transformers version (5.5.0). See kimi_tok_patch/sitecustomize.py.
export PYTHONPATH="/raid/ngetty/nemo_rl_work/kimi_tok_patch:${PYTHONPATH:-}"

echo "=== GPU state before launch ===" | tee "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$LOG"

echo "=== Launching vLLM offline throughput benchmark (TP=8) ===" | tee -a "$LOG"
"$VENV/bin/vllm" bench throughput \
  --model "$MODEL_DIR" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 512 \
  --num-prompts 200 \
  --max-num-seqs 64 \
  2>&1 | tee -a "$LOG"

echo "=== GPU state after run ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a "$LOG"

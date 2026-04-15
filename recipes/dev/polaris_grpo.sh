#!/bin/bash
# Polaris A100 GRPO benchmark launcher
# Usage: ssh <polaris_compute_node> bash /path/to/torchtune-aurora/recipes/dev/polaris_grpo.sh [model] [config]
#
# Arguments:
#   model:  8b (default) | 32b
#   config: A (default) | B
#
# Examples:
#   bash polaris_grpo.sh 8b A    # Qwen3-8B Config A (4 samples, 256 tokens)
#   bash polaris_grpo.sh 8b B    # Qwen3-8B Config B (16 samples, 512 tokens)
#   bash polaris_grpo.sh 32b A   # Qwen3-32B Config A (requires 2 nodes)
#
# For 2-node runs, use torchrun directly:
#   torchrun --nnodes 2 --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint $MASTER:29500 \
#     recipes/dev/grpo_full_finetune_distributed.py --config recipes/configs/dev/qwen32B_grpo_a100_configA.yaml
set -euo pipefail

MODEL=${1:-8b}
CONFIG=${2:-A}
NPROC=${3:-4}

echo "=== Polaris GRPO Benchmark ==="
echo "Model: Qwen3-${MODEL^^}"
echo "Config: ${CONFIG^^}"
echo "GPUs: ${NPROC}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Environment setup
module use /soft/modulefiles
module load conda/2025-09-25
conda activate base
export PATH=$HOME/.local/polaris/conda/2025-09-25/bin:$PATH
export HF_HOME=/eagle/argonne_tpc/ngetty/models
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
ulimit -c 0

# Verify torchtune
TORCHTUNE_DIR="/home/ngetty/proj/torchtune"
cd "$TORCHTUNE_DIR"
python3 -c "import torchtune; print('torchtune OK')" || {
    echo "Installing torchtune..."
    pip install -e "$TORCHTUNE_DIR"
}
python3 -c "import math_verify; print('math_verify OK')" 2>/dev/null || {
    echo "Installing math_verify..."
    pip install math_verify
}

# Model download
if [ "${MODEL,,}" = "8b" ]; then
    MODEL_PATH="/tmp/Qwen3-8B"
    HF_MODEL="Qwen/Qwen3-8B"
    CONFIG_FILE="dev/qwen8B_grpo_a100_config${CONFIG^^}"
elif [ "${MODEL,,}" = "32b" ]; then
    MODEL_PATH="/tmp/Qwen3-32B"
    HF_MODEL="Qwen/Qwen3-32B"
    CONFIG_FILE="dev/qwen32B_grpo_a100_config${CONFIG^^}"
elif [ "${MODEL,,}" = "3b" ]; then
    MODEL_PATH="/tmp/Qwen2.5-3B"
    HF_MODEL="Qwen/Qwen2.5-3B"
    CONFIG_FILE="dev/qwen3B_sync_grpo"
else
    echo "Unknown model: $MODEL (use 3b, 8b, or 32b)"
    exit 1
fi

if [ -d "$MODEL_PATH" ] && [ "$(ls -1 "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Model found at $MODEL_PATH"
else
    echo "Downloading ${HF_MODEL} to ${MODEL_PATH}..."
    huggingface-cli download "$HF_MODEL" --local-dir "$MODEL_PATH"
fi

echo ""
echo "=== Starting GRPO training ==="
echo "Config: $CONFIG_FILE"
echo ""

# Launch
tune run --nproc_per_node "$NPROC" dev/grpo_full_finetune_distributed \
    --config "$CONFIG_FILE"

echo ""
echo "=== Benchmark complete ==="
echo "End: $(date)"

#!/bin/bash -l
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -N grpo_bench_polaris
#PBS -j oe
#
# Polaris GRPO throughput benchmark (TRL GRPOTrainer, text-proxy for BioReason-Pro RL).
# 1 node = 4 A100-40GB. Companion to sft_bench/run_polaris.sh.
#
# Env knobs (override at qsub time with -v VAR=val,VAR2=val2):
#   GEN=hf|vllm   NPROC=4   MICRO=2   G=8   MAXGEN=1024   PROMPTLEN=2048
#   STEPS=20   WARMUP=5   GC=1   VLLM_MEM=0.3   BETA=0.04   TAG=...
#
# Multi-node: submit with -l select=2:system=polaris ; the script auto-detects nodes from
# $PBS_NODEFILE and drives torch.distributed.run with rdzv across them.
set -eo pipefail

cd "$PBS_O_WORKDIR" 2>/dev/null || cd /lus/eagle/projects/ModCon/ngetty/grpo_bench

module use /soft/modulefiles
module load conda
# trl-bench env: trl 1.0.0 + transformers 4.57.6 + torch 2.10+cu128 + vLLM (GRPOTrainer)
# It is a prefix conda env -> activate by path (NOT `source bin/activate`).
conda activate /home/ngetty/polaris-envs/trl-bench

GEN=${GEN:-hf}
NPROC=${NPROC:-4}
MICRO=${MICRO:-2}
G=${G:-8}
MAXGEN=${MAXGEN:-1024}
# Faithful backbone prompt length: ~2048 protein_pad + 200 GO_graph_pad + ~50 text.
# Matches the real BioReason prompt S reaching the Qwen3 backbone (throughput depends on
# sequence length, not on whether a row is a projected ESM3 vector or a token-id lookup).
PROMPTLEN=${PROMPTLEN:-2300}
STEPS=${STEPS:-20}
WARMUP=${WARMUP:-5}
GC=${GC:-1}
VLLM_MEM=${VLLM_MEM:-0.3}
BETA=${BETA:-0.04}
TAG=${TAG:-polaris_grpo_${GEN}_g${G}}

BENCH=/lus/eagle/projects/ModCon/ngetty/grpo_bench/bench_grpo.py
MODEL=/lus/eagle/projects/ModCon/ngetty/models/Qwen3-4B
OUT=/lus/eagle/projects/ModCon/ngetty/grpo_bench/results/${TAG}.json
mkdir -p /lus/eagle/projects/ModCon/ngetty/grpo_bench/results

export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
# vLLM on Polaris: keep it from re-grabbing all GPUs; colocate shares the trainer's device.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Polaris GRPO bench: GEN=$GEN NPROC=$NPROC MICRO=$MICRO G=$G MAXGEN=$MAXGEN PROMPTLEN=$PROMPTLEN STEPS=$STEPS TAG=$TAG ==="
python -c "import torch,transformers,trl;print('torch',torch.__version__,'transformers',transformers.__version__,'trl',trl.__version__)"

COMMON_ARGS=(
  --model-path "$MODEL"
  --gen "$GEN"
  --prompt-len "$PROMPTLEN"
  --num-generations "$G"
  --max-completion-length "$MAXGEN"
  --micro-bsz "$MICRO"
  --grad-checkpoint "$GC"
  --vllm-gpu-mem "$VLLM_MEM"
  --beta "$BETA"
  --steps "$STEPS" --warmup-steps "$WARMUP"
  --tag "$TAG" --out "$OUT"
)

# --- node topology from PBS ---
NODEFILE=${PBS_NODEFILE:-}
if [ -n "$NODEFILE" ] && [ -f "$NODEFILE" ]; then
  NNODES=$(sort -u "$NODEFILE" | wc -l)
else
  NNODES=1
fi
echo "=== NNODES=$NNODES ==="

# Use `python -m torch.distributed.run`, NOT `torchrun`: the conda torchrun shebang would
# launch stock conda packages even with the venv active.
if [ "$NNODES" -le 1 ]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$BENCH" "${COMMON_ARGS[@]}"
else
  HEAD=$(sort -u "$NODEFILE" | head -1)
  PORT=29521
  RANK=0
  for host in $(sort -u "$NODEFILE"); do
    ssh "$host" "cd $PWD && \
      module use /soft/modulefiles && module load conda && \
      conda activate /home/ngetty/polaris-envs/trl-bench && \
      OMP_NUM_THREADS=$OMP_NUM_THREADS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
      python -m torch.distributed.run \
        --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$NPROC \
        --rdzv_backend=c10d --rdzv_endpoint=$HEAD:$PORT \
        $BENCH ${COMMON_ARGS[*]}" &
    RANK=$((RANK + 1))
  done
  wait
fi

echo "=== done; result at $OUT ==="

#!/bin/bash
#PBS -N t211_F5_e2e
#PBS -l select=2
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A ModCon
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey/F5_lora_grpo_e2e.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey/F5_lora_grpo_e2e.err
#
# F5: LoRA-GRPO 4B GSM8K E2E smoke on torch211 venv as rollout.
# 2-node hold required (1 train + 1 vLLM).
# Reuses the validated 20-step GSM8K wrapper.
#
# Pass criteria (subset of full validation):
#   1. exit=0 over the steps that fit in the hold (target ≥10)
#   2. VALMET adapter_l2 delta non-zero each step
#   3. ratios near 1.0
#   4. NO banned:1 in vLLM logs across all steps
#   5. No regressions in step time vs the established 4B 2-node baseline (~50-65s/step at G=8)

set -o pipefail
TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${TT_DIR}/experiments/lora_grpo/t211_survey/F5_e2e_${TS}.log"

echo "Hold nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
echo "Job start: $(date)"
echo "Log: ${LOG}"

cp "${PBS_NODEFILE}" "/tmp/torchtune_lora_nodefile_${PBS_JOBID%%.*}"

# Same envelope as rung 3 GSM8K validation, but force merged-weight publish path
# (LORA_USE_RUNTIME=0) so we exercise the exact path that's now the default.
# torch211 venv is what the runner picks regardless because the merged-weight
# path doesn't depend on --enable-lora; the difference vs F1 is that this
# actually trains and broadcasts updated weights step-after-step.

export CONFIG="recipes/configs/dev/production/qwen3_4b_lora_grpo_2node_server_xpu.yaml"
export MODEL_PATH="/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B"
export NSTEPS=12
export GRPO_SAMPLES=8
export FORWARD_BATCH_SIZE=4
export GEN_BATCH_SIZE=8
export MAX_GEN_TOKENS=512
export LORA_RANK=16
export LORA_MAX_LORAS=2
export VLLM_MAX_MODEL_LEN=1536
export VLLM_MAX_NUM_SEQS=32
export VLLM_GPU_MEM=0.85
export VLLM_STARTUP_TIMEOUT=600
export ADAPTER_ROOT="${TT_DIR}/outputs/lora_grpo_qwen3_4b_t211_F5_${TS}/adapters"

# Merged-weight publish path (the new default; vLLM boots WITHOUT --enable-lora).
# This is what F1-F4 implicitly validate at the vLLM-only level; F5 closes the loop
# by exercising the trainer's broadcast handshake step-after-step.
export LORA_USE_RUNTIME=0

exec bash "${TT_DIR}/experiments/lora_grpo/run_qwen3_4b_lora_2node.sh" 2>&1 | tee "${LOG}"

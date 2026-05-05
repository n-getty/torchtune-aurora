#!/bin/bash -l
# Chain F1 → F2 → F3 → F4 sequentially on a single hold node.
# Each script is self-contained and cleans up its own vLLM processes.
ROOT=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/lora_grpo/t211_survey
CHAIN_LOG=${ROOT}/F1_to_F4_chain_$(date +%Y%m%d_%H%M%S).log
exec > "${CHAIN_LOG}" 2>&1

echo "==== CHAIN START $(date) host=$(hostname) ===="

for T in F1_defaults_ab F2_lora_soak F3_tp2_dense F4_long_ctx; do
    echo ""
    echo ">>>> Running ${T} at $(date)"
    bash ${ROOT}/${T}.sh
    RC=$?
    echo "<<<< ${T} done rc=${RC} at $(date)"
done

echo ""
echo "==== CHAIN COMPLETE $(date) ===="
ls -lat ${ROOT}/F[1-4]_* 2>&1 | head -20

#!/bin/bash
# Shared env for AuroraGPT-2B bake-off torchtune launchers.
# Source from run_torchtune.sh / 1N_smoke.sh / sweep_envelope.sh.

REPO_ROOT="/lus/flare/projects/ModCon/ngetty/torchtune"
EXPDIR="${REPO_ROOT}/experiments/auroragpt_2b_bakeoff"
CONFIG="recipes/configs/dev/production/auroragpt_2b_grpo_4n_colocate_xpu.yaml"
MODEL_SRC="/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650"
MODEL_STAGED="/tmp/torchtune/AuroraGPT-2B"

REWARD_FOR_TASK() {
    case "$1" in
        sum_digits)  echo "torchtune.dev.rl.ezpz_tasks.SumDigitsReward" ;;
        multiply)    echo "torchtune.dev.rl.ezpz_tasks.MultiplyReward" ;;
        word_sort)   echo "torchtune.dev.rl.ezpz_tasks.WordSortReward" ;;
        countdown)   echo "torchtune.dev.rl.ezpz_tasks.CountdownReward" ;;
        arithmetic)  echo "torchtune.dev.rl.ezpz_tasks.ArithmeticReward" ;;
        *) echo "UNKNOWN_TASK_${1}"; return 1 ;;
    esac
}

setup_aurora_env() {
    module load frameworks 2>/dev/null || true
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
    unset VIRTUAL_ENV

    # Single-node standalone row (CLAUDE.md launcher decision table) — used by
    # both 1N smoke and the per-node torchrun fan-out in 4N runs.
    export CCL_PROCESS_LAUNCHER=none
    export CCL_ATL_TRANSPORT=ofi
    export CCL_OP_SYNC=1
    export CCL_WORKER_COUNT=1
    export CCL_KVS_IFACE=hsn0
    export FI_PROVIDER=cxi
    export FI_CXI_RX_MATCH_MODE=hybrid
    export FI_CXI_OFLOW_BUF_SIZE=8388608
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_MR_CACHE_MONITOR=userfaultfd
    export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT

    # Validated fast paths (caller-overridable: a pre-set value wins, so an A/B
    # launcher can disable varlen to isolate the _varlen_out_cache retention).
    export TORCHTUNE_USE_IPEX_VARLEN="${TORCHTUNE_USE_IPEX_VARLEN:-1}"
    export TORCHTUNE_MASKFREE_CAUSAL="${TORCHTUNE_MASKFREE_CAUSAL:-1}"
    export TORCHTUNE_PINNED_CPU_BUF="${TORCHTUNE_PINNED_CPU_BUF:-1}"

    # Offline HF
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1

    # PYTHONPATH (recipe + vLLM usercustomize)
    local VLLM_CUSTOMIZATION="${REPO_ROOT}/recipes/dev/_usercustomize_vllm"
    export PYTHONPATH="${REPO_ROOT}:${VLLM_CUSTOMIZATION}:${PYTHONPATH:-}"

    export TORCHDYNAMO_DISABLE=1
}

stage_model() {
    mkdir -p /tmp/torchtune
    if [ ! -f "${MODEL_STAGED}/config.json" ]; then
        echo "Staging AuroraGPT-2B to /tmp ($(date))..."
        local t0=$SECONDS
        cp -r "${MODEL_SRC}" "${MODEL_STAGED}"
        echo "  staged in $((SECONDS - t0))s"
    else
        echo "Model already staged at ${MODEL_STAGED}"
    fi
}

#!/bin/bash
# Standalone-reproducer crash-rate driver for the colocate vLLM generation page fault.
#
# Runs scratch/repro_colocate_pagefault.py on all NTILES tiles IN PARALLEL (one process per
# tile, ZE_AFFINITY_MASK=0..NTILES-1) for ROUNDS rounds -> N = NTILES*ROUNDS independent
# trials per rung, fast. A trial CRASHED if its process exited nonzero OR never printed the
# REPRO_DONE line OR a banned:1/PDE/UR_RESULT signature appeared in its stderr.
#
# This is single-node; the reproducer is single-tile so no mpiexec/CCL world is needed —
# just module env + per-process affinity. Run it ON a held compute node (SSH in, then bash).
#
# Usage (on a held node):
#   RUNG=R-A bash experiments/colocate/run_repro_ladder.sh
#   RUNG=R-D MAX_GEN=768 ROUNDS=2 bash experiments/colocate/run_repro_ladder.sh
#   # full ladder:
#   for r in R-A R-B R-C R-D R-E; do RUNG=$r bash experiments/colocate/run_repro_ladder.sh; done
# NOTE: no `set -u` — sourcing _env.sh / `module load` trips unbound-var refs (repo-known trap).
set -o pipefail

REPO_ROOT="/lus/flare/projects/ModCon/ngetty/torchtune"
source "${REPO_ROOT}/experiments/auroragpt_2b_bakeoff/_env.sh"
setup_aurora_env
# _env.sh reassigns EXPDIR (to the bakeoff dir) — pin OURS AFTER sourcing (config-drift trap).
EXPDIR="${REPO_ROOT}/experiments/colocate"

# --- knobs ---
RUNG="${RUNG:-R-A}"
MODEL="${MODEL:-/lus/flare/projects/ModCon/ngetty/models/Qwen3-4B}"
NTILES="${NTILES:-12}"
ROUNDS="${ROUNDS:-1}"
MAX_GEN="${MAX_GEN:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1536}"
GPU_MEM="${GPU_MEM:-0.35}"
GRPO="${GRPO:-8}"
ITERS="${ITERS:-60}"
BURST_EVERY="${BURST_EVERY:-4}"
EXTRA_REPRO_ARGS="${EXTRA_REPRO_ARGS:-}"   # e.g. "--enable-prefix-cache" or "--fsdp on"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${EXPDIR}/repro_logs/${RUNG}_${TS}"
mkdir -p "${RUN_DIR}"
RESULTS_TSV="${EXPDIR}/repro_results.tsv"
NODE="$(hostname)"

echo "=== repro ladder ${RUNG} on ${NODE} ==="
echo "model=${MODEL} ntiles=${NTILES} rounds=${ROUNDS} max_gen=${MAX_GEN} mml=${MAX_MODEL_LEN} iters=${ITERS} burst_every=${BURST_EVERY}"
echo "logs -> ${RUN_DIR}"

# Snapshot dmesg watermark so we attribute only NEW GPU faults to this run.
DMESG_BEFORE="$(dmesg 2>/dev/null | wc -l)"

crashed=0
total=0
declare -a steps_to_crash=()

for round in $(seq 1 "${ROUNDS}"); do
    echo "--- round ${round}/${ROUNDS} : launching ${NTILES} tiles in parallel ---"
    pids=()
    for tile in $(seq 0 $((NTILES - 1))); do
        log="${RUN_DIR}/r${round}_tile${tile}.log"
        ( ZE_AFFINITY_MASK="${tile}" \
          python3 "${REPO_ROOT}/scratch/repro_colocate_pagefault.py" \
            --model "${MODEL}" --rung "${RUNG}" \
            --max-gen "${MAX_GEN}" --max-model-len "${MAX_MODEL_LEN}" \
            --gpu-mem "${GPU_MEM}" --grpo "${GRPO}" \
            --iters "${ITERS}" --burst-every "${BURST_EVERY}" \
            ${EXTRA_REPRO_ARGS} > "${log}" 2>&1 ; echo $? > "${log}.rc" ) &
        pids+=($!)
    done
    wait "${pids[@]}"

    for tile in $(seq 0 $((NTILES - 1))); do
        log="${RUN_DIR}/r${round}_tile${tile}.log"
        # tr -d '\n' collapses any multiline grep/echo output to a single token so the
        # numeric comparisons below don't misfire (false-positive CRASH on clean runs).
        rc="$(head -1 "${log}.rc" 2>/dev/null | tr -dc '0-9' )"; rc="${rc:-999}"
        total=$((total + 1))
        done_line="$(grep -c 'REPRO_DONE' "${log}" 2>/dev/null | head -1 | tr -dc '0-9')"; done_line="${done_line:-0}"
        sig="$(grep -ciE 'banned:1|NotPresent|UR_RESULT_ERROR|Segmentation fault from GPU' "${log}" 2>/dev/null | head -1 | tr -dc '0-9')"; sig="${sig:-0}"
        if [[ "${rc}" != "0" || "${done_line}" == "0" || "${sig}" != "0" ]]; then
            crashed=$((crashed + 1))
            last_iter="$(grep -oE 'iter=[0-9]+' "${log}" 2>/dev/null | tail -1 | cut -d= -f2)"
            steps_to_crash+=("${last_iter:-0}")
            echo "  tile ${tile}: CRASH (rc=${rc} done=${done_line} sig=${sig} last_iter=${last_iter:-?})"
        else
            echo "  tile ${tile}: clean"
        fi
    done
done

# Median steps-to-crash (rough).
med="na"
if [[ ${#steps_to_crash[@]} -gt 0 ]]; then
    sorted=($(printf '%s\n' "${steps_to_crash[@]}" | sort -n))
    med="${sorted[$((${#sorted[@]} / 2))]}"
fi

DMESG_AFTER="$(dmesg 2>/dev/null | wc -l)"
NEW_FAULTS="$(dmesg 2>/dev/null | tail -n $((DMESG_AFTER - DMESG_BEFORE)) | grep -ciE 'banned|GPU HANG|catastrophic' 2>/dev/null | head -1 | tr -dc '0-9')"; NEW_FAULTS="${NEW_FAULTS:-0}"

echo "=== RESULT ${RUNG}: crashed=${crashed}/${total} median_steps_to_crash=${med} new_dmesg_faults=${NEW_FAULTS} ==="
printf '%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n' \
    "${TS}" "${NODE}" "${RUNG}" "${crashed}" "${total}" "${med}" \
    "mg${MAX_GEN}" "iters${ITERS}" "${EXTRA_REPRO_ARGS:-none}" >> "${RESULTS_TSV}"
echo "appended to ${RESULTS_TSV}"

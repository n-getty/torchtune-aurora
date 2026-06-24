#!/bin/bash
# BioReason 4B GRPO — SINGLE-NODE per-rank COLOCATE (in-process TP=1 vLLM per tile).
#
# Runs recipes/dev/grpo_bioreason_distributed_xpu.py with vllm_mode=colocate_sleep
# on ONE node: each of NTILES ranks runs its own FSDP trainer + in-process vLLM
# engine; vLLM SLEEPS during the grpo backward and wakes for generation so its
# ~24 GiB and the ~60 GiB train peak don't co-reside. With PEFT-LoRA the recipe
# merges W_eff per-rank (_sync_colocated_lora_weights) into each rank's engine.
# 1 node, no dedicated vLLM node, no cross-node HTTP. frameworks/2025.3.1 stack.
#
# LAUNCH MODEL: this script runs FROM THE LOGIN/UAN NODE and SSHes into the
# compute node to run `torch.distributed.run --standalone` (single-node, interactive
# row of the CLAUDE.md launcher table: CCL_PROCESS_LAUNCHER=none / ofi). It does
# NOT use mpiexec --pmi=pmix — that fails from an SSH session ("Couldn't send RPC
# launch ... Resource temporarily unavailable" / 17006); see
# memory/feedback_mpiexec_pals_ssh.md. Training is single-node so standalone is correct.
#
# Usage (from the login node, after a 1-node hold is R, PBS_NODEFILE from exec_host):
#   export PBS_NODEFILE=/path/to/real/nodefile   # one compute hostname
#   nohup bash experiments/bioreason/run_bioreason_colocate_1node.sh > colo.log 2>&1 &
#
# Env overrides: CONFIG, MODEL_SRC, NSTEPS, NTILES (default 12), MEM_PROBE=1,
#   EXTRA_OVERRIDES (verbatim Hydra args).
set -o pipefail

TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
THIS_DIR="${TT_DIR}/experiments/bioreason"

CONFIG="${CONFIG:-recipes/configs/dev/production/bioreason_4b_lora_grpo_colocate_xpu.yaml}"
MODEL_SRC="${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft}"
MODEL_PATH="${MODEL_PATH:-/tmp/torchtune/$(basename "${MODEL_SRC}")}"
NSTEPS="${NSTEPS:-10}"
NTILES="${NTILES:-12}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
[ "${MEM_PROBE:-0}" = "1" ] && EXTRA_OVERRIDES="${EXTRA_OVERRIDES}"   # flag exported into ssh below

if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "${PBS_NODEFILE}" ]; then
    echo "ERROR: PBS_NODEFILE must point at a valid nodefile (built from exec_host)." >&2
    exit 1
fi
NODE=$(awk 'NF' "${PBS_NODEFILE}" | sort -u | head -1 | cut -d. -f1)
NNODES=$(awk 'NF' "${PBS_NODEFILE}" | sort -u | wc -l)
if [ "${NNODES}" -ne 1 ]; then
    echo "ERROR: colocate is single-node; nodefile has ${NNODES} nodes." >&2
    exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${THIS_DIR}/logs/colocate_${TS}"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/run.log"
TRAIN_LOG="/tmp/torchtune/colocate_train.log"
echo "=== BioReason colocate | config=${CONFIG} | node=${NODE} tiles=${NTILES} nsteps=${NSTEPS}" | tee "${LOG}"

# Build the BioReason train PYTHONPATH (peft from BIOREASON_DEPS + bioreason2 + recipe).
BIOREASON_SRC="${BIOREASON_SRC:-/flare/ModCon/ngetty/BioReason-Pro}"
BIOREASON_DEPS="${BIOREASON_DEPS:-/lus/flare/projects/ModCon/ngetty/bioreason_deps}"

# ── Stage model to node-local /tmp ────────────────────────────────────────────
echo "Staging ${MODEL_SRC} -> ${MODEL_PATH} on ${NODE}..." | tee -a "${LOG}"
ssh "${NODE}" "test -f '${MODEL_PATH}/config.json' || (mkdir -p $(dirname "${MODEL_PATH}") && cp -r ${MODEL_SRC} ${MODEL_PATH})"
ssh "${NODE}" "test -f '${MODEL_PATH}/config.json'" || { echo "FATAL: staging failed" | tee -a "${LOG}"; exit 1; }

# ── Drain any orphan vLLM/EngineCore from prior runs ──────────────────────────
ssh "${NODE}" "bash ${TT_DIR}/recipes/dev/clean_tiles.sh --kill" 2>&1 | tail -3 | tee -a "${LOG}" || true

_MEM_PROBE_EXPORT=""
[ "${MEM_PROBE:-0}" = "1" ] && _MEM_PROBE_EXPORT="export TORCHTUNE_MEM_PROBE=1"

# ── Run training on the compute node via SSH + torch.distributed.run --standalone.
#    Single-node interactive CCL row: none / ofi (NOT pmix — that needs the PBS
#    process tree and fails from SSH). The recipe sets ZE_AFFINITY_MASK per rank
#    from LOCAL_RANK (torchrun-provided), so no per-rank wrapper is needed. ──────
ssh "${NODE}" "
set -e
cd ${TT_DIR}
module load frameworks 2>/dev/null || true
source ${TT_DIR}/recipes/dev/_aurora_paths.sh
export BIOREASON_SRC=${BIOREASON_SRC}
export BIOREASON_DEPS=${BIOREASON_DEPS}
export PYTHONNOUSERSITE=1
export PYTHONPATH=\"${BIOREASON_DEPS}:\$(aurora_pythonpath ${TT_DIR})\"
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=hsn0
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZES_ENABLE_SYSMAN=1
export no_proxy='*'
export NO_PROXY='*'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
${_MEM_PROBE_EXPORT}
python3 -m torch.distributed.run --standalone --nproc_per_node=${NTILES} \
    recipes/dev/grpo_bioreason_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} \
    output_dir=${RUN_DIR}/out \
    ${EXTRA_OVERRIDES} 2>&1 | tee ${TRAIN_LOG}
" 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
echo "=== end rc=${RC} log=${LOG}"

if [ -x "${TT_DIR}/scripts/check_run_health.sh" ]; then
    bash "${TT_DIR}/scripts/check_run_health.sh" "${LOG}" || echo "WARN: check_run_health flagged DEGRADED"
fi
echo "--- colocate LoRA W_eff sync times ---"
grep -oE "colocate LoRA W_eff sync [0-9]+ params in [0-9.]+s" "${LOG}" | tail -5
echo "--- ratios ---"
grep -oE "ratios=[0-9.]+" "${LOG}" | tail -5
exit ${RC}

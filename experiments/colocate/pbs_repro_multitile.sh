#!/bin/bash
#PBS -N mt_repro
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/pbs_repro_multitile.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/pbs_repro_multitile.err
#
# Multi-tile standalone reproducer — 12 ranks/tiles, in-process vLLM + XCCL FSDP per rank.
# The faithful Intel handoff: reproduces the multi-rank co-residence the single-tile ladder
# could not. MUST be a PBS job (mpiexec --pmi=pmix needs the PBS process group).
#
#   qsub experiments/colocate/pbs_repro_multitile.sh                 # full (vLLM+XCCL)
#   qsub -v NO_VLLM=1 experiments/colocate/pbs_repro_multitile.sh    # control: XCCL-only
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
source "$TT/experiments/auroragpt_2b_bakeoff/_env.sh"; setup_aurora_env
# pmix multi-node row (CLAUDE.md decision table) for the XCCL training world.
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216
export MASTER_ADDR=$(hostname -i | awk '{print $1}') MASTER_PORT=29500 TRAIN_MASTER_PORT=29400

NF="$TT/experiments/colocate/mt_nodefile.txt"; sort -u "$PBS_NODEFILE" > "$NF"
NTILES="${NTILES:-12}"; WORLD="${NTILES}"; export WORLD
MODEL="${MODEL:-/tmp/models/Qwen3-4B}"; STEPS="${STEPS:-8}"; MAX_GEN="${MAX_GEN:-768}"
mkdir -p /tmp/models && rsync -a /lus/flare/projects/ModCon/ngetty/models/Qwen3-4B /tmp/models/ 2>&1 | tail -1
EXTRA=""; [ "${NO_VLLM:-0}" = "1" ] && EXTRA="--no-vllm"
[ "${LOAD_REAL:-0}" = "1" ] && EXTRA="${EXTRA} --load-real-weights"
TS=$(date +%Y%m%d_%H%M%S); LOG="$TT/experiments/colocate/repro_logs/MT_${TS}.log"
mkdir -p "$TT/experiments/colocate/repro_logs"
echo "=== multitile repro up $(date) world=${WORLD} steps=${STEPS} mg=${MAX_GEN} no_vllm=${NO_VLLM:-0} ==="

mpiexec --pmi=pmix -n ${WORLD} -ppn ${NTILES} --hostfile "$NF" --no-vni \
    --cpu-bind depth --depth 8 \
    bash "$TT/experiments/colocate/_repro_multitile_wrapper.sh" \
        "$TT/scratch/repro_colocate_pagefault_multitile.py" \
        --model "${MODEL}" --max-gen "${MAX_GEN}" --steps "${STEPS}" \
        ${EXTRA} 2>&1 | tee "$LOG"
RC=$?
echo "=== multitile repro DONE rc=${RC} $(date) ==="
echo "--- per-step ar/gen (look for ar_s explosion then abort) ---"
grep "MTREPRO_STEP" "$LOG" | sed -E 's/.*(rank=[0-9]+ step=[0-9]+ ar_s=[0-9.]+ gen_s=[0-9.]+ free_gib=[0-9.]+)/\1/' | sort
echo "--- fault? ---"
grep -ciE "banned: 1|NotPresent|Segmentation fault from GPU" "$LOG"
echo "--- clean ranks ---"; grep -c "MTREPRO_DONE" "$LOG"

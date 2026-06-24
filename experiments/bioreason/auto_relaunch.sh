#!/bin/bash
# Wait for a held PBS job to start, then launch the BioReason 2-node test against it.
#
# Usage:
#   bash auto_relaunch.sh <PBS_JOBID> [GRPO_SAMPLES] [FORWARD_BATCH_SIZE] [LOG_TAG]
#
# Polls qstat until the job is R, looks up its nodefile, then nohup-launches
# experiments/bioreason/run_bioreason_2node_server.sh against the first node
# in the assigned set. Logs go to experiments/bioreason/run_${LOG_TAG}_<ts>.log.
set -euo pipefail

JOBID=${1:?usage: auto_relaunch.sh JOBID [G] [FBS] [TAG]}
G=${2:-36}
FBS=${3:-2}
TAG=${4:-g${G}_fbs${FBS}_auto}

TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
LAUNCHER=${TT_DIR}/experiments/bioreason/run_bioreason_2node_server.sh

ts() { date -u +%Y%m%d_%H%M%S; }

echo "[$(ts)] auto_relaunch waiting for job ${JOBID}..."
while true; do
    STATE=$(qstat -f "${JOBID}" 2>/dev/null | awk -F'= ' '/job_state/ {print $2}' | tr -d ' \n')
    case "${STATE}" in
        R) break ;;
        ""|F|E)
            echo "[$(ts)] job ${JOBID} not running (state=${STATE:-GONE}); abort"
            exit 1
            ;;
        *) sleep 30 ;;
    esac
done
echo "[$(ts)] job ${JOBID} is R"

# Parse exec_host into unique nodes
NODES=($(qstat -f "${JOBID}" | awk -F'= ' '/exec_host/ {print $2}' | tr '+' '\n' | cut -d/ -f1 | sort -u))
if [[ "${#NODES[@]}" -lt 2 ]]; then
    echo "[$(ts)] need >=2 nodes, got: ${NODES[*]}"
    exit 1
fi
TRAIN_NODE=${NODES[0]}
VLLM_NODE=${NODES[1]}
NODEFILE="/var/spool/pbs/aux/${JOBID}.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"

echo "[$(ts)] TRAIN=${TRAIN_NODE} VLLM=${VLLM_NODE}"
echo "[$(ts)] NODEFILE=${NODEFILE}"

# Pre-launch hygiene: clean any orphan vLLM/python on assigned nodes.
for N in "${NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no "${N}" "pkill -9 VLLM 2>/dev/null; pkill -9 -f vllm.entrypoints 2>/dev/null; pkill -9 -f grpo_bioreason 2>/dev/null; pkill -9 -f torch.distributed.run 2>/dev/null; true" || true
done
sleep 3

LOGFILE=${TT_DIR}/experiments/bioreason/run_${TAG}_$(ts).log
echo "[$(ts)] launching → ${LOGFILE}"

ssh -o StrictHostKeyChecking=no "${TRAIN_NODE}" "PBS_NODEFILE=${NODEFILE} GRPO_SAMPLES=${G} FORWARD_BATCH_SIZE=${FBS} nohup bash ${LAUNCHER} > ${LOGFILE} 2>&1 &"
echo "[$(ts)] launched on ${TRAIN_NODE}; tail -F ${LOGFILE} to watch"

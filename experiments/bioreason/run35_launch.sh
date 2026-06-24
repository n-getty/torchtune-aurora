#!/bin/bash
# Run 35: retry of run 34 — fix absolute path for XPU_USM_ALLOC_SO (PROJDIR not yet defined
#   when the export runs in run_bioreason_dedicated.sh). Same goal: prevent zeMemFree from
#   invalidating XCCL IPC handles at step 1 via usm_caching_alloc_v2.so.
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run35_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

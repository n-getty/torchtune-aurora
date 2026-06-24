#!/bin/bash
# Run 34: 3-step test — usm_caching_alloc_v2.so to prevent zeMemFree from invalidating
#   XCCL IPC handles. gc:0.99 (run 33) didn't work — zeMemFree is called outside gc path.
#   The v2 caching allocator NEVER calls zeMemFree for cached blocks (only on OOM retry).
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run34_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

#!/bin/bash
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run30_chunked_wsync_v2.log

# Fully detach via setsid so it survives SSH disconnect
setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 1 > ${LOG} 2>&1" &
echo "PID=$!"

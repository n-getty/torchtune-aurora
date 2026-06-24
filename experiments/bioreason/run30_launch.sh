#!/bin/bash
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
pkill -9 -f grpo_full_finetune 2>/dev/null || true
sleep 5
cd ${PROJDIR}
nohup bash experiments/bioreason/run_bioreason_dedicated.sh 1 > experiments/bioreason/run30_chunked_wsync_v2.log 2>&1 &
echo "PID=$!"

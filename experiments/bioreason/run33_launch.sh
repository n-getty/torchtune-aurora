#!/bin/bash
# Run 33: 3-step test — gc:0.99 to prevent allocator GC from invalidating XCCL IPC handles
#   - Step 0: weight sync (between-step pool 62 GiB = 97%, GC threshold 99% → no GC)
#   - Steps 1+2: no banned:1 since zeMemFree not called on cached AllGather buffers
#   - Exit: os._exit(0) prevents XCCL teardown SIGSEGV
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run33_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

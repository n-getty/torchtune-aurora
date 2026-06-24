#!/bin/bash
# Run 31: 3-step test
#   - Tests rank0_only=True removal (step 1 FSDP shard stability)
#   - Tests os._exit(0) for dedicated_rank teardown (no SIGSEGV at exit)
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run31_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

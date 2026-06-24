#!/bin/bash
# Run 32: 3-step test — rank0_only=True restored + os._exit(0) teardown fix
#   - Step 0: weight sync at 32s (rank0_only=True memory-efficient)
#   - Steps 1+2: validate no FSDP shard corruption from rank0_only
#   - Exit: validate os._exit(0) prevents XCCL teardown SIGSEGV
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run32_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

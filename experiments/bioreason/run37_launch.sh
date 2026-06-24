#!/bin/bash
# Run 37: 3-step test — pre-warm summon_full_params + fbs=8
#   - fbs=4 (run 36) FAILED: 2× no_sync chunks peak at 59.7 GiB POST-BWD (worse, not better)
#   - Pre-warm puts 7.49 GiB in cache before training loop; FSDP fwd/bwd + weight sync
#     reuse cached block → no new L0 alloc → pool stays ≤54 GiB → GC never fires
#   - Expected: step 0 and step 1 clean (no banned:1/UR:40)
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run37_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

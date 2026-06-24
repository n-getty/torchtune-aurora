#!/bin/bash
# Run 36: 3-step test — fbs=4 (reduce per-tile headroom to prevent allocator GC)
#   - fbs=8 → POST-BWD pool ~54 GiB + 8 GiB wsync = 62 GiB (97%) → GC → zeMemFree → banned:1
#   - fbs=4 → pool ~35 GiB + 8 GiB = ~43 GiB (67%) → GC never fires → no stale IPC handles
#   - TORCHTUNE_USE_CHUNKED_LOSS=1 set in run_bioreason_dedicated.sh
#   - No usm_caching_alloc_v2.so (causes OOM retry → same banned:1 via different path)
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
LOG=${PROJDIR}/experiments/bioreason/run36_3step.log

setsid bash -c "cd ${PROJDIR} && bash experiments/bioreason/run_bioreason_dedicated.sh 3 > ${LOG} 2>&1" &
echo "PID=$!"

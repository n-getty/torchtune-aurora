#!/usr/bin/env bash
#PBS -N kimi_k3_hold_3n
#PBS -l walltime=04:00:00
#PBS -A AuroraGPT
#PBS -q capacity
#PBS -l select=3
#PBS -l place=scatter
#PBS -l filesystems=flare:home:daos_user_fs
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_3n_capacity.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_3n_capacity.err

set -euo pipefail
echo "job=$PBS_JOBID nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"

# daos_agent.service is masked at the systemd level and only started per-job
# by the PBS prologue when filesystems=daos_user_fs is requested. Wait for
# the socket to appear on every node before declaring ready, mirroring
# hold_3node.sh -- without this, a K3 DAOS mount attempted from an
# interactive SSH into this hold fails with DER_AGENT_COMM (-2034) even
# though the job itself is running fine.
export DAOS_AGENT_DRPC_DIR=/run/daos_agent_oneScratch
export D_AGENT_DRPC_DIR=/run/daos_agent_oneScratch
for attempt in $(seq 1 36); do
    missing=0
    for node in $(sort -u "$PBS_NODEFILE"); do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
            'test -S /run/daos_agent_oneScratch/daos_agent.sock'; then
            missing=1
        fi
    done
    if [[ "$missing" == 0 ]]; then
        echo "daos_agent_sockets_pass attempt=$attempt"
        break
    fi
    echo "daos_agent_sockets_wait attempt=$attempt"
    if [[ "$attempt" == 36 ]]; then
        echo "ERROR: daos_user agent socket did not appear on all nodes" >&2
        exit 1
    fi
    sleep 5
done

echo "hold_ready job=$PBS_JOBID time=$(date -Is)"
sleep 14220

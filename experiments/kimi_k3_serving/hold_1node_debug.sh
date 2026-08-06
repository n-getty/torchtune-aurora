#!/usr/bin/env bash
#PBS -N kimi_hold_1n
#PBS -l walltime=01:00:00
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_1n.pbs.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_1n.pbs.err

# Recoverable single-node hold for Kimi-Linear-48B decode iteration.
#
# Rationale: one-shot PBS jobs vanish on failure and force a requeue for every
# fix. Job 8740291 died at worker init (q_lora_rank=None) after waiting for a
# queue slot, and the allocation went with it. A hold keeps the node so the
# next attempt is `ssh <node> run_48b_decode.sh` -- seconds, not a requeue.
#
# The 48B checkpoint lives on Lustre (no DAOS needed at 92 GB), so this hold
# does not request daos_user_fs and does no mounting.
#
# Usage:
#   qsub hold_1node_debug.sh
#   qstat -f <jobid>            # get exec_host
#   ssh <node> bash /lus/.../run_48b_decode.sh <jobid>
#   qdel -W force <jobid>       # ALWAYS; never leave for PBS reclaim

set -euo pipefail
LOG_DIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/holds
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/hold_1n_${PBS_JOBID%%.*}.log") 2>&1

NODE=$(sort -u "$PBS_NODEFILE" | head -1)
echo "hold_ready job=$PBS_JOBID node=$NODE time=$(date -Is)"
echo "run: ssh $NODE bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/run_48b_decode.sh $PBS_JOBID"
echo "stop: qdel -W force $PBS_JOBID"

# Hold just under the 1h walltime so the job ends on its own terms if a
# force-delete is missed, rather than being reclaimed mid-write.
sleep 3540
echo "hold_expiring job=$PBS_JOBID time=$(date -Is)"

#!/usr/bin/env bash
#PBS -N kimi_hold_1n_cap
#PBS -l walltime=03:00:00
#PBS -A AuroraGPT
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l filesystems=flare:home
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_1n_cap.pbs.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/hold_1n_cap.pbs.err

# Single-node CAPACITY hold for reduced-K3 decode iteration.
#
# Why capacity and not debug: debug is max_run=1 per user and is usually
# occupied by another of my workstreams, so the reduced-K3 loop ends up waiting
# on an unrelated job. capacity allows max_run=2 and accepts 1-node jobs
# (resources_min.nodect=1), and a 1-node request backfills far sooner than the
# 16-node capacity jobs already queued. It also buys 3h instead of debug's 1h
# cap, which matters because the last debug hold expired mid-experiment.
#
# The reduced-K3 checkpoint is ~0.75 GiB and generated on-node into /tmp, so
# this hold needs no DAOS mount and no staging.
#
# Usage:
#   qsub hold_1node_capacity.sh
#   qstat -f <jobid> | grep exec_host
#   ssh <node> ...   # generate + serve + probe
#   qdel -W force <jobid>    # ALWAYS; never leave for PBS reclaim

set -euo pipefail
LOG_DIR=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/kimi_k3_serving/logs/holds
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/hold_1n_cap_${PBS_JOBID%%.*}.log") 2>&1

NODE=$(sort -u "$PBS_NODEFILE" | head -1)
echo "hold_ready job=$PBS_JOBID node=$NODE time=$(date -Is)"
echo "stop: qdel -W force $PBS_JOBID"

# Hold just under walltime so the job ends on its own terms if a force-delete
# is missed, rather than being reclaimed mid-write.
sleep 10740
echo "hold_expiring job=$PBS_JOBID time=$(date -Is)"

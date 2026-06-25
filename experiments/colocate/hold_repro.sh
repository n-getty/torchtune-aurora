#!/bin/bash
#PBS -N hold_colocate_repro
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/hold_repro.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/hold_repro.err
#
# Single-node hold on the FREE debug queue for the colocate page-fault investigation.
# Submit:  qsub experiments/colocate/hold_repro.sh
# Then SSH to the assigned node and drive the reproducer / A/B harness from there:
#   ssh <node>
#   bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/run_repro_ladder.sh
#
# NOTE: another agent may hold a debug-scaling job concurrently — that is a DIFFERENT
# queue, so this debug-queue hold does not conflict. Never SSH into a job you did not submit.

echo "=== colocate-repro hold (1 node, debug queue) ==="
echo "Start: $(date)"
echo "Job:   ${PBS_JOBID}"
echo "Node:  $(sort -u "${PBS_NODEFILE}" | tr '\n' ' ')"
echo ""
echo "To drive the investigation:"
echo "  ssh $(sort -u "${PBS_NODEFILE}" | head -1)"
echo "  cd /lus/flare/projects/ModCon/ngetty/torchtune"
echo "  RUNG=R-A bash experiments/colocate/run_repro_ladder.sh"
echo ""
# Hold for the walltime; the driver runs over SSH against this node. qdel when done.
sleep 3300
echo "End: $(date)"

#!/bin/bash
#PBS -N colocate_ab
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/pbs_colocate_ab.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/pbs_colocate_ab.err
#
# In-framework A/B for the colocate generation page fault — runs the REAL 12-rank FSDP+vLLM
# colocate recipe (the only faithful vehicle; the single-tile standalone ladder cannot exercise
# the multi-rank XCCL+FSDP+vLLM co-residence that distinguishes crashing-colocate from
# clean-server-mode). MUST run as a PBS job: run_lora_colocate.sh uses `mpiexec --pmi=pmix`,
# which fails when launched over SSH (needs the PBS process group — feedback_mpiexec_pals_ssh).
#
# Each cell runs M times back-to-back ON THIS NODE (controls node variance), classifies each
# run.log crash-vs-clean, and appends crash-count/M to ab_results.tsv.
#
# Select cells + repeats via -v:  qsub -v CELLS="baseline noreset pub999",M=4,MAX_GEN=768 \
#                                      experiments/colocate/pbs_colocate_ab.sh
set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"

NF="$TT/experiments/colocate/ab_nodefile.txt"
sort -u "$PBS_NODEFILE" > "$NF"
export PBS_NODEFILE="$NF"
echo "=== colocate A/B job up $(date) | jobid=$PBS_JOBID | node=$(cat $NF) ==="

# Stage model node-local (avoids 12x concurrent Lustre reads).
mkdir -p /tmp/models && rsync -a /lus/flare/projects/ModCon/ngetty/models/Qwen3-4B /tmp/models/ 2>&1 | tail -1

CELLS="${CELLS:-baseline noreset pub999 nofsdp}"
export M="${M:-4}"
export MAX_GEN="${MAX_GEN:-768}"
export NSTEPS="${NSTEPS:-15}"
export MODEL_PATH="/tmp/models/Qwen3-4B"

for cell in ${CELLS}; do
    echo "############ CELL=${cell} M=${M} mg=${MAX_GEN} $(date) ############"
    CELL="${cell}" bash "$TT/experiments/colocate/run_colocate_ab.sh"
done

echo "=== A/B job DONE $(date) ==="
echo "--- ab_results.tsv ---"
cat "$TT/experiments/colocate/ab_results.tsv" 2>/dev/null

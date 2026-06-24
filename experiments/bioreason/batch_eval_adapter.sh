#!/bin/bash
# One-shot PBS batch: F_max eval of OUR trained LoRA adapter vs SFT base, 12-tile
# sharded on a single node. Exercises run_eval_adapter_vs_sft.sh (adapter-aware eval
# driver). Early uplift read while the long prod RL run queues on capacity.
#
# ADAPTER_EPOCH defaults to the validated ckptfix soak adapter; override via -v.
#
# debug-scaling caps walltime at 1h. Both legs (sft_base + our_rl) must fit -> N=120
# (12-shard => ~10 proteins/shard/leg, ~20-25min/leg). For the full N=600 eval, run on
# a longer queue or split per-leg.
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_eval_adapter
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/batch_eval_adapter.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"
echo "=== BioReason adapter F_max eval start $(date) job=${PBS_JOBID} ==="
echo "node: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"

export ADAPTER_EPOCH=${ADAPTER_EPOCH:-$TT/outputs/bioreason_hsdp_4n_ckptfix/epoch_0}
export N=${N:-120}        # fits both legs in the 1h debug-scaling cap (early read)
export NSHARDS=${NSHARDS:-12}

bash "$TT/experiments/bioreason/run_eval_adapter_vs_sft.sh"
RC=$?
echo "=== BioReason adapter F_max eval end rc=$RC $(date) ==="
exit $RC

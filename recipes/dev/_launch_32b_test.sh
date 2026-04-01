#!/bin/bash
# Launch 32B multi-node GRPO test from within a held PBS job.
# Must be run on the PBS job's first node via SSH.
# Sources PBS env from the hold job's process to get PBS_JOBCOOKIE for mpiexec.

HOLD_PID=$(pgrep -u $(whoami) -f "sleep 3600" | head -1)
if [[ -z "$HOLD_PID" ]]; then
    echo "ERROR: Cannot find hold job process (sleep 3600). Is the hold job running?"
    exit 1
fi

# Source PBS/PALS environment from the hold job process
eval $(cat /proc/${HOLD_PID}/environ 2>/dev/null | tr '\0' '\n' | grep -E '^PBS_|^PALS_' | sed 's/^/export /')

if [[ -z "$PBS_JOBID" ]]; then
    echo "ERROR: Could not extract PBS_JOBID from hold process"
    exit 1
fi

export PBS_NODEFILE="/var/spool/pbs/aux/${PBS_JOBID}"
export USE_AFFINITY_MASK=1
export MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B
export CONFIG=recipes/configs/dev/qwen32B_grpo_vllm_hsdp_multinode_xpu.yaml
export NSTEPS=2
export VLLM_MAX_MODEL_LEN=1024

# Disable weight sync for initial HSDP test (TP=2 vLLM doesn't have
# /load_weights_from_path/ endpoint — only vllm_serve_xpu.py has it)
# The launch script already sets vllm_weight_sync=false

echo "PBS_JOBID=${PBS_JOBID}"
echo "PBS_JOBCOOKIE=${PBS_JOBCOOKIE:0:8}..."
echo "PBS_NODEFILE=${PBS_NODEFILE}"

cd /lus/flare/projects/ModCon/ngetty/torchtune
bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh

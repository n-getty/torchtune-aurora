# Getting Started — TorchTune Aurora/XPU

This guide covers environment setup and running GRPO training on Aurora HPC (Intel Max Series GPUs).

## Prerequisites

- **Aurora HPC allocation** (PBS project, e.g., `AuroraGPT`)
- Models staged to `/lus/flare/projects/ModCon/ngetty/models/` (Qwen2.5-3B, Qwen3-32B, etc.)

## Environment Setup

### 1. Install (one-time, on a login node)

```bash
module load frameworks
# Remove any user virtualenv that conflicts with frameworks
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

pip install -e /lus/flare/projects/ModCon/ngetty/torchtune
```

`module load frameworks` provides:
- PyTorch 2.10+ with XPU backend
- Python 3.12
- oneCCL/XCCL distributed backend
- Intel Level Zero drivers
- vLLM (bundled in frameworks)

### 2. Verify installation

```bash
# On a compute node (not login)
python -c "import torch; print(torch.xpu.device_count(), 'XPU devices')"
python -c "from torchtune.training import get_xpu_distributed_backend; print(get_xpu_distributed_backend())"
```

## Running GRPO Training

### Quick start — single-node 32B (interactive)

```bash
# Hold a node
qsub -I -l select=1:system=aurora -l walltime=1:00:00 -l filesystems=home:flare -q debug -A AuroraGPT

# On the compute node:
cd /lus/flare/projects/ModCon/ngetty/torchtune
bash recipes/dev/run_grpo_vllm_xpu.sh 2 10 \
    /lus/flare/projects/ModCon/ngetty/models/Qwen3-32B 5 \
    --config recipes/configs/dev/qwen32B_grpo_server_xpu.yaml
```

This launches:
- vLLM server on 2 tiles (TP=2, tiles 10-11)
- 10 FSDP training ranks on tiles 0-9
- Expected: ~18 s/step, ~41 GiB peak memory

### Multi-node 32B HSDP (2 nodes)

```bash
# Hold 2 nodes
qsub -l select=2:system=aurora -l walltime=1:00:00 -l filesystems=home:flare -q debug -A AuroraGPT -o logs/ -e logs/

# Interactive (after holding nodes):
export PBS_JOBID=<jobid>
export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>
bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh

# Or submit directly:
qsub recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh
```

Architecture per node:
- vLLM server on tiles 10-11 (TP=2)
- 10 training ranks on tiles 0-9
- HSDP: FSDP within node (dp_shard=10), DDP across nodes (dp_replicate=2)
- Expected: ~19.4 s/step (near-linear scaling from single-node)

### Smaller models (3B, 8B)

```bash
# 3B single-node with colocated vLLM
bash recipes/dev/run_grpo_colocate_xpu.sh \
    /lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B 5 \
    --config recipes/configs/dev/qwen3B_grpo_colocate_xpu.yaml

# 8B single-node with colocated vLLM (12 tiles)
bash recipes/dev/run_grpo_colocate_xpu.sh \
    /lus/flare/projects/ModCon/ngetty/models/Qwen3-8B 5 \
    --config recipes/configs/dev/qwen8B_grpo_colocate_xpu.yaml
```

## PBS Job Template

```bash
#!/bin/bash
#PBS -l select=N:system=aurora      # N = number of nodes
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug                        # or prod for longer runs
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/logs/job.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/logs/job.err
#PBS -N grpo_training
set -e

cd /lus/flare/projects/ModCon/ngetty/torchtune

# --- Environment ---
module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# --- CCL Configuration (CRITICAL — do not omit) ---
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_WORKER_COUNT=1              # CRITICAL: 4 → 48x AllGather regression
export CCL_ALLREDUCE=ring
# Do NOT set CCL_REDUCE_SCATTER=ring   # Causes 63x regression on multi-node
export CCL_CHUNK_SIZE=16777216
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCH_XPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

# --- Paths ---
export PYTHONPATH="/lus/flare/projects/ModCon/ngetty/torchtune:/flare/ModCon/ngetty/trl:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Launch (adjust for your model/config) ---
# See recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh for full multi-node example
```

## Key Files

| File | Purpose |
|------|---------|
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Main training recipe (~2300 lines) |
| `recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh` | Multi-node launcher (vLLM + HSDP) |
| `recipes/dev/aurora_grpo_vllm_wrapper.sh` | Per-rank wrapper (env, affinity, CCL) |
| `recipes/dev/run_grpo_vllm_xpu.sh` | Single-node vLLM server mode launcher |
| `recipes/dev/run_grpo_colocate_xpu.sh` | Single-node colocated vLLM launcher |
| `recipes/dev/_usercustomize_vllm/` | Runtime patches for vLLM on XPU (see below) |
| `recipes/configs/dev/production/` | Optimized configs (32B, 8B, 3B) |
| `recipes/configs/dev/baseline/` | Reference/comparison configs (A100, XPU baselines) |
| `docs/aurora_rl_baselines.md` | Full benchmarking results and history |

## Overriding Config Values

All YAML config values can be overridden at the command line:

```bash
bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh  # uses defaults

# Override model, steps, tile counts:
MODEL_SRC=/path/to/model NSTEPS=20 NGPUS_PER_NODE=10 VLLM_TILES=2 \
    bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh

# Override config-level params:
bash recipes/dev/aurora_grpo_vllm_hsdp_multinode.sh \
    CONFIG=recipes/configs/dev/production/qwen32B_grpo_server_xpu.yaml
```

## Syncing with Upstream TorchTune

This repo tracks [meta-pytorch/torchtune](https://github.com/meta-pytorch/torchtune) as the `upstream` remote. To incorporate upstream changes:

```bash
git fetch upstream
git rebase upstream/main  # or merge
```

Only 4 upstream files are modified (backend string, XPU util exports, minor fixes), so rebases should be clean.

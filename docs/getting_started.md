# Getting Started — TorchTune Aurora/XPU

This guide covers environment setup and running GRPO training on Aurora HPC (Intel Max Series GPUs).

## Prerequisites

- **Aurora HPC allocation** (PBS project, e.g., `AuroraGPT`)
- Models staged to `/lus/flare/projects/ModCon/ngetty/models/` (Qwen2.5-3B, Qwen3-32B, gemma-4-31B, etc.)

## Environment Setup

### 1. Install (one-time, on a login node)

```bash
cd /path/to/torchtune-aurora
module load frameworks/2025.3.1
# Remove any user virtualenv that conflicts with frameworks
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
# Optional if your TRL checkout is not at the Aurora default
export TRL_DIR=/path/to/trl

pip install -e .
```

`module load frameworks/2025.3.1` provides:
- PyTorch 2.10+ with XPU backend
- Python 3.12
- oneCCL/XCCL distributed backend
- Intel Level Zero drivers
- vLLM 0.10.1 (bundled in frameworks)

> **Note**: Do NOT co-locate vLLM and training on the same node for 32B models with XCCL weight sync.
> The L0 IPC handle cache accumulates ~10.85 GiB of external memory by step 1 backward (707 params × 9+
> peers), leaving insufficient free HBM for step 2. Fix: vLLM on a dedicated separate node (XCCL then
> goes over Slingshot, not L0 IPC). See `docs/bugs/intel_ccl_ipc_handle_accumulation.md`.

### 2. Verify installation

```bash
# On a compute node (not login)
python -c "import torch; print(torch.xpu.device_count(), 'XPU devices')"
python -c "from torchtune.training import get_xpu_distributed_backend; print(get_xpu_distributed_backend())"
```

## Running GRPO Training

> **Weight sync is required.** A run with `vllm_weight_sync=false` uses a stale vLLM model for
> generation — the policy model and generation model diverge immediately. Every step trains on
> off-policy data. This is not valid RL training. All production launchers below have weight sync
> enabled.

### Production 32B — 2-node dedicated vLLM with XCCL weight sync

This is the **only validated 32B configuration with legitimate training** (weight sync enabled,
3+ steps clean on Aurora XPU).

**Architecture:**
- Node 0 (vLLM only): 3 replicas × TP=4 = 12 tiles, `WeightSyncFromFileExtension`
- Node 1 (training only): 12-tile FSDP2 training, torch.distributed.run --standalone
- XCCL weight sync over Slingshot (cross-node, ~6.5 GB/s, ~38s for 32B)
- Weight sync fires every step (configurable via `vllm_weight_sync_interval`)

```bash
# Hold 2 nodes (interactive)
qsub -I -l select=2:system=aurora -l walltime=2:00:00 -l filesystems=home:flare -q debug -A AuroraGPT

# On the login/submit node (or inside the PBS job):
export PBS_NODEFILE=/var/spool/pbs/aux/<jobid>

# Default: VLLM_TP=4, VLLM_DP=3, G=16, 5 steps
bash recipes/dev/aurora_grpo_dedicated_vllm.sh

# Or submit directly as a PBS job:
qsub recipes/dev/aurora_grpo_dedicated_vllm.sh
```

Overridable via environment variables:

```bash
NSTEPS=20 GRPO_SAMPLES=8 MAX_GEN_TOKENS=64 \
    VLLM_TP=4 VLLM_DP=3 \
    MODEL_SRC=/lus/flare/projects/ModCon/ngetty/models/Qwen3-32B \
    bash recipes/dev/aurora_grpo_dedicated_vllm.sh
```

**Expected performance** (G=16, fbs=4, max_gen=128, Qwen3-32B):

| Phase | Time | Notes |
|-------|------|-------|
| Generation | ~20 s | 3×TP4 vLLM replicas |
| GRPO backward | ~14 s | 12-tile FSDP2, `TORCHTUNE_USE_CHUNKED_LOSS=1` |
| XCCL weight sync | ~38 s | 61 GiB at 6.5 GB/s; dominates step time |
| **Total step** | **~72 s** | |

> **Interactive sessions**: For SSH into a held 2-node job (most common for debugging),
> `aurora_grpo_dedicated_vllm.sh` uses `mpiexec --pmi=pmix` which requires the PBS process group
> context. Use `experiments/multinode_32b/test_cd_wsync.sh` instead — it uses SSH + `--standalone`
> which works from any SSH session.

### Quick start — Gemma4-31B single-node (interactive)

```bash
# On a compute node:
cd /path/to/torchtune-aurora
bash recipes/dev/run_gemma4_grpo_vllm.sh 2 10 5
```

This launches:
- vLLM server with Gemma4 overlay on 2 tiles (TP=2, tiles 10-11)
- 10 FSDP training ranks on tiles 0-9
- Uses custom `vllm_gemma4_overlay/` to add Gemma4 support to vLLM 0.10.1

> **Note**: Gemma4's larger head dimensions (local 256, global 512 vs Qwen3's 128) make vLLM generation
> ~60% slower than Qwen3-32B, but training is 31% faster due to K=V architecture. Weight sync for
> single-node Gemma4 co-located (10+2) is subject to the same L0 IPC handle accumulation issue
> as Qwen3-32B; validate step 2+ before production use.

### Smaller models (3B, 8B)

```bash
# 3B single-node with colocated vLLM + SHM weight sync
bash recipes/dev/run_grpo_colocate_xpu.sh \
    12 /lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B 5 \
    recipes/configs/dev/production/qwen3B_grpo_colocate_xpu.yaml

# 8B single-node with colocated vLLM (12 tiles)
bash recipes/dev/run_grpo_colocate_xpu.sh \
    12 /lus/flare/projects/ModCon/ngetty/models/Qwen3-8B 5 \
    recipes/configs/dev/production/qwen8B_grpo_colocate_xpu.yaml
```

Smaller models (3B, 8B) use colocated vLLM on the same node. The IPC handle accumulation issue
is proportional to model parameter count — 3B has ~200 params vs 707 for 32B, so the external
memory footprint is much smaller and single-node colocated works.

## Production Performance Summary

Validated results on Aurora (Intel Max 1550, 64 GiB/tile) **with weight sync enabled**:

| Model | Config | Nodes | Weight sync | Step time | Notes |
|-------|--------|-------|-------------|-----------|-------|
| Qwen2.5-3B | colocated vLLM, 12 tiles | 1 | SHM | ~21 s | Validated Test CI |
| Qwen3-32B | dedicated vLLM 2-node | 2 | XCCL | ~72 s | G=16/fbs=4/max_gen=128; Tests CD/CE |
| Gemma4-26B-A4B | single-node, 12 tiles | 1 | — | ~24 s | MoE; weight sync TBD |

Sync dominates 32B step time: gen=20s, grpo=14s, XCCL=38s. To reduce XCCL overhead:
- Use `vllm_weight_sync_interval=N` to sync every N steps (amortizes 38s over N steps)
- SHM weight sync requires H2D batching fix to beat XCCL at max_gen=64 (see `docs/status.md`)

A100-40GB comparison: 32B+ models are **infeasible** on A100-40GB (OOM in all configs).
Aurora's 64 GiB tiles are required.

## PBS Job Template

```bash
#!/bin/bash
#PBS -l select=N:system=aurora      # N = number of nodes
#PBS -l filesystems=home:flare
#PBS -l walltime=1:00:00
#PBS -q debug                        # or prod for longer runs
#PBS -A AuroraGPT
#PBS -o logs/job.out
#PBS -e logs/job.err
#PBS -N grpo_training
set -e

cd /path/to/torchtune-aurora

# --- Environment ---
module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export TRL_DIR=${TRL_DIR:-/flare/ModCon/ngetty/trl}

# --- CCL Configuration (CRITICAL — do not omit) ---
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
# CRITICAL: default=1000 fills at step 1 backward (707 params × peers) → eviction → banned:1.
# 65536 prevents eviction; for co-located single-node 32B, accumulation then causes OOM at
# step 2 (10.85 GiB external). Fix for 32B: separate vLLM node (XCCL over Slingshot, not L0).
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
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
unset PYTORCH_ALLOC_CONF
export TORCH_COMPILE_DISABLE=1

# RC1 fix: prevents OOM at G=16 with 32B (per-chunk fwd+bwd loop kept unsharded grads live)
export TORCHTUNE_USE_CHUNKED_LOSS=1

# --- Paths ---
export PYTHONPATH="$(pwd)${TRL_DIR:+:${TRL_DIR}}:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- Launch (adjust for your model/config) ---
# See recipes/dev/aurora_grpo_dedicated_vllm.sh for full 32B example
```

## Key Files

| File | Purpose |
|------|---------|
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | Main training recipe (~2300 lines) |
| `recipes/dev/aurora_grpo_dedicated_vllm.sh` | **Primary 32B launcher** (2-node dedicated vLLM, XCCL weight sync) |
| `recipes/dev/aurora_grpo_vllm_wrapper.sh` | Per-rank wrapper (env, affinity, CCL) — used by mpiexec launchers |
| `recipes/dev/run_gemma4_grpo_vllm.sh` | Single-node vLLM server mode launcher (Gemma4) |
| `recipes/dev/run_grpo_colocate_xpu.sh` | Single-node colocated vLLM launcher (3B/8B) |
| `recipes/dev/vllm_gemma4_overlay/` | Gemma4 model support for vLLM 0.10.1 |
| `recipes/dev/_usercustomize_vllm/` | Runtime patches for vLLM on XPU |
| `recipes/configs/dev/experimental/` | Working configs (Qwen3-32B dedicated vLLM, etc.) |
| `recipes/configs/dev/production/` | Smaller model configs (3B, 8B) |
| `torchtune/models/gemma4/` | Native torchtune Gemma4 model module |
| `docs/aurora_rl_baselines.md` | Full benchmarking results and history |
| `experiments/multinode_32b/test_cd_wsync.sh` | Validated 32B XCCL sync test (SSH+standalone) |
| `experiments/multinode_32b/test_ce_wsync_batched.sh` | Validated 32B batched XCCL test |

## Overriding Config Values

All YAML config values can be overridden at the command line:

```bash
bash recipes/dev/aurora_grpo_dedicated_vllm.sh  # uses defaults

# Override steps, samples, generation length:
NSTEPS=20 GRPO_SAMPLES=8 MAX_GEN_TOKENS=64 \
    bash recipes/dev/aurora_grpo_dedicated_vllm.sh
```

## Syncing with Upstream TorchTune

This repo tracks [meta-pytorch/torchtune](https://github.com/meta-pytorch/torchtune) as the `upstream` remote. To incorporate upstream changes:

```bash
git fetch upstream
git rebase upstream/main  # or merge
```

Only 4 upstream files are modified (backend string, XPU util exports, minor fixes), so rebases should be clean.

# Getting Started — TorchTune Aurora/XPU

This guide covers environment setup and running GRPO training on Aurora HPC (Intel Max Series GPUs).

## Prerequisites

- **Aurora HPC allocation** (PBS project, e.g., `AuroraGPT`)
- Models staged to `/lus/flare/projects/ModCon/ngetty/models/` (Qwen2.5-3B, Qwen3-32B, Gemma4-26B, etc.)

## Environment Setup

### 1. Install (one-time, on a login node)

```bash
module load frameworks/2025.3.1
# Remove any user virtualenv that conflicts with frameworks
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

pip install -e /lus/flare/projects/ModCon/ngetty/torchtune
```

`module load frameworks/2025.3.1` provides:
- PyTorch 2.10+ with XPU backend
- Python 3.12
- oneCCL/XCCL distributed backend
- Intel Level Zero drivers
- vLLM 0.10.1 (bundled)

> **Note**: Do NOT co-locate vLLM and training on the same node for 32B models. The L0 IPC
> handle cache accumulates ~10.85 GiB of external memory by step 1 backward, leaving
> insufficient free HBM for step 2. Use separate nodes (3-node config recommended for 32B).
> See `docs/bugs/intel_ccl_ipc_handle_accumulation.md`.

### 2. Verify installation

```bash
# On a compute node (not login)
python -c "import torch; print(torch.xpu.device_count(), 'XPU devices')"
```

## Holding a Node for Interactive Testing

**Always hold a node and SSH in for iterative testing.** Never submit a new job for each attempt.

```bash
# Hold 1 node (1 hour, debug-scaling queue)
qsub -I -l select=1:system=aurora -l walltime=1:00:00 -q debug-scaling -A AuroraGPT

# Or use the provided hold scripts:
bash recipes/dev/hold_node.sh          # 1 node
bash recipes/dev/hold_2nodes.sh        # 2 nodes
bash recipes/dev/hold_3nodes.sh        # 3 nodes
```

After getting a job allocation, SSH directly into the node:

```bash
qstat -u $USER                         # find your running job and its node
ssh <nodename>
```

Always use `nohup` for tests run over SSH (sessions drop after ~10 min):

```bash
nohup bash /lus/flare/.../experiments/run.sh > /lus/flare/.../run.log 2>&1 &
```

## Running GRPO Training

> **Weight sync is required.** Running with `vllm_weight_sync=false` uses a stale vLLM model
> — the policy and generation model diverge immediately. All production launchers below have
> weight sync enabled.

---

### Flagship: Qwen3-32B on 3 nodes (24-way FSDP)

This is the recommended 32B configuration: pure 24-way FSDP across 2 training nodes,
dedicated vLLM on 1 node, deferred gloo weight sync.

**Architecture:**
- Node 0 (vLLM only): TP=4, DP=1, 4 tiles
- Nodes 1-2 (training): 12+12 tile 24-way FSDP, `--standalone`
- Weight sync: deferred gloo cross-PG (training → vLLM over Slingshot) + XCCL intra-PG
- Sync interval=2 (broadcast every other step, fires during generation)

```bash
# Hold 3 nodes
qsub -I -l select=3:system=aurora -l walltime=4:00:00 -q debug-scaling -A AuroraGPT

# From a held session, SSH into the job's first node, then:
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b/run_32b_3node_24way.sh
```

**Expected performance** (Qwen3-32B, G=16, fbs=16, max_gen=128):

| Config | Step time | Memory margin | Status |
|--------|-----------|---------------|--------|
| G=16, max_gen=128 | ~41 s | 10+ GiB | **Best stability** |
| G=32, max_gen=128 | ~53 s | ~5 GiB | **Best throughput** (1.54× samples/s) |
| G=32, max_gen=192 | ~72 s | ~1.5 GiB | Marginal |
| G=48+ | hangs/OOM | — | Blocked (2-chunk rule: G/fbs > 2 triggers CCL explosion) |

Key env vars for 3-node config:
```bash
WSYNC_CROSS_METHOD=gloo     # gloo TCP for cross-node PG (eliminates CXI MR leak)
WSYNC_INTRA_METHOD=xccl     # XCCL for intra-node PG (2.4× faster than gloo)
TORCHTUNE_PINNED_CPU_BUF=1  # 8.5× gather speedup (31s → 3.7s)
TORCHTUNE_USE_CHUNKED_LOSS=1
vllm_weight_sync_interval=2
forward_batch_size=16        # 1 AllGather round (config default says 4; CLI override)
```

---

### Qwen3-32B on 2 nodes (dedicated vLLM, short runs)

2-node config is faster per-step (~43s vs 41s) but accumulates a slow CXI MR leak
(~9 MiB/step) that causes a crash around step 55-80. Use for validation runs (<30 steps).
For production runs (>30 steps), use the 3-node config above.

```bash
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/multinode_32b/run_32b_2hop_production.sh
```

Architecture: Node 0 (vLLM, 3×TP=4=12 tiles) + Node 1 (training, 12-tile FSDP2).
2-hop XCCL weight sync: ~9s per sync.

---

### Single-node: smaller models (3B, 8B, 30B-MoE)

```bash
# Qwen2.5-3B with colocated vLLM + SHM weight sync
bash recipes/dev/run_grpo_vllm_xpu.sh \
    2 10 /lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B 5 \
    recipes/configs/dev/production/qwen3B_grpo_colocate_xpu.yaml

# Qwen3-30B-A3B (MoE) — validated at G=8, ~55s/step
bash recipes/dev/run_grpo_vllm_xpu.sh \
    2 10 /lus/flare/projects/ModCon/ngetty/models/Qwen3-30B-A3B 5 \
    recipes/configs/dev/production/qwen3_30b_a3b_grpo_xpu.yaml

# Gemma4-26B-A4B with vLLM server mode
bash recipes/dev/run_gemma4_grpo_vllm.sh 2 10 5
```

Colocated vLLM (10 training + 2 vLLM tiles) works for 3B and 30B-MoE. The IPC handle
accumulation issue is proportional to parameter count — 3B/30B stay within bounds
single-node; 32B requires separate nodes.

---

### Async GRPO (Qwen3-3B)

Overlaps vLLM generation with training backward pass via `RolloutProducer`.
Phase 1 validated at 22.6s/step (vs 31s synchronous).

```bash
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/async_grpo/run_phase1_async.sh
```

Config: `recipes/configs/dev/production/qwen3B_grpo_async_xpu.yaml`

---

### BioReason multimodal GRPO

2-node setup: training node (11 ranks, FSDP1) + vLLM node (12 HTTP servers, DP=12).
ESM3 protein encoder + GO term encoder + Qwen3-4B backbone. Weight sync via shared FS.

```bash
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/hold_bioreason_2node.sh
```

Config: `recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml`

Phase 2 validated (run 50): 5/5 steps clean, ~58s/step (wsync overlap optimization pending).

---

## Production Performance Summary

| Model | Config | Nodes | Weight sync | Step time |
|-------|--------|-------|-------------|-----------|
| Qwen2.5-3B | 10+2 tiles, SHM, G=16 | 1 | SHM 1.4s (hidden) | ~21 s |
| Qwen3-3B | Async GRPO, G=8 | 1 | SHM | ~23 s |
| Qwen3-30B-A3B (MoE) | 10+2 tiles, SHM, G=8 | 1 | SHM 3.3s | ~55 s |
| Gemma4-26B-A4B | 10+2 tiles, server, G=16 | 1 | HTTP | ~24 s |
| **Qwen3-32B** | **3-node 24-way, G=16** | **3** | **gloo 47s deferred** | **~41 s** |
| Qwen3-32B | 3-node 24-way, G=32 | 3 | gloo 47s deferred | ~53 s |
| BioReason-Pro 4B | 2-node server, 11+12 tiles | 2 | shared-FS | ~58 s |

---

## PBS Job Template

```bash
#!/bin/bash
#PBS -l select=N:system=aurora      # N = number of nodes (1 for 3B/30B, 3 for 32B)
#PBS -l filesystems=home:flare
#PBS -l walltime=4:00:00
#PBS -q debug-scaling               # or prod for longer runs
#PBS -A AuroraGPT
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/<topic>/job.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/<topic>/job.err
#PBS -N grpo_training
set -e

TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
TRL_DIR=/flare/ModCon/ngetty/trl

# --- Environment ---
module load frameworks/2025.3.1 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

# --- CCL Configuration (CRITICAL — do not omit) ---
export CCL_PROCESS_LAUNCHER=pmix          # Requires: mpiexec --pmi=pmix (NOT --standalone)
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536  # CRITICAL: prevents eviction at step 1
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_WORKER_COUNT=1              # CRITICAL: 4 → 48x AllGather regression
export CCL_ALLREDUCE=ring
# Do NOT set CCL_REDUCE_SCATTER=ring   # Causes 63x regression on multi-node
export CCL_CHUNK_SIZE=16777216
export FI_MR_CACHE_MONITOR=disabled
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset PYTORCH_ALLOC_CONF               # expandable_segments:True breaks oneCCL USM check
export TORCH_COMPILE_DISABLE=1
export TORCHTUNE_USE_CHUNKED_LOSS=1   # Prevents per-chunk fwd+bwd OOM (RC1 fix)

# --- Paths ---
export PYTHONPATH="${TT_DIR}${TRL_DIR:+:${TRL_DIR}}:${PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- NOTE: For SSH+standalone (interactive, not mpiexec-launched jobs) ---
# Replace CCL_PROCESS_LAUNCHER=pmix with CCL_PROCESS_LAUNCHER=none
# CCL_ATL_TRANSPORT=ofi (not mpi — pmix env vars hang --standalone)
# See memory/feedback_pmix_envs_break_standalone.md

# --- Launch ---
# See experiments/multinode_32b/run_32b_3node_24way.sh for 32B 3-node example
# See recipes/dev/run_grpo_vllm_xpu.sh for 1-node examples
```

## Key Files

| File | Purpose |
|------|---------|
| `recipes/dev/grpo_full_finetune_distributed_xpu.py` | General GRPO base recipe (~3750 lines) |
| `recipes/dev/grpo_bioreason_distributed_xpu.py` | BioReason multimodal subclass (~1450 lines) |
| `recipes/dev/async_grpo_full_finetune_distributed.py` | Async GRPO (RolloutProducer overlap) |
| `recipes/dev/run_grpo_vllm_xpu.sh` | Single-node vLLM launcher (3B/8B/30B) |
| `recipes/dev/run_gemma4_grpo_vllm.sh` | Gemma4 single-node vLLM server launcher |
| `experiments/multinode_32b/run_32b_3node_24way.sh` | **Primary 32B launcher** (3-node, gloo cross+XCCL intra) |
| `experiments/multinode_32b/run_32b_2hop_production.sh` | 2-node 32B (short runs <30 steps) |
| `experiments/bioreason/hold_bioreason_2node.sh` | BioReason 2-node launcher |
| `experiments/async_grpo/run_phase1_async.sh` | Async GRPO Phase 1 launcher |
| `recipes/configs/dev/production/` | All production YAML configs |
| `torchtune/dev/rl/weight_sync.py` | All weight sync runtime (~1770 lines) |
| `torchtune/dev/rl/vllm_backend.py` | vLLM init and mode setup (~690 lines) |
| `torchtune/dev/bioreason/model.py` | BioReasonModel (ESM3+GO+Qwen3-4B) |
| `torchtune/modules/moe/experts.py` | GroupedExperts BMM kernel (6.3× speedup) |
| `recipes/dev/clean_tiles.sh` | Kill all vLLM processes on current node |
| `docs/status.md` | Full run history and current baselines |
| `docs/bugs/` | Specific bug investigations |

## Known Platform Constraints

- **`torch.compile` deadlocks on multi-node** with oneCCL. Single-node backbone-only compile is viable but slow (SYCL kernel compilation takes minutes per kernel). Disabled by default (`TORCH_COMPILE_DISABLE=1`).
- **`glob.glob()` hangs** on DAOS/dfuse mounts — use `os.listdir()` + filtering.
- **`torch.xpu.empty_cache()`** causes Level Zero UR handle leaks with FSDP. Never call it in FSDP training loops.
- **CCL_WORKER_COUNT must be 1** — 4 causes 48× AllGather regression.
- **Do not set `CCL_REDUCE_SCATTER=ring`** — causes 63× regression on multi-node.
- **CCL_PROCESS_LAUNCHER=pmix** requires `mpiexec --pmi=pmix` (not `--standalone`). For SSH+interactive testing, use `CCL_PROCESS_LAUNCHER=none` + `CCL_ATL_TRANSPORT=ofi`.
- **FSDP per-module wrapping** causes catastrophic overhead — use top-level-only wrapping.

## Syncing with Upstream TorchTune

This repo tracks [pytorch/torchtune](https://github.com/pytorch/torchtune) as the `upstream`
remote. Only 4 upstream files are modified, so rebases are typically clean.

```bash
git fetch upstream
git rebase upstream/main
```

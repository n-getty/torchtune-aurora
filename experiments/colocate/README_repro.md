# Standalone reproducer — in-process colocate vLLM generation GPU page fault (Aurora / Intel PVC XPU)

> **DRAFT — crash-rate tables, driver version, and the final minimal trigger set are filled in
> after the held-node runs (Phases 1-3). This file is the Intel/vLLM handoff package if the
> fault is confirmed below the torchtune RL framework.**

## One-paragraph summary

When an in-process vLLM TP=1 engine (`enforce_eager=True`, bf16) and a PyTorch/XPU "trainer"
workload are co-resident on the **same Aurora PVC tile** in a **single OS process**, a GPU page
fault fires **non-deterministically during vLLM generation**:

```
Segmentation fault from GPU at 0xff00ffff........, ctx_id: 1 (CCS) type: 0 (NotPresent),
level: 1 (PDE) [or level: 0 (PTE)], access: 0 (Read), banned: 1, aborting.
```

There is **24-28 GiB of HBM free** at the crash — it is **not** an OOM. P(crash) rises with
cumulative generation volume (longer `max_tokens`, more iterations). `vllm_mode=server` (vLLM in
a separate process) running the identical generation envelope **does not** reproduce it — the
trigger is in-process co-residence, not vLLM KV recycling in isolation.

This is distinct from the Ray TP=8 two-process co-tenancy fault
(`docs/bugs/xpu_l0_event_pool_co_tenancy.md`, which fires at the first backward, two OS
processes per tile) — here there is **one** OS process per tile and the fault is during
generation.

## Software stack (this reproduction)

| Component | Version |
|-----------|---------|
| Aurora frameworks module | `frameworks/2025.3.1` |
| oneAPI | `2025.3.1` |
| Python | 3.12.12 |
| PyTorch | 2.10.0a0+git449b176 |
| intel_extension_for_pytorch | 2.10.10+gitd0f992f |
| vLLM | 0.15.0 (+xpu) |
| i915 (KMD) | `I915_25.2.29_PSB_250224.35` (backported to kernel 5.14.21-150400.24.214) |
| compute-runtime (NEO / UMD) | `25.18.33578` |
| Level-Zero loader | `libze_loader.so.1.24.0` |
| GPU compute driver (clinfo) | `25.18.33578` |
| GPU | Intel Data Center GPU Max 1550 (Ponte Vecchio), 6×/node, 2 tiles each (FLAT = 12 tiles), 64 GiB/tile |

### Faithfulness check (standalone repro == real recipe engine)
A single-tile R-A boot on Qwen3-4B produced the **identical** vLLM KV geometry the bug doc
reports for the real LoRA-GRPO colocate recipe:
`num_gpu_blocks_override=844`, `GPU KV cache size: 54,016 tokens`, model load 7.55 GiB, free
~47 GiB after engine up. Confirms the standalone engine is configured the same as production.

## How to run

The reproducer imports **only** `torch` + `vllm` (+ `intel_extension_for_pytorch`) — **no
torchtune**. It constructs the same engine the RL recipe uses (kwargs + monkeypatches copied
verbatim from `torchtune/dev/rl/vllm_backend.py:_init_vllm_tp1`).

```bash
module load frameworks            # 2025.3.1
# CCL standalone-row env (single tile, no distributed world needed):
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi FI_PROVIDER=cxi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Single-tile trial (faithful rung R-D = trainer compute + empty_cache churn):
ZE_AFFINITY_MASK=0 python3 scratch/repro_colocate_pagefault.py \
    --model /lus/flare/projects/ModCon/ngetty/models/Qwen3-4B \
    --rung R-D --max-gen 768 --max-model-len 1536 --iters 60 --burst-every 4
```

A clean run prints exactly one `REPRO_DONE ... crashed=0` line. A crash aborts the process
(no Python traceback) and emits the `banned:1` signature to stderr + `dmesg`.

### Crash-rate harness (statistics, not anecdotes)

The fault is stochastic, so a single run proves nothing. `run_repro_ladder.sh` launches one
process per tile (12 parallel) × N rounds = N≥12 independent trials per rung and reports
`crashed/total`:

```bash
RUNG=R-D ROUNDS=2 bash experiments/colocate/run_repro_ladder.sh   # N=24
```

## The ladder — which layer owns the fault

Each rung adds one ingredient of the real colocate loop. The first rung that faults names the
responsible layer.

| Rung | resident model | torch compute | empty_cache | reset/load/fsdp | If this is the first to fault → |
|------|----------------|---------------|-------------|-----------------|--------------------------------|
| R-A  | none           | off           | off         | off             | **pure vLLM-XPU** (file vs vLLM) |
| R-B  | qwen (resident)| off           | off         | off             | **static co-residence footprint** (Intel L0) |
| R-C  | qwen           | fwd+bwd       | off         | off             | **live co-tenancy** (Intel L0, in-process analogue of Ray bug) |
| R-D  | qwen           | fwd+bwd       | each-gen    | off             | **shared-allocator free/remap** via `empty_cache` (fixable our side) |
| R-E  | qwen           | fwd+bwd       | off         | reset/load/fsdp | which specific free/remap op (reset_prefix / load_weights / FSDP resize_) |

## Crash-rate results

_Filled after Phases 1-3._

| Rung / cell | node | crashed/N | median steps-to-crash | notes |
|-------------|------|-----------|-----------------------|-------|
| R-A vllm-only | x4516c6s5b0n0 | **0/12** | na | mg768, 60 iters, ~335K tok/tile w/ bursts — CLEAN. Rules out pure-vLLM (H2). |
| R-B +resident | x4516c6s5b0n0 | **0/12** | na | +1.2 GiB synthetic resident model, no compute — CLEAN. Static co-residence alone insufficient. |
| R-D +compute+empty_cache | x4302c2s1b0n0 | **0/12** | na | non-FSDP fwd+bwd + empty_cache/gen, HBM dead-flat — CLEAN. Single-tile alloc churn does not fault. |
| R-E +fsdp resize | x4302c2s1b0n0 | HARNESS-ERR | na | world=1 FSDP→NO_SHARD, backward all_reduce on gloo hits "No backend for xpu" — a test-scaffold error, NOT a GPU fault. FSDP-resize only testable via the real multi-rank recipe (XCCL). |

**Single-tile ladder conclusion:** R-A/R-B/R-D all CLEAN at high generation volume. The fault
does **not** reproduce on a single tile under vLLM-only, +resident-model, or +compute+empty_cache.
Combined with the in-framework reproduction below, this means the trigger requires the **real
multi-rank in-process co-residence** (12 ranks each running vLLM + FSDP XCCL collectives across
the other 11 tiles) — exactly what the standalone single-tile harness cannot exercise, and what
server-mode lacks. → The faithful Intel handoff likely needs a **multi-tile (mpiexec + FSDP
world) variant** of this reproducer (Phase 4 work item).

## In-framework reproduction (real 12-rank recipe, Qwen3-4B mg768, capacity jobs)

The real LoRA-GRPO colocate recipe reproduces the fault **reliably** at mg768:

| Cell | crashed/N | crashes at | conclusion |
|------|-----------|-----------|------------|
| baseline | 4/4 | step ~3-4 | reproduces reliably (genuine `CCS NotPresent PDE banned:1`, after 3 clean steps w/ real reward) |
| noreset (`SKIP_RESET_PREFIX=1`, reset_prefix=0 confirmed) | 4/4 | step ~3-4 | **`reset_prefix_cache` NOT the trigger** |
| pub999 (`publish_every=999`, wsync=step-0 only) | 4/4 | step ~3 | **`load_weights`/publish NOT the trigger** |
| nofsdp (`NO_FSDP=1`, no FSDP wrap/collectives) | 4/4 | step ~2-3 | **FSDP NOT required** (crashes slightly earlier — FSDP mem pressure marginally delays) |
| bigkv (4000 blocks, gpu_mem 0.55) | 4/4 | step ~3 | **KV headroom does not help** |
| ccl_mid (IPC-handle threshold 8192) | 4/4 | step ~3 | **IPC-handle cache threshold does not help** |
| ccl_low (IPC-handle threshold 2048) | 4/4 | step ~3 | same |
| noreclaim (`NO_FSDP` + ~no empty_cache) | _running_ | | |

**Conclusion: every application-level lever fails.** This matches the Ray TP=8 co-tenancy
W-probe history (W4–W16 all failed; only process isolation / server mode worked). The fault is
a **driver-level L0 co-residence resource accumulation** (H3), not addressable by recipe config.
The architectural fix is `vllm_mode=server`/`dedicated` (separate process) — already the
validated production path at mg512+. The deterministic **step-3 XCCL all-reduce explosion**
(1.5s→40s, 5/5 runs) is the in-process analogue of the Ray TP=8 UR40 first-backward wedge.

**Precursor observed (baseline):** the step before the crash, `ADAPTER_AR` adapter all-reduce
inflated 1.51s → 38.99s and `TIMING` 22.9s → 130.6s — i.e. an XCCL/L0-layer slowdown
**accumulates across steps** before the page fault. This points toward L0 resource degradation
(H3-family) rather than a single clean page free/remap.

## Minimal trigger set — a 2-factor interaction

The fault requires **BOTH** of the following simultaneously; **neither alone reproduces**:

1. a real vLLM **`load_weights()`** publish into the **live in-process engine** (real HF-named
   weights — the engine's own fused param names are rejected and the call no-ops), AND
2. concurrent **multi-rank XCCL collectives** on the same tile (the FSDP/training world).

| reproducer | factor 1 (`load_weights`) | factor 2 (XCCL) | result |
|------------|---------------------------|-----------------|--------|
| `repro_colocate_pagefault.py --rung R-LW` (single tile) | yes | no | **clean** |
| `repro_colocate_pagefault_multitile.py` (no `--load-real-weights`) | no (no-op) | yes | **clean** |
| `repro_colocate_pagefault_multitile.py --load-real-weights` (12 tiles) | yes | yes | **CRASH, N=2** |

Minimal repro command (frameworks module only):
```bash
qsub -v STEPS=12,LOAD_REAL=1 experiments/colocate/pbs_repro_multitile.sh
```
Crashes at step 0 right after all 12 ranks run `load_weights`, with the byte-identical
`CCS NotPresent PDE banned:1` + NEO `drm_neo.cpp:288` abort. Jobs 8559640, 8559661 (N=2).

## What we ruled out (in-framework A/B, `run_colocate_ab.sh`)

_Filled after Phase 3: reclaim on/off, reset_prefix skip, publish cadence, KV headroom — each
M≥8 same-node back-to-back._

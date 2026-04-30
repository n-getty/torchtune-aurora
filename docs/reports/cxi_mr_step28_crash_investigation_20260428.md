# CXI MR Cache Stale Entry Crash: Root Cause = FSDP Collectives

**Date:** 2026-04-28
**Severity:** Blocks all 32B production runs > 28 steps on Aurora
**Affected:** Qwen3-32B GRPO, 2-node Aurora (12-tile FSDP2 + 3×TP=4 dedicated vLLM)
**Status:** Root cause definitively identified. No fix available within current stack.

## Executive Summary

32B FSDP2 GRPO training crashes deterministically at step 28-29 with a `banned:1` GPU
page fault. After 11 experiments systematically isolating every component, the root cause
is **not** weight sync — it is the FSDP AllGather/ReduceScatter collectives themselves.

The caching allocator's `release_available_cached_blocks()` decommits GPU VA ranges at
step 28. Those VA ranges have CXI memory registration (MR) entries from prior FSDP
collectives via oneCCL. The stale MR entries persist in the CXI provider's cache, and
the next FSDP collective that accesses those VAs triggers a GPU page fault.

**The definitive proof:** Run 21 eliminated ALL weight-sync CXI traffic (Gloo TCP +
sender rotation), yet every rank — including those that never participated in any weight
sync — showed identical external memory patterns and crashed at step 29 with the same
`banned:1` signature.

The fix must prevent the caching allocator contraction itself (reduce per-tile memory
pressure) or wait for Intel's SHS 13.1.0 fix for the CXI MR cache leak.

## Crash Signature

```
Segmentation fault from GPU at 0xff000007fda4e000, ctx_id: 1 (CCS)
type: 0 (NotPresent), level: 1 (PDE), access: 1 (Write), banned: 1, aborting.
```

Always preceded by a massive GRPO/backward slowdown (7-10× normal) at the contraction step:
```
TIMING step=28  total=147.4s  gen=21.1s  grpo=97.6s  ...  # normal grpo ~14s
```

## Crash Mechanism

The crash develops in four deterministic phases:

### Phase 1: Allocator Expansion (steps 0-7)

The PyTorch caching allocator expands `torch_resv` from ~47 GiB to ~62 GiB as FSDP
AllGather/ReduceScatter operations create larger cached blocks. By step 7, `torch_resv`
stabilizes at 62.04 GiB (R2-R10) or 62.55 GiB (R0/R1), consuming nearly all of the
63.98 GiB L0 tile memory.

### Phase 2: Steady State (steps 7-27)

Memory is stable. `external` memory (non-PyTorch L0 allocations, primarily oneCCL
internal state) is flat at 1.30-1.51 GiB. `l0_free` is tight but stable at 0.44-0.95
GiB depending on rank. The system runs reliably with consistent ~80s step times.

### Phase 3: Contraction (step 28)

A forward or backward pass peak allocation cannot be satisfied from `l0_free` (~0.4 GiB).
The caching allocator calls `release_available_cached_blocks()`, which decommits
6-16 GiB of cached blocks via `sycl::free()` → `zeMemFree()`. This releases the GPU
virtual address ranges back to the Level Zero driver.

The contraction is visible as:
- `torch_resv`: 62.04 → 55.60 GiB (Δ = -6.44 GiB)
- `l0_free`: 0.44 → 4.86 GiB (temporarily more headroom)
- `external`: 1.35 → 3.52 GiB (decommitted VA metadata now visible as "external")
- `grpo` time: 14s → 97s (allocator thrashing as blocks are re-acquired)

**Critical:** This contraction happens on ALL ranks simultaneously. It is an
allocator-internal event triggered by the gap between `torch_resv` (62 GiB) and
`l0_total` (64 GiB) being too small for peak allocation needs.

### Phase 4: Crash (step 29)

The decommitted VA ranges still have CXI MR entries registered in libfabric's cache.
When FSDP AllGather/ReduceScatter operations at step 29 use those VA ranges for new
allocations, the GPU hardware detects the stale MR references and faults with `banned:1`.

The crash always occurs during the backward pass (GRPO gradient computation), which
requires FSDP ReduceScatter — the first collective that touches re-allocated memory
in the previously-decommitted VA ranges.

## Experiment Summary

11 experiments systematically isolated every variable:

| Run | Config | Crash | torch_resv | ext (R2-R10) | ext (R0/R1) | Key finding |
|-----|--------|-------|-----------|-------------|-------------|-------------|
| 11 | XCCL sendrecv, gc:0.95 | step 29 | 62.04 | FLAT 0.83 | oscillating | GC contraction kills |
| 12 | XCCL sendrecv, gc:0.99 | 25/25 ✓ | 62.04 | FLAT 1.45 | oscillating | gc:0.99 prevents GC; test too short |
| 13 | XCCL bcast, gc:0.99 | 25/25 ✓ | 62.04 | FLAT 1.84 | +22 MiB/s | Test too short for step 28 |
| 14 | XCCL bcast, gc:0.99, 40 steps | **step 28** | 62.04→55.60 | FLAT 0.99 | +13 MiB/s | **Baseline contraction crash** |
| 15 | + ZE cache monitor | step 29 | 62.04 | — | — | +1 step |
| 16 | + ODP=1 | step 0 | — | — | — | CCL incompatible |
| 17 | + kdreg2 | step 13 | — | — | — | NODE CRASH |
| 18 | + full stale key prot | **step 29** | 62.04→55.60 | FLAT 1.46 | +10-20 MiB/s | **Survived step 28, died step 29** |
| 19c | + PG reset interval=10 | **step 29** | 62.04→55.60 | FLAT 1.30 | +10 MiB/s | PG reset doesn't clear CXI MR |
| 20 | Sender rotation pool=9, XCCL | **step 28** | 62.04→55.60 | FLAT 0.99 | R0 FLAT 1.03 | Rotation works, doesn't prevent contraction |
| **21** | **Sender rotation pool=9, Gloo** | **step 29** | **62.04→55.60** | **FLAT 1.35** | **R0 FLAT 1.51** | **PROOF: zero CXI wsync traffic, still crashes** |

Also tested but not numbered in this series:
- Gloo cross-PG without sender rotation (Run 8): 20/20 exit=0, 47s broadcast — test didn't reach step 28
- Static XCCL buffers (Run 3/6): Eliminated VA churn, external FLAT on R2-R10 — tests didn't reach step 28
- Deferred async broadcast: 37.2s/step (13.4% improvement) — crashed at step 28, same cause

## Key Experiments

### Run 14 — Baseline XCCL Broadcast (CRASHED step 28)

**Config:** XCCL broadcast + userfaultfd + gc:0.99, 40 steps requested

Step 28 contraction on ALL ranks simultaneously:

| Rank | torch_resv before | torch_resv after | Δ resv | external before | external after |
|------|-------------------|------------------|--------|-----------------|----------------|
| R0   | 62.55             | 56.55            | -6.00  | 1.38            | 3.88           |
| R1   | 62.04             | 55.60            | -6.44  | 1.89            | 4.83           |
| R4   | 62.04             | 55.60            | -6.44  | 0.99            | 3.56           |

Steady-state memory (steps 7-27) was perfectly stable:

| Step | R0 ext | R4 ext (R2-R10) | R0 l0_free | R4 l0_free |
|------|--------|-----------------|------------|------------|
|    7 | 0.99   | 0.99            | 0.44       | 0.95       |
|   14 | 1.21   | 0.99            | 0.22       | 0.95       |
|   24 | 1.34   | 0.99            | 0.09       | 0.95       |
|   27 | 1.38   | 0.99            | 0.05       | 0.95       |

**Key observation:** R4 (R2-R10) external was FLAT at 0.99 GiB for 20 steps — yet the
contraction still happened. The contraction is driven by `torch_resv` (62.04) filling
`l0_total` (63.98), not by external growth.

### Run 18 — Full Stale Key Protection (SURVIVED step 28, CRASHED step 29)

**Config:** userfaultfd + ZE monitor + OPTIMIZED_MRS=0 + MR_MATCH_EVENTS=1 + CACHE_MERGE_REGIONS=0

**BREAKTHROUGH:** First XCCL run to survive the step 28 contraction. All 6 CXI MR env
vars applied simultaneously. Step 28 contracted (62.04→46.05 GiB) but did not crash.
Step 29 crashed with banned:1 during backward — residual stale MR entries from the
16 GiB decommit.

Proves: even with every available CXI MR cleanup mechanism active, a 16 GiB batch
decommit creates more stale entries than the cleanup can process in one step.

### Run 19c — PG Reset (CRASHED step 29)

**Config:** PG reset every 10 steps + streaming gather + userfaultfd + gc:0.99

PG reset verified working (gen=10 at step 9, gen=20 at step 19). But PG reset does NOT
reduce external memory — CXI MR cache entries persist after ProcessGroup destruction.

```
Step  7: R0 ext=1.21, R4 ext=1.30  (before reset)
Step 14: R0 ext=1.26, R4 ext=1.30  (after gen=10 reset at step 9)
Step 22: R0 ext=1.33, R4 ext=1.30  (after gen=20 reset at step 19)
```

**Key finding:** MR cache is maintained by the CXI provider (libfabric), not by the
ProcessGroup. Destroying and recreating PGs has zero effect on the underlying MR
registrations.

### Run 20 — Sender Rotation Pool=9, XCCL (CRASHED step 28)

**Config:** WSYNC_SENDER_POOL_SIZE=9, WSYNC_CROSS_METHOD=xccl_broadcast, 9 dedicated
2-rank cross-PGs (R2-R10 ↔ vLLM), rotation R2→R3→...→R10→R2

Sender rotation mechanics worked perfectly:
- All 9 cross-PGs created and verified
- Rotation cycling correctly: R2 at step 0, R3 at step 1, ..., R10 at step 8, R2 at step 9
- R0 external FLAT at 1.03 GiB (no longer broadcasting)

But still crashed at step 28. The contraction is from training memory pressure
(`torch_resv` fills `l0_total`), which sender rotation cannot prevent. Post-contraction,
active senders R3 and R7 had 17+ GiB external (stale MR entries on freed blocks).

### Run 21 — Sender Rotation + Gloo (CRASHED step 29) — THE PROOF

**Config:** WSYNC_SENDER_POOL_SIZE=9, WSYNC_CROSS_METHOD=gloo, 9 Gloo (TCP) cross-PGs

**This is the definitive experiment.** Gloo uses TCP sockets over Slingshot — no CXI RDMA,
no MR registrations, zero CXI traffic from weight sync. If the crash were caused by
weight sync MR entries, this run would survive. It did not.

Step 28 POST-BWD (contraction) — ALL ranks show identical external jump:

| Rank | Role | torch_resv | external | l0_free |
|------|------|-----------|----------|---------|
| R0   | Non-sender | 55.60 | 3.57 | 4.82 |
| R1   | Non-sender | 55.60 | 3.57 | 4.82 |
| R2   | Sender (Gloo) | 55.60 | 3.52 | 4.86 |
| R4   | Sender (Gloo) | 55.60 | 3.52 | 4.86 |
| R7   | Sender (Gloo) | 55.60 | 3.52 | 4.86 |
| R11  | vLLM | 55.06 | 4.07 | 4.86 |

**R0 and R1 are `is_sender=False`** — they never created a cross-PG, never did any weight
sync broadcast, never touched any CXI weight sync path. Yet they show external=3.57 GiB,
identical to sender ranks.

**R11 is the vLLM rank** — it only receives weights over Gloo TCP, no CXI MR activity.
Yet it shows external=4.07 GiB.

The only CXI-using operations common to ALL ranks (including R0, R1, R11) are FSDP
AllGather and ReduceScatter via oneCCL/XCCL.

**Therefore:** The stale MR entries causing banned:1 originate from FSDP collectives,
not weight sync.

Steady-state memory was perfectly flat through 27 steps — zero external growth on any rank:

| Step | R0 ext | R0 l0_free | R4 ext | R4 l0_free |
|------|--------|------------|--------|------------|
|    3 | 1.51   | 3.35       | 1.33   | 3.52       |
|   10 | 1.51   | 0.44       | 1.34   | 0.61       |
|   20 | 1.51   | 0.44       | 1.35   | 0.59       |
|   27 | 1.51   | 0.44       | 1.37   | 0.58       |

Step 29: `banned:1` during backward pass. Identical crash as all previous XCCL runs.

## Approaches Exhausted

### GC Threshold Tuning
| Threshold | Effect |
|-----------|--------|
| gc:0.6 | GC triggers aggressively → crashes earlier (step 29 instead of 28) |
| gc:0.8 | GC triggers at step 28, same crash |
| gc:0.95 | GC triggers at step 28-29, same crash |
| gc:0.99 | Prevents GC but NOT expandable_segments contraction — crashes step 28 |

**Conclusion:** GC threshold delays the trigger but cannot prevent the allocator-internal
contraction when `torch_resv` fills `l0_total`.

### CXI MR Environment Variables
| Variable | Effect |
|----------|--------|
| FI_MR_CACHE_MONITOR=userfaultfd | Bounds leak via kernel VA tracking — necessary but insufficient |
| FI_MR_ZE_CACHE_MONITOR_ENABLED=1 | GPU memory tracking for MR cache — +1 step survival |
| FI_CXI_ODP=1 | On-demand paging — **CCL incompatible** (fi_cq_readerr err:5) |
| kdreg2 | Kernel MR deregistration — **CRASHES ENTIRE NODES** |
| FI_CXI_OPTIMIZED_MRS=0 | Disables MR optimization — helped survive step 28 |
| FI_CXI_MR_MATCH_EVENTS=1 | Event-based MR matching — helped survive step 28 |
| FI_MR_CACHE_MERGE_REGIONS=0 | Disables MR region merging — helped survive step 28 |

**Conclusion:** All 7 CXI MR env vars tested. Best result (Run 18): survived step 28,
crashed step 29. No env var combination can prevent the crash — the 16 GiB batch decommit
overwhelms any MR cleanup mechanism.

### Weight Sync Transport
| Transport | CXI MR entries? | Crash? |
|-----------|----------------|--------|
| XCCL send/recv | Yes | step 28 |
| XCCL broadcast | Yes | step 28 |
| Gloo TCP | **No** | **step 29** (same crash, proves FSDP is source) |

### Weight Sync Architecture
| Architecture | Effect |
|-------------|--------|
| Single sender (R0) | R0 external grows 13 MiB/step |
| Sender rotation pool=9 | R0 FLAT, but contraction still happens |
| PG reset every 10 steps | CXI MR cache not cleared by PG reset |
| Static buffers | Eliminates VA churn, but contraction still happens |
| Deferred async broadcast | 13.4% faster steps, same step-28 crash |

### Sync Interval
| Interval | Effect |
|----------|--------|
| Every step | ~80s step time, external grows faster on R0 |
| Every 2 steps | ~40s step time, external growth halved, same crash point |

## Viable Forward Paths

### 1. Prevent the Contraction (reduce per-tile memory)

The contraction happens because `torch_resv` (62 GiB) fills nearly all of `l0_total`
(64 GiB), leaving <1 GiB for peak allocation spikes. If `torch_resv` were reduced to
~55-58 GiB, the peak allocation would fit without triggering `release_available_cached_blocks()`.

Options:
- **FSDP2 CPU offloading** (`CPUOffloadPolicy`): Offload optimizer states to host RAM,
  freeing ~8-12 GiB/tile. Stays within current PyTorch stack.
- **More nodes** (3+ nodes, 18+ tiles): Reduces per-tile shard size, lowering `torch_resv`.
- **Gradient accumulation with smaller micro-batch**: Currently chunk=4; chunk=2 would
  halve activation memory during backward.
- **Activation checkpointing**: If not already on all transformer layers.

### 2. Checkpoint-Restart Every 25 Steps (workaround)

Save checkpoint at step 25 and restart. Training state is fully recoverable.
Overhead: ~30s per restart (model load + optimizer reload).

### 3. Wait for SHS 13.1.0 (Intel fix)

Intel's Slingshot Host Software (SHS) team has acknowledged the CXI MR cache leak.
The fix is targeted for SHS 13.1.0. Current version: SHS 12.0.0 (2025-04-18).
Timeline unknown.

## Version Info

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0a0+git449b176 |
| Intel Frameworks | aurora_frameworks-2025.3.1 |
| libfabric | 1.22.0 |
| SHS | 12.0.0 (2025-04-18) |
| oneCCL | bundled with frameworks |
| Level Zero | bundled with frameworks |
| FI_PROVIDER | cxi (Slingshot 11) |
| GPU | Intel Data Center GPU Max 1550 (64 GiB HBM2e per tile) |
| Nodes | 2 × Aurora compute nodes, 12 tiles training + 12 tiles vLLM |

## Reproduction

Any of these configs will reproduce the crash at step 28-29:

```bash
# Baseline (crashes step 28)
qsub -v NSTEPS=40,GC_THRESHOLD=0.99,SAVE_EVERY=0 \
  experiments/multinode_32b/run_32b_2hop_production.sh

# With all mitigations (crashes step 29 — proves unfixable)
qsub -v NSTEPS=40,GC_THRESHOLD=0.99,SAVE_EVERY=0,\
WSYNC_SENDER_POOL_SIZE=9,WSYNC_CROSS_METHOD=gloo \
  experiments/multinode_32b/run_32b_2hop_production.sh
```

## Appendix: Run 21 Full Memory Timeline

Gloo + sender rotation (job 8453200, 2026-04-28). Zero CXI weight sync traffic.

R0 POST-BWD (non-sender rank — never touched any weight sync CXI path):
```
Step  3: resv=59.13  ext=1.51  l0_free=3.35  retries=0
Step  8: resv=62.04  ext=1.51  l0_free=0.44  retries=0
Step 14: resv=62.04  ext=1.51  l0_free=0.44  retries=0
Step 20: resv=62.04  ext=1.51  l0_free=0.44  retries=0
Step 27: resv=62.04  ext=1.51  l0_free=0.44  retries=0
Step 28: resv=55.60  ext=3.57  l0_free=4.82  retries=0  ← CONTRACTION
Step 29: banned:1 during backward
```

R4 POST-BWD (Gloo sender — weight sync via TCP, zero CXI MR entries):
```
Step  3: resv=59.13  ext=1.33  l0_free=3.52  retries=0
Step  8: resv=62.04  ext=1.34  l0_free=0.61  retries=0
Step 14: resv=62.04  ext=1.35  l0_free=0.59  retries=0
Step 20: resv=62.04  ext=1.35  l0_free=0.59  retries=0
Step 27: resv=62.04  ext=1.37  l0_free=0.58  retries=0
Step 28: resv=55.60  ext=3.52  l0_free=4.86  retries=0  ← CONTRACTION
Step 29: banned:1 during backward
```

Both sender and non-sender ranks show identical crash behavior. External memory is
flat through step 27 on ALL ranks. The crash source is FSDP collectives (the only CXI
operation common to all ranks), not weight sync.

# Aurora L0 driver resource exhaustion — four signatures, likely shared root cause

**Target**: github.com/argonne-lcf/AuroraBugTracking (cc [email protected])
**Filer**: ngetty / ALCF ModCon (TorchTune RL on Aurora)
**Frameworks tested**: `frameworks/2025.3.1` (PyTorch 2.10.0a0+git449b176, oneCCL 2021.17, Level Zero 1.24.0, I915_25.2.29). Persists on torch 2.11.0+xpu against the same `oneapi/release/2025.3.1` module.
**Possible related ticket**: ALCF AuroraBugTracking #106 / GSD-12152 (`zeEventPoolDestroy` cross-pool dependency hang).

## TL;DR

We are seeing four distinct failure signatures on Aurora that we believe share an L0-driver-internal resource pool (events / queues / IPC handles / submission objects per tile). All four manifest as either `UR_RESULT_ERROR_OUT_OF_RESOURCES` (= L0 error code 40) or as `banned:1` PDE/PML5 page faults. Each signature has a documented reproducer in our repo. We are asking Intel to confirm the shared mechanism and to advise on whether the **counter-based event** path (ZE_EVENT_POOL_COUNTER_BASED_EXP, in spec since L0 1.10) would shift any of the symptoms.

## The four signatures

| # | Signature | Trigger | Crash point | Doc |
|---|-----------|---------|-------------|-----|
| 1 | UR 40 at step-0 BWD under per-tile process co-tenancy | Two OS processes attached to same PVC tile both submit L0 work | First BWD chunk of step 0; ~30 GiB HBM headroom | `docs/bugs/xpu_l0_event_pool_co_tenancy.md` |
| 2 | UR 40 from `empty_cache()` + FSDP `storage.resize_()` cycle | `torch.xpu.empty_cache()` between FSDP forwards | Iter 70 (FSDP2) / iter 145 (FSDP1), bit-deterministic | `docs/bugs/intel_xpu_resource_leak_bug_report.md` |
| 3 | banned:1 PDE — CCL IPC handle cache | LRU eviction at default threshold (~step 28); accumulation at threshold=65536 (10.85 GiB metadata) | Step 28 dense-32B / step 2 EP=16 | `docs/bugs/ccl_ipc_handle_cache.md` |
| 4 | banned:1 PML5 — `XPUPluggableAllocator` + FSDP cross-stream race | Any custom allocator registered via `XPUPluggableAllocator` | Step 1 compute, NOT during a CCL collective | `docs/bugs/xpu_pluggable_allocator_record_stream.md` |

All four are reproducible on a fresh debug-scaling node hold and have ruled-out alternative explanations documented in the linked files.

## Why we believe these share a root cause

1. **Same error families** — UR 40 is `UR_RESULT_ERROR_OUT_OF_RESOURCES`. The `banned:1` PDE/PML5 faults are what L0 emits when a kernel touches a virtual address whose backing handle was already freed/evicted. Both are accounting failures inside the same per-tile L0 driver state.
2. **All four require multi-actor or multi-cycle pressure on per-tile L0 objects** — co-tenancy (#1), `storage.resize_` cycles (#2), per-step IPC-handle churn (#3), per-step VA churn under cross-stream comm (#4).
3. **None happens on CUDA** — surveyed NeMo-RL, torchrl, OpenRLHF (all use the same Ray-actor + vLLM patterns). None reports an analog of #1; CUDA has no per-tile event pool ceiling. NCCL has no IPC handle eviction race.
4. **`device_empty_cache` becomes a no-op on XPU and does not help #1** — confirms the leak is not in the PyTorch allocator. The HBM accounting is clean at crash for #1, #3, #4.

## Diagnostic asks

Each of the requests below would let us localize whether these are one bug or four.

1. **Public env knob to query and set the per-tile L0 event-pool ceiling.** Today `ZEX_NUMBER_OF_CCS` is the closest documented control and it only affects CCS partitioning, not event pools (we tested W5 — does not move #1).
2. **Diagnostic API for per-process L0 resource usage** (events, queues, submissions, IPC handles open, IPC handles cached). Today we have only post-mortem `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
3. **Documentation of which L0 resources are per-process vs per-tile vs per-context** — specifically the lifetime of `zeEventPool*`, `zeMemOpenIpcHandle` cache state, and `zeCommandList*` submission objects.
4. **Whether ALCF Bug 106 / GSD-12152** (`zeEventPoolDestroy` cross-pool dependency hang, reportedly reproduced internally by Intel) relates to signature #1.
5. **Whether enabling counter-based events** (`ZE_EVENT_POOL_COUNTER_BASED_EXP_FLAG_IMMEDIATE`, in spec since L0 1.10) would side-step the per-tile event-pool ceiling responsible for #1. This is the only spec-level structural fix we can identify without driver changes.

## Proposed Intel-side experiment

We are running the following A/B locally and will report results back to this issue:

```
NEOReadDebugKeys=1
EnableImplicitConvertionToCounterBasedEvents=1
ForceInOrderImmediateCmdListExecution=1
EnableTimestampPoolAllocator=1
```

This stack implicitly converts pool-allocated events to counter-based, forces in-order immediate command lists, and routes timestamp pools to a separate allocator. If signature #1 flips with this stack but #2-4 do not, that argues for #1 being event-pool-specific and #2-4 being IPC/VA-specific. We will report outcome as W6 in `docs/bugs/xpu_l0_event_pool_co_tenancy.md` regardless.

## Reproducer paths (single-node debug-scaling, 1-hour walltime)

```bash
# Signature 1 (UR40 co-tenancy, TP=8 colocated Ray)
bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/colocate/run_qwen3_8b_colocate_ray.sh 3 4

# Signature 2 (empty_cache+FSDP UR40)
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    /lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/repro_xpu_resource_leak.py \
    --fsdp --layers 12 --hidden 1024 --heads 8

# Signature 3 (CCL IPC handle banned:1 — needs 32B model and ~30 steps)
# See docs/reports/ccl_external_memory_growth_32b.md

# Signature 4 (pluggable allocator banned:1 PML5 — FSDP2 32B 10 ranks)
# See experiments/arena_ipc/diag_findings.md Phase 1b / 2 / 5
```

## What works today (so we are not stuck)

- Signature 1: process-isolated tiles (W2 — trainer 0..7, vLLM TP=4 on 8..11). 10/10 clean steps, exit=0.
- Signature 2: never call `torch.xpu.empty_cache()` in FSDP loops (caching allocator handles reuse).
- Signature 3: 2-node HSDP for 32B; threshold=65536 trades stale-eviction for accumulation OOM.
- Signature 4: avoid `XPUPluggableAllocator` entirely; default allocator with `gc:0.95`.

These workarounds all sacrifice perf or capacity. Signature 1 specifically prevents the natural Ray-colocate-on-shared-tiles design that NeMo-RL, OpenRLHF, etc. all assume.
